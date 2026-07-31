"""
Inference tools: patch grid, TTA, batched prediction, stitching and
probability-to-centerline vectorization.

Author: Jakub Zapletal
Date:   2026-04-05
"""

import numpy as np
import networkx as nx
import torch
import geopandas as gpd
from shapely.geometry import LineString, box
from skimage.morphology import skeletonize, remove_small_objects, binary_closing, disk
from skimage.measure import label as label_components
from tqdm import tqdm

from .gis import Raster
from .models import normalize_patch_array


# ============================================================
# PATCH GRID
# ============================================================


def generate_patch_centers(corridor_geom, stride_m, patch_extent_m):
    """Sliding-window patch centers covering the corridor, as (x, y) tuples."""
    minx, miny, maxx, maxy = corridor_geom.bounds

    # Align grid origin to the stride for reproducibility
    x_start = (minx // stride_m) * stride_m + stride_m / 2
    y_start = (miny // stride_m) * stride_m + stride_m / 2

    centers = []
    y = y_start
    while y <= maxy + stride_m:
        x = x_start
        while x <= maxx + stride_m:
            # Keep the patch if its bounding box intersects the corridor
            patch_box = box(
                x - patch_extent_m / 2,
                y - patch_extent_m / 2,
                x + patch_extent_m / 2,
                y + patch_extent_m / 2,
            )
            if patch_box.intersects(corridor_geom):
                centers.append((x, y))
            x += stride_m
        y += stride_m

    return centers


# ============================================================
# PATCH EXTRACTION + TTA + BATCHED PREDICTION
# ============================================================


def extract_patch(cx, cy, rasters, patch_size_px, patch_extent_m, tpi_radii):
    """
    Stacked patch array at the given center, channel order:
    dsm, tpi per radius, canopy_height, canopy_height_sd, water.
    :param rasters: dict of open datasets with keys
                    dsm, canopy_height, canopy_height_sd, water
    """
    bbox = (
        cx - patch_extent_m / 2,
        cy - patch_extent_m / 2,
        cx + patch_extent_m / 2,
        cy + patch_extent_m / 2,
    )

    dsm = Raster.read_window(rasters["dsm"], bbox, patch_size_px, "bilinear")
    canopy = Raster.read_window(rasters["canopy_height"], bbox, patch_size_px, "bilinear")
    canopy_sd = Raster.read_window(
        rasters["canopy_height_sd"], bbox, patch_size_px, "bilinear"
    )
    # Water mask is categorical: nearest neighbour, then strict 0/1
    water = Raster.read_window(rasters["water"], bbox, patch_size_px, "nearest")
    water = (water > 0.5).astype(np.float32)

    tpi_channels = [Raster.compute_tpi(dsm, r) for r in tpi_radii]

    return np.stack([dsm, *tpi_channels, canopy, canopy_sd, water], axis=0)


def tta_transforms(mode):
    """
    (forward, inverse) tensor transform pairs for test-time augmentation,
    applied to (B, C, H, W); the output is mapped back before averaging.
        "flip" - identity + horizontal + vertical + both (4 passes)
        "d4"   - 4 rotations x 2 mirrors (8 passes)
    """
    def rot(x, k):
        return torch.rot90(x, k, dims=(2, 3))

    def hflip(x):
        return torch.flip(x, dims=(3,))

    if mode == "flip":
        return [
            (lambda x: x, lambda y: y),
            (hflip, hflip),
            (lambda x: torch.flip(x, dims=(2,)), lambda y: torch.flip(y, dims=(2,))),
            (lambda x: torch.flip(x, dims=(2, 3)), lambda y: torch.flip(y, dims=(2, 3))),
        ]

    transforms = []
    for f in (False, True):
        for k in (0, 1, 2, 3):
            def fwd(x, f=f, k=k):
                return rot(hflip(x) if f else x, k)

            def inv(y, f=f, k=k):
                yk = rot(y, -k)
                return hflip(yk) if f else yk

            transforms.append((fwd, inv))
    return transforms


def run_inference(
    model,
    centers,
    norm_stats,
    raster_paths,
    patch_size_px,
    patch_extent_m,
    tpi_radii,
    zscore_channel_names,
    batch_size,
    device,
    use_tta=True,
    tta_mode="d4",
):
    """Batched inference; returns a list of (cx, cy, prediction) tuples.
    :param raster_paths: dict of file paths with keys
                         dsm, canopy_height, canopy_height_sd, water
    """
    rasters = {k: Raster.open_raster(p) for k, p in raster_paths.items()}

    predictions = []
    n_centers = len(centers)
    tta = tta_transforms(tta_mode) if use_tta else [(lambda x: x, lambda y: y)]

    with torch.no_grad():
        for batch_start in tqdm(
            range(0, n_centers, batch_size),
            desc="Inference",
            total=(n_centers + batch_size - 1) // batch_size,
        ):
            batch_centers = centers[batch_start : batch_start + batch_size]

            patches = []
            for cx, cy in batch_centers:
                p = extract_patch(cx, cy, rasters, patch_size_px, patch_extent_m, tpi_radii)
                p = normalize_patch_array(p, norm_stats, zscore_channel_names)
                patches.append(p)

            batch = torch.from_numpy(np.stack(patches, axis=0)).float().to(device)

            prob_sum = None
            for fwd, inv in tta:
                logits = model(fwd(batch))
                prob = inv(torch.sigmoid(logits))
                prob_sum = prob if prob_sum is None else prob_sum + prob
            probs = (prob_sum / len(tta)).cpu().numpy()[:, 0]

            for (cx, cy), prob in zip(batch_centers, probs):
                predictions.append((cx, cy, prob))

    for k in rasters:
        rasters[k] = None

    return predictions


# ============================================================
# STITCHING
# ============================================================


def blend_window(size_px, floor=0.02):
    """2D raised-cosine (Hann) window with a small floor for feathered stitching."""
    n = np.arange(size_px)
    w1d = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (size_px - 1)))
    w1d = np.clip(w1d, floor, None)
    return np.outer(w1d, w1d).astype(np.float32)


def stitch_predictions(
    predictions, bounds, stride_m, patch_extent_m, patch_res_m, patch_size_px,
    method="feather",
):
    """
    Stitch patch predictions into one probability raster over the given bounds.
    :param bounds: (minx, miny, maxx, maxy) of the covered area
    :param method: "feather" (raised-cosine blend, no seams) | "max" | "average"
    :return: (prob_raster, geotransform)
    """
    minx, miny, maxx, maxy = bounds

    # Align to the stride grid so patch positions land cleanly
    origin_x = (minx // stride_m) * stride_m - patch_extent_m / 2
    origin_y = (maxy // stride_m + 1) * stride_m + patch_extent_m / 2

    width_m = (maxx - origin_x) + patch_extent_m
    height_m = (origin_y - miny) + patch_extent_m

    width_px = int(np.ceil(width_m / patch_res_m))
    height_px = int(np.ceil(height_m / patch_res_m))

    def _offsets(cx, cy):
        col_off = int(round((cx - patch_extent_m / 2 - origin_x) / patch_res_m))
        row_off = int(round((origin_y - cy - patch_extent_m / 2) / patch_res_m))
        r0, r1 = max(0, row_off), min(height_px, row_off + patch_size_px)
        c0, c1 = max(0, col_off), min(width_px, col_off + patch_size_px)
        pr0 = r0 - row_off
        pr1 = pr0 + (r1 - r0)
        pc0 = c0 - col_off
        pc1 = pc0 + (c1 - c0)
        return (r0, r1, c0, c1, pr0, pr1, pc0, pc1)

    if method == "max":
        prob = np.zeros((height_px, width_px), dtype=np.float32)
        for cx, cy, pred in predictions:
            r0, r1, c0, c1, pr0, pr1, pc0, pc1 = _offsets(cx, cy)
            prob[r0:r1, c0:c1] = np.maximum(prob[r0:r1, c0:c1], pred[pr0:pr1, pc0:pc1])
    else:
        # Weighted blend; uniform window for "average", raised-cosine for "feather"
        if method == "feather":
            window = blend_window(patch_size_px)
        else:
            window = np.ones((patch_size_px, patch_size_px), dtype=np.float32)

        sum_pred = np.zeros((height_px, width_px), dtype=np.float32)
        sum_wts = np.zeros((height_px, width_px), dtype=np.float32)
        for cx, cy, pred in predictions:
            r0, r1, c0, c1, pr0, pr1, pc0, pc1 = _offsets(cx, cy)
            w = window[pr0:pr1, pc0:pc1]
            sum_pred[r0:r1, c0:c1] += pred[pr0:pr1, pc0:pc1] * w
            sum_wts[r0:r1, c0:c1] += w

        prob = np.zeros_like(sum_pred)
        mask = sum_wts > 0
        prob[mask] = sum_pred[mask] / sum_wts[mask]

    geotransform = (origin_x, patch_res_m, 0.0, origin_y, 0.0, -patch_res_m)
    return prob, geotransform


# ============================================================
# VECTORIZATION: probability -> centerlines
# ============================================================
#
# Each connected high-probability region is reduced to its principal
# centerline(s) by repeatedly extracting the longest internal path through the
# region's skeleton graph; short branches fall below the length filter. Whole
# paths are emitted, so a continuous levee stays continuous.


def _neighbors(p, pts):
    """8-connectivity neighbours of pixel p present in the set pts."""
    r, c = p
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            q = (r + dr, c + dc)
            if q in pts:
                out.append(q)
    return out


def _degree_map(skel_mask):
    """Vectorized 8-connectivity degree of each skeleton pixel (0 elsewhere)."""
    s = skel_mask.astype(np.uint8)
    p = np.pad(s, 1)
    nb = (
        p[:-2, :-2]
        + p[:-2, 1:-1]
        + p[:-2, 2:]
        + p[1:-1, :-2]
        + p[1:-1, 2:]
        + p[2:, :-2]
        + p[2:, 1:-1]
        + p[2:, 2:]
    )
    return nb * s


def _chain_length_px(path):
    """Euclidean length (in pixels) of an ordered pixel path."""
    return sum(
        np.hypot(path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1])
        for k in range(len(path) - 1)
    )


def _build_reduced_graph(pts, deg, nodes):
    """
    Reduced skeleton graph: endpoints (deg 1) and junctions (deg >= 3) are
    nodes, degree-2 pixel chains between them are edges (with pixel path and
    Euclidean length). One node per junction instead of per pixel keeps
    Dijkstra fast on large AOIs. Returns (graph, loop_paths), loop_paths
    being self-closing chains handled separately.
    """
    G = nx.Graph()
    G.add_nodes_from(nodes)
    loop_paths = []
    visited_starts = set()

    for node in nodes:
        for nb in _neighbors(node, pts):
            if (node, nb) in visited_starts:
                continue
            # Walk the degree-2 chain from node until the next node
            path = [node, nb]
            prev, cur = node, nb
            while deg.get(cur, 0) == 2:
                nxts = [n for n in _neighbors(cur, pts) if n != prev]
                if not nxts:
                    break
                prev, cur = cur, nxts[0]
                path.append(cur)
            end = path[-1]
            visited_starts.add((node, nb))
            if len(path) >= 2:
                visited_starts.add((end, path[-2]))

            length = _chain_length_px(path)
            if node == end:
                loop_paths.append(path)  # self-loop chain, emit separately
                continue
            if G.has_edge(node, end) and G[node][end]["weight"] >= length:
                continue  # keep the longer of parallel chains
            G.add_edge(node, end, weight=length, path=path)

    return G, loop_paths


def _walk_loop(pts):
    """Walk a pure loop component (all pixels degree 2) into an ordered path."""
    start = next(iter(pts))
    path = [start]
    visited = {start}
    cur = start
    while True:
        nxt = [n for n in _neighbors(cur, pts) if n not in visited]
        if not nxt:
            break
        cur = nxt[0]
        visited.add(cur)
        path.append(cur)
    return path


def _longest_path_nodes(G):
    """
    Longest weighted path (graph diameter) via double Dijkstra: farthest node A
    from an arbitrary start, then farthest node B from A. Exact for trees;
    reduced skeleton graphs are nearly trees. Returns (node list, length).
    """
    start = next(iter(G.nodes))
    l1 = nx.single_source_dijkstra_path_length(G, start, weight="weight")
    node_a = max(l1, key=l1.get)
    paths_a = nx.single_source_dijkstra_path(G, node_a, weight="weight")
    lengths_a = nx.single_source_dijkstra_path_length(G, node_a, weight="weight")
    node_b = max(lengths_a, key=lengths_a.get)
    return paths_a[node_b], lengths_a[node_b]


def _reconstruct(G, node_path):
    """Concatenate edge pixel-paths along a node sequence into one pixel path."""
    pixels = []
    for a, b in zip(node_path[:-1], node_path[1:]):
        seg = list(G[a][b]["path"])
        if seg[0] != a:
            seg = seg[::-1]
        if pixels and pixels[-1] == seg[0]:
            pixels.extend(seg[1:])
        else:
            pixels.extend(seg)
    return pixels


def _path_to_linestring(path_nodes, geotransform):
    """Convert an ordered list of (row, col) pixels to a world LineString."""
    origin_x, pixel_w, _, origin_y, _, pixel_h_neg = geotransform
    pixel_h = -pixel_h_neg
    coords = [
        (origin_x + (c + 0.5) * pixel_w, origin_y - (r + 0.5) * pixel_h)
        for (r, c) in path_nodes
    ]
    return LineString(coords)


def extract_centerlines(coords, deg_map, geotransform, min_length_m):
    """
    Iterative longest-path extraction for one connected component, given its
    skeleton pixel coordinates and the global degree map (a skeleton pixel's
    8-neighbours are all in the same component, so the global degree equals
    the component degree and the raster is skeletonized only once).

    Repeatedly: take the diameter path of the current graph, emit it if long
    enough, remove its edges and process the remaining sub-graphs the same
    way. Leftover branches shorter than min_length_m are dropped.
    """
    min_length_px = min_length_m / abs(geotransform[1])

    pts = set(map(tuple, coords.tolist()))
    if len(pts) < 2:
        return []

    deg = {(r, c): int(deg_map[r, c]) for (r, c) in pts}
    nodes = {p for p in pts if deg[p] != 2}

    lines = []

    # Pure loop component (no endpoints or junctions)
    if not nodes:
        path = _walk_loop(pts)
        if _chain_length_px(path) >= min_length_px:
            lines.append(_path_to_linestring(path, geotransform))
        return lines

    G, loop_paths = _build_reduced_graph(pts, deg, nodes)

    for lp in loop_paths:
        if _chain_length_px(lp) >= min_length_px:
            lines.append(_path_to_linestring(lp, geotransform))

    queue = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    while queue:
        sub = queue.pop()
        if sub.number_of_edges() == 0:
            continue

        node_path, total_len = _longest_path_nodes(sub)
        if total_len < min_length_px:
            continue

        lines.append(_path_to_linestring(_reconstruct(sub, node_path), geotransform))

        for a, b in zip(node_path[:-1], node_path[1:]):
            if sub.has_edge(a, b):
                sub.remove_edge(a, b)
        for c in nx.connected_components(sub):
            s2 = sub.subgraph(c).copy()
            if s2.number_of_edges() >= 1:
                queue.append(s2)

    return lines


def probability_to_centerlines(
    prob_raster,
    geotransform,
    epsg,
    threshold=0.5,
    min_component_px=50,
    closing_radius_px=3,
    min_line_length_m=50,
    simplify_tolerance_m=10,
    corridor_mask=None,
):
    """
    Probability raster -> levee centerlines:
    threshold -> (optional corridor mask) -> closing -> remove small blobs
    -> skeletonize once -> per component longest-path -> simplify -> filter.
    Returns a GeoDataFrame of LineStrings.
    """
    above = prob_raster > threshold
    print(f"    pixels > {threshold}: {above.sum():,}")

    if corridor_mask is not None:
        binary = above & corridor_mask
        print(
            f"    after corridor mask:  {binary.sum():,} "
            f"({100 * binary.sum() / max(above.sum(), 1):.0f}% kept)"
        )
    else:
        binary = above

    if closing_radius_px > 0:
        binary = binary_closing(binary, disk(closing_radius_px))
    binary = remove_small_objects(binary, min_size=min_component_px, connectivity=2)

    if binary.sum() == 0:
        return gpd.GeoDataFrame({"length_m": []}, geometry=[], crs=f"EPSG:{epsg}")

    # Skeletonize the whole mask once, then split into components on the
    # skeleton; per-component skeletonization is far too slow on large rasters
    skel = skeletonize(binary)
    deg_map = _degree_map(skel)
    labels, n_comp = label_components(skel, connectivity=2, return_num=True)
    print(f"    connected components: {n_comp}")

    coords_all = np.argwhere(skel)
    if coords_all.size == 0:
        return gpd.GeoDataFrame({"length_m": []}, geometry=[], crs=f"EPSG:{epsg}")
    comp_ids = labels[coords_all[:, 0], coords_all[:, 1]]
    order = np.argsort(comp_ids, kind="stable")
    coords_all = coords_all[order]
    comp_ids = comp_ids[order]
    cut = np.flatnonzero(np.diff(comp_ids)) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [len(comp_ids)]))

    all_lines = []
    for s, e in tqdm(zip(starts, ends), total=len(starts), desc="Vectorizing components"):
        comp_lines = extract_centerlines(
            coords_all[s:e], deg_map, geotransform, min_line_length_m
        )
        all_lines.extend(comp_lines)

    print(f"    extracted paths:      {len(all_lines)}")

    if not all_lines:
        return gpd.GeoDataFrame({"length_m": []}, geometry=[], crs=f"EPSG:{epsg}")

    # Simplify to remove pixel-staircase vertices
    if simplify_tolerance_m > 0:
        all_lines = [
            ln.simplify(simplify_tolerance_m, preserve_topology=False)
            for ln in all_lines
        ]

    # Length filter after simplification
    n_before = len(all_lines)
    all_lines = [ln for ln in all_lines if ln.length >= min_line_length_m]
    print(
        f"    after length filter:  {len(all_lines)} (removed {n_before - len(all_lines)})"
    )

    gdf = gpd.GeoDataFrame(
        {"length_m": [ln.length for ln in all_lines]},
        geometry=all_lines,
        crs=f"EPSG:{epsg}",
    )
    return gdf
