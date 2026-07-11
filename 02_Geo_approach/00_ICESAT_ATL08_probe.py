"""
ATL08 filter probe - one granule, measured survival per filter
==============================================================

Opens ONE ATL08 granule and prints, aggregated over all beams:
  - per-field fill/NaN rates (h_te_median, h_te_best_fit_20m, h_te_uncertainty,
    n_te_photons, layer_flag, cloud_flag_atm, sat_flag, terrain_flg)
  - survival counts of each individual filter
  - survival of the OLD combined mask (strict: unc finite & <=1.5, photons>=50,
    terrain flag on) vs the NEW one (pass-through fills, unc<=2.5, photons>=20,
    terrain flag off)

Purpose: identify by measurement which condition dominates the filtering,
instead of guessing. Run on one granule, paste the output back.

Usage: set GRANULE below, run. Requires numpy + h5py only.
"""

import numpy as np
import h5py

# ============================================================
GRANULE = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\ATL08\ATL08_20251019002542_05262906_007_01.h5"   # <- one file
BEAMS = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]
FILL_ABS = 1e30
# ============================================================


def mask_fill(a):
    a = np.asarray(a, dtype=np.float64)
    a[np.abs(a) >= FILL_ABS] = np.nan
    return a


def col(group, name, n):
    return group[name][:] if (group is not None and name in group) else np.full(n, np.nan)


def main():
    agg = {}

    def add(key, arr):
        agg.setdefault(key, []).append(np.asarray(arr))

    with h5py.File(GRANULE, "r") as f:
        for beam in BEAMS:
            if beam not in f or f"{beam}/land_segments" not in f:
                continue
            ls = f[f"{beam}/land_segments"]
            terr = ls.get("terrain")
            n = len(ls["latitude"])

            add("h_te", mask_fill(col(terr, "h_te_median", n)))
            h20 = terr["h_te_best_fit_20m"][:] if (terr is not None and
                  "h_te_best_fit_20m" in terr) else np.full((n, 5), np.nan)
            add("h20_any", np.isfinite(mask_fill(h20)).any(axis=1))
            add("unc", mask_fill(col(terr, "h_te_uncertainty", n)))
            add("n_te", np.asarray(col(terr, "n_te_photons", n), dtype=np.float64))
            add("layer", np.asarray(col(ls, "layer_flag", n), dtype=np.float64))
            add("cloud", np.asarray(col(ls, "cloud_flag_atm", n), dtype=np.float64))
            add("sat", np.asarray(col(ls, "sat_flag", n), dtype=np.float64))
            add("tflg", np.asarray(col(ls, "terrain_flg", n), dtype=np.float64))

    d = {k: np.concatenate(v) for k, v in agg.items()}
    N = len(d["h_te"])
    pct = lambda m: f"{100.0 * np.sum(m) / N:6.1f} %  ({int(np.sum(m)):,})"

    print(f"granule: {GRANULE}")
    print(f"segments total: {N:,}\n")

    print("--- field fill rates (share of NaN/fill) ---")
    for k, lab in (("h_te", "h_te_median"), ("unc", "h_te_uncertainty"),
                   ("n_te", "n_te_photons"), ("layer", "layer_flag"),
                   ("cloud", "cloud_flag_atm"), ("sat", "sat_flag"),
                   ("tflg", "terrain_flg")):
        print(f"  {lab:18s} fill: {pct(~np.isfinite(d[k]))}")
    print(f"  h20 any-valid            : {pct(d['h20_any'].astype(bool))}\n")

    unc, nte = d["unc"], d["n_te"]
    layer, cloud, sat, tflg = d["layer"], d["cloud"], d["sat"], d["tflg"]

    print("--- individual filters (survivors) ---")
    print(f"  h_te finite              : {pct(np.isfinite(d['h_te']))}")
    print(f"  h20 any finite           : {pct(d['h20_any'].astype(bool))}")
    print(f"  unc finite & <=1.5       : {pct(np.isfinite(unc) & (unc <= 1.5))}")
    print(f"  unc <=2.5 (fill passes)  : {pct(np.where(np.isfinite(unc), unc <= 2.5, True))}")
    print(f"  photons >=50 (strict)    : {pct(nte >= 50)}")
    print(f"  photons >=20 (fill pass) : {pct(np.where(np.isfinite(nte), nte >= 20, True))}")
    print(f"  layer_flag == 0          : {pct(np.where(np.isfinite(layer), layer == 0, True))}")
    print(f"  cloud_flag_atm <= 1      : {pct(np.where(np.isfinite(cloud), cloud <= 1, True))}")
    print(f"  sat_flag == 0            : {pct(np.where(np.isfinite(sat), sat == 0, True))}")
    print(f"  terrain_flg == 0         : {pct(np.where(np.isfinite(tflg), tflg == 0, True))}\n")

    old = (np.isfinite(d["h_te"])
           & np.isfinite(unc) & (unc <= 1.5)
           & (nte >= 50)
           & np.where(np.isfinite(layer), layer == 0, True)
           & np.where(np.isfinite(cloud), cloud <= 1, True)
           & np.where(np.isfinite(sat), sat == 0, True)
           & np.where(np.isfinite(tflg), tflg == 0, True))
    new = (d["h20_any"].astype(bool)
           & np.where(np.isfinite(unc), unc <= 2.5, True)
           & np.where(np.isfinite(nte), nte >= 20, True)
           & np.where(np.isfinite(layer), layer == 0, True)
           & np.where(np.isfinite(cloud), cloud <= 1, True)
           & np.where(np.isfinite(sat), sat == 0, True))
    print("--- combined masks ---")
    print(f"  OLD (strict)             : {pct(old)}")
    print(f"  NEW (relaxed, no tflg)   : {pct(new)}")
    if int(np.sum(old)) == int(np.sum(new)):
        print("\n  !! identical survivors -> the script you run is likely NOT the edited one")


if __name__ == "__main__":
    main()