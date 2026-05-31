"""
Fill pluvial setup YAML from EUFL catalogue DB.

Set inputs below and run the script directly.
"""

from auth import get_db_password
from db_select import _build_engine
import re
import sys
from pathlib import Path
import yaml

AREA_ID = "N05"
DTM_RESOLUTION = 1
MESH_RESOLUTION = 10

# DO NOT EDIT BELOW
OUTPUT_PATH = (
    Path(__file__).parent
    / "_yaml"
    / f"setup_{AREA_ID}_{DTM_RESOLUTION}m_{MESH_RESOLUTION}m.yaml"
)

TEMPLATE_PATH = Path(__file__).parent / "setup_template.yaml"


TYPE_IDS_DTM = [5]
TYPE_IDS_ZSH_CULVERT = [10]
TYPE_IDS_ZSH_LEVEE = [8]
TYPE_IDS_ZSH_CHANNEL = [9]
TYPE_IDS_DTM_BUILDINGS = [16]
TYPE_IDS_DTM_WB = [17]
TYPE_IDS_CWF_BUILDINGS = [16]
TYPE_IDS_VP_PITS = [6]
TYPE_IDS_MANNING = [11]


if not re.match(r"^[A-Z]\d{2}$", AREA_ID):
    raise ValueError("AREA_ID must look like N05, D12, etc.")

AREA_PREFIX = AREA_ID[0]

if not TEMPLATE_PATH.exists():
    raise FileNotFoundError(f"Template file not found: {TEMPLATE_PATH}")

with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    setup_data = yaml.safe_load(f) or {}

password = get_db_password()
if not password:
    raise RuntimeError("DB password is required to fetch inputs from database.")

engine = _build_engine(password)
warnings = []


# get pluvial DTM link - 1 m + buildings, waterbodies
dtm_link = None
with engine.connect() as conn:
    for type_id in TYPE_IDS_DTM:
        row = conn.exec_driver_sql(
            """
            SELECT link
            FROM eufl_catalogue.dataset
            WHERE (area_id = %s)
              AND type_id = %s
              AND resolution = %s
            ORDER BY version DESC
            LIMIT 1
            """,
            (AREA_ID, type_id, DTM_RESOLUTION),
        ).fetchone()
        if row and row[0]:
            dtm_link = row[0]
            break

setup_data.setdefault("d_DTM_path", {}).setdefault(AREA_ID, {})[
    DTM_RESOLUTION
] = dtm_link
if not dtm_link:
    warnings.append(
        f"No DTM link found for area {AREA_ID} and resolution {DTM_RESOLUTION}."
    )


# get soil all layers
soil_layers = {}
with engine.connect() as conn:
    for layer_name, type_id in [
        ("layer_0_30", 13),
        ("layer_31_60", 14),
        ("layer_61_100", 14),
        ("layer_0_100", 12),
    ]:
        row = conn.exec_driver_sql(
            """
            SELECT link
            FROM eufl_catalogue.dataset
            WHERE area_id = %s
              AND type_id = %s
              AND (resolution = %s OR resolution IS NULL)
            ORDER BY version DESC
            LIMIT 1
            """,
            (AREA_PREFIX, type_id, MESH_RESOLUTION),
        ).fetchone()
        if row and row[0]:
            soil_layers[layer_name] = row[0]

if soil_layers:
    setup_data.setdefault("soil_file", {}).setdefault(AREA_ID, {})[
        MESH_RESOLUTION
    ] = soil_layers
else:
    warnings.append(
        f"No soil layers found for area prefix {AREA_PREFIX} and mesh resolution {MESH_RESOLUTION}."
    )

# get materials layers, all 
manning_link = None

with engine.connect() as conn:
    for test_res in [MESH_RESOLUTION, 5]:
        for type_id in TYPE_IDS_MANNING:
            row = conn.exec_driver_sql(
                """
                SELECT link
                FROM eufl_catalogue.dataset
                WHERE area_id = %s
                  AND type_id = %s
                  AND resolution = %s
                ORDER BY version DESC
                LIMIT 1
                """,
                (AREA_PREFIX, type_id, test_res),
            ).fetchone()
            if row and row[0]:
                manning_link = row[0]
                break
        if manning_link:
            break

if manning_link:
    setup_data.setdefault("manning_default", {})[MESH_RESOLUTION] = manning_link
    setup_data.setdefault("manning_path_local_raster", {}).setdefault(AREA_ID, {})[
        MESH_RESOLUTION
    ] = manning_link
else:
    if not TYPE_IDS_MANNING:
        warnings.append("Manning type_id is not set. Please provide TYPE_IDS_MANNING.")
    else:
        warnings.append(
            f"No manning raster found for area {AREA_ID} and mesh resolution {MESH_RESOLUTION}."
        )


# get ZSH culverts, levees and channels
for setup_key, type_ids in [
    ("zsh_culvert", TYPE_IDS_ZSH_CULVERT),
    ("zsh_levee", TYPE_IDS_ZSH_LEVEE),
    ("zsh_channel", TYPE_IDS_ZSH_CHANNEL),
]:
    picked_links = []

    with engine.connect() as conn:
        for test_res in [MESH_RESOLUTION, 5]:
            all_links = []
            for type_id in type_ids:
                rows = conn.exec_driver_sql(
                    """
                    SELECT link
                    FROM eufl_catalogue.dataset
                    WHERE (area_id = %s OR area_id LIKE %s)
                      AND type_id = %s
                      AND resolution = %s
                    ORDER BY version DESC
                    LIMIT 50
                    """,
                    (AREA_ID, f"{AREA_PREFIX}%", type_id, test_res),
                ).fetchall()
                all_links.extend([r[0] for r in rows if r and r[0]])

            line_links = [x for x in all_links if "_L." in str(x).upper()]
            point_links = [x for x in all_links if "_P." in str(x).upper()]

            if line_links and not point_links:
                for line in line_links:
                    point_links.append(
                        re.sub(
                            r"_L(\.[A-Za-z0-9]+)$", r"_P\1", line, flags=re.IGNORECASE
                        )
                    )
            if point_links and not line_links:
                for point in point_links:
                    line_links.append(
                        re.sub(
                            r"_P(\.[A-Za-z0-9]+)$", r"_L\1", point, flags=re.IGNORECASE
                        )
                    )

            if line_links and point_links:
                count = min(len(line_links), len(point_links))
                for i in range(count):
                    picked_links.append(line_links[i])
                    picked_links.append(point_links[i])
                break

    setup_data.setdefault(setup_key, {})[MESH_RESOLUTION] = picked_links
    if not picked_links:
        warnings.append(
            f"No {setup_key} links found for mesh resolution {MESH_RESOLUTION} (or fallback 5m)."
        )


# buildings, waterbodies, cwf
for dtm_add_key, type_ids in [
    ("dtm_buildings", TYPE_IDS_DTM_BUILDINGS),
    ("dtm_wb", TYPE_IDS_DTM_WB),
    ("cwf_buildings", TYPE_IDS_CWF_BUILDINGS),
]:
    selected_link = None

    with engine.connect() as conn:
        for test_res in [DTM_RESOLUTION, MESH_RESOLUTION, None]:
            for type_id in type_ids:
                if test_res is None:
                    row = conn.exec_driver_sql(
                        """
                        SELECT link
                        FROM eufl_catalogue.dataset
                        WHERE (area_id = %s OR area_id LIKE %s)
                          AND type_id = %s
                        ORDER BY version DESC
                        LIMIT 1
                        """,
                        (AREA_ID, f"{AREA_PREFIX}%", type_id),
                    ).fetchone()
                else:
                    row = conn.exec_driver_sql(
                        """
                        SELECT link
                        FROM eufl_catalogue.dataset
                        WHERE (area_id = %s OR area_id LIKE %s)
                          AND type_id = %s
                          AND resolution = %s
                        ORDER BY version DESC
                        LIMIT 1
                        """,
                        (AREA_ID, f"{AREA_PREFIX}%", type_id, test_res),
                    ).fetchone()

                if row and row[0]:
                    selected_link = row[0]
                    break
                if selected_link:
                    break
            if selected_link:
                break

    setup_data.setdefault("DTM_add_path", {}).setdefault(dtm_add_key, {}).setdefault(
        AREA_ID, {}
    )[DTM_RESOLUTION] = selected_link
    if not selected_link:
        warnings.append(
            f"No {dtm_add_key} link found for area {AREA_ID} (dtm={DTM_RESOLUTION}, mesh={MESH_RESOLUTION})."
        )


# Virtual pipes 
vp_link = None

with engine.connect() as conn:
    for test_res in [MESH_RESOLUTION, None]:
        if test_res is None:
            row = conn.exec_driver_sql(
                """
                SELECT link
                FROM eufl_catalogue.dataset
                WHERE (area_id = %s OR area_id LIKE %s)
                  AND type_id = %s
                ORDER BY version DESC
                LIMIT 1
                """,
                (AREA_ID, f"{AREA_PREFIX}%", TYPE_IDS_VP_PITS[0]),
            ).fetchone()
        else:
            row = conn.exec_driver_sql(
                """
                SELECT link
                FROM eufl_catalogue.dataset
                WHERE (area_id = %s OR area_id LIKE %s)
                  AND type_id = %s
                  AND resolution = %s
                ORDER BY version DESC
                LIMIT 1
                """,
                (AREA_ID, f"{AREA_PREFIX}%", TYPE_IDS_VP_PITS[0], test_res),
            ).fetchone()

        if row and row[0]:
            vp_link = row[0]
            break

setup_data.setdefault("VP_pit_points", {}).setdefault(AREA_ID, {})[
    MESH_RESOLUTION
] = vp_link
if not vp_link:
    warnings.append(
        f"No VP pit points link found for area {AREA_ID} and mesh resolution {MESH_RESOLUTION}."
    )


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump(setup_data, f, sort_keys=False, allow_unicode=False)

print("=" * 80)
print("PL01_get_inputs_from_db finished")
print(f"AREA_ID: {AREA_ID}")
print(f"DTM_RESOLUTION: {DTM_RESOLUTION}")
print(f"MESH_RESOLUTION: {MESH_RESOLUTION}")
print(f"Template: {TEMPLATE_PATH}")
print(f"Output:   {OUTPUT_PATH}")
if warnings:
    print("Warnings:")
    for w in warnings:
        print(f"  - {w}")
else:
    print("All requested fields were populated from DB.")
print("=" * 80)
print("The next step is to review the generated YAML file, add any missing fields manually, and then use it for domain extraction in PL02_extract_domains.py.")