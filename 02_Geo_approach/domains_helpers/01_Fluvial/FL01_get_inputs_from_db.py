"""
Fill the fluvial setup YAML (main_setup_fluvial.yaml) from the EUFL catalogue DB.

Mirrors PL01_get_inputs_from_db.py, but writes the *fluvial* dict shapes that
tu_input_sources.py consumes:
  d_DTM_path:            {AREA: {resolution: link}}
  manning_default:       {resolution: link}
  zsh_<x>_L / zsh_<x>_P: {AREA: link}     (lines and points kept in two keys)
  DTM_add_path:          {dtm_buildings: link, dtm_wb: link}

Soil, drainage and virtual-pipes are intentionally not fetched - the fluvial
pipeline does not use them.

Set the inputs below and run the script directly.
"""

import re
import sys
from pathlib import Path

import yaml

# auth.py / db_select.py live with the pluvial scripts; add that folder to the path.
_DB_HELPERS_DIR = (
    Path(__file__).resolve().parent.parent / "02_Pluvial" / "02_extract_domains"
)
if str(_DB_HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(_DB_HELPERS_DIR))

from auth import get_db_password
from db_select import _build_engine


# ---- INPUTS ------------------------------------------------------------------
AREA_ID = "N05"
DTM_RESOLUTIONS = [1, 20]  # DTM resolutions to fetch for this area (match the domain DTM combos)
CATALOGUE_RESOLUTION = 10  # resolution at which manning / zsh lines are catalogued (5 m fallback applied)

# Catalogue type_ids. Defaults copied from PL01 - VERIFY these are correct for the fluvial datasets.
TYPE_IDS_DTM = [5]
TYPE_IDS_MANNING = [11]
TYPE_IDS_ZSH_CULVERT = [10]
TYPE_IDS_ZSH_LEVEE = [8]
TYPE_IDS_ZSH_CHANNEL = [9]
TYPE_IDS_DTM_BUILDINGS = [16]
TYPE_IDS_DTM_WB = [17]

# DO NOT EDIT BELOW
OUTPUT_PATH = Path(__file__).parent / "main_setup_fluvial.yaml"
TEMPLATE_PATH = Path(__file__).parent / "setup_template.yaml"

if not re.match(r"^[A-Z]\d{2}$", AREA_ID):
    raise ValueError("AREA_ID must look like N05, D12, etc.")
AREA_PREFIX = AREA_ID[0]

if not TEMPLATE_PATH.exists():
    raise FileNotFoundError(f"Template file not found: {TEMPLATE_PATH}")

with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    setup_data = yaml.safe_load(f) or {}


def _ensure_dict(container, key):
    """Return container[key] as a dict, replacing a missing/None value with {}."""
    if container.get(key) is None:
        container[key] = {}
    return container[key]


def _resolution_fallbacks():
    """Resolutions to try for manning / zsh lookups, in order, de-duplicated."""
    ordered = []
    for r in [CATALOGUE_RESOLUTION, 5]:
        if r not in ordered:
            ordered.append(r)
    return ordered


password = get_db_password()
if not password:
    raise RuntimeError("DB password is required to fetch inputs from database.")

engine = _build_engine(password)
warnings = []


# ---- DTM (per area, per resolution) -----------------------------------------
area_dtm = _ensure_dict(_ensure_dict(setup_data, "d_DTM_path"), AREA_ID)
with engine.connect() as conn:
    for res in DTM_RESOLUTIONS:
        link = None
        for type_id in TYPE_IDS_DTM:
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
                (AREA_ID, type_id, res),
            ).fetchone()
            if row and row[0]:
                link = row[0]
                break
        area_dtm[res] = link
        if not link:
            warnings.append(f"No DTM link found for area {AREA_ID} at resolution {res}.")


# ---- Manning (regional; one raster keyed under each DTM resolution) ----------
manning_link = None
with engine.connect() as conn:
    for test_res in _resolution_fallbacks():
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

manning_default = _ensure_dict(setup_data, "manning_default")
if manning_link:
    # The fluvial reader looks up manning by the domain's main DTM resolution,
    # so store the (resolution-independent) landuse raster under every DTM res.
    for res in DTM_RESOLUTIONS:
        manning_default[res] = manning_link
else:
    warnings.append(
        f"No manning raster found for area prefix {AREA_PREFIX} (resolutions {_resolution_fallbacks()})."
    )


# ---- ZSH culverts / levees / channels (lines + points split) ----------------
for line_key, point_key, type_ids in [
    ("zsh_culvert_L", "zsh_culvert_P", TYPE_IDS_ZSH_CULVERT),
    ("zsh_levee_L", "zsh_levee_P", TYPE_IDS_ZSH_LEVEE),
    ("zsh_channel_L", "zsh_channel_P", TYPE_IDS_ZSH_CHANNEL),
]:
    line_pick, point_pick = None, None
    with engine.connect() as conn:
        for test_res in _resolution_fallbacks():
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

            # derive the missing counterpart from the one we have
            if line_links and not point_links:
                point_links = [
                    re.sub(r"_L(\.[A-Za-z0-9]+)$", r"_P\1", x, flags=re.IGNORECASE)
                    for x in line_links
                ]
            if point_links and not line_links:
                line_links = [
                    re.sub(r"_P(\.[A-Za-z0-9]+)$", r"_L\1", x, flags=re.IGNORECASE)
                    for x in point_links
                ]

            if line_links and point_links:
                line_pick = line_links[0]
                point_pick = point_links[0]
                break

    _ensure_dict(setup_data, line_key)[AREA_ID] = line_pick
    _ensure_dict(setup_data, point_key)[AREA_ID] = point_pick
    if not (line_pick and point_pick):
        warnings.append(
            f"No {line_key}/{point_key} pair found for area {AREA_ID} (resolutions {_resolution_fallbacks()})."
        )


# ---- DTM additional rasters (flat: key -> link) -----------------------------
dtm_add = _ensure_dict(setup_data, "DTM_add_path")
for add_key, type_ids in [
    ("dtm_buildings", TYPE_IDS_DTM_BUILDINGS),
    ("dtm_wb", TYPE_IDS_DTM_WB),
]:
    selected = None
    with engine.connect() as conn:
        for test_res in list(DTM_RESOLUTIONS) + [None]:
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
                    selected = row[0]
                    break
            if selected:
                break
    dtm_add[add_key] = selected
    if not selected:
        warnings.append(f"No {add_key} link found for area {AREA_ID}.")


# ---- write -------------------------------------------------------------------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump(setup_data, f, sort_keys=False, allow_unicode=False)

print("=" * 80)
print("FL01_get_inputs_from_db finished")
print(f"AREA_ID:         {AREA_ID}")
print(f"DTM_RESOLUTIONS: {DTM_RESOLUTIONS}")
print(f"Template:        {TEMPLATE_PATH}")
print(f"Output:          {OUTPUT_PATH}")
if warnings:
    print("Warnings:")
    for w in warnings:
        print(f"  - {w}")
else:
    print("All requested fields were populated from DB.")
print("=" * 80)
print(
    "Review the generated main_setup_fluvial.yaml, fill any missing fields manually, "
    "then run 01_tu_extract_run.py for domain extraction."
)
