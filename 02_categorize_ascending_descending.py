"""
s1_orbit_sorter.py
==================
Classifies Sentinel-1 GRD products by orbit direction and creates
symlinks (or copies) into pre-existing ascending/ and descending/ directories.

Original files are NEVER moved or modified.

Directory structure expected:
    /data/
    ├── raw/                  ← input dir(s), originals stay here
    │   ├── S1A_IW_GRDH_...zip
    │   └── S1C_IW_GRDH_...zip
    ├── ascending/            ← target dir (must exist or use --create-dirs)
    └── descending/           ← target dir (must exist or use --create-dirs)

Multiple input dirs are supported:
    python s1_orbit_sorter.py --input /data/raw1 /data/raw2
                              --ascending /data/ascending
                              --descending /data/descending

Classification fallback chain:
    1. manifest.safe XML  (most reliable)
    2. annotation XML
    3. Relative orbit number lookup table (Europe)
    4. UNKNOWN → logged, skipped

Output modes (default: symlinks, use --copy for actual copies):
    --dry-run   : preview only, no file operations
    --copy      : copy files instead of creating symlinks
    --report    : classification report only

Requirements:
    pip install tqdm

Usage:
    python s1_orbit_sorter.py \\
        --input /data/raw \\
        --ascending /data/ascending \\
        --descending /data/descending

    python s1_orbit_sorter.py \\
        --input /data/raw1 /data/raw2 \\
        --ascending /data/ascending \\
        --descending /data/descending \\
        --copy --dry-run
"""

import os
import re
import sys
import json
import shutil
import logging
import zipfile
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from tqdm import tqdm

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

class OrbitDirection(Enum):
    ASCENDING  = "ASCENDING"
    DESCENDING = "DESCENDING"
    UNKNOWN    = "UNKNOWN"


class ExtractionMethod(Enum):
    MANIFEST_XML   = "manifest.safe XML"
    ANNOTATION_XML = "annotation XML"
    NAME_LOOKUP    = "name/orbit lookup"
    UNKNOWN        = "unknown"


@dataclass
class S1Product:
    path: Path
    name: str
    direction: OrbitDirection    = OrbitDirection.UNKNOWN
    method: ExtractionMethod     = ExtractionMethod.UNKNOWN
    relative_orbit: Optional[int] = None
    acquisition_date: Optional[str] = None
    platform: Optional[str]      = None
    error: Optional[str]         = None

    @property
    def is_certain(self) -> bool:
        return self.method != ExtractionMethod.UNKNOWN

    @property
    def is_zip(self) -> bool:
        return self.path.suffix.lower() == ".zip"

    @property
    def is_safe(self) -> bool:
        return self.path.suffix.upper() == ".SAFE" and self.path.is_dir()


# ---------------------------------------------------------------------------
# KNOWN RELATIVE ORBIT → DIRECTION LOOKUP (Europe / Poland)
# ---------------------------------------------------------------------------

_ASC_ORBITS = {
    15, 22, 44, 51, 73, 80, 102, 109, 131, 138, 160, 167, 189,
}
_DESC_ORBITS = {
    8, 29, 37, 58, 66, 87, 95, 116, 124, 145, 153, 174, 182,
}


def _direction_from_orbit(orbit: int) -> Optional[OrbitDirection]:
    if orbit in _ASC_ORBITS:
        return OrbitDirection.ASCENDING
    if orbit in _DESC_ORBITS:
        return OrbitDirection.DESCENDING
    return None


# ---------------------------------------------------------------------------
# XML PARSING
# ---------------------------------------------------------------------------

_NS = {
    "safe": "http://www.esa.int/safe/sentinel-1.0",
    "s1":   "http://www.esa.int/safe/sentinel-1.0/sentinel-1",
}


def _parse_manifest(content: bytes) -> dict:
    result = {}
    try:
        root = ET.fromstring(content)
        for tag in [
            ".//safe:orbitProperties/safe:pass",
            ".//{http://www.esa.int/safe/sentinel-1.0}pass",
        ]:
            el = root.find(tag, _NS)
            if el is not None and el.text:
                result["direction"] = el.text.strip().upper()
                break
        for tag in [
            ".//safe:orbitReference/safe:relativeOrbitNumber[@type='start']",
            ".//safe:relativeOrbitNumber",
            ".//{http://www.esa.int/safe/sentinel-1.0}relativeOrbitNumber",
        ]:
            el = root.find(tag, _NS)
            if el is not None and el.text:
                try:
                    result["relative_orbit"] = int(el.text.strip())
                except ValueError:
                    pass
                break
        for tag in [
            ".//safe:startTime",
            ".//{http://www.esa.int/safe/sentinel-1.0}startTime",
        ]:
            el = root.find(tag, _NS)
            if el is not None and el.text:
                result["date"] = el.text.strip()[:10]
                break
    except ET.ParseError as e:
        result["error"] = str(e)
    return result


def _parse_annotation(content: bytes) -> dict:
    result = {}
    try:
        root = ET.fromstring(content)
        el = root.find(".//generalAnnotation/productInformation/pass")
        if el is not None and el.text:
            result["direction"] = el.text.strip().upper()
        el = root.find(".//relativeOrbitNumber")
        if el is not None and el.text:
            try:
                result["relative_orbit"] = int(el.text.strip())
            except ValueError:
                pass
    except ET.ParseError as e:
        result["error"] = str(e)
    return result


def _extract_from_name(name: str) -> dict:
    result = {}
    m = re.search(r"_(\d{8})T", name)
    if m:
        d = m.group(1)
        result["date"] = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    m = re.match(r"(S1[ABC])", name)
    if m:
        result["platform"] = m.group(1)
    parts = name.split("_")
    if len(parts) >= 7:
        try:
            result["relative_orbit"] = int(parts[6])
        except (ValueError, IndexError):
            pass
    return result


# ---------------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------------

def _apply_name_data(product: S1Product, name_data: dict):
    product.acquisition_date = name_data.get("date")
    product.platform         = name_data.get("platform")
    if "relative_orbit" in name_data:
        product.relative_orbit = name_data["relative_orbit"]


def _try_orbit_lookup(product: S1Product) -> bool:
    if product.relative_orbit is not None:
        direction = _direction_from_orbit(product.relative_orbit)
        if direction is not None:
            product.direction = direction
            product.method    = ExtractionMethod.NAME_LOOKUP
            return True
    return False


def classify_zip(path: Path) -> S1Product:
    product = S1Product(path=path, name=path.stem)
    _apply_name_data(product, _extract_from_name(path.name))

    try:
        with zipfile.ZipFile(path, "r") as zf:
            namelist = zf.namelist()

            # Strategy 1: manifest.safe
            manifests = [f for f in namelist if f.endswith("manifest.safe")]
            if manifests:
                data = _parse_manifest(zf.read(manifests[0]))
                if "direction" in data:
                    product.direction      = OrbitDirection[data["direction"]]
                    product.method         = ExtractionMethod.MANIFEST_XML
                    product.relative_orbit = data.get("relative_orbit", product.relative_orbit)
                    product.acquisition_date = data.get("date", product.acquisition_date)
                    return product

            # Strategy 2: annotation XML
            annotations = [
                f for f in namelist
                if "/annotation/" in f
                and f.endswith(".xml")
                and "calibration" not in f
                and "noise" not in f
            ]
            if annotations:
                data = _parse_annotation(zf.read(annotations[0]))
                if "direction" in data:
                    product.direction      = OrbitDirection[data["direction"]]
                    product.method         = ExtractionMethod.ANNOTATION_XML
                    product.relative_orbit = data.get("relative_orbit", product.relative_orbit)
                    return product

    except (zipfile.BadZipFile, KeyError, PermissionError, OSError) as e:
        product.error = str(e)
        log.debug(f"ZIP error for {path.name}: {e}")

    # Strategy 3: orbit number lookup
    _try_orbit_lookup(product)
    return product


def classify_safe(path: Path) -> S1Product:
    product = S1Product(path=path, name=path.stem)
    _apply_name_data(product, _extract_from_name(path.name))

    # Strategy 1: manifest.safe
    manifest = path / "manifest.safe"
    if manifest.exists():
        try:
            data = _parse_manifest(manifest.read_bytes())
            if "direction" in data:
                product.direction      = OrbitDirection[data["direction"]]
                product.method         = ExtractionMethod.MANIFEST_XML
                product.relative_orbit = data.get("relative_orbit", product.relative_orbit)
                product.acquisition_date = data.get("date", product.acquisition_date)
                return product
        except (PermissionError, OSError) as e:
            product.error = str(e)

    # Strategy 2: annotation XML
    ann_dir = path / "annotation"
    if ann_dir.exists():
        xml_files = [
            f for f in ann_dir.iterdir()
            if f.suffix == ".xml"
            and "calibration" not in f.name
            and "noise" not in f.name
        ]
        if xml_files:
            try:
                data = _parse_annotation(xml_files[0].read_bytes())
                if "direction" in data:
                    product.direction      = OrbitDirection[data["direction"]]
                    product.method         = ExtractionMethod.ANNOTATION_XML
                    product.relative_orbit = data.get("relative_orbit", product.relative_orbit)
                    return product
            except (PermissionError, OSError) as e:
                product.error = str(e)

    # Strategy 3: orbit number lookup
    _try_orbit_lookup(product)
    return product


def classify_product(path: Path) -> Optional[S1Product]:
    if path.suffix.lower() == ".zip":
        return classify_zip(path)
    if path.suffix.upper() == ".SAFE" and path.is_dir():
        return classify_safe(path)
    return None


# ---------------------------------------------------------------------------
# SCANNING
# ---------------------------------------------------------------------------

def scan_dirs(input_dirs: list[Path], recursive: bool = False) -> list[S1Product]:
    candidates = []

    for root in input_dirs:
        if not root.is_dir():
            log.warning(f"Input directory not found, skipping: {root}")
            continue

        glob_fn = root.rglob if recursive else root.glob
        for p in glob_fn("*.zip"):
            if "S1" in p.name:
                candidates.append(p)
        for p in glob_fn("*.SAFE"):
            if p.is_dir():
                candidates.append(p)

    # Deduplicate by resolved path
    seen = set()
    unique = []
    for p in candidates:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    log.info(f"Found {len(unique)} candidate products across {len(input_dirs)} input dir(s)")

    products = []
    for path in tqdm(unique, desc="Classifying", unit="product"):
        product = classify_product(path)
        if product:
            products.append(product)

    return products


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def print_report(products: list[S1Product]):
    asc   = [p for p in products if p.direction == OrbitDirection.ASCENDING]
    desc  = [p for p in products if p.direction == OrbitDirection.DESCENDING]
    unk   = [p for p in products if p.direction == OrbitDirection.UNKNOWN]

    print(f"\n{'='*72}")
    print(f"  SENTINEL-1 ORBIT CLASSIFICATION REPORT")
    print(f"{'='*72}")
    print(f"  Total      : {len(products)}")
    print(f"  ASCENDING  : {len(asc)}")
    print(f"  DESCENDING : {len(desc)}")
    print(f"  UNKNOWN    : {len(unk)}")
    print(f"{'='*72}")

    for label, group in [("ASCENDING", asc), ("DESCENDING", desc), ("UNKNOWN", unk)]:
        if not group:
            continue
        print(f"\n  {label} ({len(group)})")
        print(f"  {'-'*68}")
        for p in sorted(group, key=lambda x: x.acquisition_date or ""):
            orbit = f"rel_orbit={p.relative_orbit:03d}" if p.relative_orbit else "rel_orbit=???"
            date  = p.acquisition_date or "????"
            cert  = "" if p.is_certain else "  ⚠ unverified"
            print(f"    {date}  {orbit}  [{p.method.value}]{cert}")
            print(f"             {p.name[:62]}")
            if p.error:
                print(f"             ⚠ {p.error}")
    print(f"\n{'='*72}\n")


# ---------------------------------------------------------------------------
# FILE OPERATIONS
# ---------------------------------------------------------------------------

def link_or_copy(
    products: list[S1Product],
    asc_dir: Path,
    desc_dir: Path,
    copy: bool = False,
    dry_run: bool = False,
    create_dirs: bool = False,
):
    """
    Links (symlink) or copies products into ascending/descending dirs.
    Originals are never touched.
    UNKNOWN products are skipped with a warning.
    """
    target_map = {
        OrbitDirection.ASCENDING:  asc_dir,
        OrbitDirection.DESCENDING: desc_dir,
    }

    if not dry_run and create_dirs:
        for d in target_map.values():
            d.mkdir(parents=True, exist_ok=True)

    for d in target_map.values():
        if not dry_run and not d.exists():
            log.error(f"Target directory does not exist: {d}  (use --create-dirs)")
            sys.exit(1)

    op = "COPY" if copy else "SYMLINK"
    skipped_unknown = 0

    for p in tqdm(products, desc=op, unit="product"):
        if p.direction == OrbitDirection.UNKNOWN:
            log.warning(f"  Skipping UNKNOWN: {p.name[:60]}")
            skipped_unknown += 1
            continue

        target_dir = target_map[p.direction]
        target     = target_dir / p.path.name

        if target.exists() or target.is_symlink():
            log.debug(f"  Already exists: {target.name}")
            continue

        if dry_run:
            src = p.path.resolve()
            print(f"  [{op}] {p.direction.value[:4]}  {p.path.name} → {target_dir}/")
            continue

        try:
            if copy:
                if p.path.is_dir():
                    shutil.copytree(p.path, target)
                else:
                    shutil.copy2(p.path, target)
            else:
                target.symlink_to(p.path.resolve())
            log.info(f"  ✓ {op}: {p.path.name} → {target_dir.name}/")
        except Exception as e:
            log.error(f"  ✗ {op} failed for {p.path.name}: {e}")

    if skipped_unknown:
        log.warning(f"\n  {skipped_unknown} products skipped (UNKNOWN direction). "
                    f"Check manifest.safe or verify manually.")


def export_json(products: list[S1Product], output_path: Path):
    data = [
        {
            "name":           p.name,
            "path":           str(p.path.resolve()),
            "direction":      p.direction.value,
            "method":         p.method.value,
            "relative_orbit": p.relative_orbit,
            "date":           p.acquisition_date,
            "platform":       p.platform,
            "certain":        p.is_certain,
            "error":          p.error,
        }
        for p in products
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"Results saved → {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Classify Sentinel-1 GRD products by orbit direction and "
            "populate ascending/descending directories via symlinks or copies. "
            "Original files are never moved or modified."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Report only – no file operations
  python s1_orbit_sorter.py --input /data/raw

  # Symlink into pre-existing dirs (default)
  python s1_orbit_sorter.py --input /data/raw \\
      --ascending /data/ascending --descending /data/descending

  # Copy instead of symlink
  python s1_orbit_sorter.py --input /data/raw \\
      --ascending /data/ascending --descending /data/descending --copy

  # Multiple input dirs, dry run preview
  python s1_orbit_sorter.py --input /data/raw1 /data/raw2 \\
      --ascending /data/asc --descending /data/desc --dry-run

  # Create target dirs if missing
  python s1_orbit_sorter.py --input /data/raw \\
      --ascending /data/asc --descending /data/desc --create-dirs

  # Export JSON report
  python s1_orbit_sorter.py --input /data/raw --export results.json
        """
    )
    parser.add_argument(
        "--input", "-i", type=Path, nargs="+", required=True,
        metavar="DIR",
        help="Input directory/directories containing Sentinel-1 products (not modified)"
    )
    parser.add_argument(
        "--ascending", "-a", type=Path, default=None,
        help="Target directory for ASCENDING products"
    )
    parser.add_argument(
        "--descending", "-d", type=Path, default=None,
        help="Target directory for DESCENDING products"
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of creating symlinks (slower but portable)"
    )
    parser.add_argument(
        "--create-dirs", action="store_true",
        help="Create ascending/descending dirs if they do not exist"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Scan input directories recursively"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview operations without executing"
    )
    parser.add_argument(
        "--export", type=Path, default=None,
        help="Export classification results to JSON"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Scan and classify
    products = scan_dirs(args.input, recursive=args.recursive)

    if not products:
        log.warning("No Sentinel-1 products found in input directories.")
        sys.exit(0)

    # Report
    print_report(products)

    # Export JSON
    if args.export:
        export_json(products, args.export)

    # File operations only if both target dirs are specified
    if args.ascending and args.descending:
        link_or_copy(
            products,
            asc_dir=args.ascending,
            desc_dir=args.descending,
            copy=args.copy,
            dry_run=args.dry_run,
            create_dirs=args.create_dirs,
        )
    elif args.ascending or args.descending:
        log.error("Both --ascending and --descending must be specified together.")
        sys.exit(1)
    else:
        log.info("No target directories specified – report only mode.")

    asc_n  = sum(1 for p in products if p.direction == OrbitDirection.ASCENDING)
    desc_n = sum(1 for p in products if p.direction == OrbitDirection.DESCENDING)
    unk_n  = sum(1 for p in products if p.direction == OrbitDirection.UNKNOWN)
    log.info(f"Done.  ASC={asc_n}  DESC={desc_n}  UNKNOWN={unk_n}")


if __name__ == "__main__":
    main()
