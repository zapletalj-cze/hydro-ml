"""
Minimal Sentinel-1 orbit direction classifier.

Reads metadata from:
1. `.zip` products (from PDF file(s) inside the archive)
2. `.SAFE` folders (from PDF file(s) inside the folder)

Looks only for the keywords:
- ASCENDING
- DESCENDING
"""

import argparse
import io
import importlib
import logging
import shutil
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _load_pdf_reader():
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = importlib.import_module(module_name)
            return module.PdfReader
        except Exception:
            continue
    return None


PDF_READER = _load_pdf_reader()


def detect_direction_from_text(text: str) -> str:
    upper = text.upper()
    if "ASCENDING" in upper:
        return "ASCENDING"
    if "DESCENDING" in upper:
        return "DESCENDING"
    return "UNKNOWN"


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    if PDF_READER is None:
        return ""

    try:
        reader = PDF_READER(io.BytesIO(pdf_bytes))
        chunks = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text:
                chunks.append(page_text)
        return "\n".join(chunks)
    except Exception as exc:
        log.debug("Could not parse PDF bytes: %s", exc)
        return ""


def read_pdf_text_from_zip(path: Path) -> str:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            pdf_candidates = [
                name for name in zf.namelist() if name.lower().endswith(".pdf")
            ]
            for pdf_path in pdf_candidates:
                pdf_text = extract_text_from_pdf_bytes(zf.read(pdf_path))
                if detect_direction_from_text(pdf_text) != "UNKNOWN":
                    return pdf_text
            return ""
    except (zipfile.BadZipFile, KeyError, OSError, PermissionError) as exc:
        log.warning("Could not read ZIP %s: %s", path, exc)
        return ""


def read_pdf_text_from_safe(path: Path) -> str:
    try:
        for pdf_path in sorted(path.rglob("*.pdf")):
            if not pdf_path.is_file():
                continue
            pdf_text = extract_text_from_pdf_bytes(pdf_path.read_bytes())
            if detect_direction_from_text(pdf_text) != "UNKNOWN":
                return pdf_text
        return ""
    except (OSError, PermissionError) as exc:
        log.warning("Could not read SAFE %s: %s", path, exc)
        return ""


def classify_zip(path: Path) -> str:
    text = read_pdf_text_from_zip(path)
    return detect_direction_from_text(text)


def classify_safe(path: Path) -> str:
    text = read_pdf_text_from_safe(path)
    return detect_direction_from_text(text)


def find_products(input_dirs: list[Path], recursive: bool) -> list[Path]:
    products = []
    seen = set()

    for root in input_dirs:
        if not root.is_dir():
            log.warning("Input directory not found, skipping: %s", root)
            continue

        glob_fn = root.rglob if recursive else root.glob

        for p in glob_fn("*.zip"):
            if p.resolve() not in seen:
                seen.add(p.resolve())
                products.append(p)

        for p in glob_fn("*.SAFE"):
            if p.is_dir() and p.resolve() not in seen:
                seen.add(p.resolve())
                products.append(p)

    return sorted(products, key=lambda p: p.name)


def print_result(path: Path, direction: str):
    print(f"{direction:10}  {path}")


def move_product(path: Path, direction: str, target_root: Path) -> Path | None:
    if direction not in {"ASCENDING", "DESCENDING"}:
        return None

    folder_name = "descending" if direction == "DESCENDING" else "ascending"

    if path.parent.name.lower() == folder_name:
        return path

    target_dir = target_root / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / path.name

    if target_path.exists():
        log.warning("Target already exists, skipping move: %s", target_path)
        return None

    moved_path = Path(shutil.move(str(path), str(target_path)))
    return moved_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect ASCENDING/DESCENDING from Sentinel-1 PDF metadata"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        nargs="+",
        required=True,
        metavar="DIR",
        help="Input directory/directories containing .zip and .SAFE products",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input directories recursively",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if PDF_READER is None:
        log.error("Missing PDF parser package. Install one of: pypdf, PyPDF2")
        return

    products = find_products(args.input, recursive=args.recursive)
    target_root = args.input[0].resolve()

    if not products:
        log.warning("No .zip or .SAFE products found.")
        return

    asc = 0
    desc = 0
    unk = 0
    moved = 0

    for p in products:
        if p.suffix.lower() == ".zip":
            direction = classify_zip(p)
        elif p.suffix.upper() == ".SAFE" and p.is_dir():
            direction = classify_safe(p)
        else:
            direction = "UNKNOWN"

        if direction == "ASCENDING":
            asc += 1
        elif direction == "DESCENDING":
            desc += 1
        else:
            unk += 1

        print_result(p, direction)

        moved_path = move_product(p, direction, target_root=target_root)
        if moved_path is not None and moved_path != p:
            moved += 1
            log.info("Moved -> %s", moved_path)

    print("\nSummary")
    print(f"ASCENDING : {asc}")
    print(f"DESCENDING: {desc}")
    print(f"UNKNOWN   : {unk}")
    print(f"MOVED     : {moved}")


if __name__ == "__main__":
    main()
