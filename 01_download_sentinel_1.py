"""
Sentinel-1 GRD – Download via Copernicus Dataspace (CDSE)
============================================================
Search:   OAuth2 Client Credentials → catalogue search
Download: CDSE S3 protocol via boto3 → no 2FA issue

Requirements:
    pip install requests requests-oauthlib boto3 tqdm cryptography

Credentials are managed via credential_manager.py (Fernet encrypted storage).
On first run, GUI dialogs will prompt for credentials and store them securely.

OAuth client  → https://shapps.dataspace.copernicus.eu/dashboard/
S3 credentials → https://eodata-iam.dataspace.copernicus.eu

Usage:
    python s1_download.py --dry-run       # search only, no download
    python s1_download.py --orbit ASC     # ascending only
    python s1_download.py --orbit DESC    # descending only
    python s1_download.py --max 5         # max 5 scenes per orbit (test)
    python s1_download.py                 # download all matched scenes
"""

import logging
import argparse
from pathlib import Path

import boto3
import botocore
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient
from tqdm import tqdm

from credential_manager import get_oauth_credentials, get_s3_credentials

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BOUNDING_BOX = {
    "west":  17.9,
    "south": 53.0,
    "east":  19.1,
    "north": 54.4,
}

DATE_START = "2025-04-01T00:00:00.000Z"
DATE_END   = "2025-10-31T23:59:59.999Z"

OUTPUT_BASE = Path("./sentinel1_data")

TOKEN_URL     = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# CDSE S3 endpoint – no 2FA required
S3_ENDPOINT   = "https://eodata.dataspace.copernicus.eu"
S3_BUCKET     = "eodata"

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
# AUTHENTICATION
# ---------------------------------------------------------------------------

def create_search_session(client_id: str, client_secret: str) -> OAuth2Session:
    """OAuth2 Client Credentials – catalogue search only."""
    log.info("Fetching search token (client credentials)...")
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)
    oauth.fetch_token(
        token_url=TOKEN_URL,
        client_secret=client_secret,
        include_client_id=True,
    )
    log.info("Search session ready.")
    return oauth


def create_s3_client(access_key: str, secret_key: str):
    """
    boto3 S3 client for CDSE eodata endpoint.
    Uses S3 credentials – independent of 2FA.
    """
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="default",
        config=botocore.config.Config(signature_version="s3v4"),
    )


# ---------------------------------------------------------------------------
# PRODUCT SEARCH
# ---------------------------------------------------------------------------

def build_filter(orbit_direction: str) -> str:
    w = BOUNDING_BOX["west"]
    s = BOUNDING_BOX["south"]
    e = BOUNDING_BOX["east"]
    n = BOUNDING_BOX["north"]
    bbox_wkt = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"

    return (
        f"Collection/Name eq 'SENTINEL-1'"
        f" and contains(Name,'IW_GRDH')"
        f" and ContentDate/Start ge {DATE_START}"
        f" and ContentDate/Start le {DATE_END}"
        f" and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'orbitDirection'"
        f" and att/Value eq '{orbit_direction}')"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt}')"
        f" and Online eq true"
    )


def search_products(session: OAuth2Session, orbit_direction: str) -> list:
    log.info(f"Searching – {orbit_direction}...")
    products = []
    skip = 0
    page_size = 100

    while True:
        params = {
            "$filter": build_filter(orbit_direction),
            "$orderby": "ContentDate/Start asc",
            "$top": page_size,
            "$skip": skip,
        }
        resp = session.get(CATALOGUE_URL, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json().get("value", [])
        products.extend(batch)
        if len(batch) < page_size:
            break
        skip += page_size

    log.info(f"Found ({orbit_direction}): {len(products)} products")
    return products


def select_products(products: list, max_count: int = None) -> list:
    target = max_count if max_count is not None else len(products)
    if len(products) <= target:
        return products
    step = len(products) / target
    selected = [products[int(i * step)] for i in range(target)]
    log.info(f"Selected {len(selected)} of {len(products)} (uniform sampling)")
    return selected


def print_product_summary(products: list, orbit: str):
    print(f"\n{'='*70}")
    print(f"  {orbit} – {len(products)} products")
    print(f"{'='*70}")
    for i, p in enumerate(products):
        name = p.get("Name", "N/A")
        date = p.get("ContentDate", {}).get("Start", "N/A")[:10]
        size_mb = p.get("ContentLength", 0) / (1024 ** 2)
        print(f"  [{i+1:2d}] {date}  {name[:52]}  ({size_mb:.0f} MB)")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# DOWNLOAD VIA S3
# ---------------------------------------------------------------------------

def get_s3_path(product: dict) -> str | None:
    """
    Extracts S3 path from product S3Path attribute.
    Example: /eodata/Sentinel-1/SAR/GRD/2025/.../S1A_IW_GRDH_...SAFE
    """
    s3_path = product.get("S3Path")
    if not s3_path:
        log.warning(f"  No S3Path for product: {product.get('Name')}")
        return None
    # S3Path includes leading /eodata/ – strip bucket prefix for boto3 key
    # e.g. /eodata/Sentinel-1/... → Sentinel-1/...
    return s3_path.lstrip("/").removeprefix(f"{S3_BUCKET}/")


def download_product_s3(
    product: dict,
    output_dir: Path,
    s3_client,
) -> Path | None:
    """
    Downloads one product via CDSE S3 endpoint.
    Product is a .SAFE directory – downloads all files recursively.
    Output is zipped to match expected .zip format.
    """
    import zipfile

    product_name  = product.get("Name", "unknown")
    s3_prefix     = get_s3_path(product)
    output_zip    = output_dir / f"{product_name}.zip"
    expected_size = product.get("ContentLength", 0)

    if output_zip.exists():
        if output_zip.stat().st_size >= expected_size * 0.99:  # 1% tolerance
            log.info(f"  Skipping (exists): {product_name[:52]}")
            return output_zip
        else:
            log.warning(f"  Incomplete, re-downloading: {product_name[:52]}")
            output_zip.unlink()

    if not s3_prefix:
        return None

    log.info(f"  Downloading via S3: {product_name[:52]}")

    try:
        # List all objects under this product prefix
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix)

        objects = []
        for page in pages:
            objects.extend(page.get("Contents", []))

        if not objects:
            log.error(f"  No S3 objects found at prefix: {s3_prefix}")
            return None

        total_size = sum(o["Size"] for o in objects)
        size_mb = total_size / (1024 ** 2)
        log.info(f"  {len(objects)} files, {size_mb:.0f} MB total")

        # Download and zip in one pass
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            with tqdm(total=total_size, unit="B", unit_scale=True,
                      desc=f"    {product_name[:38]}", leave=False) as pbar:
                for obj in objects:
                    key      = obj["Key"]
                    arc_name = key.removeprefix(
                        s3_prefix.rstrip("/") + "/"
                    )
                    arc_name = f"{product_name}.SAFE/{arc_name}"

                    response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                    data     = response["Body"].read()
                    zf.writestr(arc_name, data)
                    pbar.update(len(data))

        log.info(f"  ✓ Done: {product_name[:52]}")
        return output_zip

    except Exception as e:
        log.error(f"  ✗ Error: {e}")
        if output_zip.exists():
            output_zip.unlink()
        return None


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-1 GRD from Copernicus Dataspace via S3"
    )
    parser.add_argument("--orbit", choices=["ASC", "DESC", "BOTH"], default="BOTH")
    parser.add_argument("--max", type=int, default=None,
                        help="Max scenes per orbit direction (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Search only, do not download")
    parser.add_argument("--reset-credentials", action="store_true",
                        help="Delete stored credentials and re-enter")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.reset_credentials:
        from credential_manager import delete_all_credentials
        delete_all_credentials()
        return

    log.info("Sentinel-1 GRD – CDSE S3 download")
    log.info("=" * 55)
    log.info(f"AOI:    W={BOUNDING_BOX['west']} S={BOUNDING_BOX['south']} "
             f"E={BOUNDING_BOX['east']} N={BOUNDING_BOX['north']}")
    log.info(f"Period: {DATE_START[:10]} → {DATE_END[:10]}")
    log.info(f"Orbit:  {args.orbit}")

    # Credentials via GUI (Fernet encrypted storage)
    client_id, client_secret = get_oauth_credentials()
    if not client_id:
        log.error("OAuth credentials not provided. Exiting.")
        return

    access_key, secret_key = get_s3_credentials()
    if not access_key:
        log.error("S3 credentials not provided. Exiting.")
        return

    # Sessions
    search_session = create_search_session(client_id, client_secret)
    s3_client      = create_s3_client(access_key, secret_key)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    asc_dir  = OUTPUT_BASE / "ascending"
    desc_dir = OUTPUT_BASE / "descending"
    asc_dir.mkdir(exist_ok=True)
    desc_dir.mkdir(exist_ok=True)

    orbits = []
    if args.orbit in ("ASC", "BOTH"):
        orbits.append(("ASCENDING", asc_dir))
    if args.orbit in ("DESC", "BOTH"):
        orbits.append(("DESCENDING", desc_dir))

    all_downloaded = []

    for orbit_direction, output_dir in orbits:
        products = search_products(search_session, orbit_direction)
        selected = select_products(products, max_count=args.max)
        print_product_summary(selected, orbit_direction)

        if args.dry_run:
            log.info("Dry-run – skipping download.")
            continue

        log.info(f"Starting download – {orbit_direction} ({len(selected)} scenes)")
        for i, product in enumerate(selected):
            log.info(f"\nScene {i+1}/{len(selected)}")
            result = download_product_s3(product, output_dir, s3_client)
            if result:
                all_downloaded.append(result)

    if not args.dry_run:
        print(f"\n{'='*55}")
        print(f"DONE: {len(all_downloaded)} files downloaded")
        if all_downloaded:
            total_gb = sum(f.stat().st_size for f in all_downloaded) / (1024 ** 3)
            print(f"Total size: {total_gb:.1f} GB")
        print(f"Output: {OUTPUT_BASE.resolve()}")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
