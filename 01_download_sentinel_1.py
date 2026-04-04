"""
Sentinel-1 GRD – Download via Copernicus Dataspace (CDSE)
============================================================
Authentication via OAuth2 Client Credentials – does not require 2FA TOTP.

Requirements:
    pip install requests requests-oauthlib tqdm

Preparing OAuth Client:
    1. Sign in at https://shapps.dataspace.copernicus.eu/dashboard/
    2. Navigate to "OAuth Clients" → "New OAuth Client"
    3. Copy Client ID and Client Secret
    4. Save as environment variables (see below) or enter interactively

Recommended credential storage (never hardcode):
    export CDSE_CLIENT_ID="your_client_id"
    export CDSE_CLIENT_SECRET="your_client_secret"

Usage:
    python s1_download.py
    python s1_download.py --orbit ASC     # ascending only
    python s1_download.py --orbit DESC    # descending only
    python s1_download.py --max 5         # download max 5 scenes (test)
    python s1_download.py --dry-run       # search only, no download

Fixes vs. previous version:
    1. OData filter: replaced attribute-based productType/operationalMode/polarisation
       filters with contains(Name,'IW_GRDH') – more reliable, avoids encoding issues
    2. OData Intersects syntax: area=geography'...' (was: Footprint=geography'...')
    3. Download endpoint: zipper.dataspace.copernicus.eu (was: download.dataspace...)
    4. Credentials: env variables checked first, interactive fallback only if missing
"""

import os
import time
import logging
import argparse
import getpass
from pathlib import Path
from datetime import datetime

from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIGURATION – modify as needed
# ---------------------------------------------------------------------------

# AOI – Lower Vistula (Toruń → Gdańsk)
BOUNDING_BOX = {
    "west":  17.9,
    "south": 53.0,
    "east":  19.1,
    "north": 54.4,
}

# Time range – spring/summer/autumn 2025 (avoiding winter)
DATE_START = "2025-04-01T00:00:00.000Z"
DATE_END   = "2025-10-31T23:59:59.999Z"

# Output directory
OUTPUT_BASE = Path("./sentinel1_data")

# CDSE API endpoints
TOKEN_URL     = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
# FIX #3: correct download endpoint
DOWNLOAD_URL  = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

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
# AUTHENTICATION – OAuth2 Client Credentials
# ---------------------------------------------------------------------------

def get_credentials() -> tuple:
    """
    Loads Client ID and Client Secret from environment variables.
    Falls back to interactive input only if env vars are missing.
    """
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")

    if not client_id:
        print("\nCDSE_CLIENT_ID not set as environment variable.")
        print("Create an OAuth client at: https://shapps.dataspace.copernicus.eu/dashboard/")
        client_id = input("Enter Client ID: ").strip()

    if not client_secret:
        client_secret = getpass.getpass("Enter Client Secret: ").strip()

    return client_id, client_secret


def create_session(client_id: str, client_secret: str) -> OAuth2Session:
    """
    Creates an OAuth2 session with automatic token renewal.
    Token is valid for 600 seconds (10 minutes).
    """
    log.info("Fetching OAuth2 access token...")
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)

    token = oauth.fetch_token(
        token_url=TOKEN_URL,
        client_secret=client_secret,
        include_client_id=True,
    )

    expiry = datetime.fromtimestamp(token.get("expires_at", 0))
    log.info(f"Token obtained, valid until: {expiry.strftime('%H:%M:%S')}")
    return oauth


def refresh_session(client_id: str, client_secret: str, session: OAuth2Session) -> OAuth2Session:
    """Refreshes token if close to expiration."""
    expires_at = session.token.get("expires_at", 0)
    if time.time() >= expires_at - 60:
        log.info("Token expiring – refreshing...")
        session = create_session(client_id, client_secret)
    return session


# ---------------------------------------------------------------------------
# PRODUCT SEARCH
# ---------------------------------------------------------------------------

def build_filter(orbit_direction: str) -> str:
    """
    Builds OData filter string for Sentinel-1 GRD IW.

    FIX #1: Use contains(Name,'IW_GRDH') instead of attribute filters for
    productType/operationalMode/polarisationChannels. This avoids encoding
    issues with special characters (& in VV&VH) and is consistent with
    official CDSE forum examples.

    FIX #2: OData.CSC.Intersects uses area=geography'...' not Footprint=geography'...'

    Note: IW_GRDH over continental Europe is always dual-pol VV+VH,
    so explicit polarisation filtering is not needed.
    """
    w = BOUNDING_BOX["west"]
    s = BOUNDING_BOX["south"]
    e = BOUNDING_BOX["east"]
    n = BOUNDING_BOX["north"]

    # WKT polygon – lon lat order, closed ring
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
    """
    Searches products via OData API with pagination.
    Returns list sorted chronologically.
    """
    log.info(f"Searching for Sentinel-1 GRD IW – {orbit_direction}...")

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
        data = resp.json()

        batch = data.get("value", [])
        products.extend(batch)

        log.debug(f"  Page {skip // page_size + 1}: {len(batch)} products")

        if len(batch) < page_size:
            break
        skip += page_size

    log.info(f"Products found ({orbit_direction}): {len(products)}")
    return products


def select_products(products: list, max_count: int = None) -> list:
    """
    Selects products to download.
    By default, returns all filtered products.
    """
    target = max_count if max_count is not None else len(products)

    if len(products) <= target:
        return products

    step = len(products) / target
    selected = [products[int(i * step)] for i in range(target)]
    log.info(f"Selected {len(selected)} of {len(products)} products (uniform sampling)")
    return selected


def print_product_summary(products: list, orbit: str):
    """Prints tabular summary of products."""
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
# DOWNLOAD
# ---------------------------------------------------------------------------

def download_product(
    product: dict,
    output_dir: Path,
    session: OAuth2Session,
    client_id: str,
    client_secret: str,
) -> Path | None:
    """
    Downloads one product via OData download endpoint.
    Skips if file already exists with correct size.
    """
    product_id = product.get("Id")
    product_name = product.get("Name", product_id)
    output_path = output_dir / f"{product_name}.zip"
    expected_size = product.get("ContentLength", 0)

    if output_path.exists():
        if output_path.stat().st_size == expected_size:
            log.info(f"  Skipping (already exists): {product_name[:52]}")
            return output_path
        else:
            log.warning(f"  Incomplete file, re-downloading: {product_name[:52]}")
            output_path.unlink()

    # FIX #3: zipper endpoint
    url = f"{DOWNLOAD_URL}({product_id})/$value"
    size_mb = expected_size / (1024 ** 2)
    log.info(f"  Downloading: {product_name[:52]} ({size_mb:.0f} MB)")

    session = refresh_session(client_id, client_secret, session)

    try:
        with session.get(url, stream=True, timeout=120, allow_redirects=True) as resp:
            resp.raise_for_status()

            with open(output_path, "wb") as f:
                with tqdm(
                    total=expected_size or None,
                    unit="B",
                    unit_scale=True,
                    desc=f"    {product_name[:38]}",
                    leave=False,
                ) as pbar:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

        log.info(f"  ✓ Done: {product_name[:52]}")
        return output_path

    except Exception as e:
        log.error(f"  ✗ Error: {e}")
        if output_path.exists():
            output_path.unlink()
        return None


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-1 GRD from Copernicus Dataspace (CDSE)"
    )
    parser.add_argument(
        "--orbit", choices=["ASC", "DESC", "BOTH"], default="BOTH",
        help="Orbital direction (default: BOTH)",
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Max scenes per orbit direction (default: all matched products)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Search only, do not download",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("Sentinel-1 GRD – CDSE OData download")
    log.info("=" * 55)
    log.info(
        f"AOI:     W={BOUNDING_BOX['west']} S={BOUNDING_BOX['south']} "
        f"E={BOUNDING_BOX['east']} N={BOUNDING_BOX['north']}"
    )
    log.info(f"Period:  {DATE_START[:10]} → {DATE_END[:10]}")
    log.info(f"Orbit:   {args.orbit}")

    client_id, client_secret = get_credentials()
    session = create_session(client_id, client_secret)

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
        products = search_products(session, orbit_direction)
        selected = select_products(products, max_count=args.max)
        print_product_summary(selected, orbit_direction)

        if args.dry_run:
            log.info("Dry-run mode – skipping download.")
            continue

        log.info(f"Starting download – {orbit_direction} ({len(selected)} scenes)")
        for i, product in enumerate(selected):
            log.info(f"\nScene {i+1}/{len(selected)}")
            result = download_product(
                product, output_dir, session, client_id, client_secret
            )
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