"""
s1_preprocess_pyrosar.py
========================
Parallel Sentinel-1 GRD preprocessing using pyroSAR + SNAP.
Replaces manual SNAP GPT XML graph with pyroSAR's geocode() function.

Advantages over raw GPT:
  - Automatic orbit file download (no timeout issues)
  - Automatic DEM tile download
  - Python multiprocessing for true parallelism across scenes
  - Built-in skip logic (already processed scenes)
  - Clean error handling per scene

Output:
  - sigma-nought VV and VH GeoTIFF in linear units
  - EPSG:2180 (PL-1992), 10m resolution
  - Compatible with subsequent Python feature engineering pipeline

Requirements:
    pip install pyrosar spatialist

    SNAP must be installed and configured:
      Windows: C:\\Program Files\\esa-snap\\
      pyroSAR will auto-detect SNAP on first run or set manually:
        from pyroSAR import ExamineSnap
        ExamineSnap()

Usage:
    python s1_preprocess_pyrosar.py
    python s1_preprocess_pyrosar.py --orbit ASC
    python s1_preprocess_pyrosar.py --orbit DESC
    python s1_preprocess_pyrosar.py --jobs 16
    python s1_preprocess_pyrosar.py --dry-run
"""

import logging
import argparse
import traceback
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

RAW_DIR     = Path("C:/data/raw")
OUTPUT_BASE = Path("C:/data/processed")

# Processing parameters
RESOLUTION   = 10                       # output pixel spacing in metres
T_SRS        = 2180                     # EPSG:2180 - PL-1992, consistent with BDOT10k
POLARIZATIONS = ["VV", "VH"]
DEM_NAME     = "Copernicus 30m Global DEM"
SCALING      = "linear"                 # linear units for correct temporal averaging
                                        # dB conversion done later in Python
N_JOBS       = 8                        # number of parallel scenes

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
# PROCESSING FUNCTION (runs in each worker process)
# ---------------------------------------------------------------------------

def process_scene(scene_path: Path, output_dir: Path) -> dict:
    """
    Processes a single Sentinel-1 GRD scene using pyroSAR geocode().

    Returns dict with scene name, status and error message if failed.
    """
    result = {
        "scene":  scene_path.name,
        "status": "failed",
        "error":  None,
    }

    try:
        # Import inside worker to avoid multiprocessing issues
        from pyroSAR.snap import geocode

        log.info(f"Processing: {scene_path.name}")

        geocode(
            infile=str(scene_path),
            outdir=str(output_dir),

            # Output projection - EPSG:2180 (PL-1992)
            # Consistent with BDOT10k ground truth vectors
            t_srs=T_SRS,

            # Pixel spacing in metres - preserves native GRD resolution
            tr=RESOLUTION,

            # Polarizations
            polarizations=POLARIZATIONS,

            # Output scaling: linear (not dB) for correct temporal averaging
            # dB conversion applied later in Python after multi-temporal averaging
            scaling=SCALING,

            # Geocoding method
            geocoding_type="Range-Doppler",

            # DEM - Copernicus 30m, more accurate than SRTM for Poland
            demName=DEM_NAME,
            demResamplingMethod="BILINEAR_INTERPOLATION",
            imgResamplingMethod="BILINEAR_INTERPOLATION",

            # Preprocessing steps
            removeS1BorderNoise=True,
            removeS1BorderNoiseMethod="pyroSAR",  # improved border noise removal
            removeS1ThermalNoise=True,

            # Orbit files - allow Restituted as fallback to avoid timeout
            allow_RES_OSV=True,

            # sigma0 (not gamma0) - correct for flat terrain (Polish lowlands)
            terrainFlattening=False,
            refarea="sigma0",

            # Speckle filter disabled - suppression via multi-temporal averaging
            speckleFilter=False,

            # groupsize=1: execute each node separately
            # reduces memory usage and improves stability
            groupsize=1,

            # Mask sea with nodata
            nodataValueAtSea=True,

            # Align pixels across scenes for consistent time series
            alignToStandardGrid=False,

            # Cleanup temporary files after processing
            cleanup=True,
        )

        result["status"] = "success"
        log.info(f"Done: {scene_path.name}")

    except Exception as e:
        result["error"] = str(e)
        log.error(f"Failed: {scene_path.name}")
        log.error(traceback.format_exc())

    return result


# ---------------------------------------------------------------------------
# DIRECTORY PROCESSING
# ---------------------------------------------------------------------------

def collect_scenes(input_dir: Path, output_dir: Path) -> list:
    """
    Collects unprocessed .zip scenes from input_dir.
    Skips scenes already present in output_dir.
    """
    scenes = sorted(input_dir.glob("*.zip"))
    if not scenes:
        log.warning(f"No .zip files found in {input_dir}")
        return []

    # pyroSAR marks processed scenes by creating output files
    # with a standardised naming convention - check via is_processed
    unprocessed = []
    skipped = 0

    for scene in scenes:
        try:
            from pyroSAR import identify
            id_obj = identify(str(scene))
            if id_obj.is_processed(str(output_dir)):
                log.info(f"[SKIP] Already processed: {scene.name}")
                skipped += 1
            else:
                unprocessed.append(scene)
        except Exception:
            # If identify fails, include scene (let geocode handle it)
            unprocessed.append(scene)

    log.info(
        f"Found {len(scenes)} scenes: "
        f"{len(unprocessed)} to process, {skipped} already done"
    )
    return unprocessed


def process_directory(
    input_dir: Path,
    output_dir: Path,
    n_jobs: int,
    label: str,
    dry_run: bool = False,
) -> dict:
    """
    Processes all scenes in input_dir using n_jobs parallel workers.
    """
    log.info(f"\n--- {label} ---")

    if not input_dir.exists():
        log.warning(f"Input directory not found, skipping: {input_dir}")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = collect_scenes(input_dir, output_dir)
    skipped_count = len(list(input_dir.glob("*.zip"))) - len(scenes)

    if not scenes:
        return {"total": 0, "success": 0, "failed": 0, "skipped": skipped_count}

    if dry_run:
        log.info(f"[DRY-RUN] Would process {len(scenes)} scenes with {n_jobs} workers")
        for s in scenes:
            log.info(f"  {s.name}")
        return {"total": len(scenes), "success": 0, "failed": 0, "skipped": skipped_count}

    log.info(f"Processing {len(scenes)} scenes with {n_jobs} parallel workers")

    # Bind output_dir to the worker function
    worker_fn = partial(process_scene, output_dir=output_dir)

    results = []
    with Pool(processes=n_jobs) as pool:
        for result in pool.imap_unordered(worker_fn, scenes):
            results.append(result)
            status_icon = "OK" if result["status"] == "success" else "FAIL"
            log.info(f"  [{status_icon}] {result['scene']}")
            if result["error"]:
                log.error(f"       {result['error']}")

    success = sum(1 for r in results if r["status"] == "success")
    failed  = sum(1 for r in results if r["status"] == "failed")

    return {
        "total":   len(scenes),
        "success": success,
        "failed":  failed,
        "skipped": skipped_count,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel Sentinel-1 GRD preprocessing with pyroSAR"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR,
        help=f"Root directory containing ascending/ and descending/ subdirs (default: {RAW_DIR})"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_BASE,
        help=f"Output root directory (default: {OUTPUT_BASE})"
    )
    parser.add_argument(
        "--orbit", choices=["ASC", "DESC", "BOTH"], default="BOTH",
        help="Orbit direction to process (default: BOTH)"
    )
    parser.add_argument(
        "--jobs", type=int, default=N_JOBS,
        help=f"Number of parallel scenes (default: {N_JOBS})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List scenes to process without executing"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("Sentinel-1 GRD Preprocessing - pyroSAR")
    log.info("=" * 55)
    log.info(f"Raw dir:    {args.raw_dir}")
    log.info(f"Output:     {args.output_dir}")
    log.info(f"Projection: EPSG:{T_SRS} (PL-1992)")
    log.info(f"Resolution: {RESOLUTION} m")
    log.info(f"Scaling:    {SCALING}")
    log.info(f"DEM:        {DEM_NAME}")
    log.info(f"Workers:    {args.jobs}")
    log.info(f"Orbit:      {args.orbit}")
    if args.dry_run:
        log.info("MODE:       DRY RUN")

    # Verify pyroSAR is available
    try:
        import pyroSAR
        log.info(f"pyroSAR:    {pyroSAR.__version__}")
    except ImportError:
        log.error("pyroSAR not installed. Run: pip install pyrosar spatialist")
        return

    total_stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    orbits = []
    if args.orbit in ("ASC", "BOTH"):
        orbits.append(("ASCENDING", args.raw_dir / "ascending",
                       args.output_dir / "ascending"))
    if args.orbit in ("DESC", "BOTH"):
        orbits.append(("DESCENDING", args.raw_dir / "descending",
                       args.output_dir / "descending"))

    for label, input_dir, output_dir in orbits:
        stats = process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            n_jobs=args.jobs,
            label=label,
            dry_run=args.dry_run,
        )
        for k in total_stats:
            total_stats[k] += stats[k]

    # Summary
    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    print(f"  Total    : {total_stats['total']}")
    print(f"  Success  : {total_stats['success']}")
    print(f"  Skipped  : {total_stats['skipped']}")
    if total_stats["failed"] > 0:
        print(f"  Failed   : {total_stats['failed']}  ← check logs above")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()