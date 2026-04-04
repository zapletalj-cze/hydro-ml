#!/bin/bash
# =============================================================================
# batch_preprocess.sh
# Batch execution of SNAP GPT preprocessing for Sentinel-1 GRD
# Output projection: EPSG:2180 (PL-1992) - hardcoded in XML graph
#
# Usage: ./batch_preprocess.sh <input_dir> <output_dir> [threads]
#
# Examples:
#   ./batch_preprocess.sh /data/ascending /data/processed/ascending
#   ./batch_preprocess.sh /data/descending /data/processed/descending
#   ./batch_preprocess.sh /data/ascending /data/processed/ascending 8
#
# Prerequisites:
#   - SNAP installed, gpt available in PATH
#   - S1_preprocessing_levee.xml in the same directory as this script
# =============================================================================

set -e

INPUT_DIR="${1:?Provide input directory}"
OUTPUT_DIR="${2:?Provide output directory}"
THREADS="${3:-4}"        # number of CPU threads for GPT (default: 4)
GRAPH="$(dirname "$0")/S1_preprocessing_levee.xml"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "=============================================="
echo " SNAP GPT Batch Preprocessing"
echo " Input:      $INPUT_DIR"
echo " Output:     $OUTPUT_DIR"
echo " Projection: EPSG:2180 (PL-1992)"
echo " Threads:    $THREADS"
echo "=============================================="

# Counters
TOTAL=0
SUCCESS=0
FAILED=0

# Process .zip and .SAFE files
for INPUT_FILE in "$INPUT_DIR"/*.zip "$INPUT_DIR"/*.SAFE; do
    [ -e "$INPUT_FILE" ] || continue

    BASENAME=$(basename "$INPUT_FILE" .zip)
    BASENAME=$(basename "$BASENAME" .SAFE)
    OUTPUT_FILE="${OUTPUT_DIR}/${BASENAME}_sigma0"
    LOG_FILE="${LOG_DIR}/${BASENAME}.log"

    TOTAL=$((TOTAL + 1))

    # Skip if output already exists
    if [ -f "${OUTPUT_FILE}.tif" ]; then
        echo "[SKIP] Already exists: ${BASENAME}"
        SUCCESS=$((SUCCESS + 1))
        continue
    fi

    echo ""
    echo "[${TOTAL}] Processing: ${BASENAME}"
    echo "      Started: $(date '+%H:%M:%S')"

    # Run GPT
    if gpt "$GRAPH" \
        -Pinput="$INPUT_FILE" \
        -Poutput="$OUTPUT_FILE" \
        -q "$THREADS" \
        > "$LOG_FILE" 2>&1; then

        echo "      Done:    $(date '+%H:%M:%S')"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "      FAILED – see log: $LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=============================================="
echo " DONE: ${SUCCESS}/${TOTAL} scenes processed successfully"
[ $FAILED -gt 0 ] && echo " FAILED: ${FAILED} scenes – check logs in ${LOG_DIR}"
echo "=============================================="