#!/bin/bash
# Generate Gaussian and uniform noise baselines for each epsilon.
# No GPU needed — runs on CPU in a few minutes.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/raid:${PYTHONPATH}"

DATASET=/storageC/heddoubi/raid/data/data/RAID/ELSA_TEST
OUTPUT_DIR=${SCRIPT_DIR}/output
SUBSET=200

for eps in "8/255" "16/255" "32/255"; do
    for noise_type in gaussian uniform; do
        echo "── Generating ${noise_type} noise at eps=${eps} ──"
        python3 raid/generate_noise_baseline.py \
            --path_to_dataset "$DATASET" \
            --output_dir "$OUTPUT_DIR" \
            --noise_type "$noise_type" \
            --epsilon "$eps" \
            --subset $SUBSET
    done
done

echo "Done. Noise baselines saved under ${OUTPUT_DIR}/ADV/"
