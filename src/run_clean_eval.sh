#!/bin/bash
# Clean-baseline evaluation: every detector on unperturbed test set.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/raid:${PYTHONPATH}"

MODELS=(ojha2023 corvi2023 cavia2024 chen2024_convnext chen2024_clip koutlis2024 wang2020)
DATASET=${TEST_PATH:-data/ELSA_TEST}
OUTPUT_DIR=output/evaluate_detector
DEVICE=cuda:1
BATCH_SIZE=32
NUM_WORKERS=4
SUBSET=${SUBSET:-1000}
FORCE=${FORCE:-0}

# Mirrors evaluate_detector.py's path derivation for --dataset_type subfolders:
# $OUTPUT_DIR + the dataset path minus its leading "/" + "_$SUBSET" (unless -1).
rel="${DATASET#/}"
[ "$SUBSET" != "-1" ] && rel="${rel}_${SUBSET}"

for model in "${MODELS[@]}"; do
    result_path="${OUTPUT_DIR}/${rel}/${model}/results.json"
    if [ "$FORCE" != "1" ] && [ -f "$result_path" ]; then
        echo "SKIP (exists): ${result_path}"
        continue
    fi
    echo "==== clean evaluation: ${model} ===="
    python raid/evaluate_detector.py \
        --model "$model" \
        --device "$DEVICE" \
        --path_to_dataset "$DATASET" \
        --dataset_type subfolders \
        --output_dir "$OUTPUT_DIR" \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --subset $SUBSET
done

echo "Clean evaluation complete. Results under ${OUTPUT_DIR}/${rel}/"
