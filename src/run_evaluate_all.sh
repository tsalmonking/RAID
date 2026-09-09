#!/bin/bash
# Evaluate all detectors on all attack outputs (white-box, LOO, full ensemble).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/raid:${PYTHONPATH}"

MODELS=(ojha2023 corvi2023 cavia2024 chen2024_convnext chen2024_clip koutlis2024 wang2020)
OUTPUT_DIR=output/evaluate_detector
DEVICE=cuda:1
BATCH_SIZE=32
NUM_WORKERS=4
# FORCE=1 recomputes results that already exist; default only fills in gaps.
FORCE=${FORCE:-0}

evaluate() {
    local eval_model="$1" pkl_path="$2"
    if [ ! -f "$pkl_path" ]; then
        echo "SKIP (not found): ${pkl_path}"
        return
    fi
    # Mirrors evaluate_detector.py's output path: $OUTPUT_DIR + the dataset
    # path minus its first component and filename + the model name.
    local rel="${pkl_path#*/}"
    rel="${rel%/adv_dataset.pkl}"
    local result_path="${OUTPUT_DIR}/${rel}/${eval_model}/results.json"
    if [ "$FORCE" != "1" ] && [ -f "$result_path" ]; then
        echo "SKIP (exists): ${result_path}"
        return
    fi
    echo "==== evaluating ${eval_model} on ${pkl_path} ===="
    python raid/evaluate_detector.py \
        --model "$eval_model" \
        --device "$DEVICE" \
        --path_to_dataset "$pkl_path" \
        --dataset_type dataset \
        --output_dir "$OUTPUT_DIR" \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --subset -1
}

model_list_repr() {
    # Rebuilds the exact "['a', 'b', ...]" string Python's list repr produces,
    # which is what attack_generate.py used to name the output directory.
    local arr=("$@")
    local s="["
    for i in "${!arr[@]}"; do
        [ "$i" -gt 0 ] && s+=", "
        s+="'${arr[$i]}'"
    done
    s+="]"
    echo "$s"
}

# ---- 1. White-box (single-detector attack), eps in {16, 32} ----
for eps in 16 32; do
    for m in "${MODELS[@]}"; do
        pkl="output/ADV/ELSA_TEST_1000/adv_raw_['${m}']_${eps}_255_10_0.05/adv_dataset.pkl"
        for eval_model in "${MODELS[@]}"; do
            evaluate "$eval_model" "$pkl"
        done
    done
done

# ---- 2. Leave-one-out (6-model ensemble attack), eps in {16, 32} ----
for eps in 16 32; do
    for held_out in "${MODELS[@]}"; do
        ENSEMBLE=()
        for m in "${MODELS[@]}"; do
            [ "$m" != "$held_out" ] && ENSEMBLE+=("$m")
        done
        list_str=$(model_list_repr "${ENSEMBLE[@]}")
        pkl="output/ADV/ELSA_TEST_1000/adv_raw_${list_str}_${eps}_255_10_0.05/adv_dataset.pkl"
        for eval_model in "${MODELS[@]}"; do
            evaluate "$eval_model" "$pkl"
        done
    done
done

# ---- 3. Full ensemble (7-model attack, full dataset), eps in {8, 16, 32} ----
list_str=$(model_list_repr "${MODELS[@]}")
for eps in 8 16 32; do
    pkl="output/ADV/ELSA_TEST/adv_raw_${list_str}_${eps}_255_10_0.05/adv_dataset.pkl"
    for eval_model in "${MODELS[@]}"; do
        evaluate "$eval_model" "$pkl"
    done
done

echo "All evaluations complete. Results under ${OUTPUT_DIR}/"
