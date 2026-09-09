#!/bin/bash
# PGD ablation: LOO attacks across step sizes, iteration counts, and epsilons.
# Skips runs where adv_dataset.pkl already exists (resumable).
#
# For 2 GPUs: copy this script, change DEVICE and HELD_OUT_DETECTORS.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/raid:${PYTHONPATH}"

# ──────────────── Configurable parameters ────────────────
DEVICE=cuda:0

STEP_COUNTS=(10 20 30)
STEP_SIZES=(0.01 0.03 0.05)
EPSILONS=("8/255" "16/255" "32/255")

ALL_DETECTORS=(ojha2023 corvi2023 cavia2024 chen2024_convnext chen2024_clip koutlis2024 wang2020)
HELD_OUT_DETECTORS=("${ALL_DETECTORS[@]}")

DATASET=${RAID_DATASET:-data/ELSA_TEST}
OUTPUT_DIR=${SCRIPT_DIR}/output
SUBSET=200
BATCH_SIZE=32
# ─────────────────────────────────────────────────────────

# Build the Python-style list string for the ensemble, matching attack_generate.py's output dir format
build_models_str() {
    local arr=("$@")
    local s="['${arr[0]}'"
    for det in "${arr[@]:1}"; do
        s+=", '${det}'"
    done
    s+="]"
    echo "$s"
}

# Count total runs
total_runs=0
for eps in "${EPSILONS[@]}"; do
    for steps in "${STEP_COUNTS[@]}"; do
        for alpha in "${STEP_SIZES[@]}"; do
            for held_out in "${HELD_OUT_DETECTORS[@]}"; do
                total_runs=$((total_runs + 1))
            done
        done
    done
done

echo "============================================"
echo "PGD Ablation Sweep"
echo "Device:     $DEVICE"
echo "Held-out:   ${HELD_OUT_DETECTORS[*]}"
echo "Epsilons:   ${EPSILONS[*]}"
echo "Steps:      ${STEP_COUNTS[*]}"
echo "Step sizes: ${STEP_SIZES[*]}"
echo "Subset:     $SUBSET"
echo "Total runs: $total_runs"
echo "============================================"

completed=0
skipped=0
START_TIME=$(date +%s)

for eps in "${EPSILONS[@]}"; do
    eps_str="${eps//\//_}"
    for steps in "${STEP_COUNTS[@]}"; do
        for alpha in "${STEP_SIZES[@]}"; do
            for held_out in "${HELD_OUT_DETECTORS[@]}"; do
                # Build ensemble: all detectors except held-out
                ENSEMBLE=()
                for det in "${ALL_DETECTORS[@]}"; do
                    if [ "$det" != "$held_out" ]; then
                        ENSEMBLE+=("$det")
                    fi
                done

                completed=$((completed + 1))

                # Check if output already exists
                models_str=$(build_models_str "${ENSEMBLE[@]}")
                expected_dir="${OUTPUT_DIR}/ADV/ELSA_TEST_${SUBSET}/adv_raw_${models_str}_${eps_str}_${steps}_${alpha}"
                if [ -f "${expected_dir}/adv_dataset.pkl" ]; then
                    skipped=$((skipped + 1))
                    echo "[$completed/$total_runs] SKIP (exists): eps=$eps steps=$steps alpha=$alpha held_out=$held_out"
                    continue
                fi

                DEVICES=()
                for _ in "${ENSEMBLE[@]}"; do
                    DEVICES+=("$DEVICE")
                done

                echo ""
                echo "── [$completed/$total_runs] eps=$eps steps=$steps alpha=$alpha held_out=$held_out ──"

                echo y | python3 raid/attack_generate.py \
                    --models "${ENSEMBLE[@]}" \
                    --device "${DEVICES[@]}" \
                    --path_to_dataset "$DATASET" \
                    --output_dir "$OUTPUT_DIR" \
                    --batch_size $BATCH_SIZE \
                    --epsilon "$eps" \
                    --num_steps "$steps" \
                    --step_size "$alpha" \
                    --subset $SUBSET \
                    --generate
            done
        done
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "============================================"
echo "Ablation complete: $completed runs ($skipped skipped) in $(printf '%dh %dm %ds' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))"
echo "============================================"
