#!/usr/bin/env bash
set -euo pipefail
SEG_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SEG_SCRIPT_DIR/runtime6.sh"
SEG_TRIAL="$RUNTIME/datasets/Object datasets/trials/trial_02_rfdetr_seg_hemisphere768"
SEG_MODE="${1:-both}"
case "$SEG_MODE" in a|b|both) ;; *) echo 'Usage: train_experiments6.sh a|b|both' >&2; exit 2;; esac
SEG_STAMP="$(date +%Y%m%d_%H%M%S)"
SEG_COMMON=(--epochs 100 --batch-size 1 --grad-accum-steps 16 --num-workers 8 --seed 42)
if [[ "$SEG_MODE" == a || "$SEG_MODE" == both ]]; then
  "$SEG_SCRIPT_DIR/trial02.sh" train "${SEG_COMMON[@]}" \
    --dataset "$SEG_TRIAL/dataset" \
    --run-name "trial02A_seg-small_r768_train-SYN_val-SYN_test-SYN_seed42_$SEG_STAMP" \
    --output "$SEG_TRIAL/training/experiment_A_synthetic_validation_$SEG_STAMP"
fi
if [[ "$SEG_MODE" == b || "$SEG_MODE" == both ]]; then
  "$SEG_SCRIPT_DIR/trial02.sh" prepare-dp
  "$SEG_SCRIPT_DIR/trial02.sh" train "${SEG_COMMON[@]}" \
    --dataset "$SEG_TRIAL/dataset_dp_validation" \
    --run-name "trial02B_seg-small_r768_train-SYN_val-DP_test-DP_seed42_$SEG_STAMP" \
    --output "$SEG_TRIAL/training/experiment_B_dp_validation_$SEG_STAMP"
fi
