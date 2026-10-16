#!/usr/bin/env bash
set -euo pipefail
C_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$C_SCRIPT_DIR/runtime6.sh"
cd "$ROOT"
C_PYTHON="${P4_RFDETR_PYTHON:-$RUNTIME/NSTC-Surgical-Instrument-Pick-Place/.venv-rfdetr/bin/python}"
C_TRIAL="$RUNTIME/datasets/Object datasets/trials/trial_03C_rfdetr_seg_syn_dp_balanced"
C_MODE="${1:-train}"
case "$C_MODE" in
check)
 exec "$C_PYTHON" -m training.rfdetr_seg_pipeline --dataset "$C_TRIAL/dataset" --output "$C_TRIAL/check" --check-only ;;
train) ;;
*) echo 'Usage: train-experiment-c6.sh [check|train]' >&2; exit 2 ;;
esac
C_STAMP="$(date +%Y%m%d_%H%M%S)"
C_OUTPUT="$C_TRIAL/training/experiment_C_mixed_${C_STAMP}"
"$C_PYTHON" -m training.rfdetr_seg_pipeline --dataset "$C_TRIAL/dataset" --output "$C_OUTPUT" \
 --epochs 100 --batch-size 1 --grad-accum-steps 16 --num-workers 8 --seed 42 --resolution 768 --wandb \
 --run-name "trial03C_seg-small_r768_train-SYN50-DP50_val-MIX_test-BOTH_seed42_${C_STAMP}"
for C_DOMAIN in synthetic dp; do
 "$C_PYTHON" -m training.evaluate_segmentation --dataset "$C_TRIAL/dataset/evaluation/$C_DOMAIN" \
 --split test --checkpoint "$C_OUTPUT/checkpoint_best_total.pth" --output "$C_OUTPUT/evaluations/$C_DOMAIN"
done
