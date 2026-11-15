#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="$(cd -- "$REPO/../.." && pwd)"
PY="$ROOT/.venv-dp/bin/python"
export PYTHONPATH="$REPO:$REPO/compat${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
DATA="$ROOT/datasets/DP_training/all5_rgb448_v1"
INVENTORY="$ROOT/datasets/Object datasets/trials/trial_03C_rfdetr_seg_syn_dp_balanced/dataset/source_episodes.json"
RUNS="$ROOT/datasets/DP_training/runs"
MODE="${1:-help}";shift || true
case "$MODE" in
 prepare) exec "$PY" -m training.prepare_dp_training --inventory "$INVENTORY" --output "$DATA" --image-size 448 "$@" ;;
 login) exec "$PY" -m wandb login "$@" ;;
 check)
  exec "$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads((p/"manifest.json").read_text()); assert d["export_complete"] and not d.get("smoke_only"); print(json.dumps(d["split_counts"],indent=2))' "$DATA" ;;
 pick|place|both)
  if [[ ! -f "$DATA/manifest.json" ]]; then echo 'Prepare the dataset first: ./train-dp6.sh prepare' >&2;exit 1;fi
  STAMP="$(date +%Y%m%d_%H%M%S)"
  if [[ "$MODE" == both ]]; then SKILLS=(pick place);else SKILLS=("$MODE");fi
  for SKILL in "${SKILLS[@]}";do
   OUT="$RUNS/dp_${SKILL}_all5_rgb448_b${DP_BATCH_SIZE:-24}_$STAMP"
   mkdir -p "$RUNS"
   "$PY" -m training.train_dp_epochs --dataset "$DATA" --output "$OUT" --skill "$SKILL" --epochs 100 --batch-size "${DP_BATCH_SIZE:-24}" --workers "${DP_WORKERS:-4}" --patience 15 --min-delta 0.0001 --wandb online "$@" 2>&1 | tee "$OUT.log"
  done ;;
 resume)
  OUT="${1:?Usage: ./train-dp6.sh resume /absolute/run/folder}";shift
  SKILL="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["skill"])' "$OUT/training_config.json")"
  BATCH="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["batch_size"])' "$OUT/training_config.json")"
  DATA="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["dataset"])' "$OUT/training_config.json")"
  exec "$PY" -m training.train_dp_epochs --dataset "$DATA" --output "$OUT" --skill "$SKILL" --batch-size "$BATCH" --resume "$OUT/last.ckpt" --wandb online "$@" ;;
 *) echo 'Usage: ./train-dp6.sh prepare | check | login | pick | place | both | resume RUN_FOLDER'
    echo 'Default: RGB 448, batch 24, 100 max epochs, early stopping patience 15, W&B online.' ;;
esac
