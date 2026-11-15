#!/usr/bin/env bash
# Exactly two full checkpoints, one Pick and one Place, five classes each.
set -euo pipefail
cd /home/user/NSTC
python_bin=/home/user/NSTC/.venv-dp/bin/python
run_root=/home/user/NSTC/datasets/DP_training/runs
queue_log="$run_root/target_relative_joint_two_20261103_queue.log"

for skill in pick place; do
  name="target_relative_dp_joint_${skill}_v1_20261103"
  output="$run_root/$name"
  if [[ -e "$output" ]]; then
    echo "ERROR: output exists: $output" | tee -a "$queue_log"
    exit 1
  fi
  echo "START $name $(date --iso-8601=seconds)" | tee -a "$queue_log"
  "$python_bin" experiments/dp_target_relative/train_joint.py \
    --skill "$skill" --output "$output" --epochs 100 \
    --batch-size 128 --patience 15 \
    > "$run_root/${name}.log" 2>&1
  echo "DONE $name $(date --iso-8601=seconds)" | tee -a "$queue_log"
done
echo "ALL_TRAINING_FINISHED $(date --iso-8601=seconds)" | tee -a "$queue_log"
