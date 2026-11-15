#!/usr/bin/env bash
# Sequential training: one CUDA job at a time. Existing Scissor Pick is frozen.
set -euo pipefail
cd /home/user/NSTC
python_bin=/home/user/NSTC/.venv-dp/bin/python
run_root=/home/user/NSTC/datasets/DP_training/runs
queue_log="$run_root/target_relative_family_20261103_queue.log"

run_one() {
  local instrument="$1" skill="$2"
  local name="target_relative_dp_${instrument}_${skill}_v1_20261103"
  local output="$run_root/$name"
  if [[ -e "$output" ]]; then
    echo "ERROR: output already exists: $output" | tee -a "$queue_log"
    return 1
  fi
  echo "START $name $(date --iso-8601=seconds)" | tee -a "$queue_log"
  "$python_bin" experiments/dp_target_relative/train_family.py \
    --instrument "$instrument" --skill "$skill" \
    --output "$output" --epochs 100 --batch-size 128 --patience 15 \
    > "$run_root/${name}.log" 2>&1
  echo "DONE $name $(date --iso-8601=seconds)" | tee -a "$queue_log"
}

run_one scissor place
run_one love_retractor pick
run_one love_retractor place
run_one scalpel pick
run_one scalpel place
run_one kelly pick
run_one kelly place
run_one scalpel_type2 pick
run_one scalpel_type2 place
echo "ALL_TRAINING_FINISHED $(date --iso-8601=seconds)" | tee -a "$queue_log"
