#!/bin/bash
set -e
# Overnight scaling law experiments
# Early stopping (patience=5) + single LR (1e-3)
# 3 budgets × 3 sizes × 6 configs = 54 jobs

PRESETS=("multibody_2" "multibody_23" "multibody_234")
NS=(10 20)
T=1.0
N_GPUS=8
BUDGETS="1e15,4e15,1.6e16"
SIZES="xs,small,medium"

for preset in "${PRESETS[@]}"; do
  for N in "${NS[@]}"; do
    echo "========================================"
    echo "  ${preset} N=${N} T=${T}"
    echo "========================================"
    uv run python experiments/scaling_auto.py \
      --task multibody \
      --preset "$preset" \
      --n_atoms "$N" \
      --temperature "$T" \
      --archs transformer \
      --budgets "$BUDGETS" \
      --sizes "$SIZES" \
      --n_gpus "$N_GPUS"
  done
done
echo "=== ALL SCALING EXPERIMENTS DONE ==="
