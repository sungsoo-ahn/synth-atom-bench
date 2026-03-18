#!/bin/bash
set -e

PRESETS=("multibody_2" "multibody_23" "multibody_234")
NS=(10 20)
TS=(0.5 1.0 2.0)

for preset in "${PRESETS[@]}"; do
  for N in "${NS[@]}"; do
    for T in "${TS[@]}"; do
      dir="outputs/data/${preset}_N${N}_T${T}"

      # Val: 5K samples, seed=123
      outfile="${dir}/val.npz"
      if [ -f "$outfile" ]; then
        echo "SKIP $outfile (exists)"
      else
        echo "=== ${preset} N=${N} T=${T} val (5K) ==="
        uv run python data/generate_multibody.py \
          --N "$N" --T "$T" --preset "$preset" \
          --num_samples 5000 --seed 123 \
          --output "$outfile"
      fi

      # Test: 10K samples, seed=456
      outfile="${dir}/test.npz"
      if [ -f "$outfile" ]; then
        echo "SKIP $outfile (exists)"
      else
        echo "=== ${preset} N=${N} T=${T} test (10K) ==="
        uv run python data/generate_multibody.py \
          --N "$N" --T "$T" --preset "$preset" \
          --num_samples 10000 --seed 456 \
          --output "$outfile"
      fi
    done
  done
done
echo "=== ALL DONE ==="
