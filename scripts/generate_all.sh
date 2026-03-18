#!/bin/bash
set -e
# Generate all datasets: train (50K), val (5K), test (10K) for each config.
# Also generates train_10k (10K) for data scaling experiments.

PRESETS=("multibody_2" "multibody_23" "multibody_234")
NS=(10 20)
TS=(0.5 1.0 2.0)

# Split name -> (num_samples, seed)
declare -A SPLITS=(
  ["train"]="50000 42"
  ["val"]="5000 123"
  ["test"]="10000 456"
  ["train_10k"]="10000 42"
)

for preset in "${PRESETS[@]}"; do
  for N in "${NS[@]}"; do
    for T in "${TS[@]}"; do
      dir="outputs/data/${preset}_N${N}_T${T}"
      for split in "${!SPLITS[@]}"; do
        read -r size seed <<< "${SPLITS[$split]}"
        outfile="${dir}/${split}.npz"
        if [ -f "$outfile" ]; then
          echo "SKIP $outfile (exists)"
          continue
        fi
        echo "=== ${preset} N=${N} T=${T} ${split} (${size} samples) ==="
        uv run python data/generate_multibody.py \
          --N "$N" --T "$T" --preset "$preset" \
          --num_samples "$size" --seed "$seed" \
          --output "$outfile"
      done
    done
  done
done
echo "=== ALL DONE ==="
