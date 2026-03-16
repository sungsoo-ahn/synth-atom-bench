"""Oracle g(r) distance baseline: measures irreducible noise from finite samples."""

import argparse
import json
import os
import sys

import numpy as np

from data.validate import pair_correlation
from metrics.gr_distance import gr_distance


def compute_oracle_gr_distance(data_path: str, num_bins: int = 200) -> dict:
    """Split training data in half, compute g(r) distance between halves.

    This is the theoretical best any generator can achieve — the noise floor
    from finite sample statistics.

    Returns:
        Dict with keys: gr_distance, n_samples, box_size, radius
    """
    data = np.load(data_path)
    positions = data["positions"]
    box_size = float(data["box_size"])
    radius = float(data["radius"])
    n_samples = positions.shape[0]

    # Split in half
    mid = n_samples // 2
    half_a = positions[:mid]
    half_b = positions[mid : mid + mid]  # equal sizes

    # Compute g(r) on each half
    r_a, g_r_a = pair_correlation(half_a, box_size, num_bins=num_bins)
    grd = gr_distance(half_b, r_a, g_r_a, box_size, num_bins=num_bins)

    return {
        "gr_distance": round(grd, 6),
        "n_samples": n_samples,
        "n_per_half": mid,
        "box_size": box_size,
        "radius": radius,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute oracle g(r) distance baseline")
    parser.add_argument("--data", required=True, help="Data config name (e.g. hard_sphere_N50)")
    parser.add_argument("--num_bins", type=int, default=200)
    args = parser.parse_args()

    import yaml

    config_path = os.path.join("configs", "data", f"{args.data}.yaml")
    if not os.path.isfile(config_path):
        print(f"Data config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        data_cfg = yaml.safe_load(f)

    data_path = os.path.join(data_cfg["data_dir"], "train.npz")
    if not os.path.isfile(data_path):
        print(f"Training data not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Computing oracle baseline for {args.data}...", file=sys.stderr)
    result = compute_oracle_gr_distance(data_path, num_bins=args.num_bins)
    print(f"  Oracle g(r) distance: {result['gr_distance']}", file=sys.stderr)
    print(f"  Using {result['n_per_half']} samples per half", file=sys.stderr)

    # Cache result
    cache_dir = "outputs/autoresearch"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "baselines.json")

    baselines = {}
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            baselines = json.load(f)

    baselines[args.data] = result
    with open(cache_path, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"  Cached to {cache_path}", file=sys.stderr)

    # Also print JSON to stdout for programmatic use
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
