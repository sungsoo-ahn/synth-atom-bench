"""Oracle energy Wasserstein baseline: measures irreducible noise from finite samples."""

import argparse
import json
import os
import sys

import numpy as np
import torch

from metrics.energy import energy_wasserstein, total_energy_batch


def compute_oracle_energy_wasserstein(
    data_path: str, eval_samples: int | None = None,
) -> dict:
    """Split training data in half, compute energy Wasserstein between halves.

    This measures the noise floor of the energy Wasserstein metric given finite samples.

    Args:
        eval_samples: If set, subsample the "generated" half to this count
            to match the evaluation sample size. This makes the oracle a fair
            lower bound for the harness evaluation (#6).

    Returns:
        Dict with keys: energy_wasserstein, n_samples, n_reference, n_generated,
        box_size, radius, preset, temperature
    """
    data = np.load(data_path)
    positions = data["positions"]
    box_size = float(data["box_size"])
    r0 = float(data["r0"])
    radius = r0 / 2
    preset = str(data["preset"])
    temperature = float(data["temperature"])
    k2 = float(data["k2"])
    r0 = float(data["r0"])
    k3 = float(data["k3"])
    theta0 = float(data["theta0"])
    k4 = float(data["k4"])
    phi0 = float(data["phi0"])
    n_samples = positions.shape[0]

    # Split in half
    mid = n_samples // 2
    half_a = positions[:mid]
    half_b = positions[mid : mid + mid]

    # Subsample half_b to match eval sample count if specified
    n_generated = len(half_b)
    if eval_samples is not None and eval_samples < len(half_b):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(half_b), eval_samples, replace=False)
        half_b = half_b[idx]
        n_generated = eval_samples

    # Compute total energy for both halves
    half_a_t = torch.from_numpy(half_a).float()
    half_b_t = torch.from_numpy(half_b).float()

    energies_a = total_energy_batch(
        half_a_t, preset=preset, k2=k2, r0=r0, k3=k3, theta0=theta0, k4=k4, phi0=phi0,
    )
    energies_b = total_energy_batch(
        half_b_t, preset=preset, k2=k2, r0=r0, k3=k3, theta0=theta0, k4=k4, phi0=phi0,
    )

    ew = energy_wasserstein(energies_b, energies_a)

    return {
        "energy_wasserstein": round(ew, 6),
        "n_samples": n_samples,
        "n_reference": mid,
        "n_generated": n_generated,
        "box_size": box_size,
        "radius": radius,
        "preset": preset,
        "temperature": temperature,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute oracle energy Wasserstein baseline")
    parser.add_argument("--data", required=True, help="Data config name (e.g. multibody_23_N50_T1.0)")
    parser.add_argument("--eval_samples", type=int, default=2000,
                        help="Subsample size to match evaluation (default: 2000)")
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
    result = compute_oracle_energy_wasserstein(
        data_path, eval_samples=args.eval_samples,
    )
    print(f"  Oracle energy Wasserstein: {result['energy_wasserstein']}", file=sys.stderr)
    print(f"  Reference samples: {result['n_reference']}, generated samples: {result['n_generated']}", file=sys.stderr)

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
