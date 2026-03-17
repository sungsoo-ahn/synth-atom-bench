"""Generate figures for README.md."""

import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on the path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt

from experiments.model_registry import ARCH_DISPLAY_NAMES

from viz.scaling import plot_scaling_curves
from viz.structure import plot_structure
from viz.style import save_figure, synthbench_style

OUT_DIR = _project_root / "docs" / "assets"


def hero_structures():
    """1x3 grid: multibody configurations at N=10, 20, 50."""
    Ns = [10, 20, 50]

    with synthbench_style():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                                 subplot_kw={"projection": "3d"},
                                 layout="none")

        for col, N in enumerate(Ns):
            data_path = _project_root / f"outputs/data/multibody_23_N{N}_T1.0/val.npz"
            if not data_path.exists():
                print(f"  hero_structures: SKIPPED (missing {data_path})")
                return
            data = np.load(data_path)
            positions = data["positions"][0]
            radius = float(data["radius"])
            box_size = float(data["box_size"]) if "box_size" in data else None

            ax = axes[col]
            bonds = [(i, i + 1) for i in range(N - 1)]
            plot_structure(
                positions, radius, box_size=box_size, ax=ax, title=f"N = {N}",
                bonds=bonds, draw_box=box_size is not None,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")

        fig.subplots_adjust(left=0.02, wspace=0.05)
        save_figure(fig, OUT_DIR / "hero_structures")
    print("  hero_structures")


def _load_scaling_results(results_path, metric_key="best_energy_wasserstein"):
    """Load scaling results and group by architecture.

    Returns dict mapping display name -> {"flops": array, "metric": array},
    filtered to entries with total_flops > 0.
    """
    with open(results_path) as f:
        raw = json.load(f)

    best = raw["best_per_budget"]
    arch_name_map = ARCH_DISPLAY_NAMES
    arch_data = {}
    for entry in best.values():
        if entry["total_flops"] <= 0:
            continue
        arch = entry["arch"]
        display_name = arch_name_map.get(arch, arch)
        if display_name not in arch_data:
            arch_data[display_name] = {"flops": [], "metric": []}
        arch_data[display_name]["flops"].append(entry["total_flops"])
        arch_data[display_name]["metric"].append(entry[metric_key])

    results = {}
    for name, data in arch_data.items():
        order = np.argsort(data["flops"])
        results[name] = {
            "flops": np.array(data["flops"])[order],
            "metric": np.array(data["metric"])[order],
        }
    return results


def scaling_curves():
    """Scaling curves from multibody experiment results (energy Wasserstein)."""
    results_path = _project_root / "outputs" / "scaling" / "results.json"
    if not results_path.exists():
        print("  scaling_curves: SKIPPED (no results.json)")
        return

    results = _load_scaling_results(results_path, metric_key="best_energy_wasserstein")

    with synthbench_style():
        fig = plot_scaling_curves(results, fit_curves=True, ylabel="Energy Wasserstein")
        save_figure(fig, OUT_DIR / "scaling_curves")
    print("  scaling_curves")


def main():
    print("Generating README figures...")
    hero_structures()
    scaling_curves()
    print(f"Done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
