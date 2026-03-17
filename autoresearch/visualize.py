"""Visualization for autoresearch experiment logs."""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from autoresearch.experiment_log import load_experiments
from viz.style import ARCH_COLORS, DOUBLE_COL, save_figure, synthbench_style

# Map internal arch names to display names used in ARCH_COLORS
_DISPLAY = {"transformer": "Transformer"}


def load_baseline(baseline_path: str = "outputs/autoresearch/baselines.json") -> dict:
    """Load oracle baselines."""
    if os.path.isfile(baseline_path):
        with open(baseline_path) as f:
            return json.load(f)
    return {}


def _extract_per_arch_best(experiment: dict) -> dict[str, float]:
    """Extract {arch: best_energy_wasserstein} from an experiment entry.

    Reads "best" field (best-of-N); falls back to "aggregate" for old entries.
    """
    result = {}
    per_arch = experiment.get("per_arch", {})
    for arch, arch_data in per_arch.items():
        val = arch_data.get("best") or arch_data.get("aggregate")
        if val is not None:
            result[arch] = val
    return result


def _extract_variants(experiment: dict) -> dict[str, float]:
    """Extract {display_label: energy_wasserstein} from an experiment entry.

    Handles both new per-arch format and legacy flat format.
    """
    result = {}
    per_arch = experiment.get("per_arch", {})
    if per_arch:
        archs = experiment.get("archs", list(per_arch.keys()))
        multi = len(archs) > 1
        for arch, arch_data in per_arch.items():
            variants = arch_data.get("variants", {})
            for label, v in variants.items():
                if isinstance(v, dict) and "energy_wasserstein" in v and v.get("status") != "error":
                    display = f"{arch}/{label}" if multi else label
                    result[display] = v["energy_wasserstein"]
        return result

    # Legacy format: flat "variants" dict
    variants = experiment.get("variants", {})
    for label, v in variants.items():
        if isinstance(v, dict) and "energy_wasserstein" in v:
            result[label] = v["energy_wasserstein"]
    return result


def _short_label(description: str) -> str:
    """Strip the '[arch] ' prefix, keep everything else."""
    import re
    return re.sub(r"^\[[\w_]+\]\s*", "", description).strip()


def plot_progress(experiments: list[dict], baseline: float | None, output_dir: str):
    """Per-architecture progress: Energy Wasserstein over iterations (#5).

    Plots one line per architecture so architecture-phase and flow-matching-phase
    results are not conflated on a single y-axis.
    """
    if not experiments:
        return

    # Collect per-arch data points: {arch: [(iter_idx, gr, kept, desc), ...]}
    arch_series: dict[str, list[tuple[int, float, bool, str]]] = {}
    for i, e in enumerate(experiments):
        kept = e.get("kept", False)
        desc = e.get("description", "")
        per_arch = _extract_per_arch_best(e)
        for arch, agg in per_arch.items():
            arch_series.setdefault(arch, []).append((i + 1, agg, kept, desc))

    if not arch_series:
        return

    with synthbench_style():
        # Collect ALL points sorted by iteration for numbering
        # [(iter_idx, arch, y, kept, desc), ...]
        all_points: list[tuple[int, str, float, bool, str]] = []
        for arch, points in arch_series.items():
            for x, y, kept, desc in points:
                if desc:
                    all_points.append((x, arch, y, kept, desc))
        all_points.sort(key=lambda t: t[0])  # sort by iteration

        # Map (iter_idx, arch) -> sequential number
        point_number: dict[tuple[int, str], int] = {}
        # (number, display, desc, color, kept)
        all_labels: list[tuple[int, str, str, str, bool]] = []
        for seq, (x, arch, y, kept, desc) in enumerate(all_points, 1):
            display = _DISPLAY.get(arch, arch)
            color = ARCH_COLORS.get(display, "gray")
            point_number[(x, arch)] = seq
            all_labels.append((seq, display, _short_label(desc), color, kept))

        n_labels = len(all_labels)
        row_h = 0.22
        table_height = max(0.5, n_labels * row_h + 0.3)
        plot_height = DOUBLE_COL[1] * 1.3

        fig, (ax, ax_table) = plt.subplots(
            2, 1, figsize=(DOUBLE_COL[0] * 1.5, plot_height + table_height),
            gridspec_kw={"height_ratios": [plot_height, table_height]},
        )

        for arch, points in sorted(arch_series.items()):
            display = _DISPLAY.get(arch, arch)
            color = ARCH_COLORS.get(display, "gray")

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            # Line connecting all points
            ax.plot(xs, ys, color=color, alpha=0.3, linewidth=1, zorder=1)

            # Kept = filled circle, rejected = open circle
            kx = [x for x, _, k, _ in points if k]
            ky = [y for _, y, k, _ in points if k]
            rx = [x for x, _, k, _ in points if not k]
            ry = [y for _, y, k, _ in points if not k]

            if kx:
                ax.scatter(kx, ky, c=color, marker="o", s=50, zorder=3,
                           label=f"{display} (kept)")
            if rx:
                ax.scatter(rx, ry, facecolors="none", edgecolors=color,
                           marker="o", s=50, zorder=3, alpha=0.5, linewidths=1.2,
                           label=f"{display} (rejected)")

            # Number all points on the plot
            for x, y, kept, desc in points:
                if desc and (x, arch) in point_number:
                    n = point_number[(x, arch)]
                    ax.annotate(
                        str(n), (x, y),
                        fontsize=6, fontweight="bold" if kept else "normal",
                        color=color, alpha=1.0 if kept else 0.5,
                        textcoords="offset points", xytext=(5, 5),
                        zorder=4,
                    )

        if baseline is not None:
            ax.axhline(baseline, color="#3498db", linestyle="--", alpha=0.7,
                       label=f"Oracle ({baseline:.4f})")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy Wasserstein")
        ax.set_title("Autoresearch Progress (per architecture)")
        ax.legend(frameon=False, fontsize=8, ncol=2)

        # Build legend table in the bottom panel (one label per row)
        ax_table.axis("off")
        if all_labels:
            row_height = 1.0 / max(n_labels, 1)
            for row_idx, (n, arch_name, desc, color, kept) in enumerate(all_labels):
                y_pos = 1.0 - (row_idx + 0.5) * row_height
                alpha = 1.0 if kept else 0.4
                marker = "\u2713" if kept else "\u2717"  # ✓ or ✗
                ax_table.text(
                    0.01, y_pos, f"{n}.",
                    fontsize=7, fontweight="bold" if kept else "normal",
                    color=color, alpha=alpha,
                    transform=ax_table.transAxes, va="center",
                )
                ax_table.text(
                    0.035, y_pos, marker,
                    fontsize=8, color=color, alpha=alpha,
                    transform=ax_table.transAxes, va="center",
                )
                ax_table.text(
                    0.055, y_pos, desc,
                    fontsize=6.5, color="0.2" if kept else "0.55",
                    transform=ax_table.transAxes, va="center",
                )

        fig.tight_layout()
        save_figure(fig, os.path.join(output_dir, "progress"))


def plot_variant_heatmap(experiments: list[dict], output_dir: str):
    """Heatmap of Energy Wasserstein across iterations x variants."""
    if not experiments:
        return

    # Collect all variant labels across all experiments
    all_labels = set()
    for e in experiments:
        all_labels.update(_extract_variants(e).keys())
    labels = sorted(all_labels)
    if not labels:
        return

    # Build matrix
    matrix = np.full((len(experiments), len(labels)), np.nan)
    for i, e in enumerate(experiments):
        variants = _extract_variants(e)
        for j, label in enumerate(labels):
            if label in variants:
                matrix[i, j] = variants[label]

    with synthbench_style():
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), max(4, len(experiments) * 0.3 + 1)))

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
        fig.colorbar(im, ax=ax, label="Energy Wasserstein")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(experiments)))
        ax.set_yticklabels([f"#{i+1}" for i in range(len(experiments))], fontsize=8)
        ax.set_xlabel("Variant")
        ax.set_ylabel("Iteration")
        ax.set_title("Per-Variant Energy Wasserstein")

        save_figure(fig, os.path.join(output_dir, "variant_heatmap"))


def plot_improvement_timeline(experiments: list[dict], baseline: float | None, output_dir: str):
    """Per-architecture running best over iterations (#5).

    Tracks the running best Energy Wasserstein for each architecture independently,
    avoiding cross-phase confusion.
    """
    if not experiments:
        return

    # Build per-arch running best: {arch: [(iter, running_best), ...]}
    arch_best: dict[str, float] = {}
    arch_timeline: dict[str, list[tuple[int, float]]] = {}
    for i, e in enumerate(experiments):
        if not e.get("kept", False):
            continue
        per_arch = _extract_per_arch_best(e)
        for arch, agg in per_arch.items():
            if arch not in arch_best or agg < arch_best[arch]:
                arch_best[arch] = agg
            arch_timeline.setdefault(arch, []).append((i + 1, arch_best[arch]))

    if not arch_timeline:
        return

    # Accept rate (cumulative, across all experiments)
    iters = list(range(1, len(experiments) + 1))
    cumulative_accept = []
    total_kept = 0
    for i, e in enumerate(experiments):
        total_kept += int(e.get("kept", False))
        cumulative_accept.append(total_kept / (i + 1))

    with synthbench_style():
        fig, ax1 = plt.subplots(figsize=DOUBLE_COL)

        for arch, points in sorted(arch_timeline.items()):
            display = _DISPLAY.get(arch, arch)
            color = ARCH_COLORS.get(display, "gray")
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax1.step(xs, ys, where="post", color=color, linewidth=2, label=display)

        if baseline is not None:
            ax1.axhline(baseline, color="#3498db", linestyle="--", alpha=0.7, label=f"Oracle ({baseline:.4f})")

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Running best Energy Wasserstein")
        ax1.set_title("Cumulative Improvement (per architecture)")

        # Secondary axis: accept rate
        ax2 = ax1.twinx()
        ax2.plot(iters, cumulative_accept, color="#e67e22", linewidth=1, alpha=0.5, linestyle=":")
        ax2.set_ylabel("Accept rate", color="#e67e22")
        ax2.tick_params(axis="y", labelcolor="#e67e22")
        ax2.set_ylim(0, 1)

        ax1.legend(frameon=False, loc="upper right", fontsize=8)
        save_figure(fig, os.path.join(output_dir, "improvement_timeline"))


def main():
    parser = argparse.ArgumentParser(description="Visualize autoresearch experiment log")
    parser.add_argument("--log", default="outputs/autoresearch/experiments.jsonl")
    parser.add_argument("--output", default="outputs/autoresearch/plots")
    parser.add_argument("--data", default=None, help="Data config name to select correct baseline (e.g. multibody_23_N50_T1.0)")
    args = parser.parse_args()

    if not os.path.isfile(args.log):
        print(f"Experiment log not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    experiments = load_experiments(args.log)
    if not experiments:
        print("No experiments found in log.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(experiments)} experiments", file=sys.stderr)

    # Find baseline for the specified data config (or from experiment log)
    baselines = load_baseline()
    baseline_metric = None
    data_name = args.data
    if data_name is None and experiments:
        data_name = experiments[0].get("data")
    if data_name and data_name in baselines:
        baseline_metric = baselines[data_name].get("energy_wasserstein")
    elif baselines:
        # Fallback: first baseline (legacy behavior)
        for key, val in baselines.items():
            baseline_metric = val.get("energy_wasserstein")
            break

    os.makedirs(args.output, exist_ok=True)

    plot_progress(experiments, baseline_metric, args.output)
    print(f"  Saved progress plot", file=sys.stderr)

    plot_variant_heatmap(experiments, args.output)
    print(f"  Saved variant heatmap", file=sys.stderr)

    plot_improvement_timeline(experiments, baseline_metric, args.output)
    print(f"  Saved improvement timeline", file=sys.stderr)


if __name__ == "__main__":
    main()
