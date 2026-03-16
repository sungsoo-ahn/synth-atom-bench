"""Visualization for autoresearch experiment logs."""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from viz.style import DOUBLE_COL, SINGLE_COL, save_figure, synthbench_style


def load_experiments(log_path: str) -> list[dict]:
    """Load experiments from JSONL log."""
    experiments = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    experiments.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return experiments


def load_baseline(baseline_path: str = "outputs/autoresearch/baselines.json") -> dict:
    """Load oracle baselines."""
    if os.path.isfile(baseline_path):
        with open(baseline_path) as f:
            return json.load(f)
    return {}


def plot_progress(experiments: list[dict], baseline: float | None, output_dir: str):
    """Progress curve: aggregate g(r) distance over iterations."""
    if not experiments:
        return

    iters = list(range(1, len(experiments) + 1))
    metrics = [e["aggregate_metric"] for e in experiments]
    kept = [e.get("kept", False) for e in experiments]

    with synthbench_style():
        fig, ax = plt.subplots(figsize=DOUBLE_COL)

        # Separate kept vs rejected
        kept_x = [i for i, k in zip(iters, kept) if k]
        kept_y = [m for m, k in zip(metrics, kept) if k]
        rej_x = [i for i, k in zip(iters, kept) if not k]
        rej_y = [m for m, k in zip(metrics, kept) if not k]

        if kept_x:
            ax.scatter(kept_x, kept_y, c="#2ecc71", marker="o", s=60, zorder=3, label="Kept")
        if rej_x:
            ax.scatter(rej_x, rej_y, c="#e74c3c", marker="x", s=60, zorder=3, label="Rejected")

        # Connect all points with a light line
        ax.plot(iters, metrics, color="gray", alpha=0.3, linewidth=1, zorder=1)

        # Oracle baseline
        if baseline is not None:
            ax.axhline(baseline, color="#3498db", linestyle="--", alpha=0.7, label=f"Oracle ({baseline:.4f})")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Aggregate g(r) distance")
        ax.set_title("Autoresearch Progress")
        ax.legend(frameon=False)

        save_figure(fig, os.path.join(output_dir, "progress"))


def plot_variant_heatmap(experiments: list[dict], output_dir: str):
    """Heatmap of g(r) distance across iterations x variants."""
    if not experiments:
        return

    # Collect all variant labels
    all_labels = set()
    for e in experiments:
        all_labels.update(e.get("variants", {}).keys())
    labels = sorted(all_labels)
    if not labels:
        return

    # Build matrix
    matrix = np.full((len(experiments), len(labels)), np.nan)
    for i, e in enumerate(experiments):
        for j, label in enumerate(labels):
            v = e.get("variants", {}).get(label, {})
            if "gr_distance" in v:
                matrix[i, j] = v["gr_distance"]

    with synthbench_style():
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), max(4, len(experiments) * 0.3 + 1)))

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
        fig.colorbar(im, ax=ax, label="g(r) distance")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(experiments)))
        ax.set_yticklabels([f"#{i+1}" for i in range(len(experiments))], fontsize=8)
        ax.set_xlabel("Variant")
        ax.set_ylabel("Iteration")
        ax.set_title("Per-Variant g(r) Distance")

        save_figure(fig, os.path.join(output_dir, "variant_heatmap"))


def plot_improvement_timeline(experiments: list[dict], baseline: float | None, output_dir: str):
    """Running best g(r) distance over iterations with accept/reject annotations."""
    if not experiments:
        return

    iters = list(range(1, len(experiments) + 1))
    metrics = [e["aggregate_metric"] for e in experiments]
    kept = [e.get("kept", False) for e in experiments]

    # Compute running best (only from kept experiments)
    running_best = []
    best_so_far = float("inf")
    for m, k in zip(metrics, kept):
        if k and m < best_so_far:
            best_so_far = m
        running_best.append(best_so_far if best_so_far < float("inf") else m)

    # Accept rate (cumulative)
    cumulative_accept = []
    total_kept = 0
    for i, k in enumerate(kept):
        total_kept += int(k)
        cumulative_accept.append(total_kept / (i + 1))

    with synthbench_style():
        fig, ax1 = plt.subplots(figsize=DOUBLE_COL)

        # Running best line
        ax1.plot(iters, running_best, color="#2c3e50", linewidth=2, label="Running best")
        ax1.fill_between(iters, running_best, alpha=0.1, color="#2c3e50")

        # Oracle baseline
        if baseline is not None:
            ax1.axhline(baseline, color="#3498db", linestyle="--", alpha=0.7, label=f"Oracle ({baseline:.4f})")

        # Annotate kept experiments with descriptions
        for i, (m, k, e) in enumerate(zip(metrics, kept, experiments)):
            if k and (i == 0 or m < running_best[i - 1] if i > 0 else True):
                desc = e.get("description", "")[:30]
                if desc:
                    ax1.annotate(
                        desc, (iters[i], running_best[i]),
                        textcoords="offset points", xytext=(5, 10),
                        fontsize=7, alpha=0.7, rotation=15,
                        arrowprops=dict(arrowstyle="-", alpha=0.3),
                    )

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("g(r) distance")
        ax1.set_title("Cumulative Improvement")

        # Secondary axis: accept rate
        ax2 = ax1.twinx()
        ax2.plot(iters, cumulative_accept, color="#e67e22", linewidth=1, alpha=0.5, linestyle=":")
        ax2.set_ylabel("Accept rate", color="#e67e22")
        ax2.tick_params(axis="y", labelcolor="#e67e22")
        ax2.set_ylim(0, 1)

        ax1.legend(frameon=False, loc="upper right")
        save_figure(fig, os.path.join(output_dir, "improvement_timeline"))


def main():
    parser = argparse.ArgumentParser(description="Visualize autoresearch experiment log")
    parser.add_argument("--log", default="outputs/autoresearch/experiments.jsonl")
    parser.add_argument("--output", default="outputs/autoresearch/plots")
    args = parser.parse_args()

    if not os.path.isfile(args.log):
        print(f"Experiment log not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    experiments = load_experiments(args.log)
    if not experiments:
        print("No experiments found in log.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(experiments)} experiments", file=sys.stderr)

    # Detect data config from first experiment to find baseline
    baselines = load_baseline()
    # Try to match baseline (look for any key in baselines)
    baseline_metric = None
    for key, val in baselines.items():
        baseline_metric = val.get("gr_distance")
        break  # Use first available baseline

    os.makedirs(args.output, exist_ok=True)

    plot_progress(experiments, baseline_metric, args.output)
    print(f"  Saved progress plot", file=sys.stderr)

    plot_variant_heatmap(experiments, args.output)
    print(f"  Saved variant heatmap", file=sys.stderr)

    plot_improvement_timeline(experiments, baseline_metric, args.output)
    print(f"  Saved improvement timeline", file=sys.stderr)


if __name__ == "__main__":
    main()
