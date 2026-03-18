"""Standalone evaluation: load checkpoint, generate samples, compute metrics, save plots."""

import argparse
import os
import sys

import numpy as np
import torch

from experiments.checkpointing import load_checkpoint
from experiments.model_registry import MODEL_REGISTRY
from experiments.tasks import get_task_from_data
from flow_matching.sampling import sample_batched


def build_model_from_config(config: dict, box_size: float, task=None) -> torch.nn.Module:
    """Reconstruct model from saved config dict."""
    arch = config["model"]["arch"]
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(MODEL_REGISTRY.keys())}")
    kwargs = dict(config["model"]["model_kwargs"])
    if "cutoff" in kwargs:
        kwargs["cutoff"] = box_size * 1.5
    # Inject task-specific kwargs (e.g. atom_ordering) if not already in saved config
    if task is not None:
        for k, v in task.model_kwargs().items():
            kwargs.setdefault(k, v)
    return MODEL_REGISTRY[arch](**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained velocity network")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (best.pt or latest.pt)")
    parser.add_argument("--data", required=True, help="Path to data directory (containing train.npz)")
    parser.add_argument("--output", default=None, help="Output directory (default: outputs/eval/{arch}/)")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--n_steps", type=int, default=100, help="Number of ODE steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for sampling")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    state = load_checkpoint(args.checkpoint, device=str(device))
    config = state.config
    arch = config["model"]["arch"]
    print(f"Architecture: {arch} | Step: {state.step} | Best E-W1: {state.best_energy_wasserstein:.4f}")

    # Auto-detect task and load dataset
    train_path = os.path.join(args.data, "train.npz")
    task = get_task_from_data(train_path)
    dataset = task.load_dataset(train_path)
    print(f"Data: {task.describe_data(dataset)}")
    box_size = dataset.box_size
    radius = dataset.r0 / 2
    n_atoms = dataset.positions.shape[1]

    # Build and load model
    model = build_model_from_config(config, box_size, task=task).to(device)
    model.load_state_dict(state.model_state_dict)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Generate samples
    print(f"Generating {args.n_samples} samples ({args.n_steps} ODE steps)...")
    model.eval()
    samples = sample_batched(
        model,
        n_atoms=n_atoms,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=str(device),
    )
    # Shift back to [0, box_size]
    samples = samples + box_size / 2

    # Compute energy metrics
    metrics = task.compute_metrics(samples, dataset)

    print(f"\nResults:")
    print(f"  Samples generated: {args.n_samples}")
    for k, v in metrics.items():
        print(f"  {k:20s} {v:.4f}")

    # Output directory
    output_dir = args.output or os.path.join("outputs", "eval", arch)
    os.makedirs(output_dir, exist_ok=True)

    # Save generated positions
    save_kwargs = dict(
        positions=samples.numpy(),
        radius=radius,
        box_size=box_size,
        step=state.step,
    )
    save_kwargs.update(metrics)
    out_path = os.path.join(output_dir, "generated.npz")
    np.savez(out_path, **save_kwargs)
    print(f"  Saved: {out_path}")

    # Plot structures grid
    from viz import save_figure, synthbench_style
    from viz.structure import plot_structures_grid

    with synthbench_style():
        n_show = min(8, args.n_samples)
        fig = plot_structures_grid(
            [samples[i].numpy() for i in range(n_show)],
            radius=radius,
            box_size=box_size,
        )
        save_figure(fig, os.path.join(output_dir, "structures"))
        print(f"  Saved: {output_dir}/structures.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
