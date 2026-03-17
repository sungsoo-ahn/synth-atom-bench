"""Training loop for flow matching velocity networks."""

import math
import os
import sys

# Monkey-patch argparse for Python 3.14 compatibility with Hydra's LazyCompletionHelp
import argparse

_orig_add_argument = argparse.ArgumentParser.add_argument


def _patched_add_argument(self, *args, **kwargs):
    help_val = kwargs.get("help")
    if help_val is not None and not isinstance(help_val, str):
        kwargs["help"] = repr(help_val)
    return _orig_add_argument(self, *args, **kwargs)


argparse.ArgumentParser.add_argument = _patched_add_argument

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from data.validate import pair_correlation
from experiments.checkpointing import CheckpointManager
from experiments.logger import ComputeTracker, Logger, LoggerConfig
from experiments.model_registry import MODEL_REGISTRY, SIZE_PRESETS
from experiments.tasks import get_task
from flow_matching.sampling import sample_batched
from flow_matching.training import flow_matching_loss
from metrics.clash_rate import clash_rate_batched
from metrics.gr_distance import gr_distance


def random_rotation_matrix(device: torch.device) -> torch.Tensor:
    """Sample a uniform random SO(3) rotation via QR decomposition."""
    z = torch.randn(3, 3, device=device)
    q, r = torch.linalg.qr(z)
    # Fix sign to ensure proper rotation (det=+1)
    d = torch.diag(r.sign().diag())
    q = q @ d
    if q.det() < 0:
        q[:, 0] = -q[:, 0]
    return q


def count_flops(model: nn.Module, n_atoms: int, batch_size: int, device: torch.device) -> int:
    """Estimate FLOPs for one forward+backward pass."""
    n_params = sum(p.numel() for p in model.parameters())
    # Use 6 * params * batch_size as estimate (2x forward + 4x backward)
    # Try torch FlopCounterMode if available
    try:
        from torch.utils.flop_counter import FlopCounterMode

        x = torch.randn(batch_size, n_atoms, 3, device=device)
        t = torch.rand(batch_size, device=device)
        with FlopCounterMode(display=False) as counter:
            out = model(x, t)
            loss = out.sum()
        forward_flops = counter.get_total_flops()
        # backward is ~2x forward
        return int(forward_flops * 3)
    except (ImportError, Exception):
        return 6 * n_params * batch_size


def build_model(cfg: DictConfig, box_size: float, task) -> nn.Module:
    """Instantiate velocity network from config."""
    arch = cfg.model.arch
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(MODEL_REGISTRY.keys())}")
    kwargs = dict(cfg.model.model_kwargs)
    # Override cutoff to match data
    if "cutoff" in kwargs:
        kwargs["cutoff"] = box_size * 1.5
    # Task-specific model kwargs (e.g. atom_ordering for chains)
    kwargs.update(task.model_kwargs())
    return MODEL_REGISTRY[arch](**kwargs)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup."""
    max_steps = cfg.train.max_steps
    warmup_steps = int(cfg.train.warmup_fraction * max_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(
    model: nn.Module,
    dataset,
    cfg: DictConfig,
    device: torch.device,
    task,
    gt_r: "np.ndarray | None" = None,
    gt_g_r: "np.ndarray | None" = None,
) -> dict:
    """Generate samples and compute metrics.

    Returns dict with keys: clash_rate, gr_distance, samples,
    plus any task-specific metrics.
    """
    import numpy as np

    model.eval()
    samples = sample_batched(
        model,
        n_atoms=dataset.positions.shape[1],
        n_samples=cfg.eval.n_samples,
        n_steps=cfg.eval.n_ode_steps,
        batch_size=cfg.eval.sample_batch_size,
        device=str(device),
    )
    # Shift back to [0, box_size]
    samples = samples + dataset.box_size / 2
    cr = clash_rate_batched(samples, dataset.radius)
    grd = float("inf")
    if gt_r is not None and gt_g_r is not None:
        grd = gr_distance(samples.numpy(), gt_r, gt_g_r, dataset.box_size)

    result = {"clash_rate": cr, "gr_distance": grd, "samples": samples}
    result.update(task.compute_metrics(samples, dataset))

    model.train()
    return result


def _format_eval_msg(step: int | str, ev: dict, best_grd: float) -> str:
    """Format evaluation results into a log message."""
    parts = [f"Step {step:>6s}" if isinstance(step, str) else f"Step {step:6d}",
             f"Eval clash rate: {ev['clash_rate']:.4f}",
             f"g(r) dist: {ev['gr_distance']:.4f}"]
    for k, v in ev.items():
        if k not in ("clash_rate", "gr_distance", "samples"):
            parts.append(f"{k}: {v:.4f}")
    parts.append(f"Best g(r): {best_grd:.4f}")
    return "  " + " | ".join(parts)


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    # Resolve size preset into model_kwargs
    size = cfg.model.get("size")
    if size and cfg.model.arch in SIZE_PRESETS:
        preset = SIZE_PRESETS[cfg.model.arch][size]
        with open_dict(cfg):
            for k, v in preset.items():
                cfg.model.model_kwargs[k] = v

    # Seed
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)

    # GPU assignment: distribute multirun jobs across GPUs
    if torch.cuda.is_available():
        try:
            from hydra.core.hydra_config import HydraConfig
            job_num = HydraConfig.get().job.num
        except Exception:
            job_num = 0
        gpu_id = job_num % torch.cuda.device_count()
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Task
    task = get_task(cfg)

    # Load dataset
    data_dir = cfg.data.data_dir
    train_path = os.path.join(data_dir, "train.npz")
    dataset = task.load_dataset(train_path)
    box_size = dataset.box_size
    n_atoms = dataset.positions.shape[1]
    print(f"Dataset: {len(dataset)} samples, {task.describe_data(dataset)}")

    # Precompute ground-truth g(r) for evaluation metric
    print("Precomputing ground-truth g(r)...")
    gt_r, gt_g_r = pair_correlation(dataset.positions.numpy(), box_size)

    # Center positions for flow matching (noise is N(0,I))
    dataset.positions = dataset.positions - box_size / 2

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # Persist task-specific model kwargs into config so checkpoints can reconstruct the model
    task_kwargs = task.model_kwargs()
    if task_kwargs:
        with open_dict(cfg):
            for k, v in task_kwargs.items():
                cfg.model.model_kwargs[k] = v

    # Build model
    model = build_model(cfg, box_size, task).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: {cfg.model.arch} | Parameters: {n_params:,}")

    # FLOPs estimation
    flops_per_step = count_flops(model, n_atoms, cfg.train.batch_size, device)
    print(f"FLOPs per step: {flops_per_step:.2e}")

    # Budget mode: compute max_steps from budget / flops_per_step
    budget = cfg.train.get("budget")
    if budget is not None and float(budget) > 0:
        budget = float(budget)
        computed_steps = int(budget / flops_per_step)
        if computed_steps < 2000:
            print(f"Budget {budget:.0e}: only {computed_steps} steps (< 2000 min). Skipping.")
            return
        if computed_steps > 1_000_000:
            print(f"Budget {budget:.0e}: needs {computed_steps} steps (> 1M max). Skipping.")
            return
        with open_dict(cfg):
            cfg.train.max_steps = computed_steps
        print(f"Budget {budget:.0e}: training for {computed_steps} steps")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg)

    # Checkpoint dir
    checkpoint_dir = cfg.checkpoint.get("dir") or os.path.join("outputs", "checkpoints", cfg.model.arch)
    ckpt_mgr = CheckpointManager(checkpoint_dir)

    # Resume from checkpoint if available
    start_step = 0
    state = ckpt_mgr.load_latest(device=str(device))
    if state is not None:
        model.load_state_dict(state.model_state_dict)
        optimizer.load_state_dict(state.optimizer_state_dict)
        start_step = state.step
        # Fast-forward scheduler (suppress expected warning about ordering)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_step):
                scheduler.step()
        print(f"Resumed from step {start_step}")

    # Logger
    logger_config = LoggerConfig(
        enabled=cfg.logging.enabled,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        log_dir=cfg.logging.get("log_dir", "outputs/logs"),
    )
    run_name = task.run_name(cfg, n_atoms)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    logger = Logger(logger_config, run_name=run_name, model_config=config_dict)
    logger.log_model_config(cfg.model.arch, n_params, flops_per_step)

    # Compute tracker
    tracker = ComputeTracker()

    # Training loop
    model.train()
    step = start_step
    data_iter = iter(loader)
    use_rotation = cfg.augmentation.random_rotation
    print(f"\nTraining for {cfg.train.max_steps} steps (starting from {start_step})...")

    while step < cfg.train.max_steps:
        # Get next batch, cycling through dataset
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x_0 = batch["positions"].to(device)

        # Random SO(3) augmentation
        if use_rotation:
            R = random_rotation_matrix(device)
            x_0 = x_0 @ R.T

        tracker.start()

        # Forward + backward
        loss = flow_matching_loss(model, x_0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        tracker.stop()
        step += 1

        # Log training metrics
        if step % logger_config.log_every_n_steps == 0:
            total_flops = flops_per_step * step
            lr = scheduler.get_last_lr()[0]
            logger.log_train(
                {"train/loss": loss.item(), "train/lr": lr, "train/total_flops": total_flops},
                step=step,
            )
            logger.log_compute(tracker, step)
            print(f"  Step {step:6d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | FLOPs: {total_flops:.2e}")

        # Evaluate + checkpoint
        if step % cfg.eval.every_n_steps == 0:
            ev = evaluate(model, dataset, cfg, device, task, gt_r, gt_g_r)
            total_flops = flops_per_step * step
            # Collect metrics (everything except samples tensor)
            ckpt_kwargs = {k: v for k, v in ev.items() if k != "samples"}
            extra_metrics = {k: v for k, v in ckpt_kwargs.items()
                            if k not in ("clash_rate", "gr_distance")}
            logger.log_eval(ev["samples"], dataset.radius, dataset.box_size, step,
                            extra_metrics=extra_metrics or None,
                            clash_rate=ev["clash_rate"])

            # Log all metrics
            log_metrics = {"train/total_flops": total_flops}
            for k, v in ckpt_kwargs.items():
                log_metrics[f"eval/{k}"] = v
            logger.log_train(log_metrics, step=step)

            # Save checkpoint (clash_rate and gr_distance are positional, rest are kwargs)
            cr = ckpt_kwargs.pop("clash_rate")
            grd = ckpt_kwargs.pop("gr_distance")
            ckpt_mgr.save(model, optimizer, epoch=0, step=step,
                          clash_rate=cr, config=config_dict, gr_distance=grd, **ckpt_kwargs)
            print(_format_eval_msg(step, ev, ckpt_mgr.best_gr_distance))

        # Periodic checkpoint (without eval) — carry forward best metrics
        elif step % cfg.checkpoint.every_n_steps == 0:
            ckpt_mgr.save(
                model, optimizer, epoch=0, step=step,
                clash_rate=ckpt_mgr.best_clash_rate, config=config_dict,
                gr_distance=ckpt_mgr.best_gr_distance,
                bond_violation_rate=ckpt_mgr.best_bond_violation_rate,
                nonbonded_clash_rate=ckpt_mgr.best_nonbonded_clash_rate,
            )
            print(f"  Step {step:6d} | Checkpoint saved")

    # Final evaluation
    print("\nFinal evaluation...")
    ev = evaluate(model, dataset, cfg, device, task, gt_r, gt_g_r)
    ckpt_kwargs = {k: v for k, v in ev.items() if k != "samples"}
    extra_metrics = {k: v for k, v in ckpt_kwargs.items()
                    if k not in ("clash_rate", "gr_distance")}
    logger.log_eval(ev["samples"], dataset.radius, dataset.box_size, step,
                    extra_metrics=extra_metrics or None,
                    clash_rate=ev["clash_rate"])
    cr = ckpt_kwargs.pop("clash_rate")
    grd = ckpt_kwargs.pop("gr_distance")
    ckpt_mgr.save(model, optimizer, epoch=0, step=step,
                  clash_rate=cr, config=config_dict, gr_distance=grd, **ckpt_kwargs)
    print(_format_eval_msg("Final", ev, ckpt_mgr.best_gr_distance))

    logger.finish()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
