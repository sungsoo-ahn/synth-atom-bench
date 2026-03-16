"""Autoresearch evaluation harness: quick train+eval with multi-GPU variant testing."""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import HardSphereDataset
from data.validate import pair_correlation
from experiments.model_registry import MODEL_DEFAULTS, MODEL_REGISTRY, SIZE_PRESETS
from flow_matching.sampling import sample_batched
from flow_matching.training import flow_matching_loss
from metrics.clash_rate import clash_rate_batched
from metrics.gr_distance import gr_distance

ALL_ARCHS = list(MODEL_REGISTRY.keys())


def random_rotation_matrix(device: torch.device) -> torch.Tensor:
    """Sample a uniform random SO(3) rotation via QR decomposition."""
    z = torch.randn(3, 3, device=device)
    q, r = torch.linalg.qr(z)
    d = torch.diag(r.sign().diag())
    q = q @ d
    if q.det() < 0:
        q[:, 0] = -q[:, 0]
    return q


def quick_train_eval(
    arch: str,
    data_path: str,
    train_time_s: float,
    device: str,
    model_size: str = "medium",
    lr: float = 3e-4,
    batch_size: int = 256,
    n_ode_steps: int = 50,
    eval_samples: int = 2000,
    warmup_frac: float = 0.1,
) -> dict:
    """Quick train loop + evaluation, constrained by wall-clock time.

    Args:
        train_time_s: Training time budget in seconds (excludes data loading
            and evaluation).
    """
    t0 = time.time()

    # Load data
    dataset = HardSphereDataset(data_path)
    box_size = dataset.box_size
    n_atoms = dataset.positions.shape[1]
    radius = dataset.radius

    # Precompute ground-truth g(r)
    gt_r, gt_g_r = pair_correlation(dataset.positions.numpy(), box_size)

    # Center positions for flow matching
    dataset.positions = dataset.positions - box_size / 2

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    # Build model
    kwargs = {**MODEL_DEFAULTS[arch], **SIZE_PRESETS[arch][model_size]}
    kwargs["cutoff"] = box_size * 1.5
    model = MODEL_REGISTRY[arch](**kwargs).to(device)

    # Optimizer — constant LR with linear warmup (no cosine: total steps unknown)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup_time_s = warmup_frac * train_time_s

    # Train until time budget exhausted
    model.train()
    step = 0
    data_iter = iter(loader)
    train_start = time.time()

    while True:
        elapsed = time.time() - train_start
        if elapsed >= train_time_s:
            break

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Linear warmup
        if elapsed < warmup_time_s and warmup_time_s > 0:
            warmup_scale = elapsed / warmup_time_s
            for pg in optimizer.param_groups:
                pg["lr"] = lr * warmup_scale

        x_0 = batch["positions"].to(device)
        R = random_rotation_matrix(torch.device(device))
        x_0 = x_0 @ R.T

        loss = flow_matching_loss(model, x_0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

    actual_train_time = time.time() - train_start

    # Evaluate
    model.eval()
    samples = sample_batched(
        model, n_atoms=n_atoms, n_samples=eval_samples,
        n_steps=n_ode_steps, batch_size=min(batch_size, eval_samples), device=device,
    )
    samples = samples + box_size / 2
    cr = clash_rate_batched(samples, radius)
    grd = gr_distance(samples.numpy(), gt_r, gt_g_r, box_size)

    wall_time = time.time() - t0

    # Cleanup
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "gr_distance": grd,
        "clash_rate": cr,
        "steps": step,
        "train_time_s": round(actual_train_time, 1),
        "wall_time_s": round(wall_time, 1),
        "model_size": model_size,
        "lr": lr,
    }


def parse_variants(variants_str: str | None) -> list[dict]:
    """Parse variant grid specification.

    Format: "size=xs,small,medium,large;lr=1e-4,3e-4"
    Returns list of dicts with all combinations.
    """
    if variants_str is None:
        # Default grid: 4 sizes x 2 LRs = 8 variants
        sizes = ["xs", "small", "medium", "large"]
        lrs = [1e-4, 3e-4]
    else:
        axes = {}
        for part in variants_str.split(";"):
            key, vals = part.strip().split("=")
            axes[key.strip()] = [v.strip() for v in vals.split(",")]
        sizes = axes.get("size", ["medium"])
        lrs = [float(x) for x in axes.get("lr", ["3e-4"])]

    variants = []
    for size in sizes:
        for lr_val in lrs:
            variants.append({"model_size": size, "lr": lr_val})
    return variants


def variant_label(v: dict) -> str:
    """Create a label string from variant dict."""
    return f"lr={v['lr']:.0e}_size={v['model_size']}"


# ── Subprocess-based parallel execution ───────────────────────────────────────


def _build_subprocess_cmd(arch: str, args, variant: dict) -> list[str]:
    """Build the Python command to run a single variant in a subprocess."""
    return [
        sys.executable, "-c",
        f"import json, sys; sys.path.insert(0, '.'); "
        f"from autoresearch.run import quick_train_eval; "
        f"result = quick_train_eval("
        f"arch={arch!r}, data_path={args.data_path!r}, "
        f"train_time_s={args.train_time_s}, device='cuda:0', "
        f"model_size={variant['model_size']!r}, lr={variant['lr']}, "
        f"batch_size={args.batch_size}, n_ode_steps={args.n_ode_steps}, "
        f"eval_samples={args.eval_samples}, warmup_frac={args.warmup_frac}); "
        f"print(json.dumps(result))"
    ]


def _parse_subprocess_output(stdout: str, stderr: str, returncode: int) -> dict:
    """Extract JSON result from subprocess output."""
    if returncode == 0 and stdout.strip():
        for line in reversed(stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
    return {"status": "error", "error": f"exit={returncode}, stderr={stderr[-500:]}"}


def run_variants_for_arch(
    arch: str, args, variants: list[dict], n_gpus: int,
) -> dict[str, dict]:
    """Run all variants for a single architecture across GPUs."""
    results = {}

    if n_gpus <= 1:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        for i, v in enumerate(variants):
            label = variant_label(v)
            print(f"  [{arch}] Variant {i+1}/{len(variants)}: {label}", file=sys.stderr)
            try:
                result = quick_train_eval(
                    arch=arch, data_path=args.data_path, train_time_s=args.train_time_s,
                    device=device, model_size=v["model_size"], lr=v["lr"],
                    batch_size=args.batch_size, n_ode_steps=args.n_ode_steps,
                    eval_samples=args.eval_samples, warmup_frac=args.warmup_frac,
                )
                results[label] = result
            except Exception as e:
                results[label] = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
        return results

    # Parallel execution
    job_queue = list(enumerate(variants))
    active: dict[int, tuple[subprocess.Popen, str, int]] = {}

    def _launch(gpu_id: int, job_idx: int, variant: dict):
        label = variant_label(variant)
        cmd = _build_subprocess_cmd(arch, args, variant)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        active[gpu_id] = (proc, label, job_idx)
        print(f"  [{arch}] [GPU {gpu_id}] Started: {label}", file=sys.stderr)

    for gpu_id in range(min(n_gpus, len(job_queue))):
        job_idx, variant = job_queue.pop(0)
        _launch(gpu_id, job_idx, variant)

    while active:
        time.sleep(2)
        for gpu_id in list(active.keys()):
            proc, label, job_idx = active[gpu_id]
            ret = proc.poll()
            if ret is not None:
                stdout = proc.stdout.read()
                stderr = proc.stderr.read()
                del active[gpu_id]

                parsed = _parse_subprocess_output(stdout, stderr, ret)
                results[label] = parsed
                status = "OK" if "status" not in parsed else parsed["status"]
                print(
                    f"  [{arch}] [GPU {gpu_id}] Done ({len(results)}/{len(variants)}): {label} [{status}]",
                    file=sys.stderr,
                )

                if job_queue:
                    next_idx, next_variant = job_queue.pop(0)
                    _launch(gpu_id, next_idx, next_variant)

    return results


# ── Aggregation helpers ───────────────────────────────────────────────────────


def _aggregate_variant_results(variant_results: dict[str, dict]) -> tuple[float | None, list[float]]:
    """Compute mean g(r) distance from variant results. Returns (mean, values)."""
    gr_values = [
        r["gr_distance"]
        for r in variant_results.values()
        if "gr_distance" in r and r.get("status") != "error"
    ]
    if not gr_values:
        return None, []
    return sum(gr_values) / len(gr_values), gr_values


def load_previous_best(log_path: str) -> float | None:
    """Load the best aggregate metric from the experiment log."""
    if not os.path.isfile(log_path):
        return None
    best = None
    with open(log_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("kept") and "aggregate_metric" in entry:
                    val = entry["aggregate_metric"]
                    if best is None or val < best:
                        best = val
            except json.JSONDecodeError:
                continue
    return best


def load_baseline(baseline_path: str, data_name: str) -> float | None:
    """Load oracle baseline metric."""
    if not os.path.isfile(baseline_path):
        return None
    with open(baseline_path) as f:
        baselines = json.load(f)
    return baselines.get(data_name, {}).get("gr_distance")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Autoresearch quick train+eval harness")
    parser.add_argument(
        "--archs", required=True,
        help='Comma-separated architectures, or "all" (e.g. "transformer" or "equiv_gnn,transformer,pairformer")',
    )
    parser.add_argument("--data", required=True, help="Data config name (e.g. hard_sphere_N50)")
    parser.add_argument("--train_time", type=float, default=10.0, help="Training time per variant in minutes (default: 10)")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_ode_steps", type=int, default=50)
    parser.add_argument("--eval_samples", type=int, default=2000)
    parser.add_argument("--warmup_frac", type=float, default=0.1, help="Fraction of training time for LR warmup (default: 0.1)")
    parser.add_argument("--variants", type=str, default=None, help='Variant grid: "size=xs,small;lr=1e-4,3e-4"')
    args = parser.parse_args()
    args.train_time_s = args.train_time * 60  # convert to seconds

    # Parse architectures
    if args.archs.strip().lower() == "all":
        archs = ALL_ARCHS
    else:
        archs = [a.strip() for a in args.archs.split(",")]
    for a in archs:
        if a not in MODEL_REGISTRY:
            print(json.dumps({"status": "error", "error": f"Unknown arch: {a}. Available: {ALL_ARCHS}"}))
            sys.exit(1)

    # Resolve data path from config name
    import yaml
    config_path = os.path.join("configs", "data", f"{args.data}.yaml")
    if not os.path.isfile(config_path):
        print(json.dumps({"status": "error", "error": f"Data config not found: {config_path}"}))
        sys.exit(1)
    with open(config_path) as f:
        data_cfg = yaml.safe_load(f)
    data_dir = data_cfg["data_dir"]
    args.data_path = os.path.join(data_dir, "train.npz")
    if not os.path.isfile(args.data_path):
        print(json.dumps({"status": "error", "error": f"Training data not found: {args.data_path}"}))
        sys.exit(1)

    variants = parse_variants(args.variants)
    multi_arch = len(archs) > 1
    print(
        f"Running {len(variants)} variants × {len(archs)} arch(s) on {args.n_gpus} GPU(s), "
        f"{args.train_time} min each",
        file=sys.stderr,
    )

    # ── Run each architecture ─────────────────────────────────────────────
    per_arch_results = {}   # arch -> {label: result_dict}
    per_arch_aggregate = {} # arch -> float

    for arch in archs:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Architecture: {arch}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        variant_results = run_variants_for_arch(arch, args, variants, args.n_gpus)
        per_arch_results[arch] = variant_results

        agg, _ = _aggregate_variant_results(variant_results)
        if agg is not None:
            per_arch_aggregate[arch] = agg
            print(f"  [{arch}] Aggregate g(r) distance: {agg:.6f}", file=sys.stderr)
        else:
            print(f"  [{arch}] All variants failed!", file=sys.stderr)

    # ── Cross-arch aggregation ────────────────────────────────────────────
    if not per_arch_aggregate:
        output = {
            "status": "error",
            "error": "All architectures failed",
            "archs": archs,
            "per_arch": per_arch_results,
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)

    # Grand aggregate = mean of per-arch aggregates
    grand_aggregate = sum(per_arch_aggregate.values()) / len(per_arch_aggregate)

    # Load baseline and previous best
    baseline_path = "outputs/autoresearch/baselines.json"
    baseline_metric = load_baseline(baseline_path, args.data)

    log_path = "outputs/autoresearch/experiments.jsonl"
    previous_best = load_previous_best(log_path)

    improved = previous_best is not None and grand_aggregate < previous_best
    if previous_best is None:
        improved = True  # First run

    # Per-arch improvement check
    arch_improved = {}
    if previous_best is not None:
        for arch, agg in per_arch_aggregate.items():
            arch_improved[arch] = agg < previous_best
    all_archs_improved = all(arch_improved.values()) if arch_improved else False

    # Build output
    output = {
        "status": "success",
        "archs": archs,
        "metric_name": "gr_distance",
        "per_arch": {},
        "aggregate_metric": round(grand_aggregate, 6),
        "baseline_metric": baseline_metric,
        "previous_best": previous_best,
        "improved": improved,
    }

    # Per-arch detail
    for arch in archs:
        arch_gr_values = [
            r["gr_distance"]
            for r in per_arch_results[arch].values()
            if "gr_distance" in r and r.get("status") != "error"
        ]
        output["per_arch"][arch] = {
            "variants": per_arch_results[arch],
            "aggregate": round(per_arch_aggregate[arch], 6) if arch in per_arch_aggregate else None,
            "improved": arch_improved.get(arch),
        }

    if multi_arch:
        output["all_archs_improved"] = all_archs_improved
        n_improved = sum(1 for v in arch_improved.values() if v)
        output["arch_improvement_summary"] = f"{n_improved}/{len(arch_improved)} archs improved"
    else:
        # Single-arch: also report per-variant improvement count
        arch = archs[0]
        if previous_best is not None:
            n_variants_improved = sum(
                1 for r in per_arch_results[arch].values()
                if "gr_distance" in r and r.get("status") != "error" and r["gr_distance"] < previous_best
            )
            n_variants_total = sum(
                1 for r in per_arch_results[arch].values()
                if "gr_distance" in r and r.get("status") != "error"
            )
            output["improved_count"] = f"{n_variants_improved}/{n_variants_total} variants improved"
        else:
            output["improved_count"] = "first run"

    print(json.dumps(output, indent=2))

    # Auto-update visualization if experiment log exists
    if os.path.isfile(log_path):
        try:
            subprocess.run(
                [sys.executable, "autoresearch/visualize.py",
                 "--log", log_path, "--output", "outputs/autoresearch/plots"],
                capture_output=True, timeout=60,
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
