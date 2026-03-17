"""Autoresearch evaluation harness: quick train+eval with multi-GPU variant testing."""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

# Only import the registry (frozen module) at top level.
# Editable modules (flow_matching/*, models/*, data/*, metrics/*) are imported
# inside quick_train_eval() which runs in subprocesses, so agent edits are
# always picked up fresh and a broken edit won't crash the main process
# before the preflight check runs.
from experiments.model_registry import MODEL_DEFAULTS, MODEL_REGISTRY, SIZE_PRESETS

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
    task_name: str,
    train_time_s: float,
    device: str,
    model_size: str = "medium",
    lr: float = 3e-4,
    batch_size: int = 256,
    n_ode_steps: int = 50,
    eval_samples: int = 2000,
    warmup_frac: float = 0.1,
    weight_decay: float = 0.01,
    seed: int = 42,
) -> dict:
    """Quick train loop + evaluation, constrained by wall-clock time.

    Args:
        task_name: "multibody".
        train_time_s: Training time budget in seconds (excludes data loading
            and evaluation).
        seed: Random seed for reproducibility.
    """
    # Lazy imports — these modules are editable by the agent, so we import them
    # here (inside the subprocess) rather than at module level.
    from torch.utils.data import DataLoader

    from experiments.tasks import TASK_REGISTRY
    from flow_matching.sampling import sample_batched
    from flow_matching.training import flow_matching_loss
    from metrics.energy import energy_metrics_batched

    # Seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    t0 = time.time()

    # Load data via task system
    task = TASK_REGISTRY[task_name]()
    dataset = task.load_dataset(data_path)
    box_size = dataset.box_size
    n_atoms = dataset.positions.shape[1]
    radius = dataset.radius


    # Center positions for flow matching
    dataset.positions = dataset.positions - box_size / 2

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    # Build model — merge task-specific kwargs (e.g. atom_ordering for chains)
    kwargs = {**MODEL_DEFAULTS[arch], **SIZE_PRESETS[arch][model_size]}
    kwargs["cutoff"] = box_size * 1.5
    kwargs.update(task.model_kwargs())
    model = MODEL_REGISTRY[arch](**kwargs).to(device)

    # Optimizer — constant LR with linear warmup (no cosine: total steps unknown)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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

        # Linear warmup; constant LR after (#5)
        if warmup_time_s > 0 and elapsed < warmup_time_s:
            for pg in optimizer.param_groups:
                pg["lr"] = lr * (elapsed / warmup_time_s)
        else:
            for pg in optimizer.param_groups:
                pg["lr"] = lr

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

    # Compute energy metrics
    task_metrics = task.compute_metrics(samples, dataset)

    wall_time = time.time() - t0

    # Cleanup
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "energy_wasserstein": task_metrics.get("energy_wasserstein", float("inf")),
        "steps": step,
        "train_time_s": round(actual_train_time, 1),
        "wall_time_s": round(wall_time, 1),
        "model_size": model_size,
        "lr": lr,
    }
    result.update(task_metrics)
    return result


def parse_variants(variants_str: str | None) -> list[dict]:
    """Parse variant grid specification.

    Format: "size=xs,small,medium,large;lr=1e-4,3e-4"
    Returns list of dicts with all combinations.
    """
    if variants_str is None:
        # Default grid: 2 sizes x 4 LRs = 8 variants
        sizes = ["small", "medium"]
        lrs = [3e-5, 1e-4, 3e-4, 1e-3]
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


def _batch_size_for_variant(base_batch_size: int, model_size: str) -> int:
    """Scale batch size down for larger model sizes to avoid OOM.

    xs/small use the full batch size. medium halves it, large quarters it.
    """
    scale = {"xs": 1.0, "small": 1.0, "medium": 0.5, "large": 0.25, "xl": 0.125}
    factor = scale.get(model_size, 0.25)
    return max(16, int(base_batch_size * factor))


def _build_subprocess_cmd(arch: str, args, variant: dict) -> list[str]:
    """Build the Python command to run a single variant in a subprocess.

    Wraps in try/except so import errors and crashes produce JSON, not silent failure (#2).
    """
    batch_size = _batch_size_for_variant(args.batch_size, variant["model_size"])
    script = (
        f"import json, sys, traceback, torch\n"
        f"sys.path.insert(0, '.')\n"
        f"_dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'\n"
        f"try:\n"
        f"    from autoresearch.run import quick_train_eval\n"
        f"    result = quick_train_eval(\n"
        f"        arch={arch!r}, data_path={args.data_path!r},\n"
        f"        task_name={args.task_name!r},\n"
        f"        train_time_s={args.train_time_s}, device=_dev,\n"
        f"        model_size={variant['model_size']!r}, lr={variant['lr']},\n"
        f"        batch_size={batch_size}, n_ode_steps={args.n_ode_steps},\n"
        f"        eval_samples={args.eval_samples}, warmup_frac={args.warmup_frac},\n"
        f"        weight_decay={args.weight_decay})\n"
        f"    print(json.dumps(result))\n"
        f"except Exception as e:\n"
        f"    print(json.dumps({{'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()}}))\n"
    )
    return [sys.executable, "-c", script]


def _parse_subprocess_output(stdout: str, stderr: str, returncode: int) -> dict:
    """Extract JSON result from subprocess output."""
    if stdout and stdout.strip():
        for line in reversed(stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
    return {"status": "error", "error": f"exit={returncode}, stderr={stderr[-500:] if stderr else '(empty)'}"}


def _preflight_import_check():
    """Verify that editable modules (flow_matching, models) parse without error.

    Runs a quick import in a subprocess so that a syntax error in agent-edited
    code fails fast instead of wasting GPU time on 24 identical import failures.
    """
    script = (
        "import sys; sys.path.insert(0, '.'); "
        "import flow_matching.training, flow_matching.sampling, flow_matching.interpolation; "
        "from experiments.model_registry import MODEL_REGISTRY; "
        "[cls() for cls in []]"  # no-op, just ensure imports succeed
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        error_msg = result.stderr.strip().split("\n")[-3:]  # last 3 lines of traceback
        print(json.dumps({
            "status": "error",
            "error": "Import check failed — edited code has errors",
            "detail": "\n".join(error_msg),
        }, indent=2))
        sys.exit(1)


def run_all_jobs(
    archs: list[str], args, variants: list[dict], n_gpus: int,
) -> dict[str, dict[str, dict]]:
    """Run all (arch x variant) combinations across GPUs in a single scheduling pass.

    Always uses subprocesses, even for n_gpus=1, so that agent edits to
    flow_matching/ or models/ are picked up fresh (no stale module cache) (#6).

    Returns {arch: {variant_label: result_dict}}.
    """
    results = {arch: {} for arch in archs}

    # Flatten all (arch, variant) into a single job queue — subprocess per job
    # so agent edits to flow_matching/ or models/ are always picked up fresh.
    n_slots = max(n_gpus, 1)  # at least 1 slot even for CPU-only
    use_gpu = n_gpus >= 1 and torch.cuda.is_available()
    job_queue = [(arch, v) for arch in archs for v in variants]
    total = len(job_queue)
    # slot -> (proc, arch, label, start_time)
    active: dict[int, tuple[subprocess.Popen, str, str, float]] = {}
    completed = 0

    # Per-job timeout: training time + generous margin for data loading + eval
    job_timeout_s = args.train_time_s + 600

    def _launch(slot_id: int, arch: str, variant: dict):
        label = variant_label(variant)
        cmd = _build_subprocess_cmd(arch, args, variant)
        if use_gpu:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(slot_id)}
        else:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        active[slot_id] = (proc, arch, label, time.time())
        gpu_str = f"GPU {slot_id}" if use_gpu else "CPU"
        print(f"  [{arch}] [{gpu_str}] Started: {label}", file=sys.stderr)

    def _finish(slot_id: int, proc: subprocess.Popen, arch: str, label: str, timed_out: bool):
        nonlocal completed
        if timed_out:
            proc.kill()
            proc.wait(timeout=10)
            try:
                stdout = proc.stdout.read()
                stderr = proc.stderr.read()
            except Exception:
                stdout, stderr = "", ""
            parsed = {"status": "error", "error": f"Killed: exceeded {job_timeout_s:.0f}s timeout"}
        else:
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            parsed = _parse_subprocess_output(stdout, stderr, proc.returncode)

        results[arch][label] = parsed
        completed += 1
        status = "OK" if "status" not in parsed else parsed["status"]
        gpu_str = f"GPU {slot_id}" if use_gpu else "CPU"
        timeout_str = " [TIMEOUT]" if timed_out else ""
        print(
            f"  [{arch}] [{gpu_str}] Done ({completed}/{total}): {label} [{status}]{timeout_str}",
            file=sys.stderr,
        )

    # Fill initial slots
    for slot_id in range(min(n_slots, len(job_queue))):
        arch, variant = job_queue.pop(0)
        _launch(slot_id, arch, variant)

    while active:
        time.sleep(2)
        for slot_id in list(active.keys()):
            proc, arch, label, start_t = active[slot_id]
            ret = proc.poll()
            elapsed = time.time() - start_t

            if ret is not None:
                # Process exited normally
                del active[slot_id]
                _finish(slot_id, proc, arch, label, timed_out=False)
                if job_queue:
                    next_arch, next_variant = job_queue.pop(0)
                    _launch(slot_id, next_arch, next_variant)
            elif elapsed > job_timeout_s:
                # Process hung — kill it
                del active[slot_id]
                _finish(slot_id, proc, arch, label, timed_out=True)
                if job_queue:
                    next_arch, next_variant = job_queue.pop(0)
                    _launch(slot_id, next_arch, next_variant)

    return results


# ── Aggregation helpers ───────────────────────────────────────────────────────


def _best_variant_result(variant_results: dict[str, dict]) -> tuple[float | None, str | None, list[str]]:
    """Find best (lowest) energy Wasserstein distance from variant results.

    Returns (best_value, best_label, all_succeeded_labels).
    """
    best_val = None
    best_label = None
    succeeded = []
    for label, r in variant_results.items():
        if "energy_wasserstein" in r and r.get("status") != "error":
            succeeded.append(label)
            if best_val is None or r["energy_wasserstein"] < best_val:
                best_val = r["energy_wasserstein"]
                best_label = label
    return best_val, best_label, succeeded


def load_previous_best_per_arch(log_path: str, data_name: str | None = None) -> dict[str, float]:
    """Load best energy Wasserstein per architecture from the experiment log.

    Scans all kept entries and tracks the best per-arch metric seen.
    Reads the "best" field (best-of-N); falls back to "aggregate" for old entries.

    Args:
        data_name: If provided, only consider entries matching this data config
            (e.g. "multibody_23_N50_T1.0"). Prevents cross-task contamination when switching
            between tasks.

    Returns:
        {arch: best_metric} for every architecture that has a kept result.
    """
    if not os.path.isfile(log_path):
        return {}
    best: dict[str, float] = {}
    with open(log_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not entry.get("kept"):
                continue

            # Filter by data config if specified (skip entries from other tasks)
            if data_name is not None and entry.get("data") is not None:
                if entry["data"] != data_name:
                    continue

            for arch, arch_data in entry.get("per_arch", {}).items():
                # Prefer "best" (best-of-N), fall back to "aggregate" (old entries)
                val = arch_data.get("best") or arch_data.get("aggregate")
                if val is not None and (arch not in best or val < best[arch]):
                    best[arch] = val

    return best


def load_baseline(baseline_path: str, data_name: str) -> float | None:
    """Load oracle baseline metric."""
    if not os.path.isfile(baseline_path):
        return None
    with open(baseline_path) as f:
        baselines = json.load(f)
    return baselines.get(data_name, {}).get("energy_wasserstein")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Autoresearch quick train+eval harness")
    parser.add_argument(
        "--archs", required=True,
        help='Architecture to test (e.g. "transformer", )',
    )
    parser.add_argument("--data", required=True, help="Data config name (e.g. multibody_23_N50_T1.0)")
    parser.add_argument("--train_time", type=float, default=10.0, help="Training time per variant in minutes (default: 10)")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_ode_steps", type=int, default=50)
    parser.add_argument("--eval_samples", type=int, default=2000)
    parser.add_argument("--warmup_frac", type=float, default=0.1, help="Fraction of training time for LR warmup (default: 0.1)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--variants", type=str, default=None, help='Variant grid: "size=xs,small;lr=1e-4,3e-4"')
    parser.add_argument("--description", type=str, default=None, help="Experiment description (auto-logs when provided)")
    args = parser.parse_args()
    args.train_time_s = args.train_time * 60  # convert to seconds

    # Parse architecture (single arch for autoresearch)
    arch = args.archs.strip()
    if arch not in MODEL_REGISTRY:
        print(json.dumps({"status": "error", "error": f"Unknown arch: {arch}. Available: {ALL_ARCHS}"}))
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
    args.task_name = data_cfg.get("task", "multibody")
    if not os.path.isfile(args.data_path):
        print(json.dumps({"status": "error", "error": f"Training data not found: {args.data_path}"}))
        sys.exit(1)

    variants = parse_variants(args.variants)
    print(
        f"Running {len(variants)} variants for {arch} on {args.n_gpus} GPU(s), "
        f"{args.train_time} min each",
        file=sys.stderr,
    )

    # ── Sanity check: verify editable modules parse before launching jobs ──
    _preflight_import_check()

    # ── Run all variant jobs across GPUs ──────────────────────────────────
    all_results = run_all_jobs([arch], args, variants, args.n_gpus)
    best_val, best_label, succeeded = _best_variant_result(all_results[arch])

    if best_val is None:
        output = {
            "status": "error",
            "error": f"All {arch} variants failed",
            "per_arch": {arch: {"variants": all_results[arch]}},
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)

    n_ok = len(succeeded)
    n_total = len(all_results[arch])
    print(f"  [{arch}] Best energy Wasserstein: {best_val:.6f} ({best_label}, {n_ok}/{n_total} ok)", file=sys.stderr)

    # Load baseline and previous best
    baseline_path = "outputs/autoresearch/baselines.json"
    baseline_metric = load_baseline(baseline_path, args.data)

    log_path = "outputs/autoresearch/experiments.jsonl"
    previous_best_per_arch = load_previous_best_per_arch(log_path, data_name=args.data)
    prev = previous_best_per_arch.get(arch)
    kept = best_val < prev if prev is not None else True

    # Build per-arch output (used in both JSON output and experiment log)
    per_arch_output = {
        arch: {
            "variants": all_results[arch],
            "best": round(best_val, 6),
            "best_variant": best_label,
            "previous_best": round(prev, 6) if prev is not None else None,
            "improved": kept,
        },
    }

    output = {
        "status": "success",
        "arch": arch,
        "metric_name": "energy_wasserstein",
        "per_arch": per_arch_output,
        "best_metric": round(best_val, 6),
        "baseline_metric": baseline_metric,
        "previous_best": round(prev, 6) if prev is not None else None,
        "delta": round(best_val - prev, 6) if prev is not None else None,
        "best_variant": best_label,
        "kept": kept,
    }

    print(json.dumps(output, indent=2))

    # ── Auto-log when --description is provided (#1) ──────────────────────
    if args.description is not None:
        from autoresearch.experiment_log import get_changed_files, log_experiment

        # Only report files that are actually editable by autoresearch sessions
        all_changed = get_changed_files(committed=False)
        editable_prefixes = [f"models/{arch}.py", "flow_matching/"]
        relevant_changed = [
            f for f in all_changed
            if any(f == p or f.startswith(p) for p in editable_prefixes)
        ]

        log_experiment(
            log_path=log_path,
            description=args.description,
            archs=[arch],
            per_arch=per_arch_output,
            best_metric=round(best_val, 6),
            baseline_metric=baseline_metric,
            previous_best_per_arch={
                a: round(v, 6) for a, v in previous_best_per_arch.items()
            },
            kept=kept,
            data=args.data,
            files_changed=relevant_changed,
        )
        print(f"Logged to {log_path} (kept={kept})", file=sys.stderr)

    # Auto-update visualization if experiment log exists
    if os.path.isfile(log_path):
        try:
            subprocess.run(
                [sys.executable, "autoresearch/visualize.py",
                 "--log", log_path, "--output", "outputs/autoresearch/plots",
                 "--data", args.data],
                capture_output=True, timeout=60,
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
