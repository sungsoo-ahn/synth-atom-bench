"""Fully automated scaling law pipeline: data → train → collect → fit → report."""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from experiments.model_registry import ARCH_DISPLAY_NAMES, MODEL_DEFAULTS, MODEL_REGISTRY, SIZE_PRESETS
from experiments.tasks import TASK_REGISTRY

BUDGETS_DEFAULT = [1e15, 4e15, 1.6e16, 6.4e16, 2.56e17]
LEARNING_RATES = [1e-4, 1e-3]
ALL_SIZES = ["xs", "small", "medium", "large", "xl"]
MIN_STEPS = 2000
MAX_STEPS = 1_000_000

DATA_SPLITS = {
    "train": 50_000,
    "val": 5_000,
    "test": 10_000,
}

# Framework registry — hook point for future alternative frameworks
FRAMEWORK_REGISTRY = {
    "flow_matching": "flow_matching.training.flow_matching_loss",
}


def log(msg: str):
    print(f"[scaling_auto] {msg}", flush=True)


# ── Stage 1: Data check / generation ──────────────────────────────────────────


def ensure_data(task_name: str, n_atoms: int, data_dir: str, eta: float = 0.3):
    """Generate train/val/test data if it doesn't exist."""
    task = TASK_REGISTRY[task_name]()
    # For HardSphereTask, pass eta
    if hasattr(task, "eta"):
        task.eta = eta

    all_exist = all(
        os.path.isfile(os.path.join(data_dir, f"{split}.npz"))
        for split in DATA_SPLITS
    )
    if all_exist:
        log(f"Data exists at {data_dir}")
        return

    os.makedirs(data_dir, exist_ok=True)
    for split, n_samples in DATA_SPLITS.items():
        path = os.path.join(data_dir, f"{split}.npz")
        if os.path.isfile(path):
            log(f"  {split}.npz exists, skipping")
            continue
        cmd = task.data_generator_cmd(n_atoms, n_samples, path)
        # Use different seeds per split
        seed = {"train": 42, "val": 123, "test": 456}[split]
        cmd += f" --seed {seed}"
        log(f"  Generating {split}: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            log(f"  ERROR: data generation failed for {split}")
            sys.exit(1)


# ── Stage 2: Oracle baseline ──────────────────────────────────────────────────


def compute_baseline(data_name: str, data_dir: str) -> float | None:
    """Compute oracle g(r) distance using autoresearch/baseline.py."""
    cache_path = "outputs/autoresearch/baselines.json"
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            baselines = json.load(f)
        if data_name in baselines:
            log(f"Oracle baseline (cached): {baselines[data_name]['gr_distance']}")
            return baselines[data_name]["gr_distance"]

    log("Computing oracle baseline...")
    result = subprocess.run(
        [sys.executable, "autoresearch/baseline.py", "--data", data_name],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"  Warning: baseline computation failed: {result.stderr[-200:]}")
        return None

    # Parse JSON from stdout
    for line in reversed(result.stdout.strip().split("\n")):
        if line.strip().startswith("{"):
            try:
                data = json.loads(line)
                log(f"Oracle baseline: {data.get('gr_distance')}")
                return data.get("gr_distance")
            except json.JSONDecodeError:
                pass

    # Try loading from cache (baseline.py writes it)
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            baselines = json.load(f)
        if data_name in baselines:
            return baselines[data_name]["gr_distance"]

    return None


# ── Stage 3: FLOP measurement ────────────────────────────────────────────────


def measure_all_flops(
    archs: list[str], sizes: list[str], batch_size: int, n_atoms: int, cache_path: str,
) -> dict[str, dict]:
    """Measure FLOPs for all (arch, size) combos, with caching."""
    from experiments.scaling import measure_flops

    # Load cache
    cache = {}
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    results = {}
    for arch in archs:
        for size in sizes:
            key = f"{arch}_{size}_bs{batch_size}_n{n_atoms}"
            if key in cache:
                results[(arch, size)] = cache[key]
                continue

            log(f"  Measuring FLOPs: {arch} {size}...")
            flops, n_params = measure_flops(arch, size, batch_size, n_atoms)
            entry = {"flops_per_step": flops, "n_params": n_params}
            results[(arch, size)] = entry
            cache[key] = entry

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    return results


# ── Stage 4: Grid generation ─────────────────────────────────────────────────


def generate_grid(
    archs: list[str],
    sizes: list[str],
    lrs: list[float],
    budgets: list[float],
    flops_table: dict,
    data_name: str,
    scaling_dir: str,
    batch_size: int,
) -> list[dict]:
    """Generate list of job configs. Returns list of job dicts."""
    from itertools import product

    jobs = []
    skipped = 0
    for budget, arch, size, lr in product(budgets, archs, sizes, lrs):
        entry = flops_table.get((arch, size))
        if entry is None:
            skipped += 1
            continue
        flops_per_step = entry["flops_per_step"]
        max_steps = int(budget / flops_per_step)

        if max_steps < MIN_STEPS or max_steps > MAX_STEPS:
            skipped += 1
            continue

        run_name = f"{arch}_{size}_lr{lr:.0e}_budget{budget:.2e}"
        ckpt_dir = os.path.join(scaling_dir, run_name)

        preset = SIZE_PRESETS[arch][size]
        model_overrides = " ".join(f"model.model_kwargs.{k}={v}" for k, v in preset.items())

        cmd = (
            f"uv run python experiments/train.py "
            f"data={data_name} "
            f"model={arch} "
            f"model.size={size} "
            f"train.lr={lr} "
            f"train.max_steps={max_steps} "
            f"{model_overrides} "
            f"checkpoint.dir={ckpt_dir} "
            f"logging.enabled=true "
            f"hydra.run.dir={ckpt_dir}"
        )

        jobs.append({
            "run_name": run_name,
            "cmd": cmd,
            "arch": arch,
            "size": size,
            "lr": lr,
            "budget": budget,
            "max_steps": max_steps,
            "flops_per_step": flops_per_step,
        })

    log(f"Grid: {len(jobs)} jobs ({skipped} skipped)")
    return jobs


# ── Stage 5: Execution ───────────────────────────────────────────────────────


def is_job_complete(scaling_dir: str, run_name: str) -> bool:
    """Check if a job has already completed."""
    run_dir = os.path.join(scaling_dir, run_name)
    latest_pt = os.path.join(run_dir, "latest.pt")
    if not os.path.isfile(latest_pt):
        return False
    try:
        data = torch.load(latest_pt, map_location="cpu", weights_only=False)
        saved_step = data.get("step", 0)
        config = data.get("config", {})
        max_steps = config.get("train", {}).get("max_steps", 0)
        return max_steps > 0 and saved_step >= max_steps
    except Exception:
        return False


def run_jobs(jobs: list[dict], scaling_dir: str, n_gpus: int, max_retries: int = 2):
    """Execute jobs with parallel GPU scheduling and retry on failure."""
    # Filter already-complete jobs
    remaining = []
    for job in jobs:
        if is_job_complete(scaling_dir, job["run_name"]):
            log(f"  Skipping (done): {job['run_name']}")
        else:
            remaining.append(job)

    log(f"Executing {len(remaining)} jobs ({len(jobs) - len(remaining)} already done)")
    if not remaining:
        return

    # Track retries
    retry_counts: dict[str, int] = {}
    failed_jobs: list[dict] = []

    def _execute_batch(batch: list[dict]):
        if n_gpus <= 1:
            for job in batch:
                log(f"  Running: {job['run_name']}")
                result = subprocess.run(job["cmd"], shell=True)
                if result.returncode != 0:
                    count = retry_counts.get(job["run_name"], 0)
                    if count < max_retries:
                        retry_counts[job["run_name"]] = count + 1
                        log(f"  Retrying ({count+1}/{max_retries}): {job['run_name']}")
                        failed_jobs.append(job)
                    else:
                        log(f"  FAILED after {max_retries} retries: {job['run_name']}")
        else:
            active: dict[int, tuple[subprocess.Popen, dict]] = {}
            job_queue = list(batch)
            completed = 0
            total = len(batch)

            def _launch(gpu_id: int, job: dict):
                run_dir = os.path.join(scaling_dir, job["run_name"])
                os.makedirs(run_dir, exist_ok=True)
                log_path = os.path.join(run_dir, "train.log")
                log_f = open(log_path, "w")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
                proc = subprocess.Popen(
                    job["cmd"], shell=True, stdout=log_f, stderr=subprocess.STDOUT, env=env,
                )
                proc._log_file = log_f
                active[gpu_id] = (proc, job)
                log(f"  [GPU {gpu_id}] Started: {job['run_name']}")

            # Fill initial slots
            for gpu_id in range(min(n_gpus, len(job_queue))):
                _launch(gpu_id, job_queue.pop(0))

            while active:
                time.sleep(5)
                for gpu_id in list(active.keys()):
                    proc, job = active[gpu_id]
                    ret = proc.poll()
                    if ret is not None:
                        proc._log_file.close()
                        del active[gpu_id]
                        completed += 1

                        if ret == 0:
                            log(f"  [GPU {gpu_id}] Done ({completed}/{total}): {job['run_name']}")
                        else:
                            count = retry_counts.get(job["run_name"], 0)
                            if count < max_retries:
                                retry_counts[job["run_name"]] = count + 1
                                log(f"  [GPU {gpu_id}] Failed, queuing retry ({count+1}/{max_retries}): {job['run_name']}")
                                job_queue.append(job)
                            else:
                                log(f"  [GPU {gpu_id}] FAILED after {max_retries} retries: {job['run_name']}")

                        if job_queue:
                            _launch(gpu_id, job_queue.pop(0))

    _execute_batch(remaining)

    # Process retries
    if failed_jobs:
        log(f"Retrying {len(failed_jobs)} failed jobs...")
        _execute_batch(failed_jobs)


# ── Stage 6: Collection ──────────────────────────────────────────────────────


def collect_results(scaling_dir: str, jobs: list[dict]) -> dict:
    """Collect results from completed runs. Returns structured results dict."""
    # Build job metadata lookup
    job_meta = {j["run_name"]: j for j in jobs}

    results = []
    for run_name in sorted(os.listdir(scaling_dir)):
        run_dir = os.path.join(scaling_dir, run_name)
        if not os.path.isdir(run_dir):
            continue

        ckpt_path = os.path.join(run_dir, "latest.pt")
        if not os.path.isfile(ckpt_path):
            ckpt_path = os.path.join(run_dir, "best.pt")
        if not os.path.isfile(ckpt_path):
            continue

        try:
            data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            config = data.get("config", {})
            arch = config.get("model", {}).get("arch", "unknown")
            size = config.get("model", {}).get("size", "unknown")
            lr = config.get("train", {}).get("lr", 0)
            step = data.get("step", 0)
            cr = data.get("best_clash_rate", float("inf"))
            grd = data.get("best_gr_distance", float("inf"))

            meta = job_meta.get(run_name, {})
            budget = meta.get("budget", 0)
            flops_per_step = meta.get("flops_per_step", 0)

            results.append({
                "run": run_name,
                "arch": arch,
                "size": size,
                "lr": lr,
                "budget": budget,
                "best_clash_rate": cr,
                "best_gr_distance": grd,
                "step": step,
                "flops_per_step": flops_per_step,
                "total_flops": flops_per_step * step,
            })
        except Exception as e:
            log(f"  Warning: failed to load {ckpt_path}: {e}")

    if not results:
        log("No results found!")
        return {"all_results": [], "best_per_budget": {}}

    # Best per (arch, budget)
    best_per_budget = {}
    for r in results:
        key = (r["arch"], r["budget"])
        if key not in best_per_budget:
            best_per_budget[key] = r
        else:
            prev = best_per_budget[key]
            if r["best_gr_distance"] < prev["best_gr_distance"]:
                best_per_budget[key] = r

    log(f"Collected {len(results)} results, {len(best_per_budget)} unique (arch, budget) pairs")

    out = {
        "all_results": results,
        "best_per_budget": {
            f"{arch}_{budget:.0e}": r for (arch, budget), r in best_per_budget.items()
        },
    }

    results_path = os.path.join(scaling_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    return out


# ── Stage 7: Fitting ─────────────────────────────────────────────────────────


def fit_and_plot(
    scaling_dir: str, results: dict, oracle_baseline: float | None,
):
    """Fit scaling laws and generate plots."""
    from viz.scaling import fit_scaling_law, plot_scaling_curves
    from viz.style import ARCH_COLORS, ARCH_MARKERS, save_figure, synthbench_style

    import matplotlib.pyplot as plt

    best_per_budget = results["best_per_budget"]
    if not best_per_budget:
        log("No results to fit")
        return {}

    # Organize by architecture
    arch_data = {}
    for key, r in best_per_budget.items():
        if r["budget"] <= 0:
            continue
        arch = r["arch"]
        if arch not in arch_data:
            arch_data[arch] = {"flops": [], "clash_rate": [], "gr_distance": []}
        arch_data[arch]["flops"].append(r["budget"])
        arch_data[arch]["clash_rate"].append(r["best_clash_rate"])
        arch_data[arch]["gr_distance"].append(r.get("best_gr_distance", float("inf")))

    # Sort by budget
    for arch in arch_data:
        order = np.argsort(arch_data[arch]["flops"])
        arch_data[arch]["flops"] = np.array(arch_data[arch]["flops"])[order]
        arch_data[arch]["clash_rate"] = np.array(arch_data[arch]["clash_rate"])[order]
        arch_data[arch]["gr_distance"] = np.array(arch_data[arch]["gr_distance"])[order]

    plots_dir = os.path.join(scaling_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Fit and report
    fits = {}
    log("\nScaling Law Fits (clash_rate):")
    log(f"  {'Architecture':<15} {'alpha':>8} {'prefactor':>12} {'floor':>10}")
    for arch, d in arch_data.items():
        flops = d["flops"]
        cr = d["clash_rate"]
        if len(flops) < 3:
            continue
        try:
            a, alpha, floor = fit_scaling_law(flops, cr)
            fits[f"{arch}_clash_rate"] = {"a": a, "alpha": alpha, "floor": floor}
            log(f"  {arch:<15} {alpha:>8.3f} {a:>12.4f} {floor:>10.5f}")
        except Exception as e:
            log(f"  {arch:<15} fit failed: {e}")

    # Fit g(r) distance
    log("\nScaling Law Fits (gr_distance):")
    log(f"  {'Architecture':<15} {'alpha':>8} {'prefactor':>12} {'floor':>10}")
    for arch, d in arch_data.items():
        flops = d["flops"]
        grd = d["gr_distance"]
        valid = np.isfinite(grd)
        if valid.sum() < 3:
            continue
        try:
            a, alpha, floor = fit_scaling_law(flops[valid], grd[valid])
            fits[f"{arch}_gr_distance"] = {"a": a, "alpha": alpha, "floor": floor}
            log(f"  {arch:<15} {alpha:>8.3f} {a:>12.4f} {floor:>10.5f}")
        except Exception as e:
            log(f"  {arch:<15} fit failed: {e}")

    # Save fits
    fits_path = os.path.join(scaling_dir, "scaling_fits.json")
    with open(fits_path, "w") as f:
        json.dump(fits, f, indent=2)

    # Plot clash_rate scaling curves
    plot_data = {}
    for arch, d in arch_data.items():
        display_name = ARCH_DISPLAY_NAMES.get(arch, arch)
        plot_data[display_name] = d

    with synthbench_style():
        fig = plot_scaling_curves(plot_data, fit_curves=True)
        if oracle_baseline is not None:
            ax = fig.get_axes()[0]
            ax.axhline(oracle_baseline, color="#3498db", linestyle="--", alpha=0.5,
                       label=f"Oracle baseline ({oracle_baseline:.4f})")
            ax.legend(frameon=False)
        save_figure(fig, os.path.join(plots_dir, "scaling_clash_rate"))
        log(f"Saved {plots_dir}/scaling_clash_rate.png")

    # Plot g(r) distance scaling curves
    gr_plot_data = {}
    for arch, d in arch_data.items():
        valid = np.isfinite(d["gr_distance"])
        if valid.any():
            display_name = ARCH_DISPLAY_NAMES.get(arch, arch)
            gr_plot_data[display_name] = {
                "flops": d["flops"][valid],
                "clash_rate": d["gr_distance"][valid],
            }

    if gr_plot_data:
        with synthbench_style():
            fig = plot_scaling_curves(gr_plot_data, fit_curves=True, ylabel="g(r) L1 distance")
            if oracle_baseline is not None:
                ax = fig.get_axes()[0]
                ax.axhline(oracle_baseline, color="#3498db", linestyle="--", alpha=0.5,
                           label=f"Oracle baseline ({oracle_baseline:.4f})")
                ax.legend(frameon=False)
            save_figure(fig, os.path.join(plots_dir, "scaling_gr_distance"))
            log(f"Saved {plots_dir}/scaling_gr_distance.png")

    return fits


# ── Stage 8: Report ──────────────────────────────────────────────────────────


def generate_report(
    scaling_dir: str, results: dict, fits: dict, oracle_baseline: float | None,
):
    """Generate a summary report."""
    report_lines = ["# Scaling Law Report", ""]

    if oracle_baseline is not None:
        report_lines.append(f"Oracle g(r) baseline: {oracle_baseline:.6f}")
        report_lines.append("")

    # Best results per architecture
    best_per_budget = results.get("best_per_budget", {})
    if best_per_budget:
        report_lines.append("## Best Results per (Architecture, Budget)")
        report_lines.append("")
        report_lines.append(f"{'Arch':<12} {'Budget':>10} {'CR':>10} {'g(r)':>10} {'Size':<6}")
        report_lines.append("-" * 55)
        for key in sorted(best_per_budget.keys()):
            r = best_per_budget[key]
            grd_str = f"{r['best_gr_distance']:>10.4f}" if r["best_gr_distance"] < float("inf") else "       n/a"
            report_lines.append(
                f"{r['arch']:<12} {r['budget']:>10.0e} {r['best_clash_rate']:>10.4f} {grd_str} {r['size']:<6}"
            )
        report_lines.append("")

    # Scaling exponents
    if fits:
        report_lines.append("## Scaling Exponents")
        report_lines.append("")
        report_lines.append(f"{'Metric':<30} {'alpha':>8} {'floor':>10}")
        report_lines.append("-" * 52)
        for key in sorted(fits.keys()):
            f = fits[key]
            report_lines.append(f"{key:<30} {f['alpha']:>8.3f} {f['floor']:>10.5f}")
        report_lines.append("")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(scaling_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    log(f"Report saved to {report_path}")
    print(report_text)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Fully automated scaling law pipeline")
    parser.add_argument("--task", default="hard_sphere", choices=list(TASK_REGISTRY.keys()))
    parser.add_argument("--n_atoms", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.3, help="Packing fraction (hard_sphere only)")
    parser.add_argument("--archs", default="equiv_gnn,transformer,pairformer")
    parser.add_argument("--budgets", default=None, help="Comma-separated FLOP budgets")
    parser.add_argument("--sizes", default=None, help="Comma-separated sizes (default: all)")
    parser.add_argument("--lrs", default=None, help="Comma-separated learning rates")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", default=None, help="Output directory (default: auto)")
    parser.add_argument("--max_retries", type=int, default=2)
    args = parser.parse_args()

    archs = [a.strip() for a in args.archs.split(",")]
    sizes = [s.strip() for s in args.sizes.split(",")] if args.sizes else ALL_SIZES
    lrs = [float(x) for x in args.lrs.split(",")] if args.lrs else LEARNING_RATES
    budgets = [float(x) for x in args.budgets.split(",")] if args.budgets else BUDGETS_DEFAULT

    data_name = f"{args.task}_N{args.n_atoms}"
    data_dir = f"outputs/data/{data_name}"
    scaling_dir = args.output_dir or f"outputs/scaling/{data_name}"
    flops_cache_path = os.path.join(scaling_dir, "flops_cache.json")

    log(f"Task: {args.task}, N={args.n_atoms}, archs={archs}")
    log(f"Budgets: {[f'{b:.0e}' for b in budgets]}")
    log(f"Output: {scaling_dir}")

    # Stage 1: Data
    log("\n=== Stage 1: Data check/generation ===")
    ensure_data(args.task, args.n_atoms, data_dir, eta=args.eta)

    # Stage 2: Oracle baseline
    log("\n=== Stage 2: Oracle baseline ===")
    oracle_baseline = compute_baseline(data_name, data_dir)

    # Stage 3: FLOP measurement
    log("\n=== Stage 3: FLOP measurement ===")
    flops_table = measure_all_flops(archs, sizes, args.batch_size, args.n_atoms, flops_cache_path)

    # Stage 4: Grid generation
    log("\n=== Stage 4: Grid generation ===")
    jobs = generate_grid(archs, sizes, lrs, budgets, flops_table, data_name, scaling_dir, args.batch_size)

    # Save grid metadata
    os.makedirs(scaling_dir, exist_ok=True)
    meta = {
        "task": args.task,
        "n_atoms": args.n_atoms,
        "archs": archs,
        "budgets": budgets,
        "runs": [
            {k: v for k, v in j.items() if k != "cmd"}
            for j in jobs
        ],
    }
    with open(os.path.join(scaling_dir, "grid_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Stage 5: Execution
    log("\n=== Stage 5: Training ===")
    run_jobs(jobs, scaling_dir, args.n_gpus, max_retries=args.max_retries)

    # Stage 6: Collection
    log("\n=== Stage 6: Collecting results ===")
    results = collect_results(scaling_dir, jobs)

    # Stage 7: Fitting
    log("\n=== Stage 7: Fitting scaling laws ===")
    fits = fit_and_plot(scaling_dir, results, oracle_baseline)

    # Stage 8: Report
    log("\n=== Stage 8: Report ===")
    generate_report(scaling_dir, results, fits, oracle_baseline)

    log("\nDone!")


if __name__ == "__main__":
    main()
