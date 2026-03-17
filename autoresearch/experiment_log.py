"""Append-only JSONL experiment tracker for autoresearch sessions."""

import json
import os
import subprocess
from datetime import datetime, timezone


def get_git_sha() -> str:
    """Get current git SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_changed_files(committed: bool = False) -> list[str]:
    """Get list of files relevant to this experiment.

    Args:
        committed: True if changes were already committed (use HEAD~1..HEAD).
            False if changes are still uncommitted (use diff against HEAD).
            When called from auto-log in run.py, changes are always uncommitted
            at log time (agent commits/reverts after), so default is False.
    """
    try:
        if committed:
            cmd = ["git", "diff", "--name-only", "HEAD~1", "HEAD"]
        else:
            cmd = ["git", "diff", "--name-only", "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return [f for f in result.stdout.strip().split("\n") if f]
    except Exception:
        return []


def log_experiment(
    log_path: str,
    description: str,
    archs: list[str],
    per_arch: dict[str, dict],
    best_metric: float,
    baseline_metric: float | None,
    previous_best_per_arch: dict[str, float | None],
    kept: bool,
    data: str | None = None,
    files_changed: list[str] | None = None,
) -> None:
    """Append one experiment entry to the JSONL log.

    Args:
        archs: List of architectures tested.
        per_arch: {arch: {"variants": {...}, "best": float}} per architecture.
        best_metric: Best g(r) distance across all variants (for single-arch runs).
        previous_best_per_arch: {arch: best_metric_for_that_arch} from history.
        data: Data config name (e.g. "chain_N50", "hard_sphere_N50").
        files_changed: Explicit file list; auto-detected from git if None.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if files_changed is None:
        files_changed = get_changed_files(committed=False)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "description": description,
        "data": data,
        "archs": archs,
        "per_arch": per_arch,
        "best_metric": best_metric,
        "baseline_metric": baseline_metric,
        "previous_best_per_arch": previous_best_per_arch,
        "kept": kept,
        "files_changed": files_changed,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_experiments(log_path: str) -> list[dict]:
    """Load all experiments from the JSONL log."""
    if not os.path.isfile(log_path):
        return []
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
