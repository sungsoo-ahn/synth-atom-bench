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


def get_changed_files() -> list[str]:
    """Get list of files changed since last commit."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return [f for f in result.stdout.strip().split("\n") if f]
    except Exception:
        return []


def log_experiment(
    log_path: str,
    description: str,
    arch: str,
    variant_results: dict,
    aggregate_metric: float,
    baseline_metric: float | None,
    previous_best: float | None,
    kept: bool,
) -> None:
    """Append one experiment entry to the JSONL log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "description": description,
        "arch": arch,
        "variants": variant_results,
        "aggregate_metric": aggregate_metric,
        "baseline_metric": baseline_metric,
        "previous_best": previous_best,
        "kept": kept,
        "files_changed": get_changed_files(),
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
