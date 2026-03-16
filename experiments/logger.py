"""File-based logger and GPU compute tracker."""

import json
import os
import time
from dataclasses import dataclass

import torch


@dataclass
class LoggerConfig:
    """Configuration for experiment logging."""

    enabled: bool = True
    log_every_n_steps: int = 100
    eval_every_n_epochs: int = 1
    checkpoint_every_n_epochs: int = 5
    log_dir: str = "outputs/logs"


class ComputeTracker:
    """Track elapsed GPU-hours using CUDA events or wall clock."""

    def __init__(self):
        self._use_cuda = torch.cuda.is_available()
        self._elapsed_seconds: float = 0.0
        self._running = False
        if self._use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
        else:
            self._wall_start: float = 0.0

    def start(self) -> None:
        self._running = True
        if self._use_cuda:
            self._start_event.record()
        else:
            self._wall_start = time.monotonic()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._use_cuda:
            self._end_event.record()
            torch.cuda.synchronize()
            self._elapsed_seconds += self._start_event.elapsed_time(self._end_event) / 1000.0
        else:
            self._elapsed_seconds += time.monotonic() - self._wall_start

    def reset(self) -> None:
        self._elapsed_seconds = 0.0
        self._running = False

    @property
    def gpu_hours(self) -> float:
        return self._elapsed_seconds / 3600.0


class Logger:
    """File-based experiment logger. Writes JSON lines to a log directory.

    Log directory structure:
        {log_dir}/{run_name}/
            train.jsonl     - training metrics (one JSON object per line)
            eval.jsonl      - evaluation metrics
            config.json     - model/run configuration
            scaling.jsonl   - scaling results
    """

    def __init__(
        self,
        config: LoggerConfig,
        run_name: str | None = None,
        model_config: dict | None = None,
    ):
        self._config = config
        self._log_dir: str | None = None

        if not config.enabled:
            return

        run_name = run_name or "default"
        self._log_dir = os.path.join(config.log_dir, run_name)
        os.makedirs(self._log_dir, exist_ok=True)

        if model_config:
            config_path = os.path.join(self._log_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(model_config, f, indent=2, default=str)

    def _append(self, filename: str, data: dict) -> None:
        if self._log_dir is None:
            return
        path = os.path.join(self._log_dir, filename)
        with open(path, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def log_train(self, metrics: dict, step: int) -> None:
        if self._log_dir is None:
            return
        self._append("train.jsonl", {"step": step, **metrics})

    def log_eval(
        self,
        positions: torch.Tensor,
        radius: float,
        box_size: float,
        step: int,
    ) -> None:
        if self._log_dir is None:
            return

        import numpy as np
        from data.validate import pair_correlation
        from metrics.clash_rate import clash_rate_batched

        cr = clash_rate_batched(positions, radius)
        metrics: dict = {"step": step, "eval/clash_rate": cr}

        pos_np = positions.cpu().numpy()
        r, g_r = pair_correlation(pos_np, box_size)

        # Save plots to log dir
        import matplotlib.pyplot as plt
        from viz.metrics import plot_gr, plot_min_distance_hist

        plots_dir = os.path.join(self._log_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        fig = plot_gr(r, g_r, radius)
        fig.savefig(os.path.join(plots_dir, f"gr_step{step}.png"), dpi=150)
        plt.close(fig)

        fig = plot_min_distance_hist(pos_np, radius)
        fig.savefig(os.path.join(plots_dir, f"min_dist_step{step}.png"), dpi=150)
        plt.close(fig)

        self._append("eval.jsonl", metrics)

    def log_model_config(
        self,
        architecture: str,
        param_count: int,
        flops_per_step: float | None = None,
    ) -> None:
        if self._log_dir is None:
            return
        model_info = {
            "architecture": architecture,
            "param_count": param_count,
            "flops_per_step": flops_per_step,
        }
        path = os.path.join(self._log_dir, "config.json")
        # Merge with existing config if present
        existing = {}
        if os.path.isfile(path):
            with open(path) as f:
                existing = json.load(f)
        existing.update(model_info)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    def log_compute(self, tracker: ComputeTracker, step: int) -> None:
        if self._log_dir is None:
            return
        self._append("train.jsonl", {"step": step, "compute/gpu_hours": tracker.gpu_hours})

    def log_scaling_point(
        self,
        architecture: str,
        compute_budget: float,
        best_clash_rate: float,
        best_gr_distance: float = float("inf"),
        param_count: int | None = None,
        flops_per_step: float | None = None,
    ) -> None:
        if self._log_dir is None:
            return
        self._append("scaling.jsonl", {
            "architecture": architecture,
            "compute_budget": compute_budget,
            "best_clash_rate": best_clash_rate,
            "best_gr_distance": best_gr_distance,
            "param_count": param_count,
            "flops_per_step": flops_per_step,
        })

    def finish(self) -> None:
        pass
