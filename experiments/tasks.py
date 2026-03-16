"""Task abstraction: dataset, metrics, model kwargs, run naming."""

from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig


class Task(ABC):
    """Base class for benchmark tasks."""

    @abstractmethod
    def load_dataset(self, path: str):
        """Return a PyTorch Dataset from a .npz path."""

    @abstractmethod
    def model_kwargs(self) -> dict:
        """Extra kwargs to inject into the model constructor."""

    @abstractmethod
    def compute_metrics(self, samples: torch.Tensor, dataset) -> dict[str, float]:
        """Compute task-specific metrics on generated samples.

        Returns dict of metric_name -> value beyond the base metrics
        (clash_rate and gr_distance are always computed by the caller).
        """

    @abstractmethod
    def run_name(self, cfg: DictConfig, n_atoms: int) -> str:
        """Return the run name for logging."""

    @abstractmethod
    def describe_data(self, dataset) -> str:
        """Return a human-readable description of the dataset."""


class HardSphereTask(Task):
    def __init__(self, eta: float | None = None):
        self.eta = eta

    def load_dataset(self, path):
        from data.dataset import HardSphereDataset
        return HardSphereDataset(path)

    def model_kwargs(self):
        return {}

    def compute_metrics(self, samples, dataset):
        return {}

    def run_name(self, cfg, n_atoms):
        return f"{cfg.model.arch}_N{n_atoms}_eta{cfg.data.eta}"

    def describe_data(self, dataset):
        N = dataset.positions.shape[1]
        return f"hard-sphere, N={N}, radius={dataset.radius}, box_size={dataset.box_size:.4f}"


class ChainTask(Task):
    def load_dataset(self, path):
        from data.chain_dataset import ChainDataset
        return ChainDataset(path)

    def model_kwargs(self):
        return {"atom_ordering": True}

    def compute_metrics(self, samples, dataset):
        from metrics.bond_violation import (
            bond_violation_rate_batched,
            nonbonded_clash_rate_batched,
        )
        return {
            "bond_violation_rate": bond_violation_rate_batched(
                samples, dataset.bond_length
            ),
            "nonbonded_clash_rate": nonbonded_clash_rate_batched(
                samples, dataset.radius
            ),
        }

    def run_name(self, cfg, n_atoms):
        return f"{cfg.model.arch}_N{n_atoms}_chain"

    def describe_data(self, dataset):
        N = dataset.positions.shape[1]
        return f"chain, N={N}, bond_length={dataset.bond_length}, radius={dataset.radius}, box_size={dataset.box_size:.4f}"


TASK_REGISTRY = {
    "hard_sphere": HardSphereTask,
    "chain": ChainTask,
}


def get_task(cfg: DictConfig) -> Task:
    """Select task from config's explicit task field."""
    task_name = cfg.data.task
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_name]()


def get_task_from_data(path: str) -> Task:
    """Auto-detect task from .npz file contents (for standalone eval)."""
    import numpy as np
    data = np.load(path)
    has_bond = "bond_length" in data.files
    data.close()
    return ChainTask() if has_bond else HardSphereTask()
