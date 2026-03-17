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
        """Compute task-specific metrics on generated samples."""

    @abstractmethod
    def run_name(self, cfg: DictConfig, n_atoms: int) -> str:
        """Return the run name for logging."""

    @abstractmethod
    def describe_data(self, dataset) -> str:
        """Return a human-readable description of the dataset."""

    @abstractmethod
    def data_generator_cmd(self, n_atoms: int, n_samples: int, output_path: str) -> str:
        """Return shell command to generate data for this task."""


class MultibodyTask(Task):
    def load_dataset(self, path):
        from data.multibody_dataset import MultibodyDataset
        return MultibodyDataset(path)

    def model_kwargs(self):
        return {"atom_ordering": True}

    def compute_metrics(self, samples, dataset):
        from metrics.energy import energy_metrics_batched
        return energy_metrics_batched(samples, dataset)

    def run_name(self, cfg, n_atoms):
        preset = cfg.data.preset
        T = cfg.data.temperature
        return f"{cfg.model.arch}_N{n_atoms}_{preset}_T{T}"

    def describe_data(self, dataset):
        N = dataset.positions.shape[1]
        return (
            f"multibody, N={N}, preset={dataset.preset}, "
            f"T={dataset.temperature}, box_size={dataset.box_size:.4f}"
        )

    def data_generator_cmd(self, n_atoms: int, n_samples: int, output_path: str) -> str:
        return (
            f"uv run python data/generate_multibody.py --N {n_atoms} "
            f"--num_samples {n_samples} --output {output_path}"
        )


TASK_REGISTRY = {
    "multibody": MultibodyTask,
}


def get_task(cfg: DictConfig) -> Task:
    """Select task from config's explicit task field."""
    task_name = cfg.data.task
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_name]()


def get_task_from_data(path: str) -> Task:
    """Auto-detect task from .npz file contents (for standalone eval)."""
    return MultibodyTask()
