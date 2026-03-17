"""PyTorch dataset for multi-body potential samples."""

import numpy as np
import torch
from torch.utils.data import Dataset


class MultibodyDataset(Dataset):
    """Dataset of multi-body potential configurations loaded from .npz files.

    Loads entire dataset into memory as float32 tensors at init time.
    Positions are shifted to [0, box_size] for training code compatibility.
    """

    def __init__(self, path: str):
        data = np.load(path, allow_pickle=True)
        required = {"positions", "box_size", "temperature", "preset"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"Multibody dataset {path} is missing required fields: {missing}. "
                f"Available: {data.files}"
            )
        positions = data["positions"].astype(np.float32)
        self.box_size = float(data["box_size"])
        self.temperature = float(data["temperature"])
        self.preset = str(data["preset"])

        # Potential parameters
        self.k2 = float(data["k2"]) if "k2" in data.files else 1.0
        self.r0 = float(data["r0"]) if "r0" in data.files else 1.5
        self.k3 = float(data["k3"]) if "k3" in data.files else 1.0
        self.theta0 = float(data["theta0"]) if "theta0" in data.files else np.pi / 3
        self.k4 = float(data["k4"]) if "k4" in data.files else 0.5
        self.phi0 = float(data["phi0"]) if "phi0" in data.files else np.pi / 3

        # Reference energies for evaluation
        self.energies = None
        if "energies" in data.files:
            self.energies = torch.from_numpy(data["energies"].astype(np.float32))

        # Shift from origin-centered to [0, box_size] for training compatibility
        self.positions = torch.from_numpy(positions + self.box_size / 2)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> dict:
        return {
            "positions": self.positions[idx],
            "radius": self.r0 / 2,
            "box_size": self.box_size,
        }
