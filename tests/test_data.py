"""Tests for multibody data generation and dataset loading."""

import numpy as np
import pytest
import torch

from data.generate_multibody import (
    bond_energy,
    angle_energy,
    dihedral_energy,
    total_energy,
    langevin_sample,
    PRESET_COMPONENTS,
)
from data.multibody_dataset import MultibodyDataset


class TestEnergyFunctions:
    def test_bond_energy_zero_at_r0(self):
        """Perfectly spaced chain should have zero bond energy."""
        pos = torch.zeros(5, 3)
        for i in range(5):
            pos[i, 0] = i * 1.5
        assert bond_energy(pos, k2=1.0, r0=1.5).item() == pytest.approx(0.0, abs=1e-6)

    def test_bond_energy_positive(self):
        """Stretched bonds should have positive energy."""
        pos = torch.zeros(3, 3)
        for i in range(3):
            pos[i, 0] = i * 2.0  # spacing 2.0, r0=1.5
        assert bond_energy(pos, k2=1.0, r0=1.5).item() > 0

    def test_angle_energy_too_few_atoms(self):
        pos = torch.zeros(2, 3)
        assert angle_energy(pos, k3=1.0, theta0=1.047).item() == 0.0

    def test_dihedral_energy_too_few_atoms(self):
        pos = torch.zeros(3, 3)
        assert dihedral_energy(pos, k4=0.5, phi0=1.047).item() == 0.0

    def test_batched(self):
        """Batched energy should match unbatched."""
        pos = torch.randn(5, 3)
        pos_batch = pos.unsqueeze(0)
        params = {"k2": 1.0, "r0": 1.5, "k3": 1.0, "theta0": 1.047, "k4": 0.5, "phi0": 1.047}
        e_single = total_energy(pos, "multibody_234", params)
        e_batch = total_energy(pos_batch, "multibody_234", params)
        assert e_single.item() == pytest.approx(e_batch.item(), rel=1e-5)

    def test_gradient_finite(self):
        """Gradients should be finite."""
        from data.generate_multibody import compute_gradient
        pos = torch.randn(4, 8, 3)
        params = {"k2": 1.0, "r0": 1.5, "k3": 1.0, "theta0": 1.047, "k4": 0.5, "phi0": 1.047}
        grad = compute_gradient(pos, "multibody_234", params)
        assert grad.shape == (4, 8, 3)
        assert torch.isfinite(grad).all()


class TestLangevinSampler:
    def test_shape(self):
        params = {"k2": 1.0, "r0": 1.5, "k3": 1.0, "theta0": 1.047, "k4": 0.5, "phi0": 1.047}
        samples, energies = langevin_sample(
            N=5, temperature=1.0, preset="multibody_2",
            params=params, num_samples=10, equilibration_steps=100,
            batch_chains=10, seed=42, device="cpu",
        )
        assert samples.shape == (10, 5, 3)
        assert energies.shape == (10,)

    def test_centered(self):
        params = {"k2": 1.0, "r0": 1.5, "k3": 1.0, "theta0": 1.047, "k4": 0.5, "phi0": 1.047}
        samples, _ = langevin_sample(
            N=5, temperature=1.0, preset="multibody_2",
            params=params, num_samples=10, equilibration_steps=100,
            batch_chains=10, seed=42, device="cpu",
        )
        for s in range(len(samples)):
            np.testing.assert_allclose(samples[s].mean(axis=0), 0.0, atol=0.1)


class TestMultibodyDataset:
    def test_roundtrip(self, tmp_path):
        params = {"k2": 1.0, "r0": 1.5, "k3": 1.0, "theta0": 1.047, "k4": 0.5, "phi0": 1.047}
        samples, energies = langevin_sample(
            N=5, temperature=1.0, preset="multibody_23",
            params=params, num_samples=20, equilibration_steps=100,
            batch_chains=20, seed=42, device="cpu",
        )
        max_ext = np.max(np.abs(samples))
        box_size = 2.0 * max_ext * 1.2
        path = str(tmp_path / "test.npz")
        np.savez(
            path,
            positions=samples,
            box_size=np.float32(box_size),
            temperature=np.float32(1.0),
            preset="multibody_23",
            k2=np.float32(1.0), r0=np.float32(1.5),
            k3=np.float32(1.0), theta0=np.float32(1.047),
            k4=np.float32(0.5), phi0=np.float32(1.047),
            energies=energies,
            N=5, seed=42, burn_in=100,
        )

        ds = MultibodyDataset(path)
        assert len(ds) == 20
        item = ds[0]
        assert item["positions"].shape == (5, 3)
        assert item["positions"].dtype == torch.float32
        assert ds.preset == "multibody_23"
        assert ds.temperature == 1.0
