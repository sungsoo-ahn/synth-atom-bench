"""Tests for energy metrics."""

import numpy as np
import torch
import pytest

from metrics.energy import (
    bond_energy_batch,
    angle_energy_batch,
    dihedral_energy_batch,
    total_energy_batch,
    energy_wasserstein,
)


class TestBondEnergyBatch:
    def test_zero_at_r0(self):
        positions = torch.zeros(2, 5, 3)
        for i in range(5):
            positions[:, i, 0] = i * 1.5
        E = bond_energy_batch(positions, k2=1.0, r0=1.5)
        assert E.shape == (2,)
        assert torch.allclose(E, torch.zeros(2), atol=1e-6)

    def test_positive_for_stretched(self):
        positions = torch.zeros(1, 3, 3)
        for i in range(3):
            positions[0, i, 0] = i * 2.0
        E = bond_energy_batch(positions, k2=1.0, r0=1.5)
        assert E.item() > 0


class TestAngleEnergyBatch:
    def test_too_few_atoms(self):
        positions = torch.zeros(2, 2, 3)
        E = angle_energy_batch(positions, k3=1.0, theta0=1.0)
        assert torch.allclose(E, torch.zeros(2))


class TestDihedralEnergyBatch:
    def test_too_few_atoms(self):
        positions = torch.zeros(2, 3, 3)
        E = dihedral_energy_batch(positions, k4=0.5, phi0=1.0)
        assert torch.allclose(E, torch.zeros(2))


class TestTotalEnergyBatch:
    def test_preset_selection(self):
        positions = torch.randn(4, 10, 3)
        E2 = total_energy_batch(positions, "multibody_2")
        E23 = total_energy_batch(positions, "multibody_23")
        E234 = total_energy_batch(positions, "multibody_234")
        assert (E23 >= E2).all()
        assert (E234 >= E23).all()


class TestEnergyWasserstein:
    def test_identical(self):
        E = torch.randn(100)
        w = energy_wasserstein(E, E)
        assert w < 1e-6

    def test_different(self):
        E1 = torch.zeros(100)
        E2 = torch.ones(100) * 10
        w = energy_wasserstein(E1, E2)
        assert w > 5.0
