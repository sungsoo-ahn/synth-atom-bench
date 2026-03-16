"""Tests for model architectures."""

import torch
import pytest

from models.common import GaussianRBF, SinusoidalTimestepEmbedding
from models.equiv_gnn import EquivGNNVelocityNetwork, CosineCutoff
from models.transformer import TransformerVelocityNetwork
from models.pairformer import PairformerVelocityNetwork


class TestSinusoidalTimestepEmbedding:
    def test_output_shape(self):
        emb = SinusoidalTimestepEmbedding(64)
        t = torch.rand(8)
        result = emb(t)
        assert result.shape == (8, 64)

    def test_odd_dim(self):
        emb = SinusoidalTimestepEmbedding(65)
        t = torch.rand(4)
        result = emb(t)
        assert result.shape == (4, 65)

    def test_different_timesteps_give_different_embeddings(self):
        emb = SinusoidalTimestepEmbedding(32)
        t = torch.tensor([0.0, 0.5, 1.0])
        result = emb(t)
        # All three embeddings should be different
        assert not torch.allclose(result[0], result[1])
        assert not torch.allclose(result[1], result[2])


class TestGaussianRBF:
    def test_output_shape_1d(self):
        rbf = GaussianRBF(20, cutoff=5.0)
        d = torch.rand(100)
        result = rbf(d)
        assert result.shape == (100, 20)

    def test_output_shape_2d(self):
        rbf = GaussianRBF(20, cutoff=5.0)
        d = torch.rand(4, 10)
        result = rbf(d)
        assert result.shape == (4, 10, 20)

    def test_center_activation(self):
        rbf = GaussianRBF(20, cutoff=5.0)
        # Distance at first center (0.0) should have max activation at index 0
        d = torch.tensor([0.0])
        result = rbf(d)
        assert result[0, 0] > result[0, -1]


class TestCosineCutoff:
    def test_zero_at_cutoff(self):
        cutoff = CosineCutoff(5.0)
        d = torch.tensor([5.0])
        assert cutoff(d).item() == pytest.approx(0.0, abs=1e-6)

    def test_one_at_zero(self):
        cutoff = CosineCutoff(5.0)
        d = torch.tensor([0.0])
        assert cutoff(d).item() == pytest.approx(1.0, abs=1e-6)

    def test_beyond_cutoff(self):
        cutoff = CosineCutoff(5.0)
        d = torch.tensor([6.0])
        assert cutoff(d).item() == pytest.approx(0.0, abs=1e-6)


class TestEquivGNNVelocityNetwork:
    @pytest.fixture
    def model(self):
        return EquivGNNVelocityNetwork(hidden_dim=32, num_layers=2, num_rbf=10, cutoff=5.0)

    def test_forward_shape(self, model):
        positions = torch.randn(4, 10, 3)
        t = torch.rand(4)
        velocity = model(positions, t)
        assert velocity.shape == (4, 10, 3)

    def test_single_sample(self, model):
        positions = torch.randn(1, 5, 3)
        t = torch.rand(1)
        velocity = model(positions, t)
        assert velocity.shape == (1, 5, 3)

    def test_backward(self, model):
        positions = torch.randn(2, 5, 3, requires_grad=True)
        t = torch.rand(2)
        velocity = model(positions, t)
        loss = velocity.sum()
        loss.backward()
        assert positions.grad is not None
        assert positions.grad.shape == (2, 5, 3)

    def test_equivariance(self, model):
        """Equiv-GNN should be equivariant: rotating input should rotate output."""
        torch.manual_seed(42)
        positions = torch.randn(2, 5, 3)
        t = torch.rand(2)

        # Random rotation matrix
        q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.det(q) < 0:
            q[:, 0] *= -1  # ensure proper rotation (det=+1)

        # Forward on original
        v_orig = model(positions, t)

        # Forward on rotated input
        positions_rot = positions @ q.T  # (batch, N, 3) @ (3, 3)
        v_rot = model(positions_rot, t)

        # Rotated output should match rotation of original output
        v_orig_rot = v_orig @ q.T
        torch.testing.assert_close(v_rot, v_orig_rot, atol=1e-4, rtol=1e-4)

    def test_different_timesteps(self, model):
        """Different timesteps should produce different outputs."""
        positions = torch.randn(2, 5, 3)
        t1 = torch.tensor([0.1, 0.1])
        t2 = torch.tensor([0.9, 0.9])
        v1 = model(positions, t1)
        v2 = model(positions, t2)
        assert not torch.allclose(v1, v2, atol=1e-6)


class TestTransformerVelocityNetwork:
    @pytest.fixture
    def model(self):
        return TransformerVelocityNetwork(
            hidden_dim=32, num_layers=2, num_heads=4, num_rbf=10, cutoff=5.0
        )

    def test_forward_shape(self, model):
        positions = torch.randn(4, 10, 3)
        t = torch.rand(4)
        velocity = model(positions, t)
        assert velocity.shape == (4, 10, 3)

    def test_single_sample(self, model):
        positions = torch.randn(1, 5, 3)
        t = torch.rand(1)
        velocity = model(positions, t)
        assert velocity.shape == (1, 5, 3)

    def test_backward(self, model):
        positions = torch.randn(2, 5, 3, requires_grad=True)
        t = torch.rand(2)
        velocity = model(positions, t)
        loss = velocity.sum()
        loss.backward()
        assert positions.grad is not None
        assert positions.grad.shape == (2, 5, 3)

    def test_zero_init(self, model):
        """DiT-style zero-init means fresh model outputs near-zero."""
        positions = torch.randn(2, 5, 3)
        t = torch.rand(2)
        velocity = model(positions, t)
        assert velocity.abs().max() < 1e-5

    def test_nonzero_after_training_step(self, model):
        """After a gradient step, output should be non-zero."""
        positions = torch.randn(2, 5, 3)
        t = torch.rand(2)
        target = torch.randn(2, 5, 3)
        loss = (model(positions, t) - target).pow(2).mean()
        loss.backward()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        opt.step()
        velocity = model(positions, t)
        assert velocity.abs().max() > 1e-5


class TestPairformerVelocityNetwork:
    @pytest.fixture
    def model(self):
        return PairformerVelocityNetwork(
            hidden_dim=32, pair_dim=16, num_layers=1, num_heads=4,
            num_rbf=10, cutoff=5.0,
        )

    def test_forward_shape(self, model):
        positions = torch.randn(4, 10, 3)
        t = torch.rand(4)
        velocity = model(positions, t)
        assert velocity.shape == (4, 10, 3)

    def test_single_sample(self, model):
        positions = torch.randn(1, 5, 3)
        t = torch.rand(1)
        velocity = model(positions, t)
        assert velocity.shape == (1, 5, 3)

    def test_backward(self, model):
        positions = torch.randn(2, 5, 3, requires_grad=True)
        t = torch.rand(2)
        velocity = model(positions, t)
        loss = velocity.sum()
        loss.backward()
        assert positions.grad is not None
        assert positions.grad.shape == (2, 5, 3)

    def test_zero_init(self, model):
        """Zero-init output projection means fresh model outputs near-zero."""
        positions = torch.randn(2, 5, 3)
        t = torch.rand(2)
        velocity = model(positions, t)
        assert velocity.abs().max() < 1e-5

    def test_nonzero_after_training_step(self, model):
        """After a gradient step, output should be non-zero."""
        positions = torch.randn(2, 5, 3)
        t = torch.rand(2)
        target = torch.randn(2, 5, 3)
        loss = (model(positions, t) - target).pow(2).mean()
        loss.backward()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        opt.step()
        velocity = model(positions, t)
        assert velocity.abs().max() > 1e-5
