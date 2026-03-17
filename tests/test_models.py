"""Tests for model architectures."""

import torch
import pytest

from models.common import GaussianRBF, SinusoidalTimestepEmbedding
from models.transformer import TransformerVelocityNetwork


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
        d = torch.tensor([0.0])
        result = rbf(d)
        assert result[0, 0] > result[0, -1]


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
