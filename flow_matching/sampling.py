"""Euler ODE sampler for flow matching."""

import torch
from torch import Tensor
import torch.nn as nn


@torch.no_grad()
def sample(
    model: nn.Module,
    n_atoms: int,
    n_samples: int,
    n_steps: int = 100,
    device: str = "cpu",
) -> Tensor:
    """Generate samples via Euler ODE integration.

    Integrates from t=0 (noise) to t=1 (data).

    Args:
        model: Velocity network, callable as model(x, t) -> (batch, N, 3).
        n_atoms: Number of atoms per sample.
        n_samples: Number of samples to generate.
        n_steps: Number of Euler steps.
        device: Device for generation.

    Returns:
        Generated positions, shape (n_samples, n_atoms, 3).
    """
    model.eval()
    x = torch.randn(n_samples, n_atoms, 3, device=device)

    # Cosine step spacing: concentrate steps near t=1 where trajectories curve most
    import math
    t_schedule = [0.5 * (1 - math.cos(math.pi * i / n_steps)) for i in range(n_steps + 1)]

    for i in range(n_steps):
        t_i = t_schedule[i]
        t_next_i = t_schedule[i + 1]
        dt = t_next_i - t_i
        t = torch.full((n_samples,), t_i, device=device)
        v1 = model(x, t)
        x_pred = x + v1 * dt
        t_next = torch.full((n_samples,), t_next_i, device=device)
        v2 = model(x_pred, t_next)
        x = x + 0.5 * dt * (v1 + v2)

    return x


@torch.no_grad()
def sample_batched(
    model: nn.Module,
    n_atoms: int,
    n_samples: int,
    n_steps: int = 100,
    batch_size: int = 256,
    device: str = "cpu",
) -> Tensor:
    """Generate samples in chunks to avoid OOM.

    Args:
        model: Velocity network.
        n_atoms: Number of atoms per sample.
        n_samples: Total number of samples to generate.
        n_steps: Number of Euler steps.
        batch_size: Samples per chunk.
        device: Device for generation.

    Returns:
        Generated positions, shape (n_samples, n_atoms, 3).
    """
    chunks = []
    remaining = n_samples
    while remaining > 0:
        chunk_size = min(batch_size, remaining)
        chunk = sample(model, n_atoms, chunk_size, n_steps, device)
        chunks.append(chunk.cpu())
        remaining -= chunk_size
    return torch.cat(chunks, dim=0)
