"""Shared components for velocity networks."""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional encoding for scalar timesteps."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: Tensor) -> Tensor:
        """Embed timesteps.

        Args:
            t: Timesteps, shape (batch,).

        Returns:
            Embeddings, shape (batch, embed_dim).
        """
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        # If embed_dim is odd, pad with a zero
        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class AtomOrderingEmbedding(nn.Module):
    """Sinusoidal positional encoding for atom ordering in chains.

    Generates a fixed sinusoidal embedding based on atom index (0, 1, ..., N-1),
    projected to hidden_dim via a linear layer.
    """

    def __init__(self, hidden_dim: int, max_atoms: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Precompute sinusoidal embeddings for positions 0..max_atoms-1
        half_dim = hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        positions = torch.arange(max_atoms, dtype=torch.float32)
        args = positions[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if hidden_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros(max_atoms, 1)], dim=-1)
        self.register_buffer("embedding", embedding)  # (max_atoms, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, n_atoms: int) -> Tensor:
        """Return ordering embedding for n_atoms.

        Returns:
            Embedding, shape (n_atoms, hidden_dim).
        """
        return self.proj(self.embedding[:n_atoms])
