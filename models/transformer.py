"""Transformer velocity network — inspired by SimpleFold FoldingDiT.

Reference: SimpleFold (Apple, 2025), "SimpleFold: Folding Proteins is Simpler
than You Think". Uses DiT-style adaptive layer norm zero (adaLN-Zero)
conditioning with pairwise distance attention bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.common import AtomOrderingEmbedding, GaussianRBF, SinusoidalTimestepEmbedding


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Adaptive layer norm modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """Root mean square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.scale * x / (rms + self.eps)


def _apply_3d_rope(x: Tensor, positions: Tensor, head_dim: int) -> Tensor:
    """Apply 3D rotary position embeddings from spatial coordinates.

    Args:
        x: (B, H, N, D) query or key tensor.
        positions: (B, N, 3) spatial coordinates.
        head_dim: dimension per head.
    Returns:
        (B, H, N, D) with RoPE applied.
    """
    # Split head_dim into 3 pairs for x,y,z (remaining dims unchanged)
    n_rope_pairs = min(head_dim // 2, 3)  # up to 3 coordinate dims
    freqs = torch.exp(
        -torch.arange(n_rope_pairs, device=x.device, dtype=x.dtype) * (4.0 / n_rope_pairs)
    )  # (n_rope_pairs,)
    # positions: (B, N, 3) -> angles: (B, N, n_rope_pairs)
    angles = positions[..., :n_rope_pairs] * freqs  # (B, N, n_rope_pairs)
    cos_a = angles.cos().unsqueeze(1)  # (B, 1, N, n_rope_pairs)
    sin_a = angles.sin().unsqueeze(1)  # (B, 1, N, n_rope_pairs)
    # Apply rotation to first 2*n_rope_pairs dims of head
    x1 = x[..., :n_rope_pairs]
    x2 = x[..., n_rope_pairs:2 * n_rope_pairs]
    x_rot = torch.cat([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a, x[..., 2 * n_rope_pairs:]], dim=-1)
    return x_rot


class SelfAttention(nn.Module):
    """Multi-head self-attention with QK normalization, 3D RoPE, and attention bias."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, bias: Tensor | None = None, positions: Tensor | None = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        # Apply 3D RoPE from spatial coordinates
        if positions is not None:
            q = _apply_3d_rope(q, positions, self.head_dim)
            k = _apply_3d_rope(k, positions, self.head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if bias is not None:
            attn = attn + bias
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning.

    At initialization, the adaLN modulation is zero-initialized so each
    block acts as an identity, following the DiT design.
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0, residual_scale: float = 1.0):
        super().__init__()
        self.residual_scale = residual_scale
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLUFeedForward(hidden_dim, int(hidden_dim * mlp_ratio))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.attn.apply(_basic_init)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, c: Tensor, bias: Tensor | None = None, positions: Tensor | None = None) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        h = self.attn(modulate(self.norm1(x), shift_msa, scale_msa), bias=bias, positions=positions)
        x = x + self.residual_scale * gate_msa.unsqueeze(1) * h
        h = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + self.residual_scale * gate_mlp.unsqueeze(1) * h
        return x


class FinalLayer(nn.Module):
    """Output layer with adaLN modulation and zero-initialized projection."""

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class TransformerVelocityNetwork(nn.Module):
    """Transformer-based velocity network for flow matching.

    Uses DiT-style adaLN-Zero conditioning and pairwise distance attention
    bias with Gaussian RBF expansion, following the SimpleFold architecture.
    Supports arbitrary spatial dimensions via the spatial_dim parameter.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        num_rbf: int = 64,
        cutoff: float = 10.0,
        mlp_ratio: float = 4.0,
        atom_ordering: bool = False,
        spatial_dim: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.atom_ordering = atom_ordering
        self.spatial_dim = spatial_dim

        # Input: D-dimensional coordinates → hidden_dim
        self.input_proj = nn.Linear(spatial_dim, hidden_dim)

        # Pairwise distance bias: distances → RBF → MLP → per-head bias
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.pair_proj = nn.Sequential(
            nn.Linear(num_rbf, num_rbf, bias=False),
            nn.SiLU(),
            nn.Linear(num_rbf, num_heads, bias=False),
        )

        # Timestep → conditioning vector (sinusoidal + MLP)
        self.time_embed = SinusoidalTimestepEmbedding(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.normal_(self.time_proj[0].weight, std=0.02)
        nn.init.normal_(self.time_proj[2].weight, std=0.02)

        # Atom ordering embedding for chain tasks
        if atom_ordering:
            self.ordering_embed = AtomOrderingEmbedding(hidden_dim)

        # Transformer blocks with residual scaling for training stability
        residual_scale = num_layers ** -0.5
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, mlp_ratio, residual_scale=residual_scale) for _ in range(num_layers)]
        )

        # Output: adaLN + projection → D-dimensional velocity
        self.final_layer = FinalLayer(hidden_dim, spatial_dim)

    def _compute_pair_bias(self, positions: Tensor) -> Tensor:
        """Compute pairwise distance attention bias.

        Args:
            positions: (batch, N, D).

        Returns:
            Attention bias (batch, num_heads, N, N).
        """
        dists = torch.cdist(positions, positions)  # (batch, N, N)
        rbf_feats = self.rbf(dists)  # (batch, N, N, num_rbf)
        bias = self.pair_proj(rbf_feats)  # (batch, N, N, num_heads)
        return bias.permute(0, 3, 1, 2)  # (batch, num_heads, N, N)

    def forward(self, positions: Tensor, t: Tensor) -> Tensor:
        """Predict velocity field.

        Args:
            positions: Atom positions (batch, N, D).
            t: Timestep (batch,).

        Returns:
            Predicted velocity (batch, N, D).
        """
        pair_bias = self._compute_pair_bias(positions)
        x = self.input_proj(positions)

        # Add atom ordering embedding for chain tasks
        if self.atom_ordering:
            N = positions.shape[1]
            x = x + self.ordering_embed(N).unsqueeze(0)

        c = self.time_proj(self.time_embed(t))

        for block in self.blocks:
            x = block(x, c, bias=pair_bias, positions=positions)

        return self.final_layer(x, c)
