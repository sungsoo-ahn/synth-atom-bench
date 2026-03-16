"""Equivariant GNN velocity network — faithful reimplementation of PaiNN from SchNetPack.

Reference: Schütt et al., "Equivariant message passing for the prediction of
tensorial properties and molecular spectra" (2021).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from models.common import AtomOrderingEmbedding, SinusoidalTimestepEmbedding


class BesselRBF(nn.Module):
    """Bessel radial basis functions (DimeNet-style). Orthogonal basis."""

    def __init__(self, num_rbf: int = 20, cutoff: float = 10.0):
        super().__init__()
        self.cutoff = cutoff
        freqs = torch.arange(1, num_rbf + 1, dtype=torch.float32) * math.pi / cutoff
        self.register_buffer("freqs", freqs)
        self.prefactor = math.sqrt(2.0 / cutoff)

    def forward(self, distances: Tensor) -> Tensor:
        d = distances.unsqueeze(-1)
        return self.prefactor * torch.sin(self.freqs * d) / (d + 1e-8)


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope."""

    def __init__(self, cutoff: float = 10.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        """Apply cosine cutoff.

        Args:
            distances: (n_pairs,).

        Returns:
            Cutoff values in [0, 1], shape (n_pairs,).
        """
        return 0.5 * (1.0 + torch.cos(torch.pi * distances / self.cutoff)) * (distances < self.cutoff).float()


class EquivGNNInteraction(nn.Module):
    """Equivariant GNN message passing block (PaiNN-style)."""

    def __init__(self, hidden_dim: int, num_rbf: int):
        super().__init__()
        # Context net on neighbor scalar features -> 3H for (ds, dvs, dvv)
        self.context_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )
        # Filter net on RBF -> 3H, modulated by cutoff
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        rbf: Tensor,
        f_cut: Tensor,
        dir_ij: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Message passing update.

        Args:
            s: Scalar features (n_atoms, H).
            v: Vector features (n_atoms, 3, H).
            idx_i: Receiver indices (n_edges,).
            idx_j: Sender indices (n_edges,).
            rbf: Radial basis values (n_edges, n_rbf).
            f_cut: Cutoff values (n_edges,).
            dir_ij: Unit direction vectors (n_edges, 3).

        Returns:
            Updated (s, v).
        """
        H = s.shape[-1]

        # Context from neighbor scalar features
        context = self.context_net(s[idx_j])  # (n_edges, 3H)

        # Filter from radial basis, modulated by cutoff
        W = self.filter_net(rbf) * f_cut[:, None]  # (n_edges, 3H)

        # Element-wise product
        msg = context * W  # (n_edges, 3H)

        # Split into scalar, vector-scalar, vector-vector contributions
        ds, dvs, dvv = msg[:, :H], msg[:, H:2*H], msg[:, 2*H:]

        # Scatter-add messages to receivers
        s_update = torch.zeros_like(s)
        s_update.scatter_add_(0, idx_i[:, None].expand_as(ds), ds)

        # Vector updates: dvs * dir_ij + dvv * v_j
        v_update = torch.zeros_like(v)
        # dvs contribution: (n_edges, H) -> (n_edges, 3, H) via dir_ij
        v_msg_s = dvs[:, None, :] * dir_ij[:, :, None]  # (n_edges, 3, H)
        # dvv contribution: scale neighbor vectors
        v_msg_v = dvv[:, None, :] * v[idx_j]  # (n_edges, 3, H)
        v_msg = v_msg_s + v_msg_v
        v_update.scatter_add_(0, idx_i[:, None, None].expand_as(v_msg), v_msg)

        s = s + s_update
        v = v + v_update
        return s, v


class EquivGNNMixing(nn.Module):
    """Equivariant GNN intra-atomic mixing block (PaiNN-style)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )
        # Scalar-to-vector gate: controls which vector channels get updated
        self.v_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, s: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Intra-atomic refinement.

        Args:
            s: Scalar features (n_atoms, H).
            v: Vector features (n_atoms, 3, H).

        Returns:
            Updated (s, v).
        """
        H = s.shape[-1]

        # Linear transforms on vector feature channel dim
        # v: (n_atoms, 3, H) -> apply linear on last dim
        Uv = self.U(v)  # (n_atoms, 3, H)
        Vv = self.V(v)  # (n_atoms, 3, H)

        # Norm of Vv: (n_atoms, H)
        Vv_norm = torch.sqrt(torch.sum(Vv ** 2, dim=1) + 1e-8)

        # Context from concatenated [s, |Vv|]
        ctx_input = torch.cat([s, Vv_norm], dim=-1)  # (n_atoms, 2H)
        ctx = self.context_net(ctx_input)  # (n_atoms, 3H)
        a_ss, a_sv, a_vv = ctx[:, :H], ctx[:, H:2*H], ctx[:, 2*H:]

        # Dot product of Uv and Vv: sum over spatial dim -> (n_atoms, H)
        dot_uv = torch.sum(Uv * Vv, dim=1)

        # Updates
        gate = self.v_gate(s)  # (n_atoms, H)
        s = s + a_ss + a_sv * dot_uv
        v = v + gate[:, None, :] * a_vv[:, None, :] * Uv

        return s, v


class EquivGNNVelocityNetwork(nn.Module):
    """Equivariant GNN velocity network for flow matching (PaiNN architecture).

    All atoms are identical (hard spheres), so we use a single learned
    embedding instead of per-element embeddings. Timestep conditioning
    is additive on scalar features.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 5,
        num_rbf: int = 20,
        cutoff: float = 10.0,
        atom_ordering: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.atom_ordering = atom_ordering

        # Radial basis and cutoff
        self.rbf = BesselRBF(num_rbf, cutoff)
        self.cosine_cutoff = CosineCutoff(cutoff)

        # Atom embedding: single learned embedding for identical atoms
        self.atom_embedding = nn.Parameter(torch.randn(1, hidden_dim))

        # Timestep embedding (2-layer MLP with SiLU, matching transformer)
        self.time_embed = SinusoidalTimestepEmbedding(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Atom ordering embedding for chain tasks
        if atom_ordering:
            self.ordering_embed = AtomOrderingEmbedding(hidden_dim)

        # Message passing layers
        self.interactions = nn.ModuleList([
            EquivGNNInteraction(hidden_dim, num_rbf) for _ in range(num_layers)
        ])
        self.mixings = nn.ModuleList([
            EquivGNNMixing(hidden_dim) for _ in range(num_layers)
        ])

        # adaLN-Zero: per-layer timestep modulation (scale, shift, gate for interaction + mixing)
        # Each layer produces 6H values: (scale_i, shift_i, gate_i, scale_m, shift_m, gate_m)
        self.adaln_projs = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, 6 * hidden_dim),
            ) for _ in range(num_layers)
        ])
        # Zero-initialize the final linear so layers start as identity
        for proj in self.adaln_projs:
            nn.init.zeros_(proj[1].weight)
            nn.init.zeros_(proj[1].bias)

        # Position-to-vector projection: scalar gate for initial vector features
        self.pos_to_vec = nn.Linear(hidden_dim, hidden_dim)

        # Velocity readout from vector features: (n_atoms, 3, H) -> (n_atoms, 3)
        self.velocity_readout = nn.Linear(hidden_dim, 1, bias=False)

    def _build_graph(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Build all-pairs graph from batched positions.

        Args:
            positions: (batch, N, 3).

        Returns:
            idx_i, idx_j: Edge indices (n_edges,).
            rbf: Radial basis values (n_edges, n_rbf).
            f_cut: Cutoff values (n_edges,).
            dir_ij: Unit direction vectors (n_edges, 3).
        """
        batch_size, N, _ = positions.shape
        device = positions.device

        # Flatten to (batch*N, 3)
        pos_flat = positions.reshape(-1, 3)

        # Build all-pairs edges within each sample (excluding self-loops)
        # For each sample: N*(N-1) edges
        arange_N = torch.arange(N, device=device)
        # All pairs (i, j) where i != j within one sample
        src = arange_N.repeat_interleave(N - 1)  # [0,0,...,1,1,...,N-1,N-1,...]
        # For each i, all j != i
        dst_list = []
        for i in range(N):
            dst_list.append(torch.cat([arange_N[:i], arange_N[i+1:]]))
        dst = torch.cat(dst_list)  # (N*(N-1),)

        # Expand across batch with offsets
        batch_offsets = torch.arange(batch_size, device=device)[:, None] * N  # (batch, 1)
        idx_i = (src[None, :] + batch_offsets).reshape(-1)  # (batch * N*(N-1),)
        idx_j = (dst[None, :] + batch_offsets).reshape(-1)

        # Compute displacement vectors and distances
        diff = pos_flat[idx_j] - pos_flat[idx_i]  # (n_edges, 3)
        dist = torch.norm(diff, dim=-1, keepdim=False)  # (n_edges,)

        # Unit direction (with safe division)
        dir_ij = diff / (dist[:, None] + 1e-8)

        # Radial basis and cutoff
        rbf = self.rbf(dist)
        f_cut = self.cosine_cutoff(dist)

        return idx_i, idx_j, rbf, f_cut, dir_ij

    def forward(self, positions: Tensor, t: Tensor) -> Tensor:
        """Predict velocity field.

        Args:
            positions: Atom positions (batch, N, 3).
            t: Timestep (batch,).

        Returns:
            Predicted velocity (batch, N, 3).
        """
        batch_size, N, _ = positions.shape

        # Build graph
        idx_i, idx_j, rbf, f_cut, dir_ij = self._build_graph(positions)

        # Initialize scalar features: learned embedding + timestep
        s = self.atom_embedding.expand(batch_size * N, -1).clone()  # (batch*N, H)
        t_emb = self.time_proj(self.time_embed(t))  # (batch, H)
        # Repeat timestep embedding for each atom in the sample
        t_emb = t_emb[:, None, :].expand(-1, N, -1).reshape(batch_size * N, -1)
        s = s + t_emb

        # Add atom ordering embedding for chain tasks
        if self.atom_ordering:
            ord_emb = self.ordering_embed(N)  # (N, H)
            ord_emb = ord_emb.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * N, -1)
            s = s + ord_emb

        # Initialize vector features from input positions
        # positions_flat: (batch*N, 3) -> (batch*N, 3, 1) * gate -> (batch*N, 3, H)
        pos_flat = positions.reshape(batch_size * N, 3)
        gate = self.pos_to_vec(s)  # (batch*N, H) — scalar-gated projection
        v = pos_flat[:, :, None] * gate[:, None, :]  # (batch*N, 3, H)

        # Message passing with adaLN-Zero timestep modulation
        for interaction, mixing, adaln_proj in zip(self.interactions, self.mixings, self.adaln_projs):
            # Get per-layer modulation from timestep
            adaln = adaln_proj(t_emb)  # (batch*N, 6H)
            scale_i, shift_i, gate_i, scale_m, shift_m, gate_m = adaln.chunk(6, dim=-1)

            # Modulate scalar features before interaction
            s_mod = (1 + scale_i) * s + shift_i
            s_new, v_new = interaction(s_mod, v, idx_i, idx_j, rbf, f_cut, dir_ij)
            # Gate the update (residual is already inside interaction, so we gate the full output)
            s = s + gate_i * (s_new - s_mod)
            v = v_new  # vector features don't get gated (equivariance)

            # Modulate before mixing
            s_mod = (1 + scale_m) * s + shift_m
            s_new, v_new = mixing(s_mod, v)
            s = s + gate_m * (s_new - s_mod)
            v = v_new

        # Velocity readout: (batch*N, 3, H) -> (batch*N, 3, 1) -> (batch*N, 3)
        velocity = self.velocity_readout(v).squeeze(-1)

        return velocity.reshape(batch_size, N, 3)
