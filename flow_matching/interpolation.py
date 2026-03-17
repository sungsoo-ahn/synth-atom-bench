"""Linear interpolation for conditional flow matching."""

import torch
from torch import Tensor


def _ot_permute(noise: Tensor, x_0: Tensor) -> Tensor:
    """Mini-batch OT: permute noise to minimize transport cost to x_0.

    Uses greedy nearest-neighbor assignment (fast approximation of Hungarian).
    """
    B = noise.shape[0]
    # Flatten spatial dims for cost computation
    noise_flat = noise.reshape(B, -1)  # (B, N*3)
    x0_flat = x_0.reshape(B, -1)  # (B, N*3)
    # Pairwise cost matrix between noise and data samples
    cost = torch.cdist(noise_flat, x0_flat)  # (B, B)
    # Greedy assignment: for each data sample, find nearest unused noise
    perm = torch.zeros(B, dtype=torch.long, device=noise.device)
    used = torch.zeros(B, dtype=torch.bool, device=noise.device)
    for i in range(B):
        costs_i = cost[:, i].clone()
        costs_i[used] = float('inf')
        j = costs_i.argmin()
        perm[i] = j
        used[j] = True
    return noise[perm]


def interpolate(x_0: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Compute interpolated positions, noise, and velocity target.

    Uses mini-batch OT coupling to pair noise with nearest data samples.

    Args:
        x_0: Clean data positions, shape (batch, N, 3).
        t: Timesteps in [0, 1], shape (batch,).

    Returns:
        x_t: Interpolated positions (batch, N, 3).
        noise: Sampled noise (batch, N, 3).
        velocity_target: Target velocity x_0 - noise (batch, N, 3).
    """
    noise = torch.randn_like(x_0)
    noise = _ot_permute(noise, x_0)
    t = t[:, None, None]  # (batch, 1, 1) for broadcasting
    x_t = (1 - t) * noise + t * x_0
    velocity_target = x_0 - noise
    return x_t, noise, velocity_target
