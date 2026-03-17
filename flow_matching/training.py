"""Flow matching training loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flow_matching.interpolation import interpolate


def flow_matching_loss(model: nn.Module, x_0: Tensor) -> Tensor:
    """Compute flow matching MSE loss.

    Args:
        model: Velocity network, callable as model(x_t, t) -> (batch, N, 3).
        x_0: Clean data positions, shape (batch, N, 3).

    Returns:
        Scalar MSE loss.
    """
    batch_size = x_0.shape[0]
    t = torch.sigmoid(torch.randn(batch_size, device=x_0.device))
    x_t, noise, velocity_target = interpolate(x_0, t)
    v_pred = model(x_t, t)
    weight = (t / (1 - t + 1e-5)) ** 2
    main_loss = (weight[:, None, None] * (v_pred - velocity_target) ** 2).mean()

    # Auxiliary pairwise distance loss, upweighted late in the flow
    x_1_pred = x_t + (1 - t[:, None, None]) * v_pred
    dist_pred = torch.cdist(x_1_pred, x_1_pred)  # (B, N, N)
    dist_true = torch.cdist(x_0, x_0)  # (B, N, N)
    t_weight = 1 + 8 * torch.relu(t - 0.5)
    dist_loss = (t_weight[:, None, None] * (dist_pred - dist_true).abs()).mean()

    return main_loss + 0.3 * dist_loss
