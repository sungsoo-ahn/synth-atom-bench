"""GPU-accelerated energy computation for 3D multi-body potentials."""

import torch
from scipy.stats import wasserstein_distance


def bond_energy_batch(positions: torch.Tensor, k2: float, r0: float) -> torch.Tensor:
    """Harmonic bond stretch energy for consecutive pairs. (batch,)."""
    diffs = positions[:, 1:] - positions[:, :-1]  # (B, N-1, 3)
    dists = torch.linalg.norm(diffs, dim=-1)  # (B, N-1)
    return k2 * ((dists - r0) ** 2).sum(dim=1)


def angle_energy_batch(positions: torch.Tensor, k3: float, theta0: float) -> torch.Tensor:
    """Harmonic angle bending energy for consecutive triplets. (batch,)."""
    if positions.shape[1] < 3:
        return torch.zeros(positions.shape[0], device=positions.device)
    v1 = positions[:, :-2] - positions[:, 1:-1]  # (B, N-2, 3)
    v2 = positions[:, 2:] - positions[:, 1:-1]  # (B, N-2, 3)
    cos_theta = (v1 * v2).sum(dim=-1) / (
        torch.linalg.norm(v1, dim=-1) * torch.linalg.norm(v2, dim=-1) + 1e-10
    )
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    return k3 * ((theta - theta0) ** 2).sum(dim=1)


def dihedral_energy_batch(positions: torch.Tensor, k4: float, phi0: float) -> torch.Tensor:
    """Harmonic dihedral energy for consecutive quadruplets. (batch,).

    Standard 3D dihedral: angle between planes (i,i+1,i+2) and (i+1,i+2,i+3).
    """
    if positions.shape[1] < 4:
        return torch.zeros(positions.shape[0], device=positions.device)
    b1 = positions[:, 1:-2] - positions[:, :-3]  # (B, N-3, 3)
    b2 = positions[:, 2:-1] - positions[:, 1:-2]  # (B, N-3, 3)
    b3 = positions[:, 3:] - positions[:, 2:-1]    # (B, N-3, 3)
    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)
    n1 = n1 / (torch.linalg.norm(n1, dim=-1, keepdim=True) + 1e-10)
    n2 = n2 / (torch.linalg.norm(n2, dim=-1, keepdim=True) + 1e-10)
    b2_hat = b2 / (torch.linalg.norm(b2, dim=-1, keepdim=True) + 1e-10)
    m1 = torch.linalg.cross(n1, b2_hat)
    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)
    phi = torch.atan2(y, x)
    return k4 * ((phi - phi0) ** 2).sum(dim=1)


PRESET_COMPONENTS = {
    "multibody_2": ["bond"],
    "multibody_23": ["bond", "angle"],
    "multibody_234": ["bond", "angle", "dihedral"],
}


def total_energy_batch(
    positions: torch.Tensor, preset: str,
    k2: float = 1.0, r0: float = 1.5,
    k3: float = 1.0, theta0: float = 1.0472,
    k4: float = 0.5, phi0: float = 1.0472,
) -> torch.Tensor:
    """Compute total energy for a batch. Returns (batch,)."""
    components = PRESET_COMPONENTS[preset]
    E = torch.zeros(positions.shape[0], device=positions.device)
    if "bond" in components:
        E = E + bond_energy_batch(positions, k2, r0)
    if "angle" in components:
        E = E + angle_energy_batch(positions, k3, theta0)
    if "dihedral" in components:
        E = E + dihedral_energy_batch(positions, k4, phi0)
    return E


def energy_wasserstein(gen_energies: torch.Tensor, ref_energies: torch.Tensor) -> float:
    """Wasserstein-1 distance between generated and reference energy distributions."""
    return float(wasserstein_distance(
        gen_energies.cpu().numpy(),
        ref_energies.cpu().numpy(),
    ))


def energy_metrics_batched(
    samples: torch.Tensor, dataset, chunk_size: int = 500,
) -> dict[str, float]:
    """Compute energy metrics between generated samples and reference dataset.

    Returns:
        energy_wasserstein: W1 distance between energy distributions
        mean_energy_ratio: mean(E_gen) / mean(E_ref)
    """
    preset = dataset.preset
    k2, r0 = dataset.k2, dataset.r0
    k3, theta0 = dataset.k3, dataset.theta0
    k4, phi0 = dataset.k4, dataset.phi0

    # Compute generated energies in chunks
    gen_energies = []
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        E = total_energy_batch(chunk, preset, k2, r0, k3, theta0, k4, phi0)
        gen_energies.append(E)
    gen_energies = torch.cat(gen_energies)

    # Reference energies
    if dataset.energies is not None:
        ref_energies = dataset.energies.to(samples.device)
    else:
        # Compute from dataset positions (shifted back to origin-centered)
        ref_positions = dataset.positions - dataset.box_size / 2
        ref_energies = []
        for i in range(0, len(ref_positions), chunk_size):
            chunk = ref_positions[i:i + chunk_size].to(samples.device)
            E = total_energy_batch(chunk, preset, k2, r0, k3, theta0, k4, phi0)
            ref_energies.append(E)
        ref_energies = torch.cat(ref_energies)

    w1 = energy_wasserstein(gen_energies, ref_energies)
    mean_ratio = gen_energies.mean().item() / (ref_energies.mean().item() + 1e-10)

    return {
        "energy_wasserstein": w1,
        "mean_energy_ratio": mean_ratio,
    }
