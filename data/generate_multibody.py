"""Langevin dynamics sampler for 3D multi-body potential configurations (Boltzmann distribution).

Uses overdamped Langevin dynamics with parallel chains on GPU for efficient sampling.
All energy functions are in PyTorch for autograd compatibility.
"""

import argparse
import math
import os
import time

import numpy as np
import torch


# --- Energy functions (PyTorch, all O(N), support batched and unbatched) ---


def bond_energy(positions: torch.Tensor, k2: float, r0: float) -> torch.Tensor:
    """Harmonic bond stretch energy over consecutive pairs. O(N).

    Args:
        positions: (N, 3) or (batch, N, 3)
        k2: bond spring constant
        r0: equilibrium bond length

    Returns:
        Scalar or (batch,) energy.
    """
    diffs = positions[..., 1:, :] - positions[..., :-1, :]  # (..., N-1, 2)
    dists = torch.linalg.norm(diffs, dim=-1)  # (..., N-1)
    return k2 * ((dists - r0) ** 2).sum(dim=-1)


def angle_energy(positions: torch.Tensor, k3: float, theta0: float) -> torch.Tensor:
    """Harmonic angle bending energy over consecutive triplets. O(N).

    Args:
        positions: (N, 3) or (batch, N, 3)
        k3: angle spring constant
        theta0: equilibrium angle (radians)

    Returns:
        Scalar or (batch,) energy.
    """
    N = positions.shape[-2]
    if N < 3:
        return torch.zeros(positions.shape[:-2], device=positions.device, dtype=positions.dtype)
    v1 = positions[..., :-2, :] - positions[..., 1:-1, :]  # (..., N-2, 2)
    v2 = positions[..., 2:, :] - positions[..., 1:-1, :]  # (..., N-2, 2)
    cos_theta = (v1 * v2).sum(dim=-1) / (
        torch.linalg.norm(v1, dim=-1) * torch.linalg.norm(v2, dim=-1) + 1e-10
    )
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    theta = torch.arccos(cos_theta)
    return k3 * ((theta - theta0) ** 2).sum(dim=-1)


def _dihedral_angles(positions: torch.Tensor) -> torch.Tensor:
    """Compute dihedral angles for consecutive quadruplets.

    Standard 3D dihedral: angle between planes (i,i+1,i+2) and (i+1,i+2,i+3).

    Args:
        positions: (N, 3) or (batch, N, 3)

    Returns:
        (..., N-3) tensor of dihedral angles.
    """
    b1 = positions[..., 1:-2, :] - positions[..., :-3, :]  # (..., N-3, 3)
    b2 = positions[..., 2:-1, :] - positions[..., 1:-2, :]  # (..., N-3, 3)
    b3 = positions[..., 3:, :] - positions[..., 2:-1, :]    # (..., N-3, 3)
    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)
    n1 = n1 / (torch.linalg.norm(n1, dim=-1, keepdim=True) + 1e-10)
    n2 = n2 / (torch.linalg.norm(n2, dim=-1, keepdim=True) + 1e-10)
    b2_hat = b2 / (torch.linalg.norm(b2, dim=-1, keepdim=True) + 1e-10)
    m1 = torch.linalg.cross(n1, b2_hat)
    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)
    return torch.atan2(y, x)


def dihedral_energy(positions: torch.Tensor, k4: float, phi0: float) -> torch.Tensor:
    """Harmonic dihedral energy over consecutive quadruplets. O(N).

    Args:
        positions: (N, 3) or (batch, N, 3)
        k4: dihedral force constant
        phi0: equilibrium dihedral angle (radians)

    Returns:
        Scalar or (batch,) energy.
    """
    N = positions.shape[-2]
    if N < 4:
        return torch.zeros(positions.shape[:-2], device=positions.device, dtype=positions.dtype)
    phi = _dihedral_angles(positions)
    return k4 * ((phi - phi0) ** 2).sum(dim=-1)


PRESET_COMPONENTS = {
    "multibody_2": ["bond"],
    "multibody_23": ["bond", "angle"],
    "multibody_234": ["bond", "angle", "dihedral"],
}


def total_energy(positions: torch.Tensor, preset: str, params: dict) -> torch.Tensor:
    """Compute total energy for a given preset.

    Args:
        positions: (N, 3) or (batch, N, 3)
        preset: one of PRESET_COMPONENTS keys
        params: dict with k2, r0, k3, theta0, k4, phi0

    Returns:
        Scalar or (batch,) energy.
    """
    E = torch.zeros(positions.shape[:-2], device=positions.device, dtype=positions.dtype)
    components = PRESET_COMPONENTS[preset]
    if "bond" in components:
        E = E + bond_energy(positions, params["k2"], params["r0"])
    if "angle" in components:
        E = E + angle_energy(positions, params["k3"], params["theta0"])
    if "dihedral" in components:
        E = E + dihedral_energy(positions, params["k4"], params["phi0"])
    return E


# --- Analytic gradient functions ---


def bond_gradient(positions: torch.Tensor, k2: float, r0: float) -> torch.Tensor:
    """Analytic gradient of bond stretch energy. O(N).

    Args:
        positions: (..., N, 3)
    Returns:
        (..., N, 3) gradient tensor.
    """
    d = positions[..., 1:, :] - positions[..., :-1, :]  # (..., N-1, 3)
    r = torch.linalg.norm(d, dim=-1, keepdim=True)  # (..., N-1, 1)
    factor = 2 * k2 * (r - r0) / (r + 1e-10)  # (..., N-1, 1)
    bond_force = factor * d
    grad = torch.zeros_like(positions)
    grad[..., 1:, :] += bond_force
    grad[..., :-1, :] -= bond_force
    return grad


def angle_gradient(positions: torch.Tensor, k3: float, theta0: float) -> torch.Tensor:
    """Analytic gradient of angle bending energy. O(N).

    Args:
        positions: (..., N, 3)
    Returns:
        (..., N, 3) gradient tensor.
    """
    N = positions.shape[-2]
    if N < 3:
        return torch.zeros_like(positions)
    # Vectors from central atom j to neighbors i (=j-1) and k (=j+1)
    u = positions[..., :-2, :] - positions[..., 1:-1, :]  # r_{i} - r_{j}
    v = positions[..., 2:, :] - positions[..., 1:-1, :]   # r_{k} - r_{j}
    u_norm = torch.linalg.norm(u, dim=-1, keepdim=True) + 1e-10
    v_norm = torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-10
    cos_theta = (u * v).sum(dim=-1, keepdim=True) / (u_norm * v_norm)
    cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.arccos(cos_theta)
    sin_theta = torch.sin(theta).clamp(min=1e-10)
    # dE/dtheta * dtheta/d(cos_theta) = 2*k3*(theta-theta0) * (-1/sin_theta)
    prefactor = 2 * k3 * (theta - theta0) / (-sin_theta)
    # d(cos_theta)/dr_i and d(cos_theta)/dr_k
    grad_i = prefactor * (v / (u_norm * v_norm) - cos_theta * u / u_norm**2)
    grad_k = prefactor * (u / (u_norm * v_norm) - cos_theta * v / v_norm**2)
    grad_j = -(grad_i + grad_k)
    grad = torch.zeros_like(positions)
    grad[..., :-2, :] += grad_i
    grad[..., 1:-1, :] += grad_j
    grad[..., 2:, :] += grad_k
    return grad


def dihedral_gradient(positions: torch.Tensor, k4: float, phi0: float) -> torch.Tensor:
    """Analytic gradient of dihedral energy. O(N).

    Uses atan2(Y, X) formulation with explicit chain rule:
      X = m·n, Y = |b2|*(b1·n), where m = b1×b2, n = b2×b3.

    Args:
        positions: (..., N, 3)
    Returns:
        (..., N, 3) gradient tensor.
    """
    N = positions.shape[-2]
    if N < 4:
        return torch.zeros_like(positions)
    eps = 1e-10
    b1 = positions[..., 1:-2, :] - positions[..., :-3, :]   # r_j - r_i
    b2 = positions[..., 2:-1, :] - positions[..., 1:-2, :]   # r_k - r_j
    b3 = positions[..., 3:, :] - positions[..., 2:-1, :]     # r_l - r_k
    m = torch.linalg.cross(b1, b2)  # (..., N-3, 3)
    n = torch.linalg.cross(b2, b3)  # (..., N-3, 3)
    b2_norm = torch.linalg.norm(b2, dim=-1, keepdim=True) + eps
    b2_hat = b2 / b2_norm
    # atan2 quantities: phi = atan2(Y, X) matching _dihedral_angles convention
    X = (m * n).sum(dim=-1, keepdim=True)              # m·n
    b1_dot_n = (b1 * n).sum(dim=-1, keepdim=True)      # b1·n
    Y = -b2_norm * b1_dot_n                             # -|b2|*(b1·n)
    phi = torch.atan2(Y, X)
    dE_dphi = 2 * k4 * (phi - phi0)
    # dphi/dX = -Y/R², dphi/dY = X/R²
    R2 = X ** 2 + Y ** 2 + eps
    dphi_dX = -Y / R2
    dphi_dY = X / R2
    # Cross products needed for position derivatives
    b2_cross_n = torch.linalg.cross(b2, n)
    n_cross_b1 = torch.linalg.cross(n, b1)
    b3_cross_m = torch.linalg.cross(b3, m)
    m_cross_b2 = torch.linalg.cross(m, b2)
    b3_cross_b1 = torch.linalg.cross(b3, b1)
    # Y = -|b2|*(b1·n), so dY/dr = -d(|b2|*(b1·n))/dr
    # Atom i (only b1 depends on r_i via -I):
    dX_i = -b2_cross_n
    dY_i = b2_norm * n  # -(−|b2|*n)
    # Atom l (only b3 depends on r_l via +I):
    dX_l = m_cross_b2
    dY_l = -b2_norm * m  # -(|b2|*m)
    # Atom j (b1 via +I, b2 via -I):
    dX_j = b2_cross_n - n_cross_b1 - b3_cross_m
    dY_j = -(b2_norm * n - b2_hat * b1_dot_n - b2_norm * b3_cross_b1)
    # Atom k (b2 via +I, b3 via -I):
    dX_k = n_cross_b1 + b3_cross_m - m_cross_b2
    dY_k = -(b2_hat * b1_dot_n + b2_norm * b3_cross_b1 - b2_norm * m)
    # Chain rule: dE/dr = dE/dphi * (dphi/dX * dX/dr + dphi/dY * dY/dr)
    grad = torch.zeros_like(positions)
    grad[..., :-3, :] += dE_dphi * (dphi_dX * dX_i + dphi_dY * dY_i)
    grad[..., 1:-2, :] += dE_dphi * (dphi_dX * dX_j + dphi_dY * dY_j)
    grad[..., 2:-1, :] += dE_dphi * (dphi_dX * dX_k + dphi_dY * dY_k)
    grad[..., 3:, :] += dE_dphi * (dphi_dX * dX_l + dphi_dY * dY_l)
    return grad


# --- Langevin dynamics sampler ---


def compute_gradient(positions: torch.Tensor, preset: str, params: dict) -> torch.Tensor:
    """Compute gradient of total energy w.r.t. positions using analytic formulas.

    Args:
        positions: (batch, N, 3)

    Returns:
        (batch, N, 3) gradient tensor.
    """
    grad = torch.zeros_like(positions)
    components = PRESET_COMPONENTS[preset]
    if "bond" in components:
        grad = grad + bond_gradient(positions, params["k2"], params["r0"])
    if "angle" in components:
        grad = grad + angle_gradient(positions, params["k3"], params["theta0"])
    if "dihedral" in components:
        grad = grad + dihedral_gradient(positions, params["k4"], params["phi0"])
    return grad


def initialize_chains(
    num_chains: int, N: int, r0: float, device: torch.device, dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Place atoms along a line with spacing r0, plus small perturbation.

    Returns:
        (num_chains, N, 3) tensor.
    """
    positions = torch.zeros(num_chains, N, 3, device=device, dtype=dtype)
    for i in range(N):
        positions[:, i, 0] = i * r0
    # Small random perturbation
    positions = positions + 0.05 * r0 * torch.randn(
        num_chains, N, 3, device=device, dtype=dtype, generator=generator,
    )
    return positions


def langevin_sample(
    N: int,
    temperature: float,
    preset: str,
    params: dict,
    num_samples: int,
    equilibration_steps: int | None = None,
    dt: float = 0.001,
    batch_chains: int = 1000,
    seed: int = 42,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run overdamped Langevin dynamics with parallel chains on GPU.

    Each chain produces one independent sample after equilibration.

    Returns:
        (positions (num_samples, N, 3), energies (num_samples,)) as numpy arrays.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    dtype = torch.float32

    if equilibration_steps is None:
        equilibration_steps = max(10_000, 500 * N)

    r0 = params["r0"]
    noise_scale = math.sqrt(2 * temperature * dt)

    all_positions = np.empty((num_samples, N, 3), dtype=np.float32)
    all_energies = np.empty(num_samples, dtype=np.float32)
    collected = 0

    t0 = time.time()

    while collected < num_samples:
        # How many chains to run this batch
        n_chains = min(batch_chains, num_samples - collected)

        # Each batch gets a deterministic sub-seed
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed + collected)

        x = initialize_chains(n_chains, N, r0, dev, dtype, gen)
        current_dt = dt

        # Equilibration
        for step in range(equilibration_steps):
            grad = compute_gradient(x, preset, params)

            # Overdamped Langevin update
            noise = math.sqrt(2 * temperature * current_dt) * torch.randn(
                n_chains, N, 3, device=dev, dtype=dtype, generator=gen,
            )
            x = x - grad * current_dt + noise

            # Adaptive: if positions diverge, reduce dt
            max_extent = x.abs().max().item()
            if max_extent > 100 * r0 * N:
                current_dt *= 0.5
                # Re-initialize diverged chains
                diverged = x.abs().amax(dim=(-2, -1)) > 100 * r0 * N
                if diverged.any():
                    n_diverged = diverged.sum().item()
                    x[diverged] = initialize_chains(
                        n_diverged, N, r0, dev, dtype, gen,
                    )

        # Collect samples: center at origin, filter NaN/Inf
        with torch.no_grad():
            x_centered = x - x.mean(dim=-2, keepdim=True)
            energies = total_energy(x_centered, preset, params)

            # Filter out diverged chains (NaN or very large positions)
            valid = torch.isfinite(x_centered).all(dim=(-2, -1)) & torch.isfinite(energies)
            x_valid = x_centered[valid].cpu().numpy()
            e_valid = energies[valid].cpu().numpy()

        n_valid = len(x_valid)
        n_to_store = min(n_valid, num_samples - collected)
        all_positions[collected:collected + n_to_store] = x_valid[:n_to_store]
        all_energies[collected:collected + n_to_store] = e_valid[:n_to_store]
        collected += n_to_store

        if n_valid < n_chains:
            n_bad = n_chains - n_valid
            print(f" ({n_bad} diverged)", end="", flush=True)

        elapsed = time.time() - t0
        rate = collected / elapsed if elapsed > 0 else 0
        print(
            f"\r  {collected}/{num_samples} samples | "
            f"dt={current_dt:.6f} | {rate:.0f} samples/s",
            end="", flush=True,
        )

    print()
    elapsed = time.time() - t0
    print(f"  Done: {num_samples} samples in {elapsed:.1f}s")
    return all_positions, all_energies


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D multi-body potential samples via Langevin dynamics"
    )
    parser.add_argument("--N", type=int, required=True, help="Number of atoms")
    parser.add_argument("--T", type=float, default=1.0, help="Temperature (default: 1.0)")
    parser.add_argument("--preset", type=str, required=True,
                        choices=list(PRESET_COMPONENTS.keys()),
                        help="Potential preset")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--burn_in", type=int, default=None,
                        help="Equilibration steps per chain (default: max(10000, 500*N))")
    parser.add_argument("--dt", type=float, default=0.001, help="Langevin step size")
    parser.add_argument("--batch_chains", type=int, default=1000,
                        help="Number of parallel chains per batch")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    parser.add_argument("--k2", type=float, default=1.0, help="Bond spring constant")
    parser.add_argument("--r0", type=float, default=1.5, help="Equilibrium bond length")
    parser.add_argument("--k3", type=float, default=1.0, help="Angle spring constant")
    parser.add_argument("--theta0", type=float, default=np.pi / 3, help="Equilibrium angle (rad)")
    parser.add_argument("--k4", type=float, default=0.5, help="Dihedral force constant")
    parser.add_argument("--phi0", type=float, default=np.pi / 3, help="Equilibrium dihedral (rad)")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path")
    args = parser.parse_args()

    params = {
        "k2": args.k2, "r0": args.r0,
        "k3": args.k3, "theta0": args.theta0,
        "k4": args.k4, "phi0": args.phi0,
    }

    equilibration_steps = args.burn_in or max(10_000, 500 * args.N)

    print(f"Generating {args.num_samples} samples: N={args.N}, T={args.T}, preset={args.preset}")
    print(f"  Params: {params}")
    print(f"  Langevin: dt={args.dt}, equilibration={equilibration_steps}, "
          f"batch_chains={args.batch_chains}")

    samples, energies = langevin_sample(
        N=args.N,
        temperature=args.T,
        preset=args.preset,
        params=params,
        num_samples=args.num_samples,
        equilibration_steps=args.burn_in,
        dt=args.dt,
        batch_chains=args.batch_chains,
        seed=args.seed,
        device=args.device,
    )

    # Compute box_size: 2 * max_extent * 1.2
    max_extent = np.abs(samples).max()
    box_size = 2 * max_extent * 1.2

    print(f"  Box size: {box_size:.4f}")
    print(f"  Energy: mean={energies.mean():.3f}, std={energies.std():.3f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        positions=samples,
        box_size=np.float32(box_size),
        N=args.N,
        temperature=np.float32(args.T),
        preset=args.preset,
        k2=np.float32(args.k2),
        r0=np.float32(args.r0),
        k3=np.float32(args.k3),
        theta0=np.float32(args.theta0),
        k4=np.float32(args.k4),
        phi0=np.float32(args.phi0),
        seed=args.seed,
        burn_in=equilibration_steps,
        energies=energies,
    )
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
