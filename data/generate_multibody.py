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


# --- Langevin dynamics sampler ---


def compute_gradient(positions: torch.Tensor, preset: str, params: dict) -> torch.Tensor:
    """Compute gradient of total energy w.r.t. positions using autograd.

    Args:
        positions: (batch, N, 3) requires_grad must be set by caller or will be set here

    Returns:
        (batch, N, 3) gradient tensor (detached).
    """
    if not positions.requires_grad:
        positions.requires_grad_(True)
    E = total_energy(positions, preset, params).sum()
    grad = torch.autograd.grad(E, positions)[0]
    return grad.detach()


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
            # Compute gradient with autograd
            x_grad = x.detach().requires_grad_(True)
            grad = compute_gradient(x_grad, preset, params)

            # Overdamped Langevin update
            noise = math.sqrt(2 * temperature * current_dt) * torch.randn(
                n_chains, N, 3, device=dev, dtype=dtype, generator=gen,
            )
            with torch.no_grad():
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

        # Collect samples: center at origin
        with torch.no_grad():
            x_centered = x - x.mean(dim=-2, keepdim=True)
            energies = total_energy(x_centered, preset, params)

        all_positions[collected:collected + n_chains] = x_centered.cpu().numpy()
        all_energies[collected:collected + n_chains] = energies.cpu().numpy()
        collected += n_chains

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
