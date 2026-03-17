"""Validation script for multi-body potential MCMC samples.

Checks:
1. Bond length distribution — should peak at r0
2. Angle distribution — should peak at theta0 (if preset includes angles)
3. Dihedral distribution — should peak at phi0 (if preset includes dihedrals)
4. Energy distribution — should follow Boltzmann P(E) ∝ exp(-E/T) × g(E)
5. Autocorrelation of energy — checks mixing quality
"""

import argparse
from pathlib import Path

import numpy as np


def compute_bond_lengths(positions: np.ndarray) -> np.ndarray:
    """Consecutive bond lengths. Returns (num_samples * (N-1),)."""
    diffs = positions[:, 1:] - positions[:, :-1]  # (S, N-1, 3)
    return np.linalg.norm(diffs, axis=2).ravel()


def compute_angles(positions: np.ndarray) -> np.ndarray:
    """Consecutive bond angles in radians. Returns (num_samples * (N-2),)."""
    v1 = positions[:, :-2] - positions[:, 1:-1]  # (S, N-2, 3)
    v2 = positions[:, 2:] - positions[:, 1:-1]
    cos_theta = np.sum(v1 * v2, axis=2) / (
        np.linalg.norm(v1, axis=2) * np.linalg.norm(v2, axis=2) + 1e-10
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta).ravel()


def compute_dihedrals(positions: np.ndarray) -> np.ndarray:
    """Consecutive dihedral angles in radians. Returns (num_samples * (N-3),)."""
    b1 = positions[:, 1:-2] - positions[:, :-3]
    b2 = positions[:, 2:-1] - positions[:, 1:-2]
    b3 = positions[:, 3:] - positions[:, 2:-1]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 = n1 / (np.linalg.norm(n1, axis=2, keepdims=True) + 1e-10)
    n2 = n2 / (np.linalg.norm(n2, axis=2, keepdims=True) + 1e-10)
    b2_hat = b2 / (np.linalg.norm(b2, axis=2, keepdims=True) + 1e-10)
    m1 = np.cross(n1, b2_hat)
    x = np.sum(n1 * n2, axis=2)
    y = np.sum(m1 * n2, axis=2)
    return np.arctan2(y, x).ravel()


def boltzmann_theoretical_bond(r, k2, r0, T):
    """Theoretical bond length distribution: P(r) ∝ r² exp(-k2(r-r0)²/T)."""
    p = r**2 * np.exp(-k2 * (r - r0) ** 2 / T)
    return p / (np.trapezoid(p, r) + 1e-30)


def boltzmann_theoretical_angle(theta, k3, theta0, T):
    """Theoretical angle distribution: P(θ) ∝ sin(θ) exp(-k3(θ-θ0)²/T)."""
    p = np.sin(theta) * np.exp(-k3 * (theta - theta0) ** 2 / T)
    return p / (np.trapezoid(p, theta) + 1e-30)


def energy_autocorrelation(energies: np.ndarray, max_lag: int = 200) -> np.ndarray:
    """Normalized energy autocorrelation function."""
    e = energies - energies.mean()
    var = np.var(e)
    if var < 1e-12:
        return np.zeros(max_lag)
    acf = np.correlate(e, e, mode="full")
    acf = acf[len(e) - 1 :]  # positive lags only
    acf = acf[:max_lag] / (var * len(e))
    return acf


def main():
    parser = argparse.ArgumentParser(description="Validate multi-body potential samples")
    parser.add_argument("--input", type=str, required=True, help="Input .npz file")
    parser.add_argument("--output_dir", type=str, default="outputs/plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    positions = data["positions"]
    energies = data["energies"]
    T = float(data["temperature"])
    preset = str(data["preset"])
    N = positions.shape[1]
    num_samples = positions.shape[0]
    k2, r0 = float(data["k2"]), float(data["r0"])
    k3, theta0 = float(data["k3"]), float(data["theta0"])
    k4, phi0 = float(data["k4"]), float(data["phi0"])

    from data.generate_multibody import PRESET_COMPONENTS
    components = PRESET_COMPONENTS[preset]

    print(f"Loaded {num_samples} samples: N={N}, T={T}, preset={preset}")
    print(f"  Components: {components}")
    print(f"  Energy: mean={energies.mean():.3f}, std={energies.std():.3f}, "
          f"min={energies.min():.3f}, max={energies.max():.3f}")

    # Compute distributions
    bond_lengths = compute_bond_lengths(positions)
    print(f"  Bond lengths: mean={bond_lengths.mean():.4f} (r0={r0}), "
          f"std={bond_lengths.std():.4f}")

    if "angle" in components:
        angles = compute_angles(positions)
        print(f"  Angles: mean={np.degrees(angles.mean()):.1f}° "
              f"(θ0={np.degrees(theta0):.1f}°), std={np.degrees(angles.std()):.1f}°")

    if "dihedral" in components:
        dihedrals = compute_dihedrals(positions)
        print(f"  Dihedrals: circular mean not shown, spread={np.degrees(np.std(dihedrals)):.1f}°")

    # Autocorrelation
    acf = energy_autocorrelation(energies)
    # Find first zero crossing as rough decorrelation length
    zero_cross = np.where(acf < 0)[0]
    decorr = zero_cross[0] if len(zero_cross) > 0 else len(acf)
    print(f"  Energy autocorrelation: decorrelation ≈ {decorr} samples")

    # --- Plots ---
    import matplotlib.pyplot as plt
    from viz.style import save_figure, synthbench_style

    out = Path(args.output_dir)
    tag = f"{preset}_N{N}_T{T}"

    with synthbench_style():
        n_panels = 2 + ("angle" in components) + ("dihedral" in components)
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.5))
        ax_idx = 0

        # 1) Bond length distribution
        ax = axes[ax_idx]; ax_idx += 1
        ax.hist(bond_lengths, bins=80, density=True, alpha=0.7, color="#4C72B0",
                label="MCMC samples")
        r_range = np.linspace(max(0.01, r0 - 3), r0 + 3, 500)
        p_theory = boltzmann_theoretical_bond(r_range, k2, r0, T)
        ax.plot(r_range, p_theory, "r-", lw=2, label="Boltzmann theory")
        ax.axvline(r0, color="gray", ls="--", alpha=0.5, label=f"r₀={r0}")
        ax.set_xlabel("Bond length")
        ax.set_ylabel("Density")
        ax.set_title("Bond lengths")
        ax.legend(fontsize=8)

        # 2) Angle distribution (if applicable)
        if "angle" in components:
            ax = axes[ax_idx]; ax_idx += 1
            ax.hist(np.degrees(angles), bins=80, density=True, alpha=0.7, color="#8172B3",
                    label="MCMC samples")
            theta_range = np.linspace(0.01, np.pi - 0.01, 500)
            p_theory = boltzmann_theoretical_angle(theta_range, k3, theta0, T)
            ax.plot(np.degrees(theta_range), p_theory * np.pi / 180, "r-", lw=2,
                    label="Boltzmann theory")
            ax.axvline(np.degrees(theta0), color="gray", ls="--", alpha=0.5,
                       label=f"θ₀={np.degrees(theta0):.0f}°")
            ax.set_xlabel("Angle (degrees)")
            ax.set_ylabel("Density")
            ax.set_title("Bond angles")
            ax.legend(fontsize=8)

        # 3) Dihedral distribution (if applicable)
        if "dihedral" in components:
            ax = axes[ax_idx]; ax_idx += 1
            ax.hist(np.degrees(dihedrals), bins=80, density=True, alpha=0.7, color="#C44E52",
                    label="MCMC samples")
            ax.axvline(np.degrees(phi0), color="gray", ls="--", alpha=0.5,
                       label=f"φ₀={np.degrees(phi0):.0f}°")
            ax.set_xlabel("Dihedral (degrees)")
            ax.set_ylabel("Density")
            ax.set_title("Dihedral angles")
            ax.legend(fontsize=8)

        # 4) Energy distribution
        ax = axes[ax_idx]
        ax.hist(energies, bins=80, density=True, alpha=0.7, color="#55A868",
                label="MCMC samples")
        ax.set_xlabel("Total energy")
        ax.set_ylabel("Density")
        ax.set_title("Energy distribution")
        ax.legend(fontsize=8)

        fig.suptitle(f"{preset}, N={N}, T={T}", fontsize=14, y=1.02)
        save_figure(fig, out / f"validate_{tag}.png")

    # Autocorrelation plot
    with synthbench_style():
        fig, ax = plt.subplots(figsize=(5, 3))
        lags = np.arange(min(200, len(acf)))
        ax.plot(lags, acf[:len(lags)], color="#4C72B0")
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Lag (samples)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Energy autocorrelation — {tag}")
        save_figure(fig, out / f"validate_{tag}_acf.png")

    print(f"\n  Plots saved to {out}/validate_{tag}*.png")


if __name__ == "__main__":
    main()
