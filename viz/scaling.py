"""Scaling law curves and capability heatmap."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from viz.style import ARCH_COLORS, ARCH_MARKERS, DOUBLE_COL


def _power_law(C, a, alpha, floor):
    return a * np.power(C, -alpha) + floor


def fit_scaling_law(
    flops: np.ndarray, metric: np.ndarray,
) -> tuple[float, float, float]:
    """Fit metric(C) = a * C^(-alpha) + floor.

    Args:
        flops: total training FLOPs per run.
        metric: corresponding metric values.

    Returns:
        (a, alpha, floor) tuple.
    """
    p0 = [1.0, 0.5, max(min(metric) * 0.5, 1e-6)]
    bounds = ([0, 0, 0], [np.inf, 5.0, np.inf])
    popt, _ = curve_fit(_power_law, flops, metric, p0=p0, bounds=bounds,
                        maxfev=10000)
    return tuple(popt)


def plot_scaling_curves(
    results: dict,
    ax: plt.Axes | None = None,
    fit_curves: bool = True,
    extrapolate_factor: float = 3.0,
    ylabel: str = "Energy Wasserstein",
) -> plt.Figure:
    """Plot scaling curves: metric vs. total training FLOPs.

    Args:
        results: dict mapping architecture name to a dict with keys:
            "flops": array of total training FLOPs,
            "metric": array of metric values,
            "metric_std": (optional) array of standard deviations.
        ax: optional existing axes.
        fit_curves: whether to fit and plot power law curves.
        extrapolate_factor: how far beyond data to extend fitted curves.
        ylabel: label for the y-axis.

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=DOUBLE_COL)
    else:
        fig = ax.get_figure()

    for arch, data in results.items():
        color = ARCH_COLORS.get(arch, "gray")
        marker = ARCH_MARKERS.get(arch, "x")
        flops = np.asarray(data["flops"], dtype=float)
        vals = np.asarray(data["metric"], dtype=float)
        vals_std = np.asarray(data.get("metric_std", np.zeros_like(vals)), dtype=float)

        # Data points
        label = arch
        ax.plot(flops, vals, marker=marker, color=color, linewidth=1.5,
                markersize=7, label=label, zorder=3)

        # Confidence interval
        if np.any(vals_std > 0):
            lo = np.clip(vals - vals_std, 1e-8, None)
            hi = vals + vals_std
            ax.fill_between(flops, lo, hi, alpha=0.15, color=color)

        # Fitted curve
        if fit_curves and len(flops) >= 3:
            try:
                a, alpha, floor = fit_scaling_law(flops, vals)
                f_min, f_max = flops.min(), flops.max()
                f_ext = np.geomspace(f_min / 2, f_max * extrapolate_factor, 200)
                vals_fit = _power_law(f_ext, a, alpha, floor)
                ax.plot(f_ext, vals_fit, color=color, linestyle="--", linewidth=1.0,
                        alpha=0.7)
                ax.lines[-2].set_label(f"{arch} (\u03b1={alpha:.2f})")
            except RuntimeError:
                pass

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
               label="perfect")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total training FLOPs")
    ax.set_ylabel(ylabel)
    ax.set_title("Compute Scaling")
    ax.legend(frameon=False)

    return fig


def plot_capability_heatmap(
    data: np.ndarray,
    ax: plt.Axes | None = None,
    architectures: list[str] | None = None,
    datasets: list[str] | None = None,
    cmap: str = "RdYlGn_r",
    fmt: str = ".2f",
) -> plt.Figure:
    """Heatmap of metric values across architectures and dataset variants.

    Args:
        data: (num_archs, num_datasets) metric matrix.
        ax: optional existing axes.
        architectures: row labels.
        datasets: column labels.
        cmap: colormap name.
        fmt: format string for cell annotations.

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    im = ax.imshow(data, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, label="Energy Wasserstein", shrink=0.8)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            rgba = im.cmap(im.norm(val))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if brightness < 0.5 else "black"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                    color=text_color, fontsize=10)

    if architectures is not None:
        ax.set_yticks(range(len(architectures)))
        ax.set_yticklabels(architectures)
    if datasets is not None:
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45, ha="right")

    ax.set_title("Energy Wasserstein by Architecture and Dataset")

    return fig
