"""
Uncertainty-aware statistical graphics.

Most published figures need more than a point estimate. This module provides
composable building blocks — every function draws into an optional ``ax`` and
returns the artists or fitted values — for:

* bootstrap confidence intervals (:func:`bootstrap_ci`, :func:`plot_mean_ci`)
* group comparisons with CIs and significance brackets
  (:func:`bar_with_ci`, :func:`add_significance_bar`, :func:`p_to_stars`)
* regression with confidence and prediction bands (:func:`linear_fit_with_ci`)
* distribution diagnostics (:func:`ecdf_plot`, :func:`qq_plot`, :func:`raincloud_plot`)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats as _st

from ._utils import as_array, get_axes

__all__ = [
    "add_significance_bar",
    "bar_with_ci",
    "bootstrap_ci",
    "ecdf_plot",
    "linear_fit_with_ci",
    "mean_ci",
    "p_to_stars",
    "plot_ci_band",
    "plot_mean_ci",
    "qq_plot",
    "raincloud_plot",
]


# --------------------------------------------------------------------------- intervals


def bootstrap_ci(
    data: Sequence[float] | np.ndarray,
    statistic: Callable[..., Any] = np.mean,
    *,
    n_boot: int = 2000,
    ci: float = 95,
    axis: int = 0,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap confidence interval of ``statistic`` along ``axis``.

    Works on 1-D samples (returns two floats) and on 2-D ``(n_samples, n_points)``
    arrays (returns two length-``n_points`` arrays), which is what you need for a
    confidence band around a mean curve.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(data, dtype=float)
    if arr.shape[axis] < 2:
        raise ValueError("Need at least two observations to bootstrap.")
    n = arr.shape[axis]
    idx = rng.integers(0, n, size=(n_boot, n))
    resampled = np.take(arr, idx, axis=axis)  # (n_boot, n, ...) along axis
    boot = statistic(resampled, axis=axis + 1 if axis >= 0 else axis)
    alpha = (100 - ci) / 2
    low, high = np.percentile(boot, [alpha, 100 - alpha], axis=0)
    return low, high


def mean_ci(
    data: Sequence[float] | np.ndarray,
    *,
    ci: float = 95,
    method: str = "t",
    axis: int = 0,
    n_boot: int = 2000,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mean, low, high)`` using a t-interval, ±SEM, ±SD or the bootstrap."""
    arr = np.asarray(data, dtype=float)
    mean = arr.mean(axis=axis)
    n = arr.shape[axis]
    if method == "bootstrap":
        low, high = bootstrap_ci(arr, np.mean, n_boot=n_boot, ci=ci, axis=axis, seed=seed)
    elif method in {"t", "sem", "std"}:
        sd = arr.std(axis=axis, ddof=1)
        if method == "std":
            half = sd
        else:
            sem = sd / np.sqrt(n)
            half = sem if method == "sem" else _st.t.ppf(0.5 + ci / 200, df=n - 1) * sem
        low, high = mean - half, mean + half
    else:
        raise ValueError("method must be 't', 'sem', 'std' or 'bootstrap'.")
    return mean, low, high


def plot_ci_band(
    ax: Axes,
    x: Sequence[float],
    low: Sequence[float],
    high: Sequence[float],
    *,
    color: str | None = None,
    alpha: float = 0.25,
    label: str | None = None,
    **fill_kw: Any,
):
    """Shade the region between ``low`` and ``high``."""
    return ax.fill_between(x, low, high, color=color, alpha=alpha, linewidth=0, label=label, **fill_kw)


def plot_mean_ci(
    x: Sequence[float],
    samples: np.ndarray | Sequence[Sequence[float]],
    *,
    ci: float = 95,
    method: str = "bootstrap",
    label: str | None = None,
    color: str | None = None,
    alpha: float = 0.25,
    n_boot: int = 2000,
    seed: int | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    **line_kw: Any,
) -> tuple[Figure, Axes]:
    """Mean curve with a confidence band from repeated measurements.

    ``samples`` has shape ``(n_repeats, len(x))`` — e.g. one row per random seed,
    subject or simulation run. The band is a bootstrap (default), t, SEM or SD interval.
    """
    arr = np.asarray(samples, dtype=float)
    x = as_array(x)
    if arr.ndim != 2 or arr.shape[1] != len(x):
        raise ValueError("samples must have shape (n_repeats, len(x)).")
    mean, low, high = mean_ci(arr, ci=ci, method=method, axis=0, n_boot=n_boot, seed=seed)
    fig, ax = get_axes(ax, figsize)
    (line,) = ax.plot(x, mean, color=color, label=label, **line_kw)
    band_label = None if label is None else f"{ci:g}% CI" if method != "std" else "±1 SD"
    plot_ci_band(ax, x, low, high, color=line.get_color(), alpha=alpha, label=band_label)
    if label:
        ax.legend()
    return fig, ax


# --------------------------------------------------------------------------- group comparison


def bar_with_ci(
    groups: Mapping[str, Sequence[float]],
    *,
    ci: float = 95,
    method: str = "bootstrap",
    statistic: Callable[..., Any] = np.mean,
    colors: Sequence[str] | None = None,
    show_points: bool = True,
    jitter: float = 0.08,
    point_kw: dict[str, Any] | None = None,
    seed: int | None = 0,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    ylabel: str = "",
    title: str = "",
) -> tuple[Figure, Axes, dict[str, tuple[float, float, float]]]:
    """Bars of a statistic per group with CI error bars and (optionally) the raw points.

    Returns ``(fig, ax, summary)`` where ``summary[name] = (estimate, low, high)``.
    """
    rng = np.random.default_rng(seed)
    fig, ax = get_axes(ax, figsize)
    names = list(groups)
    summary: dict[str, tuple[float, float, float]] = {}
    for i, name in enumerate(names):
        vals = np.asarray(groups[name], dtype=float)
        est = float(statistic(vals))
        if method == "bootstrap":
            low, high = bootstrap_ci(vals, statistic, ci=ci, seed=rng)
        else:
            _, low, high = mean_ci(vals, ci=ci, method=method)
        summary[name] = (est, float(low), float(high))
        color = colors[i % len(colors)] if colors else f"C{i}"
        ax.bar(i, est, color=color, alpha=0.8, zorder=2)
        ax.errorbar(i, est, yerr=[[est - low], [high - est]], fmt="none", ecolor="black", capsize=4, zorder=3)
        if show_points:
            xs = i + rng.uniform(-jitter, jitter, size=vals.size)
            ax.scatter(
                xs, vals, **({"s": 12, "color": "black", "alpha": 0.5, "zorder": 4} | (point_kw or {}))
            )
    ax.set_xticks(range(len(names)), labels=names)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax, summary


def p_to_stars(p: float, *, ns: str = "n.s.") -> str:
    """Conventional significance stars: ``***`` <0.001, ``**`` <0.01, ``*`` <0.05."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ns


def add_significance_bar(
    ax: Axes,
    x1: float,
    x2: float,
    y: float | None = None,
    text: str | float = "*",
    *,
    height: float | None = None,
    color: str = "black",
    linewidth: float = 1.0,
    fontsize: float | None = None,
) -> None:
    """Draw a bracket between ``x1`` and ``x2`` with a label (a string or a p-value).

    ``y`` defaults to just above the current data range; ``height`` is the tick length.
    Passing a float for ``text`` converts it with :func:`p_to_stars`.
    """
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if y is None:
        y = ymax - 0.05 * span
    if height is None:
        height = 0.02 * span
    label = p_to_stars(text) if isinstance(text, (int, float)) else text
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color=color, linewidth=linewidth, clip_on=False)
    ax.text((x1 + x2) / 2, y + height, label, ha="center", va="bottom", color=color, fontsize=fontsize)
    if y + 4 * height > ymax:
        ax.set_ylim(ymin, y + 6 * height)


# --------------------------------------------------------------------------- regression


def linear_fit_with_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    ci: float = 95,
    prediction_band: bool = True,
    scatter: bool = True,
    color: str | None = None,
    label: str | None = "OLS fit",
    n_grid: int = 200,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes, dict[str, float]]:
    """Ordinary least squares fit with confidence (mean) and prediction (new obs.) bands.

    Returns ``(fig, ax, result)`` where ``result`` holds ``slope``, ``intercept``,
    ``r``, ``r2``, ``p``, ``stderr`` and ``n``.
    """
    xa, ya = as_array(x).astype(float), as_array(y).astype(float)
    if xa.size != ya.size or xa.size < 3:
        raise ValueError("x and y must have equal length ≥ 3.")
    res = _st.linregress(xa, ya)
    n = xa.size
    dof = n - 2
    t = _st.t.ppf(0.5 + ci / 200, dof)
    yhat = res.intercept + res.slope * xa
    s_err = np.sqrt(np.sum((ya - yhat) ** 2) / dof)
    grid = np.linspace(xa.min(), xa.max(), n_grid)
    fit = res.intercept + res.slope * grid
    sxx = np.sum((xa - xa.mean()) ** 2)
    se_mean = s_err * np.sqrt(1 / n + (grid - xa.mean()) ** 2 / sxx)
    se_pred = s_err * np.sqrt(1 + 1 / n + (grid - xa.mean()) ** 2 / sxx)

    fig, ax = get_axes(ax, figsize)
    if scatter:
        ax.scatter(xa, ya, s=18, alpha=0.6, color=color, zorder=3)
    (line,) = ax.plot(grid, fit, color=color, label=label, zorder=4)
    c = line.get_color()
    ax.fill_between(
        grid,
        fit - t * se_mean,
        fit + t * se_mean,
        color=c,
        alpha=0.25,
        linewidth=0,
        label=f"{ci:g}% CI (mean)",
    )
    if prediction_band:
        ax.fill_between(
            grid,
            fit - t * se_pred,
            fit + t * se_pred,
            color=c,
            alpha=0.10,
            linewidth=0,
            label=f"{ci:g}% prediction band",
        )
    if label:
        ax.legend()
    result = {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r": float(res.rvalue),
        "r2": float(res.rvalue**2),
        "p": float(res.pvalue),
        "stderr": float(res.stderr),
        "n": int(n),
    }
    return fig, ax, result


# --------------------------------------------------------------------------- distributions


def _orientation_kw(ax: Axes, vertical: bool) -> dict[str, Any]:
    """Return the orientation keyword accepted by this Matplotlib's boxplot/violinplot.

    Matplotlib ≥ 3.10 takes ``orientation=...``; older releases only know ``vert=...``.
    """
    import inspect

    if "orientation" in inspect.signature(ax.violinplot).parameters:
        return {"orientation": "vertical" if vertical else "horizontal"}
    return {"vert": vertical}


def ecdf_plot(
    data: Sequence[float],
    *,
    complementary: bool = False,
    label: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    **step_kw: Any,
) -> tuple[Figure, Axes]:
    """Empirical (complementary) cumulative distribution function as a step plot."""
    arr = np.sort(as_array(data).astype(float))
    if arr.size == 0:
        raise ValueError("data must be non-empty.")
    y = np.arange(1, arr.size + 1) / arr.size
    if complementary:
        y = 1 - y + 1 / arr.size
    fig, ax = get_axes(ax, figsize)
    ax.step(arr, y, where="post", label=label, **step_kw)
    ax.set_ylabel("1 − ECDF" if complementary else "ECDF")
    ax.set_ylim(0, 1.02)
    if label:
        ax.legend()
    return fig, ax


def qq_plot(
    data: Sequence[float],
    dist: str | Any = "norm",
    *,
    sparams: tuple = (),
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    color: str | None = None,
) -> tuple[Figure, Axes, dict[str, float]]:
    """Quantile–quantile plot against a scipy distribution, with the fitted reference line.

    Returns ``(fig, ax, {"slope", "intercept", "r"})``; ``r`` close to 1 indicates a good fit.
    """
    arr = as_array(data).astype(float)
    (osm, osr), (slope, intercept, r) = _st.probplot(arr, dist=dist, sparams=sparams)
    fig, ax = get_axes(ax, figsize)
    ax.scatter(osm, osr, s=14, alpha=0.7, color=color)
    xs = np.array([osm.min(), osm.max()])
    ax.plot(xs, intercept + slope * xs, color="tab:red", linewidth=1)
    name = dist if isinstance(dist, str) else getattr(dist, "name", "dist")
    ax.set_xlabel(f"Theoretical quantiles ({name})")
    ax.set_ylabel("Sample quantiles")
    return fig, ax, {"slope": float(slope), "intercept": float(intercept), "r": float(r)}


def raincloud_plot(
    groups: Mapping[str, Sequence[float]],
    *,
    orient: str = "h",
    colors: Sequence[str] | None = None,
    point_size: float = 8,
    jitter: float = 0.08,
    box_width: float = 0.12,
    violin_width: float = 0.7,
    alpha: float = 0.6,
    seed: int | None = 0,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    xlabel: str = "",
    title: str = "",
) -> tuple[Figure, Axes]:
    """Raincloud plot (Allen et al., 2019): half-violin + box plot + jittered raw points.

    Shows the distribution's shape, its summary statistics and every observation
    at once, which is why it is increasingly preferred over bar charts of means.
    """
    rng = np.random.default_rng(seed)
    names = list(groups)
    data = [np.asarray(groups[n], dtype=float) for n in names]
    vertical = orient.lower().startswith("v")
    fig, ax = get_axes(ax, figsize)
    positions = np.arange(len(names), dtype=float)

    orient_kw = _orientation_kw(ax, vertical)
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=violin_width,
        showextrema=False,
        showmeans=False,
        showmedians=False,
        **orient_kw,
    )
    for i, body in enumerate(parts["bodies"]):
        color = colors[i % len(colors)] if colors else f"C{i}"
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(alpha)
        # keep only the upper (vertical) / right-hand (horizontal) half → "cloud"
        verts = body.get_paths()[0].vertices
        if vertical:
            verts[:, 0] = np.clip(verts[:, 0], -np.inf, positions[i])
        else:
            verts[:, 1] = np.clip(verts[:, 1], positions[i], np.inf)

    offset = 0.18
    box_pos = positions - offset
    bp = ax.boxplot(
        data,
        positions=box_pos,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black"},
        **orient_kw,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor("black")

    for i, vals in enumerate(data):
        color = colors[i % len(colors)] if colors else f"C{i}"
        noise = rng.uniform(-jitter, jitter, size=vals.size)
        cat = positions[i] - 2 * offset + noise
        if vertical:
            ax.scatter(cat, vals, s=point_size, color=color, alpha=0.7, zorder=3, linewidths=0)
        else:
            ax.scatter(vals, cat, s=point_size, color=color, alpha=0.7, zorder=3, linewidths=0)

    if vertical:
        ax.set_xticks(positions, labels=names)
        if xlabel:
            ax.set_ylabel(xlabel)
    else:
        ax.set_yticks(positions, labels=names)
        if xlabel:
            ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="x" if not vertical else "y", alpha=0.3)
    return fig, ax
