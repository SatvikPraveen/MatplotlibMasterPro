"""
High-level plotting helpers used throughout the notebooks.

Design contract
---------------
* Every function accepts ``ax=None``; pass an existing ``Axes`` to draw into a
  subplot, or omit it to get a fresh figure.
* Functions return the objects they create — ``(fig, ax)`` for single-axes
  plots, ``(fig, axes)`` for grids, an :class:`~matplotlib.animation.FuncAnimation`
  for animations, and the written paths for ``save_*`` helpers.
* Nothing calls ``plt.show()`` unless you pass ``show=True``. Jupyter's inline
  backend displays open figures automatically at the end of a cell.
"""

from __future__ import annotations

import shutil
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._utils import as_array, ensure_dir, finalize, get_axes, validate_xy
from .colors import text_color_for

try:  # optional dependency used only by the interactive helpers
    from ipywidgets import interact, widgets

    HAS_IPYWIDGETS = True
except ImportError:  # pragma: no cover - exercised only without ipywidgets
    HAS_IPYWIDGETS = False

__all__ = [
    # basics
    "line_plot",
    "multi_line_plot",
    "bar_plot",
    "grouped_bar_plot",
    "scatter_plot",
    "histogram_plot",
    "pie_chart",
    "grid_plot",
    "dual_axis_plot",
    "fill_between_plot",
    "log_scale_plot",
    "twin_axes_fill_plot",
    # annotations
    "annotate_point",
    "highlight_region",
    "label_line",
    # images
    "imshow_matrix",
    "plot_image",
    "grid_heatmap",
    # interactive
    "interactive_slider_plot",
    "dropdown_plot",
    # export & style
    "save_plot",
    "set_global_style",
    # animation
    "animate_line_plot",
    "animate_dual_line_plot",
    "animate_bar_chart",
    "animate_scatter_growth",
    "save_animation",
    "save_line_animation",
    "save_dual_line_animation",
    "save_bar_animation",
    "save_scatter_animation",
    # statistics
    "plot_histogram_with_stats",
    "plot_boxplot",
    "plot_violinplot",
    # comparative
    "stacked_area_plot",
    "subplot_groupwise",
    "save_grouped_bar_plot",
    "save_stacked_area_plot",
    "save_subplot_groupwise",
    # colormaps
    "display_colormap_samples",
    "save_sequential_bar_plot",
    "save_diverging_change_plot",
    "save_qualitative_grouped_bar",
    # time series
    "plot_timeseries_trend",
    "save_timeseries_trend",
    "plot_rolling_mean_std",
    "save_rolling_stats_plot",
    "plot_multi_product_timeseries",
    "save_multi_product_timeseries",
    # dashboards
    "create_dashboard",
]

FigAx = tuple[Figure, Axes]


def _save_and_close(fig: Figure, path: str | Path, dpi: int) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


# =============================================================================
# Basic plots
# =============================================================================


def line_plot(
    x: Sequence[Any],
    y: Sequence[Any],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    label: str | None = None,
    color: str = "tab:blue",
    linestyle: str = "-",
    marker: str | None = None,
    figsize: tuple[float, float] = (10, 5),
    grid: bool = True,
    *,
    linewidth: float | None = None,
    markersize: float | None = None,
    rotate_xticks: float | None = 45,
    ax: Axes | None = None,
    show: bool = False,
    **plot_kw: Any,
) -> FigAx:
    """Single line plot with sensible defaults. Extra keywords go to ``Axes.plot``."""
    validate_xy(x, y)
    fig, ax = get_axes(ax, figsize)
    kw: dict[str, Any] = {"label": label, "color": color, "linestyle": linestyle, "marker": marker}
    if linewidth is not None:
        kw["linewidth"] = linewidth
    if markersize is not None:
        kw["markersize"] = markersize
    kw.update(plot_kw)
    ax.plot(x, y, **kw)
    return finalize(
        fig,
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=grid,
        legend=bool(label),
        rotate_xticks=rotate_xticks,
        show=show,
    )


def multi_line_plot(
    df: pd.DataFrame | Sequence[Any] | None = None,
    x_col: str | Sequence[Any] | None = None,
    y_cols: Sequence[str] | Sequence[Sequence[Any]] | Mapping[str, Sequence[Any]] | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    markers: Sequence[str | None] | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple[float, float] = (10, 6),
    *,
    x: Sequence[Any] | None = None,
    ys: Sequence[Sequence[Any]] | Mapping[str, Sequence[Any]] | None = None,
    linestyles: Sequence[str] | None = None,
    grid: bool = True,
    rotate_xticks: float | None = 45,
    ax: Axes | None = None,
    show: bool = False,
) -> FigAx:
    """Plot several series on one axes.

    Two calling conventions are supported:

    * **DataFrame**: ``multi_line_plot(df, "Month", ["Laptop", "Tablet"])`` — columns of ``df``.
    * **Arrays**: ``multi_line_plot(x, [y1, y2], labels=["a", "b"])`` or the explicit
      keywords ``x=..., ys=...``; ``ys`` may also be a ``{label: y}`` mapping.
    """
    if isinstance(df, pd.DataFrame):
        if x_col is None or y_cols is None:
            raise ValueError("DataFrame mode requires x_col and y_cols.")
        x_vals = df[x_col]
        series = {str(c): df[c] for c in y_cols}
    else:
        x_vals = x if x is not None else df
        y_input = ys if ys is not None else (y_cols if y_cols is not None else x_col)
        if x_vals is None or y_input is None:
            raise ValueError("Provide either (df, x_col, y_cols) or (x, ys).")
        if isinstance(y_input, Mapping):
            series = {str(k): v for k, v in y_input.items()}
        else:
            series = {f"Series {i + 1}": v for i, v in enumerate(y_input)}
    if labels is not None:
        series = dict(zip(labels, series.values()))

    fig, ax = get_axes(ax, figsize)
    for i, (label, y) in enumerate(series.items()):
        validate_xy(x_vals, y)
        ax.plot(
            x_vals,
            y,
            label=label,
            color=colors[i] if colors else None,
            marker=markers[i] if markers else None,
            linestyle=linestyles[i] if linestyles else "-",
        )
    return finalize(
        fig,
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=grid,
        legend=True,
        rotate_xticks=rotate_xticks,
        show=show,
    )


def bar_plot(
    categories: Sequence[Any],
    values: Sequence[float],
    *,
    horizontal: bool = False,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    color: str | Sequence[str] = "skyblue",
    edgecolor: str = "black",
    grid: bool = True,
    figsize: tuple[float, float] = (8, 5),
    annotate: bool = False,
    fmt: str = "{:,.0f}",
    ax: Axes | None = None,
    show: bool = False,
    **bar_kw: Any,
) -> FigAx:
    """Vertical or horizontal bar chart. ``annotate=True`` prints each value on its bar."""
    validate_xy(categories, values)
    fig, ax = get_axes(ax, figsize)
    if horizontal:
        bars = ax.barh(categories, values, color=color, edgecolor=edgecolor, **bar_kw)
        if grid:
            ax.grid(True, axis="x", alpha=0.4)
    else:
        bars = ax.bar(categories, values, color=color, edgecolor=edgecolor, **bar_kw)
        if grid:
            ax.grid(True, axis="y", alpha=0.4)
    if annotate:
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=3, fontsize=9)
    return finalize(fig, ax, title=title, xlabel=xlabel, ylabel=ylabel, show=show)


def grouped_bar_plot(
    x_labels: Sequence[Any] | pd.DataFrame | None = None,
    data_dict: Mapping[str, Sequence[float]] | str | None = None,
    *,
    df: pd.DataFrame | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    value: str | None = None,
    agg: str = "sum",
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    bar_width: float | None = None,
    figsize: tuple[float, float] = (10, 6),
    legend_title: str | None = None,
    rotate_xticks: float | None = 45,
    ax: Axes | None = None,
    show: bool = False,
) -> FigAx:
    """Grouped (side-by-side) bars.

    * **Dict mode**: ``grouped_bar_plot(x_labels, {"Group A": [...], "Group B": [...]})``.
    * **DataFrame mode**: ``grouped_bar_plot(df=df, category="Month", subcategory="Product",
      value="Revenue")`` aggregates ``value`` with ``agg`` per (category, subcategory).
    """
    if df is None and isinstance(x_labels, pd.DataFrame):
        df, x_labels = x_labels, None
        if isinstance(data_dict, str):  # positional (df, category, ...) form
            category, data_dict = data_dict, None
    if df is not None:
        if not (category and subcategory and value):
            raise ValueError("DataFrame mode requires category, subcategory and value.")
        pivot = df.pivot_table(index=category, columns=subcategory, values=value, aggfunc=agg).fillna(0)
        x_labels = list(pivot.index)
        data_dict = {str(col): pivot[col].to_numpy() for col in pivot.columns}
        xlabel = category if xlabel is None else xlabel
        ylabel = value if ylabel is None else ylabel
        legend_title = subcategory if legend_title is None else legend_title
    if x_labels is None or not isinstance(data_dict, Mapping):
        raise ValueError("Provide (x_labels, data_dict) or a DataFrame with category/subcategory/value.")

    n_groups = len(data_dict)
    width = bar_width if bar_width is not None else 0.8 / max(n_groups, 1)
    x = np.arange(len(x_labels))
    fig, ax = get_axes(ax, figsize)
    for i, (name, vals) in enumerate(data_dict.items()):
        if len(vals) != len(x_labels):
            raise ValueError(f"Series {name!r} has {len(vals)} values but there are {len(x_labels)} labels.")
        ax.bar(x + i * width, vals, width=width, label=str(name))
    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels([str(lbl) for lbl in x_labels])
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(title=legend_title)
    return finalize(
        fig,
        ax,
        title=title,
        xlabel=xlabel or "",
        ylabel=ylabel or "",
        rotate_xticks=rotate_xticks,
        show=show,
    )


def scatter_plot(
    x: Sequence[float],
    y: Sequence[float],
    *,
    color: str = "green",
    size: float | Sequence[float] = 80,
    alpha: float = 0.7,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    edgecolor: str = "white",
    cmap: str | None = None,
    color_values: Sequence[float] | None = None,
    use_colorbar: bool = False,
    colorbar_label: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    grid: bool = True,
    c: Sequence[float] | str | None = None,
    s: float | Sequence[float] | None = None,
    ax: Axes | None = None,
    show: bool = False,
    **scatter_kw: Any,
) -> FigAx:
    """Scatter plot; ``c``/``s`` are accepted as aliases for ``color_values``/``size``."""
    validate_xy(x, y)
    if c is not None:
        color_values = c
    if s is not None:
        size = s
    fig, ax = get_axes(ax, figsize)
    if color_values is not None and not isinstance(color_values, str):
        sc = ax.scatter(
            x,
            y,
            c=color_values,
            s=size,
            cmap=cmap or "viridis",
            alpha=alpha,
            edgecolors=edgecolor,
            **scatter_kw,
        )
        if use_colorbar:
            fig.colorbar(sc, ax=ax, label=colorbar_label or "")
    else:
        ax.scatter(x, y, c=color_values or color, s=size, alpha=alpha, edgecolors=edgecolor, **scatter_kw)
    return finalize(fig, ax, title=title, xlabel=xlabel, ylabel=ylabel, grid=grid, show=show)


def histogram_plot(
    data: Sequence[float],
    *,
    bins: int | Sequence[float] | str = 10,
    color: str = "steelblue",
    edgecolor: str = "black",
    alpha: float = 0.7,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Frequency",
    density: bool = False,
    grid: bool = True,
    figsize: tuple[float, float] = (8, 5),
    ax: Axes | None = None,
    show: bool = False,
    **hist_kw: Any,
) -> FigAx:
    """Histogram; ``density=True`` normalises the area to one and relabels the y-axis."""
    if len(data) == 0:
        raise ValueError("data must be non-empty.")
    fig, ax = get_axes(ax, figsize)
    ax.hist(data, bins=bins, color=color, edgecolor=edgecolor, alpha=alpha, density=density, **hist_kw)
    return finalize(
        fig,
        ax,
        title=title,
        xlabel=xlabel,
        ylabel="Density" if density else ylabel,
        grid=grid,
        show=show,
    )


def pie_chart(
    values: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    *,
    colors: Sequence[str] | None = None,
    explode: Sequence[float] | None = None,
    autopct: str | Callable | None = "%1.1f%%",
    title: str = "",
    startangle: float = 90,
    shadow: bool = False,
    figsize: tuple[float, float] = (6, 6),
    ax: Axes | None = None,
    show: bool = False,
    **pie_kw: Any,
) -> FigAx:
    """Pie chart. Both ``pie_chart(values, labels=...)`` and ``pie_chart(labels=..., values=...)`` work."""
    if values is None:
        raise ValueError("values is required.")
    if labels is not None and not _is_numeric(values) and _is_numeric(labels):
        # The caller passed (labels, values) positionally — swap them.
        values, labels = labels, values
    fig, ax = get_axes(ax, figsize)
    ax.pie(
        values,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct=autopct,
        startangle=startangle,
        shadow=shadow,
        **pie_kw,
    )
    ax.axis("equal")
    return finalize(fig, ax, title=title, show=show)


def _is_numeric(values: Sequence[Any]) -> bool:
    try:
        np.asarray(values, dtype=float)
        return True
    except (TypeError, ValueError):
        return False


def grid_plot(
    plot_funcs: Sequence[Callable[[Axes], Any]],
    *,
    titles: Sequence[str] | None = None,
    nrows: int = 1,
    ncols: int = 2,
    figsize: tuple[float, float] = (12, 5),
    suptitle: str | None = None,
    sharex: bool = False,
    sharey: bool = False,
    show: bool = False,
) -> tuple[Figure, np.ndarray]:
    """Call each function in ``plot_funcs`` with its own ``Axes`` in an ``nrows × ncols`` grid.

    Unused axes are removed. Returns ``(fig, axes)`` with ``axes`` flattened.
    """
    if len(plot_funcs) > nrows * ncols:
        raise ValueError(f"{len(plot_funcs)} plot functions do not fit in a {nrows}x{ncols} grid.")
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey, squeeze=False
    )
    axes = axes.ravel()
    for i, func in enumerate(plot_funcs):
        func(axes[i])
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
    for j in range(len(plot_funcs), len(axes)):
        fig.delaxes(axes[j])
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes[: len(plot_funcs)]


def dual_axis_plot(
    x: Sequence[Any],
    y1: Sequence[float],
    y2: Sequence[float],
    *,
    label1: str = "Primary",
    label2: str = "Secondary",
    color1: str = "tab:blue",
    color2: str = "tab:red",
    xlabel: str = "",
    ylabel1: str = "",
    ylabel2: str = "",
    title: str = "",
    figsize: tuple[float, float] = (10, 5),
    marker1: str | None = None,
    marker2: str | None = None,
    legend: bool = True,
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Two series sharing ``x`` on independent y-axes. Returns ``(fig, (ax_left, ax_right))``."""
    validate_xy(x, y1)
    validate_xy(x, y2)
    fig, ax1 = get_axes(ax, figsize)
    l1 = ax1.plot(x, y1, color=color1, marker=marker1, label=label1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2 = ax1.twinx()
    l2 = ax2.plot(x, y2, color=color2, marker=marker2, label=label2)
    ax2.set_ylabel(ylabel2, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    if legend:
        ax1.legend(l1 + l2, [label1, label2], loc="upper left")
    if title:
        ax1.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, (ax1, ax2)


def fill_between_plot(
    x: Sequence[Any],
    y1: Sequence[float],
    y2: float | Sequence[float] = 0,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    color: str = "skyblue",
    alpha: float = 0.4,
    label: str | None = None,
    edge: bool = True,
    linestyle: str = "--",
    figsize: tuple[float, float] = (10, 5),
    rotate_xticks: float | None = 45,
    ax: Axes | None = None,
    show: bool = False,
) -> FigAx:
    """Shade the area between ``y1`` and ``y2`` (default: the x-axis)."""
    validate_xy(x, y1)
    fig, ax = get_axes(ax, figsize)
    if edge:
        ax.plot(x, y1, label=label, linestyle=linestyle, color=color)
    ax.fill_between(x, y1, y2, color=color, alpha=alpha, label=None if edge else label)
    return finalize(
        fig,
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=True,
        legend=bool(label),
        rotate_xticks=rotate_xticks,
        show=show,
    )


def log_scale_plot(
    x: Sequence[float],
    y: Sequence[float],
    *,
    log_axis: str = "y",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    label: str | None = None,
    color: str = "navy",
    marker: str | None = "o",
    linestyle: str = "-",
    figsize: tuple[float, float] = (8, 5),
    rotate_xticks: float | None = 45,
    ax: Axes | None = None,
    show: bool = False,
) -> FigAx:
    """Line plot with logarithmic ``"x"``, ``"y"`` or ``"both"`` axes."""
    validate_xy(x, y)
    if log_axis not in {"x", "y", "both"}:
        raise ValueError("log_axis must be 'x', 'y' or 'both'.")
    fig, ax = get_axes(ax, figsize)
    ax.plot(x, y, label=label, color=color, linestyle=linestyle, marker=marker)
    if log_axis in {"x", "both"}:
        ax.set_xscale("log")
    if log_axis in {"y", "both"}:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    return finalize(
        fig,
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=bool(label),
        rotate_xticks=rotate_xticks,
        show=show,
    )


def twin_axes_fill_plot(
    x: Sequence[Any],
    y1: Sequence[float],
    y2: Sequence[float],
    *,
    label1: str = "Y1",
    label2: str = "Y2",
    xlabel: str = "",
    ylabel1: str = "",
    ylabel2: str = "",
    color1: str = "tab:blue",
    color2: str = "tab:green",
    alpha1: float = 0.5,
    alpha2: float = 0.3,
    title: str = "",
    figsize: tuple[float, float] = (10, 5),
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Filled areas for two series on twin y-axes."""
    validate_xy(x, y1)
    validate_xy(x, y2)
    fig, ax1 = get_axes(ax, figsize)
    ax1.fill_between(x, y1, color=color1, alpha=alpha1, label=label1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2 = ax1.twinx()
    ax2.fill_between(x, y2, color=color2, alpha=alpha2, label=label2)
    ax2.set_ylabel(ylabel2, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    h1, n1 = ax1.get_legend_handles_labels()
    h2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, n1 + n2, loc="upper left")
    if title:
        ax1.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, (ax1, ax2)


# =============================================================================
# Annotations
# =============================================================================


def annotate_point(
    ax: Axes,
    x: Any,
    y: float,
    text: str,
    *,
    xytext: tuple[float, float] = (10, 10),
    textcolor: str = "black",
    arrowprops: dict[str, Any] | None = None,
    fontsize: float = 10,
    **kw: Any,
):
    """Annotate a single data point with an arrow (offset in points)."""
    if arrowprops is None:
        arrowprops = {"arrowstyle": "->", "color": "black"}
    return ax.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        fontsize=fontsize,
        color=textcolor,
        arrowprops=arrowprops,
        **kw,
    )


def highlight_region(
    ax: Axes,
    x_start: Any,
    x_end: Any,
    *,
    color: str = "yellow",
    alpha: float = 0.3,
    label: str | None = None,
):
    """Shade a vertical band between two x positions."""
    return ax.axvspan(x_start, x_end, color=color, alpha=alpha, label=label)


def label_line(
    ax: Axes,
    x: Sequence[Any],
    y: Sequence[float],
    text: str,
    *,
    color: str = "black",
    fontsize: float = 10,
    location: str = "right",
    offset: tuple[float, float] = (0, 0),
    use_index: bool | None = None,
):
    """Write ``text`` next to a line at its ``'left'``, ``'center'`` or ``'right'`` end.

    Categorical x values are positioned by index; numeric/datetime values by value
    (override with ``use_index``).
    """
    idx = {"left": 0, "center": len(x) // 2, "right": len(x) - 1}[location]
    x_arr = as_array(x)
    if use_index is None:
        use_index = x_arr.dtype.kind in "OUS"
    xpos = idx + offset[0] if use_index else x_arr[idx]
    return ax.text(xpos, y[idx] + offset[1], text, fontsize=fontsize, color=color)


# =============================================================================
# Images & grids
# =============================================================================


def imshow_matrix(
    matrix: np.ndarray,
    *,
    cmap: str = "viridis",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    colorbar: bool = True,
    figsize: tuple[float, float] = (6, 5),
    ax: Axes | None = None,
    show: bool = False,
    **imshow_kw: Any,
) -> FigAx:
    """Display a 2-D array with ``imshow`` (``aspect='auto'``) and an optional colourbar."""
    fig, ax = get_axes(ax, figsize)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", **imshow_kw)
    if colorbar:
        fig.colorbar(im, ax=ax)
    return finalize(fig, ax, title=title, xlabel=xlabel, ylabel=ylabel, show=show)


def plot_image(
    image_array: np.ndarray,
    *,
    cmap: str | None = None,
    title: str = "",
    figsize: tuple[float, float] = (6, 6),
    ax: Axes | None = None,
    show: bool = False,
) -> FigAx:
    """Show an image (grayscale or RGB) without axes."""
    fig, ax = get_axes(ax, figsize)
    ax.imshow(image_array, cmap=cmap)
    ax.set_axis_off()
    return finalize(fig, ax, title=title, show=show)


def grid_heatmap(
    data: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    cmap: str = "YlGnBu",
    annot: bool = False,
    fmt: str = ".2f",
    title: str = "",
    figsize: tuple[float, float] = (8, 6),
    cbar_label: str = "",
    ax: Axes | None = None,
    show: bool = False,
) -> FigAx:
    """Labelled heat map; annotations automatically switch to white on dark cells."""
    data = np.asarray(data)
    if data.shape != (len(row_labels), len(col_labels)):
        raise ValueError("data shape must be (len(row_labels), len(col_labels)).")
    fig, ax = get_axes(ax, figsize)
    im = ax.imshow(data, cmap=cmap)
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    if annot:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                ax.text(
                    j,
                    i,
                    format(data[i, j], fmt),
                    ha="center",
                    va="center",
                    color=text_color_for(im.cmap(im.norm(data[i, j]))),
                )
    fig.colorbar(im, ax=ax, label=cbar_label)
    return finalize(fig, ax, title=title, show=show)


# =============================================================================
# Interactive (ipywidgets)
# =============================================================================


def _require_ipywidgets() -> None:
    if not HAS_IPYWIDGETS:
        raise ImportError("ipywidgets is required for interactive plots: pip install ipywidgets")


def interactive_slider_plot(
    x: Sequence[Any],
    y_series_dict: Mapping[str, Sequence[float]],
    *,
    xlabel: str = "X",
    ylabel: str = "Y",
    title_prefix: str = "Value at index",
):
    """Slider that highlights the value of each series at the selected ``x`` index."""
    _require_ipywidgets()
    x = list(x)

    @interact(index=widgets.IntSlider(min=0, max=len(x) - 1, step=1, value=0))
    def _plot(index):
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, y in y_series_dict.items():
            ax.plot(x, y, label=label)
            ax.scatter([x[index]], [y[index]], s=80, label=f"{label} @ {x[index]} = {y[index]}")
        finalize(
            fig,
            ax,
            title=f"{title_prefix}: {x[index]}",
            xlabel=xlabel,
            ylabel=ylabel,
            grid=True,
            legend=True,
            rotate_xticks=45,
            show=True,
        )

    return _plot


def dropdown_plot(
    x: Sequence[Any],
    y_series_dict: Mapping[str, Sequence[float]],
    *,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Dropdown Series Viewer",
):
    """Dropdown to choose which series to display."""
    _require_ipywidgets()

    @interact(series=widgets.Dropdown(options=list(y_series_dict.keys()), description="Series"))
    def _plot(series):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y_series_dict[series], label=series, color="tab:blue", marker="o")
        finalize(
            fig,
            ax,
            title=f"{title} — {series}",
            xlabel=xlabel,
            ylabel=ylabel,
            grid=True,
            legend=True,
            rotate_xticks=45,
            show=True,
        )

    return _plot


# =============================================================================
# Export & global style
# =============================================================================


def save_plot(
    fig: Figure,
    filename: str,
    folder: str | Path = "exports",
    formats: Sequence[str] = ("png", "pdf", "svg"),
    *,
    dpi: int = 300,
    verbose: bool = True,
    **savefig_kw: Any,
) -> list[Path]:
    """Save ``fig`` as ``folder/filename.<fmt>`` for each format; returns the paths."""
    folder = ensure_dir(folder)
    paths = []
    for fmt in formats:
        path = folder / f"{filename}.{fmt}"
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight", **savefig_kw)
        paths.append(path)
    if verbose:
        print(f"Saved {filename} as {', '.join(formats)} in '{folder}/'")
    return paths


def set_global_style(style_name: str = "ggplot", font: str = "DejaVu Sans", size: float = 12) -> None:
    """Apply a Matplotlib style sheet plus a global font family and size."""
    plt.style.use(style_name)
    plt.rcParams.update({"font.family": font, "font.size": size})


# =============================================================================
# Animation
# =============================================================================


def _anim_axes(x, y_min, y_max, *, xlabel, ylabel, title, figsize=(10, 5), grid=True) -> FigAx:
    fig, ax = plt.subplots(figsize=figsize)
    x_arr = as_array(x)
    if x_arr.dtype.kind in "OUS":  # categorical x: use index positions for limits
        ax.set_xlim(-0.5, len(x_arr) - 0.5)
    else:
        ax.set_xlim(np.min(x_arr), np.max(x_arr))
    pad = 0.05 * ((y_max - y_min) or 1)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.4)
    return fig, ax


def animate_line_plot(
    x: Sequence[Any],
    y: Sequence[float],
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    interval: int = 200,
    color: str = "tab:blue",
    figsize: tuple[float, float] = (10, 5),
    repeat: bool = False,
    show: bool = False,
) -> FuncAnimation:
    """Progressively reveal a line. Returns the animation (keep a reference to it!)."""
    validate_xy(x, y)
    y_arr = as_array(y).astype(float)
    fig, ax = _anim_axes(
        x, float(y_arr.min()), float(y_arr.max()), xlabel=xlabel, ylabel=ylabel, title=title, figsize=figsize
    )
    (line,) = ax.plot([], [], color=color, linewidth=2)
    x_list = list(x)

    def update(frame):
        line.set_data(x_list[:frame], y_arr[:frame])
        return (line,)

    anim = FuncAnimation(fig, update, frames=len(x_list) + 1, interval=interval, blit=False, repeat=repeat)
    if show:
        plt.show()
    return anim


def animate_dual_line_plot(
    x: Sequence[Any],
    y1: Sequence[float],
    y2: Sequence[float],
    *,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Dual Line Animation",
    labels: tuple[str, str] = ("Line 1", "Line 2"),
    colors: tuple[str, str] = ("tab:blue", "tab:green"),
    interval: int = 300,
    figsize: tuple[float, float] = (10, 5),
    repeat: bool = False,
    show: bool = False,
) -> FuncAnimation:
    """Progressively reveal two lines that share an x-axis."""
    validate_xy(x, y1)
    validate_xy(x, y2)
    a1, a2 = as_array(y1).astype(float), as_array(y2).astype(float)
    fig, ax = _anim_axes(
        x,
        float(min(a1.min(), a2.min())),
        float(max(a1.max(), a2.max())),
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        figsize=figsize,
    )
    (l1,) = ax.plot([], [], color=colors[0], label=labels[0])
    (l2,) = ax.plot([], [], color=colors[1], label=labels[1])
    ax.legend(loc="upper left")
    x_list = list(x)

    def update(frame):
        l1.set_data(x_list[:frame], a1[:frame])
        l2.set_data(x_list[:frame], a2[:frame])
        return l1, l2

    anim = FuncAnimation(fig, update, frames=len(x_list) + 1, interval=interval, blit=False, repeat=repeat)
    if show:
        plt.show()
    return anim


def animate_bar_chart(
    categories: Sequence[Any],
    values: Sequence[float],
    *,
    xlabel: str = "Category",
    ylabel: str = "Value",
    title: str = "Animated Bar Chart",
    color: str = "skyblue",
    interval: int = 200,
    figsize: tuple[float, float] = (10, 5),
    repeat: bool = False,
    show: bool = False,
) -> FuncAnimation:
    """Bars grow in one at a time."""
    validate_xy(categories, values)
    vals = as_array(values).astype(float)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(list(categories), np.zeros(len(vals)), color=color)
    ax.set_ylim(0, float(vals.max()) * 1.1 if vals.max() > 0 else 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45)

    def update(frame):
        for bar, val in zip(bars, vals[:frame]):
            bar.set_height(val)
        return bars

    anim = FuncAnimation(fig, update, frames=len(vals) + 1, interval=interval, blit=False, repeat=repeat)
    if show:
        plt.show()
    return anim


def animate_scatter_growth(
    x: Sequence[float],
    y: Sequence[float],
    *,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Animated Scatter Growth",
    color: str = "purple",
    interval: int = 200,
    figsize: tuple[float, float] = (10, 5),
    repeat: bool = False,
    show: bool = False,
) -> FuncAnimation:
    """Points appear one at a time."""
    validate_xy(x, y)
    xa, ya = as_array(x).astype(float), as_array(y).astype(float)
    fig, ax = _anim_axes(
        xa, float(ya.min()), float(ya.max()), xlabel=xlabel, ylabel=ylabel, title=title, figsize=figsize
    )
    sc = ax.scatter([], [], color=color, s=50, alpha=0.8)

    def update(frame):
        sc.set_offsets(np.c_[xa[:frame], ya[:frame]])
        return (sc,)

    anim = FuncAnimation(fig, update, frames=len(xa) + 1, interval=interval, blit=False, repeat=repeat)
    if show:
        plt.show()
    return anim


def save_animation(
    anim: FuncAnimation,
    filename: str | Path,
    *,
    fps: int = 10,
    dpi: int = 100,
    writer: str | None = None,
    verbose: bool = True,
) -> Path:
    """Write an animation to disk, falling back to an animated GIF when ffmpeg is unavailable.

    ``.mp4``/``.mov``/``.webm`` need the ``ffmpeg`` binary; ``.gif`` uses Pillow and always works.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if writer is None:
        if path.suffix.lower() == ".gif":
            writer = "pillow"
        elif shutil.which("ffmpeg") and mpl.animation.writers.is_available("ffmpeg"):
            writer = "ffmpeg"
        else:
            warnings.warn(
                f"ffmpeg not found; writing {path.with_suffix('.gif').name} with Pillow "
                f"instead of {path.name}.",
                RuntimeWarning,
                stacklevel=2,
            )
            path = path.with_suffix(".gif")
            writer = "pillow"
    anim.save(path, writer=writer, fps=fps, dpi=dpi)
    plt.close(anim._fig)
    if verbose:
        print(f"Saved animation to {path}")
    return path


def save_line_animation(
    x, y, filename: str | Path = "line_animation.mp4", dpi: int = 100, fps: int = 10
) -> Path:
    """Render :func:`animate_line_plot` to a file (see :func:`save_animation`)."""
    anim = animate_line_plot(x, y, xlabel="X", ylabel="Y", title="Line Animation Export", interval=100)
    return save_animation(anim, filename, fps=fps, dpi=dpi)


def save_dual_line_animation(
    x,
    y1,
    y2,
    labels: tuple[str, str] = ("Series A", "Series B"),
    filename: str | Path = "dual_line_animation.mp4",
    dpi: int = 100,
    fps: int = 10,
) -> Path:
    """Render :func:`animate_dual_line_plot` to a file."""
    anim = animate_dual_line_plot(
        x,
        y1,
        y2,
        labels=labels,
        xlabel="X",
        ylabel="Y",
        title="Dual Line Animation Export",
        interval=100,
        colors=("tab:blue", "tab:orange"),
    )
    return save_animation(anim, filename, fps=fps, dpi=dpi)


def save_bar_animation(
    categories, values, filename: str | Path = "bar_animation.mp4", dpi: int = 100, fps: int = 10
) -> Path:
    """Render :func:`animate_bar_chart` to a file."""
    anim = animate_bar_chart(
        categories,
        values,
        xlabel="",
        ylabel="Value",
        title="Bar Chart Animation Export",
        interval=100,
        color="tab:blue",
    )
    return save_animation(anim, filename, fps=fps, dpi=dpi)


def save_scatter_animation(
    x, y, filename: str | Path = "scatter_animation.mp4", dpi: int = 100, fps: int = 10
) -> Path:
    """Render :func:`animate_scatter_growth` to a file."""
    anim = animate_scatter_growth(
        x, y, xlabel="X", ylabel="Y", title="Scatter Animation Export", interval=100, color="tab:green"
    )
    return save_animation(anim, filename, fps=fps, dpi=dpi)


# =============================================================================
# Distribution / statistical summaries
# =============================================================================


def plot_histogram_with_stats(
    data: Sequence[float],
    bins: int | Sequence[float] | str = 10,
    title: str = "Histogram with Statistics",
    xlabel: str = "",
    ylabel: str = "Frequency",
    *,
    color: str = "skyblue",
    figsize: tuple[float, float] = (8, 5),
    ax: Axes | None = None,
    show: bool = False,
) -> FigAx:
    """Histogram annotated with mean, median and ±1 standard deviation."""
    arr = as_array(data).astype(float)
    if arr.size == 0:
        raise ValueError("data must be non-empty.")
    mean, median = arr.mean(), np.median(arr)
    std = arr.std(ddof=1) if arr.size > 1 else 0.0
    fig, ax = get_axes(ax, figsize)
    ax.hist(arr, bins=bins, color=color, edgecolor="black", alpha=0.7)
    ax.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.2f}")
    ax.axvline(median, color="green", linestyle="--", label=f"Median: {median:.2f}")
    ax.axvline(mean + std, color="orange", linestyle=":", label=f"+1 SD: {mean + std:.2f}")
    ax.axvline(mean - std, color="orange", linestyle=":", label=f"-1 SD: {mean - std:.2f}")
    return finalize(fig, ax, title=title, xlabel=xlabel, ylabel=ylabel, legend=True, show=show)


def _set_group_ticks(ax: Axes, names: Sequence[str]) -> None:
    ax.set_xticks(range(1, len(names) + 1), labels=list(names))


def plot_boxplot(
    data_dict: Mapping[str, Sequence[float]],
    title: str = "Boxplot Comparison",
    ylabel: str = "Value",
    *,
    figsize: tuple[float, float] = (8, 5),
    colors: Sequence[str] | None = None,
    showfliers: bool = True,
    ax: Axes | None = None,
    show: bool = False,
) -> FigAx:
    """Side-by-side box plots from a ``{group: values}`` mapping."""
    fig, ax = get_axes(ax, figsize)
    bp = ax.boxplot(list(data_dict.values()), patch_artist=True, showfliers=showfliers)
    if colors:
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
    _set_group_ticks(ax, list(data_dict.keys()))
    ax.grid(True, linestyle="--", alpha=0.6)
    return finalize(fig, ax, title=title, ylabel=ylabel, show=show)


def plot_violinplot(
    data_dict: Mapping[str, Sequence[float]],
    title: str = "Violin Plot",
    ylabel: str = "Value",
    *,
    figsize: tuple[float, float] = (8, 5),
    showmeans: bool = True,
    showmedians: bool = False,
    ax: Axes | None = None,
    show: bool = False,
) -> FigAx:
    """Side-by-side violin plots from a ``{group: values}`` mapping."""
    fig, ax = get_axes(ax, figsize)
    ax.violinplot(list(data_dict.values()), showmeans=showmeans, showmedians=showmedians)
    _set_group_ticks(ax, list(data_dict.keys()))
    ax.grid(True, linestyle="--", alpha=0.4)
    return finalize(fig, ax, title=title, ylabel=ylabel, show=show)


# =============================================================================
# Comparative plots
# =============================================================================


def stacked_area_plot(
    x: Sequence[Any],
    y_series_dict: Mapping[str, Sequence[float]],
    xlabel: str = "",
    ylabel: str = "",
    title: str = "Stacked Area Chart",
    *,
    figsize: tuple[float, float] = (10, 6),
    alpha: float = 0.8,
    rotate_xticks: float | None = 45,
    ax: Axes | None = None,
    show: bool = False,
) -> FigAx:
    """Stacked area chart from a ``{label: values}`` mapping."""
    ys = np.vstack([as_array(v) for v in y_series_dict.values()])
    fig, ax = get_axes(ax, figsize)
    ax.stackplot(x, ys, labels=list(y_series_dict.keys()), alpha=alpha)
    ax.grid(True, linestyle="--", alpha=0.5)
    return finalize(
        fig,
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=True,
        legend_kw={"loc": "upper left"},
        rotate_xticks=rotate_xticks,
        show=show,
    )


def subplot_groupwise(
    df: pd.DataFrame,
    group_col: str,
    x_col: str,
    y_col: str,
    title_prefix: str = "Group",
    *,
    ncols: int = 2,
    sharey: bool = False,
    show: bool = False,
) -> tuple[Figure, np.ndarray]:
    """One small-multiple panel per unique value of ``group_col``."""
    groups = list(pd.unique(df[group_col]))
    n = len(groups)
    if n == 0:
        raise ValueError("No groups to plot.")
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharey=sharey, squeeze=False)
    axes = axes.ravel()
    for ax, group in zip(axes, groups):
        sub = df[df[group_col] == group]
        ax.plot(sub[x_col], sub[y_col], marker="o")
        ax.set_title(f"{title_prefix} {group}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", labelrotation=45)
    for ax in axes[n:]:
        fig.delaxes(ax)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes[:n]


def save_grouped_bar_plot(
    df: pd.DataFrame,
    category: str,
    subcategory: str,
    value: str,
    title: str,
    filename: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    """Render a DataFrame grouped bar chart straight to ``filename``."""
    fig, _ = grouped_bar_plot(
        df=df, category=category, subcategory=subcategory, value=value, title=title, figsize=(12, 6)
    )
    return _save_and_close(fig, filename, dpi)


def save_stacked_area_plot(
    x, y_series_dict, xlabel: str, ylabel: str, title: str, filename: str | Path, *, dpi: int = 300
) -> Path:
    """Render a stacked area chart straight to ``filename``."""
    fig, _ = stacked_area_plot(x, y_series_dict, xlabel=xlabel, ylabel=ylabel, title=title, figsize=(12, 6))
    return _save_and_close(fig, filename, dpi)


def save_subplot_groupwise(
    df: pd.DataFrame,
    group_col: str,
    x_col: str,
    y_col: str,
    title_prefix: str,
    filename_prefix: str | Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Save one PNG per group as ``<filename_prefix>_<group>.png``."""
    prefix = Path(filename_prefix)
    written = []
    for group in pd.unique(df[group_col]):
        sub = df[df[group_col] == group]
        fig, _ = line_plot(
            sub[x_col],
            sub[y_col],
            title=f"{title_prefix} {group}",
            xlabel=x_col,
            ylabel=y_col,
            marker="o",
            figsize=(8, 4),
        )
        written.append(_save_and_close(fig, prefix.parent / f"{prefix.name}_{group}.png", dpi))
    return written


# =============================================================================
# Colormaps
# =============================================================================

_DEFAULT_CMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "coolwarm",
    "bwr",
    "seismic",
    "Pastel1",
    "Set1",
    "Set2",
    "Paired",
    "Dark2",
]


def display_colormap_samples(n: int = 256, cmaps: Sequence[str] | None = None, *, show: bool = False):
    """Horizontal gradient strips for a list of colormaps (sequential, diverging, qualitative)."""
    maps = list(cmaps) if cmaps else _DEFAULT_CMAPS
    fig, axes = plt.subplots(len(maps), 1, figsize=(8, 0.5 * len(maps)), squeeze=False)
    gradient = np.linspace(0, 1, n).reshape(1, -1)
    for ax, name in zip(axes.ravel(), maps):
        ax.imshow(gradient, aspect="auto", cmap=mpl.colormaps[name])
        ax.set_title(name, fontsize=9, loc="left")
        ax.set_axis_off()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes.ravel()


def save_sequential_bar_plot(
    months, values, filename: str | Path, cmap: str = "plasma", *, dpi: int = 150
) -> Path:
    """Bars coloured by position along a sequential colormap."""
    colors = mpl.colormaps[cmap](np.linspace(0, 1, len(months)))
    fig, ax = bar_plot(
        months,
        values,
        color=colors,
        edgecolor="none",
        figsize=(10, 5),
        title=f"Sequential Bar Plot — {cmap} colormap",
    )
    ax.tick_params(axis="x", labelrotation=45)
    return _save_and_close(fig, filename, dpi)


def save_diverging_change_plot(
    months, changes, filename: str | Path, cmap: str = "coolwarm", *, dpi: int = 150
) -> Path:
    """Bars coloured by a diverging colormap centred on zero."""
    changes = as_array(changes).astype(float)
    vmax = float(np.nanmax(np.abs(changes))) or 1.0
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    colors = mpl.colormaps[cmap](norm(changes))
    fig, ax = bar_plot(
        months,
        changes,
        color=colors,
        edgecolor="none",
        figsize=(10, 5),
        title=f"Diverging Change Plot — {cmap} colormap",
    )
    ax.axhline(0, color="gray", linestyle="--")
    ax.tick_params(axis="x", labelrotation=45)
    return _save_and_close(fig, filename, dpi)


def save_qualitative_grouped_bar(df: pd.DataFrame, filename: str | Path, *, dpi: int = 150) -> Path:
    """Units sold per product per month with the qualitative default colour cycle."""
    data = df.copy()
    data["_month"] = pd.to_datetime(data["Month"])
    pivot = (
        data.pivot_table(index="_month", columns="Product", values="Units Sold", aggfunc="sum")
        .sort_index()
        .fillna(0)
    )
    labels = [d.strftime("%b") for d in pivot.index]
    fig, _ = grouped_bar_plot(
        labels,
        {str(c): pivot[c].to_numpy() for c in pivot.columns},
        title="Units Sold per Product per Month — qualitative colours",
        xlabel="Month",
        ylabel="Units Sold",
        legend_title="Product",
        figsize=(10, 5),
    )
    return _save_and_close(fig, filename, dpi)


# =============================================================================
# Time series
# =============================================================================


def plot_timeseries_trend(
    x: Sequence[Any],
    y: Sequence[float],
    ylabel: str = "Value",
    title: str = "Time Series Trend",
    *,
    xlabel: str = "Date",
    marker: str | None = "o",
    figsize: tuple[float, float] = (10, 5),
    ax: Axes | None = None,
    show: bool = False,
) -> FigAx:
    """Basic time-series line with auto-formatted dates."""
    validate_xy(x, y)
    fig, ax = get_axes(ax, figsize)
    ax.plot(x, y, marker=marker, linestyle="-")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    return finalize(fig, ax, title=title, xlabel=xlabel, ylabel=ylabel, show=show)


def save_timeseries_trend(x, y, path: str | Path, *, dpi: int = 300, **kw: Any) -> Path:
    """Save :func:`plot_timeseries_trend` to ``path``."""
    fig, _ = plot_timeseries_trend(x, y, **kw)
    return _save_and_close(fig, path, dpi)


def plot_rolling_mean_std(
    series: pd.Series,
    window: int = 3,
    title: str = "Rolling Statistics",
    *,
    ylabel: str = "Value",
    figsize: tuple[float, float] = (10, 5),
    ax: Axes | None = None,
    show: bool = False,
) -> FigAx:
    """Series with its rolling mean and a ±1 rolling-SD band."""
    series = pd.Series(series)
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    fig, ax = get_axes(ax, figsize)
    ax.plot(series.index, series.to_numpy(), label="Original", color="tab:blue")
    ax.plot(series.index, rolling_mean.to_numpy(), label=f"Rolling mean ({window})", color="tab:orange")
    ax.fill_between(
        series.index,
        (rolling_mean - rolling_std).to_numpy(),
        (rolling_mean + rolling_std).to_numpy(),
        color="tab:orange",
        alpha=0.2,
        label="±1 rolling SD",
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    return finalize(fig, ax, title=title, xlabel="Date", ylabel=ylabel, legend=True, show=show)


def save_rolling_stats_plot(
    series: pd.Series, window: int = 3, path: str | Path | None = None, *, dpi: int = 300, **kw: Any
) -> Path:
    """Save :func:`plot_rolling_mean_std` to ``path``."""
    if path is None:
        raise ValueError("path is required.")
    fig, _ = plot_rolling_mean_std(series, window=window, **kw)
    return _save_and_close(fig, path, dpi)


def plot_multi_product_timeseries(
    df: pd.DataFrame,
    title: str = "Multiple Product Time Series",
    *,
    ylabel: str = "Value",
    figsize: tuple[float, float] = (10, 6),
    ax: Axes | None = None,
    show: bool = False,
) -> FigAx:
    """One line per column of a date-indexed DataFrame."""
    fig, ax = get_axes(ax, figsize)
    for col in df.columns:
        ax.plot(df.index, df[col], label=str(col))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    return finalize(
        fig,
        ax,
        title=title,
        xlabel="Date",
        ylabel=ylabel,
        legend=True,
        legend_kw={"loc": "upper left"},
        show=show,
    )


def save_multi_product_timeseries(df: pd.DataFrame, path: str | Path, *, dpi: int = 300, **kw: Any) -> Path:
    """Save :func:`plot_multi_product_timeseries` to ``path``."""
    fig, _ = plot_multi_product_timeseries(df, **kw)
    return _save_and_close(fig, path, dpi)


# =============================================================================
# Dashboards
# =============================================================================


def create_dashboard(
    df: pd.DataFrame,
    monthly: pd.DataFrame,
    save_path: str | Path | None = None,
    *,
    figsize: tuple[float, float] = (12, 8),
    dpi: int = 300,
    show: bool = False,
) -> tuple[Figure, np.ndarray]:
    """2×2 sales dashboard: monthly revenue, units over time, units histogram, units-vs-revenue.

    ``df`` needs ``"Units Sold"`` and ``"Revenue"``; ``monthly`` needs ``"Month"``,
    ``"Revenue"`` and ``"Units Sold"``.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs[0, 0].bar(monthly["Month"], monthly["Revenue"], color="teal")
    axs[0, 0].set_title("Monthly Revenue")
    axs[0, 0].tick_params(axis="x", rotation=45)
    axs[0, 1].plot(monthly["Month"], monthly["Units Sold"], marker="o", color="orange")
    axs[0, 1].set_title("Units Sold Over Time")
    axs[0, 1].tick_params(axis="x", rotation=45)
    axs[1, 0].hist(df["Units Sold"], bins=10, color="purple", edgecolor="white")
    axs[1, 0].set_title("Distribution of Units Sold")
    axs[1, 1].scatter(df["Units Sold"], df["Revenue"], color="crimson", alpha=0.7)
    axs[1, 1].set_title("Units Sold vs Revenue")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    return fig, axs
