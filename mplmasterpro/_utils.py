"""Internal helpers shared across the plotting modules."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = ["as_array", "ensure_dir", "finalize", "get_axes", "validate_xy"]


def get_axes(
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    **subplot_kw: Any,
) -> tuple[Figure, Axes]:
    """Return ``(fig, ax)``, creating a new figure only when ``ax`` is ``None``.

    This is the composability primitive used by every plotting helper: pass an
    existing ``ax`` to draw into a subplot of a larger layout, or omit it to get
    a standalone figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, **subplot_kw)
    else:
        fig = ax.figure
    return fig, ax


def finalize(
    fig: Figure,
    ax: Axes,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    grid: bool | None = None,
    grid_kw: dict[str, Any] | None = None,
    legend: bool = False,
    legend_kw: dict[str, Any] | None = None,
    rotate_xticks: float | None = None,
    tight: bool = True,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Apply labels, grid, legend and layout, then optionally call ``plt.show()``."""
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid is not None:
        ax.grid(grid, **(grid_kw or {}))
    if legend:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(**(legend_kw or {}))
    if rotate_xticks is not None:
        ax.tick_params(axis="x", labelrotation=rotate_xticks)
    if tight:
        with contextlib.suppress(Exception):  # layout engines can refuse (e.g. constrained)
            fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def as_array(values: Any) -> np.ndarray:
    """Convert pandas/list inputs to a NumPy array without copying when possible."""
    return np.asarray(values)


def validate_xy(x: Sequence[Any], y: Sequence[Any], *, allow_empty: bool = False) -> None:
    """Raise ``ValueError`` for empty or length-mismatched ``x``/``y`` inputs."""
    nx, ny = len(x), len(y)
    if not allow_empty and (nx == 0 or ny == 0):
        raise ValueError("x and y must be non-empty sequences.")
    if nx != ny:
        raise ValueError(f"x and y must have the same length (got {nx} and {ny}).")


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (a directory) if needed and return it as a :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
