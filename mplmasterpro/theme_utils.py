"""
Reusable Matplotlib themes.

Each ``apply_*_theme`` function mutates the global ``rcParams`` so that every
figure created afterwards picks up the style. For scoped styling prefer
:func:`theme_context`, which restores the previous settings on exit::

    with theme_context("ieee"):
        fig, ax = plt.subplots()
        ...

Available themes are listed by :func:`list_themes`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = [
    "COLORBLIND_PALETTE",
    "THEMES",
    "apply_colorblind_friendly_theme",
    "apply_corporate_theme",
    "apply_dark_theme",
    "apply_high_contrast_theme",
    "apply_ieee_theme",
    "apply_minimal_theme",
    "apply_nature_theme",
    "apply_pastel_theme",
    "apply_publication_theme",
    "apply_theme",
    "get_colorblind_palette",
    "list_themes",
    "reset_theme",
    "theme_context",
]

# Okabe & Ito (2008) — the de-facto standard colour-blind-safe qualitative palette.
COLORBLIND_PALETTE: dict[str, str] = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky_blue": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}


def reset_theme() -> None:
    """Restore Matplotlib's built-in defaults (the backend setting is left untouched)."""
    mpl.rcdefaults()
    plt.style.use("default")


def apply_dark_theme() -> None:
    """Dark background theme for slides and dashboards viewed on screens."""
    plt.style.use("dark_background")
    mpl.rcParams.update(
        {
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "text.color": "white",
            "figure.facecolor": "#222222",
            "axes.facecolor": "#333333",
            "grid.color": "#555555",
            "axes.grid": True,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "font.size": 12,
        }
    )


def apply_corporate_theme() -> None:
    """Clean, presentation-friendly style with light horizontal gridlines."""
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.labelcolor": "#333333",
            "axes.edgecolor": "#CCCCCC",
            "axes.grid": True,
            "grid.color": "#E0E0E0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def apply_minimal_theme() -> None:
    """Minimal style: open spines, faint grid, no legend frame."""
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


def apply_publication_theme() -> None:
    """Serif, high-contrast theme suitable for academic papers and greyscale printing."""
    plt.style.use("classic")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 1,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.grid": True,
            "grid.color": "#CCCCCC",
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "black",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def apply_colorblind_friendly_theme() -> None:
    """Okabe–Ito colour cycle with a light grid; safe for the common forms of CVD."""
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "axes.prop_cycle": mpl.cycler(color=list(COLORBLIND_PALETTE.values())),
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.frameon": False,
        }
    )


def apply_high_contrast_theme() -> None:
    """Large fonts, thick lines and strong gridlines for projectors and posters."""
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "axes.linewidth": 2,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.grid": True,
            "grid.color": "#333333",
            "grid.linestyle": "--",
            "grid.linewidth": 1.5,
            "lines.linewidth": 3,
            "lines.markersize": 10,
            "legend.fontsize": 12,
            "legend.frameon": True,
            "legend.edgecolor": "black",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def apply_pastel_theme() -> None:
    """Soft pastel palette for reports aimed at non-technical audiences."""
    plt.style.use("seaborn-v0_8-pastel")
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.4,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "axes.facecolor": "#FAFAFA",
            "figure.facecolor": "white",
            "legend.frameon": False,
        }
    )


def apply_ieee_theme() -> None:
    """IEEE Transactions style: 3.5 in single-column figures, 8 pt serif type, 600 dpi."""
    plt.style.use("classic")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.linewidth": 0.5,
            "axes.grid": False,
            "legend.fontsize": 7,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "figure.figsize": (3.5, 2.625),  # IEEE single-column width, 4:3
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 600,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,  # embed TrueType so text stays editable in Illustrator
            "ps.fonttype": 42,
        }
    )


def apply_nature_theme() -> None:
    """Nature-family style: 89 mm single-column figures, 7 pt sans-serif type, no grid."""
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "legend.fontsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            "figure.figsize": (3.504, 2.336),  # 89 mm × 59 mm
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.prop_cycle": mpl.cycler(color=list(COLORBLIND_PALETTE.values())),
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


THEMES: dict[str, Callable[[], None]] = {
    "default": reset_theme,
    "dark": apply_dark_theme,
    "corporate": apply_corporate_theme,
    "minimal": apply_minimal_theme,
    "publication": apply_publication_theme,
    "colorblind": apply_colorblind_friendly_theme,
    "high_contrast": apply_high_contrast_theme,
    "pastel": apply_pastel_theme,
    "ieee": apply_ieee_theme,
    "nature": apply_nature_theme,
}


def list_themes() -> list[str]:
    """Return the names accepted by :func:`apply_theme` and :func:`theme_context`."""
    return list(THEMES)


def apply_theme(name: str) -> None:
    """Apply a theme by name (see :func:`list_themes`)."""
    try:
        THEMES[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown theme {name!r}. Available: {list_themes()}") from exc


@contextmanager
def theme_context(name: str) -> Iterator[None]:
    """Apply a theme inside a ``with`` block and restore the previous rcParams afterwards."""
    with mpl.rc_context():
        apply_theme(name)
        yield


def get_colorblind_palette() -> list[str]:
    """Return the Okabe–Ito palette as a list of hex colours."""
    return list(COLORBLIND_PALETTE.values())
