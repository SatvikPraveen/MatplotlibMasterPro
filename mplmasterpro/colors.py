"""
Colour tooling for accessible, quantitatively-checked figures.

* Curated palettes: Okabe–Ito, Paul Tol (bright / muted / vibrant / light) and IBM.
* :func:`simulate_cvd` applies the Machado, Oliveira & Fernandes (2009) matrices to
  preview how colours appear under protanopia, deuteranopia or tritanopia.
* :func:`contrast_ratio` (WCAG 2.1) and :func:`min_pairwise_distance` (CIE76 ΔE)
  give numbers you can put in a methods section instead of "we used nice colours".
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from itertools import combinations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure

__all__ = [
    "IBM_COLORBLIND",
    "OKABE_ITO",
    "PALETTES",
    "TOL_BRIGHT",
    "TOL_LIGHT",
    "TOL_MUTED",
    "TOL_VIBRANT",
    "contrast_ratio",
    "cvd_preview",
    "delta_e",
    "get_palette",
    "min_pairwise_distance",
    "palette_context",
    "preview_palette",
    "relative_luminance",
    "rgb_to_lab",
    "set_palette",
    "simulate_cvd",
    "text_color_for",
]

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
TOL_BRIGHT = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]
TOL_MUTED = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#DDDDDD",
]
TOL_VIBRANT = ["#EE7733", "#0077BB", "#33BBEE", "#EE3377", "#CC3311", "#009988", "#BBBBBB"]
TOL_LIGHT = [
    "#77AADD",
    "#EE8866",
    "#EEDD88",
    "#FFAABB",
    "#99DDFF",
    "#44BB99",
    "#BBCC33",
    "#AAAA00",
    "#DDDDDD",
]
IBM_COLORBLIND = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]

PALETTES: dict[str, list[str]] = {
    "okabe_ito": OKABE_ITO,
    "tol_bright": TOL_BRIGHT,
    "tol_muted": TOL_MUTED,
    "tol_vibrant": TOL_VIBRANT,
    "tol_light": TOL_LIGHT,
    "ibm": IBM_COLORBLIND,
    "tab10": [to_hex(c) for c in plt.get_cmap("tab10").colors],
}

# Machado, Oliveira & Fernandes (2009), severity = 1.0, applied in linear RGB.
_CVD_MATRICES: dict[str, np.ndarray] = {
    "protanopia": np.array(
        [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]
    ),
    "deuteranopia": np.array(
        [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]
    ),
    "tritanopia": np.array(
        [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]
    ),
}


def get_palette(name: str = "okabe_ito", n: int | None = None) -> list[str]:
    """Return a named palette as hex strings, optionally cycled/truncated to ``n`` colours."""
    try:
        base = PALETTES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown palette {name!r}. Available: {sorted(PALETTES)}") from exc
    if n is None:
        return list(base)
    return [base[i % len(base)] for i in range(n)]


def set_palette(name_or_colors: str | Sequence[str] = "okabe_ito") -> list[str]:
    """Make a palette the global colour cycle (``axes.prop_cycle``) and return it."""
    colors = get_palette(name_or_colors) if isinstance(name_or_colors, str) else list(name_or_colors)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)
    return colors


@contextmanager
def palette_context(name_or_colors: str | Sequence[str]) -> Iterator[list[str]]:
    """Temporarily set the colour cycle inside a ``with`` block."""
    with mpl.rc_context():
        yield set_palette(name_or_colors)


# --------------------------------------------------------------------------- colour science


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(np.asarray(lin, dtype=float), 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def _to_rgb_array(colors: str | Sequence | np.ndarray) -> np.ndarray:
    if isinstance(colors, str):
        return np.array([to_rgb(colors)])
    arr = np.asarray(colors)
    if arr.dtype.kind in "USO":  # sequence of colour specs
        return np.array([to_rgb(c) for c in colors])
    if arr.ndim == 1 and arr.shape[0] in (3, 4):
        return arr[None, :3].astype(float)
    return arr[..., :3].astype(float)


def simulate_cvd(colors: str | Sequence | np.ndarray, kind: str = "deuteranopia") -> list[str] | np.ndarray:
    """Simulate how ``colors`` look under a colour-vision deficiency.

    Parameters
    ----------
    colors
        A colour spec, a sequence of colour specs, or an ``(..., 3|4)`` float RGB(A) array.
    kind
        ``"protanopia"``, ``"deuteranopia"`` or ``"tritanopia"``.

    Returns
    -------
    list[str] | numpy.ndarray
        Hex strings when given colour specs; an RGB float array when given an array
        (an image, for example), so you can inspect a whole figure with ``imshow``.
    """
    try:
        matrix = _CVD_MATRICES[kind]
    except KeyError as exc:
        raise ValueError(f"kind must be one of {sorted(_CVD_MATRICES)}") from exc
    rgb = _to_rgb_array(colors)
    lin = _srgb_to_linear(rgb)
    sim = _linear_to_srgb(lin @ matrix.T)
    if isinstance(colors, np.ndarray) and colors.dtype.kind == "f":
        return sim
    return [to_hex(c) for c in sim.reshape(-1, 3)]


def relative_luminance(color: str | Sequence[float]) -> float:
    """WCAG 2.1 relative luminance of an sRGB colour (0 = black, 1 = white)."""
    r, g, b = _srgb_to_linear(np.asarray(to_rgb(color)))
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def contrast_ratio(color_a: str | Sequence[float], color_b: str | Sequence[float]) -> float:
    """WCAG 2.1 contrast ratio between two colours (1:1 identical … 21:1 black/white).

    Body text needs ≥ 4.5, large text and graphical objects ≥ 3.0.
    """
    la, lb = relative_luminance(color_a), relative_luminance(color_b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def text_color_for(background: str | Sequence[float], light: str = "white", dark: str = "black") -> str:
    """Pick the higher-contrast text colour (``light`` or ``dark``) for a background."""
    return light if contrast_ratio(background, light) >= contrast_ratio(background, dark) else dark


def rgb_to_lab(color: str | Sequence[float]) -> np.ndarray:
    """Convert an sRGB colour to CIE L*a*b* (D65 reference white)."""
    lin = _srgb_to_linear(np.asarray(to_rgb(color)))
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = m @ lin / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[1] - 16
    a = 500 * (f[0] - f[1])
    b = 200 * (f[1] - f[2])
    return np.array([L, a, b])


def delta_e(color_a: str | Sequence[float], color_b: str | Sequence[float]) -> float:
    """CIE76 colour difference ΔE*ab between two colours (≈2.3 is a just-noticeable difference)."""
    return float(np.linalg.norm(rgb_to_lab(color_a) - rgb_to_lab(color_b)))


def min_pairwise_distance(colors: Sequence[str], cvd: str | None = None) -> tuple[float, tuple[str, str]]:
    """Smallest ΔE between any two palette colours, optionally after CVD simulation.

    Returns ``(distance, (color_i, color_j))`` for the closest pair, so you can report
    e.g. "all colours differ by ΔE ≥ 20 under deuteranopia".
    """
    colors = list(colors)
    if len(colors) < 2:
        raise ValueError("Need at least two colours.")
    test = simulate_cvd(colors, cvd) if cvd else colors
    best = (np.inf, (colors[0], colors[1]))
    for (i, ca), (j, cb) in combinations(enumerate(test), 2):
        d = delta_e(ca, cb)
        if d < best[0]:
            best = (d, (colors[i], colors[j]))
    return best


# --------------------------------------------------------------------------- previews


def preview_palette(
    colors: Sequence[str] | str,
    *,
    ax: Axes | None = None,
    labels: bool = True,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Draw colour swatches for a palette (name or list of colours)."""
    palette = get_palette(colors) if isinstance(colors, str) else list(colors)
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, 0.9 * len(palette)), 1.2))
    else:
        fig = ax.figure
    for i, c in enumerate(palette):
        ax.add_patch(mpl.patches.Rectangle((i, 0), 1, 1, facecolor=c, edgecolor="none"))
        if labels:
            ax.text(i + 0.5, 0.5, c, ha="center", va="center", fontsize=7, color=text_color_for(c))
    ax.set_xlim(0, len(palette))
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9, loc="left")
    return fig, ax


def cvd_preview(colors: Sequence[str] | str, *, figsize: tuple[float, float] | None = None) -> Figure:
    """Show a palette as seen with normal vision and the three dichromatic deficiencies."""
    palette = get_palette(colors) if isinstance(colors, str) else list(colors)
    kinds = ["normal", "protanopia", "deuteranopia", "tritanopia"]
    fig, axes = plt.subplots(len(kinds), 1, figsize=figsize or (max(4, 0.9 * len(palette)), 4.5))
    for ax, kind in zip(axes, kinds):
        cols = palette if kind == "normal" else simulate_cvd(palette, kind)
        preview_palette(cols, ax=ax, labels=False, title=kind)
    fig.tight_layout()
    return fig
