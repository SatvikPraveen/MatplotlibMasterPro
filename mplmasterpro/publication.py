"""
Helpers for camera-ready figures.

* :func:`figure_size` — exact widths for common journal templates (IEEE, Nature, Science, …)
* :func:`set_size` — size a figure from a LaTeX ``\\columnwidth`` measured in points
* :func:`add_panel_labels` — "(a)", "(b)" … labels placed consistently across subplots
* :func:`save_figure` — multi-format export with embedded provenance metadata (git hash, author)
* :func:`despine`, :func:`set_math_fonts` — small finishing touches
"""

from __future__ import annotations

import datetime as _dt
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "GOLDEN_RATIO",
    "JOURNAL_WIDTHS_MM",
    "add_panel_labels",
    "despine",
    "figure_size",
    "git_revision",
    "mm_to_inches",
    "save_figure",
    "set_math_fonts",
    "set_size",
]

GOLDEN_RATIO = (5**0.5 - 1) / 2  # ≈ 0.618, height / width

# Column widths in millimetres: (single, double). Sources: publisher author guidelines.
JOURNAL_WIDTHS_MM: dict[str, tuple[float, float]] = {
    "ieee": (88.9, 181.9),  # 3.5 in / 7.16 in
    "acm": (84.6, 177.8),  # 3.33 in / 7 in
    "aps": (86.0, 178.0),  # Physical Review
    "nature": (89.0, 183.0),
    "science": (55.0, 120.0),  # 2- and 3-column layout uses 55 / 120 / 183 mm
    "elsevier": (90.0, 190.0),
    "pnas": (87.0, 178.0),
    "plos": (83.0, 173.0),
    "springer": (84.0, 174.0),
    "a4_text": (160.0, 160.0),  # a typical single-column A4 text width
    "letter_text": (165.1, 165.1),  # 6.5 in
}


def mm_to_inches(mm: float) -> float:
    """Convert millimetres to inches."""
    return mm / 25.4


def figure_size(
    journal: str = "ieee",
    columns: int | float = 1,
    *,
    aspect: float = GOLDEN_RATIO,
    height_mm: float | None = None,
) -> tuple[float, float]:
    """Return ``(width, height)`` in inches for a journal column width.

    Parameters
    ----------
    journal
        Key of :data:`JOURNAL_WIDTHS_MM`.
    columns
        ``1`` for single-column, ``2`` for full width. ``1.5`` interpolates linearly.
    aspect
        ``height / width``; defaults to the golden ratio.
    height_mm
        Explicit height in millimetres (overrides ``aspect``).
    """
    try:
        single, double = JOURNAL_WIDTHS_MM[journal]
    except KeyError as exc:
        raise ValueError(f"Unknown journal {journal!r}. Available: {sorted(JOURNAL_WIDTHS_MM)}") from exc
    if columns == 1:
        width_mm = single
    elif columns == 2:
        width_mm = double
    else:
        width_mm = single + (double - single) * (float(columns) - 1)
    height = mm_to_inches(height_mm) if height_mm is not None else mm_to_inches(width_mm) * aspect
    return (round(mm_to_inches(width_mm), 3), round(height, 3))


def set_size(
    width_pt: float,
    fraction: float = 1.0,
    subplots: tuple[int, int] = (1, 1),
    *,
    aspect: float = GOLDEN_RATIO,
) -> tuple[float, float]:
    """Figure size in inches from a LaTeX length in points (``\\showthe\\columnwidth``).

    ``fraction`` is the share of the text width to use, and ``subplots=(rows, cols)``
    scales the height so each panel keeps the requested aspect ratio.
    """
    width_in = width_pt * fraction / 72.27
    height_in = width_in * aspect * (subplots[0] / subplots[1])
    return (width_in, height_in)


def add_panel_labels(
    axes: Iterable[Axes] | np.ndarray | Axes,
    labels: Sequence[str] | None = None,
    *,
    style: str = "(a)",
    loc: tuple[float, float] = (-0.12, 1.04),
    fontsize: float | None = None,
    fontweight: str = "bold",
    **text_kw: Any,
) -> list[Any]:
    """Label subplots ``(a)``, ``(b)``, … in axes coordinates.

    ``style`` is a template applied to a lowercase letter: ``"(a)"``, ``"a"``, ``"A"``,
    ``"a)"`` or ``"A."`` are all understood. Returns the created ``Text`` artists.
    """
    ax_list = [axes] if isinstance(axes, Axes) else list(np.ravel(np.asarray(axes, dtype=object)))
    if labels is None:
        letters = [chr(ord("a") + i) for i in range(len(ax_list))]
        upper = any(ch.isupper() for ch in style)
        labels = [style.replace("A" if upper else "a", ch.upper() if upper else ch) for ch in letters]
    artists = []
    for ax, label in zip(ax_list, labels):
        artists.append(
            ax.text(
                loc[0],
                loc[1],
                label,
                transform=ax.transAxes,
                fontsize=fontsize or mpl.rcParams["axes.titlesize"],
                fontweight=fontweight,
                va="bottom",
                ha="left",
                **text_kw,
            )
        )
    return artists


def git_revision(short: bool = True) -> str | None:
    """Return the current git commit hash, or ``None`` outside a repository."""
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


_PDF_KEYS = {
    "Title",
    "Author",
    "Subject",
    "Keywords",
    "Creator",
    "Producer",
    "CreationDate",
    "ModDate",
    "Trapped",
}
_SVG_KEYS = {
    "Title",
    "Creator",
    "Date",
    "Description",
    "Keywords",
    "Publisher",
    "Rights",
    "Source",
    "Type",
    "Coverage",
    "Contributor",
    "Format",
    "Identifier",
    "Language",
    "Relation",
}


def _metadata_for(fmt: str, meta: dict[str, str]) -> dict[str, str]:
    if fmt == "pdf":
        return {k: v for k, v in meta.items() if k in _PDF_KEYS}
    if fmt == "svg":
        out = {k: v for k, v in meta.items() if k in _SVG_KEYS}
        if "Subject" in meta and "Description" not in out:
            out["Description"] = meta["Subject"]
        return out
    if fmt == "png":
        return meta
    return {}


def save_figure(
    fig: Figure,
    path: str | Path,
    formats: Sequence[str] = ("pdf", "png"),
    *,
    dpi: int = 300,
    metadata: dict[str, str] | None = None,
    include_git_hash: bool = True,
    transparent: bool = False,
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.02,
    close: bool = False,
    **savefig_kw: Any,
) -> list[Path]:
    """Save ``fig`` once per format with embedded provenance metadata.

    ``path`` may or may not carry a suffix; the suffix is replaced by each entry in
    ``formats``. Metadata (title, author, creator, ISO timestamp and, when available,
    the git commit) is written into PDF/PNG/SVG headers so a figure can always be
    traced back to the code that produced it. Returns the written paths.
    """
    from . import __version__  # local import to avoid a cycle at module import time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    known = {f.lower().lstrip(".") for f in formats} | {"png", "pdf", "svg", "eps", "jpg", "tif", "tiff"}
    base = path.with_suffix("") if path.suffix.lower().lstrip(".") in known else path

    meta: dict[str, str] = {
        "Creator": f"mplmasterpro {__version__} / matplotlib {mpl.__version__}",
        "Date": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
    title = fig._suptitle.get_text() if getattr(fig, "_suptitle", None) else None
    if not title and fig.axes:
        title = fig.axes[0].get_title()
    if title:
        meta["Title"] = title
    if include_git_hash and (rev := git_revision()):
        meta["Subject"] = f"git:{rev}"
    if metadata:
        meta.update(metadata)

    written = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        out = base.parent / f"{base.name}.{fmt}"
        fig.savefig(
            out,
            format=fmt,
            dpi=dpi,
            transparent=transparent,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            metadata=_metadata_for(fmt, meta) or None,
            **savefig_kw,
        )
        written.append(out)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return written


def despine(
    ax: Axes | Iterable[Axes],
    *,
    top: bool = True,
    right: bool = True,
    left: bool = False,
    bottom: bool = False,
    offset: float | None = None,
) -> None:
    """Hide selected spines (Tufte-style), optionally offsetting the remaining ones outward."""
    axes = [ax] if isinstance(ax, Axes) else list(np.ravel(np.asarray(ax, dtype=object)))
    for a in axes:
        for side, hide in (("top", top), ("right", right), ("left", left), ("bottom", bottom)):
            a.spines[side].set_visible(not hide)
            if not hide and offset is not None:
                a.spines[side].set_position(("outward", offset))


def set_math_fonts(fontset: str = "stix", usetex: bool = False, preamble: str | None = None) -> None:
    """Choose the math text engine: a bundled mathtext font set or full LaTeX rendering.

    ``fontset`` is one of ``"dejavusans"``, ``"dejavuserif"``, ``"cm"``, ``"stix"``,
    ``"stixsans"``. ``usetex=True`` requires a working TeX installation.
    """
    mpl.rcParams["mathtext.fontset"] = fontset
    mpl.rcParams["text.usetex"] = usetex
    if preamble is not None:
        mpl.rcParams["text.latex.preamble"] = preamble
