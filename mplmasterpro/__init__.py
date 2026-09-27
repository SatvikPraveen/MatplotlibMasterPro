"""
MatplotlibMasterPro — a research-grade toolkit for publication-quality Matplotlib figures.

The package is organised into focused modules:

- :mod:`mplmasterpro.plot_utils`   — high-level plotting helpers (return ``(fig, ax)``)
- :mod:`mplmasterpro.theme_utils`  — reusable rcParams themes and a theme context manager
- :mod:`mplmasterpro.publication`  — journal figure sizes, panel labels, metadata-aware export
- :mod:`mplmasterpro.stats`        — uncertainty visualisation: bootstrap CIs, fits, raincloud plots
- :mod:`mplmasterpro.colors`       — colour-blind-safe palettes, CVD simulation, contrast metrics
- :mod:`mplmasterpro.datasets`     — deterministic synthetic datasets used throughout the notebooks

Every plotting function follows the same contract: it accepts an optional ``ax``
so it can be composed into larger layouts, never calls ``plt.show()`` on your
behalf, and returns the Matplotlib objects it created.
"""

from __future__ import annotations

from . import colors, datasets, plot_utils, publication, stats, theme_utils
from .colors import get_palette, set_palette, simulate_cvd
from .datasets import load_dataset
from .publication import add_panel_labels, figure_size, save_figure
from .theme_utils import apply_theme, list_themes, theme_context

__version__ = "1.0.0"
__author__ = "Satvik Praveen"
__email__ = "satvikpraveen707@gmail.com"
__license__ = "GPL-3.0-or-later"

__all__ = [
    "__version__",
    "add_panel_labels",
    "apply_theme",
    "colors",
    "datasets",
    "figure_size",
    "get_palette",
    "list_themes",
    "load_dataset",
    "plot_utils",
    "publication",
    "save_figure",
    "set_palette",
    "simulate_cvd",
    "stats",
    "theme_context",
    "theme_utils",
]
