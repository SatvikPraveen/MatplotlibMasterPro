# utils/theme_utils.py

import matplotlib.pyplot as plt
import matplotlib as mpl

def apply_dark_theme():
    """
    Apply a dark background theme for all plots.
    """
    plt.style.use('dark_background')
    mpl.rcParams.update({
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'text.color': 'white',
        'figure.facecolor': '#222222',
        'axes.facecolor': '#333333',
        'grid.color': '#555555',
        'axes.grid': True,
        'grid.linestyle': '--',
        'legend.frameon': False,
        'font.size': 12
    })

def apply_corporate_theme():
    """
    Apply a clean, presentation-friendly corporate style.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.labelcolor': '#333333',
        'axes.edgecolor': '#CCCCCC',
        'axes.grid': True,
        'grid.color': '#E0E0E0',
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'legend.frameon': False
    })

def apply_minimal_theme():
    """
    Apply a minimal style with no gridlines or distractions.
    """
    plt.style.use('default')
    mpl.rcParams.update({
        'axes.grid': False,
        'legend.frameon': False,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
    })

def reset_theme():
    """
    Reset to default matplotlib settings.
    """
    mpl.rcParams.update(mpl.rcParamsDefault)


def apply_publication_theme():
    """
    Apply a publication-ready theme suitable for academic papers.
    High-contrast, clean, suitable for black & white printing.
    """
    plt.style.use('classic')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'axes.linewidth': 1,
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.grid': True,
        'grid.color': '#CCCCCC',
        'grid.linestyle': ':',
        'grid.linewidth': 0.5,
        'legend.frameon': True,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })


def apply_colorblind_friendly_theme():
    """
    Apply a theme optimized for colorblind accessibility.
    Uses colorblind-friendly palette.
    """
    plt.style.use('seaborn-v0_8-colorblind')
    mpl.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'legend.frameon': False,
    })


def apply_high_contrast_theme():
    """
    Apply a high-contrast theme for presentations and projectors.
    """
    plt.style.use('default')
    mpl.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'axes.linewidth': 2,
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.grid': True,
        'grid.color': '#333333',
        'grid.linestyle': '--',
        'grid.linewidth': 1.5,
        'lines.linewidth': 3,
        'lines.markersize': 10,
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })


def apply_pastel_theme():
    """
    Apply a soft pastel theme with light colors.
    Great for reports and non-technical audiences.
    """
    plt.style.use('seaborn-v0_8-pastel')
    mpl.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': 'white',
        'legend.frameon': False
    })


def apply_ieee_theme():
    """
    IEEE-style publication theme.
    Follows IEEE Transactions formatting guidelines.
    """
    plt.style.use('classic')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'axes.linewidth': 0.5,
        'axes.grid': False,
        'legend.fontsize': 7,
        'legend.frameon': True,
        'legend.framealpha': 1.0,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'lines.linewidth': 1.5,
        'figure.figsize': (3.5, 2.625),  # IEEE single-column width
        'savefig.dpi': 600,
        'savefig.format': 'pdf'
    })


# Colorblind-friendly color palettes
COLORBLIND_PALETTE = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#CC78BC',
    'purple': '#CA9161',
    'brown': '#949494',
    'pink': '#ECE133',
    'gray': '#56B4E9'
}

def get_colorblind_palette():
    """
    Return a list of colorblind-friendly colors.
    """
    return list(COLORBLIND_PALETTE.values())

