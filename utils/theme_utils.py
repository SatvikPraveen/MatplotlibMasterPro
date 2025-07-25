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
