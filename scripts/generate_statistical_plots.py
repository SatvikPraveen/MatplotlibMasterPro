#!/usr/bin/env python3
"""
Generate Statistical Analysis Plots

Creates box plots, violin plots, and statistical comparisons for data analysis.

Usage:
    python scripts/generate_statistical_plots.py

Output:
    exports/statistical_analysis/*.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from utils.theme_utils import apply_minimal_theme

# Configuration
OUTPUT_DIR = Path("exports/statistical_analysis")
DPI = 300


def generate_sample_data():
    """Generate sample datasets for statistical analysis."""
    np.random.seed(42)
    
    data = {
        'Group A': np.random.normal(100, 15, 200),
        'Group B': np.random.normal(110, 20, 200),
        'Group C': np.random.normal(95, 10, 200),
        'Group D': np.random.normal(105, 25, 200)
    }
    
    return data


def create_box_plots(data):
    """Generate comprehensive box plot comparison."""
    apply_minimal_theme()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Vertical box plot
    bp1 = ax1.boxplot(data.values(), labels=data.keys(), patch_artist=True,
                       notch=True, showmeans=True, meanline=True)
    
    # Customize colors
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Values', fontweight='bold', fontsize=12)
    ax1.set_title('Box Plot Comparison (Vertical)', fontweight='bold', fontsize=14, pad=15)
    ax1.grid(axis='y', alpha=0.3)
    
    # Horizontal box plot
    bp2 = ax2.boxplot(data.values(), labels=data.keys(), patch_artist=True,
                       vert=False, showmeans=True)
    
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Values', fontweight='bold', fontsize=12)
    ax2.set_title('Box Plot Comparison (Horizontal)', fontweight='bold', fontsize=14, pad=15)
    ax2.grid(axis='x', alpha=0.3)
    
    fig.suptitle('📊 Statistical Distribution Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig, 'box_plots.png'


def create_violin_plots(data):
    """Generate violin plot showing density distributions."""
    apply_minimal_theme()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    parts = ax.violinplot(data.values(), positions=range(1, len(data)+1),
                          showmeans=True, showmedians=True, showextrema=True)
    
    # Customize violin colors
    colors = ['blue', 'green', 'red', 'orange']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    ax.set_xticks(range(1, len(data)+1))
    ax.set_xticklabels(data.keys())
    ax.set_ylabel('Values', fontweight='bold', fontsize=12)
    ax.set_title('🎻 Violin Plot: Distribution Density', fontweight='bold', fontsize=14, pad=15)
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    ax.text(0.02, 0.98, 'Wider sections = Higher density\nWhite dot = Median\nThick bar = IQR',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return fig, 'violin_plots.png'


def create_combined_plot(data):
    """Create combined box and violin plot."""
    apply_minimal_theme()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Violin plot as base
    parts = ax.violinplot(data.values(), positions=range(1, len(data)+1),
                          showmeans=False, showmedians=False, showextrema=False)
    
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.3)
        pc.set_edgecolor('blue')
    
    # Overlay box plot
    bp = ax.boxplot(data.values(), positions=range(1, len(data)+1),
                    widths=0.3, patch_artist=True, showmeans=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_alpha(0.8)
        patch.set_edgecolor('darkblue')
        patch.set_linewidth(2)
    
    ax.set_xticks(range(1, len(data)+1))
    ax.set_xticklabels(data.keys())
    ax.set_ylabel('Values', fontweight='bold', fontsize=12)
    ax.set_title('📊 Combined: Violin + Box Plot', fontweight='bold', fontsize=14, pad=15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return fig, 'combined_violin_box.png'


def create_summary_statistics(data):
    """Create statistical summary visualization."""
    apply_minimal_theme()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate statistics
    groups = list(data.keys())
    means = [np.mean(data[g]) for g in groups]
    stds = [np.std(data[g]) for g in groups]
    medians = [np.median(data[g]) for g in groups]
    
    x = np.arange(len(groups))
    width = 0.25
    
    # Create grouped bar chart
    ax.bar(x - width, means, width, label='Mean', color='steelblue', alpha=0.8)
    ax.bar(x, stds, width, label='Std Dev', color='orange', alpha=0.8)
    ax.bar(x + width, medians, width, label='Median', color='green', alpha=0.8)
    
    # Add error bars on means
    ax.errorbar(x - width, means, yerr=stds, fmt='none', ecolor='black',
                capsize=5, capthick=2, alpha=0.5)
    
    ax.set_xlabel('Groups', fontweight='bold', fontsize=12)
    ax.set_ylabel('Values', fontweight='bold', fontsize=12)
    ax.set_title('📈 Statistical Summary', fontweight='bold', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return fig, 'statistical_summary.png'


def main():
    """Main execution function."""
    print("=" * 60)
    print("📊 Statistical Plots Generator")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate sample data
    print("\nGenerating sample data...")
    data = generate_sample_data()
    print(f"✓ Created {len(data)} groups with 200 samples each")
    
    # Generate plots
    generators = [
        lambda: create_box_plots(data),
        lambda: create_violin_plots(data),
        lambda: create_combined_plot(data),
        lambda: create_summary_statistics(data)
    ]
    
    for generator in generators:
        try:
            fig, filename = generator()
            output_path = OUTPUT_DIR / filename
            fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {filename}")
            plt.close(fig)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ Generated {len(generators)} statistical plots")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
