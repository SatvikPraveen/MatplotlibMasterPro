#!/usr/bin/env python3
"""
Create Publication-Ready Figures

Generates IEEE and academic-style figures suitable for research papers.

Usage:
    python scripts/create_publication_figures.py

Output:
    exports/publication/*.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from utils.theme_utils import apply_publication_theme, apply_ieee_theme, apply_colorblind_friendly_theme

# Configuration
OUTPUT_DIR = Path("exports/publication")
DPI = 600  # Publication quality


def create_ieee_figure():
    """Create IEEE Transactions style figure."""
    apply_ieee_theme()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))  # IEEE column width
    
    # Left plot: Line graph with error bars
    x = np.linspace(0, 10, 20)
    y = 2 * x + 1 + np.random.randn(20) * 2
    y_smooth = 2 * x + 1
    errors = np.abs(np.random.randn(20) * 1.5)
    
    ax1.errorbar(x, y, yerr=errors, fmt='o', markersize=4, capsize=3, 
                 label='Measured', color='black', alpha=0.6)
    ax1.plot(x, y_smooth, '-', linewidth=1.5, label='Theoretical', color='red')
    ax1.set_xlabel('Input Parameter (units)')
    ax1.set_ylabel('Output Response (units)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontsize=10, 
             fontweight='bold', verticalalignment='top')
    
    # Right plot: Bar chart with comparison
    categories = ['Method A', 'Method B', 'Method C', 'Proposed']
    values = [78, 82, 85, 92]
    colors = ['gray', 'gray', 'gray', 'red']
    
    bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Performance (%)')
    ax2.set_ylim(70, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')
    ax2.tick_params(axis='x', rotation=15)
    ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontsize=10,
             fontweight='bold', verticalalignment='top')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    return fig, 'ieee_figure.pdf'


def create_academic_figure():
    """Create academic publication figure."""
    apply_publication_theme()
    
    fig = plt.figure(figsize=(6, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.35)
    
    # Top: Full-width plot
    ax1 = fig.add_subplot(gs[0, :])
    x = np.linspace(0, 4*np.pi, 200)
    y1 = np.sin(x)
    y2 = np.sin(2*x) / 2
    y3 = np.sin(3*x) / 3
    
    ax1.plot(x, y1, 'k-', linewidth=1.5, label='Fundamental')
    ax1.plot(x, y2, 'r--', linewidth=1.5, label='2nd Harmonic')
    ax1.plot(x, y3, 'b-.', linewidth=1.5, label='3rd Harmonic')
    ax1.fill_between(x, 0, y1+y2+y3, alpha=0.2, color='gray', label='Composite')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right', ncol=2, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=14, 
             fontweight='bold')
    
    # Bottom left: Scatter with regression
    ax2 = fig.add_subplot(gs[1, 0])
    np.random.seed(42)
    x_data = np.random.randn(50)
    y_data = 2*x_data + 1 + np.random.randn(50)*0.5
    
    ax2.scatter(x_data, y_data, alpha=0.6, s=30, color='steelblue', edgecolor='black', linewidth=0.5)
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    ax2.plot(sorted(x_data), p(sorted(x_data)), 'r--', linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
    ax2.set_xlabel('Variable X')
    ax2.set_ylabel('Variable Y')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold')
    
    # Bottom right: Box plot
    ax3 = fig.add_subplot(gs[1, 1])
    data = [np.random.normal(0, std, 100) for std in range(1, 5)]
    bp = ax3.boxplot(data, labels=['T1', 'T2', 'T3', 'T4'], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax3.set_ylabel('Response')
    ax3.set_xlabel('Treatment')
    ax3.grid(axis='y', alpha=0.3)
    ax3.text(-0.15, 1.05, 'C', transform=ax3.transAxes, fontsize=14, fontweight='bold')
    
    # Bottom: Full-width heatmap
    ax4 = fig.add_subplot(gs[2, :])
    data_matrix = np.random.rand(5, 10)
    im = ax4.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Feature')
    ax4.set_yticks(range(5))
    ax4.set_yticklabels([f'F{i+1}' for i in range(5)])
    cbar = plt.colorbar(im, ax=ax4, orientation='vertical', pad=0.02)
    cbar.set_label('Intensity', rotation=270, labelpad=15)
    ax4.text(-0.1, 1.05, 'D', transform=ax4.transAxes, fontsize=14, fontweight='bold')
    
    return fig, 'academic_figure.pdf'


def create_colorblind_friendly_figure():
    """Create colorblind-friendly visualization."""
    apply_colorblind_friendly_theme()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Line plot with distinct patterns
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.cos(x)
    
    ax1.plot(x, y1, '-', linewidth=2.5, label='Pattern A', color='#0173B2')
    ax1.plot(x, y2, '--', linewidth=2.5, label='Pattern B', color='#DE8F05')
    ax1.plot(x, y3, '-.', linewidth=2.5, label='Pattern C', color='#029E73')
    ax1.set_xlabel('Time (arbitrary units)')
    ax1.set_ylabel('Response')
    ax1.set_title('Colorblind-Friendly Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot with distinct markers
    np.random.seed(42)
    for i, (marker, color, label) in enumerate([('o', '#0173B2', 'Group 1'),
                                                  ('s', '#DE8F05', 'Group 2'),
                                                  ('^', '#029E73', 'Group 3')]):
        x_rand = np.random.randn(30) + i*3
        y_rand = np.random.randn(30) + i*2
        ax2.scatter(x_rand, y_rand, marker=marker, s=80, alpha=0.7,
                   color=color, edgecolor='black', linewidth=1, label=label)
    
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('Accessible Data Visualization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, 'colorblind_friendly_figure.pdf'


def main():
    """Main execution function."""
    print("=" * 60)
    print("📄 Publication Figure Generator")
    print("=" * 60)
    print(f"\nResolution: {DPI} DPI (publication quality)")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    generators = [
        ("IEEE Transactions Style", create_ieee_figure),
        ("Academic Multi-Panel", create_academic_figure),
        ("Colorblind-Friendly", create_colorblind_friendly_figure)
    ]
    
    for name, generator in generators:
        try:
            print(f"📊 Creating: {name}")
            fig, filename = generator()
            output_path = OUTPUT_DIR / filename
            fig.savefig(output_path, dpi=DPI, bbox_inches='tight', format='pdf')
            print(f"  ✓ Saved: {filename}\n")
            plt.close(fig)
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
    
    print("=" * 60)
    print("✅ Publication figures created successfully")
    print("=" * 60)
    
    print("\n💡 Usage Tips:")
    print("  - All figures are 600 DPI (suitable for print)")
    print("  - PDF format preserves vector graphics")
    print("  - Use IEEE style for journal submissions")
    print("  - Use colorblind-friendly for accessibility")


if __name__ == "__main__":
    main()
