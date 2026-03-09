#!/usr/bin/env python3
"""
Batch Export Plots in Multiple Formats

Exports the same plot in PNG, PDF, and SVG formats for different use cases.

Usage:
    python scripts/batch_export.py

Output:
    exports/batch/*.{png,pdf,svg}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.theme_utils import apply_publication_theme

# Configuration
OUTPUT_DIR = Path("exports/batch")
FORMATS = ['png', 'pdf', 'svg']
DPI = 600  # High resolution for publication


def create_sample_plot():
    """Create a sample publication-quality plot."""
    apply_publication_theme()
    
    # Generate data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.sin(x) * np.exp(-x/10)
    y3 = np.cos(x) * np.exp(-x/10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x, y1, 'b-', linewidth=2, label='sin(x)', alpha=0.8)
    ax.plot(x, y2, 'r--', linewidth=2, label='sin(x)·exp(-x/10)', alpha=0.8)
    ax.plot(x, y3, 'g-.', linewidth=2, label='cos(x)·exp(-x/10)', alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax.set_title('Damped Oscillations', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return fig


def create_data_visualization():
    """Create a data-driven visualization."""
    apply_publication_theme()
    
    # Create sample data
    categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    values1 = [23, 45, 56, 78, 32]
    values2 = [34, 38, 49, 61, 44]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, values1, width, label='Dataset 1', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, values2, width, label='Dataset 2', color='coral', alpha=0.8)
    
    ax.set_xlabel('Categories', fontsize=12, fontweight='bold')
    ax.set_ylabel('Values', fontsize=12, fontweight='bold')
    ax.set_title('Comparative Analysis', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    return fig


def export_plot(fig, basename, formats=FORMATS):
    """Export plot in multiple formats."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    exported = []
    for fmt in formats:
        output_path = OUTPUT_DIR / f"{basename}.{fmt}"
        
        # Format-specific parameters
        save_params = {
            'bbox_inches': 'tight',
            'format': fmt
        }
        
        if fmt == 'png':
            save_params['dpi'] = DPI
            save_params['facecolor'] = 'white'
        elif fmt == 'pdf':
            save_params['dpi'] = DPI
        elif fmt == 'svg':
            save_params['transparent'] = True
        
        fig.savefig(output_path, **save_params)
        exported.append(output_path)
        print(f"  ✓ {fmt.upper()}: {output_path.name}")
    
    return exported


def main():
    """Main execution function."""
    print("=" * 60)
    print("📦 Batch Export Tool")
    print("=" * 60)
    print(f"\nExport formats: {', '.join(FORMATS)}")
    print(f"Resolution: {DPI} DPI")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    plots_to_export = [
        ("damped_oscillations", create_sample_plot),
        ("comparative_analysis", create_data_visualization)
    ]
    
    total_files = 0
    
    for basename, generator in plots_to_export:
        try:
            print(f"\n📊 Processing: {basename}")
            fig = generator()
            exported = export_plot(fig, basename)
            total_files += len(exported)
            plt.close(fig)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ Exported {total_files} files ({len(plots_to_export)} plots × {len(FORMATS)} formats)")
    print("=" * 60)
    
    print("\n💡 Format Guidelines:")
    print("  PNG  - Web, presentations, quick sharing")
    print("  PDF  - Publications, print, LaTeX documents")
    print("  SVG  - Vector graphics, scaling, web (modern)")


if __name__ == "__main__":
    main()
