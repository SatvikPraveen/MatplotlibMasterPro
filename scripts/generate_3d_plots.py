#!/usr/bin/env python3
"""
Generate 3D Visualization Suite

Batch generates various 3D plots for scientific and technical visualization.

Usage:
    python scripts/generate_3d_plots.py

Output:
    exports/3d_visualizations/*.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Configuration
OUTPUT_DIR = Path("exports/3d_visualizations")
DPI = 300


def create_3d_surface():
    """Generate 3D surface plot."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create data
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, edgecolor='none')
    
    ax.set_xlabel('X Axis', fontweight='bold')
    ax.set_ylabel('Y Axis', fontweight='bold')
    ax.set_zlabel('Z Axis', fontweight='bold')
    ax.set_title('3D Surface: Radial Sine Wave', fontsize=14, fontweight='bold', pad=20)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.view_init(elev=30, azim=45)
    
    return fig, '3d_surface.png'


def create_3d_scatter():
    """Generate 3D scatter plot."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate random data
    np.random.seed(42)
    n = 300
    x = np.random.randn(n)
    y = np.random.randn(n)
    z = np.random.randn(n)
    colors = np.sqrt(x**2 + y**2 + z**2)
    
    # Create scatter plot
    scatter = ax.scatter(x, y, z, c=colors, cmap='plasma', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('X Axis', fontweight='bold')
    ax.set_ylabel('Y Axis', fontweight='bold')
    ax.set_zlabel('Z Axis', fontweight='bold')
    ax.set_title('3D Scatter: Random Distribution', fontsize=14, fontweight='bold', pad=20)
    
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Distance from Origin')
    
    return fig, '3d_scatter.png'


def create_3d_wireframe():
    """Generate 3D wireframe plot."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create data
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    
    # Convert to 2D for wireframe
    theta_grid = np.linspace(-4 * np.pi, 4 * np.pi, 50)
    z_grid = np.linspace(-2, 2, 50)
    Theta, Z = np.meshgrid(theta_grid, z_grid)
    R = Z**2 + 1
    X = R * np.sin(Theta)
    Y = R * np.cos(Theta)
    
    # Plot wireframe
    ax.plot_wireframe(X, Y, Z, color='blue', alpha=0.6, linewidth=1)
    
    ax.set_xlabel('X Axis', fontweight='bold')
    ax.set_ylabel('Y Axis', fontweight='bold')
    ax.set_zlabel('Z Axis', fontweight='bold')
    ax.set_title('3D Wireframe: Spiral', fontsize=14, fontweight='bold', pad=20)
    
    ax.view_init(elev=25, azim=60)
    
    return fig, '3d_wireframe.png'


def create_3d_contour():
    """Generate 3D contour plot."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create data
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2))
    
    # Plot contours
    ax.contour3D(X, Y, Z, 50, cmap='coolwarm')
    
    ax.set_xlabel('X Axis', fontweight='bold')
    ax.set_ylabel('Y Axis', fontweight='bold')
    ax.set_zlabel('Z Axis', fontweight='bold')
    ax.set_title('3D Contour: Gaussian Peak', fontsize=14, fontweight='bold', pad=20)
    
    ax.view_init(elev=35, azim=45)
    
    return fig, '3d_contour.png'


def main():
    """Main execution function."""
    print("=" * 60)
    print("📦 3D Visualization Generator")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    generators = [
        create_3d_surface,
        create_3d_scatter,
        create_3d_wireframe,
        create_3d_contour
    ]
    
    for generator in generators:
        try:
            print(f"\nGenerating {generator.__name__}...")
            fig, filename = generator()
            
            output_path = OUTPUT_DIR / filename
            fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {output_path}")
            
            plt.close(fig)
            
        except Exception as e:
            print(f"❌ Error in {generator.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ Generated {len(generators)} 3D visualizations")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
