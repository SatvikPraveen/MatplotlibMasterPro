#!/usr/bin/env python3
"""
Generate Comprehensive Sales Dashboard

This script creates a professional multi-panel dashboard from sales data,
demonstrating real-world data visualization for business analytics.

Usage:
    python scripts/generate_dashboard.py

Output:
    exports/dashboards/sales_dashboard_automated.png
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.theme_utils import apply_corporate_theme

# Configuration
DATA_PATH = Path("datasets/sales_data.csv")
OUTPUT_DIR = Path("exports/dashboards")
OUTPUT_FILE = "sales_dashboard_automated.png"
DPI = 300


def load_data(filepath):
    """Load and validate sales data."""
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    required_cols = ['Month', 'Product', 'Units_Sold', 'Revenue']
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Expected: {required_cols}")
    
    print(f"✓ Loaded {len(df)} rows from {filepath.name}")
    return df


def create_dashboard(df):
    """Create multi-panel sales dashboard."""
    # Apply corporate theme
    apply_corporate_theme()
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Revenue by Product (Bar Chart)
    ax1 = fig.add_subplot(gs[0, :2])
    product_revenue = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
    bars = ax1.bar(range(len(product_revenue)), product_revenue.values, color='steelblue', edgecolor='navy', linewidth=1.5)
    ax1.set_xticks(range(len(product_revenue)))
    ax1.set_xticklabels(product_revenue.index, rotation=45, ha='right')
    ax1.set_ylabel('Total Revenue ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Revenue by Product', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Market Share (Pie Chart)
    ax2 = fig.add_subplot(gs[0, 2])
    colors = plt.cm.Set3(range(len(product_revenue)))
    wedges, texts, autotexts = ax2.pie(product_revenue.values, labels=product_revenue.index,
                                         autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Market Share', fontsize=14, fontweight='bold', pad=15)
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    # 3. Revenue Trend (Line Chart)
    ax3 = fig.add_subplot(gs[1, :])
    monthly_revenue = df.groupby('Month')['Revenue'].sum()
    ax3.plot(monthly_revenue.index, monthly_revenue.values, marker='o', linewidth=2.5,
             markersize=8, color='darkgreen', label='Revenue')
    ax3.fill_between(monthly_revenue.index, monthly_revenue.values, alpha=0.3, color='lightgreen')
    ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
    ax3.set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Add trend line
    z = np.polyfit(range(len(monthly_revenue)), monthly_revenue.values, 1)
    p = np.poly1d(z)
    ax3.plot(monthly_revenue.index, p(range(len(monthly_revenue))),
             "--", color='red', alpha=0.7, linewidth=2, label='Trend')
    ax3.legend(fontsize=10)
    
    # 4. Units Sold by Product (Horizontal Bar)
    ax4 = fig.add_subplot(gs[2, :2])
    product_units = df.groupby('Product')['Units_Sold'].sum().sort_values()
    ax4.barh(range(len(product_units)), product_units.values, color='coral', edgecolor='darkred', linewidth=1.5)
    ax4.set_yticks(range(len(product_units)))
    ax4.set_yticklabels(product_units.index)
    ax4.set_xlabel('Units Sold', fontsize=12, fontweight='bold')
    ax4.set_title('Total Units Sold by Product', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Summary Statistics (Text Box)
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    total_revenue = df['Revenue'].sum()
    total_units = df['Units_Sold'].sum()
    avg_price = total_revenue / total_units if total_units > 0 else 0
    best_product = product_revenue.index[0]
    
    stats_text = f"""
    📊 SUMMARY STATISTICS
    
    Total Revenue: ${total_revenue:,.2f}
    Total Units: {total_units:,}
    Avg Price/Unit: ${avg_price:.2f}
    
    Top Product: {best_product}
    Products: {len(product_revenue)}
    Months: {len(monthly_revenue)}
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange', linewidth=2),
             family='monospace')
    
    # Main title
    fig.suptitle('📈 Sales Performance Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    return fig


def save_dashboard(fig, output_path):
    """Save dashboard to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"✓ Dashboard saved to: {output_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("📊 Sales Dashboard Generator")
    print("=" * 60)
    
    try:
        # Load data
        df = load_data(DATA_PATH)
        
        # Create dashboard
        print("Creating dashboard...")
        fig = create_dashboard(df)
        
        # Save output
        output_path = OUTPUT_DIR / OUTPUT_FILE
        save_dashboard(fig, output_path)
        
        print("\n" + "=" * 60)
        print("✅ Dashboard generation complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    finally:
        plt.close('all')


if __name__ == "__main__":
    main()
