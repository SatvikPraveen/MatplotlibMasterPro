#!/usr/bin/env python3
"""Process multiple CSV files and create plots automatically."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

# Configuration
DATA_DIR = Path("datasets")
OUTPUT_DIR = Path("exports/batch_processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find all CSV files
csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    print("No CSV files found in datasets/ folder")
    sys.exit(1)

print(f"Found {len(csv_files)} CSV files. Processing...\n")

# Process each file
for csv_path in csv_files:
    try:
        # Load data
        df = pd.read_csv(csv_path)
        
        # Create a simple plot (customize based on your data)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot first two numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) >= 2:
            ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
        
        ax.set_title(f'Data from {csv_path.stem}')
        ax.grid(True, alpha=0.3)
        
        # Save
        output_path = OUTPUT_DIR / f"{csv_path.stem}_plot.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ {csv_path.name} → {output_path.name}")
        
    except Exception as e:
        print(f"✗ {csv_path.name}: {e}")

print(f"\n✅ Batch processing complete! Check {OUTPUT_DIR}/")
