#!/usr/bin/env python3
"""Apply custom themes with a single function call."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from utils.theme_utils import (apply_dark_theme, apply_minimal_theme, 
                                apply_corporate_theme, apply_colorblind_friendly_theme)

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Try different themes
themes = [
    ("Dark Theme", apply_dark_theme),
    ("Minimal Theme", apply_minimal_theme),
    ("Corporate Theme", apply_corporate_theme),
    ("Colorblind-Friendly", apply_colorblind_friendly_theme)
]

for name, theme_func in themes:
    # Apply theme
    theme_func()
    
    # Create plot
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, linewidth=2)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title(f'{name} Demo')
    plt.grid(True, alpha=0.3)
    
    # Save
    filename = name.lower().replace(' ', '_').replace('-', '_') + '.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {filename}")

print("\n✅ All themes demonstrated!")
