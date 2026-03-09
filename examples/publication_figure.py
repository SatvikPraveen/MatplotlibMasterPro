#!/usr/bin/env python3
"""Create a publication-ready figure with proper formatting."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from utils.theme_utils import apply_publication_theme

# Apply publication theme
apply_publication_theme()

# Create high-quality data
x = np.linspace(0, 10, 200)
y1 = np.sin(x) * np.exp(-x/10)
y2 = np.cos(x) * np.exp(-x/10)

# Create figure
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y1, 'k-', linewidth=1.5, label='sin(x)·exp(-x/10)')
ax.plot(x, y2, 'k--', linewidth=1.5, label='cos(x)·exp(-x/10)')

# Format for publication
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Amplitude (V)', fontsize=11)
ax.set_title('Damped Oscillations', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', frameon=True)
ax.grid(True, alpha=0.3, linestyle=':')

# Save with publication settings (600 DPI, tight layout)
plt.savefig('publication_figure.pdf', dpi=600, bbox_inches='tight')
print("✓ Publication figure saved as 'publication_figure.pdf' (600 DPI)")

plt.show()
