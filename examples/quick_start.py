#!/usr/bin/env python3
"""Quick Start - Your first matplotlib plot in 10 lines!"""

import matplotlib.pyplot as plt

# Create data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create plot
plt.plot(x, y, marker='o')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('My First Plot')
plt.grid(True)

# Save and show
plt.savefig('my_first_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Plot saved as 'my_first_plot.png'")
