#!/usr/bin/env python3
"""Create a simple animation in <20 lines."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Setup
fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))
ax.set_ylim(-1.5, 1.5)
ax.set_title('Animated Sine Wave')

# Animation function
def animate(frame):
    line.set_ydata(np.sin(x + frame/10))
    return line,

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=True)

# Save as GIF
anim.save('animated_sine.gif', writer='pillow', fps=20)
print("✓ Animation saved as 'animated_sine.gif'")

plt.show()
