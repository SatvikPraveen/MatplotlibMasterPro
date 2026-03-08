# 🎯 Matplotlib Cheatsheet

A one-stop syntax reference for mastering `matplotlib.pyplot`.

---

## 📦 Setup

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

---

## 🧱 Plot Types

### 📈 Line Plot

```python
plt.plot(x, y, label="Series A", linestyle="--", color="blue")
plt.title("Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
```

### 📊 Bar Plot

```python
plt.bar(categories, values, color="teal")
```

### 🟣 Scatter Plot

```python
plt.scatter(x, y, c="red", alpha=0.6)
```

### 📉 Histogram

```python
plt.hist(data, bins=10, edgecolor="black")
```

### 🧁 Pie Chart

```python
plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
```

---

## 🪟 Subplots

### Basic Layout

```python
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(x, y)
axs[1, 1].bar(x, y)
plt.tight_layout()
```

### GridSpec Layout

```python
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 2)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
```

---

## 🎨 Styling & Themes

```python
plt.style.use("seaborn-vibrant")  # or "ggplot", "bmh", "dark_background"
```

Customizing plots:

```python
plt.title("Title", fontsize=14, fontweight="bold")
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.xticks(rotation=45)
```

---

## 🧷 Annotations

```python
plt.annotate("Peak", xy=(x1, y1), xytext=(x1+1, y1+20),
             arrowprops=dict(facecolor='black', arrowstyle="->"))
```

---

## � 3D Plots

```python
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3D Scatter
ax.scatter(x, y, z, c='red', marker='o')

# 3D Surface
ax.plot_surface(X, Y, Z, cmap='viridis')

# 3D Wireframe
ax.plot_wireframe(X, Y, Z, color='blue')

# Set viewing angle
ax.view_init(elev=30, azim=45)
```

---

## 📊 Statistical Plots

### Box Plot

```python
plt.boxplot(data, labels=['Group 1', 'Group 2'], 
            patch_artist=True, notch=True, showmeans=True)
```

### Violin Plot

```python
parts = plt.violinplot(data, positions=[1, 2, 3], 
                       showmeans=True, showmedians=True)
```

---

## 📉 Error Bars & Confidence Intervals

### Error Bars

```python
plt.errorbar(x, y, yerr=errors, fmt='o-', capsize=5, 
             capthick=2, ecolor='red')
```

### Fill Between (Confidence Interval)

```python
plt.fill_between(x, y_lower, y_upper, alpha=0.3, 
                 color='blue', label='95% CI')
```

### Asymmetric Errors

```python
plt.errorbar(x, y, yerr=[lower_errors, upper_errors], 
             fmt='s-', capsize=7)
```

---

## 🗺️ Contour Plots & Heatmaps

### Contour Lines

```python
contour = plt.contour(X, Y, Z, levels=10, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)
```

### Filled Contour

```python
plt.contourf(X, Y, Z, levels=20, cmap='RdBu_r')
plt.colorbar(label='Value')
```

### Heatmap

```python
plt.imshow(matrix, cmap='hot', aspect='auto')
plt.colorbar()
```

---

## 🔵 Polar Plots

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')

# Polar plot
theta = np.linspace(0, 2*np.pi, 100)
r = 1 + np.sin(3*theta)
ax.plot(theta, r)
ax.fill(theta, r, alpha=0.3)

# Radar chart
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
ax.plot(angles, values, 'o-', linewidth=2)
ax.set_xticks(angles)
ax.set_xticklabels(categories)
```

---

## 🔄 Twin Axes (Composite Plots)

### Two Y-Axes

```python
fig, ax1 = plt.subplots()

# First y-axis
ax1.plot(x, y1, 'b-', label='Primary')
ax1.set_ylabel('Y1', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Second y-axis
ax2 = ax1.twinx()
ax2.plot(x, y2, 'r-', label='Secondary')
ax2.set_ylabel('Y2', color='r')
ax2.tick_params(axis='y', labelcolor='r')
```

### Shared Axes

```python
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# All subplots share the same x-axis
```

---

## 🔍 Inset Plots & Zoom

### Basic Inset

```python
ax_inset = ax.inset_axes([0.6, 0.6, 0.35, 0.35])  # [x, y, width, height]
ax_inset.plot(x_zoom, y_zoom, 'r-')
ax_inset.set_xlim(x_min, x_max)
```

### Zoomed Inset with Connection

```python
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

axins = zoomed_inset_axes(ax, zoom=3, loc='upper right')
axins.plot(x, y)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="red")
```

### Indicate Inset Zoom

```python
ax_inset = ax.inset_axes([0.1, 0.6, 0.3, 0.3])
ax_inset.plot(x_detail, y_detail)
ax.indicate_inset_zoom(ax_inset, edgecolor='red')
```

---

## 💾 Saving Figures

```python
# Single format
plt.savefig("plot.png", dpi=300, bbox_inches="tight")

# Multiple formats
for fmt in ['png', 'pdf', 'svg']:
    plt.savefig(f"plot.{fmt}", dpi=300, bbox_inches="tight")

# High quality for publication
plt.savefig("figure.pdf", dpi=600, bbox_inches="tight", 
            format='pdf', transparent=True)
```

---

## 🎨 Color & Style

### Colormaps

```python
# Sequential: viridis, plasma, inferno, magma, cividis
# Diverging: RdBu, RdYlGn, coolwarm, seismic
# Qualitative: tab10, tab20, Set1, Set2, Set3

plt.scatter(x, y, c=values, cmap='viridis')
plt.colorbar(label='Value')
```

### Custom Colors

```python
# Hex colors
plt.plot(x, y, color='#FF5733')

# RGB tuples
plt.plot(x, y, color=(0.2, 0.4, 0.6))

# Named colors
plt.plot(x, y, color='coral')
```

### Themes from utils

```python
from utils.theme_utils import *

apply_dark_theme()
apply_corporate_theme()
apply_minimal_theme()
apply_publication_theme()
apply_colorblind_friendly_theme()
```

---

## 🧰 Extras

### Reference Lines

```python
plt.axhline(y=0, color='k', linestyle='--', linewidth=1)  # Horizontal
plt.axvline(x=5, color='r', linestyle='--', linewidth=1)  # Vertical
plt.axline((0, 0), slope=1, color='gray', linestyle=':')  # Diagonal
```

### Grid

```python
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
```

### Legends

```python
plt.legend(loc='best')  # Auto placement
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Outside plot
plt.legend(ncol=2)  # Two columns
```

### Limits & Ticks

```python
plt.xlim(0, 10)
plt.ylim(-5, 5)
plt.xticks(range(0, 11, 2))
plt.yticks([-5, 0, 5], ['Low', 'Medium', 'High'])
```

---

## 📊 Quick Plot Types Reference

| Type | Command | Use Case |
|------|---------|----------|
| Line | `plt.plot(x, y)` | Trends over time |
| Scatter | `plt.scatter(x, y)` | Correlation |
| Bar | `plt.bar(x, height)` | Categorical comparison |
| Histogram | `plt.hist(data)` | Distribution |
| Box | `plt.boxplot(data)` | Statistical summary |
| Violin | `plt.violinplot(data)` | Distribution density |
| Pie | `plt.pie(sizes)` | Proportions |
| Heatmap | `plt.imshow(matrix)` | 2D data/matrix |
| Contour | `plt.contour(X, Y, Z)` | 3D surface data |
| Polar | `ax.plot(theta, r)` | Circular data |
| 3D Surface | `ax.plot_surface(X, Y, Z)` | 3D relationships |
| Error bars | `plt.errorbar(x, y, yerr)` | Uncertainty |

---

## 🔧 Common Patterns

### Save and Close

```python
fig, ax = plt.subplots()
# ... plotting code ...
plt.savefig('output.png')
plt.close(fig)  # Free memory
```

### Context Manager

```python
with plt.rc_context({"font.size": 14}):
    # Plot with custom settings
    plt.plot(x, y)
# Settings auto-reset after block
```

### Turn Off Interactive Mode

```python
plt.ioff()  # Don't show plots immediately
# ... create multiple plots ...
plt.show()  # Show all at once
```

---

## 📌 Tips & Best Practices

1. ✅ Always use `plt.tight_layout()` to prevent label overlap
2. ✅ Save figures **before** calling `plt.show()`
3. ✅ Use `figsize=(width, height)` to control size
4. ✅ Set DPI to 300+ for publication-quality figures
5. ✅ Use vector formats (PDF, SVG) for scalable graphics
6. ✅ Close figures with `plt.close()` to free memory
7. ✅ Use descriptive labels and titles
8. ✅ Add legends when plotting multiple series
9. ✅ Choose appropriate colormaps (sequential, diverging, qualitative)
10. ✅ Test plots with colorblind-friendly palettes

---

## 📚 Jupyter Notebook Magic

```python
%matplotlib inline        # Static plots
%matplotlib notebook      # Interactive (deprecated)
%matplotlib widget        # Interactive (ipympl)
%config InlineBackend.figure_format = 'retina'  # High DPI
```

---

## 🚀 Performance Tips

- Use `plt.ioff()` for batch processing
- Downsample large datasets before plotting
- Use `rasterized=True` for plots with many elements
- Close unused figures: `plt.close('all')`
- Use Agg backend for non-interactive: `matplotlib.use('Agg')`

---

## 📌 Tip

Use `%matplotlib inline` in Jupyter Notebooks or `%matplotlib notebook` for interactivity.

---

🧠 **Use this with**: `utils/plot_utils.py` and the `notebooks/` for fast recall.
