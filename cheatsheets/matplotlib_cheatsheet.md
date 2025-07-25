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

## 💾 Saving Figures

```python
plt.savefig("plot.png", dpi=300, bbox_inches="tight")
```

---

## 🧰 Extras

- `plt.grid(True)` → Show grid
- `plt.axhline()`, `plt.axvline()` → Draw reference lines
- `fig.tight_layout()` → Prevent overlapping

---

## 📌 Tip

Use `%matplotlib inline` in Jupyter Notebooks or `%matplotlib notebook` for interactivity.

---

🧠 **Use this with**: `utils/plot_utils.py` and the `notebooks/` for fast recall.
