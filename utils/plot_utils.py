# utils/plot_utils.py

import os
from ipywidgets import interact, widgets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib import animation
from matplotlib.animation import FuncAnimation

def line_plot(
    x, y, title="", xlabel="", ylabel="", label=None,
    color="blue", linestyle="-", marker=None, figsize=(10, 5), grid=True
):
    """
    Wrapper to create a simple line plot with common defaults.
    """
    plt.figure(figsize=figsize)
    plt.plot(x, y, label=label, color=color, linestyle=linestyle, marker=marker)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if label:
        plt.legend()
    if grid:
        plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


def multi_line_plot(
    df, x_col, y_cols, labels=None, colors=None, markers=None,
    title="", xlabel="", ylabel="", figsize=(10, 6)
):
    """
    Plot multiple lines from a DataFrame column-wise.
    """
    plt.figure(figsize=figsize)

    for i, y_col in enumerate(y_cols):
        label = labels[i] if labels else y_col
        color = colors[i] if colors else None
        marker = markers[i] if markers else None
        plt.plot(df[x_col], df[y_col], label=label, color=color, marker=marker)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


def bar_plot(
    categories, values, *,
    horizontal=False, title="", xlabel="", ylabel="",
    color="skyblue", edgecolor="black", grid=True,
    figsize=(8, 5)
):
    """
    Plot a simple vertical or horizontal bar chart.
    """
    plt.figure(figsize=figsize)
    if horizontal:
        plt.barh(categories, values, color=color, edgecolor=edgecolor)
    else:
        plt.bar(categories, values, color=color, edgecolor=edgecolor)

    plt.title(title)
    if horizontal:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if grid:
            plt.grid(axis='x')
    else:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if grid:
            plt.grid(axis='y')

    plt.tight_layout()
    plt.show()


def grouped_bar_plot(
    x_labels, data_dict, *,
    title="", xlabel="", ylabel="",
    bar_width=0.2, figsize=(10, 6)
):
    """
    Plot grouped bar charts from a dictionary of label -> list of values.
    """
    categories = list(data_dict.keys())
    x = list(range(len(x_labels)))

    plt.figure(figsize=figsize)

    for i, cat in enumerate(categories):
        plt.bar(
            [val + i * bar_width for val in x],
            data_dict[cat],
            width=bar_width,
            label=cat
        )

    plt.xticks([r + (bar_width * len(categories)) / 2 for r in x], x_labels, rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def scatter_plot(
    x, y, *,
    color="green", size=80, alpha=0.7,
    title="", xlabel="", ylabel="",
    edgecolor="white", cmap=None, color_values=None,
    use_colorbar=False, figsize=(8, 6)
):
    """
    Plot a scatter plot with customization options.
    """
    plt.figure(figsize=figsize)

    if color_values is not None and cmap:
        scatter = plt.scatter(x, y, c=color_values, s=size, cmap=cmap, alpha=alpha, edgecolors=edgecolor)
        if use_colorbar:
            plt.colorbar(scatter, label=xlabel)
    else:
        plt.scatter(x, y, c=color, s=size, alpha=alpha, edgecolors=edgecolor)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def histogram_plot(
    data, *,
    bins=10, color="steelblue", edgecolor="black", alpha=0.7,
    title="", xlabel="", ylabel="Frequency",
    density=False, grid=True, figsize=(8, 5)
):
    """
    Plot a histogram for continuous or discrete values.
    """
    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, color=color, edgecolor=edgecolor, alpha=alpha, density=density)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density" if density else ylabel)
    if grid:
        plt.grid(True)
    plt.tight_layout()
    plt.show()


def pie_chart(
    labels, values, *,
    colors=None, explode=None, autopct="%1.1f%%",
    title="", startangle=90, shadow=True, figsize=(6, 6)
):
    """
    Plot a pie chart with optional explode, color, and percentage formatting.
    """
    plt.figure(figsize=figsize)
    plt.pie(
        values,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct=autopct,
        startangle=startangle,
        shadow=shadow
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


def grid_plot(
    plot_funcs, *,
    titles=None, nrows=1, ncols=2, figsize=(12, 5),
    suptitle=None, sharex=False, sharey=False
):
    """
    Create a grid of subplots using a list of functions that accept ax.
    
    Parameters:
    - plot_funcs: list of callables that accept a matplotlib Axes object
    - titles: optional list of subplot titles
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    axes = np.array(axes).reshape(-1)  # flatten in case of 2D layout

    for i, func in enumerate(plot_funcs):
        func(axes[i])
        if titles:
            axes[i].set_title(titles[i])

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    plt.tight_layout()
    plt.show()


def dual_axis_plot(
    x, y1, y2, *,
    label1="Primary", label2="Secondary",
    color1="tab:blue", color2="tab:red",
    xlabel="", ylabel1="", ylabel2="",
    title="", figsize=(10, 5)
):
    """
    Create a dual-axis plot (primary and secondary Y-axis).
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(x, y1, color=color1, label=label1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(x, y2, color=color2, label=label2)
    ax2.set_ylabel(ylabel2, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(title)
    fig.tight_layout()
    plt.show()


def fill_between_plot(
    x, y1, y2=0, *,
    title="", xlabel="", ylabel="",
    color="skyblue", alpha=0.4, label=None,
    edge=True, linestyle="--", figsize=(10, 5)
):
    """
    Plot area between y1 and y2 (default 0).
    """
    plt.figure(figsize=figsize)
    plt.plot(x, y1, label=label, linestyle=linestyle if edge else "None")
    plt.fill_between(x, y1, y2, color=color, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if label:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


def log_scale_plot(
    x, y, *,
    log_axis="y",
    title="", xlabel="", ylabel="", label=None,
    color="navy", marker="o", linestyle="-",
    figsize=(8, 5)
):
    """
    Plot a line or scatter with log scale.
    """
    plt.figure(figsize=figsize)
    plt.plot(x, y, label=label, color=color, linestyle=linestyle, marker=marker)
    if log_axis == "y":
        plt.yscale("log")
    elif log_axis == "x":
        plt.xscale("log")
    elif log_axis == "both":
        plt.xscale("log")
        plt.yscale("log")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if label:
        plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


def twin_axes_fill_plot(
    x, y1, y2, *,
    label1="Y1", label2="Y2",
    xlabel="", ylabel1="", ylabel2="",
    color1="tab:blue", color2="tab:green",
    alpha1=0.5, alpha2=0.3,
    title="", figsize=(10, 5)
):
    """
    Plot fill areas for y1 and y2 on twin y-axes.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.fill_between(x, y1, color=color1, alpha=alpha1, label=label1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.fill_between(x, y2, color=color2, alpha=alpha2, label=label2)
    ax2.set_ylabel(ylabel2, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(title)
    fig.tight_layout()
    plt.show()

def annotate_point(ax, x, y, text, *,
                   xytext=(10, 10), textcolor='black',
                   arrowprops=None, fontsize=10):
    """
    Annotate a single point on a plot.
    """
    if arrowprops is None:
        arrowprops = dict(arrowstyle='->', color='black')
    
    ax.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        textcoords='offset points',
        fontsize=fontsize,
        color=textcolor,
        arrowprops=arrowprops
    )


def highlight_region(ax, x_start, x_end, *,
                     color='yellow', alpha=0.3, label=None):
    """
    Highlight vertical region on the x-axis.
    """
    ax.axvspan(x_start, x_end, color=color, alpha=alpha, label=label)


def label_line(ax, x, y, text, *,
               color='black', fontsize=10, location='right', offset=(0, 0)):
    """
    Label a line segment directly on the line.
    location: 'left', 'right', or 'center'
    Works for both numerical and categorical x-axis values.
    """
    idx = {
        'left': 0,
        'center': len(x) // 2,
        'right': -1
    }[location]

    ax.text(
        idx + offset[0],  # use index as x position
        y[idx] + offset[1],
        text,
        fontsize=fontsize,
        color=color
    )


import matplotlib.pyplot as plt
import numpy as np

def imshow_matrix(matrix, *,
                  cmap='viridis', title='', xlabel='', ylabel='',
                  colorbar=True, figsize=(6, 5)):
    """
    Visualize a 2D matrix using imshow.
    """
    plt.figure(figsize=figsize)
    im = plt.imshow(matrix, cmap=cmap, aspect='auto')
    if colorbar:
        plt.colorbar(im)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_image(image_array, *,
               cmap=None, title='', figsize=(6, 6)):
    """
    Display an image (grayscale or RGB).
    """
    plt.figure(figsize=figsize)
    plt.imshow(image_array, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def grid_heatmap(data, row_labels, col_labels, *,
                 cmap="YlGnBu", annot=False, fmt=".2f",
                 title="", figsize=(8, 6)):
    """
    Create a labeled grid heatmap using imshow.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap)

    # Show all ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate tick labels and align
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    if annot:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, format(data[i, j], fmt),
                               ha="center", va="center", color="black")

    ax.set_title(title)
    fig.colorbar(im)
    fig.tight_layout()
    plt.show()


def interactive_slider_plot(x, y_series_dict, *,
                             xlabel="X", ylabel="Y", title_prefix="Value at index"):
    """
    Create an interactive slider to explore different Y-series with common X.
    """
    @interact(index=widgets.IntSlider(min=0, max=len(x)-1, step=1, value=0))
    def plot(index):
        plt.figure(figsize=(8, 5))
        for label, y in y_series_dict.items():
            plt.plot(x, y, label=label)
            plt.scatter(x[index], y[index], s=80, label=f"{label} @ {x[index]} = {y[index]}")
        plt.title(f"{title_prefix}: {x[index]}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()


def dropdown_plot(x, y_series_dict, *,
                  xlabel="X", ylabel="Y", title="Dropdown Series Viewer"):
    """
    Create a dropdown to select which Y-series to plot.
    """
    @interact(series=widgets.Dropdown(options=list(y_series_dict.keys()), description="Series"))
    def plot(series):
        y = y_series_dict[series]
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label=series, color="tab:blue", marker="o")
        plt.title(f"{title} — {series}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()


def save_plot(fig, filename, folder="exports", formats=("png", "pdf", "svg")):
    """
    Save a matplotlib figure in multiple formats to a specified folder.
    """
    os.makedirs(folder, exist_ok=True)
    for fmt in formats:
        path = os.path.join(folder, f"{filename}.{fmt}")
        fig.savefig(path, format=fmt, bbox_inches='tight')
    print(f"✅ Plot saved as {formats} in '{folder}/'")


def set_global_style(style_name="ggplot", font="DejaVu Sans", size=12):
    """
    Apply a global matplotlib style and font settings.
    """
    plt.style.use(style_name)
    plt.rcParams.update({
        "font.family": font,
        "font.size": size
    })
    print(f"🎨 Global style set: '{style_name}' with font '{font}' and size {size}")


def animate_line_plot(x, y, *, xlabel="", ylabel="", title="", interval=200, color="blue"):
    """
    Create a simple animated line plot.

    Parameters:
    - x, y: Data for the line
    - xlabel, ylabel, title: Axis and title labels
    - interval: Delay between frames in milliseconds
    - color: Line color
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y) * 1.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    line, = ax.plot([], [], color=color, linewidth=2)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        line.set_data(x[:frame], y[:frame])
        return line,

    ani = animation.FuncAnimation(
        fig, update, frames=len(x) + 1,
        init_func=init, blit=True, interval=interval, repeat=False
    )

    plt.show()


def animate_dual_line_plot(x, y1, y2, *,
                            xlabel="X", ylabel="Y",
                            title="📈 Dual Line Animation",
                            labels=("Line 1", "Line 2"),
                            colors=("blue", "green"),
                            interval=300):
    fig, ax = plt.subplots(figsize=(10, 5))
    line1, = ax.plot([], [], color=colors[0], label=labels[0])
    line2, = ax.plot([], [], color=colors[1], label=labels[1])
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(min(np.min(y1), np.min(y2)), max(np.max(y1), np.max(y2)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    def update(frame):
        line1.set_data(x[:frame], y1[:frame])
        line2.set_data(x[:frame], y2[:frame])
        return line1, line2

    anim = FuncAnimation(fig, update, frames=len(x)+1, interval=interval, repeat=False)
    plt.show()


def animate_bar_chart(categories, values, *,
                      xlabel="Category", ylabel="Value",
                      title="📊 Animated Bar Chart",
                      color="skyblue",
                      interval=200):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(categories, [0]*len(values), color=color)
    ax.set_ylim(0, max(values) * 1.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    def update(frame):
        for bar, val in zip(bars, values[:frame]):
            bar.set_height(val)
        return bars

    anim = FuncAnimation(fig, update, frames=len(values)+1, interval=interval, repeat=False)
    plt.show()


def animate_scatter_growth(x, y, *,
                           xlabel="X", ylabel="Y",
                           title="🔵 Animated Scatter Growth",
                           color="purple",
                           interval=200):
    fig, ax = plt.subplots(figsize=(10, 5))
    scat = ax.scatter([], [], color=color)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    def update(frame):
        scat.set_offsets(np.c_[x[:frame], y[:frame]])
        return scat,

    anim = FuncAnimation(fig, update, frames=len(x)+1, interval=interval, repeat=False)
    plt.show()


# 📦 Export Single Line Animation
def save_line_animation(x, y, filename="line_animation.mp4", dpi=100, fps=10):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2, color="tab:blue")
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Line Animation Export")

    def update(frame):
        line.set_data(x[:frame], y[:frame])
        return line,

    anim = FuncAnimation(fig, update, frames=len(x)+1, interval=100)
    anim.save(filename, dpi=dpi, fps=fps, writer='ffmpeg')
    print(f"✅ Saved line animation to: {filename}")


# 🔁 Export Dual Line Animation
def save_dual_line_animation(x, y1, y2, labels=("Series A", "Series B"),
                              filename="dual_line_animation.mp4", dpi=100, fps=10):
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], lw=2, label=labels[0], color="tab:blue")
    line2, = ax.plot([], [], lw=2, label=labels[1], color="tab:orange")
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(min(np.min(y1), np.min(y2)), max(np.max(y1), np.max(y2)))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Dual Line Animation Export")
    ax.legend()

    def update(frame):
        line1.set_data(x[:frame], y1[:frame])
        line2.set_data(x[:frame], y2[:frame])
        return line1, line2

    anim = FuncAnimation(fig, update, frames=len(x)+1, interval=100)
    anim.save(filename, dpi=dpi, fps=fps, writer='ffmpeg')
    print(f"✅ Saved dual-line animation to: {filename}")


# 📊 Export Animated Bar Chart
def save_bar_animation(categories, values, filename="bar_animation.mp4", dpi=100, fps=10):
    fig, ax = plt.subplots()
    bars = ax.bar(categories, np.zeros_like(values))
    ax.set_ylim(0, max(values) * 1.1)
    ax.set_ylabel("Value")
    ax.set_title("Bar Chart Animation Export")
    plt.xticks(rotation=45)

    def update(frame):
        for bar, height in zip(bars, values[:frame]):
            bar.set_height(height)
        return bars

    anim = FuncAnimation(fig, update, frames=len(values)+1, interval=100)
    anim.save(filename, dpi=dpi, fps=fps, writer='ffmpeg')
    print(f"✅ Saved bar animation to: {filename}")


# 🔵 Export Animated Scatter Growth
def save_scatter_animation(x, y, filename="scatter_animation.mp4", dpi=100, fps=10):
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], alpha=0.7, s=50, color="tab:green")
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Scatter Animation Export")

    def update(frame):
        sc.set_offsets(np.c_[x[:frame], y[:frame]])
        return sc,

    anim = FuncAnimation(fig, update, frames=len(x)+1, interval=100)
    anim.save(filename, dpi=dpi, fps=fps, writer='ffmpeg')
    print(f"✅ Saved scatter animation to: {filename}")


def plot_histogram_with_stats(data, bins=10, title="📊 Histogram with Stats", xlabel="", ylabel="Frequency"):
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.2f}")
    plt.axvline(median, color="green", linestyle="--", label=f"Median: {median:.2f}")
    plt.axvline(mean + std, color="orange", linestyle=":", label=f"+1 STD: {mean + std:.2f}")
    plt.axvline(mean - std, color="orange", linestyle=":", label=f"-1 STD: {mean - std:.2f}")
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_boxplot(data_dict, title="📦 Boxplot Comparison", ylabel="Value"):
    plt.figure(figsize=(8, 5))
    plt.boxplot(data_dict.values(), labels=data_dict.keys(), patch_artist=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_violinplot(data_dict, title="🎻 Violin Plot", ylabel="Value"):
    plt.figure(figsize=(8, 5))
    parts = plt.violinplot(data_dict.values(), showmeans=True)
    plt.xticks(ticks=range(1, len(data_dict)+1), labels=list(data_dict.keys()))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def grouped_bar_plot(df, category, subcategory, value, title="Grouped Bar Chart"):
    """
    Create a grouped bar chart showing `value` across `subcategory` for each `category`.
    Example: Revenue by Product per Month.
    """

    # 🧮 Step 1: Aggregate
    agg_df = df.groupby([category, subcategory])[value].sum().reset_index()

    # 🔄 Step 2: Pivot for plotting
    pivot_df = agg_df.pivot(index=category, columns=subcategory, values=value).fillna(0)

    categories = pivot_df.index
    subcats = pivot_df.columns
    x = np.arange(len(categories))
    width = 0.8 / len(subcats)

    # 📊 Step 3: Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, sub in enumerate(subcats):
        ax.bar(x + i * width, pivot_df[sub], width, label=sub)

    ax.set_xlabel(category)
    ax.set_ylabel(value)
    ax.set_title(title)
    ax.set_xticks(x + width * (len(subcats) - 1) / 2)
    ax.set_xticklabels(categories, rotation=45)
    ax.legend(title=subcategory)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()



def stacked_area_plot(x, y_series_dict, xlabel="", ylabel="", title="Stacked Area Chart"):
    """
    Plot a stacked area chart using multiple series in y_series_dict.
    Keys are used as labels.
    """

    y_values = np.row_stack([y_series_dict[k] for k in y_series_dict.keys()])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(x, y_values, labels=list(y_series_dict.keys()), alpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def subplot_groupwise(df, group_col, x_col, y_col, title_prefix="Group"):
    """
    Create individual subplots for each group in `group_col`.
    Useful for visualizing the same metric across groups like regions.
    """

    groups = df[group_col].unique()
    n = len(groups)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten()

    for i, group in enumerate(groups):
        sub_df = df[df[group_col] == group]
        axes[i].plot(sub_df[x_col], sub_df[y_col], marker='o')
        axes[i].set_title(f"{title_prefix} {group}")
        axes[i].set_xlabel(x_col)
        axes[i].set_ylabel(y_col)
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].tick_params(axis='x', rotation=45)

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ✅ Save Grouped Bar Plot
def save_grouped_bar_plot(df, category, subcategory, value, title, filename):
    """
    Save a grouped bar chart showing `value` across `subcategory` for each `category`.
    """
    pivot_df = df.pivot_table(index=category, columns=subcategory, values=value, aggfunc="sum").fillna(0)
    categories = pivot_df.index
    subcats = pivot_df.columns

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.8 / len(subcats)
    for i, subcat in enumerate(subcats):
        ax.bar(
            [x + i * width for x in range(len(categories))],
            pivot_df[subcat],
            width=width,
            label=subcat
        )
    ax.set_xticks([x + width * (len(subcats)-1)/2 for x in range(len(categories))])
    ax.set_xticklabels(categories, rotation=45)
    ax.set_ylabel(value)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


# ✅ Save Stacked Area Plot
def save_stacked_area_plot(x, y_series_dict, xlabel, ylabel, title, filename):
    """
    Save a stacked area plot from multiple y-series dict.
    Example: {"Revenue": [...], "Units Sold": [...]}
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    y_values = list(y_series_dict.values())
    labels = list(y_series_dict.keys())
    ax.stackplot(x, *y_values, labels=labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


# ✅ Save Subplots by Group
def save_subplot_groupwise(df, group_col, x_col, y_col, title_prefix, filename_prefix):
    """
    Save individual subplots for each group (e.g., Product-wise Revenue over Month)
    """
    groups = df[group_col].unique()
    for group in groups:
        subset = df[df[group_col] == group]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(subset[x_col], subset[y_col], marker='o')
        ax.set_title(f"{title_prefix} {group}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        output_file = f"{filename_prefix}_{group}.png"
        fig.savefig(output_file, dpi=300)
        plt.close(fig)


def display_colormap_samples(n=256):

    maps = [
        "viridis", "plasma", "inferno", "magma", "cividis",
        "coolwarm", "bwr", "seismic",
        "Pastel1", "Set1", "Set2", "Paired", "Dark2"
    ]

    fig, axes = plt.subplots(len(maps), 1, figsize=(8, 0.5 * len(maps)))
    gradient = np.linspace(0, 1, n).reshape(1, -1)

    for ax, name in zip(axes, maps):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        ax.set_title(name, fontsize=9, loc="left")
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


def save_sequential_bar_plot(months, values, filename, cmap="plasma"):

    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(months)))
    plt.figure(figsize=(10, 5))
    plt.bar(months, values, color=colors)
    plt.title(f"📊 Sequential Bar Plot — {cmap} Colormap")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_diverging_change_plot(months, changes, filename, cmap="coolwarm"):

    normalized = (changes - changes.min()) / (changes.max() - changes.min())
    colors = plt.cm.get_cmap(cmap)(normalized)

    plt.figure(figsize=(10, 5))
    plt.bar(months, changes, color=colors)
    plt.title(f"🔁 Diverging Change Plot — {cmap} Colormap")
    plt.xticks(rotation=45)
    plt.axhline(0, color="gray", linestyle="--")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_qualitative_grouped_bar(df, filename):

    df["Month_Str"] = pd.to_datetime(df["Month"]).dt.strftime("%b")
    months = sorted(df["Month_Str"].unique(), key=lambda m: pd.to_datetime(m, format='%b').month)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    products = df["Product"].unique()
    width = 0.8 / len(products)
    x = np.arange(len(months))

    plt.figure(figsize=(10, 5))
    for i, product in enumerate(products):
        product_vals = df[df["Product"] == product].groupby("Month_Str")["Units Sold"].sum().reindex(months)
        plt.bar(x + i * width, product_vals.values, width=width, label=product, color=color_cycle[i % len(color_cycle)])

    plt.title("🌈 Units Sold per Product per Month — Qualitative Colors")
    plt.xticks(x + width * (len(products)-1)/2, months, rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_timeseries_trend(x, y, ylabel="Value", title="Time Series Trend"):
    """
    Plot a basic time-series line chart.
    """

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker='o', linestyle='-')
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def save_timeseries_trend(x, y, path):
    """
    Save a time-series line chart to the given path.
    """

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker='o', linestyle='-')
    ax.set_title("Time Series Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()


def plot_rolling_mean_std(series, window=3, title="Rolling Statistics"):
    """
    Plot original series with rolling mean and std deviation.
    """

    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series, label="Original", color="blue")
    ax.plot(rolling_mean, label=f"Rolling Mean ({window})", color="orange")
    ax.fill_between(series.index, rolling_mean - rolling_std, rolling_mean + rolling_std,
                    color="orange", alpha=0.2, label="Rolling Std")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def save_rolling_stats_plot(series, window=3, path=None):
    """
    Save rolling mean and std deviation plot.
    """

    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series, label="Original", color="blue")
    ax.plot(rolling_mean, label=f"Rolling Mean ({window})", color="orange")
    ax.fill_between(series.index, rolling_mean - rolling_std, rolling_mean + rolling_std,
                    color="orange", alpha=0.2, label="Rolling Std")
    ax.set_title("Rolling Statistics")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()


def plot_multi_product_timeseries(df, title="Multiple Product Time Series"):
    """
    Plot multiple time series from a DataFrame (columns as series).
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def save_multi_product_timeseries(df, path):
    """
    Save multiple product time series to a file.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col)

    ax.set_title("Multiple Product Time Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()


def create_dashboard(df, monthly, save_path=None):
    """
    Create a 2x2 dashboard layout visualizing sales trends and stats.

    Parameters:
    - df: raw sales dataframe with 'Units Sold' and 'Revenue'
    - monthly: pre-aggregated monthly dataframe with 'Month', 'Revenue', and 'Units Sold'
    - save_path: optional path to save the figure (e.g., "exports/dashboards/dashboard.png")
    """

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Top Left - Bar: Monthly Revenue
    axs[0, 0].bar(monthly["Month"], monthly["Revenue"], color="teal")
    axs[0, 0].set_title("Monthly Revenue")
    axs[0, 0].tick_params(axis="x", rotation=45)

    # Top Right - Line: Units Sold Over Time
    axs[0, 1].plot(monthly["Month"], monthly["Units Sold"], marker="o", color="orange")
    axs[0, 1].set_title("Units Sold Over Time")
    axs[0, 1].tick_params(axis="x", rotation=45)

    # Bottom Left - Histogram: Units Sold
    axs[1, 0].hist(df["Units Sold"], bins=10, color="purple", edgecolor="white")
    axs[1, 0].set_title("Distribution of Units Sold")

    # Bottom Right - Scatter: Revenue vs Units Sold
    axs[1, 1].scatter(df["Units Sold"], df["Revenue"], color="crimson", alpha=0.7)
    axs[1, 1].set_title("Units Sold vs Revenue")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    else:
        plt.show()
