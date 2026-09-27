"""Behavioural tests for plot_utils beyond the basic API: compat modes, saving, animation, themes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.animation import FuncAnimation

from mplmasterpro import plot_utils as pu
from mplmasterpro import theme_utils as tu

# --------------------------------------------------------------------------- calling conventions


def test_multi_line_plot_dataframe_mode(monthly_df: pd.DataFrame) -> None:
    fig, ax = pu.multi_line_plot(
        df=monthly_df,
        x_col="Month",
        y_cols=["Units Sold", "Revenue"],
        labels=["u", "r"],
        colors=["red", "blue"],
    )
    assert [line.get_label() for line in ax.get_lines()] == ["u", "r"]
    assert ax.get_lines()[0].get_color() == "red"
    with pytest.raises(ValueError):
        pu.multi_line_plot(df=monthly_df)


def test_multi_line_plot_mapping_mode() -> None:
    x = np.arange(5)
    fig, ax = pu.multi_line_plot(x=x, ys={"a": x, "b": x * 2})
    assert [line.get_label() for line in ax.get_lines()] == ["a", "b"]


def test_grouped_bar_plot_dataframe_mode(sales_df: pd.DataFrame) -> None:
    fig, ax = pu.grouped_bar_plot(df=sales_df, category="Month", subcategory="Product", value="Revenue")
    assert len(ax.patches) == 12 * 4
    assert ax.get_xlabel() == "Month" and ax.get_ylabel() == "Revenue"
    assert ax.get_legend().get_title().get_text() == "Product"
    fig2, ax2 = pu.grouped_bar_plot(sales_df, "Month", subcategory="Product", value="Units Sold")
    assert len(ax2.patches) == 48
    with pytest.raises(ValueError):
        pu.grouped_bar_plot(["a", "b"], {"g": [1, 2, 3]})


def test_scatter_plot_alias_arguments() -> None:
    x = np.arange(6)
    fig, ax = pu.scatter_plot(x, x, c=x, cmap="plasma", s=x * 10 + 5, use_colorbar=True, colorbar_label="v")
    assert len(fig.axes) == 2  # main axes + colourbar
    fig, ax = pu.scatter_plot(x, x, color_values=x, size=30, cmap="viridis")
    assert len(ax.collections) == 1


def test_pie_chart_argument_order_tolerance() -> None:
    fig, ax = pu.pie_chart(["a", "b"], [1, 3])
    fig2, ax2 = pu.pie_chart([1, 3], ["a", "b"])
    assert len(ax.patches) == len(ax2.patches) == 2
    with pytest.raises(ValueError):
        pu.pie_chart(labels=["a"])


def test_validation_errors() -> None:
    with pytest.raises(ValueError):
        pu.line_plot([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        pu.histogram_plot([])
    with pytest.raises(ValueError):
        pu.log_scale_plot([1, 2], [1, 2], log_axis="z")
    with pytest.raises(ValueError):
        pu.grid_heatmap(np.zeros((2, 2)), ["r"], ["c", "d"])
    with pytest.raises(ValueError):
        pu.grid_plot([lambda ax: None] * 5, nrows=2, ncols=2)


# --------------------------------------------------------------------------- composition & return values


def test_functions_draw_into_supplied_axes(monthly_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3)
    x = np.arange(12)
    pu.line_plot(x, x, ax=axes[0, 0])
    pu.bar_plot(list("abcdefghijkl"), x, ax=axes[0, 1])
    pu.histogram_plot(x, ax=axes[0, 2])
    pu.fill_between_plot(x, x, ax=axes[1, 0])
    pu.plot_timeseries_trend(pd.to_datetime(monthly_df["Month"]), monthly_df["Revenue"], ax=axes[1, 1])
    pu.imshow_matrix(np.eye(3), ax=axes[1, 2], colorbar=False)
    assert len(fig.axes) == 6
    assert all(len(ax.get_children()) > 10 for ax in axes.ravel())


def test_twin_axes_and_grid_plot_return_shapes(monthly_df: pd.DataFrame) -> None:
    fig, (a1, a2) = pu.dual_axis_plot(monthly_df["Month"], monthly_df["Units Sold"], monthly_df["Revenue"])
    assert a2 in a1.get_shared_x_axes().get_siblings(a1)
    fig, (b1, b2) = pu.twin_axes_fill_plot(np.arange(3), np.arange(3), np.arange(3) * 2)
    assert len(b1.collections) == 1 and len(b2.collections) == 1
    fig, axes = pu.grid_plot([lambda ax: ax.plot([0, 1])] * 3, nrows=2, ncols=2, titles=["a", "b", "c"])
    assert len(axes) == 3 and len(fig.axes) == 3  # unused axis removed
    assert axes[2].get_title() == "c"


def test_grid_heatmap_annotation_contrast() -> None:
    data = np.array([[0.0, 1.0]])
    fig, ax = pu.grid_heatmap(data, ["r"], ["lo", "hi"], annot=True, cmap="gray")
    colours = {t.get_text(): t.get_color() for t in ax.texts}
    assert colours["0.00"] == "white" and colours["1.00"] == "black"


def test_annotations_return_artists() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    assert pu.annotate_point(ax, 1, 1, "p").get_text() == "p"
    assert pu.highlight_region(ax, 0, 1, label="band") in ax.patches
    t = pu.label_line(ax, ["a", "b", "c"], [0, 1, 4], "cat", location="center")
    assert t.get_position()[0] == 1  # categorical x → index position
    t2 = pu.label_line(ax, [0.0, 0.5, 1.0], [0, 1, 4], "num", location="right")
    assert t2.get_position()[0] == pytest.approx(1.0)


def test_stats_summary_plots(rng: np.random.Generator) -> None:
    groups = {"a": rng.normal(size=30), "b": rng.normal(size=30)}
    fig, ax = pu.plot_histogram_with_stats(groups["a"], bins=8)
    assert len(ax.lines) == 4 and ax.get_legend() is not None
    fig, ax = pu.plot_boxplot(groups, colors=["red", "blue"], showfliers=False)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
    fig, ax = pu.plot_violinplot(groups)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]


def test_comparative_and_dashboard(sales_df: pd.DataFrame, monthly_df: pd.DataFrame, tmp_path: Path) -> None:
    fig, ax = pu.stacked_area_plot(
        monthly_df["Month"], {"u": monthly_df["Units Sold"].to_numpy(), "r": monthly_df["Revenue"].to_numpy()}
    )
    assert len(ax.collections) == 2
    fig, axes = pu.subplot_groupwise(sales_df, "Product", "Month", "Revenue", ncols=3)
    assert len(axes) == 4 and len(fig.axes) == 4
    fig, axs = pu.create_dashboard(sales_df, monthly_df, save_path=tmp_path / "d" / "dash.png")
    assert axs.shape == (2, 2) and (tmp_path / "d" / "dash.png").exists()


# --------------------------------------------------------------------------- saving helpers


def test_save_helpers_write_files(sales_df: pd.DataFrame, monthly_df: pd.DataFrame, tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    paths = pu.save_plot(fig, "p", folder=tmp_path / "multi", formats=("png", "svg"), verbose=False)
    assert [p.suffix for p in paths] == [".png", ".svg"] and all(p.exists() for p in paths)
    plt.close(fig)

    assert pu.save_grouped_bar_plot(sales_df, "Month", "Product", "Revenue", "t", tmp_path / "g.png").exists()
    assert pu.save_stacked_area_plot(
        monthly_df["Month"], {"u": monthly_df["Units Sold"].to_numpy()}, "x", "y", "t", tmp_path / "s.png"
    ).exists()
    subs = pu.save_subplot_groupwise(sales_df, "Product", "Month", "Revenue", "P", tmp_path / "sub")
    assert len(subs) == 4 and all(p.exists() for p in subs)

    months, revenue = monthly_df["Month"], monthly_df["Revenue"].to_numpy()
    assert pu.save_sequential_bar_plot(months, revenue, tmp_path / "seq.png").exists()
    assert pu.save_diverging_change_plot(
        months, np.diff(revenue, prepend=revenue[0]), tmp_path / "div.png"
    ).exists()
    assert pu.save_qualitative_grouped_bar(sales_df, tmp_path / "q.png").exists()
    assert "_month" not in sales_df.columns  # input must not be mutated

    dates = pd.to_datetime(months)
    series = pd.Series(revenue, index=dates)
    assert pu.save_timeseries_trend(dates, revenue, tmp_path / "ts.png").exists()
    assert pu.save_rolling_stats_plot(series, 3, tmp_path / "roll.png").exists()
    wide = sales_df.pivot_table(index="Month", columns="Product", values="Revenue")
    assert pu.save_multi_product_timeseries(wide, tmp_path / "wide.png").exists()
    with pytest.raises(ValueError):
        pu.save_rolling_stats_plot(series)
    assert plt.get_fignums() == []  # every save helper closes its figure


# --------------------------------------------------------------------------- animation


def test_animations_return_funcanimation(monthly_df: pd.DataFrame) -> None:
    x = np.arange(6)
    anims = [
        pu.animate_line_plot(monthly_df["Month"], monthly_df["Revenue"]),
        pu.animate_dual_line_plot(x, x, x * 2),
        pu.animate_bar_chart(list("abcdef"), x + 1),
        pu.animate_scatter_growth(x, x),
    ]
    assert all(isinstance(a, FuncAnimation) for a in anims)
    for a in anims:
        a._draw_frame(len(x))  # draws the final frame without an event loop


def test_save_animation_gif_and_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    x = np.arange(4)
    gif = pu.save_bar_animation(list("abcd"), x + 1, filename=tmp_path / "bars.gif", fps=4)
    assert gif.exists() and gif.stat().st_size > 0
    monkeypatch.setattr(pu.shutil, "which", lambda _name: None)
    with pytest.warns(RuntimeWarning, match="ffmpeg not found"):
        out = pu.save_line_animation(x, x, filename=tmp_path / "line.mp4", fps=4)
    assert out.suffix == ".gif" and out.exists()


# --------------------------------------------------------------------------- themes


def test_theme_registry_and_context() -> None:
    assert {"ieee", "nature", "colorblind", "default"} <= set(tu.list_themes())
    tu.reset_theme()
    base = plt.rcParams["font.size"]
    with tu.theme_context("ieee"):
        assert plt.rcParams["font.size"] == 8
        assert plt.rcParams["figure.figsize"] == [3.5, 2.625]
    assert plt.rcParams["font.size"] == base
    tu.apply_theme("nature")
    assert plt.rcParams["font.size"] == 7
    with pytest.raises(ValueError, match="Unknown theme"):
        tu.apply_theme("neon")


def test_colorblind_theme_uses_okabe_ito() -> None:
    tu.apply_colorblind_friendly_theme()
    assert plt.rcParams["axes.prop_cycle"].by_key()["color"] == tu.get_colorblind_palette()
    assert tu.get_colorblind_palette()[0] == "#0072B2"
