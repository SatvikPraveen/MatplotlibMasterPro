"""Tests for the uncertainty-visualisation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mplmasterpro import stats as st


def test_bootstrap_ci_covers_true_mean(rng: np.random.Generator) -> None:
    data = rng.normal(loc=5, scale=2, size=400)
    low, high = st.bootstrap_ci(data, ci=95, seed=0)
    assert low < 5 < high
    assert high - low < 1.0  # n=400, sd=2 → half-width ≈ 0.2
    lo2, hi2 = st.bootstrap_ci(data, ci=95, seed=0)
    assert (low, high) == (lo2, hi2)  # seeded → deterministic


def test_bootstrap_ci_2d_and_validation(rng: np.random.Generator) -> None:
    samples = rng.normal(size=(50, 7))
    low, high = st.bootstrap_ci(samples, seed=1)
    assert low.shape == high.shape == (7,)
    assert np.all(low < high)
    with pytest.raises(ValueError):
        st.bootstrap_ci([1.0])


@pytest.mark.parametrize("method", ["t", "sem", "std", "bootstrap"])
def test_mean_ci_methods_nest_sensibly(method: str, rng: np.random.Generator) -> None:
    data = rng.normal(size=100)
    mean, low, high = st.mean_ci(data, method=method, seed=0)
    assert low <= mean <= high
    if method == "t":
        _, lo_sem, hi_sem = st.mean_ci(data, method="sem")
        assert low < lo_sem and high > hi_sem  # t-interval wider than ±SEM
    with pytest.raises(ValueError):
        st.mean_ci(data, method="magic")


def test_plot_mean_ci_draws_line_and_band(rng: np.random.Generator) -> None:
    x = np.linspace(0, 1, 20)
    samples = rng.normal(size=(10, 20)) + x
    fig, ax = st.plot_mean_ci(x, samples, label="runs", seed=0)
    assert len(ax.lines) == 1 and len(ax.collections) == 1
    assert ax.get_legend() is not None
    with pytest.raises(ValueError):
        st.plot_mean_ci(x, samples[:, :5])


def test_bar_with_ci_summary_and_significance(rng: np.random.Generator) -> None:
    groups = {"ctrl": rng.normal(0, 1, 40), "treat": rng.normal(1, 1, 40)}
    fig, ax, summary = st.bar_with_ci(groups, ylabel="score", title="t")
    assert set(summary) == {"ctrl", "treat"}
    est, low, high = summary["treat"]
    assert low <= est <= high
    assert len(ax.patches) == 2
    st.add_significance_bar(ax, 0, 1, text=0.0004)
    assert any(t.get_text() == "***" for t in ax.texts)
    assert st.p_to_stars(0.03) == "*" and st.p_to_stars(0.2) == "n.s." and st.p_to_stars(0.005) == "**"


def test_linear_fit_recovers_slope(rng: np.random.Generator) -> None:
    x = np.linspace(0, 10, 60)
    y = 3 * x - 2 + rng.normal(scale=0.5, size=x.size)
    fig, ax, res = st.linear_fit_with_ci(x, y)
    assert res["slope"] == pytest.approx(3, abs=0.1)
    assert res["intercept"] == pytest.approx(-2, abs=0.4)
    assert res["r2"] > 0.99 and res["n"] == 60
    assert len(ax.collections) == 3  # scatter + CI band + prediction band
    with pytest.raises(ValueError):
        st.linear_fit_with_ci([1, 2], [1, 2])


def test_ecdf_plot_monotone(rng: np.random.Generator) -> None:
    fig, ax = st.ecdf_plot(rng.normal(size=50), label="x")
    (line,) = ax.lines
    ys = line.get_ydata()
    assert np.all(np.diff(ys) >= 0) and ys[-1] == pytest.approx(1.0)
    fig, ax = st.ecdf_plot(rng.normal(size=50), complementary=True)
    assert ax.get_ylabel().startswith("1")
    with pytest.raises(ValueError):
        st.ecdf_plot([])


def test_qq_plot_normal_data_is_linear(rng: np.random.Generator) -> None:
    fig, ax, res = st.qq_plot(rng.normal(size=500))
    assert res["r"] > 0.99
    assert "norm" in ax.get_xlabel()


@pytest.mark.parametrize("orient", ["h", "v"])
def test_raincloud_plot_components(orient: str, rng: np.random.Generator) -> None:
    groups = {"a": rng.normal(size=30), "b": rng.normal(1, size=30), "c": rng.normal(2, size=30)}
    fig, ax = st.raincloud_plot(groups, orient=orient, title="rain", xlabel="value")
    labels = [t.get_text() for t in (ax.get_xticklabels() if orient == "v" else ax.get_yticklabels())]
    assert labels == ["a", "b", "c"]
    # 3 violin bodies + 3 jittered scatter collections
    assert len(ax.collections) >= 6
    assert ax.get_title() == "rain"
