"""Tests for palettes, CVD simulation and the colour-science helpers."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest

from mplmasterpro import colors


@pytest.mark.parametrize("name", sorted(colors.PALETTES))
def test_palettes_are_valid_unique_hex(name: str) -> None:
    pal = colors.get_palette(name)
    assert len(pal) >= 5
    assert len(set(pal)) == len(pal)
    for c in pal:
        assert c.startswith("#") and len(c) == 7


def test_get_palette_cycles_and_rejects_unknown() -> None:
    assert len(colors.get_palette("ibm", 12)) == 12
    assert colors.get_palette("ibm", 2) == colors.IBM_COLORBLIND[:2]
    with pytest.raises(ValueError, match="Unknown palette"):
        colors.get_palette("not-a-palette")


def test_set_palette_and_context_restore_cycle() -> None:
    before = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    with colors.palette_context("tol_bright") as pal:
        assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == pal
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == before
    colors.set_palette(["#000000", "#ffffff"])
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == ["#000000", "#ffffff"]


def test_simulate_cvd_shapes_and_greys_are_preserved() -> None:
    hexes = colors.simulate_cvd(colors.OKABE_ITO, "deuteranopia")
    assert len(hexes) == len(colors.OKABE_ITO)
    # neutral greys stay (almost) neutral under every deficiency
    for kind in ("protanopia", "deuteranopia", "tritanopia"):
        (grey,) = colors.simulate_cvd("#808080", kind)
        rgb = np.array(mpl.colors.to_rgb(grey))
        assert np.ptp(rgb) < 0.05, (kind, grey)
    img = np.random.default_rng(0).random((3, 4, 3))
    out = colors.simulate_cvd(img, "tritanopia")
    assert isinstance(out, np.ndarray) and out.shape == (3, 4, 3)
    assert out.min() >= 0 and out.max() <= 1
    with pytest.raises(ValueError):
        colors.simulate_cvd("#ff0000", "monochromacy")


def test_deuteranopia_collapses_red_green() -> None:
    red, green = colors.simulate_cvd(["#ff0000", "#00aa00"], "deuteranopia")
    assert colors.delta_e(red, green) < colors.delta_e("#ff0000", "#00aa00")


def test_wcag_contrast_and_luminance() -> None:
    assert colors.relative_luminance("white") == pytest.approx(1.0)
    assert colors.relative_luminance("black") == pytest.approx(0.0)
    assert colors.contrast_ratio("black", "white") == pytest.approx(21.0)
    assert colors.contrast_ratio("white", "black") == pytest.approx(21.0)
    assert colors.contrast_ratio("#777777", "#777777") == pytest.approx(1.0)
    assert colors.text_color_for("#001f3f") == "white"
    assert colors.text_color_for("#ffdc00") == "black"


def test_lab_conversion_reference_values() -> None:
    L, a, b = colors.rgb_to_lab("white")
    assert pytest.approx(100, abs=0.05) == L and abs(a) < 0.05 and abs(b) < 0.05
    L, a, b = colors.rgb_to_lab("red")  # canonical sRGB red ≈ (53.2, 80.1, 67.2)
    assert (L, a, b) == pytest.approx((53.2, 80.1, 67.2), abs=0.3)
    assert colors.delta_e("red", "red") == 0


def test_min_pairwise_distance_reports_closest_pair() -> None:
    d, pair = colors.min_pairwise_distance(["#000000", "#ffffff", "#010101"])
    assert set(pair) == {"#000000", "#010101"}
    assert d < 1
    d_normal, _ = colors.min_pairwise_distance(colors.OKABE_ITO)
    d_deutan, _ = colors.min_pairwise_distance(colors.OKABE_ITO, cvd="deuteranopia")
    assert d_deutan <= d_normal
    assert d_deutan > 10  # Okabe–Ito stays distinguishable under deuteranopia
    with pytest.raises(ValueError):
        colors.min_pairwise_distance(["#000000"])


def test_previews_create_expected_axes() -> None:
    fig, ax = colors.preview_palette("okabe_ito", title="p")
    assert len(ax.patches) == len(colors.OKABE_ITO)
    fig = colors.cvd_preview(["#ff0000", "#00ff00", "#0000ff"])
    assert len(fig.axes) == 4
