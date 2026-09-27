"""Tests for journal figure sizing, panel labels and metadata-aware export."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from mplmasterpro import publication as pub


def test_figure_size_matches_journal_widths() -> None:
    w, h = pub.figure_size("ieee")
    assert w == pytest.approx(3.5, abs=0.01)
    assert h == pytest.approx(w * pub.GOLDEN_RATIO, abs=0.01)
    w2, _ = pub.figure_size("nature", columns=2)
    assert w2 == pytest.approx(183 / 25.4, abs=0.01)
    w15, _ = pub.figure_size("elsevier", columns=1.5)
    assert pub.figure_size("elsevier")[0] < w15 < pub.figure_size("elsevier", 2)[0]
    _, h_fixed = pub.figure_size("pnas", height_mm=50)
    assert h_fixed == pytest.approx(50 / 25.4, abs=0.01)
    with pytest.raises(ValueError, match="Unknown journal"):
        pub.figure_size("zine")


def test_set_size_scales_with_fraction_and_grid() -> None:
    w, h = pub.set_size(345)  # a typical LaTeX \columnwidth
    assert w == pytest.approx(345 / 72.27)
    assert h == pytest.approx(w * pub.GOLDEN_RATIO)
    w_half, _ = pub.set_size(345, fraction=0.5)
    assert w_half == pytest.approx(w / 2)
    _, h_grid = pub.set_size(345, subplots=(2, 1))
    assert h_grid == pytest.approx(2 * h)


@pytest.mark.parametrize(
    ("style", "expected"),
    [("(a)", ["(a)", "(b)", "(c)", "(d)"]), ("A", ["A", "B", "C", "D"]), ("a)", ["a)", "b)", "c)", "d)"])],
)
def test_add_panel_labels_styles(style: str, expected: list[str]) -> None:
    fig, axes = plt.subplots(2, 2)
    artists = pub.add_panel_labels(axes, style=style)
    assert [t.get_text() for t in artists] == expected
    assert all(t.get_fontweight() == "bold" for t in artists)


def test_add_panel_labels_custom_and_single_axes() -> None:
    fig, ax = plt.subplots()
    (artist,) = pub.add_panel_labels(ax, labels=["Fig. 1"], fontsize=6)
    assert artist.get_text() == "Fig. 1" and artist.get_fontsize() == 6


def test_save_figure_writes_all_formats_with_metadata(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("Provenance test")
    paths = pub.save_figure(
        fig, tmp_path / "fig.png", formats=("png", "pdf", "svg"), dpi=72, metadata={"Author": "Tester"}
    )
    assert [p.suffix for p in paths] == [".png", ".pdf", ".svg"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    svg = (tmp_path / "fig.svg").read_text(encoding="utf-8")
    assert "Provenance test" in svg  # title embedded as <dc:title>
    assert "mplmasterpro" in svg  # creator string
    pdf = (tmp_path / "fig.pdf").read_bytes()
    assert b"Tester" in pdf and b"Provenance test" in pdf


def test_save_figure_keeps_unusual_stems(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    (path,) = pub.save_figure(fig, tmp_path / "run.v2", formats=("png",), include_git_hash=False, close=True)
    assert path.name == "run.v2.png"
    assert not plt.fignum_exists(fig.number)


def test_despine_and_math_fonts() -> None:
    fig, axes = plt.subplots(1, 2)
    pub.despine(axes, offset=5)
    for ax in axes:
        assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
        assert ax.spines["left"].get_visible()
    pub.set_math_fonts("cm")
    assert plt.rcParams["mathtext.fontset"] == "cm"
    assert plt.rcParams["text.usetex"] is False


def test_git_revision_is_string_or_none() -> None:
    rev = pub.git_revision()
    assert rev is None or (isinstance(rev, str) and len(rev) >= 7)
