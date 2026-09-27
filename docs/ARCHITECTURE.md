# Architecture

This document explains how `mplmasterpro` is organised and the decisions behind it, so that new helpers fit in without discussion.

## Goals

1. **One code path for notebooks, scripts and tests.** The old `utils/` folder was reached through `sys.path` hacks and could not be installed, tested in isolation or imported from another project. It is now a package with a `pyproject.toml`.
2. **Composable helpers.** A function that creates its own figure and calls `plt.show()` cannot be placed in a subplot. Every helper accepts `ax=None` and returns its artists instead.
3. **Research features that are hard to get right by hand**: journal sizes, uncertainty bands, colour-blindness checks, provenance metadata.
4. **Verifiable claims.** Anything the README asserts (notebooks run, datasets reproduce, figures carry metadata) is checked by a test or a CI job.

## Package map

```
mplmasterpro/
├── __init__.py       version, curated top-level re-exports
├── _utils.py         get_axes(), finalize(), validate_xy(), ensure_dir()   ← private
├── plot_utils.py     teaching helpers used by notebooks 01–16
├── theme_utils.py    rcParams presets, THEMES registry, theme_context()
├── publication.py    figure_size(), set_size(), add_panel_labels(), save_figure(), despine()
├── stats.py          bootstrap_ci(), plot_mean_ci(), bar_with_ci(), linear_fit_with_ci(),
│                     ecdf_plot(), qq_plot(), raincloud_plot(), add_significance_bar()
├── colors.py         palettes, simulate_cvd(), contrast_ratio(), delta_e(), previews
└── datasets.py       seeded generators, load_dataset(), CLI
```

Dependency direction is strictly downward: `plot_utils` may import `_utils` and `colors`; `stats` and `publication` import `_utils`; nothing imports `plot_utils`. `datasets` depends only on NumPy and pandas.

## The plotting contract

```python
def some_plot(data, *, ..., ax: Axes | None = None, show: bool = False) -> tuple[Figure, Axes]:
    validate_inputs(...)
    fig, ax = get_axes(ax, figsize)      # new figure only if ax is None
    ...draw...
    return finalize(fig, ax, title=..., xlabel=..., legend=..., show=show)
```

- `get_axes` is the single place that decides whether to create a figure.
- `finalize` applies labels, grid, legend, tick rotation and `tight_layout`, and calls `plt.show()` only when asked.
- Functions that draw several axes return `(fig, axes)`; twin-axis helpers return `(fig, (ax_left, ax_right))`.
- `save_*` helpers build the figure through the corresponding plotting function, save, **close** the figure and return the path(s). Tests assert that no figure is left open.
- Animations return the `FuncAnimation`; the caller owns the reference (Matplotlib garbage-collects unreferenced animations).

### Why no implicit `plt.show()`

In Jupyter's inline backend, open figures are rendered at the end of every cell, so `plt.show()` inside a helper is redundant. In scripts it blocks (interactive backends) or does nothing (Agg). Making it opt-in keeps helpers usable in both contexts and in tests.

### Two calling conventions

Notebooks 01–16 pass DataFrames (`multi_line_plot(df, "Month", ["Laptop", "Tablet"])`, `grouped_bar_plot(df=..., category=..., subcategory=..., value=...)`) while library users and the original test suite pass arrays (`multi_line_plot(x, [y1, y2])`, `grouped_bar_plot(labels, {"A": ..., "B": ...})`). Both are supported explicitly, dispatching on `isinstance(first_arg, pd.DataFrame)`. Keep this pattern when adding helpers that are used from the notebooks.

## Themes

Themes are plain functions that mutate `rcParams`, registered in `THEMES` so they can be selected by name. `theme_context(name)` wraps `matplotlib.rc_context()` so a theme can be applied to a single figure without leaking. `reset_theme()` uses `mpl.rcdefaults()` (which deliberately leaves the backend untouched) followed by `plt.style.use("default")`.

Journal presets (`ieee`, `nature`) set `figure.figsize` to the single-column width, `pdf.fonttype = 42` so text stays editable in vector editors, and `savefig.dpi = 600`. Pair them with `publication.figure_size()` for double-column figures.

## Colour science

- Palettes are hex lists; `PALETTES` maps names to them.
- `simulate_cvd()` converts sRGB → linear RGB, applies the Machado et al. (2009) severity-1.0 matrix, and converts back. It accepts colour specs (returns hex) or float arrays such as rendered images (returns an array).
- `contrast_ratio()` implements WCAG 2.1; `rgb_to_lab()` uses the D65 white point and the sRGB matrix from IEC 61966-2-1; `delta_e()` is CIE76 (adequate for "are these distinguishable?" questions; CIEDE2000 is on the roadmap).
- `min_pairwise_distance(palette, cvd=...)` is the number to quote in a paper: the smallest ΔE between any two colours after simulation.

## Statistics

- Bootstrap intervals use the percentile method with `numpy.random.default_rng(seed)`; pass `seed` for reproducible bands.
- `mean_ci(method="t")` uses Student's *t* with *n* − 1 degrees of freedom; `"sem"` and `"std"` are provided because some fields plot them, but the label says which one was drawn.
- `linear_fit_with_ci()` uses `scipy.stats.linregress`; the confidence band is for the mean response and the prediction band for a new observation, both from the standard OLS formulas.
- `raincloud_plot()` builds the half violin by clipping the violin body's path to one side of its position, then overlays a narrow box plot and jittered points. It works with the `orientation=` keyword on Matplotlib ≥ 3.10 and falls back to `vert=` on older versions.

## Publication helpers

- `JOURNAL_WIDTHS_MM` stores (single, double) column widths taken from publisher author guidelines; `figure_size()` converts to inches and applies an aspect ratio (golden ratio by default).
- `save_figure()` writes one file per format and embeds metadata filtered to what each backend accepts: PDF (Title, Author, Subject, Keywords, Creator, …), SVG (Dublin Core fields) and PNG (free-form). The `Subject` carries `git:<short sha>` when the code is run inside a repository.
- `add_panel_labels()` places labels in axes coordinates (default just outside the top-left corner) so they do not collide with data.

## Datasets

The four CSVs are generated from `numpy.random.RandomState(seed)` streams that replicate the original script call-for-call, so the committed files are reproducible byte-for-byte (`tests/test_datasets.py` and the CI "scripts" job verify this). `load_dataset()` reads the CSV when present and regenerates it in memory otherwise, so notebooks work from a shallow checkout.

## Testing strategy

- `tests/conftest.py` forces the Agg backend, resets `rcParams` around every test and closes all figures.
- Tests assert on artists (line colours, patch counts, tick labels, legend presence) rather than on pixels, which keeps them stable across Matplotlib versions.
- Numerical helpers are tested against known values (Lab of sRGB red, WCAG 21:1, slope recovery) and statistical properties (coverage of the true mean, monotone ECDF).
- Notebook execution, scripts and Docker are exercised in CI rather than in pytest to keep the unit suite under five seconds.

## Compatibility policy

- Python ≥ 3.10 (uses `X | Y` unions and `str.removesuffix`).
- Matplotlib ≥ 3.10, NumPy ≥ 1.24 (also tested on 2.x), pandas ≥ 2.0 (also tested on 3.x), SciPy ≥ 1.10.
- Deprecations from the supported Matplotlib range fail the test suite (`filterwarnings = error::MatplotlibDeprecationWarning:mplmasterpro`), so upstream removals are caught before they break users.
