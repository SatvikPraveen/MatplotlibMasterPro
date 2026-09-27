# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-09-27

First research-grade release.

### Added
- Installable `mplmasterpro` package (`pip install -e .`) replacing the `utils/` folder.
- `mplmasterpro.publication`: journal column widths (IEEE, Nature, Science, Elsevier, PNAS, ACM, APS, PLOS, Springer), `set_size()` for LaTeX `\columnwidth`, `add_panel_labels()`, `save_figure()` with embedded title/author/git-hash metadata, `despine()`, `set_math_fonts()`.
- `mplmasterpro.stats`: `bootstrap_ci()`, `mean_ci()`, `plot_mean_ci()`, `bar_with_ci()`, `add_significance_bar()`, `linear_fit_with_ci()` with confidence and prediction bands, `ecdf_plot()`, `qq_plot()`, `raincloud_plot()`.
- `mplmasterpro.colors`: Okabe–Ito, Paul Tol and IBM palettes, `simulate_cvd()` (Machado et al. 2009), WCAG `contrast_ratio()`, CIELAB `delta_e()` and `min_pairwise_distance()`, `cvd_preview()`.
- `mplmasterpro.datasets`: seeded generators that reproduce the committed CSVs byte-for-byte, `load_dataset()`, `mplmasterpro-datasets` console script.
- `theme_context()` / `apply_theme()` registry plus a Nature theme.
- Test suite of 110 tests (89 % line coverage), executed on Python 3.10–3.13 across Linux, macOS and Windows.
- CI that also executes all 23 notebooks, every script and example, verifies dataset reproducibility, builds the wheel and the Docker image.
- `scripts/run_notebooks.py` headless notebook runner; `Makefile`, `pre-commit` config, `environment.yml`, `docker-compose.yml`, `CITATION.cff`, `SECURITY.md`.
- Executed outputs for notebooks 17–23 so they render on GitHub.

### Changed
- All plotting helpers accept `ax=None`, return `(fig, ax)` and no longer call `plt.show()` implicitly.
- `multi_line_plot()` and `grouped_bar_plot()` accept both DataFrame and array/dict calling conventions.
- Colour-blind palette corrected to the canonical Okabe–Ito values.
- Minimum Matplotlib is 3.10; `boxplot(labels=…, vert=…)` replaced by `tick_labels=` / `orientation=`.
- Dockerfile rebuilt: non-root user, ffmpeg, healthcheck, multi-command entrypoint.
- `pyproject.toml` is the single source of dependency truth; `requirements*.txt` delegate to it.

### Fixed
- 22 of the 43 original unit tests failed against an API that did not exist; the API now matches the tests.
- `scripts/generate_dashboard.py` required a `Units_Sold` column the dataset never had.
- Duplicate `grouped_bar_plot` definitions and mid-file imports in the old `plot_utils`.
- Removed use of `np.row_stack` and `plt.cm.get_cmap`, both deleted upstream.
- Commit history attributed uniformly to the maintainer.

## [0.x] - 2025

Notebook-only project: 23 notebooks, `utils/` helpers, Streamlit viewer, Docker image.

[1.0.0]: https://github.com/SatvikPraveen/MatplotlibMasterPro/releases/tag/v1.0.0
