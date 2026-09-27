# Roadmap

Planned work, roughly in priority order. Items marked **help wanted** are good first contributions; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Done in 1.0.0

- Installable `mplmasterpro` package with the `(fig, ax)` contract
- Publication sizing, provenance-carrying export, panel labels
- Bootstrap/t intervals, regression bands, raincloud, ECDF, Q-Q plots
- Colour-vision-deficiency simulation, WCAG contrast, CIELAB ΔE palette checks
- Ten themes including IEEE and Nature presets
- Seeded, byte-reproducible datasets
- CI executing notebooks, scripts, tests (3 OSes × 4 Pythons), packaging and Docker
- Executed outputs for all notebooks; Binder-ready `requirements.txt`

## Next (1.1)

- **Notebook 24 – Uncertainty visualisation** using `mplmasterpro.stats` end to end (bootstrap bands, raincloud, significance brackets). *help wanted*
- **Notebook 25 – Journal-ready figures**: `figure_size`, `theme_context("nature")`, panel labels, `save_figure` metadata, checking the result with `cvd_preview`.
- **CIEDE2000** colour difference alongside CIE76 in `colors.delta_e`.
- **`stats.paired_plot`** (slope/“spaghetti” plot for before–after designs) and **`stats.forest_plot`** for effect sizes with CIs.
- **`publication.export_for_latex`**: PGF backend export with a matching `\includegraphics` snippet and font-size guidance.
- **Palette linter CLI** (`mplmasterpro-palette check '#…' '#…' --cvd all`) printing the ΔE matrix.
- PyPI release via trusted publishing once the name is reserved.

## Later

- Documentation site (MkDocs + mkdocstrings) built from `docs/` and the docstrings, deployed from CI.
- Property-based tests (Hypothesis) for the colour-space round trips.
- Image-comparison tests for a small set of reference figures, tolerant to Matplotlib minor versions.
- Optional Plotly/Bokeh exporters for the interactive notebook so it works outside Jupyter.
- Real-world case-study notebooks (climate reanalysis, clinical trial summary, benchmark tables → figures).
- Translation of the cheat sheet.

## Open questions

- Should `plot_utils` be split into topic modules (`timeseries`, `comparative`, `animation`) in 2.0, with the current module kept as a facade? The notebooks import individual names, so a facade would keep them working.
- Should the datasets grow (e.g. longer time series with seasonality) even though that changes the committed CSVs? A new dataset name avoids breaking reproducibility of the old ones.
