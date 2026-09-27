# Tests

110 pytest tests covering every `mplmasterpro` module (89 % line coverage). They run in about four seconds on the Agg backend.

| File | Covers |
| --- | --- |
| `conftest.py` | Forces the Agg backend, isolates `rcParams` per test, closes figures, provides `rng`, `sales_df`, `monthly_df` fixtures |
| `test_plot_utils.py` | Original API tests: line, bar, scatter, histogram, pie, multi-line, grouped bars, error handling, `ax=` composition |
| `test_plot_utils_extended.py` | DataFrame/array calling conventions, drawing into supplied axes, every `save_*` helper closes its figure, animations, GIF fallback, theme registry |
| `test_theme_utils.py` | Each theme applies expected rcParams; palette validity; persistence and switching |
| `test_publication.py` | Journal widths, LaTeX-point sizing, panel-label styles, multi-format export with embedded metadata |
| `test_stats.py` | Bootstrap coverage and determinism, t/SEM/SD nesting, slope recovery, ECDF monotonicity, Q-Q linearity, raincloud components, significance stars |
| `test_colors.py` | Palette uniqueness, CVD simulation properties, WCAG contrast, CIELAB reference values, closest-pair ΔE |
| `test_datasets.py` | Generators reproduce the committed CSVs byte-for-byte; loader, regeneration and CLI |

```bash
pytest                                   # everything
pytest -k stats -v                       # one topic
pytest --cov=mplmasterpro --cov-report=html && open htmlcov/index.html
```

## Conventions

- Assert on artists (colours, patch counts, tick labels, legend presence), not on rendered pixels.
- Numerical helpers are checked against known values or statistical properties, with seeded generators.
- Matplotlib deprecation warnings raised from inside `mplmasterpro` fail the suite (`pyproject.toml → filterwarnings`), so upstream API removals are caught early.
- Notebook execution, scripts and Docker are verified in CI rather than here to keep the unit suite fast.
