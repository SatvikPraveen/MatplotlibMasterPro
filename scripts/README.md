# Scripts

Production-style entry points built on `mplmasterpro`. Every script runs from the repository root, writes into `exports/<topic>/` and is executed in CI on each push, so they double as integration tests for the package.

| Script | Output | What it demonstrates |
| --- | --- | --- |
| `generate_dashboard.py` | `exports/dashboards/sales_dashboard_automated.png` | GridSpec dashboard with KPI text box, corporate theme |
| `generate_3d_plots.py` | `exports/3d_visualizations/*.png` | Surface, wireframe, 3-D scatter and contour |
| `generate_statistical_plots.py` | `exports/statistical_analysis/*.png` | Box, violin and combined statistical summaries |
| `batch_export.py` | `exports/batch/*.{png,pdf,svg}` | One figure, three formats, publication theme |
| `create_publication_figures.py` | `exports/publication/*.pdf` | IEEE, academic multi-panel and colour-blind-safe figures |
| `run_notebooks.py` | `build/notebooks/*.ipynb` | Headless execution of every notebook (the CI gate) |
| `generate_api_docs.py` | `docs/API.md` | API reference generated from docstrings |

```bash
python scripts/generate_dashboard.py
python scripts/run_notebooks.py --keep-going          # all notebooks
python scripts/run_notebooks.py 17 21 --inplace       # refresh two notebooks' outputs
python scripts/generate_api_docs.py
```

## Writing a new script

- Import from `mplmasterpro` (the scripts add the repository root to `sys.path`, so `pip install -e .` is optional).
- Load data with `mplmasterpro.datasets.load_dataset(...)`.
- Prefer `mplmasterpro.publication.save_figure()` over bare `savefig` so the output carries provenance metadata.
- Create the output directory with `Path(...).mkdir(parents=True, exist_ok=True)` and close figures you do not return.
- Add the script to the `scripts` job in `.github/workflows/ci.yml` and to the table above.
