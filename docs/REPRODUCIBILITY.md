# Reproducibility

"Reproducible" is a claim that needs a mechanism behind it. This page lists what is guaranteed, which mechanism enforces it, and how to reproduce any artefact in the repository yourself.

## Guarantees and their enforcement

| Guarantee | Mechanism | Where |
| --- | --- | --- |
| The package installs and imports on Python 3.10–3.13, Linux/macOS/Windows | pytest matrix | `ci.yml` → `test` |
| 110 unit tests pass; Matplotlib deprecations inside the package are errors | pytest, `filterwarnings` | `pyproject.toml`, `tests/` |
| All 23 notebooks execute top to bottom with a fresh kernel | `scripts/run_notebooks.py` | `ci.yml` → `notebooks` |
| Every script in `scripts/` and `examples/` runs to completion | shell loop | `ci.yml` → `scripts` |
| `datasets/*.csv` are exactly what the seeded generators produce | `cmp` of regenerated files; `tests/test_datasets.py` | `ci.yml` → `scripts` |
| The sdist and wheel build cleanly and pass metadata checks | `python -m build`, `twine check` | `ci.yml` → `build` |
| The Docker image builds | buildx | `ci.yml` → `docker` |
| A release tag matches the package version | version check step | `release.yml` |
| Exported figures record their provenance | `save_figure()` metadata | `mplmasterpro/publication.py` |

## Reproducing artefacts locally

### Datasets

```bash
python generate_all_datasets.py --out /tmp/regen
for f in datasets/*.csv; do cmp "$f" "/tmp/regen/$(basename "$f")" && echo "identical: $f"; done
```

The generators use `numpy.random.RandomState(seed)` (the legacy stream) on purpose: it is frozen by NumPy's compatibility policy, so the CSVs are stable across NumPy versions.

### Notebook outputs

```bash
python scripts/run_notebooks.py --keep-going              # executed copies in build/notebooks/
python scripts/run_notebooks.py 17 21 --inplace           # refresh committed outputs for two notebooks
```

Figures in executed notebooks depend on the Matplotlib version (fonts, default DPI, anti-aliasing), so pixel-identical outputs are **not** promised across environments; error-free execution is.

### Exported figures

Notebooks 10, 13, 14, 15 and 16 write into `exports/`. Re-run those notebooks (or the corresponding script) to regenerate them. `exports/research/research_showcase.png` is produced by the snippet in the README.

### Provenance metadata

```bash
python - <<'PY'
import matplotlib.pyplot as plt
from mplmasterpro import save_figure
fig, ax = plt.subplots(); ax.plot([0, 1]); ax.set_title("demo")
print(save_figure(fig, "/tmp/demo", formats=("pdf", "png", "svg")))
PY
exiftool /tmp/demo.pdf | grep -E "Title|Subject|Creator"     # or: pdfinfo /tmp/demo.pdf
grep -o "<dc:title>.*</dc:title>" /tmp/demo.svg
```

Expected keys: `Title` (axes/suptitle), `Creator` (`mplmasterpro <version> / matplotlib <version>`), `Date` (UTC ISO-8601) and `Subject: git:<short-sha>` when run inside the repository.

## Randomness policy

- Library code that draws random numbers (bootstrap, jitter) takes a `seed` argument and uses `numpy.random.default_rng(seed)`. Defaults are `seed=0` for jitter (visual reproducibility) and `seed=None` for the bootstrap (statistical honesty) — pass an explicit seed when a figure must be regenerated exactly.
- Notebooks and scripts seed at the top of the cell that generates data.

## Environment capture

For a paper, record the exact environment alongside the figure:

```bash
pip freeze > figure_env.txt
python -c "import mplmasterpro, matplotlib, numpy, pandas, scipy; print(mplmasterpro.__version__, matplotlib.__version__, numpy.__version__, pandas.__version__, scipy.__version__)"
git rev-parse HEAD
```

The Docker image (`docker build -t matplotlibmasterpro .`) is the strongest option: it pins the base Python, installs ffmpeg and fonts, and runs the same test suite and notebooks as CI.

## Known non-determinism

- Font fallbacks differ between platforms (Times New Roman on Windows/macOS vs. Nimbus Roman/DejaVu Serif on Linux); text metrics therefore vary slightly. The Docker image ships Liberation and DejaVu fonts for consistency.
- `ffmpeg` encodes are not byte-identical across ffmpeg versions; GIF output via Pillow is deterministic for a given Pillow version.
- Interactive notebook 09 relies on `ipympl`; its widget state is not stored in the committed notebook.
