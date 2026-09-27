# Environment Setup

Three supported ways to run the project: a Python virtual environment (recommended for development), Conda, or Docker. All of them install the `mplmasterpro` package in editable mode so notebooks, scripts and tests import the same code.

## 1. Virtual environment

```bash
git clone https://github.com/SatvikPraveen/MatplotlibMasterPro.git
cd MatplotlibMasterPro
python -m venv venv
source venv/bin/activate          # Windows PowerShell: .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[all]"           # package + notebooks + Streamlit + dev tools
pre-commit install                # optional: run ruff and nbstripout on every commit
```

Python 3.10 or newer is required (3.10–3.13 are tested in CI).

### Choosing extras

| Command | Installs |
| --- | --- |
| `pip install -e .` | Library only: matplotlib ≥ 3.10, numpy, pandas, scipy |
| `pip install -e ".[notebooks]"` | + JupyterLab, ipywidgets, ipympl, Pillow |
| `pip install -e ".[app]"` | + Streamlit viewer |
| `pip install -e ".[dev]"` | + pytest, pytest-cov, ruff, nbclient, pre-commit, build |
| `pip install -e ".[all]"` | Everything above |

`requirements.txt` (runtime) and `requirements_dev.txt` (`-e .[all]`) remain for tools that expect them, such as Binder.

### Animations

Saving `.mp4` animations needs the `ffmpeg` binary on your `PATH` (`brew install ffmpeg`, `apt install ffmpeg`, `conda install ffmpeg`). Without it, `save_animation()` warns and writes an animated GIF instead, so notebook 11 still runs.

## 2. Conda

```bash
conda env create -f environment.yml
conda activate mplmasterpro
```

The environment mirrors `pyproject.toml` and includes ffmpeg.

## 3. Docker

```bash
docker build -t matplotlibmasterpro .
docker run --rm -p 8888:8888 matplotlibmasterpro              # JupyterLab (token-less, for local use)
docker run --rm -p 8501:8501 matplotlibmasterpro streamlit    # export viewer
docker run --rm matplotlibmasterpro test                      # pytest inside the image
docker run --rm matplotlibmasterpro notebooks --keep-going    # execute every notebook
```

Or with Compose, which mounts `notebooks/`, `exports/` and `datasets/` so your work persists:

```bash
docker compose up jupyter
docker compose up streamlit
```

## Running things

| Task | Command |
| --- | --- |
| JupyterLab | `jupyter lab` (open `notebooks/`) |
| One notebook headlessly | `python scripts/run_notebooks.py 17` |
| All notebooks headlessly | `python scripts/run_notebooks.py --keep-going` |
| Refresh committed outputs | `python scripts/run_notebooks.py 17 --inplace` |
| Production scripts | `python scripts/generate_dashboard.py` (see `scripts/README.md`) |
| Examples | `python examples/quick_start.py` |
| Streamlit viewer | `streamlit run streamlit_app.py` |
| Regenerate datasets | `python generate_all_datasets.py` or `mplmasterpro-datasets` |
| Tests | `pytest` (add `--cov=mplmasterpro` for coverage) |
| Lint / format | `ruff check .` / `ruff format .` or `make lint` / `make format` |

`make help` lists every Make target.

## Verifying the installation

```bash
python -c "import mplmasterpro; print(mplmasterpro.__version__)"
python -c "from mplmasterpro import figure_size; print(figure_size('nature', 2))"
pytest -q
```

Expected: the version string, `(7.205, 4.453)`, and `110 passed`.

## Notebook kernel

The notebooks add the repository root to `sys.path` in their first cell, so they work even without `pip install -e .` as long as JupyterLab is started from the repository root. If you register a dedicated kernel:

```bash
python -m ipykernel install --user --name mplmasterpro --display-name "Python (mplmasterpro)"
```

## Troubleshooting

Backend, font and animation issues are collected in [TROUBLESHOOTING.md](TROUBLESHOOTING.md). Two common ones:

- **`RuntimeError: 'widget' is not a recognised GUI loop or backend name`** in notebook 09 → install `ipympl` (`pip install -e ".[notebooks]"`).
- **`Axes.boxplot() got an unexpected keyword argument 'labels'`** → you are on Matplotlib ≥ 3.11 with old code; use `tick_labels=` (the repository already does).
