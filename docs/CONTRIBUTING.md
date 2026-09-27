# Contributing to MatplotlibMasterPro

Thank you for considering a contribution. This page explains what is welcome, how the code is organised, and what a pull request needs to pass.

## What to contribute

- **New helpers** in `mplmasterpro/` — especially uncertainty visualisation, accessibility checks and journal-specific presets.
- **Notebooks** that teach a technique not yet covered (see [ROADMAP.md](ROADMAP.md) for ideas). Keep them runnable top to bottom with a fresh kernel.
- **Bug reports** with a minimal reproducible example — use the issue template.
- **Documentation**: corrections, clearer explanations, translated cheat sheets.

## Development setup

```bash
git clone https://github.com/<you>/MatplotlibMasterPro.git
cd MatplotlibMasterPro
python -m venv venv && source venv/bin/activate
pip install -e ".[all]"
pre-commit install
make test
```

## The plotting contract

Every public plotting function in `mplmasterpro` follows the same rules, and reviewers will ask for them:

1. **Accept `ax=None`.** Use `get_axes(ax, figsize)` from `mplmasterpro._utils`; create a figure only when no axes were supplied so the function can be composed into larger layouts.
2. **Return what you create.** `(fig, ax)` for single-axes plots, `(fig, axes)` for grids, the `FuncAnimation` for animations, and the written `Path`s for `save_*` helpers. Never return `None` from a plotting function.
3. **Do not call `plt.show()`** unless the caller passes `show=True`. Jupyter displays open figures automatically.
4. **Validate inputs early** with a clear `ValueError` (`validate_xy` covers the common case).
5. **Type hints and a docstring** on every public function; the first docstring line is what `docs/API.md` shows.
6. **Do not mutate caller data.** Copy DataFrames before adding columns.
7. **Stay within the supported Matplotlib range** (≥ 3.10). Avoid arguments deprecated upstream (`boxplot(labels=…)`, `vert=…`, `plt.cm.get_cmap`).

## Checks a pull request must pass

| Check | Local command |
| --- | --- |
| Lint and formatting | `make lint` (`ruff check`, `ruff format --check`) |
| Tests | `make test` — add tests under `tests/` for new behaviour |
| Notebooks you touched | `python scripts/run_notebooks.py <nn> --inplace`, then commit the executed notebook |
| Datasets unchanged | `python generate_all_datasets.py --out /tmp/d && cmp datasets/sales_data.csv /tmp/d/sales_data.csv` (CI does this for all four) |
| Changelog | Add a line under **Unreleased** in `CHANGELOG.md` |

CI runs the same steps on Linux, macOS and Windows.

## Notebook conventions

- First cell: imports, then the two-line `sys.path` bootstrap, then `from mplmasterpro... import ...`.
- Load data with `mplmasterpro.datasets.load_dataset("sales_data")` rather than hard-coded paths.
- End a cell whose last statement is a plotting call with `;` to suppress the echoed `(fig, ax)`.
- Write exports under `exports/<topic>/` and keep individual files under ~1 MB.
- Re-execute before committing so outputs match the code; `nbstripout` (via pre-commit) removes execution counters but keeps outputs.

## Commit messages

Use an imperative subject line (`Add raincloud_plot orientation option`), wrap the body at 72 characters and explain *why*, not just *what*. Reference issues with `Fixes #123`.

## Releasing (maintainers)

1. Update `__version__` in `mplmasterpro/__init__.py`, `CITATION.cff` and `CHANGELOG.md`.
2. `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z`.
3. The release workflow verifies the tag matches the package version, builds the distribution and publishes a GitHub release.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind, be specific, assume good faith.

## Questions

Open a discussion or email satvikpraveen707@gmail.com.
