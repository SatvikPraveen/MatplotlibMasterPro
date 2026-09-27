# Documentation

| Document | Read it when you want to… |
| --- | --- |
| [SETUP.md](SETUP.md) | Install the project (venv, Conda or Docker), run notebooks, scripts, the Streamlit viewer and the tests |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Understand the package layout, the `(fig, ax)` contract, and why helpers behave the way they do |
| [API.md](API.md) | Look up the signature and purpose of every public function (generated from the docstrings) |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | See exactly what CI verifies and how to regenerate datasets, notebook outputs and figures with provenance |
| [ROADMAP.md](ROADMAP.md) | Find planned work or pick something to contribute |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Fix backend, font, animation-writer and Jupyter problems |
| [RESOURCES.md](RESOURCES.md) | Go further: books, courses, style guides, colour tools, datasets |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Open a pull request that passes review and CI |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Know the community standards |

Project-level files worth knowing: [`README.md`](../README.md) (overview and quick start), [`CHANGELOG.md`](../CHANGELOG.md), [`CITATION.cff`](../CITATION.cff), [`SECURITY.md`](../SECURITY.md), [`pyproject.toml`](../pyproject.toml) (dependencies, extras, ruff and pytest configuration) and the [cheat sheet](../cheatsheets/matplotlib_cheatsheet.md).

Regenerate `API.md` after changing public signatures:

```bash
python scripts/generate_api_docs.py
```
