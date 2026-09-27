# Convenience targets. `make help` lists them.
PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Editable install with every optional extra
	$(PIP) install -e ".[all]"
	pre-commit install

lint:  ## Ruff lint + format check
	ruff check mplmasterpro tests scripts examples generate_all_datasets.py streamlit_app.py
	ruff format --check mplmasterpro tests

format:  ## Auto-fix lint issues and format
	ruff check --fix mplmasterpro tests scripts examples generate_all_datasets.py streamlit_app.py
	ruff format mplmasterpro tests scripts examples generate_all_datasets.py streamlit_app.py

test:  ## Run the test suite with coverage
	$(PYTHON) -m pytest --cov=mplmasterpro --cov-report=term-missing

notebooks:  ## Execute every notebook headlessly into build/notebooks
	$(PYTHON) scripts/run_notebooks.py --keep-going

notebooks-refresh:  ## Re-execute notebooks in place (updates committed outputs)
	$(PYTHON) scripts/run_notebooks.py --inplace --keep-going

scripts:  ## Run every production script
	@for f in scripts/generate_dashboard.py scripts/generate_3d_plots.py scripts/generate_statistical_plots.py scripts/batch_export.py scripts/create_publication_figures.py; do echo "== $$f"; $(PYTHON) $$f || exit 1; done

datasets:  ## Regenerate datasets/*.csv from the seeded generators
	$(PYTHON) generate_all_datasets.py

build:  ## Build sdist and wheel into dist/
	$(PYTHON) -m build

docker:  ## Build the Docker image
	docker build -t matplotlibmasterpro .

lab:  ## Launch JupyterLab
	jupyter lab

app:  ## Launch the Streamlit export viewer
	streamlit run streamlit_app.py

clean:  ## Remove caches and build artefacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov coverage.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: help install lint format test notebooks notebooks-refresh scripts datasets build docker lab app clean
