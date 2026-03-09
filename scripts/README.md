# 🐍 Scripts

This folder contains **production-ready Python scripts** that demonstrate real-world use cases for matplotlib visualizations.

## 📋 Available Scripts

### 1. `generate_dashboard.py`
Creates a comprehensive multi-panel dashboard from sales data.
```bash
python scripts/generate_dashboard.py
```
**Output:** Professional dashboard in `exports/dashboards/`

### 2. `generate_3d_plots.py`
Batch generates various 3D visualizations (surface, wireframe, scatter).
```bash
python scripts/generate_3d_plots.py
```
**Output:** 3D plots in `exports/3d_visualizations/`

### 3. `generate_statistical_plots.py`
Creates box plots, violin plots, and statistical comparisons.
```bash
python scripts/generate_statistical_plots.py
```
**Output:** Statistical plots in `exports/statistical_analysis/`

### 4. `batch_export.py`
Exports plots in multiple formats (PNG, PDF, SVG) for publication.
```bash
python scripts/batch_export.py
```
**Output:** Multi-format exports in `exports/batch/`

### 5. `create_publication_figures.py`
Generates publication-ready figures with IEEE/academic formatting.
```bash
python scripts/create_publication_figures.py
```
**Output:** Publication figures in `exports/publication/`

---

## 🎯 Use Cases

- **Automated reporting** - Run scripts on schedule to generate updated dashboards
- **Batch processing** - Process multiple datasets at once
- **Production pipelines** - Integrate into data processing workflows
- **Command-line usage** - Run without Jupyter/interactive environment

---

## 🔧 Requirements

All scripts use the project's existing utilities and datasets:
- `utils/plot_utils.py` - Plotting helper functions
- `utils/theme_utils.py` - Custom themes
- `datasets/*.csv` - Sample data

Make sure your virtual environment is activated:
```bash
source venv/bin/activate  # or source activate.sh
```

---

## 💡 Tips

- All scripts can be run independently
- Outputs are saved to the `exports/` directory
- Scripts demonstrate professional coding practices (error handling, logging, documentation)
- Modify parameters at the top of each script to customize output
