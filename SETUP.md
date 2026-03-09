# 🚀 Environment Setup Guide

This guide will help you set up the development environment for **MatplotlibMasterPro**.

---

## 🐍 Virtual Environment Setup

A virtual environment named `venv` has been created in this project to isolate dependencies.

### **Activate the Virtual Environment**

#### On macOS/Linux:
```bash
source venv/bin/activate
```

#### On Windows:
```bash
venv\Scripts\activate
```

When activated, you'll see `(venv)` in your terminal prompt.

### **Deactivate the Virtual Environment**
```bash
deactivate
```

---

## 📦 Installing Dependencies

### **Core Dependencies** (required to run notebooks and streamlit app)
```bash
pip install -r requirements.txt
```

This installs:
- `matplotlib` - Plotting library
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `jupyterlab` - Interactive notebooks
- `streamlit` - Dashboard app
- `Pillow` - Image processing

### **Development Dependencies** (optional, for full dev environment)
```bash
pip install -r requirements_dev.txt
```

This includes all dev tools like ipywidgets, GitPython, and more.

---

## 🎯 Quick Start

1. **Activate the environment:**
   ```bash
   source venv/bin/activate   # macOS/Linux
   ```

2. **Launch JupyterLab:**
   ```bash
   jupyter lab
   ```

3. **Or run the Streamlit dashboard:**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Run production scripts:**
   ```bash
   python scripts/generate_dashboard.py
   python scripts/generate_3d_plots.py
   python scripts/batch_export.py
   ```

5. **Try quick examples:**
   ```bash
   python examples/quick_start.py
   python examples/publication_figure.py
   ```

---

## 🐍 Running Python Scripts

The project includes production-ready scripts and minimal examples for different use cases.

### **Production Scripts** (`scripts/`)

These demonstrate real-world automation and batch processing:

```bash
# Generate a comprehensive sales dashboard
python scripts/generate_dashboard.py

# Create 3D visualizations
python scripts/generate_3d_plots.py

# Generate statistical plots
python scripts/generate_statistical_plots.py

# Batch export in multiple formats
python scripts/batch_export.py

# Create publication-ready figures
python scripts/create_publication_figures.py
```

### **Quick Examples** (`examples/`)

Minimal, copy-paste ready scripts for learning:

```bash
# Your first plot (10 lines)
python examples/quick_start.py

# Publication-quality figure
python examples/publication_figure.py

# Process multiple datasets
python examples/batch_process.py

# Apply custom themes
python examples/custom_theme_example.py

# Simple animation
python examples/animation_example.py
```

---

## 🧪 Running Tests

The project includes comprehensive unit tests demonstrating professional testing practices.

### **Install Testing Dependencies**

```bash
pip install pytest pytest-cov
```

Or use the dev requirements:
```bash
pip install -r requirements_dev.txt
```

### **Run All Tests**

```bash
python -m pytest tests/ -v
```

### **Run Specific Test File**

```bash
python -m pytest tests/test_plot_utils.py -v
python -m pytest tests/test_theme_utils.py -v
```

### **Run with Coverage Report**

```bash
python -m pytest tests/ --cov=utils --cov-report=html
```

This creates an HTML coverage report in `htmlcov/index.html`.

---

## 🔍 Verify Installation

Check installed packages:
```bash
pip list
```

Check specific package versions:
```bash
pip show matplotlib pandas numpy
```

---

## 🆘 Troubleshooting

### **Virtual environment not activating?**
Make sure you're in the project directory:
```bash
cd /path/to/MatplotlibMasterPro
```

### **Import errors?**
Ensure the virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### **Jupyter kernel not found?**
Install ipykernel in the virtual environment:
```bash
python -m ipykernel install --user --name=matplotlib-env --display-name="Python (MatplotlibMasterPro)"
```

Then select this kernel from the Jupyter interface.

---

## 📋 Current Environment

- **Python Version:** 3.9.6
- **Virtual Environment:** `venv/` (local to project)
- **Core Packages Installed:**
  - matplotlib 3.9.4
  - pandas 2.3.3
  - numpy 2.0.2
  - jupyterlab 4.5.5
  - streamlit 1.50.0
  - Pillow 11.3.0

---

## 🔄 Updating Dependencies

To update a specific package:
```bash
pip install --upgrade matplotlib
```

To update all packages:
```bash
pip install --upgrade -r requirements.txt
```

---

## 🐳 Alternative: Using Docker

If you prefer Docker, use the included Dockerfile:
```bash
docker build -t matplotlib-master-pro .
docker run -p 8888:8888 -p 8501:8501 matplotlib-master-pro
```

---

## 📝 Notes

- The `venv/` folder is git-ignored and won't be committed
- Always activate the virtual environment before working on the project
- Use `pip freeze > requirements.txt` to save new dependencies (if needed)

Happy plotting! 📊
