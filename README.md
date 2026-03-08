# 📊 MatplotlibMasterPro

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-darkgreen.svg)](https://www.python.org/)
[![Jupyter Notebooks](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Matplotlib Focused](https://img.shields.io/badge/Matplotlib-100%25-brightgreen.svg)](https://matplotlib.org/)
[![Project Status](https://img.shields.io/badge/Status-Active-success.svg)](#)
[![Open Source](https://img.shields.io/badge/Open%20Source-Yes-critical.svg)](#)
[![Dashboard Ready](https://img.shields.io/badge/Dashboards-Included-blueviolet.svg)](#)
[![Animations](https://img.shields.io/badge/Animations-MP4/GIF-red.svg)](#)
[![Streamlit Compatible](https://img.shields.io/badge/Streamlit-Ready-ff4b4b.svg)](#)
[![Portfolio Project](https://img.shields.io/badge/Use%20Case-Portfolio%20Project-lightgrey.svg)](#)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-blue.svg)](#)

---

## 🧠 Project Overview

**MatplotlibMasterPro** is a complete, portfolio-grade project designed to **master data visualization using `matplotlib.pyplot`**.  
It’s structured to serve both as a:

- 📘 **Self-paced learning notebook series**
- 💼 **Professional showcase project**

Whether you’re revisiting fundamentals or creating complex dashboards — this project brings it all together in one place.

---

## 📁 Project Structure

```bash
MatplotlibMasterPro/
├── notebooks/               # Step-by-step concept notebooks
├── utils/                   # Plotting utility scripts
├── cheatsheets/             # Markdown/PDF visual guides
├── datasets/                # Toy + Realistic datasets
├── exports/                 # Exported plots and dashboards
├── streamlit_app.py         # Streamlit dashboard viewer
├── requirements.txt         # Minimal dependencies to run the project
├── requirements_dev.txt     # Full dev environment
├── Dockerfile               # Dockerized Jupyter environment
├── .dockerignore            # Docker ignore rules
├── .gitignore               # Git ignore rules
├── README.md
└── LICENSE
```

---

## 📚 Notebooks Roadmap

| Notebook                     | Description                                         |
| ---------------------------- | --------------------------------------------------- |
| `01_line_plot.ipynb`         | Basics of `plot()`, labels, legend                  |
| `02_bar_scatter.ipynb`       | Bar charts and scatter plots                        |
| `03_histogram_pie.ipynb`     | Distributions and pie charts                        |
| `04_subplots_axes.ipynb`     | Subplotting and axes control                        |
| `05_customization.ipynb`     | Colors, linestyles, themes                          |
| `06_advanced_plots.ipynb`    | Log plots, heatmaps, fill areas                     |
| `07_annotations.ipynb`       | Labels, arrows, text, highlights                    |
| `08_images_and_grids.ipynb`  | `imshow`, `matshow`, grids                          |
| `09_interactive.ipynb`       | Widgets, sliders, `%matplotlib notebook`            |
| `10_export_style.ipynb`      | Save figures, DPI, formats, themes                  |
| `11_animation.ipynb`         | Animated plots, FuncAnimation, saving MP4/GIF       |
| `12_stats_distribution.ipynb`| Statistical distributions and plots                 |
| `13_comparative_plots.ipynb` | Grouped bars, stacked areas, side-by-side views     |
| `14_colormaps_themes.ipynb`  | Colormaps, gradients, diverging schemes             |
| `15_timeseries.ipynb`        | Time-series: trends, seasonal cycles                |
| `16_dashboards.ipynb`        | Multi-panel dashboards using `subplots`, `gridspec` |
| `17_3d_plots.ipynb`          | 3D scatter, surface, wireframe plots                |
| `18_statistical_plots.ipynb` | Box plots, violin plots, swarm plots                |
| `19_error_visualization.ipynb`| Error bars, confidence intervals, fill between     |
| `20_contour_plots.ipynb`     | Contour, contourf, filled contours, heatmaps        |
| `21_polar_plots.ipynb`       | Polar coordinates, radial plots, circular data      |
| `22_composite_plots.ipynb`   | Layered plots, twin axes, broken axes               |
| `23_inset_zoom.ipynb`        | Inset plots, zoomed views, anchored boxes           |

---

## ✨ Advanced Visualization Features

This project covers **comprehensive matplotlib capabilities** including:

- 📦 **3D Visualizations** — Surface plots, wireframes, 3D scatter plots
- 📊 **Statistical Analysis** — Box plots, violin plots, distribution comparisons
- 📉 **Error Visualization** — Error bars, confidence intervals, uncertainty quantification
- 🗺️ **Field Representation** — Contour plots, filled contours, heatmaps
- 🔵 **Polar & Circular Data** — Radar charts, rose plots, wind roses
- 🔄 **Multi-Scale Plots** — Twin axes, layered visualizations, broken axes
- 🔍 **Detail Views** — Inset plots, zoomed views, magnified regions
- 🎨 **Publication-Ready Themes** — IEEE, academic, colorblind-friendly palettes
- 🎬 **Animations** — FuncAnimation, timeline effects, dynamic visualizations
- 📐 **Complex Layouts** — GridSpec, nested subplots, multi-panel dashboards

---

## 📸 Sample Visualizations

Here are two dashboards from the project:

![🧩 Gridspec Dashboard](exports/dashboards/sales_dashboard_gridspec.png)  
_Advanced layout using `GridSpec` for flexible placement_
<br>

![🪟 Subplots Layout](exports/dashboards/sales_dashboard_subplots_2x2.png)  
_Subplots with shared axes and tight layout for cleaner visuals_
<br>

## 🎞️ Animated Visualizations

Here are animated visualizations exported from the project:

- 🎬 [`product_revenue_bars.mp4`](exports/product_revenue_bars.mp4)  
  _Animated bar chart showing revenue distribution by product_

- 📈 [`revenue_growth.mp4`](exports/revenue_growth.mp4)  
  _Revenue growth over time with animated line movement_

- 📊 [`units_revenue_growth.mp4`](exports/units_revenue_growth.mp4)  
  _Dual-plot animation comparing units sold and revenue growth_

- 🔄 [`revenue_vs_units_scatter.mp4`](exports/revenue_vs_units_scatter.mp4)  
  _Dynamic scatter plot showing correlation over time_

## 🧪 Datasets Created and Used

| Filename           | Description                                  |
| ------------------ | -------------------------------------------- |
| `sales_data.csv`   | Monthly product-wise sales and revenue       |
| `covid_cases.csv`  | Cumulative COVID-19 cases across U.S. states |
| `stock_prices.csv` | OHLC & volume for multiple stock tickers     |
| `weather_data.csv` | Daily city-level temperature and humidity    |

> All datasets are generated using `pandas` and `numpy`, and stored under [`datasets/`](datasets/).

---

## 🛠️ Utilities

- `utils/plot_utils.py` — Custom plot wrappers (comparative, themed, exportable)
- `utils/theme_utils.py` — Reusable themes like `dark`, `minimal`, and `corporate`

---

## 🧾 Cheatsheets

Quick-reference syntax guides available at:

- [`cheatsheets/matplotlib_cheatsheet.md`](cheatsheets/matplotlib_cheatsheet.md)

---

## 📖 Documentation & Resources

This project includes comprehensive documentation to support your learning:

- 📋 **[SETUP.md](SETUP.md)** — Complete environment setup guide with activation instructions
- 💡 **[RECOMMENDATIONS.md](RECOMMENDATIONS.md)** — Prioritized improvement suggestions and project roadmap
- 🔧 **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — Solutions for common matplotlib issues (backends, fonts, performance, etc.)
- 📚 **[RESOURCES.md](RESOURCES.md)** — Curated learning resources (courses, books, communities, datasets)
- 🤝 **[CONTRIBUTING.md](CONTRIBUTING.md)** — Contribution guidelines for the project
- 📜 **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** — Community standards and behavior expectations

---

## 🚀 Getting Started

### **Option 1: Virtual Environment (Recommended)**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SatvikPraveen/MatplotlibMasterPro.git
   cd MatplotlibMasterPro
   ```

2. **Use the quick activation script:**
   ```bash
   source activate.sh
   ```

   Or activate manually:
   ```bash
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Launch JupyterLab:**
   ```bash
   jupyter lab
   ```

📖 **For detailed setup instructions, see [SETUP.md](SETUP.md)**  
💡 **For improvement suggestions, see [RECOMMENDATIONS.md](RECOMMENDATIONS.md)**  
🔧 **For troubleshooting help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)**  
📚 **For learning resources, see [RESOURCES.md](RESOURCES.md)**

### **Option 2: Docker (Isolated Environment)**

See the [Dockerized Setup](#-dockerized-setup) section below.

---

## 🌐 Streamlit App

Explore exported dashboards interactively:

```bash
streamlit run streamlit_app.py
```

Or via Docker:

```bash
docker build -t matplotlibmasterpro .
docker run -p 8501:8501 matplotlibmasterpro
```

---

## 🐳 Dockerized Setup

Run a fully isolated Jupyter + Streamlit environment with ease.

```bash
# Build the container
docker build -t matplotlibmasterpro .

# Launch Jupyter
docker run -p 8888:8888 matplotlibmasterpro
```

> Tokenless access enabled by default. Use `--rm -d` to run in background.

---

## 🚀 Future Enhancements

- [x] Streamlit integration for dashboard browsing
- [x] JupyterLab with Docker
- [x] Advanced 3D visualizations
- [x] Statistical plotting techniques
- [x] Publication-ready themes
- [x] Comprehensive troubleshooting guide
- [x] Curated learning resources
- [ ] PDF report export
- [ ] Pip-installable library version
- [ ] Interactive Plotly/Bokeh integrations
- [ ] Real-world case studies

---

## 💼 License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0). See the [LICENSE](./LICENSE) file for more details.

---

## 🙌 Contributing

Want to contribute?

- ✅ Fork the repo
- 🔧 Create a feature branch
- 🔁 Submit a PR with your improvements
- 🐛 Open issues for bugs or suggestions

---
