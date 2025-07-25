# 📊 streamlit_app.py — MatplotlibMasterPro Viewer

import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path

# 📁 Set project root
PROJECT_ROOT = Path(__file__).resolve().parent
EXPORTS_DIR = PROJECT_ROOT / "exports"

# 🧱 Layout
st.set_page_config(page_title="MatplotlibMasterPro Dashboard", layout="wide")
st.title("📊 MatplotlibMasterPro")
st.markdown(
    """
    Welcome to **MatplotlibMasterPro** — your all-in-one, portfolio-grade project for mastering data visualization with `matplotlib`.

    Explore exported dashboards, understand layout strategies, and build powerful multi-panel plots step-by-step.
    """
)

# 🔍 Helper: Display image if it exists
def display_image(image_path, caption):
    if image_path.exists():
        st.image(str(image_path), caption=caption, use_container_width=True)
    else:
        st.warning(f"❌ Missing: {caption}")

# 📁 Exported categories
categories = {
    "Grouped & Stacked Plots": EXPORTS_DIR / "comparative_plots",
    "Color Maps & Themes": EXPORTS_DIR / "colormaps_themes",
    "Time Series": EXPORTS_DIR / "timeseries_plots",
    "Dashboards": EXPORTS_DIR / "dashboards",
}

# 📦 Sidebar: Select section
section = st.sidebar.selectbox("📁 Choose Plot Category", list(categories.keys()))
folder_path = categories[section]

# 🖼️ Display images in selected folder
if folder_path.exists():
    img_files = sorted(folder_path.glob("*.png"))
    if img_files:
        for img_path in img_files:
            display_image(img_path, caption=img_path.name)
    else:
        st.info("No exported plots found in this category yet.")
else:
    st.error(f"Folder not found: `{folder_path}`")
