"""
MatplotlibMasterPro export viewer.

Browse every figure the notebooks and scripts wrote under ``exports/``::

    streamlit run streamlit_app.py

Sub-folders of ``exports/`` become categories automatically, so new notebooks
or scripts show up without editing this file.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
EXPORTS_DIR = PROJECT_ROOT / "exports"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg"}
VIDEO_SUFFIXES = {".mp4", ".webm"}
DOC_SUFFIXES = {".pdf"}

st.set_page_config(page_title="MatplotlibMasterPro", page_icon="📊", layout="wide")
st.title("MatplotlibMasterPro — export viewer")
st.markdown(
    "Figures produced by the 23 notebooks and the production scripts. "
    "Regenerate them with `python scripts/run_notebooks.py --inplace` or the scripts in `scripts/`."
)


def pretty(name: str) -> str:
    return name.replace("_", " ").strip().title()


def collect_categories(root: Path) -> dict[str, Path]:
    if not root.exists():
        return {}
    categories = {
        pretty(p.name): p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")
    }
    if any(p.is_file() for p in root.iterdir()):
        categories["Top-Level Exports"] = root
    return categories


categories = collect_categories(EXPORTS_DIR)
if not categories:
    st.error(f"No exports found under `{EXPORTS_DIR}`. Run a notebook or script first.")
    st.stop()

with st.sidebar:
    st.header("Browse")
    choice = st.selectbox("Category", list(categories))
    columns = st.slider("Columns", 1, 3, 2)
    show_paths = st.checkbox("Show file paths", value=False)
    st.divider()
    st.caption(
        f"{sum(1 for c in categories.values() for f in c.iterdir() if f.is_file())} files in {len(categories)} categories"
    )

folder = categories[choice]
files = sorted(f for f in folder.iterdir() if f.is_file() and not f.name.startswith("."))
images = [f for f in files if f.suffix.lower() in IMAGE_SUFFIXES]
videos = [f for f in files if f.suffix.lower() in VIDEO_SUFFIXES]
docs = [f for f in files if f.suffix.lower() in DOC_SUFFIXES]

st.subheader(choice)

if images:
    cols = st.columns(columns)
    for i, path in enumerate(images):
        with cols[i % columns]:
            st.image(
                str(path),
                caption=str(path.relative_to(PROJECT_ROOT)) if show_paths else path.stem,
                width="stretch",
            )

if videos:
    st.markdown("### Animations")
    cols = st.columns(min(columns, len(videos)))
    for i, path in enumerate(videos):
        with cols[i % len(cols)]:
            st.video(str(path))
            st.caption(path.name)

if docs:
    st.markdown("### PDF exports")
    for path in docs:
        st.download_button(
            f"Download {path.name}", data=path.read_bytes(), file_name=path.name, mime="application/pdf"
        )

if not (images or videos or docs):
    st.info("This category has no displayable files yet.")
