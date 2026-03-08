#!/bin/bash
# 🚀 Quick activation script for MatplotlibMasterPro virtual environment

echo "🐍 Activating virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📦 Python location: $(which python)"
echo "📊 Matplotlib version: $(python -c 'import matplotlib; print(matplotlib.__version__)')"
echo ""
echo "💡 Quick commands:"
echo "   - Launch JupyterLab:  jupyter lab"
echo "   - Run Streamlit app:  streamlit run streamlit_app.py"
echo "   - Deactivate:         deactivate"
echo ""
