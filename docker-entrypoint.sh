#!/usr/bin/env bash
# Entrypoint for the MatplotlibMasterPro image.
#   jupyter    (default) start JupyterLab on :8888 without a token
#   streamlit  start the export viewer on :8501
#   test       run the pytest suite
#   notebooks  execute every notebook headlessly
#   anything else is executed verbatim (e.g. `bash`, `python -c ...`)
set -euo pipefail

case "${1:-jupyter}" in
  jupyter)
    exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser \
      --ServerApp.token='' --ServerApp.password='' --ServerApp.root_dir=/app ;;
  streamlit)
    exec streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true ;;
  test)
    shift; exec python -m pytest "$@" ;;
  notebooks)
    shift; exec python scripts/run_notebooks.py "$@" ;;
  *)
    exec "$@" ;;
esac
