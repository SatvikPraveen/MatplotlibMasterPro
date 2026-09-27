# syntax=docker/dockerfile:1.7
# Reproducible JupyterLab + Streamlit environment for MatplotlibMasterPro.
#
#   docker build -t matplotlibmasterpro .
#   docker run --rm -p 8888:8888 matplotlibmasterpro                 # JupyterLab
#   docker run --rm -p 8501:8501 matplotlibmasterpro streamlit       # Streamlit viewer
#   docker run --rm matplotlibmasterpro test                         # run the test suite
#
# Or use `docker compose up jupyter` / `docker compose up streamlit`.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# ffmpeg for MP4 animation export; fonts so IEEE/Nature themes have serif/sans fallbacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core fonts-liberation tini \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 mpl
WORKDIR /app

# Install dependencies first so source edits do not invalidate the layer cache.
COPY pyproject.toml README.md LICENSE ./
COPY mplmasterpro ./mplmasterpro
RUN pip install --upgrade pip \
    && pip install -e ".[notebooks,app,dev]"

COPY --chown=mpl:mpl . .
RUN chown -R mpl:mpl /app
USER mpl

EXPOSE 8888 8501

COPY --chown=mpl:mpl docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["jupyter"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import mplmasterpro" || exit 1
