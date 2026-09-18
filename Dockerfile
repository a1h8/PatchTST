# Pipeline + KB + ingest runtime image for the K3s deployment.
#
# Deliberately Beam-free: the K3s deployment runs `engine: local`, so Apache
# Beam (the heaviest connector dep) is not installed. The image therefore ships
# the local engine, the Parquet sink, the Mimir source, the CSV ingest CLI and
# the KB HTTP service.
#
# Build (CPU, dependency-light — zscore detector works out of the box):
#   docker build -t patchtst-pipeline:dev .
#
# Build with PatchTST/reconstruction detectors (pulls torch, large image):
#   docker build --build-arg INSTALL_TORCH=1 -t patchtst-pipeline:torch .
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# KB stack (pyarrow + duckdb + fastapi + uvicorn + httpx) and pipeline YAML
# support cover the local-engine cycle, the Parquet sink and the KB service.
# requirements-connectors.txt (Apache Beam) is intentionally NOT installed.
COPY requirements-kb.txt ./
# cramjam: snappy compression for connectors/ingest's live remote-write push
# (requirements-ingest.txt) — a pure wheel, no libsnappy system package needed
# in this slim image, unlike python-snappy. numpy: detection/detector.py
# imports it unconditionally at module load, so even the "dependency-free"
# ZScoreDetector needs it despite torch/transformers being INSTALL_TORCH-gated.
# Neither was previously installed: the ingest-seed Job and every pipeline
# CronJob tick failed on every real deployment, only caught by actually
# running this on a real cluster (k3s).
RUN pip install -r requirements-kb.txt "pyyaml>=6.0" "cramjam>=2.7" "numpy>=1.23"

# Optional: deep-learning detectors. Off by default to keep the image small.
ARG INSTALL_TORCH=0
COPY requirements-detection-patchtst.txt ./
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
        pip install -r requirements-detection-patchtst.txt; \
    fi

# Application code. Only the packages the deployment runs are copied.
# inference/ is required even for the zscore-only path: detection/__init__.py
# unconditionally imports ForecastInferenceDetector, which imports `inference`
# at module load — previously missing here entirely (ModuleNotFoundError on
# every real deployment). It's cheap: numpy only at import time, torch stays
# lazy inside PatchTSTInference's methods.
COPY connectors/ ./connectors/
COPY detection/ ./detection/
COPY inference/ ./inference/
COPY kb/ ./kb/
COPY pipeline/ ./pipeline/

# Drop privileges.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

# No default CMD: each workload (pipeline CronJob, ingest Job, KB Deployment)
# sets its own command. See deploy/k3s/.
