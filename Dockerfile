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
RUN pip install -r requirements-kb.txt "pyyaml>=6.0"

# Optional: deep-learning detectors. Off by default to keep the image small.
ARG INSTALL_TORCH=0
COPY requirements-detection-patchtst.txt ./
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
        pip install -r requirements-detection-patchtst.txt; \
    fi

# Application code. Only the packages the deployment runs are copied.
COPY connectors/ ./connectors/
COPY detection/ ./detection/
COPY kb/ ./kb/
COPY pipeline/ ./pipeline/

# Drop privileges.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

# No default CMD: each workload (pipeline CronJob, ingest Job, KB Deployment)
# sets its own command. See deploy/k3s/.
