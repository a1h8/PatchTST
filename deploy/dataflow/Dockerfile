# Dataflow SDK worker image (M6).
#
# Unlike the top-level Dockerfile (Beam-free, `engine: local` on K3s), Dataflow
# workers must run the Beam SDK harness AND import the pipeline code. This image
# is built FROM the Beam SDK container matching the launch-time versions
# (apache-beam 2.74.0 / python 3.11) and layers the connector deps + code on top.
#
# The base tag MUST match the apache-beam version used to submit the job, or the
# runner rejects the workers. Keep it in lockstep with requirements-connectors.txt.
#
# Build & push (Artifact Registry):
#   REGION=europe-west1; PROJECT=my-gcp-project
#   IMAGE=$REGION-docker.pkg.dev/$PROJECT/patchtst/dataflow-worker:2.74.0
#   docker build -f deploy/dataflow/Dockerfile -t $IMAGE .
#   docker push $IMAGE
# then reference it as engine.options.sdk_container_image (see
# config/dataflow-streaming.example.yaml).
FROM apache/beam_python3.11_sdk:2.74.0

ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Connector / pipeline runtime deps (Apache Beam is already in the base image).
COPY requirements-connectors.txt ./
RUN pip install -r requirements-connectors.txt "pyyaml>=6.0" "s3fs>=2024.2"

# Optional deep-learning detectors (off by default — the example uses zscore).
ARG INSTALL_TORCH=0
COPY requirements-detection-patchtst.txt ./
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
        pip install -r requirements-detection-patchtst.txt; \
    fi

# Application code the workers import (source/detector/sink/engine).
COPY connectors/ ./connectors/
COPY detection/ ./detection/
COPY kb/ ./kb/
COPY pipeline/ ./pipeline/

ENV PYTHONPATH=/app

# The base image already sets the SDK harness ENTRYPOINT; do not override it.
