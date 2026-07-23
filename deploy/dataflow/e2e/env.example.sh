# Ephemeral Dataflow e2e — copy to env.sh and fill in, then:
#   source deploy/dataflow/e2e/env.sh
#   deploy/dataflow/e2e/e2e.sh setup && deploy/dataflow/e2e/e2e.sh run
#   deploy/dataflow/e2e/e2e.sh teardown        # deletes everything below
#
# Every resource name derives from PROJECT + PREFIX so teardown is deterministic.
# Nothing here is committed with real values — env.sh is gitignored.

# --- required ---------------------------------------------------------------
export PROJECT="my-gcp-project"        # gcloud project id
export REGION="europe-west1"           # Dataflow + Artifact Registry region

# --- knobs (safe defaults for a cheap smoke test) ---------------------------
export PREFIX="patchtst-e2e"           # names buckets/repo/job; keep it unique
export MAX_WORKERS="1"                  # 1 worker = cheapest; the sim load needs no more
export MACHINE_TYPE="n1-standard-2"    # default Dataflow streaming worker

# --- pinned, must match the SDK (do not bump casually) ----------------------
# The image tag == the apache-beam version the launcher submits with (preflight
# enforces this). Keep in lockstep with deploy/dataflow/Dockerfile's FROM tag.
export BEAM_VERSION="2.74.0"

# --- derived (usually leave as-is) ------------------------------------------
export AR_REPO="$PREFIX"
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/dataflow-worker:${BEAM_VERSION}"
export DATAFLOW_BUCKET="${PREFIX}-dataflow"   # temp/staging
export KB_BUCKET="${PREFIX}-kb"               # signal-store sink output
export JOB_NAME="${PREFIX}-stream"
export RENDERED_CONFIG="deploy/dataflow/e2e/config.rendered.yaml"  # gitignored
