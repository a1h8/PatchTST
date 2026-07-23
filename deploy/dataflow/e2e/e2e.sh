#!/usr/bin/env bash
# Ephemeral Dataflow end-to-end test — create everything, run, tear it all down.
#
#   source deploy/dataflow/e2e/env.sh        # your copy of env.example.sh
#   deploy/dataflow/e2e/e2e.sh setup         # APIs, repo, buckets, build+push image  (~5-8 min)
#   deploy/dataflow/e2e/e2e.sh run           # preflight + submit (detached)          (~5 min to start)
#   deploy/dataflow/e2e/e2e.sh status        # job state + where the GCS output lands
#   deploy/dataflow/e2e/e2e.sh teardown      # drain job + delete buckets & repo      (~3-5 min)
#   deploy/dataflow/e2e/e2e.sh all           # setup -> run (then teardown yourself)
#
# Idempotent: re-running setup skips resources that already exist. teardown is
# safe to run whether or not a job is live. Nothing here disables APIs (cheap,
# and toggling them is noisy) or touches anything outside the $PREFIX names.
set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo="$(cd "$_here/../../.." && pwd)"

# Auto-source env.sh if present and the vars are not already exported.
if [[ -z "${PROJECT:-}" && -f "$_here/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$_here/env.sh"
fi

_require_vars() {
  local missing=0
  for v in PROJECT REGION PREFIX BEAM_VERSION IMAGE AR_REPO \
           DATAFLOW_BUCKET KB_BUCKET JOB_NAME RENDERED_CONFIG MAX_WORKERS \
           MACHINE_TYPE; do
    if [[ -z "${!v:-}" ]]; then echo "  missing env: $v"; missing=1; fi
  done
  [[ $missing -eq 0 ]] || { echo "source your env.sh first (see env.example.sh)"; exit 2; }
}

_require_tools() {
  for t in gcloud docker envsubst python; do
    command -v "$t" >/dev/null || { echo "missing tool: $t"; exit 2; }
  done
}

_gc() { gcloud --project "$PROJECT" "$@"; }

_render() {
  # Only substitute our known vars so any literal $ in the template is safe.
  envsubst '${PROJECT} ${REGION} ${IMAGE} ${DATAFLOW_BUCKET} ${KB_BUCKET} ${JOB_NAME} ${MAX_WORKERS} ${MACHINE_TYPE}' \
    < "$_here/config.template.yaml" > "$_repo/$RENDERED_CONFIG"
  echo "rendered -> $RENDERED_CONFIG"
}

_job_id() {
  _gc dataflow jobs list --region "$REGION" --status active \
    --filter="name=$JOB_NAME" --format='value(JOB_ID)' 2>/dev/null | head -n1
}

cmd_setup() {
  _require_vars; _require_tools
  echo "== enable APIs =="
  _gc services enable dataflow.googleapis.com compute.googleapis.com \
    artifactregistry.googleapis.com storage.googleapis.com

  echo "== Artifact Registry repo ($AR_REPO) =="
  _gc artifacts repositories describe "$AR_REPO" --location "$REGION" >/dev/null 2>&1 || \
    _gc artifacts repositories create "$AR_REPO" --location "$REGION" \
      --repository-format=docker --description="patchtst dataflow e2e (ephemeral)"
  gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

  echo "== GCS buckets =="
  for b in "$DATAFLOW_BUCKET" "$KB_BUCKET"; do
    _gc storage buckets describe "gs://$b" >/dev/null 2>&1 || \
      _gc storage buckets create "gs://$b" --location "$REGION"
  done

  echo "== build + push worker image ($IMAGE) =="
  echo "   (base tag must == BEAM_VERSION=$BEAM_VERSION; preflight re-checks)"
  docker build -f "$_repo/deploy/dataflow/Dockerfile" -t "$IMAGE" "$_repo"
  docker push "$IMAGE"
  echo "== setup done =="
}

cmd_run() {
  _require_vars; _require_tools
  _render
  echo "== preflight (offline gate) =="
  ( cd "$_repo" && python -m pipeline --check "$RENDERED_CONFIG" )
  echo "== submit (detached) =="
  ( cd "$_repo" && python -m pipeline "$RENDERED_CONFIG" )
  echo
  echo "submitted. watch it:"
  echo "  deploy/dataflow/e2e/e2e.sh status"
  echo "  console: https://console.cloud.google.com/dataflow/jobs?project=$PROJECT"
}

cmd_status() {
  _require_vars
  echo "== active jobs named $JOB_NAME =="
  _gc dataflow jobs list --region "$REGION" --status active \
    --filter="name=$JOB_NAME" \
    --format='table(JOB_ID, STATE, CREATE_TIME)'
  echo "== sink output (first objects) =="
  _gc storage ls "gs://$KB_BUCKET/kb/**" 2>/dev/null | head -n 5 || echo "  (nothing yet)"
  echo "metrics: Dataflow console -> Job metrics -> namespace 'pipeline'"
  echo "  rows_in / records_out / event_lag_ms / records_failed"
}

cmd_teardown() {
  _require_vars
  local jid; jid="$(_job_id || true)"
  if [[ -n "$jid" ]]; then
    echo "== draining job $jid (finish in-flight windows) =="
    _gc dataflow jobs drain "$jid" --region "$REGION" || \
      _gc dataflow jobs cancel "$jid" --region "$REGION" || true
    echo "  drain requested; it may take a few minutes to stop billing"
  else
    echo "== no active job named $JOB_NAME =="
  fi
  echo "== delete buckets =="
  for b in "$DATAFLOW_BUCKET" "$KB_BUCKET"; do
    _gc storage rm -r "gs://$b" 2>/dev/null || echo "  gs://$b already gone"
  done
  echo "== delete Artifact Registry repo =="
  _gc artifacts repositories delete "$AR_REPO" --location "$REGION" --quiet 2>/dev/null || \
    echo "  repo $AR_REPO already gone"
  rm -f "$_repo/$RENDERED_CONFIG"
  echo "== teardown done (APIs left enabled) =="
}

case "${1:-}" in
  setup)    cmd_setup ;;
  run)      cmd_run ;;
  status)   cmd_status ;;
  teardown) cmd_teardown ;;
  all)      cmd_setup && cmd_run ;;
  *) echo "usage: $0 {setup|run|status|teardown|all}"; exit 2 ;;
esac
