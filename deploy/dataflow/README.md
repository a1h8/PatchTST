# Dataflow submit (M6)

Run the **same** detection pipeline on Google Cloud Dataflow, unbounded. Nothing
in the pipeline code changes — only the `engine:` block of the config selects the
`dataflow` runner (see [`config/dataflow-streaming.example.yaml`](../../config/dataflow-streaming.example.yaml)).
Dataflow-first is the deliberate M6 order (managed, least ops); Flink-on-K8s is
the self-hosted alternative that comes next.

## Why a custom worker image

Dataflow workers run in isolated containers and must import the pipeline code
(`connectors`, `detection`, `kb`, `pipeline`) plus the Beam SDK harness. The
top-level `Dockerfile` is deliberately Beam-free (it targets the `engine: local`
K3s CronJob), so Dataflow uses its own image built from the Beam SDK base:
[`deploy/dataflow/Dockerfile`](./Dockerfile).

> The base tag (`apache/beam_python3.11_sdk:2.74.0`) **must** match the
> `apache-beam` version used to submit the job — see `requirements-connectors.txt`.

## Prerequisites

- A GCP project with the Dataflow, Compute, and Artifact Registry APIs enabled.
- Two GCS locations: `temp_location` and `staging_location`.
- Credentials on the submitting machine (`gcloud auth application-default login`
  or a service-account key).
- A metrics endpoint reachable from the workers (Mimir — the D2 Mimir-first
  ingress) and an object-store sink root (`s3://…` / `gs://…`), not a
  worker-local path.

## 1. Build & push the worker image

```sh
REGION=europe-west1
PROJECT=my-gcp-project
IMAGE=$REGION-docker.pkg.dev/$PROJECT/patchtst/dataflow-worker:2.74.0

docker build -f deploy/dataflow/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"
```

## 2. Fill in the config

Copy `config/dataflow-streaming.example.yaml` and set, under `engine.options`:
`project`, `region`, `temp_location`, `staging_location`, `job_name`, and
`sdk_container_image` (the `$IMAGE` above). `experiments: [use_runner_v2,
enable_streaming_engine]` enables Runner v2 + Streaming Engine.

## 3. Pre-flight (offline, no submit)

Gate the config before the slow, billable run — this contacts no infra:

```sh
python -m pipeline --check config/dataflow-streaming.example.yaml
```

It catches the boring first-submit failures: the launch-time `apache-beam`
version drifting from the `sdk_container_image` tag (Dataflow rejects workers
whose SDK differs from the launcher — the reason `requirements-connectors.txt`'s
`>=` floor must be pinned to the image tag before submit), a missing/`non-gs://`
`temp_location` or `staging_location`, a worker-unreachable local sink `root`,
and a streaming graph with no window or one that would hang the driver. Exit
code is non-zero on any error. A real submit (next step) runs the same checks
first and aborts on error.

## 4. Submit

```sh
python -m pipeline config/dataflow-streaming.example.yaml
```

`python -m pipeline` builds the pipeline and, because the runner is `dataflow`,
`pipeline.run()` submits the job to Dataflow. The example sets `engine.block:
false`, so the driver returns the job handle immediately instead of blocking on
the long-running stream (with `block` unset/true it would `wait_until_finish()`
and hang until the job drains). The job then appears in the Dataflow console.

## Monitoring

`BeamEngine.run()` returns the `PipelineResult` (the Dataflow job handle). The
pipeline metrics (namespace `pipeline`) are visible in the Dataflow **Job
Metrics** tab: throughput (`rows_in` / `records_out`), event lag
(`event_lag_ms`), and failures (`records_failed`) — alongside Dataflow's own
system lag and watermark.

## Cost & teardown

A streaming Dataflow job runs until drained. Stop it from the console or:

```sh
gcloud dataflow jobs drain <JOB_ID> --region "$REGION"   # finish in-flight windows
gcloud dataflow jobs cancel <JOB_ID> --region "$REGION"  # hard stop
```
