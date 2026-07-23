# Ephemeral Dataflow e2e test

Create everything, run one throwaway streaming job, tear it all down. Validates
the live M6 path — submit → windowing → detection → **GCS Parquet write** — with
a synthetic unbounded source, so it needs **no live Mimir**. Expect **~25–45 min
wall-clock** (mostly image build/push + Dataflow worker startup) and, if you tear
down at the end, **< €1** (a new GCP account's $300 credit covers it entirely).

## Run it

```sh
cp deploy/dataflow/e2e/env.example.sh deploy/dataflow/e2e/env.sh
$EDITOR deploy/dataflow/e2e/env.sh          # set PROJECT + REGION
source deploy/dataflow/e2e/env.sh

deploy/dataflow/e2e/e2e.sh setup            # APIs, AR repo, buckets, build+push image
deploy/dataflow/e2e/e2e.sh run              # preflight --check, then submit (detached)
deploy/dataflow/e2e/e2e.sh status           # job state + GCS output + where metrics live
deploy/dataflow/e2e/e2e.sh teardown         # drain job, delete buckets + repo
```

`env.sh` and the rendered config are gitignored. Every resource name derives
from `PROJECT` + `PREFIX`, so `teardown` only ever removes what `setup` made.

## Why it stays cheap

- **`MAX_WORKERS=1`, `n1-standard-2`** — the synthetic load needs nothing more.
- **`enable_streaming_engine`** — 30 GB worker disk instead of ~400 GB.
- **`block: false`** — submit detaches; you control the run length.
- A streaming job bills **wall-clock, not CPU** — the one thing that runs up a
  bill is forgetting to `teardown`. Two guardrails:
  1. Set a **GCP budget alert** (e.g. €10) before `setup` — the real safety net.
  2. `teardown` drains the job even if you never called `status`.

## What "green" looks like

`status` shows the job `Running`, `gs://$KB_BUCKET/kb/*.parquet` objects
appearing, and — in the Dataflow console **Job metrics**, namespace `pipeline` —
`rows_in` / `records_out` climbing, `event_lag_ms` populated, `records_failed`
at 0. That exercises every joint the DirectRunner tests could not: real worker
provisioning, the SDK-container version match, and the object-store sink write.
