# Kafka demo (connector C7)

Runs the KafkaSource path (D2: on-demand, low-latency alternative to Mimir)
side by side with the existing Mimir path, against a single-node Redpanda
broker — no ZooKeeper, no separate cluster to operate:

```
kafka-producer-seed Job ──produce──▶ Redpanda ──consume──▶ pipeline-kafka Job ──▶ KB datalake
```

Same shared `kb-datalake` PVC and `kb` service as `deploy/k3s`, so
`signal_history` serves signals from both the Mimir and Kafka paths — this is
what "swap the source, nothing else changes" (the SPI's point) looks like end
to end.

## 1. Prerequisite

The base stack must already be applied (namespace, `kb-datalake` PVC, `kb`
service):

```sh
kubectl apply -k deploy/k3s
```

## 2. Build the image and deploy

Same image as `deploy/k3s`, now with `kafka-python` installed (see the root
`Dockerfile`):

```sh
docker build -t patchtst-pipeline:dev .
kubectl apply -k deploy/kafka
```

Wait for Redpanda to be ready:

```sh
kubectl -n patchtst rollout status deploy/redpanda
```

## 3. Seed the topic and run detection

The `kafka-producer-seed` Job runs once on apply (60 points, 60s apart, a CPU
spike near the end — the same shape as `ingest-seed`, group `node2/demo` so
it's distinguishable from the Mimir path's `node1/demo`). Re-seed any time:

```sh
kubectl -n patchtst delete job kafka-producer-seed --ignore-not-found
kubectl apply -k deploy/kafka
```

`pipeline-kafka` is a plain Job, not a CronJob: `KafkaSource.read()` drains
whatever's on the topic within `poll_timeout_s` (no rolling time window to
recompute the way the Mimir CronJob has), so re-run it manually after
re-seeding:

```sh
kubectl -n patchtst delete job pipeline-kafka --ignore-not-found
kubectl apply -k deploy/kafka
kubectl -n patchtst logs job/pipeline-kafka -f
```

## 4. Verify

```sh
kubectl -n patchtst port-forward svc/kb 8080:80 &
curl 'http://localhost:8080/api/v1/signals/history?entity=node2/demo'
```

Expect `sim_cpu` flagged `severity=warning` (the injected spike) and `sim_mem`
`normal` — the same detection outcome as the Mimir path, proving the source
swap changed nothing else in the cycle.

## Notes / caveats

- **Demo broker, not production Kafka.** `redpandadata/redpanda` in
  `--mode=dev-container`: single node, no persistence guarantees beyond the
  pod's lifetime, no auth/TLS. Fine for validating the connector end-to-end;
  not a sizing or HA reference.
- **No native Beam path exercised here.** This demo runs the `local` engine
  (`KafkaSource.read()`'s bounded poll); `native_beam_read()`'s unbounded
  `ReadFromKafka` path (streaming engine, M5/M6) needs a Beam-capable runner
  (see `deploy/flink` or a Dataflow submit), not covered by this manifest set.
- **Wire format.** Producers must publish one JSON-encoded, already-aligned
  `PivotRow` per message (`{group_id, ts, values, channels, labels}`) — see
  `connectors/sources/kafka.py`'s module docstring. Alignment across
  heterogeneous channel cadences is the producer's job, same boundary Mimir's
  connector owns for its own ingestion path.

## Teardown

```sh
kubectl delete -k deploy/kafka
```
