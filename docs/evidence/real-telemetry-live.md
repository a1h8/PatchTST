# Real cluster telemetry, end to end — evidence

Captured artifact, not a claim (same discipline as `docs/evidence/flink-live.md`
and `tools/capture_signals.py`). Answers the next open item after the batch
POC and the Flink-on-K8s proof: does the pipeline detect on **real cluster
telemetry**, not `sim_*` fixtures or synthetic scenarios — all the way from a
Prometheus scrape to a `SignalRecord` in the KB?

- **Date:** 2026-09-28
- **Cluster:** Rancher Desktop k3s (`lima-rancher-desktop`)
- **Stack:** `kubeverdict-obs` Prometheus + Alloy/Grafana/Loki/Tempo, `patchtst`
  Mimir + the `pipeline` CronJob, both pre-existing

## Starting state

Mimir (`patchtst` namespace, 10 days up) had never received a single real
sample: `query up` returned an empty vector. The `kubeverdict-obs` Prometheus
had no `remote_write` configured, so nothing fed it. Separately, that
Prometheus pod was crash-looping (39 restarts over ~3 days,
`bind: address already in use` on :9090 — a stuck listener surviving
in-pod-netns container restarts; a full pod delete, not just a container
restart, was needed to clear it).

## What was changed

1. Deleted the wedged `prometheus-server` pod (fresh network namespace ->
   the stale listener cleared, container came back healthy).
2. `helm upgrade prometheus prometheus-community/prometheus -n kubeverdict-obs
   --reuse-values -f -` with `server.remoteWrite[0].url:
   http://mimir-gateway.patchtst.svc.cluster.local:80/api/v1/push` (Mimir here
   runs with `auth_enabled: false`, confirmed by `connectors/sources/mimir.py`'s
   optional `tenant` and the absence of any `X-Scope-OrgID` in the existing
   `deploy/k3s` manifests — no tenant header needed).
3. Same mechanism, `kube-state-metrics.enabled: true` (was `false`) — gives a
   real, meaningful per-pod signal (`kube_pod_container_status_restarts_total`)
   instead of the otel-collector's internal-only metrics that were the only
   other thing already being scraped.
4. `deploy/k3s/30-pipeline.yaml`'s `promql` repointed from `{__name__=~"sim_.+"}`
   to `increase(kube_pod_container_status_restarts_total[10m])` — the live
   counterpart of `scenarios.library.noisy_baseline_no_incident`'s
   `pod_restart_count` shape, now on the actual cluster instead of a fixture.

## Result

`kubectl create job --from=cronjob/pipeline` (one manual tick, same image and
config the CronJob runs every 5 minutes):

```
mimir-alertmanager-0/patchtst                        normal  score=0.0
flink-taskmanager-678d9f5ddc-7hbhf/patchtst          normal  score=0.0
coredns-9fc4f469d-7pmtg/kube-system                  normal  score=0.0
kube-verdict-kubeverdict-5488fc669-cv72s/kubeverdict normal  score=0.0
... (20 real pods across 4 namespaces)
```

Read back from the KB's Parquet datalake (`duckdb.read_parquet` over
`/data/kb`, mounted from the same PVC the pipeline writes to) — not a log
line, the actual persisted `SignalRecord`s, keyed by real `entity_uid`s the
cluster's scheduler assigned, not `node1/demo` fixture labels.

## What this does and doesn't prove

**Proves:** the full chain — kube-state-metrics scrape -> Prometheus
`remote_write` -> Mimir -> the `mimir` source connector -> `ZScoreDetector` ->
KB write — runs on real telemetry with no code changes, only deploy config
(Helm values + one ConfigMap's `promql`).

**Doesn't prove:** detection sensitivity on a real incident. Every pod scored
`normal`/`0.0` because no container actually restarted in the lookback
window — the correct answer for a quiet cluster, but this run doesn't
exercise the `warning`/`critical` path on real data. That needs either a
longer observation window or a deliberately induced restart, and is the
natural next step before claiming production-grade evidence (see README
Status).

## Reruns

```
kubectl -n patchtst create job --from=cronjob/pipeline pipeline-manual-test
kubectl -n patchtst wait --for=condition=complete job/pipeline-manual-test
kubectl -n patchtst exec deploy/kb -- python3 -c "
import duckdb, glob
files = glob.glob('/data/kb/**/*.parquet', recursive=True)
print(duckdb.connect().execute(f'select entity_uid, metric_name, severity, score, ts from read_parquet({files!r}) order by ts desc limit 20').fetchall())
"
kubectl -n patchtst delete job pipeline-manual-test
```
