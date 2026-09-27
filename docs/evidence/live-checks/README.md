# Live checks

Repeatable, automated versions of the manual live proofs captured elsewhere
in `docs/evidence/` — each run against a real cluster (never mocked) and
writes a timestamped JSON result here, same "captured artifact, not a claim"
discipline as `docs/evidence/signal-captures/`.

| Script | Proves | Needs |
|---|---|---|
| `python -m tools.live_check_flink` | Flink-on-K8s: submit → windowed detection → KB write (`s3://`) | `deploy/k3s` + `deploy/flink` applied |
| `python -m tools.live_check_kafka` | Kafka/Redpanda source (D2): drain → detect → KB write, via the real `kb` HTTP API | `deploy/k3s` + `deploy/kafka` applied |
| `python -m tools.live_check_alert` | `KubeVerdictAlertSink`: a real POST lands with labels intact, and a genuinely unreachable endpoint doesn't crash the pusher | `deploy/k3s` + `deploy/mock-kubeverdict` applied |

None of these run in the default `pytest` suite — they need a live cluster,
which CI does not have. Run them by hand before trusting a deployment target
that unit tests (mocked by design) cannot prove.

Still not covered here: Dataflow and `gs://` (no GCP project provisioned —
see the sovereignty gate in `docs/ARCHITECTURE.md`), and OTLP (no collector
deployed yet).
