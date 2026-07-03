# Roadmap

Milestones are sequential; the **Connector SPI** is the cross-cutting backbone.
See [ARCHITECTURE.md](./ARCHITECTURE.md) and [CONNECTORS.md](./CONNECTORS.md).

## Milestones

| Milestone | Deliverable | Priority |
|-----------|-------------|----------|
| **M0** | Frozen decisions (D1–D5 below) — gate before any code beyond M2 | ✅ D1–D5 decided |
| **M1** | PatchTST inference module decoupled from the training `Learner`, exposing **both heads**: `forecast(window)` and `reconstruct(window)`; RevIN normalization, checkpoint loaded once per worker — *engine + reference checkpoints (ETTh1 pretrain→finetune, `inference/train_reference.py`); load-once inference variant (`patchtst-infer`/`reconstruction-infer`) wired in M3 (#15), train-on-the-fly path kept for scoped/no-checkpoint assessment* | ✅ done |
| **M1.5** | **Connector SPI**: pivot schema + `SourceConnector`/`SinkConnector` contracts + `registry` + contract/conformance test suite — *implemented (PR #2), 100% coverage* | ✅ done |
| **M2** | Beam batch skeleton on DirectRunner: source → windowing → sink, no model. Validates pivot schema end-to-end (dev/test only, never prod) — *`BeamEngine` on DirectRunner via engine-agnostic ports & adapters (#4); real Mimir → detection → signal-store write path + DirectRunner integration test (#6). Batch windowing lives in the pivot/detector layer; native `WindowInto` / watermarks are M5* | ✅ done |
| **M3** | PatchTST in the pipeline via `RunInference` with a custom PyTorch `ModelHandler`: per-worker load, batching, device. Output enriched with **forecast residual + reconstruction error** — *inference-backed D1 detectors wired to the load-once M1 engine, per-window forecast residual + reconstruction error (#15)* | ✅ done |
| **M4** | **Regime-switching detection** per `group_id` (NORMAL→INCIDENT state machine): forecast anticipation (early WARN, `h ≤ remediation time`) in NORMAL, reconstruction detective verdict in INCIDENT; adaptive thresholds (rolling quantile / MAD), per-channel residual aggregation, anti-flapping — *state machine (#9), anti-flapping (#16), adaptive thresholds (#17), entity aggregation (#18), KB-seeded regime state (#19)* | ✅ done |
| **M5** | Streaming: same pipeline unbounded — sliding windows, watermarks, late data, triggering — *Beam streaming path on the DirectRunner: event-time sliding windows, watermark + allowed-lateness gate, per-`(entity, window)` detection, configurable early/late firing policy, validated with a synthetic unbounded source (#24); a real broker (Kafka/OTLP) and a production runner are M6* | ✅ skeleton (DirectRunner) |
| **M6** | Production runner (Flink on-K8s or Dataflow) + pipeline monitoring (lag, throughput, failures) — *config-driven runner selection (`direct`/`dataflow`/`flink` alias → `PipelineOptions`, wired through `build_engine`) + throughput counters on the `PipelineResult`; **Dataflow submit path shipped** (SDK worker image, streaming config example, submit docs) and **Flink-on-K8s** self-hosted portable runner (`deploy/flink`: JobManager + TaskManagers with SDK-harness sidecar + Beam job server, RocksDB→S3 checkpoints; `portable` runner alias). **order: Dataflow first (managed), then Flink-on-K8s**. Lag/failure dashboards next* | 🚧 in progress |
| **M7** | KubeVerdict alerting + optional retraining loop back to the datalake | P2 |

## Critical path (batch POC) — ✅ complete

```
M0 → M1 → M1.5 → M2 → M3 → M4   ✅ all merged
```

End-to-end variation detection on historical data, without touching streaming or
Flink. This de-risked the two fragile joints — inference-in-Beam and the model's
real value — before investing in streaming ops. **Next frontier: M5 (streaming).**

## Connector workstream

Once the SPI (M1.5) is frozen, each connector is a small, parallelizable PR:
drop a file under `connectors/sources/` or `connectors/sinks/` with
`@connector("name")` and pass the conformance suite. P0 connectors for the POC:
a metrics source (**Mimir, C9a**) and an **object-store + table-format** sink
(**S3 API + Iceberg, C11/C18** — backend-agnostic). Other backends are added on
demand, not upfront.

## Open decisions

| #  | Question | Decision |
|----|----------|----------|
| D1 | Detection mechanism | ✅ **Both, regime-switching**: forecast (anticipate the wall) in NORMAL, reconstruction (detective) at the break |
| D2 | Mimir as sole ingress, or Kafka/OTLP in parallel for low-latency live? | ✅ **Mimir-first**: Mimir (C9a) is the sole ingress for M6 — one connector, no broker, the same source as the historical KB; accepted latency is scrape + remote-write + query (~tens of s). Kafka/OTLP (C2/C7) added on-demand when sub-second forecast anticipation requires it. |
| D3 | Plugin discovery: internal registry vs Python entry-points | ✅ **Internal registry** |
| D4 | Pivot schema: univariate vs native multivariate | ✅ **Native multivariate** |
| D5 | Datalake purpose: retraining vs analytics/compliance vs both | ✅ **Knowledge base** — longitudinal signal history kube-verdict queries as RCA evidence |
| D6 | Engine coupling: Beam in the contract vs engine-agnostic core | ✅ **Engine-agnostic** (ports & adapters): Local + Beam engines, Spark/Databricks next |
| D7 | How kube-verdict consumes the knowledge base | ✅ **Structured datalake first** (`kb/`: SignalRecord + DuckDB query + `signal_history` API); semantic Weaviate face next; don't re-implement its detector |

## Scope note — SPI vs catalog

The real P0 deliverable is **not** "write the Mimir and MinIO connectors". It is
freezing the **pivot contract + registry + conformance suite** (M1.5). After
that, Mimir / Kafka / MinIO / ClickHouse are a few files each, written in
parallel without touching the core.