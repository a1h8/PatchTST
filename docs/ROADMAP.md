# Roadmap

Milestones are sequential; the **Connector SPI** is the cross-cutting backbone.
See [ARCHITECTURE.md](./ARCHITECTURE.md) and [CONNECTORS.md](./CONNECTORS.md).

## Milestones

| Milestone | Deliverable | Priority |
|-----------|-------------|----------|
| **M0** | Frozen decisions (D1–D5 below) — gate before any code beyond M2 | ✅ D1/D3/D4 decided |
| **M1** | PatchTST inference module decoupled from the training `Learner`, exposing **both heads**: `forecast(window)` and `reconstruct(window)`; RevIN normalization, checkpoint loaded once per worker | P0 |
| **M1.5** | **Connector SPI**: pivot schema + `SourceConnector`/`SinkConnector` contracts + `registry` + contract/conformance test suite — *implemented (PR #2), 100% coverage* | ✅ done |
| **M2** | Beam batch skeleton on DirectRunner: source → windowing → sink, no model. Validates pivot schema end-to-end (dev/test only, never prod) | P0 |
| **M3** | PatchTST in the pipeline via `RunInference` with a custom PyTorch `ModelHandler`: per-worker load, batching, device. Output enriched with **forecast residual + reconstruction error** | P0 |
| **M4** | **Regime-switching detection** per `group_id` (NORMAL→INCIDENT state machine): forecast anticipation (early WARN, `h ≤ remediation time`) in NORMAL, reconstruction detective verdict in INCIDENT; adaptive thresholds (rolling quantile / MAD), per-channel residual aggregation, anti-flapping | P0 |
| **M5** | Streaming: same pipeline unbounded — sliding windows, watermarks, late data, triggering | P1 |
| **M6** | Production runner (Flink on-K8s or Dataflow) + pipeline monitoring (lag, throughput, failures) | P1 |
| **M7** | KubeVerdict alerting + optional retraining loop back to the datalake | P2 |

## Critical path (batch POC)

```
M0 → M1 → M1.5 → M2 → M3 → M4
```

End-to-end variation detection on historical data, without touching streaming or
Flink. This de-risks the two fragile joints — inference-in-Beam and the model's
real value — before investing in streaming ops.

## Connector workstream

Once the SPI (M1.5) is frozen, each connector is a small, parallelizable PR:
drop a file under `connectors/sources/` or `connectors/sinks/` with
`@connector("name")` and pass the conformance suite. P0 connectors for the POC:
**Mimir (C9a)** source and **MinIO/Iceberg (C11/C18)** sink.

## Open decisions

| #  | Question | Decision |
|----|----------|----------|
| D1 | Detection mechanism | ✅ **Both, regime-switching**: forecast (anticipate the wall) in NORMAL, reconstruction (detective) at the break |
| D2 | Mimir as sole ingress, or Kafka/OTLP in parallel for low-latency live? | TBD |
| D3 | Plugin discovery: internal registry vs Python entry-points | ✅ **Internal registry** |
| D4 | Pivot schema: univariate vs native multivariate | ✅ **Native multivariate** |
| D5 | Datalake purpose: retraining vs analytics/compliance vs both | TBD |

## Scope note — SPI vs catalog

The real P0 deliverable is **not** "write the Mimir and MinIO connectors". It is
freezing the **pivot contract + registry + conformance suite** (M1.5). After
that, Mimir / Kafka / MinIO / ClickHouse are a few files each, written in
parallel without touching the core.