# Architecture — Metrics Variation Detection on PatchTST

## Goal

Aggregate time-series **signals** into a datalake and detect variations
(anomalies / drift) using PatchTST, with unified batch and streaming processing.

The core is **domain-agnostic**: it knows only generic signals — an entity, a
metric, values over time (`PivotRow` / `SignalRecord` carry no domain-specific
field). It therefore serves any time-series domain (infrastructure, IoT,
application or business KPIs, …). **Kubernetes observability — feeding
[kube-verdict](https://github.com/a1h8/kube-verdict) — is the reference
application, not a constraint.**

The design is also **provider-agnostic**: deployment target (managed cloud,
self-hosted, or sovereign) is a configuration choice, not baked into the core.

## Core principle

The processing core (windowing → PatchTST inference → variation detection)
never knows where data comes from or goes to. Sources and sinks are **plugins**
behind a stable contract. Apache Beam is the engine because the same pipeline
runs in batch (replay history) and streaming (live ingestion), and is portable
across runners (DirectRunner for dev, Flink/Spark on-cluster or Dataflow for prod).

## Reference flow

The boxes below are *roles*, each filled by an interchangeable connector. The
names in parentheses are one possible instantiation; any backend that implements
the SPI role can be substituted (see [CONNECTORS.md](./CONNECTORS.md)).

```
Agents (Prometheus / OTEL)
        │  remote-write
        ▼
   metrics store  ◄──── blocks ────►  object store (S3 API)
   (e.g. Mimir)                            ▲
        │  query / pull                    │
        ▼                                  │
   Beam runner  (windowing)                │
   (Direct / Flink / Spark / Dataflow)     │
        │                                  │
        ▼                                  │
   PatchTST inference (forecast ⇄ reconstruction)
        │                                  │
        ▼                                  │
   Variation detection                     │
        │                                  │
        ├──► datalake (table format) ──────┘  (same object-store substrate)
        ├──► analytics store
        └──► alerting / verdict sink
```

A single object store (any S3-compatible backend) can serve as the substrate for
both the metrics store's blocks and the datalake — one storage layer to secure
and back up, whichever provider you choose.

## Detection mechanism

**Decided (D1): dual detector, regime-switching.** Two signals on one PatchTST
pipeline, each used where it is reliable:

- **Forecast — anticipation** (`PatchTST_supervised`). In the normal/trending
  regime the model forecasts the trajectory; a predicted threshold crossing
  within horizon `h` raises an early **WARN**. This is the "see the wall coming"
  path, scoped to slow-saturation metrics (disk, memory, quota, latency drift).
- **Reconstruction — detective** (`PatchTST_self_supervised`). A brutal break
  pushes the input out-of-distribution, where the forecaster collapses (its
  predictions become unreliable exactly when needed). Reconstruction error spikes
  cleanly on OOD input, so at the break the verdict switches to this signal — no
  waiting for `actual[t+h]`.

**Why both, not one.** It is not a compromise but the technically correct split:
forecast is accurate pre-break and buys lead time; reconstruction is the clean
signal during the incident *because* forecast degrades under regime change.

**State machine per `group_id`:**

```
NORMAL  ──(break: reconstruction error spikes)──►  INCIDENT
   ▲                                                   │
   └──────────(recovery: error back to baseline)───────┘

NORMAL   : forecast drives anticipation (early WARN)
INCIDENT : reconstruction drives the verdict; anticipation suspended (model OOD)
```

**Design constraint:** anticipation only pays if it is actionable — the horizon
`h` must be ≤ the remediation time (autoscale / drain / page), otherwise an early
WARN is cosmetic and detective alone is preferable.

**Implementation status.** Detection is a pluggable `Detector` (`detection/`).
Implemented today:
- `ZScoreDetector` — real statistical detection (z-score on the recent tail),
  matching kube-verdict's `zscore` method and severity bands;
- `PatchTSTDetector` — the **forecast face** of D1: trains a small PatchTST
  (HF `transformers`) on the early signal and scores the recent window by
  forecast-error ratio (`method="patchtst"`), falling back to z-score for short
  signals. torch/transformers are optional, lazy-imported.

The **reconstruction face** (detective) and the NORMAL↔INCIDENT regime switch
remain the target, behind the same `Detector` interface. The write path is wired
through the SPI: `MimirSource → make_detection_transform(detector) →
signal-store sink`, runnable on `LocalEngine` (the K3s demo) or `BeamEngine`.

## Connector SPI — open to N plugins

Connectors are not architecture decisions, they are interchangeable
implementations of one contract.

**Decided (D4): native multivariate pivot.** A row carries an aligned vector of
channel values at one timestamp:
`{group_id: str, ts: int, values: tuple[float], channels: tuple[str], labels: dict}`.

Intentional model mismatch, accepted with eyes open: PatchTST is
**channel-independent** — it processes each channel as an independent univariate
sequence with no cross-channel attention. So the grouping buys batching and a
group-level detection decision, **not** joint modeling of cross-channel
correlation. The real cost of multivariate — temporal alignment of
heterogeneous K8s cadences onto a common grid — is paid inside the source
connector (`connectors/alignment.py`), never in the core.

The contract is **engine-agnostic** (decision D6): a source *yields* rows, a sink
*consumes* them — no execution engine appears in the contract.

```python
# connectors/base.py — no engine import
from abc import ABC, abstractmethod
from typing import Iterable

# Pivot schema — the only language the core understands:
# {group_id: str, ts: int, values: tuple[float], channels: tuple[str], labels: dict}

class SourceConnector(ABC):
    @abstractmethod
    def read(self) -> Iterable[PivotRow]:        # pure Python
        ...
    def native_beam_read(self): return None      # optional engine-native override

class SinkConnector(ABC):
    @abstractmethod
    def write(self, rows: Iterable) -> None:      # pure Python
        ...
    def native_beam_write(self): return None      # optional engine-native override
```

```python
# connectors/registry.py
_REGISTRY: dict[str, type] = {}

def connector(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco

def build(name: str, **cfg):
    return _REGISTRY[name](**cfg)
```

Adding a connector = dropping one file with `@connector("name")`. The core and
the pipeline never change. That openness is the point of having N connectors.

## Execution engines — ports & adapters (D6)

The core and connectors are engine-agnostic; an **engine adapter** runs a
`source → sinks` flow on a concrete runtime. The engine depends on connectors,
never the reverse — the hexagonal boundary that lets the system plug into any
engine.

```
core (PivotRow, connectors, detection)  ── engine-agnostic
    │
    ├─ engines/local.py   pure Python — no third-party dep (dev, tests, small jobs)
    ├─ engines/beam.py    Apache Beam (current production path)
    └─ engines/spark.py   Spark / Databricks (planned, "one engine among others")
```

```python
src   = build(cfg.source.type, **cfg.source.params)
sinks = [build(s.type, **s.params) for s in cfg.sinks]

LocalEngine().run(src, sinks)   # or BeamEngine().run(src, sinks)
```

**Native capability hook.** A pure iterator cannot express an engine's native
distributed/unbounded I/O (unbounded Kafka, a parallel writer). A connector may
expose `native_beam_read` / `native_beam_write`; the engine adapter uses them
when present and falls back to gather-and-call otherwise. This keeps connectors
portable without sacrificing engine-native streaming where it matters.

Trade-off (accepted): "plug any engine" is not free — each engine needs its
adapter, and engine-native features are not portable for nothing. But the core,
connectors, and detection logic are written once, engine-free.

## Knowledge base — feeding kube-verdict (D7)

This pipeline is the **signal-aggregation / knowledge-base layer** for
[kube-verdict](https://github.com/a1h8/kube-verdict), an evidence-first K8s
incident decision engine. The two are complementary, not duplicative:

| | kube-verdict | this pipeline |
|---|---|---|
| When | reactive, per incident | continuous |
| Scope | one entity in RCA | cluster-wide, longitudinal |
| Output | point-in-time verdict / RCA | queryable **signal history** |

We do **not** re-implement detection — kube-verdict already has
`signals/patchtst_detector.py`. Our value is aggregating signals over time into a
**queryable knowledge base** kube-verdict draws on as historical evidence.

**Structured face (`kb/`, implemented).** Aggregated `SignalRecord`s (schema
aligned with kube-verdict's `AnomalyResult`) are written as Parquet and queried
by DuckDB. The store exposes the contract kube-verdict's `rca/context_builder`
calls during RCA:

```
GET /api/v1/signals/history?entity=Pod/prod/api&metric=cpu_usage&since=..&until=..
```

This is "interrogate the service with a signal" — decoupled, touching neither
kube-verdict's embedder nor its FAISS index. Backend is Parquet+DuckDB now
(S3/Iceberg-ready), swappable for ClickHouse at scale behind the same interface.

**Semantic face (planned).** A parallel vector store (Weaviate as a shared
service, owning its vectorizer) for kube-verdict's RAG `example_lookup_node`.
Kept separate from the structured face to avoid coupling to its hardcoded
`all-MiniLM-L6-v2` FAISS index.

**Write path goes through the SPI** — the store is exposed as a
`@connector("signal-store")` `SinkConnector`, so signals are written via the
normal connector cycle (`build → Engine.run → sink.write`), not a standalone
write. **Read path** (`SignalStore.query` / the HTTP service) stays *outside*
the SPI on purpose: it is request/response serving, not streaming dataflow.
Optional secondary push: emit Alertmanager-format alerts to
`/api/v1/webhook/alertmanager` to trigger RCA.

## Deployment is independent of design

The architecture is provider-agnostic: *where* it runs is a deployment choice
layered on top, not a property of the core or the SPI. The same pipeline runs
against a managed cloud, a self-hosted OSS stack, or a sovereign/EU provider by
swapping connector config and the runner flag — no code change. See the
deployment profiles in [CONNECTORS.md](./CONNECTORS.md).

Each profile carries its own trade-offs (managed convenience vs operational
debt, jurisdiction/compliance, cost). Sovereignty — e.g. SecNumCloud-qualified
providers with no extra-territorial exposure, plus catalog/lineage to prove
*where data lives and who accesses it* — is one such profile, selectable when
required, never assumed.

See [CONNECTORS.md](./CONNECTORS.md) for the plugin catalog and
[ROADMAP.md](./ROADMAP.md) for milestones and open decisions.