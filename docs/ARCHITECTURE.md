# Architecture — Metrics Variation Detection on PatchTST

## Goal

Consolidate K8s metrics into a sovereign datalake and detect variations
(anomalies / drift) on time-series using PatchTST, with unified batch and
streaming processing.

## Core principle

The processing core (windowing → PatchTST inference → variation detection)
never knows where data comes from or goes to. Sources and sinks are **plugins**
behind a stable contract. Apache Beam is the engine because the same pipeline
runs in batch (replay history) and streaming (live ingestion), and is portable
across runners (DirectRunner for dev, Flink/Spark on-cluster for sovereign prod).

## Target flow (sovereign stack)

```
Agents (Prometheus / OTEL)
        │  remote-write
        ▼
   Grafana Mimir  ◄──── metric blocks ────►  MinIO (S3)
        │  PromQL query_range                      ▲
        ▼                                          │
   Beam / Flink  (windowing)                       │
        │                                          │
        ▼                                          │
   PatchTST inference (forecast ⇄ reconstruction)  │
        │                                          │
        ▼                                          │
   Variation detection                             │
        │                                          │
        ├──► Iceberg datalake ────────────────────┘  (same MinIO substrate)
        ├──► ClickHouse (analytics)
        └──► KubeVerdict (alerting / verdict)
```

MinIO is the single object-storage substrate, backing both Mimir blocks and the
Iceberg datalake — one storage layer to secure, back up, and keep sovereign.

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

```python
# connectors/base.py
from abc import ABC, abstractmethod
import apache_beam as beam

# Pivot schema — the only language the core understands:
# {group_id: str, ts: int, values: tuple[float], channels: tuple[str], labels: dict}

class SourceConnector(ABC):
    @abstractmethod
    def read(self) -> beam.PTransform:   # -> PCollection[PivotRow]
        ...

class SinkConnector(ABC):
    @abstractmethod
    def write(self) -> beam.PTransform:  # PCollection -> writes out
        ...
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

The pipeline is config-driven and source/sink agnostic:

```python
src = build(cfg.source.type, **cfg.source.params)
sinks = [build(s.type, **s.params) for s in cfg.sinks]

p | src.read() | Window() | RunInference(patchtst) | Detect() | *[s.write() for s in sinks]
```

Adding a connector = dropping one file with `@connector("name")`. The core and
the pipeline never change. That openness is the point of having N connectors.

## Sovereignty model

Three tiers, to be explicit about what "sovereign" means here:

- **SecNumCloud-qualified** (OVHcloud select offers, Outscale/Dassault, Cloud
  Temple): French/EU law, immune to extra-territorial reach.
- **"Trusted cloud" but US tech under license** (S3NS = Thales+Google, Bleu =
  Capgemini/Orange+Microsoft): excluded if strict immunity is the goal.
- **Open-source self-hosted** (MinIO, Ceph, ClickHouse, Kafka, Mimir): maximum
  sovereignty since we control the substrate; the operational debt is ours.

Sovereignty is not only runtime: a data catalog (Nessie/Polaris) and lineage
(OpenMetadata/DataHub) matter too, because proving *where data lives and who
accesses it* is part of SecNumCloud / GDPR requirements.

See [CONNECTORS.md](./CONNECTORS.md) for the plugin catalog and
[ROADMAP.md](./ROADMAP.md) for milestones and open decisions.