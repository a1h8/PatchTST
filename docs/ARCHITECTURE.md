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
   PatchTST inference (reconstruction error)       │
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

Default: **self-supervised reconstruction error** (`PatchTST_self_supervised`).
The model reconstructs masked patches; the gap between reconstruction and actual
value is the variation signal. This needs no labels and gives a clean reactive
signal. The forecast-plus-residual path remains an alternative (decision D1).

## Connector SPI — open to N plugins

Connectors are not architecture decisions, they are interchangeable
implementations of one contract.

```python
# connectors/base.py
from abc import ABC, abstractmethod
import apache_beam as beam

# Pivot schema — the only language the core understands:
# {series_id: str, ts: int, value: float, labels: dict}

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