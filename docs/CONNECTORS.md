# Connectors — Plugin Catalog

Connectors implement the SPI defined in [ARCHITECTURE.md](./ARCHITECTURE.md):
`SourceConnector.read()` and `SinkConnector.write()`, both speaking the pivot
schema `{group_id, ts, values[], channels[], labels}`. Each connector is a plugin
registered with `@connector("name")`; the core never depends on a concrete
connector.

**The SPI is provider-agnostic.** A connector is just an implementation of the
contract — it can target a managed cloud service, a self-hosted OSS system, or a
sovereign provider. The catalog below is grouped by *function*; within each
function the backends are interchangeable, and none is privileged. Pick whichever
fits your environment; the pipeline code does not change.

## Sources — metrics ingestion

| ID  | Plugin | Mechanism | Note |
|-----|--------|-----------|------|
| C9a | **Grafana Mimir** | remote-write in + PromQL read | multi-tenant, single entry point, blocks on object storage |
| C1  | Prometheus | PromQL `query_range` | direct scrape target or long-term store front-end |
| C2  | OTLP / remote-write | live push | vendor-neutral OpenTelemetry |
| C7  | Kafka / Redpanda | streaming bus | Redpanda lighter, no ZooKeeper |
| C8  | NATS JetStream | streaming bus | low footprint, edge-friendly |
| C10 | VictoriaMetrics | TSDB read/write | compact TSDB alternative |
| C23 | Cloud Pub/Sub / Kinesis | managed streaming bus | managed-cloud option |

## Sinks — object storage / datalake

All speak the **S3 API**, so a single connector covers every S3-compatible
backend; the target is a config/credentials concern, not a code change.

| ID  | Plugin | Backend examples |
|-----|--------|------------------|
| C11 | **S3-compatible object store** | AWS S3 · GCS (interop) · Azure Blob · MinIO · Ceph RGW · OVHcloud / Scaleway / Outscale |

## Sinks — analytics / query

| ID  | Plugin | Note |
|-----|--------|------|
| C15 | **ClickHouse** | strong on time-series OLAP |
| C16 | TimescaleDB / PostgreSQL | for moderate volume |
| C17 | Trino / DuckDB | query-on-lake over Parquet/Iceberg |
| C24 | BigQuery / Snowflake / Redshift | managed-cloud warehouses |

## Table format & catalog

| ID  | Plugin | Role |
|-----|--------|------|
| C18 | **Apache Iceberg** / Delta | ACID table format, time-travel, open datalake |
| C19 | Nessie / Polaris | versioned "git-for-data" catalog |
| C22 | OpenMetadata / DataHub | lineage, governance, traceability |

## Alerting

| ID  | Plugin | Role |
|-----|--------|------|
| C6  | **KubeVerdict** | emit verdict + context (in-house) |
| C25 | Alertmanager / PagerDuty / Opsgenie | standard alert routing |

## Runners — execution axis (not connectors)

Runners are the *other* axis: the Beam pipeline is portable, so the runner is a
`--runner` flag, not connector code. Listed here only for completeness.

| Runner | Use |
|--------|-----|
| DirectRunner | local dev / test |
| Flink / Spark on-K8s | self-hosted production |
| Dataflow | managed-cloud production (GCP) |

## Deployment profiles (examples — none privileged)

The same pipeline runs against different stacks by swapping connector config and
the runner flag:

```
Managed cloud (GCP):  Pub/Sub → Dataflow → GCS + BigQuery
Self-hosted (OSS):    Mimir   → Flink    → MinIO + Iceberg → ClickHouse
Sovereign (EU):       Mimir   → Flink    → OVH/Outscale S3 → ClickHouse
```

Sovereignty (SecNumCloud-qualified providers, no US-hyperscaler dependency) is
**one** such profile — selectable, not assumed. Because the object-storage sink
speaks the S3 API and the runner is a flag, moving between profiles needs no code
change. The trade-offs that distinguish profiles (managed vs operational debt,
jurisdiction, cost) are deployment decisions, independent of the SPI.
