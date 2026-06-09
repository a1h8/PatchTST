# Connectors — Plugin Catalog

Connectors implement the SPI defined in [ARCHITECTURE.md](./ARCHITECTURE.md):
`SourceConnector.read()` and `SinkConnector.write()`, both speaking the pivot
schema `{series_id, ts, value, labels}`. Each connector is a plugin registered
with `@connector("name")`; the core never depends on a concrete connector.

The catalog below is oriented toward a sovereign cloud (no US hyperscaler), but
any connector that satisfies the contract and passes the conformance suite can
be added.

## Sources

| ID  | Plugin | Mechanism | Replaces (GCP) | Sovereignty / Note |
|-----|--------|-----------|----------------|--------------------|
| C9a | **Grafana Mimir** | remote-write in + PromQL read | — | OSS (AGPLv3) · blocks on MinIO · multi-tenant, single entry point |
| C1  | Prometheus PromQL | `query_range` | — | OSS · typically points at Mimir |
| C2  | OTLP / remote-write | live push | — | OSS, vendor-neutral |
| C7  | Kafka / Redpanda | streaming bus | PubSub | OSS self-hosted · Redpanda lighter, no ZooKeeper |
| C8  | NATS JetStream | streaming bus | PubSub | OSS · low footprint, edge-friendly |
| C10 | VictoriaMetrics | TSDB read/write | — | OSS · compact Thanos/Mimir alternative |

## Sinks — object storage / datalake

| ID  | Plugin | Replaces (GCP) | Sovereignty / Note |
|-----|--------|----------------|--------------------|
| C11 | **MinIO** | GCS | OSS · S3 API · single storage substrate |
| C12 | Ceph RGW | GCS | OSS · on-prem at scale |
| C13 | OVHcloud / Scaleway Object Storage | GCS | SecNumCloud (select offers) · EU |
| C14 | Outscale OOS | GCS | SecNumCloud · Dassault Systèmes |

## Sinks — analytics / query

| ID  | Plugin | Replaces (GCP) | Sovereignty / Note |
|-----|--------|----------------|--------------------|
| C15 | **ClickHouse** | BigQuery | OSS · strong on time-series OLAP |
| C16 | TimescaleDB / PostgreSQL | BigQuery | OSS · for moderate volume |
| C17 | Trino / DuckDB | BigQuery | OSS · query-on-lake over Parquet/Iceberg |

## Table format, catalog, governance

| ID  | Plugin | Role | Sovereignty / Note |
|-----|--------|------|--------------------|
| C18 | **Apache Iceberg** (on MinIO) | ACID table format, time-travel | OSS · open datalake |
| C19 | Nessie / Polaris | versioned "git-for-data" catalog | OSS · table governance |
| C22 | OpenMetadata / DataHub | lineage, governance | OSS · traceability for SecNumCloud/GDPR |

## Alerting

| ID  | Plugin | Role | Note |
|-----|--------|------|------|
| C6  | **KubeVerdict** | emit verdict + context | in-house |

## Runners (Beam execution, not connectors but sovereignty-relevant)

| ID  | Runner | Replaces (GCP) | Note |
|-----|--------|----------------|------|
| C20 | Flink on-K8s | Dataflow | OSS · streaming, the J6 operational debt |
| C21 | Spark on-K8s | Dataflow | OSS · batch, if Spark already in place |

## Recommended sovereign stack

```
Agents → Mimir (C9a) → Beam/Flink (C20) → MinIO+Iceberg (C11/C18)
       → ClickHouse (C15) → KubeVerdict (C6)
```

All OSS, self-hostable on the K8s cluster, zero US-hyperscaler dependency, and
migratable to a SecNumCloud provider (C13/C14) without touching pipeline code
thanks to the shared S3 API. MinIO is the single substrate under both Mimir and
the Iceberg datalake.