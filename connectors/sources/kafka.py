"""Kafka / Redpanda source (connector C7).

Reads pivot rows from a topic where each message is one JSON-encoded, already
-aligned ``PivotRow``: ``{group_id, ts, values, channels, labels}`` — the wire
mirror of ``ParquetSink.as_record`` (see ``connectors/sinks/parquet.py``).
Kafka does not align heterogeneous channel cadences the way ``alignment.py``
does for Mimir's batch query (decision D4); a producer publishing raw
per-channel samples is responsible for aligning them before they land on the
topic this connector reads.

Two views of the same topic, same shape as the other M5/M6 streaming sources:

- ``read()`` — engine-agnostic: a bounded poll (``kafka-python``) that drains
  whatever is currently available within ``poll_timeout_s``. Fine for the
  ``LocalEngine``, tests, and small backlogs; not a substitute for the native
  unbounded read.
- ``native_beam_read()`` — Beam's cross-language ``ReadFromKafka`` (unbounded,
  per-partition watermarks), used by the streaming engine path (roadmap M6);
  it requires the Kafka expansion service (Docker/JDK) that ships with Beam.

Both are lazy imports — ``kafka-python`` and ``apache_beam`` are optional
dependencies (see ``requirements-connectors.txt``), so importing this module
never requires either.
"""
from __future__ import annotations

import json
from typing import Any

from ..base import SourceConnector
from ..pivot import PivotRow
from ..registry import connector


def _row_from_json(payload: bytes | str) -> PivotRow:
    d = json.loads(payload)
    return PivotRow(
        group_id=d["group_id"],
        ts=int(d["ts"]),
        values=tuple(float(v) for v in d["values"]),
        channels=tuple(d["channels"]),
        labels=dict(d.get("labels", {})),
    )


@connector("kafka")
class KafkaSource(SourceConnector):
    """Streaming source backed by a Kafka/Redpanda topic of pivot-row JSON."""

    def __init__(
        self,
        bootstrap_servers: str | list[str],
        topic: str,
        *,
        group_id: str = "patchtst-connector",
        auto_offset_reset: str = "latest",
        poll_timeout_s: float = 5.0,
    ) -> None:
        self.bootstrap_servers = (
            list(bootstrap_servers)
            if isinstance(bootstrap_servers, list)
            else [s.strip() for s in bootstrap_servers.split(",")]
        )
        self.topic = topic
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.poll_timeout_s = poll_timeout_s

    def read(self) -> list[PivotRow]:
        from kafka import KafkaConsumer  # lazy: non-stdlib, only needed live

        consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset=self.auto_offset_reset,
            consumer_timeout_ms=int(self.poll_timeout_s * 1000),
            enable_auto_commit=True,
        )
        try:
            return [_row_from_json(msg.value) for msg in consumer]
        finally:
            consumer.close()

    def native_beam_read(self):
        import apache_beam as beam  # lazy
        from apache_beam.io.kafka import ReadFromKafka

        consumer_config = {
            "bootstrap.servers": ",".join(self.bootstrap_servers),
            "group.id": self.group_id,
            "auto.offset.reset": self.auto_offset_reset,
        }
        topic = self.topic

        class _ReadPivotRows(beam.PTransform):
            def expand(self, pbegin):
                return (
                    pbegin
                    | "ReadFromKafka"
                    >> ReadFromKafka(consumer_config=consumer_config, topics=[topic])
                    | "ParsePivotRow" >> beam.Map(lambda kv: _row_from_json(kv[1]))
                )

        return _ReadPivotRows()

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "bootstrap_servers": self.bootstrap_servers,
            "topic": self.topic,
            "group_id": self.group_id,
        }
