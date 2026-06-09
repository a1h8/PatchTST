"""Parquet sink (connector C11/C18 stepping stone).

Writes records as Parquet. Targets a local path and, via the S3 API, any
S3-compatible store; Iceberg (C18) layers on later without changing the contract.

Engine-agnostic ``write`` uses pyarrow directly (no engine needed). Under Beam,
the native hook ``native_beam_write`` provides a distributed ``WriteToParquet``
instead. pyarrow / beam are imported lazily so the core stays dependency-free.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

from ..base import SinkConnector
from ..registry import connector


@connector("parquet")
class ParquetSink(SinkConnector):
    """Write rows (e.g. PivotRow-derived detections) to Parquet files."""

    def __init__(
        self,
        path: str,
        *,
        file_name_suffix: str = ".parquet",
        num_shards: int = 0,
    ) -> None:
        self.path = path
        self.file_name_suffix = file_name_suffix
        self.num_shards = num_shards

    def _pyarrow_schema(self):
        import pyarrow as pa  # lazy

        # labels stored as a JSON string for a robust, schema-stable round-trip.
        return pa.schema(
            [
                ("group_id", pa.string()),
                ("ts", pa.int64()),
                ("channels", pa.list_(pa.string())),
                ("values", pa.list_(pa.float64())),
                ("labels", pa.string()),
            ]
        )

    @staticmethod
    def as_record(row) -> dict:
        """PivotRow (or detection row) -> flat Arrow-friendly dict."""
        return {
            "group_id": row.group_id,
            "ts": int(row.ts),
            "channels": list(row.channels),
            "values": [float(v) for v in row.values],
            "labels": json.dumps(dict(row.labels), sort_keys=True),
        }

    def local_path(self) -> str:
        """Single-file output path used by the agnostic writer."""
        return f"{self.path}{self.file_name_suffix}"

    def write(self, rows: Iterable[Any]) -> None:
        """Engine-agnostic write: materialize records and write one Parquet file."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        records = [self.as_record(r) for r in rows]
        table = pa.Table.from_pylist(records, schema=self._pyarrow_schema())
        pq.write_table(table, self.local_path())

    def native_beam_write(self):
        import apache_beam as beam  # lazy
        from apache_beam.io.parquetio import WriteToParquet

        schema = self._pyarrow_schema()
        path, suffix, shards = self.path, self.file_name_suffix, self.num_shards
        as_record = self.as_record

        class _ToParquet(beam.PTransform):
            def expand(self, pcoll):
                return (
                    pcoll
                    | "AsRecord" >> beam.Map(as_record)
                    | "WriteParquet"
                    >> WriteToParquet(
                        file_path_prefix=path,
                        schema=schema,
                        file_name_suffix=suffix,
                        num_shards=shards,
                    )
                )

        return _ToParquet()

    def describe(self):
        return {**super().describe(), "path": self.path}