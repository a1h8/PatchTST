"""Parquet sink (connector C11/C18 stepping stone).

Writes pipeline output as Parquet. Targets a local path for the batch POC and an
S3-compatible store (MinIO) via an ``s3://`` path once ``s3fs``/filesystem is
configured — same code, sovereign substrate. Iceberg table format (C18) layers
on top later without changing the connector contract.

Beam is imported lazily in ``write``.
"""
from __future__ import annotations

import json

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

    def write(self):
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