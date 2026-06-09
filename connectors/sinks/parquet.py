"""Parquet sink (connector C11/C18 stepping stone).

Writes pipeline output as Parquet. Targets a local path for the batch POC and an
S3-compatible store (MinIO) via an ``s3://`` path once ``s3fs``/filesystem is
configured — same code, sovereign substrate. Iceberg table format (C18) layers
on top later without changing the connector contract.

Beam is imported lazily in ``write``.
"""
from __future__ import annotations

from ..base import SinkConnector
from ..registry import connector

# Arrow-friendly flat schema for detection output.
_FIELDS = ["group_id", "ts", "channels", "values", "labels"]


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

        return pa.schema(
            [
                ("group_id", pa.string()),
                ("ts", pa.int64()),
                ("channels", pa.list_(pa.string())),
                ("values", pa.list_(pa.float64())),
                ("labels", pa.map_(pa.string(), pa.string())),
            ]
        )

    def write(self):
        import apache_beam as beam  # lazy
        from apache_beam.io.parquetio import WriteToParquet

        def as_record(row) -> dict:
            return {
                "group_id": row.group_id,
                "ts": row.ts,
                "channels": list(row.channels),
                "values": [float(v) for v in row.values],
                "labels": dict(row.labels),
            }

        return "ToParquet" >> (
            beam.Map(as_record)
            | WriteToParquet(
                file_path_prefix=self.path,
                schema=self._pyarrow_schema(),
                file_name_suffix=self.file_name_suffix,
                num_shards=self.num_shards,
            )
        )

    def describe(self):
        return {**super().describe(), "path": self.path}