"""CSV → Mimir/Prometheus ingestion (remote-write).

A small tool to push wide CSVs (e.g. the ETTh1 reference data) into a real
Grafana Mimir or Prometheus via the remote-write API, so the rest of the
pipeline (``MimirSource`` → detection → knowledge base) can run end-to-end on
real-ish data. See ``python -m connectors.ingest --help``.
"""
from __future__ import annotations

from .csv_source import csv_to_timeseries
from .remote_write import RemoteWriter, TimeSeries, encode_write_request

__all__ = ["csv_to_timeseries", "RemoteWriter", "TimeSeries", "encode_write_request"]
