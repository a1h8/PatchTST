"""Local engine — pure-Python execution, zero third-party dependency.

Materializes the source and hands the records to each sink. Ideal for dev,
tests, and small/batch jobs. Not distributed: a whole read is pulled into one
process, so it is unsuitable for large or unbounded workloads — use ``BeamEngine``
(or a Spark/Databricks engine) there.
"""
from __future__ import annotations

from typing import Optional, Sequence

from ..base import SinkConnector, SourceConnector
from .base import Engine, Transform


class LocalEngine(Engine):
    def run(
        self,
        source: SourceConnector,
        sinks: Sequence[SinkConnector],
        transform: Optional[Transform] = None,
    ) -> None:
        rows: object = source.read()
        if transform is not None:
            rows = transform(rows)
        rows = list(rows)  # materialize once for all sinks
        for sink in sinks:
            sink.write(rows)
