"""Beam engine — adapter running connectors on Apache Beam.

Uses a connector's engine-native hook when present (e.g. an unbounded Kafka
source, or a distributed Parquet writer), and otherwise lifts the agnostic
iterator: ``beam.Create(source.read())`` for reads, gather-and-call ``write()``
for sinks. ``apache_beam`` is imported lazily so the core never needs it.
"""
from __future__ import annotations

from typing import Optional, Sequence

from ..base import SinkConnector, SourceConnector
from .base import Engine, Transform


class BeamEngine(Engine):
    def __init__(self, pipeline_options=None) -> None:
        self._options = pipeline_options

    def run(
        self,
        source: SourceConnector,
        sinks: Sequence[SinkConnector],
        transform: Optional[Transform] = None,
    ) -> None:
        import apache_beam as beam

        with beam.Pipeline(options=self._options) as p:
            native_read = source.native_beam_read()
            pcoll = p | "Read" >> (
                native_read
                if native_read is not None
                else beam.Create(list(source.read()))
            )
            if transform is not None:
                # Iterable->Iterable transform (e.g. detection): gather the
                # bundle, apply, fan back out. Batch-oriented; engine-native
                # windowed detection is a later optimization.
                pcoll = (
                    pcoll
                    | "ToListT" >> beam.combiners.ToList()
                    | "Transform" >> beam.FlatMap(lambda rows: list(transform(rows)))
                )
            for i, sink in enumerate(sinks):
                native_write = sink.native_beam_write()
                if native_write is not None:
                    pcoll | f"Write{i}" >> native_write
                else:
                    # Fallback: gather to one bundle and call the agnostic
                    # write(). Fine for batch/small; native writers should be
                    # provided for large or unbounded sinks.
                    (
                        pcoll
                        | f"ToList{i}" >> beam.combiners.ToList()
                        | f"Write{i}" >> beam.Map(sink.write)
                    )
