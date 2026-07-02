"""Beam engine — adapter running connectors on Apache Beam.

Uses a connector's engine-native hook when present (e.g. an unbounded Kafka
source, or a distributed Parquet writer), and otherwise lifts the agnostic
iterator: ``beam.Create(source.read())`` for reads, gather-and-call ``write()``
for sinks. ``apache_beam`` is imported lazily so the core never needs it.

Two execution modes (decision: same pipeline, batch and streaming — roadmap M5):

- **batch** (default): one bounded read, the optional transform is applied over
  the whole gathered bundle, sinks gather-and-call. Validates the pivot schema
  end-to-end (M2).
- **streaming** (``streaming=True``): the read is treated as unbounded; rows are
  stamped with their event time (``PivotRow.ts``), assigned to **sliding
  windows** with a watermark trigger + **allowed lateness**, then the transform
  runs **per (entity, window)** — no global gather. This is the same detection
  code, executed continuously. Validated on the DirectRunner with a synthetic
  unbounded source (``synthetic-stream``); a real broker (Kafka/OTLP) and a
  production runner (Flink/Dataflow) are M6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..base import SinkConnector, SourceConnector
from .base import Engine, Transform


# -- monitoring (M6) -------------------------------------------------------
# Throughput counters read off the PipelineResult after the run — the runner-
# agnostic slice of "pipeline monitoring (lag, throughput, failures)". Both are
# module-level so they pickle for distributed runners; the Metrics import is
# deferred to worker execution.
_METRICS_NS = "pipeline"


def _count_in(row):
    from apache_beam.metrics import Metrics

    Metrics.counter(_METRICS_NS, "rows_in").inc()
    return row


def _count_out(record):
    from apache_beam.metrics import Metrics

    Metrics.counter(_METRICS_NS, "records_out").inc()
    return record


@dataclass(frozen=True)
class WindowSpec:
    """Sliding-window + triggering + late-data policy for the streaming path.

    ``size_s`` is the window length and ``period_s`` the slide step (overlapping
    when ``period_s < size_s``). ``allowed_lateness_s`` is how far past the
    watermark a late element is still admitted into its window (and re-fires its
    pane); elements later than that are dropped. ``accumulation`` controls
    whether a re-firing re-emits the full window (``"accumulating"``) or only
    the new elements (``"discarding"``).

    **Triggering** decides *when* a window's pane is emitted, relative to the
    watermark (which marks the window as on-time-complete):

    - The on-time pane always fires once the watermark passes the window end.
    - ``early_firing_count``: if set, also emit a *speculative* pane after every
      N new elements **before** the watermark closes the window — low-latency
      partial verdicts that are later corrected by the on-time/late panes. Left
      ``None`` for watermark-only (the default, lowest-volume behavior).
    - ``late_firing_count``: re-fire after every N late elements admitted within
      ``allowed_lateness_s`` (default 1: re-fire on each late arrival).
    """

    size_s: float = 60.0
    period_s: float = 30.0
    allowed_lateness_s: float = 0.0
    accumulation: str = "accumulating"
    early_firing_count: Optional[int] = None
    late_firing_count: int = 1

    def __post_init__(self) -> None:
        if self.accumulation not in ("accumulating", "discarding"):
            raise ValueError(
                f"accumulation must be 'accumulating' or 'discarding', "
                f"got {self.accumulation!r}"
            )
        if self.late_firing_count < 1:
            raise ValueError("late_firing_count must be >= 1")
        if self.early_firing_count is not None and self.early_firing_count < 1:
            raise ValueError("early_firing_count must be >= 1 when set")


class BeamEngine(Engine):
    def __init__(
        self,
        pipeline_options=None,
        *,
        streaming: bool = False,
        window: Optional[WindowSpec] = None,
    ) -> None:
        self._options = pipeline_options
        self._streaming = streaming
        self._window = window or WindowSpec()

    def run(
        self,
        source: SourceConnector,
        sinks: Sequence[SinkConnector],
        transform: Optional[Transform] = None,
    ):
        """Run the flow on the configured runner; returns the ``PipelineResult``.

        The result carries the throughput counters (``rows_in`` /
        ``records_out`` under the ``pipeline`` namespace) and, on a real runner,
        the job handle for lag/failure monitoring.
        """
        import apache_beam as beam

        options = self._options
        if self._streaming:
            from apache_beam.options.pipeline_options import (
                PipelineOptions,
                StandardOptions,
            )

            options = options or PipelineOptions()
            options.view_as(StandardOptions).streaming = True

        pipeline = beam.Pipeline(options=options)
        native_read = source.native_beam_read()
        pcoll = pipeline | "Read" >> (
            native_read
            if native_read is not None
            else beam.Create(list(source.read()))
        )
        pcoll = pcoll | "CountIn" >> beam.Map(_count_in)
        if self._streaming:
            self._run_streaming(beam, pcoll, sinks, transform)
        else:
            self._run_batch(beam, pcoll, sinks, transform)
        result = pipeline.run()
        result.wait_until_finish()
        return result

    # -- batch -------------------------------------------------------------
    def _run_batch(self, beam, pcoll, sinks, transform) -> None:
        if transform is not None:
            # Iterable->Iterable transform (e.g. detection): gather the
            # bundle, apply, fan back out. Batch-oriented; the streaming path
            # below windows instead of gathering globally.
            pcoll = (
                pcoll
                | "ToListT" >> beam.combiners.ToList()
                | "Transform" >> beam.FlatMap(lambda rows: list(transform(rows)))
            )
        out = pcoll | "CountOut" >> beam.Map(_count_out)
        for i, sink in enumerate(sinks):
            native_write = sink.native_beam_write()
            if native_write is not None:
                out | f"Write{i}" >> native_write
            else:
                # Fallback: gather to one bundle and call the agnostic
                # write(). Fine for batch/small; native writers should be
                # provided for large or unbounded sinks.
                (
                    out
                    | f"ToList{i}" >> beam.combiners.ToList()
                    | f"Write{i}" >> beam.Map(sink.write)
                )

    # -- streaming ---------------------------------------------------------
    def _run_streaming(self, beam, pcoll, sinks, transform) -> None:
        from apache_beam.transforms.trigger import (
            AccumulationMode,
            AfterCount,
            AfterWatermark,
        )
        from apache_beam.transforms.window import SlidingWindows

        w = self._window
        accumulation = (
            AccumulationMode.ACCUMULATING
            if w.accumulation == "accumulating"
            else AccumulationMode.DISCARDING
        )
        # Composed trigger: optional speculative early panes (every N elements
        # before the watermark), the on-time pane at the watermark, then a late
        # pane per N late arrivals still within allowed lateness. Elements later
        # than that are dropped.
        early = (
            AfterCount(w.early_firing_count)
            if w.early_firing_count is not None
            else None
        )
        trigger = AfterWatermark(
            early=early, late=AfterCount(w.late_firing_count)
        )
        windowed = (
            pcoll
            | "Timestamp"
            >> beam.Map(lambda r: beam.window.TimestampedValue(r, r.ts / 1000.0))
            | "Window"
            >> beam.WindowInto(
                SlidingWindows(int(w.size_s), int(w.period_s)),
                trigger=trigger,
                allowed_lateness=w.allowed_lateness_s,
                accumulation_mode=accumulation,
            )
        )
        if transform is not None:
            # Detect per (entity, window): group this window's rows by entity,
            # apply the same transform to each bounded per-window bundle.
            out = (
                windowed
                | "KeyByEntity" >> beam.Map(lambda r: (r.group_id, r))
                | "GroupByEntity" >> beam.GroupByKey()
                | "Detect"
                >> beam.FlatMap(lambda kv, t=transform: list(t(list(kv[1]))))
            )
        else:
            out = windowed
        out = out | "CountOut" >> beam.Map(_count_out)
        for i, sink in enumerate(sinks):
            native_write = sink.native_beam_write()
            if native_write is not None:
                out | f"Write{i}" >> native_write
            else:
                # Streaming-safe fallback: call the agnostic write() per element
                # (one-record bundles) instead of an unbounded global gather.
                out | f"Write{i}" >> beam.Map(lambda r, s=sink: s.write([r]))