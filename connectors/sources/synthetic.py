"""Synthetic streaming source — a deterministic unbounded series for M5.

The streaming analog of the Mimir batch source: it fabricates aligned
multivariate ``PivotRow``s on a fixed grid so the streaming engine path
(sliding windows, watermarks, triggers, late data) can be exercised on the
DirectRunner without a real broker. A real low-latency ingress (Kafka/OTLP) is a
separate connector with its own native unbounded read (decision D2 / roadmap M6).

Two views of the *same* logical data, by design:

- ``read()`` — engine-agnostic, returns the full clean series in event-time
  order. Lateness is an event-time/streaming concept the agnostic view does not
  model, so ``read()`` never reorders or holds anything back.
- ``native_beam_read()`` — a Beam ``TestStream`` that *replays* that series with
  a delivery schedule: it advances the watermark as it emits on-time elements
  and (if ``late_at`` is set) delivers one element **after** the watermark has
  passed its window, so the engine's ``allowed_lateness`` policy decides whether
  it is admitted or dropped.
"""
from __future__ import annotations

from typing import Any

from ..base import SourceConnector
from ..pivot import PivotRow
from ..registry import connector


@connector("synthetic-stream")
class SyntheticStreamSource(SourceConnector):
    """Deterministic multivariate series generator for streaming tests/demos."""

    def __init__(
        self,
        *,
        group_ids: list[str] | None = None,
        channels: list[str] | None = None,
        start_ms: int = 0,
        n_points: int = 8,
        step_s: int = 15,
        base: float = 1.0,
        slope: float = 0.0,
        anomaly_at: int | None = None,
        anomaly_value: float | None = None,
        late_at: int | None = None,
    ) -> None:
        self.group_ids = group_ids or ["g0"]
        self.channels = channels or ["m0"]
        self.start_ms = start_ms
        self.n_points = n_points
        self.step_s = step_s
        self.base = base
        self.slope = slope
        self.anomaly_at = anomaly_at
        self.anomaly_value = anomaly_value
        self.late_at = late_at

    def _value(self, i: int) -> float:
        if self.anomaly_at is not None and i == self.anomaly_at:
            return self.anomaly_value if self.anomaly_value is not None else self.base * 10
        return self.base + self.slope * i

    def _generate(self) -> list[PivotRow]:
        """Full clean series, in event-time order."""
        step_ms = self.step_s * 1000
        channels = tuple(self.channels)
        rows: list[PivotRow] = []
        for i in range(self.n_points):
            ts = self.start_ms + i * step_ms
            values = tuple(self._value(i) for _ in channels)
            for gid in self.group_ids:
                rows.append(
                    PivotRow(group_id=gid, ts=ts, values=values, channels=channels)
                )
        return rows

    def read(self) -> list[PivotRow]:
        # Agnostic view: the whole logical series, no lateness modeling.
        return self._generate()

    def native_beam_read(self):
        import apache_beam as beam
        from apache_beam.testing.test_stream import TestStream

        rows = self._generate()
        # one point index (across all groups) replayed late, if requested
        late_ts = (
            self.start_ms + self.late_at * self.step_s * 1000
            if self.late_at is not None
            else None
        )

        stream = TestStream()
        held: list[PivotRow] = []
        for r in rows:
            if late_ts is not None and r.ts == late_ts:
                held.append(r)  # deliver after the watermark passes its window
                continue
            stream = stream.advance_watermark_to(r.ts / 1000.0)
            stream = stream.add_elements(
                [beam.window.TimestampedValue(r, r.ts / 1000.0)]
            )
        for r in held:
            # watermark is already past r.ts -> this is a late element
            stream = stream.add_elements(
                [beam.window.TimestampedValue(r, r.ts / 1000.0)]
            )
        return stream.advance_watermark_to_infinity()

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "group_ids": self.group_ids,
            "channels": self.channels,
            "n_points": self.n_points,
            "step_s": self.step_s,
            "late_at": self.late_at,
        }