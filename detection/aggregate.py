"""Detection transform — PivotRows (raw metrics) → SignalRecords (signals).

This is the pipeline stage between a source and the signal-store sink. It groups
the multivariate ``PivotRow`` stream by entity (``group_id``) and channel into
per-metric series, runs the detector once per (entity, metric), and emits one
``SignalRecord`` per assessment.

Pure Python and engine-agnostic: it is the ``transform`` an engine applies
between ``read`` and ``write`` (LocalEngine / BeamEngine).
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Iterable, Iterator

from connectors.pivot import PivotRow
from kb.signal import SignalRecord

from .detector import Detector


def detect_signals(
    rows: Iterable[PivotRow],
    detector: Detector,
    *,
    now_ms: int | None = None,
) -> Iterator[SignalRecord]:
    """One SignalRecord per (entity, metric) assessed over its window."""
    ts = now_ms if now_ms is not None else int(time.time() * 1000)

    # group_id -> channel -> [(ts, value)]
    series: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    labels: dict[str, dict] = {}
    for r in rows:
        labels[r.group_id] = dict(r.labels)
        for ch, val in zip(r.channels, r.values):
            series[r.group_id][ch].append((r.ts, float(val)))

    for group_id, channels in series.items():
        for channel, points in channels.items():
            points.sort()
            values = [v for _, v in points]
            yield detector.detect(
                group_id, channel, values, ts, labels=labels.get(group_id)
            )


def make_detection_transform(
    detector: Detector, *, now_ms: int | None = None
) -> Callable[[Iterable[PivotRow]], Iterator[SignalRecord]]:
    """Bind a detector into a transform callable for ``Engine.run(..., transform=)``."""

    def _transform(rows: Iterable[PivotRow]) -> Iterator[SignalRecord]:
        return detect_signals(rows, detector, now_ms=now_ms)

    return _transform
