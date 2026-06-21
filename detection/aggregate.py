"""Detection transform — PivotRows (raw metrics) → SignalRecords (signals).

This is the pipeline stage between a source and the signal-store sink. It groups
the multivariate ``PivotRow`` stream by entity (``group_id``) and channel into
per-metric series, runs the detector once per (entity, metric), and emits one
``SignalRecord`` per assessment.

Optionally (``entity_rollup=True``) it also emits, per entity, a **multivariate
verdict** that aggregates the per-channel residuals into a single entity-level
signal (roadmap M4). The per-channel detail is kept as RCA evidence; the rollup
is the "is this entity in trouble?" answer for downstream alerting.

Pure Python and engine-agnostic: it is the ``transform`` an engine applies
between ``read`` and ``write`` (LocalEngine / BeamEngine).
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Iterable, Iterator, Sequence

from connectors.pivot import PivotRow
from kb.signal import SEVERITIES, SignalRecord

from .detector import Detector

#: sentinel ``metric_name`` of an entity-level (multivariate) rollup signal.
ENTITY_METRIC = "__entity__"

#: severity ordering for "worst channel drives the entity verdict".
_SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}


def detect_signals(
    rows: Iterable[PivotRow],
    detector: Detector,
    *,
    now_ms: int | None = None,
    entity_rollup: bool = False,
    rollup_policy: str = "max",
) -> Iterator[SignalRecord]:
    """One SignalRecord per (entity, metric) assessed over its window.

    With ``entity_rollup``, also yield one multivariate rollup per entity (after
    its per-channel signals), aggregating their scores via ``rollup_policy``.
    """
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
        per_channel: list[SignalRecord] = []
        for channel, points in channels.items():
            points.sort()
            values = [v for _, v in points]
            sig = detector.detect(
                group_id, channel, values, ts, labels=labels.get(group_id)
            )
            per_channel.append(sig)
            yield sig
        if entity_rollup and per_channel:
            yield aggregate_entity(
                per_channel, policy=rollup_policy, labels=labels.get(group_id)
            )


def _rollup_score(scores: Sequence[float], policy: str) -> float:
    """Aggregate per-channel scores into one entity score.

    ``max`` (worst channel, aligns with the max-severity verdict), ``mean``
    (average pressure), or ``l2`` (magnitude of the residual vector across
    channels — the natural multivariate norm).
    """
    if policy == "max":
        return max(scores)
    if policy == "mean":
        return sum(scores) / len(scores)
    if policy == "l2":
        return sum(s * s for s in scores) ** 0.5
    raise ValueError(f"unknown rollup policy {policy!r}; use 'max' | 'mean' | 'l2'")


def aggregate_entity(
    signals: Sequence[SignalRecord],
    *,
    policy: str = "max",
    labels: dict | None = None,
) -> SignalRecord:
    """Roll up one entity's per-channel signals into one multivariate verdict.

    Severity is the worst channel's (the entity is as sick as its sickest
    channel); the score aggregates the channel scores via ``policy``; labels note
    the contributing channels. All inputs must share one ``entity_uid``/``ts``.
    """
    if not signals:
        raise ValueError("aggregate_entity needs at least one signal")

    worst = max(signals, key=lambda s: (_SEVERITY_RANK[s.severity], s.score))
    top = max(signals, key=lambda s: s.score)
    n_anom = sum(1 for s in signals if s.is_anomalous)

    rollup_labels = dict(labels or {})
    rollup_labels.update(
        n_channels=str(len(signals)),
        n_anomalous=str(n_anom),
        top_channel=top.metric_name,
        rollup_policy=policy,
    )
    return SignalRecord(
        entity_uid=signals[0].entity_uid,
        metric_name=ENTITY_METRIC,
        ts=signals[0].ts,
        severity=worst.severity,
        score=round(_rollup_score([s.score for s in signals], policy), 4),
        method="aggregate",
        horizon=worst.horizon,
        n_points=max(s.n_points for s in signals),
        labels=rollup_labels,
    )


def aggregate_signals(
    signals: Iterable[SignalRecord], *, policy: str = "max"
) -> Iterator[SignalRecord]:
    """Group a flat per-channel signal stream by entity, yield one rollup each.

    Rollups already in the stream (``metric_name == ENTITY_METRIC``) are skipped
    so the function is idempotent.
    """
    by_entity: dict[str, list[SignalRecord]] = defaultdict(list)
    for s in signals:
        if s.metric_name == ENTITY_METRIC:
            continue
        by_entity[s.entity_uid].append(s)
    for group in by_entity.values():
        yield aggregate_entity(group, policy=policy)


def make_detection_transform(
    detector: Detector,
    *,
    now_ms: int | None = None,
    entity_rollup: bool = False,
    rollup_policy: str = "max",
) -> Callable[[Iterable[PivotRow]], Iterator[SignalRecord]]:
    """Bind a detector into a transform callable for ``Engine.run(..., transform=)``."""

    def _transform(rows: Iterable[PivotRow]) -> Iterator[SignalRecord]:
        return detect_signals(
            rows,
            detector,
            now_ms=now_ms,
            entity_rollup=entity_rollup,
            rollup_policy=rollup_policy,
        )

    return _transform
