"""Entity-rollup tests — per-channel residuals → one multivariate verdict (M4)."""
import math

import pytest

from connectors.pivot import PivotRow
from detection import (
    ENTITY_METRIC,
    ZScoreDetector,
    aggregate_entity,
    aggregate_signals,
    detect_signals,
    make_detection_transform,
)
from kb.signal import SignalRecord


def _sig(entity, metric, score, severity, *, ts=1000, n=50):
    return SignalRecord(
        entity_uid=entity,
        metric_name=metric,
        ts=ts,
        severity=severity,
        score=score,
        method="zscore",
        n_points=n,
    )


# --- aggregate_entity: the multivariate verdict ---------------------------

def test_severity_is_worst_channel():
    sigs = [
        _sig("e", "cpu", 1.0, "normal"),
        _sig("e", "mem", 2.0, "warning"),
        _sig("e", "io", 0.5, "normal"),
    ]
    roll = aggregate_entity(sigs)
    assert roll.severity == "warning"          # mem drives the entity verdict
    assert roll.entity_uid == "e"
    assert roll.metric_name == ENTITY_METRIC
    assert roll.method == "aggregate"


def test_score_policy_max_mean_l2():
    sigs = [_sig("e", "cpu", 3.0, "warning"), _sig("e", "mem", 4.0, "critical")]
    assert aggregate_entity(sigs, policy="max").score == 4.0
    assert aggregate_entity(sigs, policy="mean").score == 3.5
    assert aggregate_entity(sigs, policy="l2").score == round(math.hypot(3.0, 4.0), 4)


def test_labels_record_contributors():
    sigs = [
        _sig("e", "cpu", 5.0, "critical"),
        _sig("e", "mem", 2.0, "warning"),
        _sig("e", "io", 0.1, "normal"),
    ]
    roll = aggregate_entity(sigs, policy="mean")
    assert roll.labels["n_channels"] == "3"
    assert roll.labels["n_anomalous"] == "2"
    assert roll.labels["top_channel"] == "cpu"     # highest score
    assert roll.labels["rollup_policy"] == "mean"


def test_entity_labels_are_carried_through():
    sigs = [_sig("e", "cpu", 1.0, "normal")]
    roll = aggregate_entity(sigs, labels={"env": "prod"})
    assert roll.labels["env"] == "prod"


def test_ts_and_n_points_from_channels():
    sigs = [_sig("e", "cpu", 1.0, "normal", ts=7, n=10), _sig("e", "mem", 1.0, "normal", ts=7, n=40)]
    roll = aggregate_entity(sigs)
    assert roll.ts == 7 and roll.n_points == 40


def test_empty_raises():
    with pytest.raises(ValueError):
        aggregate_entity([])


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        aggregate_entity([_sig("e", "cpu", 1.0, "normal")], policy="median")


# --- aggregate_signals: group a flat stream by entity ---------------------

def test_aggregate_signals_one_rollup_per_entity():
    stream = [
        _sig("e1", "cpu", 1.0, "normal"),
        _sig("e1", "mem", 4.0, "critical"),
        _sig("e2", "cpu", 0.0, "normal"),
    ]
    rolls = list(aggregate_signals(stream))
    by_entity = {r.entity_uid: r for r in rolls}
    assert set(by_entity) == {"e1", "e2"}
    assert by_entity["e1"].severity == "critical"
    assert by_entity["e2"].severity == "normal"


def test_aggregate_signals_is_idempotent_skips_rollups():
    base = [_sig("e", "cpu", 4.0, "critical")]
    once = list(aggregate_signals(base))
    # feeding rollups back in must not produce rollups-of-rollups
    twice = list(aggregate_signals(base + once))
    assert len(twice) == 1 and twice[0].metric_name == ENTITY_METRIC


# --- detect_signals / transform: per-channel + rollup ---------------------

def test_detect_signals_rollup_emitted_per_entity():
    rows = [
        PivotRow("e1", 1000, (10.0, 1.0), ("cpu", "mem")),
        PivotRow("e2", 1000, (5.0,), ("cpu",)),
    ]
    out = list(detect_signals(rows, ZScoreDetector(), now_ms=999, entity_rollup=True))
    rollups = [s for s in out if s.metric_name == ENTITY_METRIC]
    assert {r.entity_uid for r in rollups} == {"e1", "e2"}
    # per-channel signals are still present alongside the rollups
    assert any(s.metric_name == "cpu" for s in out)
    assert all(s.ts == 999 for s in out)


def test_detect_signals_no_rollup_by_default():
    rows = [PivotRow("e", 1000, (1.0,), ("cpu",))]
    out = list(detect_signals(rows, ZScoreDetector(), now_ms=1))
    assert not any(s.metric_name == ENTITY_METRIC for s in out)


def test_transform_forwards_rollup_flag():
    rows = [PivotRow("e", 1000, (1.0, 2.0), ("cpu", "mem"))]
    transform = make_detection_transform(
        ZScoreDetector(), now_ms=1, entity_rollup=True, rollup_policy="l2"
    )
    out = list(transform(rows))
    (roll,) = [s for s in out if s.metric_name == ENTITY_METRIC]
    assert roll.labels["rollup_policy"] == "l2"