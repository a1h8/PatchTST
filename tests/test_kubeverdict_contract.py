"""Contract tests: the *shape* PatchTST owes kube-verdict, not detection quality.

kube-verdict's webhook contract (its ``api/models.py::SignalAlert``, a separate
repo) is mirrored here by hand -- there is nothing to import across repos, so
this file is the one place that freezes the exact field/type contract
``KubeVerdictAlertSink`` must keep producing. ``scenarios/library.py`` +
``tools/capture_signals.py`` validate whether PatchTST's detectors catch the
right incidents; this validates whether what they emit still speaks
kube-verdict's language once it reaches the webhook.
"""
from __future__ import annotations

import json
import math

import pytest

import kb.alert as alert_mod
from connectors.pivot import PivotRow
from detection.aggregate import detect_signals
from detection.detector import ZScoreDetector
from kb.alert import KubeVerdictAlertSink
from kb.signal import SEVERITIES, SignalRecord

# Mirrors kube-verdict's api/models.py::SignalAlert field-by-field. Update by
# hand if that schema changes -- there is no shared package to import from.
SIGNAL_ALERT_SCHEMA: dict[str, type | tuple[type, ...]] = {
    "entity_uid": str,
    "metric_name": str,
    "ts": int,
    "severity": str,
    "score": (int, float),
    "method": str,
    "horizon": str,
    "n_points": int,
    "labels": dict,
    "text": str,
}


def _assert_matches_contract(alert: dict) -> None:
    assert set(alert) == set(SIGNAL_ALERT_SCHEMA), (
        f"alert keys {sorted(alert)} != contract keys {sorted(SIGNAL_ALERT_SCHEMA)}"
    )
    for field, expected_type in SIGNAL_ALERT_SCHEMA.items():
        assert isinstance(alert[field], expected_type), (
            f"{field}: expected {expected_type}, got {type(alert[field]).__name__}"
        )
    assert alert["severity"] in SEVERITIES
    assert not isinstance(alert["score"], bool)
    assert not (math.isnan(alert["score"]) or math.isinf(alert["score"]))
    assert all(
        isinstance(k, str) and isinstance(v, str) for k, v in alert["labels"].items()
    )
    json.dumps(alert)  # must round-trip through JSON exactly as posted


def _push(records: list[SignalRecord], *, min_severity: str = "normal") -> list[dict]:
    """Run records through the real sink (severity filter included) and
    capture what it would have POSTed, via the same seam the existing
    tests/test_kubeverdict_alert.py mock uses."""
    sent: list[dict] = []
    orig = alert_mod.post_alerts
    alert_mod.post_alerts = lambda ep, alerts, **kw: sent.extend(alerts)
    try:
        KubeVerdictAlertSink("http://kv", min_severity=min_severity).write(records)
    finally:
        alert_mod.post_alerts = orig
    return sent


@pytest.mark.parametrize(
    "record",
    [
        SignalRecord(
            entity_uid="Pod/prod/api-1", metric_name="cpu", ts=1_000,
            severity="warning", score=3.2, method="zscore",
        ),
        SignalRecord(
            entity_uid="Pod/prod/api-1", metric_name="cpu", ts=1_000,
            severity="critical", score=9.9, method="patchtst",
            horizon="short", n_points=64,
            labels={"namespace": "demo", "pod": "node1"},
        ),
        SignalRecord(
            entity_uid="node1/demo", metric_name="__entity__", ts=1_000,
            severity="critical", score=5.5, method="aggregate",
            labels={"namespace": "demo", "n_channels": "2"},
        ),
    ],
    ids=["minimal-defaults", "full-fields-with-labels", "entity-rollup"],
)
def test_alert_matches_signal_alert_contract(record):
    sent = _push([record])
    assert len(sent) == 1
    _assert_matches_contract(sent[0])


def test_severities_match_kubeverdict_enum():
    # kube-verdict's own zscore/patchtst methods only ever emit these three;
    # kb/alert.py's min_severity ranking assumes this exact ordering.
    assert SEVERITIES == ("normal", "warning", "critical")


def test_resource_labels_survive_the_full_detection_pipeline():
    """namespace/pod are what kube-verdict's signal_mapper needs to scope an
    RCA investigation -- verify they reach the posted alert through the real
    detect_signals transform, not a hand-built SignalRecord."""
    labels = {"namespace": "demo", "pod": "node1"}
    rows = [
        PivotRow(group_id="node1/demo", ts=60_000 * i, values=(0.2,),
                 channels=("cpu",), labels=labels)
        for i in range(9)
    ] + [
        PivotRow(group_id="node1/demo", ts=60_000 * 9, values=(50.0,),
                 channels=("cpu",), labels=labels),
    ]

    signals = list(detect_signals(rows, ZScoreDetector(), now_ms=600_000))
    assert signals, "detect_signals produced nothing"

    sent = _push(signals)
    assert sent, "the anomalous spike should have cleared min_severity=normal"
    for alert in sent:
        _assert_matches_contract(alert)
    assert any(
        alert["labels"].get("namespace") == "demo"
        and alert["labels"].get("pod") == "node1"
        for alert in sent
    )
