"""Detection tests — real ZScoreDetector + the PivotRow→SignalRecord transform,
and the full write path Mimir → detection → signal-store → query."""
import pytest

import kb  # noqa: F401  (registers signal-store sink)
from connectors import LocalEngine, build
from connectors.pivot import PivotRow
from detection import ZScoreDetector, detect_signals, make_detection_transform

STABLE_THEN_SPIKE = [10.0] * 50 + [200.0]   # clear anomaly in the tail (z ~ 7)


# --- ZScoreDetector (real detection) --------------------------------------

def test_detector_flags_anomaly_critical():
    r = ZScoreDetector().detect("Pod/p/a", "cpu", STABLE_THEN_SPIKE, ts=1000)
    assert r.severity == "critical" and r.score > 4.5
    assert r.method == "zscore" and r.n_points == 51


def test_detector_warning_band_via_thresholds():
    # same spike, thresholds bracket the score into the warning band
    r = ZScoreDetector(warning=2.0, critical=100.0).detect("e", "cpu", STABLE_THEN_SPIKE, ts=0)
    assert r.severity == "warning"


def test_detector_constant_series_is_normal():
    r = ZScoreDetector().detect("e", "cpu", [5.0] * 20, ts=0)
    assert r.severity == "normal" and r.score == 0.0


def test_detector_short_series_is_normal():
    r = ZScoreDetector(min_points=8).detect("e", "cpu", [1.0, 99.0], ts=0)
    assert r.severity == "normal" and r.score == 0.0 and r.n_points == 2


# --- transform: PivotRows -> SignalRecords --------------------------------

def test_detect_signals_one_per_entity_metric():
    rows = [
        PivotRow("e1", 1000, (1.0, 10.0), ("cpu", "mem")),
        PivotRow("e1", 2000, (1.1, 11.0), ("cpu", "mem")),
        PivotRow("e2", 1000, (5.0,), ("cpu",)),
    ]
    out = list(detect_signals(rows, ZScoreDetector(), now_ms=999))
    keys = {(s.entity_uid, s.metric_name) for s in out}
    assert keys == {("e1", "cpu"), ("e1", "mem"), ("e2", "cpu")}
    assert all(s.ts == 999 for s in out)


def test_detect_signals_default_now_ms_is_set():
    rows = [PivotRow("e", 1, (1.0,), ("cpu",))]
    (sig,) = list(detect_signals(rows, ZScoreDetector()))
    assert sig.ts > 0


# --- full write path: Mimir → detection → signal-store → query ------------

def _mimir_result(values):
    return [
        {
            "metric": {"__name__": "cpu", "instance": "pod-a"},
            "values": [[i * 15, str(v)] for i, v in enumerate(values)],
        }
    ]


def test_write_path_local_engine_produces_real_signal(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "connectors.sources.mimir.query_range",
        lambda *a, **k: _mimir_result(STABLE_THEN_SPIKE),
    )
    source = build("mimir", endpoint="http://m", promql="cpu", start=0, end=999, step_s=15)
    sink = build("signal-store", root=str(tmp_path / "kb"))
    transform = make_detection_transform(ZScoreDetector(), now_ms=1700000000000)

    LocalEngine().run(source, [sink], transform=transform)

    signals = sink.store.query("pod-a")
    assert len(signals) == 1
    sig = signals[0]
    assert sig.entity_uid == "pod-a" and sig.metric_name == "cpu"
    assert sig.severity == "critical" and sig.is_anomalous
    assert sig.ts == 1700000000000


def test_write_path_beam_engine_with_transform(monkeypatch, tmp_path):
    pytest.importorskip("apache_beam")
    from connectors.engines.beam import BeamEngine

    monkeypatch.setattr(
        "connectors.sources.mimir.query_range",
        lambda *a, **k: _mimir_result(STABLE_THEN_SPIKE),
    )
    source = build("mimir", endpoint="http://m", promql="cpu", start=0, end=999, step_s=15)
    sink = build("signal-store", root=str(tmp_path / "kb"))
    transform = make_detection_transform(ZScoreDetector(), now_ms=1700000000000)

    BeamEngine().run(source, [sink], transform=transform)

    signals = sink.store.query("pod-a")
    assert len(signals) == 1 and signals[0].severity == "critical"
