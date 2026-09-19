"""Level-shift check: the score itself, and its effect on the regime switch
(stub faces, no torch)."""
import numpy as np

from detection import RegimeSwitchDetector, level_shift_score
from detection.detector import Detector
from kb.signal import SignalRecord


def _spiky(n=120, shift_from=None, shift=30.0):
    """Baseline ~5 with periodic 3-point spikes; optionally a sustained shift."""
    rng = np.random.default_rng(0)
    v = 5.0 + rng.normal(0.0, 0.5, n)
    for i in range(n):
        if i % 20 < 3:
            v[i] += 40.0
        elif shift_from is not None and i >= shift_from:
            v[i] += shift
    return v


def test_too_short_returns_none():
    assert level_shift_score([1.0] * 10) is None


def test_stationary_series_scores_low_despite_spikes():
    assert level_shift_score(_spiky()) < 2.0


def test_sustained_plateau_scores_high():
    assert level_shift_score(_spiky(shift_from=90)) > 8.0


def test_slow_ramp_stays_below_critical():
    ramp = np.linspace(0.4, 0.6, 200) + np.random.default_rng(1).normal(0, 0.01, 200)
    assert level_shift_score(ramp) < 8.0


def test_constant_baseline_tiny_change_is_not_enormous():
    v = [10.0] * 60 + [10.1] * 20
    assert level_shift_score(v) < 1.0


class _Fixed(Detector):
    def __init__(self, method, severity):
        self.method, self.severity = method, severity

    def detect(self, entity_uid, metric_name, values, ts, labels=None):
        return SignalRecord(
            entity_uid, metric_name, ts, self.severity, score=1.0,
            method=self.method, labels=dict(labels or {}),
        )


def _detector(**kw):
    return RegimeSwitchDetector(
        forecast=_Fixed("patchtst", "normal"),
        detective=_Fixed("patchtst-recon", "normal"),
        **kw,
    )


def test_displaced_level_enters_incident_and_escalates_severity():
    d = _detector()
    sig = d.detect("e", "m", _spiky(shift_from=90).tolist(), ts=0)
    assert sig.labels["regime"] == "incident"
    assert sig.severity == "critical"          # forecast said normal
    assert float(sig.labels["level_shift"]) > 8.0


def test_incident_holds_while_level_displaced_then_exits_on_return():
    d = _detector()
    shifted = _spiky(shift_from=90).tolist()
    d.detect("e", "m", shifted, ts=0)                       # -> incident
    held = d.detect("e", "m", shifted, ts=1)                # recon normal, still displaced
    assert held.labels["regime"] == "incident"
    back = d.detect("e", "m", _spiky().tolist(), ts=2)      # level recovered
    assert back.labels["regime"] == "normal"


def test_stationary_series_does_not_flip():
    sig = _detector().detect("e", "m", _spiky().tolist(), ts=0)
    assert sig.labels["regime"] == "normal" and sig.severity == "normal"


def test_disabled_check_restores_previous_behaviour():
    d = _detector(level_critical=None)
    sig = d.detect("e", "m", _spiky(shift_from=90).tolist(), ts=0)
    assert sig.labels["regime"] == "normal" and "level_shift" not in sig.labels
