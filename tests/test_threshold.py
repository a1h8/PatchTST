"""Tests for the M4 threshold policies and their wiring into inference detectors."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from detection import (
    FixedThreshold,
    ForecastInferenceDetector,
    MADThreshold,
    RollingQuantileThreshold,
    build_threshold,
    clear_engine_cache,
)
from inference import ModelSpec

KEY = ("Pod/p", "cpu")
SPEC = ModelSpec(c_in=1, context_length=16, target_length=4, patch_len=4, stride=4)


# --- fixed ------------------------------------------------------------------

def test_fixed_threshold_boundaries():
    t = FixedThreshold(warning=1.8, critical=3.0)
    assert t.classify(KEY, 1.0) == "normal"
    assert t.classify(KEY, 1.8) == "warning"   # >= warning
    assert t.classify(KEY, 3.0) == "critical"  # >= critical


# --- rolling quantile -------------------------------------------------------

def test_rolling_quantile_cold_start_uses_fixed_cut():
    t = RollingQuantileThreshold(min_history=10, warning=1.8, critical=3.0)
    # empty history → fixed cut applies
    assert t.classify(KEY, 1.0) == "normal"
    assert t.classify(KEY, 3.5) == "critical"


def test_rolling_quantile_adapts_after_warmup():
    t = RollingQuantileThreshold(window=64, min_history=10, warn_q=0.95, crit_q=0.99)
    for s in [1.0, 1.1, 0.9, 1.05, 0.95] * 4:   # 20 samples, tight around ~1
        t.classify(KEY, s)
    # a value inside the learned spread is normal; a clear spike is critical
    assert t.classify(KEY, 1.0) == "normal"
    assert t.classify(KEY, 10.0) == "critical"


def test_rolling_window_forgets_old_scores():
    t = RollingQuantileThreshold(window=10, min_history=5)
    for _ in range(10):
        t.classify(KEY, 1.0)              # fill the window with ~1.0
    assert len(t._hist[KEY]) == 10        # capped at window


# --- MAD --------------------------------------------------------------------

def test_mad_is_robust_to_a_single_outlier():
    t = MADThreshold(window=64, min_history=10, k_warn=3.0, k_crit=5.0)
    hist = [1.0] * 19 + [50.0]            # one big outlier in the history
    for s in hist:
        t.classify(KEY, s)
    # median/MAD shrug off the lone outlier → a normal value stays normal
    assert t.classify(KEY, 1.0) == "normal"
    assert t.classify(KEY, 100.0) == "critical"


def test_mad_flat_history_does_not_false_positive():
    t = MADThreshold(min_history=5)
    for _ in range(10):
        t.classify(KEY, 2.0)             # perfectly flat history (MAD == 0)
    assert t.classify(KEY, 2.0) == "normal"   # equal value not flagged
    assert t.classify(KEY, 9.0) == "critical"


# --- per-key isolation ------------------------------------------------------

def test_threshold_history_is_per_key():
    t = RollingQuantileThreshold(window=64, min_history=5)
    for _ in range(10):
        t.classify(("a", "m"), 1.0)
    # a different key is still cold → fixed cut, not 'a's learned spread
    assert t.classify(("b", "m"), 1.0) == "normal"
    assert len(t._hist[("b", "m")]) == 1


# --- config builder ---------------------------------------------------------

def test_build_threshold_types():
    assert isinstance(build_threshold({"type": "fixed"}), FixedThreshold)
    assert isinstance(
        build_threshold({"type": "rolling-quantile", "params": {"window": 32}}),
        RollingQuantileThreshold,
    )
    assert isinstance(build_threshold({"type": "mad"}), MADThreshold)


def test_build_threshold_unknown_raises():
    with pytest.raises(KeyError, match="unknown threshold policy"):
        build_threshold({"type": "nope"})


# --- wiring into the inference detector -------------------------------------

class FakeEngine:
    def __init__(self, spec=SPEC):
        self.spec = spec

    def forecast(self, window, future):
        return SimpleNamespace(rmse=abs(float(np.asarray(future)[-1, 0])))

    def reconstruct(self, window):
        return SimpleNamespace(error=abs(float(np.asarray(window)[-1, 0])))


@pytest.fixture(autouse=True)
def _clear():
    clear_engine_cache()
    yield
    clear_engine_cache()


def test_detector_defaults_to_fixed_threshold():
    det = ForecastInferenceDetector(engine=FakeEngine())
    assert isinstance(det.threshold, FixedThreshold)
    assert det.threshold.warning == det.warning and det.threshold.critical == det.critical


def test_detector_coerces_threshold_dict():
    det = ForecastInferenceDetector(
        engine=FakeEngine(),
        threshold={"type": "rolling-quantile", "params": {"min_history": 4}},
    )
    assert isinstance(det.threshold, RollingQuantileThreshold)
    assert det.threshold.min_history == 4


def test_detector_uses_adaptive_threshold_end_to_end():
    # forecast score = |last future value|; feed many calm series then a spike.
    det = ForecastInferenceDetector(
        engine=FakeEngine(),
        threshold={"type": "rolling-quantile", "params": {"min_history": 5, "window": 50}},
    )
    calm = [0.1] * 40
    for i in range(8):
        sig = det.detect("e", "m", calm, ts=i)
        assert sig.severity == "normal"
    spike = [0.1] * 40
    spike[-1] = 10.0
    assert det.detect("e", "m", spike, ts=99).severity == "critical"
