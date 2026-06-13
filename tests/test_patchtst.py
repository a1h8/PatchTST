"""PatchTSTDetector tests — fallback path (no torch needed) and the real
forecast-residual path (guarded by torch/transformers, tiny config for speed)."""
import pytest

from detection import PatchTSTDetector
from detection.detector import ZScoreDetector


def _tiny() -> PatchTSTDetector:
    # tiny config keeps the trained model fast in tests
    return PatchTSTDetector(
        context_length=16, patch_length=4, prediction_length=4,
        d_model=8, num_heads=2, num_layers=1, epochs=2,
    )


def test_patchtst_short_series_falls_back_to_zscore():
    r = _tiny().detect("e", "cpu", [1.0, 2.0, 3.0], ts=5)
    assert r.method == "zscore" and r.severity == "normal" and r.n_points == 3


def test_patchtst_is_detector_with_zscore_fallback():
    assert PatchTSTDetector.method == "patchtst"
    assert isinstance(_tiny().fallback, ZScoreDetector)


def test_patchtst_severity_thresholds():
    d = PatchTSTDetector(warning=1.8, critical=3.0)
    assert d._severity(3.5) == "critical"
    assert d._severity(2.0) == "warning"
    assert d._severity(0.5) == "normal"


def test_patchtst_path_produces_signal():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import numpy as np

    vals = (10 + np.sin(np.linspace(0, 12, 60))).tolist()
    r = _tiny().detect("Pod/p/a", "cpu", vals, ts=1700000000000)

    assert r.method == "patchtst"
    assert r.n_points == 60 and r.ts == 1700000000000
    assert r.score >= 0.0 and r.severity in ("normal", "warning", "critical")


def test_patchtst_too_few_training_windows_falls_back():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    # len 22 >= context+pred (20) so it enters the PatchTST path, but the 80%
    # train split (17) yields no full training window -> z-score fallback.
    r = _tiny().detect("e", "cpu", [10.0] * 22, ts=1)
    assert r.method == "zscore"
