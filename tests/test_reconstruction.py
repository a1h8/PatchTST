"""ReconstructionDetector tests — fallback path (no torch) and the real
masked-reconstruction path (guarded by torch/transformers, tiny config)."""
import pytest

from detection import ReconstructionDetector
from detection.detector import ZScoreDetector


def _tiny() -> ReconstructionDetector:
    return ReconstructionDetector(
        context_length=16, patch_length=4, d_model=8,
        num_heads=2, num_layers=1, epochs=2,
    )


def test_recon_short_series_falls_back_to_zscore():
    r = _tiny().detect("e", "cpu", [1.0, 2.0, 3.0], ts=5)
    assert r.method == "zscore" and r.n_points == 3


def test_recon_is_detector_with_zscore_fallback():
    assert ReconstructionDetector.method == "patchtst-recon"
    assert isinstance(_tiny().fallback, ZScoreDetector)


def test_recon_severity_thresholds():
    d = ReconstructionDetector(warning=1.8, critical=3.0)
    assert d._severity(3.1) == "critical"
    assert d._severity(2.0) == "warning"
    assert d._severity(0.9) == "normal"


def test_recon_path_produces_signal():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import numpy as np

    vals = (10 + np.sin(np.linspace(0, 12, 60))).tolist()
    r = _tiny().detect("Pod/p/a", "cpu", vals, ts=1700000000000)

    assert r.method == "patchtst-recon"
    assert r.n_points == 60 and r.ts == 1700000000000
    assert r.score >= 0.0 and r.severity in ("normal", "warning", "critical")
