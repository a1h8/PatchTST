"""Tests for the inference-backed PatchTST detectors (roadmap M3).

A fake engine stands in for ``inference.PatchTSTInference`` so the detector logic
(window iteration, ratio-to-baseline scoring, severity, fallback, load-once cache,
runner wiring) is exercised without torch or a checkpoint. The fake reports a
per-window scalar driven by the window's *last* value, so a spike planted at the
tail makes the eval window's error dominate the rolling baseline → a clean ratio.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from detection import (
    ForecastInferenceDetector,
    ReconstructionInferenceDetector,
    clear_engine_cache,
)
from detection import inference_detector as idet
from inference import ModelSpec
from pipeline.runner import build_detector

SPEC = ModelSpec(c_in=1, context_length=16, target_length=4, patch_len=4, stride=4)


class FakeEngine:
    """Stand-in engine: error driven by the last value of the scored segment."""

    def __init__(self, spec: ModelSpec = SPEC) -> None:
        self.spec = spec

    def forecast(self, window, future):
        return SimpleNamespace(rmse=abs(float(np.asarray(future)[-1, 0])))

    def reconstruct(self, window):
        return SimpleNamespace(error=abs(float(np.asarray(window)[-1, 0])))


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_engine_cache()
    yield
    clear_engine_cache()


def _flat(n: int, value: float = 0.1) -> list[float]:
    return [value] * n


# --- forecast face ----------------------------------------------------------

def test_forecast_flat_series_is_normal():
    det = ForecastInferenceDetector(engine=FakeEngine())
    sig = det.detect("e", "m", _flat(40, 1.0), ts=1)
    assert sig.method == "patchtst"
    assert sig.severity == "normal"
    assert sig.score == pytest.approx(1.0, abs=1e-6)


def test_forecast_tail_spike_is_critical():
    v = _flat(40, 0.1)
    v[-1] = 10.0  # eval window's last future value spikes
    sig = ForecastInferenceDetector(engine=FakeEngine()).detect("e", "m", v, ts=1)
    assert sig.method == "patchtst"
    assert sig.severity == "critical"
    assert sig.score > 3.0


def test_forecast_short_series_falls_back_to_zscore():
    sig = ForecastInferenceDetector(engine=FakeEngine()).detect("e", "m", _flat(10), ts=1)
    assert sig.method == "zscore"


def test_no_engine_no_checkpoint_falls_back_to_zscore():
    # long-enough series, but neither engine nor checkpoints → load fails →
    # the defensive guard falls back to z-score instead of raising.
    det = ForecastInferenceDetector(spec=SPEC)  # spec known, but no ckpt/engine
    sig = det.detect("e", "m", _flat(40, 1.0), ts=1)
    assert sig.method == "zscore"


# --- reconstruction face ----------------------------------------------------

def test_reconstruction_flat_series_is_normal():
    sig = ReconstructionInferenceDetector(engine=FakeEngine()).detect(
        "e", "m", _flat(40, 1.0), ts=1
    )
    assert sig.method == "patchtst-recon"
    assert sig.severity == "normal"


def test_reconstruction_tail_spike_is_critical():
    v = _flat(40, 0.1)
    v[-1] = 10.0
    sig = ReconstructionInferenceDetector(engine=FakeEngine()).detect("e", "m", v, ts=1)
    assert sig.method == "patchtst-recon"
    assert sig.severity == "critical"


def test_reconstruction_short_series_falls_back():
    sig = ReconstructionInferenceDetector(engine=FakeEngine()).detect("e", "m", _flat(8), ts=1)
    assert sig.method == "zscore"


# --- spec coercion ----------------------------------------------------------

def test_spec_dict_is_coerced(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(idet, "_load_engine", lambda *a, **k: fake)
    det = ForecastInferenceDetector(
        forecast_ckpt="f.pt",
        reconstruct_ckpt="r.pt",
        spec={"c_in": 1, "context_length": 16, "target_length": 4,
              "patch_len": 4, "stride": 4, "ignored": "x"},
    )
    assert isinstance(det.spec, ModelSpec)
    assert det.spec.context_length == 16
    # min_points known from spec without loading the engine
    assert det.detect("e", "m", _flat(40, 1.0), ts=1).method == "patchtst"


# --- checkpoint_dir auto-resume (M7 retrain pointer) ------------------------

def _write_latest(out_dir, *, with_spec=True, version="v1"):
    from dataclasses import asdict

    from inference.retrain import write_version

    meta = {"spec": asdict(SPEC)} if with_spec else {}
    return write_version(
        out_dir,
        forecast_ckpt="f.pt",
        reconstruct_ckpt="r.pt",
        meta=meta,
        version=version,
    )


def test_checkpoint_dir_resolves_latest_and_spec(tmp_path, monkeypatch):
    out = str(tmp_path / "models")
    _write_latest(out)
    seen = {}

    def fake_from_ckpts(fc, rc, spec, device="cpu"):
        seen["ckpts"] = (fc, rc)
        return FakeEngine(spec)

    monkeypatch.setattr(
        "inference.PatchTSTInference.from_checkpoints", staticmethod(fake_from_ckpts)
    )
    det = ForecastInferenceDetector(checkpoint_dir=out)  # no ckpts, no spec
    sig = det.detect("e", "m", _flat(40, 1.0), ts=1)

    assert sig.method == "patchtst" and sig.severity == "normal"
    assert seen["ckpts"] == ("f.pt", "r.pt")  # followed latest.json
    assert isinstance(det.spec, ModelSpec)  # spec read back from the manifest
    assert det.spec.context_length == 16


def test_checkpoint_dir_before_first_retrain_falls_back(tmp_path):
    # No latest.json yet (loop never ran) → graceful z-score fallback, no raise.
    det = ForecastInferenceDetector(checkpoint_dir=str(tmp_path / "models"))
    assert det.detect("e", "m", _flat(40, 1.0), ts=1).method == "zscore"


def test_explicit_ckpts_take_precedence_over_checkpoint_dir(tmp_path, monkeypatch):
    out = str(tmp_path / "models")
    _write_latest(out)  # manifest points at f.pt/r.pt
    seen = {}
    monkeypatch.setattr(
        "inference.PatchTSTInference.from_checkpoints",
        staticmethod(lambda fc, rc, spec, device="cpu": seen.setdefault("ckpts", (fc, rc)) or FakeEngine(spec)),
    )
    det = ForecastInferenceDetector(
        checkpoint_dir=out, forecast_ckpt="explicit-f.pt",
        reconstruct_ckpt="explicit-r.pt", spec=SPEC,
    )
    det.detect("e", "m", _flat(40, 1.0), ts=1)
    assert seen["ckpts"] == ("explicit-f.pt", "explicit-r.pt")


def test_checkpoint_dir_via_runner_config(tmp_path, monkeypatch):
    out = str(tmp_path / "models")
    _write_latest(out)
    monkeypatch.setattr(
        "inference.PatchTSTInference.from_checkpoints",
        staticmethod(lambda fc, rc, spec, device="cpu": FakeEngine(spec)),
    )
    det = build_detector({"type": "patchtst-infer", "params": {"checkpoint_dir": out}})
    assert det.detect("e", "m", _flat(40, 1.0), ts=1).method == "patchtst"


# --- load-once engine cache -------------------------------------------------

def test_engine_cache_loads_once(monkeypatch):
    calls = {"n": 0}

    def fake_from_ckpts(fc, rc, spec, device="cpu"):
        calls["n"] += 1
        return FakeEngine(spec)

    monkeypatch.setattr(
        "inference.PatchTSTInference.from_checkpoints", staticmethod(fake_from_ckpts)
    )
    a = idet._load_engine("f.pt", "r.pt", SPEC, "cpu")
    b = idet._load_engine("f.pt", "r.pt", SPEC, "cpu")
    assert a is b
    assert calls["n"] == 1

    c = idet._load_engine("other.pt", "r.pt", SPEC, "cpu")
    assert c is not a
    assert calls["n"] == 2


# --- runner wiring: shared engine across regime-switch faces -----------------

def test_regime_switch_faces_share_one_engine(monkeypatch):
    # Patch the real loader (not _load_engine) so the process cache dedupes:
    # two detectors built from the same checkpoints must resolve one engine.
    engines = []

    def fake_from_ckpts(fc, rc, spec, device="cpu"):
        eng = FakeEngine(spec)
        engines.append(eng)
        return eng

    monkeypatch.setattr(
        "inference.PatchTSTInference.from_checkpoints", staticmethod(fake_from_ckpts)
    )

    params = {
        "forecast_ckpt": "f.pt",
        "reconstruct_ckpt": "r.pt",
        "spec": {"c_in": 1, "context_length": 16, "target_length": 4,
                 "patch_len": 4, "stride": 4},
    }
    det = build_detector({
        "type": "regime-switch",
        "forecast": {"type": "patchtst-infer", "params": params},
        "detective": {"type": "reconstruction-infer", "params": params},
    })

    # drive both faces: NORMAL→forecast (spike → incident), then INCIDENT→recon
    v = _flat(40, 0.1)
    v[-1] = 10.0
    first = det.detect("e", "m", v, ts=1)
    second = det.detect("e", "m", v, ts=2)

    assert first.method == "patchtst" and first.labels["mode"] == "anticipation"
    assert second.method == "patchtst-recon" and second.labels["mode"] == "detective"
    # both faces resolved the same cached engine → loaded exactly once
    assert len(engines) == 1
