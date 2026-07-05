"""Retraining loop (M7) — KB-guided window selection + checkpoint versioning.

Torch-free: the heavy training (train_reference.pretrain / finetune) is
monkeypatched, so these tests exercise the selection logic, the version manifest
+ latest pointer, and the orchestration wiring — fast, no model, no GPU.
"""
import numpy as np
import pytest

from inference import retrain as R
from inference.config import ModelSpec
from kb.signal import SignalRecord


class _FakeStore:
    """Duck-typed SignalStore: serves a fixed signal list, filtered by time."""

    def __init__(self, signals):
        self._signals = signals

    def query(self, entity_uid, metric=None, since=None, until=None, limit=None):
        out = [s for s in self._signals if s.entity_uid == entity_uid]
        if metric is not None:
            out = [s for s in out if s.metric_name == metric]
        if since is not None:
            out = [s for s in out if s.ts >= since]
        if until is not None:
            out = [s for s in out if s.ts <= until]
        return out


def _sig(ts, severity, entity="e", metric="m"):
    return SignalRecord(
        entity_uid=entity,
        metric_name=metric,
        ts=ts,
        severity=severity,
        score=9.0 if severity != "normal" else 0.0,
        method="zscore",
    )


# -- selection --------------------------------------------------------------
def test_normal_keep_mask_drops_anomalous_points():
    store = _FakeStore([_sig(300, "critical"), _sig(0, "normal")])
    mask = R.normal_keep_mask(store, "e", ts_start=0, step_ms=100, n=5)
    assert list(mask) == [True, True, True, False, True]  # only ts=300 dropped


def test_keep_mask_exclude_radius_widens_around_incident():
    store = _FakeStore([_sig(200, "warning")])
    mask = R.normal_keep_mask(store, "e", 0, 100, 5, exclude_radius=1)
    assert list(mask) == [True, False, False, False, True]


def test_longest_normal_segment_picks_longest_run():
    series = np.arange(6).reshape(6, 1).astype(float)
    keep = np.array([True, False, True, True, True, False])
    seg = R.longest_normal_segment(series, keep)
    assert [v for v in seg[:, 0]] == [2.0, 3.0, 4.0]


def test_select_training_series_returns_clean_slice():
    store = _FakeStore([_sig(100, "critical")])  # idx1 anomalous
    series = np.arange(5).reshape(5, 1).astype(float)  # ts 0..400 step 100
    seg = R.select_training_series(store, "e", series, 0, 100)
    assert [v for v in seg[:, 0]] == [2.0, 3.0, 4.0]  # longest clean run


# -- versioning -------------------------------------------------------------
def test_write_version_and_resolve_latest(tmp_path):
    out = str(tmp_path / "models")
    manifest = R.write_version(
        out,
        forecast_ckpt="fc.pth",
        reconstruct_ckpt="rc.pth",
        meta={"n_train_rows": 10},
        version="v1",
    )
    assert manifest["version"] == "v1"
    assert (tmp_path / "models" / "v1" / "manifest.json").exists()

    latest = R.resolve_latest(out)
    assert latest["version"] == "v1"
    assert latest["forecast_ckpt"] == "fc.pth"
    assert latest["reconstruct_ckpt"] == "rc.pth"
    assert latest["n_train_rows"] == 10


def test_resolve_latest_none_when_never_trained(tmp_path):
    assert R.resolve_latest(str(tmp_path)) is None


# -- orchestration ----------------------------------------------------------
def test_retrain_excludes_incident_and_writes_version(tmp_path, monkeypatch):
    captured = {}

    def fake_pretrain(spec, train, args, device):
        captured["train_len"] = len(train)
        return "recon.pth"

    def fake_finetune(spec, train, rc, args, device):
        captured["rc_in"] = rc
        return "forecast.pth"

    monkeypatch.setattr("inference.train_reference.pretrain", fake_pretrain)
    monkeypatch.setattr("inference.train_reference.finetune", fake_finetune)

    spec = ModelSpec(c_in=1, context_length=2, target_length=1)
    store = _FakeStore([_sig(500, "critical")])  # incident at idx5 of 10
    series = np.arange(10).reshape(10, 1).astype(float)

    manifest = R.retrain(
        store,
        spec,
        {"e": series},
        args=object(),
        out_dir=str(tmp_path / "m"),
        ts_start=0,
        step_ms=100,
    )

    assert captured["train_len"] == 5  # longest clean run idx0..4, incident cut
    assert captured["rc_in"] == "recon.pth"
    assert manifest["forecast_ckpt"] == "forecast.pth"
    assert manifest["reconstruct_ckpt"] == "recon.pth"
    assert manifest["n_entities"] == 1
    assert manifest["n_train_rows"] == 5
    assert R.resolve_latest(str(tmp_path / "m"))["version"] == manifest["version"]


def test_retrain_raises_when_no_clean_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "inference.train_reference.pretrain",
        lambda *a, **k: pytest.fail("must not train on all-anomalous history"),
    )
    spec = ModelSpec(c_in=1, context_length=5, target_length=1)
    store = _FakeStore([_sig(t, "critical") for t in range(0, 500, 100)])
    series = np.arange(5).reshape(5, 1).astype(float)

    with pytest.raises(ValueError, match="no clean training windows"):
        R.retrain(
            store,
            spec,
            {"e": series},
            args=object(),
            out_dir=str(tmp_path),
            ts_start=0,
            step_ms=100,
        )
