"""Tests for the reference-checkpoint driver (PR-C).

Covers the pure data helpers and a tiny end-to-end pretrain -> finetune -> report
cycle on synthetic data (no network, no real dataset): a 1-epoch run with a
minimal spec, exercising the same code path that produces the ETTh1 checkpoints.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inference import train_reference as tr
from inference.config import ModelSpec


class _Args:
    def __init__(self, **kw):
        defaults = dict(
            dset="synth", context_length=16, target_length=4, patch_len=4, stride=4,
            n_layers=1, d_model=8, n_heads=2, d_ff=16, mask_ratio=0.5, batch_size=8,
            win_stride=2, lr=1e-3, pretrain_epochs=1, finetune_epochs=1,
        )
        defaults.update(kw)
        self.__dict__.update(defaults)


def _series(n=200, c=7):
    t = np.linspace(0, 20 * np.pi, n)
    return np.stack([np.sin(t + i) for i in range(c)], axis=1).astype(np.float32)


# ---- pure helpers ----------------------------------------------------------

def test_load_channels(tmp_path):
    csv = tmp_path / "ETTh1.csv"
    csv.write_text("date,a,b,c,d,e,f,g\n2016,1,2,3,4,5,6,7\n2016,8,9,10,11,12,13,14\n")
    out = tr._load_channels(str(csv))
    assert out.shape == (2, 7)
    assert out[1, 0] == 8.0


def test_standardize_zero_mean_unit_std_on_train():
    data = _series(100)
    z = tr._standardize(data, train_end=80)
    assert np.allclose(z[:80].mean(axis=0), 0, atol=1e-4)
    assert np.allclose(z[:80].std(axis=0), 1, atol=1e-4)


def test_standardize_handles_constant_channel():
    data = np.ones((10, 2), dtype=np.float32)
    z = tr._standardize(data, train_end=10)
    assert np.isfinite(z).all()  # sd==0 guarded


def test_windows_shape():
    w = tr._windows(_series(50), length=16, stride=4)
    assert w.shape[1:] == (16, 7) and w.shape[0] == (50 - 16) // 4 + 1


def test_pairs_alignment():
    s = np.arange(60).reshape(60, 1).astype(np.float32)
    x, y = tr._pairs(s, ctx=10, horizon=4, stride=5)
    assert x.shape[1:] == (10, 1) and y.shape[1:] == (4, 1)
    # y immediately follows x
    assert y[0, 0, 0] == x[0, -1, 0] + 1


def test_batches_cover_all_indices():
    seen = np.concatenate([sel for sel in tr._batches(10, 4)])
    assert sorted(seen) == list(range(10))


def test_build_spec_uses_seven_channels():
    spec = tr.build_spec(_Args())
    assert spec.c_in == 7 and spec.context_length == 16


# ---- end-to-end (tiny) -----------------------------------------------------

def test_pretrain_finetune_report_cycle(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tr, "CKPT_DIR", str(tmp_path))
    args = _Args()
    spec = ModelSpec(c_in=7, context_length=16, target_length=4, patch_len=4,
                     stride=4, n_layers=1, d_model=8, n_heads=2, d_ff=16)
    data = tr._standardize(_series(200), train_end=150)
    train, val = data[:150], data[150:]

    rc = tr.pretrain(spec, train, args, device="cpu")
    fc = tr.finetune(spec, train, rc, args, device="cpu")
    tr.report(spec, val, fc, rc, device="cpu")

    import os
    assert os.path.exists(rc) and os.path.exists(fc)
    out = capsys.readouterr().out
    assert "forecast RMSE" in out and "reconstruction error" in out


def test_transfer_backbone_raises_without_overlap(tmp_path):
    spec = ModelSpec(c_in=7, context_length=16, target_length=4, patch_len=4,
                     stride=4, n_layers=1, d_model=8, n_heads=2, d_ff=16)
    # save an unrelated state_dict -> no shared backbone keys
    bad = tmp_path / "bad.pth"
    torch.save({"nope": torch.zeros(1)}, bad)
    model = tr.PatchTSTInference._build(spec, head_type="prediction")
    with pytest.raises(RuntimeError, match="no shared backbone"):
        tr._transfer_backbone(str(bad), model, device="cpu")
