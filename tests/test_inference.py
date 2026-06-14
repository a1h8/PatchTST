"""PatchTSTInference (M1) tests.

Validates the inference engine against *synthetic* checkpoints — random-weight
PatchTST models saved to disk and reloaded — so the module is fully exercised
without training, data, or a GPU. Covers: checkpoint round-trip, both heads'
output shapes (single & batched), forecast residual/RMSE, RevIN invariance,
load-once behavior, and input validation.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inference import ModelSpec, PatchTSTInference
from inference.engine import _select_state_dict
from PatchTST_self_supervised.src.models.patchTST import PatchTST


def _spec() -> ModelSpec:
    return ModelSpec(
        c_in=3, context_length=16, target_length=4, patch_len=4, stride=4,
        n_layers=1, d_model=8, n_heads=2, d_ff=16,
    )


def _build(spec: ModelSpec, head_type: str) -> PatchTST:
    return PatchTST(
        c_in=spec.c_in, target_dim=spec.target_length, patch_len=spec.patch_len,
        stride=spec.stride, num_patch=spec.num_patch, n_layers=spec.n_layers,
        d_model=spec.d_model, n_heads=spec.n_heads, d_ff=spec.d_ff,
        shared_embedding=True, act="relu", res_attention=False, head_type=head_type,
    )


@pytest.fixture
def engine(tmp_path):
    spec = _spec()
    fc, rc = _build(spec, "prediction"), _build(spec, "pretrain")
    fp, rp = tmp_path / "fc.pth", tmp_path / "rc.pth"
    torch.save(fc.state_dict(), fp)
    torch.save({"model": rc.state_dict()}, rp)  # the {'model': ...} variant
    return PatchTSTInference.from_checkpoints(str(fp), str(rp), spec)


# ---- spec ------------------------------------------------------------------

def test_num_patch():
    assert _spec().num_patch == 4  # (16-4)//4 + 1


def test_from_dict_ignores_unknown_keys():
    spec = ModelSpec.from_dict(
        {"c_in": 2, "context_length": 8, "target_length": 2, "bogus": 99}
    )
    assert spec.c_in == 2 and spec.context_length == 8


# ---- checkpoint loading ----------------------------------------------------

def test_select_state_dict_handles_both_shapes():
    raw = {"a": 1}
    assert _select_state_dict(raw) is raw
    assert _select_state_dict({"model": raw, "opt": {}}) is raw


def test_non_strict_load_warns_on_missing_keys(tmp_path, caplog):
    spec = _spec()
    fc, rc = _build(spec, "prediction"), _build(spec, "pretrain")
    fp, rp = tmp_path / "fc.pth", tmp_path / "rc.pth"
    partial = fc.state_dict()
    partial.pop(next(iter(partial)))  # drop a key -> missing on load
    torch.save(partial, fp)
    torch.save(rc.state_dict(), rp)

    with caplog.at_level("WARNING"):
        PatchTSTInference.from_checkpoints(str(fp), str(rp), spec, strict=False)
    assert any("missing" in r.message for r in caplog.records)


def test_loads_once_and_does_not_reload_on_inference(tmp_path):
    spec = _spec()
    fc, rc = _build(spec, "prediction"), _build(spec, "pretrain")
    fp, rp = tmp_path / "fc.pth", tmp_path / "rc.pth"
    torch.save(fc.state_dict(), fp)
    torch.save(rc.state_dict(), rp)

    calls = {"n": 0}

    def counting_loader(path):
        calls["n"] += 1
        return torch.load(path, map_location="cpu")

    eng = PatchTSTInference.from_checkpoints(
        str(fp), str(rp), spec, _loader=counting_loader
    )
    assert calls["n"] == 2  # one load per checkpoint, at construction

    w = np.random.randn(16, 3).astype("float32")
    eng.forecast(w)
    eng.reconstruct(w)
    assert calls["n"] == 2  # inference must not reload the model


# ---- forecast --------------------------------------------------------------

def test_forecast_shape_single(engine):
    out = engine.forecast(np.random.randn(16, 3).astype("float32"))
    assert out.prediction.shape == (4, 3)
    assert out.residual is None and out.rmse is None


def test_forecast_shape_batched(engine):
    out = engine.forecast(np.random.randn(5, 16, 3).astype("float32"))
    assert out.prediction.shape == (5, 4, 3)


def test_forecast_with_future_fills_residual(engine):
    w = np.random.randn(16, 3).astype("float32")
    fut = np.random.randn(4, 3).astype("float32")
    out = engine.forecast(w, future=fut)
    assert out.residual.shape == (4, 3)
    assert out.rmse_per_channel.shape == (3,)
    assert out.rmse == pytest.approx(
        float(np.sqrt(np.mean(out.residual**2))), rel=1e-5
    )


def test_forecast_deterministic_in_eval(engine):
    w = np.random.randn(16, 3).astype("float32")
    a = engine.forecast(w).prediction
    b = engine.forecast(w).prediction
    np.testing.assert_allclose(a, b)  # eval() => dropout off => repeatable


# ---- reconstruct -----------------------------------------------------------

def test_reconstruct_shape_single(engine):
    out = engine.reconstruct(np.random.randn(16, 3).astype("float32"))
    assert out.reconstruction.shape == (4, 3, 4)  # [num_patch, c_in, patch_len]
    assert out.error_per_channel.shape == (3,)
    assert out.error >= 0.0


def test_reconstruct_shape_batched(engine):
    out = engine.reconstruct(np.random.randn(2, 16, 3).astype("float32"))
    assert out.reconstruction.shape == (2, 4, 3, 4)
    assert out.error_per_channel.shape == (2, 3)


# ---- RevIN -----------------------------------------------------------------

def test_revin_roundtrip_invariance(engine):
    from PatchTST_self_supervised.src.models.layers.revin import RevIN

    r = RevIN(3, affine=False)
    x = torch.randn(2, 16, 3)
    back = r(r(x, "norm"), "denorm")
    assert torch.max(torch.abs(x - back)).item() < 1e-4


# ---- validation ------------------------------------------------------------

def test_wrong_window_length_raises(engine):
    with pytest.raises(ValueError, match="length"):
        engine.forecast(np.random.randn(10, 3).astype("float32"))


def test_wrong_channel_count_raises(engine):
    with pytest.raises(ValueError, match="channels"):
        engine.forecast(np.random.randn(16, 5).astype("float32"))


def test_bad_ndim_raises(engine):
    with pytest.raises(ValueError, match="2D|3D"):
        engine.forecast(np.random.randn(16).astype("float32"))
