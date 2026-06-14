"""PatchTSTInference — the M1 engine.

Loads two PatchTST checkpoints once (forecast = prediction head, reconstruct =
pretrain head) and exposes them as ``forecast(window)`` / ``reconstruct(window)``.
RevIN instance-normalizes each window the same way training did (affine-free):
forecast denormalizes its prediction back to the input's scale; reconstruction
error is reported in normalized space (scale-invariant, the right unit for an
anomaly score).

Heavy deps (``torch``, the vendored ``PatchTST_self_supervised`` package) are
imported lazily so ``import inference`` stays cheap for callers that only need the
config. Construct the engine once per worker; ``forecast``/``reconstruct`` then
run under ``torch.no_grad`` with no per-call model rebuild.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import ModelSpec

log = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Forecast head output, denormalized to the input window's scale.

    ``prediction`` has shape ``[target_length, c_in]`` (or ``[bs, target_length,
    c_in]`` for a batched call). ``residual``/``rmse_per_channel``/``rmse`` are
    populated only when the caller passes the realized ``future`` values.
    """

    prediction: np.ndarray
    residual: np.ndarray | None = None
    rmse_per_channel: np.ndarray | None = None
    rmse: float | None = None


@dataclass
class ReconstructResult:
    """Reconstruction head output and error, in normalized space.

    ``reconstruction`` is the patched reconstruction ``[num_patch, c_in,
    patch_len]`` (or batched with a leading ``bs``), the native pretrain-head
    layout. ``error_per_channel`` is the per-channel reconstruction MSE against
    the patched input; higher means the window is less in-distribution.
    """

    reconstruction: np.ndarray
    error_per_channel: np.ndarray
    error: float


def _select_state_dict(raw: dict) -> dict:
    """Accept either a bare state_dict or ``{'model': ..., 'opt': ...}``."""
    if "model" in raw and isinstance(raw["model"], dict):
        return raw["model"]
    return raw


class PatchTSTInference:
    """Checkpoint-backed PatchTST inference with both heads.

    Build via :meth:`from_checkpoints`. The forecast and reconstruct models are
    independent ``PatchTST`` instances (their backbones diverged during finetune),
    each loaded from its own checkpoint and set to ``eval()`` on ``device``.
    """

    def __init__(self, forecast_model, reconstruct_model, spec: ModelSpec, device: str):
        self.spec = spec
        self.device = device
        self._forecast_model = forecast_model
        self._reconstruct_model = reconstruct_model

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_checkpoints(
        cls,
        forecast_ckpt: str,
        reconstruct_ckpt: str,
        spec: ModelSpec,
        device: str = "cpu",
        strict: bool = True,
        _loader: Callable | None = None,
    ) -> "PatchTSTInference":
        """Load both checkpoints once and return a ready engine.

        ``_loader`` is an injection seam for tests (defaults to ``torch.load``);
        production callers leave it unset.
        """
        import torch

        load = _loader or (lambda p: torch.load(p, map_location=device))

        forecast_model = cls._build(spec, head_type="prediction")
        reconstruct_model = cls._build(spec, head_type="pretrain")

        cls._load_into(forecast_model, load(forecast_ckpt), strict, "forecast")
        cls._load_into(reconstruct_model, load(reconstruct_ckpt), strict, "reconstruct")

        forecast_model.to(device).eval()
        reconstruct_model.to(device).eval()
        return cls(forecast_model, reconstruct_model, spec, device)

    @staticmethod
    def _build(spec: ModelSpec, head_type: str):
        from PatchTST_self_supervised.src.models.patchTST import PatchTST

        return PatchTST(
            c_in=spec.c_in,
            target_dim=spec.target_length,
            patch_len=spec.patch_len,
            stride=spec.stride,
            num_patch=spec.num_patch,
            n_layers=spec.n_layers,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            shared_embedding=spec.shared_embedding,
            d_ff=spec.d_ff,
            dropout=spec.dropout,
            head_dropout=spec.head_dropout,
            act=spec.activation,
            res_attention=spec.res_attention,
            head_type=head_type,
        )

    @staticmethod
    def _load_into(model, raw: dict, strict: bool, label: str) -> None:
        result = model.load_state_dict(_select_state_dict(raw), strict=strict)
        missing = getattr(result, "missing_keys", [])
        unexpected = getattr(result, "unexpected_keys", [])
        if missing or unexpected:
            log.warning(
                "%s checkpoint loaded with missing=%s unexpected=%s",
                label, missing, unexpected,
            )

    # ---- inference ----------------------------------------------------------

    def forecast(self, window: np.ndarray, future: np.ndarray | None = None) -> ForecastResult:
        """Forecast the next ``target_length`` steps from a context window.

        ``window``: ``[context_length, c_in]`` or batched ``[bs, context_length,
        c_in]``. If ``future`` (the realized horizon, same channel layout) is
        given, residual and RMSE are filled in.
        """
        import torch

        x, batched = self._to_btc(window, self.spec.context_length, "window")
        revin = self._revin()
        with torch.no_grad():
            xn = revin(x, "norm")
            patched = self._patch(xn)
            pred = self._forecast_model(patched)          # [bs, target_length, c_in]
            pred = revin(pred, "denorm")
        pred_np = self._out(pred, batched)

        res = ForecastResult(prediction=pred_np)
        if future is not None:
            fut, _ = self._to_btc(future, self.spec.target_length, "future")
            fut_np = self._out(fut, batched)
            diff = pred_np - fut_np
            res.residual = diff
            axis = tuple(range(diff.ndim - 1))            # all but channel axis
            res.rmse_per_channel = np.sqrt(np.mean(diff**2, axis=axis))
            res.rmse = float(np.sqrt(np.mean(diff**2)))
        return res

    def reconstruct(self, window: np.ndarray) -> ReconstructResult:
        """Reconstruct a window and report per-channel reconstruction error.

        ``window``: ``[context_length, c_in]`` or batched. Error is the MSE
        between the patched input and its reconstruction, in normalized space.
        """
        import torch

        x, batched = self._to_btc(window, self.spec.context_length, "window")
        revin = self._revin()
        with torch.no_grad():
            xn = revin(x, "norm")
            patched = self._patch(xn)                      # [bs, num_patch, c_in, patch_len]
            recon = self._reconstruct_model(patched)       # same shape
            err = (recon - patched) ** 2

        # mean over patches & patch_len, keep batch and channel: [bs, c_in]
        err_pc = err.mean(dim=(1, 3)).cpu().numpy()
        recon_np = recon.cpu().numpy()
        if not batched:
            recon_np = recon_np[0]
            err_pc = err_pc[0]
        return ReconstructResult(
            reconstruction=recon_np,
            error_per_channel=err_pc,
            error=float(err_pc.mean()),
        )

    # ---- helpers ------------------------------------------------------------

    def _revin(self):
        from PatchTST_self_supervised.src.models.layers.revin import RevIN

        return RevIN(self.spec.c_in, affine=False).to(self.device)

    def _patch(self, xn):
        from PatchTST_self_supervised.src.callback.patch_mask import create_patch

        patched, _ = create_patch(xn, self.spec.patch_len, self.spec.stride)
        return patched

    def _to_btc(self, arr: np.ndarray, expect_len: int, name: str):
        """Coerce to a ``[bs, length, c_in]`` float tensor; report if batched."""
        import torch

        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 2:
            a = a[None, ...]
            batched = False
        elif a.ndim == 3:
            batched = True
        else:
            raise ValueError(f"{name} must be 2D [L, C] or 3D [B, L, C], got shape {a.shape}")

        if a.shape[1] != expect_len:
            raise ValueError(
                f"{name} length {a.shape[1]} != expected {expect_len}"
            )
        if a.shape[2] != self.spec.c_in:
            raise ValueError(
                f"{name} channels {a.shape[2]} != c_in {self.spec.c_in}"
            )
        return torch.from_numpy(a).to(self.device), batched

    @staticmethod
    def _out(t, batched: bool) -> np.ndarray:
        a = t.cpu().numpy()
        return a if batched else a[0]
