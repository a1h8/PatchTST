"""Inference-backed PatchTST detectors (roadmap M3).

Replace the train-on-the-fly path of ``PatchTSTDetector`` / ``ReconstructionDetector``
with the **load-once M1 engine** (``inference.PatchTSTInference``): the model is
loaded once per process from checkpoints and scored under ``torch.no_grad``, never
refit per window. The scoring keeps the *same* ratio-to-baseline + severity
semantics as the trained detectors, so the emitted ``SignalRecord``s speak the
same language to kube-verdict — only the model source changes (a pretrained
checkpoint instead of an on-the-fly fit). For high-frequency cluster-wide
scoring this is the path; the trained detectors stay valid for scoped, no-checkpoint
assessment.

Univariate by construction: detection runs per ``(entity, metric)`` (see
``detection.aggregate``), so each window is ``[context_length, 1]`` and the engine
spec carries ``c_in=1``. The engine holds *both* heads, so one loaded instance
serves the forecast (anticipation) and reconstruction (detective) faces of a
``RegimeSwitchDetector`` — see the process-level engine cache below.

``torch`` and the vendored model package are imported lazily inside the engine, so
importing this module (and constructing a detector) stays cheap; only the first
``detect`` that actually loads a checkpoint pulls them in.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from inference import ModelSpec
from kb.signal import SignalRecord

from .detector import Detector, ZScoreDetector

log = logging.getLogger(__name__)


# --- process-level engine cache (load-once) ---------------------------------

# Keyed by (forecast_ckpt, reconstruct_ckpt, device, spec). ``ModelSpec`` is a
# frozen dataclass, hence hashable, so the whole key is hashable. One entry per
# distinct checkpoint pair → the forecast and reconstruction detectors that share
# the same checkpoints share a single loaded engine.
_ENGINE_CACHE: dict[tuple, object] = {}


def _load_engine(forecast_ckpt: str, reconstruct_ckpt: str, spec: ModelSpec, device: str):
    """Return a cached ``PatchTSTInference`` for these checkpoints (load once)."""
    key = (forecast_ckpt, reconstruct_ckpt, device, spec)
    engine = _ENGINE_CACHE.get(key)
    if engine is None:
        from inference import PatchTSTInference  # lazy: pulls torch + model pkg

        engine = PatchTSTInference.from_checkpoints(
            forecast_ckpt, reconstruct_ckpt, spec, device=device
        )
        _ENGINE_CACHE[key] = engine
    return engine


def clear_engine_cache() -> None:
    """Drop all cached engines (test/teardown helper)."""
    _ENGINE_CACHE.clear()


# --- base -------------------------------------------------------------------

@dataclass
class _InferenceDetector(Detector):
    """Common wiring for checkpoint-backed PatchTST detectors.

    Provide *either* a pre-built ``engine`` (injection seam — tests, or a
    worker that loaded it once) *or* ``(forecast_ckpt, reconstruct_ckpt, spec)``
    to lazy-load and cache one. ``spec`` may be a plain dict (parsed YAML); it is
    coerced to :class:`ModelSpec`.
    """

    forecast_ckpt: str | None = None
    reconstruct_ckpt: str | None = None
    spec: ModelSpec | dict | None = None
    device: str = "cpu"
    warning: float = 1.8
    critical: float = 3.0
    engine: object | None = None
    fallback: Detector = field(default_factory=ZScoreDetector)

    # set by subclasses
    method = "patchtst"

    def __post_init__(self) -> None:
        if isinstance(self.spec, dict):
            self.spec = ModelSpec.from_dict(self.spec)

    # ---- engine access ------------------------------------------------------

    def _get_engine(self):
        if self.engine is not None:
            return self.engine
        if not (self.forecast_ckpt and self.reconstruct_ckpt and self.spec):
            raise ValueError(
                "inference detector needs either an `engine` or "
                "(forecast_ckpt, reconstruct_ckpt, spec)"
            )
        self.engine = _load_engine(
            self.forecast_ckpt, self.reconstruct_ckpt, self.spec, self.device
        )
        return self.engine

    def _model_spec(self) -> ModelSpec:
        if isinstance(self.spec, ModelSpec):
            return self.spec
        return self._get_engine().spec

    # ---- scoring (subclass hooks) ------------------------------------------

    def _min_points(self, spec: ModelSpec) -> int:
        """Shortest series this detector can score (else z-score fallback)."""
        raise NotImplementedError

    def _score(self, engine, v: np.ndarray) -> float:
        """Ratio of the most recent window's error to the rolling baseline."""
        raise NotImplementedError

    # ---- Detector interface -------------------------------------------------

    def _severity(self, score: float) -> str:
        if score >= self.critical:
            return "critical"
        if score >= self.warning:
            return "warning"
        return "normal"

    def detect(
        self,
        entity_uid: str,
        metric_name: str,
        values: Sequence[float],
        ts: int,
        labels: dict | None = None,
    ) -> SignalRecord:
        v = np.asarray(values, dtype=np.float32)
        if len(v) >= self._min_points(self._model_spec()):
            try:
                score = self._score(self._get_engine(), v)
                return SignalRecord(
                    entity_uid=entity_uid,
                    metric_name=metric_name,
                    ts=ts,
                    severity=self._severity(score),
                    score=round(float(score), 4),
                    method=self.method,
                    n_points=int(len(v)),
                    labels=dict(labels or {}),
                )
            except Exception as exc:  # pragma: no cover - defensive
                log.warning(
                    "%s inference detect failed (%s); falling back to z-score",
                    self.method, exc,
                )
        return self.fallback.detect(entity_uid, metric_name, values, ts, labels)


def _baseline(errors: list[float], eval_err: float) -> float:
    """Rolling baseline = mean of the last few earlier-window errors.

    With no earlier window (series only long enough for the eval window) the
    baseline is the eval error itself → score 1.0 (normal): we cannot call a
    deviation without a reference.
    """
    return float(np.mean(errors[-5:])) if errors else eval_err


# --- forecast face (anticipation) -------------------------------------------

@dataclass
class ForecastInferenceDetector(_InferenceDetector):
    """Forecast-residual detector backed by the M1 engine's prediction head.

    Drop-in for ``PatchTSTDetector`` (``method="patchtst"``): the recent window's
    forecast RMSE relative to the model's own rolling baseline RMSE.
    """

    method = "patchtst"

    def _min_points(self, spec: ModelSpec) -> int:
        return spec.context_length + spec.target_length

    def _score(self, engine, v: np.ndarray) -> float:
        spec = engine.spec
        c, h = spec.context_length, spec.target_length
        win = c + h
        step = max(1, c // 4)
        split = max(win, int(len(v) * 0.80))

        errors: list[float] = []
        for i in range(0, split - win + 1, step):
            ctx = v[i : i + c, None]
            fut = v[i + c : i + win, None]
            errors.append(engine.forecast(ctx, fut).rmse)

        start = len(v) - win
        eval_err = engine.forecast(v[start : start + c, None], v[start + c :, None]).rmse
        return eval_err / max(_baseline(errors, eval_err), 1e-8)


# --- reconstruction face (detective) ----------------------------------------

@dataclass
class ReconstructionInferenceDetector(_InferenceDetector):
    """Reconstruction-error detector backed by the M1 engine's pretrain head.

    Drop-in for ``ReconstructionDetector`` (``method="patchtst-recon"``): the recent
    window's reconstruction error relative to the rolling baseline. A model that
    learned normal patterns reconstructs a broken window poorly, so the error spikes.
    """

    method = "patchtst-recon"

    def _min_points(self, spec: ModelSpec) -> int:
        return spec.context_length

    def _score(self, engine, v: np.ndarray) -> float:
        c = engine.spec.context_length
        step = max(1, c // 4)
        split = max(c, int(len(v) * 0.80))

        errors: list[float] = []
        for i in range(0, split - c + 1, step):
            errors.append(engine.reconstruct(v[i : i + c, None]).error)

        eval_err = engine.reconstruct(v[len(v) - c :, None]).error
        return eval_err / max(_baseline(errors, eval_err), 1e-8)
