"""PatchTST forecast-residual detector (decision D1, forecast face).

Trains a small PatchTST forecaster on the early part of a signal, then scores the
recent window by forecast error relative to the model's own baseline error:

    score = eval_RMSE / baseline_RMSE        (ratio, >1 = worse than usual)
    score >= warning  -> warning ; >= critical -> critical

This deliberately mirrors kube-verdict's `signals/patchtst_detector.py` method so
the signals we aggregate match its language. It plugs in behind the ``Detector``
interface (drop-in for ``ZScoreDetector``); short signals fall back to z-score.

Heavy deps (torch + transformers) are imported lazily inside ``detect`` so the
``detection`` package and the rest of the pipeline import without them.

Note: training on the fly is accurate and self-contained but compute-heavy — fit
for periodic/scoped assessment, not high-frequency cluster-wide scoring. A
pretrained inference variant can replace the training path behind this same class.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from kb.signal import SignalRecord

from .detector import Detector, ZScoreDetector

log = logging.getLogger(__name__)


def _sliding_windows(
    signal: np.ndarray, context_length: int, prediction_length: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    step = max(1, context_length // 4)
    window = context_length + prediction_length
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(0, len(signal) - window + 1, step):
        ctx = signal[i : i + context_length].copy()
        tgt = signal[i + context_length : i + window].copy()
        pairs.append((ctx, tgt))
    return pairs


@dataclass
class PatchTSTDetector(Detector):
    context_length: int = 64
    patch_length: int = 8
    prediction_length: int = 8
    d_model: int = 32
    num_heads: int = 4
    num_layers: int = 2
    epochs: int = 30
    lr: float = 5e-4
    warning: float = 1.8
    critical: float = 3.0
    fallback: Detector = field(default_factory=ZScoreDetector)

    method = "patchtst"

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
        if len(v) >= self.context_length + self.prediction_length:
            try:
                return self._detect_patchtst(entity_uid, metric_name, v, ts, labels)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("PatchTST detect failed (%s); falling back to z-score", exc)
        return self.fallback.detect(entity_uid, metric_name, values, ts, labels)

    def _detect_patchtst(
        self,
        entity_uid: str,
        metric_name: str,
        v: np.ndarray,
        ts: int,
        labels: dict | None,
    ) -> SignalRecord:
        import torch
        from transformers import PatchTSTConfig, PatchTSTForPrediction

        mu = float(v.mean())
        sigma = float(v.std()) or 1.0
        norm = (v - mu) / sigma

        split = max(self.context_length, int(len(norm) * 0.80))
        windows = _sliding_windows(norm[:split], self.context_length, self.prediction_length)
        if not windows:
            return self.fallback.detect(entity_uid, metric_name, v, ts, labels)

        config = PatchTSTConfig(
            num_input_channels=1,
            context_length=self.context_length,
            patch_length=self.patch_length,
            stride=self.patch_length,
            prediction_length=self.prediction_length,
            d_model=self.d_model,
            num_attention_heads=self.num_heads,
            num_hidden_layers=self.num_layers,
            ffn_dim=self.d_model * 4,
            dropout=0.1,
            head_dropout=0.0,
            pooling_type=None,
            channel_attention=False,
            scaling="std",
            loss="mse",
            pre_norm=True,
        )
        model = PatchTSTForPrediction(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        model.train()
        for _ in range(self.epochs):
            for ctx, tgt in windows:
                optimizer.zero_grad()
                past = torch.from_numpy(ctx).float().unsqueeze(0).unsqueeze(-1)
                future = torch.from_numpy(tgt).float().unsqueeze(0).unsqueeze(-1)
                model(past_values=past, future_values=future).loss.backward()
                optimizer.step()

        model.eval()
        start = len(norm) - self.context_length - self.prediction_length
        ctx = norm[start : start + self.context_length]
        tgt = norm[start + self.context_length : start + self.context_length + self.prediction_length]
        with torch.no_grad():
            past = torch.from_numpy(ctx).float().unsqueeze(0).unsqueeze(-1)
            pred = model(past_values=past).prediction_outputs.squeeze().numpy()

        eval_rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))
        baseline = _baseline_rmse(model, windows[-5:], torch)
        score = eval_rmse / max(baseline, 1e-8)

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


def _baseline_rmse(model, windows, torch) -> float:
    errors: list[float] = []
    model.eval()
    for ctx, tgt in windows:
        with torch.no_grad():
            past = torch.from_numpy(ctx).float().unsqueeze(0).unsqueeze(-1)
            pred = model(past_values=past).prediction_outputs.squeeze().numpy()
        errors.append(float(np.sqrt(np.mean((pred - tgt) ** 2))))
    return float(np.mean(errors)) if errors else 1.0
