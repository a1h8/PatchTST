"""PatchTST reconstruction detector (decision D1, detective face).

Trains a self-supervised PatchTST (masked-patch reconstruction) on the early
part of a signal, then scores the recent window by reconstruction error relative
to the model's own baseline error:

    score = eval_recon_error / baseline_recon_error      (method="patchtst-recon")
    score >= warning -> warning ; >= critical -> critical

This is the detective face of D1: a model that learned to reconstruct *normal*
patterns reconstructs an out-of-distribution (broken) window poorly, so the error
spikes cleanly — unlike the forecaster, which degrades under regime change. Short
signals fall back to z-score. Plugs in behind the ``Detector`` interface.

torch / transformers are imported lazily inside ``detect`` so the package imports
without them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from kb.signal import SignalRecord

from .detector import Detector, ZScoreDetector

log = logging.getLogger(__name__)


def _recon_windows(signal: np.ndarray, context_length: int) -> list[np.ndarray]:
    step = max(1, context_length // 4)
    return [
        signal[i : i + context_length].copy()
        for i in range(0, len(signal) - context_length + 1, step)
    ]


@dataclass
class ReconstructionDetector(Detector):
    context_length: int = 64
    patch_length: int = 8
    d_model: int = 32
    num_heads: int = 4
    num_layers: int = 2
    epochs: int = 30
    lr: float = 5e-4
    mask_ratio: float = 0.4
    warning: float = 1.8
    critical: float = 3.0
    fallback: Detector = field(default_factory=ZScoreDetector)

    method = "patchtst-recon"

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
        if len(v) >= self.context_length:
            try:
                return self._detect_recon(entity_uid, metric_name, v, ts, labels)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("recon detect failed (%s); falling back to z-score", exc)
        return self.fallback.detect(entity_uid, metric_name, values, ts, labels)

    def _detect_recon(
        self,
        entity_uid: str,
        metric_name: str,
        v: np.ndarray,
        ts: int,
        labels: dict | None,
    ) -> SignalRecord:
        import torch
        from transformers import PatchTSTConfig, PatchTSTForPretraining

        mu = float(v.mean())
        sigma = float(v.std()) or 1.0
        norm = (v - mu) / sigma

        split = max(self.context_length, int(len(norm) * 0.80))
        # entry guard (len >= context_length) ⇒ split >= context_length ⇒ ≥1 window
        windows = _recon_windows(norm[:split], self.context_length)

        config = PatchTSTConfig(
            num_input_channels=1,
            context_length=self.context_length,
            patch_length=self.patch_length,
            stride=self.patch_length,
            d_model=self.d_model,
            num_attention_heads=self.num_heads,
            num_hidden_layers=self.num_layers,
            ffn_dim=self.d_model * 4,
            dropout=0.1,
            mask_type="random",
            random_mask_ratio=self.mask_ratio,
            scaling="std",
        )
        model = PatchTSTForPretraining(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for _ in range(self.epochs):
            for ctx in windows:
                optimizer.zero_grad()
                past = torch.from_numpy(ctx).float().unsqueeze(0).unsqueeze(-1)
                model(past_values=past).loss.backward()
                optimizer.step()

        model.eval()
        baseline = _recon_loss(model, windows[-5:], torch)
        eval_ctx = norm[-self.context_length :]
        eval_loss = _recon_loss(model, [eval_ctx], torch)
        score = eval_loss / max(baseline, 1e-8)

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


def _recon_loss(model, windows, torch) -> float:
    losses: list[float] = []
    model.eval()
    for ctx in windows:
        with torch.no_grad():
            past = torch.from_numpy(ctx).float().unsqueeze(0).unsqueeze(-1)
            losses.append(float(model(past_values=past).loss))
    return float(np.mean(losses)) if losses else 1.0
