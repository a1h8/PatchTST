"""Variation detectors — turn a metric window into a SignalRecord.

The ``Detector`` interface is the seam where detection strategy plugs in.
``ZScoreDetector`` is a real statistical detector (not a stub): it mirrors the
z-score method and severity semantics of kube-verdict's own fallback, so the
signals we aggregate speak its language. A PatchTST-based detector (forecast /
reconstruction) can later implement the same interface without touching the
pipeline — see docs/ARCHITECTURE.md (D1).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from kb.signal import SignalRecord


class Detector(ABC):
    """Assess one metric series for one entity → a SignalRecord."""

    @abstractmethod
    def detect(
        self,
        entity_uid: str,
        metric_name: str,
        values: Sequence[float],
        ts: int,
        labels: dict | None = None,
    ) -> SignalRecord:
        ...


@dataclass
class ZScoreDetector(Detector):
    """Robust z-score on the most recent fraction of the window.

    score = max |z| over the recent tail; severity by threshold. Real math on
    real values — kube-verdict uses the same method as its short-signal fallback.
    """

    warning: float = 3.0
    critical: float = 4.5
    min_points: int = 8
    recent_fraction: float = 0.25

    method = "zscore"

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
        v = np.asarray(values, dtype=np.float64)
        n = int(v.size)

        if n < max(3, self.min_points):
            score = 0.0
        else:
            mu = float(v.mean())
            sigma = float(v.std())
            if sigma < 1e-8:
                score = 0.0
            else:
                z = np.abs((v - mu) / sigma)
                q = max(1, int(n * self.recent_fraction))
                score = float(z[-q:].max())

        return SignalRecord(
            entity_uid=entity_uid,
            metric_name=metric_name,
            ts=ts,
            severity=self._severity(score),
            score=round(score, 4),
            method=self.method,
            n_points=n,
            labels=dict(labels or {}),
        )
