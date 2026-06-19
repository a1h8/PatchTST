"""Threshold policies — turn a detector score into a severity (roadmap M4).

A fixed ``warning``/``critical`` pair can't fit every metric: a noisy p99 latency
and a smooth error-rate have very different "normal" score spreads, so one global
cut yields false positives on the noisy one and false negatives on the smooth one.
An **adaptive** policy learns each key's own spread from a rolling window of recent
scores and sets the cut from that.

Policies:

  * :class:`FixedThreshold` — the original static cut (default; behaviour-preserving).
  * :class:`RollingQuantileThreshold` — ``warning`` = ``q(warn_q)``, ``critical`` =
    ``q(crit_q)`` of the recent scores.
  * :class:`MADThreshold` — ``median + k·(1.4826·MAD)``; robust to outliers (a single
    spike in the history barely moves the cut, unlike mean+k·σ).

All policies are **stateful per key** ``(entity, metric)`` and judge the incoming
score against history *before* appending it, so a window is never compared to
itself. A sustained shift is eventually absorbed into the baseline (the rolling
window forgets) — by design: this flags the *transition*; the regime state machine
(``RegimeSwitchDetector``) carries the sustained-incident case. Until a key has
``min_history`` samples the policy falls back to the fixed cut (cold start).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


class ThresholdPolicy(ABC):
    """Map a score to ``'normal' | 'warning' | 'critical'`` for a key."""

    @abstractmethod
    def classify(self, key: tuple[str, str], score: float) -> str:
        ...


@dataclass
class FixedThreshold(ThresholdPolicy):
    """Static cut — the original behaviour."""

    warning: float = 1.8
    critical: float = 3.0

    def classify(self, key: tuple[str, str], score: float) -> str:
        if score >= self.critical:
            return "critical"
        if score >= self.warning:
            return "warning"
        return "normal"


@dataclass
class _RollingPolicy(ThresholdPolicy):
    """Per-key rolling-window base: cold-start fixed cut, then adaptive cuts."""

    window: int = 64
    min_history: int = 16
    warning: float = 1.8      # cold-start fallback cut
    critical: float = 3.0

    def __post_init__(self) -> None:
        self._hist: dict[tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

    def classify(self, key: tuple[str, str], score: float) -> str:
        hist = self._hist[key]
        if len(hist) < self.min_history:
            warn, crit = self.warning, self.critical
        else:
            warn, crit = self._cuts(np.fromiter(hist, dtype=float))

        # strict '>': a score *within* the observed spread stays normal, so a
        # flat history (collapsed cuts) doesn't flag an equal score.
        sev = "critical" if score > crit else "warning" if score > warn else "normal"
        hist.append(score)
        return sev

    def _cuts(self, arr: np.ndarray) -> tuple[float, float]:
        raise NotImplementedError


@dataclass
class RollingQuantileThreshold(_RollingPolicy):
    """Cuts at the ``warn_q`` / ``crit_q`` quantiles of the recent scores."""

    warn_q: float = 0.95
    crit_q: float = 0.99

    def _cuts(self, arr: np.ndarray) -> tuple[float, float]:
        return float(np.quantile(arr, self.warn_q)), float(np.quantile(arr, self.crit_q))


@dataclass
class MADThreshold(_RollingPolicy):
    """Cuts at ``median + k·scale`` with ``scale = 1.4826·MAD`` (robust)."""

    k_warn: float = 3.0
    k_crit: float = 5.0

    def _cuts(self, arr: np.ndarray) -> tuple[float, float]:
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        scale = 1.4826 * mad or 1e-8          # ~std for normal data; guard flat history
        return med + self.k_warn * scale, med + self.k_crit * scale


_POLICIES: dict[str, type[ThresholdPolicy]] = {
    "fixed": FixedThreshold,
    "rolling-quantile": RollingQuantileThreshold,
    "mad": MADThreshold,
}


def build_threshold(cfg: dict) -> ThresholdPolicy:
    """Build a ThresholdPolicy from config: ``{type: mad, params: {...}}``."""
    kind = cfg["type"]
    try:
        cls = _POLICIES[kind]
    except KeyError:
        raise KeyError(
            f"unknown threshold policy {kind!r}; available: {sorted(_POLICIES)}"
        ) from None
    return cls(**cfg.get("params", {}))
