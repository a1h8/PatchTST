"""Level-shift check — the signal both PatchTST faces are blind to.

The forecast and reconstruction faces both normalise the level away (global
z-normalisation plus the model's per-window ``scaling="std"``), and a forecaster
re-trained on a window that already contains a sustained shift simply learns the
new level. So a persistent plateau — e.g. latency that stops recovering between
spikes — is only visible at the instant it starts, and a coarse tick spacing can
step over that instant entirely.

This compares the *recent* median against the *baseline* median (the earlier part
of the same window) in robust units (MAD), which spikes and ordinary noise barely
move. It is a second opinion feeding the regime switch, not a replacement for
either face.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Below this many points there is no meaningful baseline to compare against.
MIN_POINTS = 32


def level_shift_score(
    values: Sequence[float],
    *,
    recent: int = 8,
    baseline_frac: float = 0.6,
    min_points: int = MIN_POINTS,
) -> float | None:
    """Robust z-score of ``median(last recent points)`` vs the baseline median.

    Returns ``None`` when the series is too short to judge. The baseline is the
    first ``baseline_frac`` of the window; its spread is the MAD (scaled to a
    standard deviation), floored at 5% of the baseline level so a near-constant
    baseline does not turn a negligible change into an enormous score.
    """
    v = np.asarray(values, dtype=float)
    if len(v) < min_points:
        return None
    base = v[: max(1, int(len(v) * baseline_frac))]
    median = float(np.median(base))
    mad = 1.4826 * float(np.median(np.abs(base - median)))
    scale = max(mad, 0.05 * abs(median), 1e-9)
    return abs(float(np.median(v[-recent:])) - median) / scale
