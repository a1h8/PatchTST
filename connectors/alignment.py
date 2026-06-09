"""Temporal alignment — where the multivariate 'what doesn't match' is paid.

K8s channels arrive on different cadences (CPU @15s, custom @60s, remote-write
with jitter). To build the [n_channels, context_len] window PatchTST expects,
every channel of a group must sit on a common time grid. This module turns
per-channel timestamped points into aligned multivariate ``PivotRow`` records.

This is the alignment cost that decision D4 (native multivariate) moves into the
source connector. It is pure-Python and unit-testable without Beam.
"""
from __future__ import annotations

from typing import Iterable, Literal

from .pivot import PivotRow

FillPolicy = Literal["ffill", "zero", "drop"]


def _snap(ts: int, step_ms: int) -> int:
    """Floor a timestamp onto the grid of width ``step_ms``."""
    return (ts // step_ms) * step_ms


def align_group(
    group_id: str,
    series: dict[str, Iterable[tuple[int, float]]],
    *,
    step_ms: int,
    fill: FillPolicy = "ffill",
    labels: dict[str, str] | None = None,
) -> list[PivotRow]:
    """Align one group's channels onto a common grid.

    Args:
        group_id: identity shared by all channels in this group.
        series: channel name -> iterable of (epoch_ms, value), any cadence.
        step_ms: grid resolution; timestamps are floored to this step.
        fill: gap policy when a channel has no sample in a grid bucket.
            ``ffill`` carries the last known value forward, ``zero`` uses 0.0,
            ``drop`` omits any grid point not covered by every channel.
        labels: optional metadata copied onto each row.

    Returns:
        Aligned rows, ascending by ts. ``channels`` order is the sorted channel
        names (stable for a given group), so output positions are deterministic.
    """
    channels = sorted(series)
    if not channels:
        return []

    # bucket -> {channel: last value in/under that bucket}
    bucketed: dict[str, dict[int, float]] = {}
    grid: set[int] = set()
    for ch in channels:
        last_per_bucket: dict[int, float] = {}
        for ts, val in sorted(series[ch]):
            last_per_bucket[_snap(ts, step_ms)] = float(val)
        bucketed[ch] = last_per_bucket
        grid.update(last_per_bucket)

    rows: list[PivotRow] = []
    carried: dict[str, float | None] = {ch: None for ch in channels}
    for g in sorted(grid):
        values: list[float] = []
        complete = True
        for ch in channels:
            if g in bucketed[ch]:
                carried[ch] = bucketed[ch][g]
            if carried[ch] is None:
                # no value seen yet for this channel
                if fill == "zero":
                    values.append(0.0)
                else:  # ffill or drop: cannot fill from nothing
                    complete = False
                    break
            else:
                values.append(carried[ch])
        if not complete:
            if fill == "drop":
                continue
            # ffill with no prior value at grid start: skip until covered
            continue
        rows.append(
            PivotRow(
                group_id=group_id,
                ts=g,
                values=tuple(values),
                channels=tuple(channels),
                labels=dict(labels or {}),
            )
        )
    return rows