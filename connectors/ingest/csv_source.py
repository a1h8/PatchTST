"""CSV → Prometheus ``TimeSeries`` for remote-write ingestion.

Wide CSV layout (one timestamp column + one column per metric), e.g. ETTh1:

    date,HUFL,HULL,...,OT
    2016-07-01 00:00:00,5.827,2.009,...,30.531

Each metric column becomes a ``TimeSeries`` labelled ``__name__=<prefix><col>``
plus an ``instance`` (and any extra labels). Timestamps are parsed as epoch
seconds/ms or a datetime format. ``end_at_now`` shifts the whole series so its
last point lands ~now — handy when a real Mimir rejects samples older than its
retention/out-of-order window.
"""
from __future__ import annotations

import csv as _csv
import time
from datetime import datetime, timezone

from .remote_write import TimeSeries


def _parse_ts(raw: str, fmt: str) -> int:
    """Parse a timestamp cell to epoch milliseconds."""
    raw = raw.strip()
    try:
        n = float(raw)
        # Heuristic: >1e12 is already ms, else seconds.
        return int(n if n > 1e12 else n * 1000)
    except ValueError:
        dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)


def csv_to_timeseries(
    path: str,
    *,
    instance: str,
    timestamp_column: str | None = None,
    metric_prefix: str = "",
    extra_labels: dict[str, str] | None = None,
    time_format: str = "%Y-%m-%d %H:%M:%S",
    end_at_now: bool = False,
) -> list[TimeSeries]:
    """Read a wide CSV into one ``TimeSeries`` per metric column."""
    with open(path, newline="") as f:
        reader = _csv.reader(f)
        header = next(reader)
        ts_idx = header.index(timestamp_column) if timestamp_column else 0
        metric_cols = [i for i in range(len(header)) if i != ts_idx]

        base = {"instance": instance, **(extra_labels or {})}
        series = {
            i: TimeSeries(labels={"__name__": f"{metric_prefix}{header[i]}", **base})
            for i in metric_cols
        }
        for row in reader:
            if not row:
                continue
            ts_ms = _parse_ts(row[ts_idx], time_format)
            for i in metric_cols:
                cell = row[i].strip()
                if cell == "":
                    continue
                series[i].samples.append((ts_ms, float(cell)))

    out = [s for s in series.values() if s.samples]
    if end_at_now:
        _shift_to_now(out)
    return out


def _shift_to_now(series: list[TimeSeries]) -> None:
    """Shift every series in place so the latest sample lands ~now."""
    last = max((s.samples[-1][0] for s in series), default=0)
    if not last:
        return
    delta = int(time.time() * 1000) - last
    for s in series:
        s.samples = [(ts + delta, v) for ts, v in s.samples]
