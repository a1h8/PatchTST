"""Real-run CLI for the Mimir source — the read side of ``connectors.ingest``.

Run one live ``query_range`` against a Mimir endpoint and print the aligned
``PivotRow``s, so the connector can be smoke-tested against a real cluster:

    python -m connectors.sources.mimir_cli \\
        --endpoint http://localhost:9009 \\
        --promql 'rate(container_cpu_usage_seconds_total{container!="POD"}[5m])' \\
        --lookback 1h --step 60 --group-by pod,namespace --tenant demo

Kept in its own module (not ``mimir``): the source registers ``@connector("mimir")``
at import, and ``python -m connectors.sources.mimir`` would re-execute that module
as ``__main__`` and register the connector twice. This module carries no connector,
so running it as ``__main__`` is safe.
"""
from __future__ import annotations

import json

from .mimir import MimirSource

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(text: str) -> int:
    """Parse a Prometheus-style duration (``90s``, ``30m``, ``1h``, ``2d``) to seconds.

    A bare integer is read as seconds. Raises ``ValueError`` on a bad unit.
    """
    text = text.strip()
    if not text:
        raise ValueError("empty duration")
    if text[-1].isdigit():
        return int(text)
    value, unit = text[:-1], text[-1].lower()
    try:
        return int(value) * _UNIT_SECONDS[unit]
    except KeyError:
        raise ValueError(
            f"unknown duration unit {unit!r} in {text!r} (use s/m/h/d)"
        ) from None


def resolve_window(
    *,
    start: int | None,
    end: int | None,
    lookback: str,
    now_s: int | None = None,
) -> tuple[int, int]:
    """Resolve the ``[start, end]`` epoch-second query window.

    Explicit ``start``/``end`` win; otherwise the window is ``[now - lookback, now]``.
    ``now_s`` is injectable for tests; it defaults to wall-clock ``time.time()``.
    """
    import time

    now = int(time.time()) if now_s is None else now_s
    resolved_end = now if end is None else end
    resolved_start = (resolved_end - parse_duration(lookback)) if start is None else start
    if resolved_start >= resolved_end:
        raise ValueError(
            f"empty query window: start={resolved_start} >= end={resolved_end}"
        )
    return resolved_start, resolved_end


def main(argv: list[str] | None = None) -> int:
    """Run one real ``query_range`` and report the rows (summary or ``--json``)."""
    import argparse

    p = argparse.ArgumentParser(
        prog="connectors.sources.mimir_cli",
        description="Run a real Mimir query_range and print aligned pivot rows",
    )
    p.add_argument("--endpoint", required=True, help="e.g. http://localhost:9009")
    p.add_argument("--promql", required=True, help="range query to run")
    p.add_argument("--lookback", default="1h",
                   help="window ending now, e.g. 90s/30m/1h/2d (default 1h)")
    p.add_argument("--start", type=int, default=None, help="epoch seconds (overrides lookback)")
    p.add_argument("--end", type=int, default=None, help="epoch seconds (default now)")
    p.add_argument("--step", type=int, default=60, help="query resolution, seconds")
    p.add_argument("--group-by", default="instance",
                   help="comma-separated labels forming the group_id / entity")
    p.add_argument("--channel-label", default="__name__",
                   help="label naming each channel within a group")
    p.add_argument("--fill", default="ffill", help="alignment fill policy")
    p.add_argument("--tenant", default=None, help="X-Scope-OrgID (Mimir tenant)")
    p.add_argument("--limit", type=int, default=10, help="rows to print (<=0 for all)")
    p.add_argument("--json", action="store_true", help="emit NDJSON rows, not a summary")
    args = p.parse_args(argv)

    start, end = resolve_window(start=args.start, end=args.end, lookback=args.lookback)
    source = MimirSource(
        args.endpoint,
        args.promql,
        start,
        end,
        step_s=args.step,
        group_by=[s for s in args.group_by.split(",") if s],
        channel_label=args.channel_label,
        fill=args.fill,
        tenant=args.tenant,
    )
    rows = source.read()

    if args.json:
        for row in rows:
            print(json.dumps({
                "group_id": row.group_id,
                "ts": row.ts,
                "channels": list(row.channels),
                "values": list(row.values),
                "labels": row.labels,
            }))
        return 0

    groups = sorted({r.group_id for r in rows})
    channels = sorted({c for r in rows for c in r.channels})
    span = (f"{min(r.ts for r in rows)}..{max(r.ts for r in rows)} ms" if rows else "-")
    print(
        f"queried {args.endpoint} [{start}..{end} step {args.step}s]: "
        f"{len(rows)} rows, {len(groups)} groups, {len(channels)} channels, span {span}"
    )
    shown = rows if args.limit <= 0 else rows[: args.limit]
    for row in shown:
        vals = ",".join(f"{v:.4g}" for v in row.values)
        print(f"  {row.group_id} @{row.ts}  {','.join(row.channels)}={vals}")
    if 0 < args.limit < len(rows):
        print(f"  ... ({len(rows) - args.limit} more)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())