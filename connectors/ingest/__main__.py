"""CLI: push a wide CSV into Mimir/Prometheus via remote-write.

    python -m connectors.ingest --csv dataset/ETT-small/ETTh1.csv \
        --endpoint http://localhost:9009 --instance etth1 --tenant demo --end-at-now

    python -m connectors.ingest --csv data.csv --endpoint x --instance i --dry-run
"""
from __future__ import annotations

import argparse
import logging

from .csv_source import csv_to_timeseries
from .remote_write import RemoteWriter

log = logging.getLogger(__name__)


def _parse_labels(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        k, _, v = item.partition("=")
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="CSV -> Mimir/Prometheus remote-write")
    p.add_argument("--csv", required=True)
    p.add_argument("--endpoint", required=True, help="e.g. http://localhost:9009")
    p.add_argument("--instance", required=True, help="instance label value")
    p.add_argument("--tenant", default=None, help="X-Scope-OrgID (Mimir tenant)")
    p.add_argument("--timestamp-column", default=None)
    p.add_argument("--metric-prefix", default="")
    p.add_argument("--time-format", default="%Y-%m-%d %H:%M:%S")
    p.add_argument("--label", action="append", help="extra label k=v (repeatable)")
    p.add_argument("--end-at-now", action="store_true",
                   help="shift series so the latest sample is ~now")
    p.add_argument("--dry-run", action="store_true",
                   help="encode only; no compression, no network")
    args = p.parse_args(argv)

    series = csv_to_timeseries(
        args.csv,
        instance=args.instance,
        timestamp_column=args.timestamp_column,
        metric_prefix=args.metric_prefix,
        extra_labels=_parse_labels(args.label),
        time_format=args.time_format,
        end_at_now=args.end_at_now,
    )
    writer = RemoteWriter(args.endpoint, tenant=args.tenant)
    sent = writer.push(series, dry_run=args.dry_run)
    verb = "encoded (dry-run)" if args.dry_run else f"pushed to {args.endpoint}"
    log.info("%s: %d series, %d samples", verb, len(series), sent)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
