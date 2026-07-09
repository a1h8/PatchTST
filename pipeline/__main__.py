"""CLI entrypoint: ``python -m pipeline <config.yaml|.json>``.

This is what the K3s CronJob runs each tick. ``--check`` runs the offline
pre-flight validation (version alignment, required Dataflow options, reachable
paths, streaming/window/detach) and exits without submitting — a cheap gate
before a slow, billable remote run. A normal invocation runs the same checks
first and aborts on any error.
"""
from __future__ import annotations

import argparse

from .preflight import format_report, has_errors, preflight
from .runner import load_config, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline", description="Run the detection pipeline")
    parser.add_argument("config", help="path to pipeline config (.yaml / .json)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="run offline pre-flight validation and exit (no submit)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    issues = preflight(config)

    if args.check:
        print(format_report(issues))
        return 1 if has_errors(issues) else 0

    if has_errors(issues):
        print(format_report(issues))
        print("aborting submit: fix the errors above or re-check with --check")
        return 1

    run_pipeline(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
