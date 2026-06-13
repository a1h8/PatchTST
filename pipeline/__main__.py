"""CLI entrypoint: ``python -m pipeline <config.yaml|.json>``.

This is what the K3s CronJob runs each tick.
"""
from __future__ import annotations

import argparse

from .runner import load_config, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline", description="Run the detection pipeline")
    parser.add_argument("config", help="path to pipeline config (.yaml / .json)")
    args = parser.parse_args(argv)

    run_pipeline(load_config(args.config))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
