"""CLI entrypoint: serve the signal knowledge base over HTTP.

    python -m kb --root /data/kb --host 0.0.0.0 --port 8080

This is what the K3s KB Deployment runs; kube-verdict's RCA queries it via
``GET /api/v1/signals/history``. ``--root`` (or ``$KB_ROOT``) points at the
Parquet datalake the pipeline writes to.
"""
from __future__ import annotations

import argparse
import os

from .store import SignalStore


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kb", description="Serve the signal knowledge base")
    p.add_argument("--root", default=os.environ.get("KB_ROOT", "/data/kb"),
                   help="datalake root (local path or s3://...); env: KB_ROOT")
    p.add_argument("--host", default=os.environ.get("KB_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("KB_PORT", "8080")))
    args = p.parse_args(argv)

    import uvicorn  # lazy: only the server entrypoint needs it

    from . import create_app

    app = create_app(SignalStore(args.root))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
