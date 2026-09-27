#!/usr/bin/env python3
"""Repeatable live check for the Flink-on-K8s deployment target (M6).

Automates the manual sequence used to first prove this path live
(docs/evidence/flink-live.md, 2026-09-26): reseed synthetic Mimir data,
submit the pipeline to the real Flink cluster, poll the Flink job to
completion, and confirm a fresh signal landed in the KB. Needs a live
cluster with deploy/k3s + deploy/flink already applied (kubectl pointed at
it) -- this is not a unit test, it is the thing unit tests with fixtures
cannot prove.

Usage:
    python -m tools.live_check_flink [--namespace patchtst] [--timeout 90]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
NAMESPACE_DEFAULT = "patchtst"


def _kubectl(*args: str, namespace: str) -> str:
    cmd = ["kubectl", "-n", namespace, *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _flink_jobs(namespace: str) -> list[dict]:
    out = _kubectl(
        "exec", "deploy/flink-jobmanager", "--",
        "curl", "-s", "localhost:8081/jobs/overview",
        namespace=namespace,
    )
    return json.loads(out)["jobs"]


def seed_data(namespace: str) -> None:
    print("[1/4] seeding fresh sim_* metrics into Mimir...")
    _kubectl("delete", "job", "ingest-seed", "--ignore-not-found", namespace=namespace)
    _kubectl("apply", "-f", str(ROOT / "deploy/k3s/40-ingest.yaml"), namespace=namespace)
    _kubectl(
        "wait", "--for=condition=complete", "job/ingest-seed", "--timeout=60s",
        namespace=namespace,
    )


def submit(namespace: str) -> set[str]:
    print("[2/4] submitting the pipeline to the Flink cluster...")
    before = {j["jid"] for j in _flink_jobs(namespace)}
    _kubectl("delete", "job", "flink-submit-mimir", "--ignore-not-found", namespace=namespace)
    _kubectl(
        "apply", "-f", str(ROOT / "deploy/flink/50-submit-mimir.example.yaml"),
        namespace=namespace,
    )
    _kubectl(
        "wait", "--for=condition=complete", "job/flink-submit-mimir", "--timeout=60s",
        namespace=namespace,
    )
    return before


def wait_for_flink_job(namespace: str, before: set[str], timeout: float) -> dict:
    print("[3/4] waiting for the Flink job to finish...")
    deadline = time.monotonic() + timeout
    latest_seen: dict | None = None
    while time.monotonic() < deadline:
        candidates = [j for j in _flink_jobs(namespace) if j["jid"] not in before]
        if candidates:
            latest_seen = max(candidates, key=lambda j: j["start-time"])
            if latest_seen["state"] in ("FINISHED", "FAILED", "CANCELED"):
                return latest_seen
        time.sleep(3)
    raise TimeoutError(f"Flink job did not finish within {timeout}s (last seen: {latest_seen})")


def verify_signal_landed(namespace: str, timeout: float) -> int:
    print("[4/4] verifying a fresh signal landed in the KB...")
    minutes = max(1, int(timeout / 60) + 1)
    out = _kubectl(
        "exec", "deploy/mimir-minio", "--",
        "sh", "-c",
        "mc alias set local http://localhost:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD "
        f">/dev/null 2>&1; mc find local/patchtst-kb/kb-flink --newer-than {minutes}m | wc -l",
        namespace=namespace,
    )
    return int(out.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--namespace", default=NAMESPACE_DEFAULT)
    parser.add_argument("--timeout", type=float, default=90.0, help="seconds to wait for the Flink job")
    parser.add_argument("--out", default=str(ROOT / "docs/evidence/live-checks"))
    args = parser.parse_args(argv)

    started = datetime.now(timezone.utc)
    try:
        seed_data(args.namespace)
        before = submit(args.namespace)
        job = wait_for_flink_job(args.namespace, before, args.timeout)
        n_files = verify_signal_landed(args.namespace, args.timeout)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    ok = job["state"] == "FINISHED" and n_files > 0
    result = {
        "checked_at": started.isoformat(),
        "target": "flink-on-k8s",
        "flink_job_id": job["jid"],
        "flink_job_state": job["state"],
        "flink_job_duration_ms": job["duration"],
        "new_signal_files": n_files,
        "pass": ok,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"flink-{started.strftime('%Y%m%dT%H%M%SZ')}.json").write_text(
        json.dumps(result, indent=2)
    )

    print(json.dumps(result, indent=2))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
