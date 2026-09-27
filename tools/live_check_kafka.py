#!/usr/bin/env python3
"""Repeatable live check for the Kafka/Redpanda source (connector C7).

Seeds Redpanda's ``patchtst-metrics`` topic (``deploy/kafka/10-producer-seed.yaml``),
runs the Kafka-sourced detection pipeline (``deploy/kafka/20-pipeline-kafka.yaml``,
a bounded drain, not a rolling window -- see that manifest's docstring), then
queries the KB HTTP API for the signal it should have written. This is the
one path (D2, docs/ROADMAP.md) never proven against a real broker before --
unit tests mock ``KafkaSource``; this needs Redpanda actually running in the
cluster.

Usage:
    python -m tools.live_check_kafka [--namespace patchtst] [--timeout 90]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
NAMESPACE_DEFAULT = "patchtst"
ENTITY_UID = "node2/demo"  # fixed group_id the producer seeds (10-producer-seed.yaml)


def _kubectl(*args: str, namespace: str) -> str:
    cmd = ["kubectl", "-n", namespace, *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def seed_topic(namespace: str) -> None:
    print("[1/3] seeding the Redpanda topic with synthetic PivotRows...")
    _kubectl("delete", "job", "kafka-producer-seed", "--ignore-not-found", namespace=namespace)
    _kubectl(
        "apply", "-f", str(ROOT / "deploy/kafka/10-producer-seed.yaml"), namespace=namespace
    )
    _kubectl(
        "wait", "--for=condition=complete", "job/kafka-producer-seed", "--timeout=60s",
        namespace=namespace,
    )


def run_pipeline(namespace: str, timeout: float) -> None:
    print("[2/3] draining the topic through the Kafka-sourced pipeline...")
    _kubectl("delete", "job", "pipeline-kafka", "--ignore-not-found", namespace=namespace)
    _kubectl(
        "apply", "-f", str(ROOT / "deploy/kafka/20-pipeline-kafka.yaml"), namespace=namespace
    )
    _kubectl(
        "wait", "--for=condition=complete", "job/pipeline-kafka",
        f"--timeout={int(timeout)}s", namespace=namespace,
    )


def verify_signal_landed(namespace: str, since_ms: int) -> int:
    print("[3/3] verifying the signal landed in the KB (via the kb HTTP API)...")
    pod = "live-check-kafka-curl"
    _kubectl("delete", "pod", pod, "--ignore-not-found", "--force", "--grace-period=0", namespace=namespace)
    url = f"http://kb.{namespace}.svc.cluster.local/api/v1/signals/history?entity={ENTITY_UID}&since={since_ms}"
    out = _kubectl(
        "run", pod, "--rm", "-i", "--restart=Never", "--image=curlimages/curl",
        "--command", "--", "curl", "-s", url,
        namespace=namespace,
    )
    # stdout is the JSON body immediately followed by kubectl's own
    # 'pod "..." deleted' line with no separator; parse just the JSON prefix.
    payload, _ = json.JSONDecoder().raw_decode(out.strip())
    return payload["count"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--namespace", default=NAMESPACE_DEFAULT)
    parser.add_argument("--timeout", type=float, default=90.0, help="seconds to wait for the pipeline Job")
    parser.add_argument("--out", default=str(ROOT / "docs/evidence/live-checks"))
    args = parser.parse_args(argv)

    started = datetime.now(timezone.utc)
    since_ms = int(started.timestamp() * 1000) - 3600_000  # producer backdates ~1h of points
    try:
        seed_topic(args.namespace)
        run_pipeline(args.namespace, args.timeout)
        count = verify_signal_landed(args.namespace, since_ms)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    ok = count > 0
    result = {
        "checked_at": started.isoformat(),
        "target": "kafka-redpanda",
        "entity_uid": ENTITY_UID,
        "signal_count": count,
        "pass": ok,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"kafka-{started.strftime('%Y%m%dT%H%M%SZ')}.json").write_text(
        json.dumps(result, indent=2)
    )

    print(json.dumps(result, indent=2))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
