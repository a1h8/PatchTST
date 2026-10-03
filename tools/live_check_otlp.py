#!/usr/bin/env python3
"""Repeatable live check for the OTLP push source (connector C2).

Starts the OTLP-sourced detection pipeline (deploy/otlp/10-pipeline-otlp.yaml,
a bounded drain of whatever's pushed within poll_timeout_s -- see that
manifest's docstring), pushes one real OTLP/HTTP JSON export request to it
from a second in-cluster pod while it's listening, then queries the KB HTTP
API for the signal it should have written. This is the other half of D2
(docs/ROADMAP.md) never proven against a real push -- unit tests exercise
OTLPSource with a direct urllib call in the same process; this needs two
separate pods talking over a real Service, the same push protocol an actual
OTel Collector would use.

Usage:
    python -m tools.live_check_otlp [--namespace patchtst] [--timeout 90]
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
ENTITY_UID = "otlp-livecheck-demo"  # service.name the pusher sets

_PUSHER_SCRIPT = """
import json
import time
import urllib.error
import urllib.request

now_ns = int(time.time() * 1e9)
body = {
    "resourceMetrics": [
        {
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": "otlp-livecheck-demo"}}
                ]
            },
            "scopeMetrics": [
                {
                    "metrics": [
                        {
                            "name": "cpu_usage",
                            "gauge": {"dataPoints": [{"timeUnixNano": str(now_ns), "asDouble": 0.42}]},
                        }
                    ]
                }
            ],
        }
    ]
}
data = json.dumps(body).encode()
url = "http://pipeline-otlp.patchtst.svc.cluster.local:4318/v1/metrics"

# the receiver only starts listening once the Job's read() is first called;
# retry past the startup race instead of needing exact timing from outside.
deadline = time.monotonic() + 20.0
last_err = None
while time.monotonic() < deadline:
    try:
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            assert resp.status == 200
            print("PUSHER_OK")
            raise SystemExit(0)
    except (urllib.error.URLError, ConnectionRefusedError, OSError) as exc:
        last_err = exc
        time.sleep(1)
print(f"PUSHER_FAILED: {last_err}")
raise SystemExit(1)
"""


def _kubectl(*args: str, namespace: str) -> str:
    cmd = ["kubectl", "-n", namespace, *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def start_pipeline(namespace: str) -> None:
    print("[1/4] starting the OTLP-sourced pipeline (receiver binds on first read())...")
    _kubectl("delete", "job", "pipeline-otlp", "--ignore-not-found", namespace=namespace)
    _kubectl(
        "apply", "-f", str(ROOT / "deploy/otlp/10-pipeline-otlp.yaml"), namespace=namespace
    )


def push_metric(namespace: str, timeout: float) -> str:
    print("[2/4] pushing one OTLP/HTTP export request from a second pod...")
    pod = "live-check-otlp-pusher"
    _kubectl("delete", "pod", pod, "--ignore-not-found", "--force", "--grace-period=0", namespace=namespace)
    _kubectl(
        "run", pod, "--restart=Never", "--image=patchtst-pipeline:dev",
        "--image-pull-policy=IfNotPresent",
        "--command", "--", "python", "-c", _PUSHER_SCRIPT,
        namespace=namespace,
    )
    deadline = time.monotonic() + timeout
    phase = ""
    while time.monotonic() < deadline:
        phase = _kubectl(
            "get", "pod", pod, "-o", "jsonpath={.status.phase}", namespace=namespace
        ).strip()
        if phase in ("Succeeded", "Failed"):
            break
        time.sleep(2)
    logs = _kubectl("logs", pod, namespace=namespace)
    _kubectl("delete", "pod", pod, "--ignore-not-found", "--force", "--grace-period=0", namespace=namespace)
    if phase != "Succeeded" or "PUSHER_OK" not in logs:
        raise RuntimeError(f"pusher pod ended in phase={phase!r}, logs:\n{logs}")
    return logs


def wait_pipeline_done(namespace: str, timeout: float) -> None:
    print("[3/4] waiting for the pipeline Job to drain and write to the KB...")
    _kubectl(
        "wait", "--for=condition=complete", "job/pipeline-otlp",
        f"--timeout={int(timeout)}s", namespace=namespace,
    )


def verify_signal_landed(namespace: str, since_ms: int) -> int:
    print("[4/4] verifying the signal landed in the KB (via the kb HTTP API)...")
    pod = "live-check-otlp-curl"
    _kubectl("delete", "pod", pod, "--ignore-not-found", "--force", "--grace-period=0", namespace=namespace)
    url = f"http://kb.{namespace}.svc.cluster.local/api/v1/signals/history?entity={ENTITY_UID}&since={since_ms}"
    out = _kubectl(
        "run", pod, "--rm", "-i", "--restart=Never", "--image=curlimages/curl",
        "--command", "--", "curl", "-s", url,
        namespace=namespace,
    )
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
    since_ms = int(started.timestamp() * 1000) - 60_000
    try:
        start_pipeline(args.namespace)
        push_metric(args.namespace, args.timeout)
        wait_pipeline_done(args.namespace, args.timeout)
        count = verify_signal_landed(args.namespace, since_ms)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    ok = count > 0
    result = {
        "checked_at": started.isoformat(),
        "target": "otlp-push-receiver",
        "entity_uid": ENTITY_UID,
        "signal_count": count,
        "pass": ok,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"otlp-{started.strftime('%Y%m%dT%H%M%SZ')}.json").write_text(
        json.dumps(result, indent=2)
    )

    print(json.dumps(result, indent=2))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
