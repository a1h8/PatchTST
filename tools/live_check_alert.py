#!/usr/bin/env python3
"""Repeatable live check for KubeVerdictAlertSink against a real socket.

tests/test_kubeverdict_alert.py and tests/test_kubeverdict_contract.py mock
``post_alerts``/``urlopen`` -- solid for payload shape and severity filtering,
but neither proves a POST actually crosses a real network boundary and lands.
No real kube-verdict is available here, so this stands up a minimal mock
webhook in-cluster (deploy/mock-kubeverdict, stdlib-only, records what it
receives) and checks two things against it:

  1. anomalous signals (>= min_severity) actually arrive, with labels intact.
  2. a genuinely unreachable endpoint (bad DNS, not a mocked exception) does
     not crash the pusher -- kb/alert.py's best-effort promise, proven live.

Usage:
    python -m tools.live_check_alert [--namespace patchtst] [--timeout 90]
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

_PUSHER_SCRIPT = """
import sys
from kb.alert import KubeVerdictAlertSink
from kb.signal import SignalRecord

records = [
    SignalRecord(entity_uid="node3/demo", metric_name="sim_cpu", ts=1_000,
                 severity="normal", score=0.1, method="zscore",
                 labels={"namespace": "demo", "pod": "node3"}),
    SignalRecord(entity_uid="node3/demo", metric_name="sim_cpu", ts=1_001,
                 severity="warning", score=3.5, method="zscore",
                 labels={"namespace": "demo", "pod": "node3"}),
    SignalRecord(entity_uid="node3/demo", metric_name="sim_mem", ts=1_002,
                 severity="critical", score=9.9, method="zscore",
                 labels={"namespace": "demo", "pod": "node3"}),
]

good = KubeVerdictAlertSink(
    "http://mock-kubeverdict.patchtst.svc.cluster.local:8080/api/v1/webhook/signal"
)
good.write(records)

# best-effort against a genuinely unreachable endpoint (real DNS failure, not
# a mocked exception) -- must not raise with the default raise_on_error=False.
bad = KubeVerdictAlertSink(
    "http://mock-kubeverdict-does-not-exist.patchtst.svc.cluster.local:8080/api/v1/webhook/signal",
    timeout=3.0,
)
bad.write(records)
print("PUSHER_OK")
"""


def _kubectl(*args: str, namespace: str) -> str:
    cmd = ["kubectl", "-n", namespace, *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def deploy_mock_server(namespace: str, timeout: float) -> None:
    print("[1/3] deploying the mock kube-verdict webhook...")
    _kubectl("apply", "-k", str(ROOT / "deploy/mock-kubeverdict"), namespace=namespace)
    # restart so each run starts from a clean in-memory `received` list
    _kubectl("rollout", "restart", "deployment/mock-kubeverdict", namespace=namespace)
    _kubectl(
        "rollout", "status", "deployment/mock-kubeverdict",
        f"--timeout={int(timeout)}s", namespace=namespace,
    )


def push_alerts(namespace: str, timeout: float) -> str:
    # `kubectl run --rm -i` races a fast-completing pod's own exit (observed
    # dropping captured output); create + poll for phase + logs + delete
    # instead, the pattern already proven reliable this session.
    print("[2/3] pushing signals -- one real endpoint, one deliberately broken...")
    pod = "live-check-alert-pusher"
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
    if phase != "Succeeded":
        raise RuntimeError(f"pusher pod ended in phase={phase!r}, logs:\n{logs}")
    return logs


def fetch_received(namespace: str) -> list[dict]:
    print("[3/3] fetching what the mock webhook actually received...")
    pod = "live-check-alert-curl"
    _kubectl("delete", "pod", pod, "--ignore-not-found", "--force", "--grace-period=0", namespace=namespace)
    out = _kubectl(
        "run", pod, "--rm", "-i", "--restart=Never", "--image=curlimages/curl",
        "--command", "--", "curl", "-s",
        "http://mock-kubeverdict.patchtst.svc.cluster.local:8080/received",
        namespace=namespace,
    )
    payload, _ = json.JSONDecoder().raw_decode(out.strip())
    return payload["alerts"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--namespace", default=NAMESPACE_DEFAULT)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--out", default=str(ROOT / "docs/evidence/live-checks"))
    args = parser.parse_args(argv)

    started = datetime.now(timezone.utc)
    try:
        deploy_mock_server(args.namespace, args.timeout)
        pusher_out = push_alerts(args.namespace, args.timeout)
        pusher_ok = "PUSHER_OK" in pusher_out
        received = fetch_received(args.namespace)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    labels_intact = any(
        a.get("labels", {}).get("namespace") == "demo"
        and a.get("labels", {}).get("pod") == "node3"
        for a in received
    )
    # default min_severity="warning" -> exactly the warning+critical record,
    # the normal one never gets pushed at all.
    ok = pusher_ok and len(received) == 2 and labels_intact
    result = {
        "checked_at": started.isoformat(),
        "target": "kubeverdict-alert-webhook",
        "pusher_survived_bad_endpoint": pusher_ok,
        "received_count": len(received),
        "labels_intact": labels_intact,
        "pass": ok,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"alert-{started.strftime('%Y%m%dT%H%M%SZ')}.json").write_text(
        json.dumps(result, indent=2)
    )

    print(json.dumps(result, indent=2))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
