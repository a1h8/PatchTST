#!/usr/bin/env python3
"""Local, offline signal-capture harness — validates the real PatchTST
detection path (RegimeSwitchDetector: forecast + reconstruction faces, not the
zscore fallback) against the h013+ synthetic scenarios, before committing any
cloud spend on a live deployment.

No network, no cloud, no Mimir: pure Python + local torch training on
synthetic series (scenarios/library.py). Ticks the detector forward over
growing windows (mirroring how a periodic CronJob tick sees a growing/rolling
window), and measures, per scenario:

  - detection_latency_ticks — how many ticks after the true incident onset the
    regime actually flips to "incident" (None if it never does — a miss)
  - false_positive_ticks    — any tick before the true onset where the regime
    is already "incident"

Results are frozen as versioned JSON under docs/evidence/signal-captures/ —
the same "captured artifact, not a claim" discipline kube-verdict's B13 uses.

Usage:
    python -m tools.capture_signals [--epochs N] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from detection import PatchTSTDetector, ReconstructionDetector, RegimeSwitchDetector
from scenarios.library import SCENARIOS

ROOT = Path(__file__).parent.parent


def _checkpoints(n_points: int, min_len: int, step: int) -> list[int]:
    if n_points < min_len:
        return [n_points]
    return list(range(min_len, n_points + 1, step))


def run_scenario(name: str, values, incident_at: int, description: str, *, epochs: int, step: int) -> dict:
    forecast = PatchTSTDetector(epochs=epochs, d_model=16, num_layers=1)
    detective = ReconstructionDetector(epochs=epochs, d_model=16, num_layers=1)
    detector = RegimeSwitchDetector(forecast=forecast, detective=detective, enter_after=2, exit_after=2)

    min_len = forecast.context_length + forecast.prediction_length
    ticks = _checkpoints(len(values), min_len, step)

    timeline: list[dict] = []
    first_incident_tick: int | None = None
    t0 = time.monotonic()
    for tick in ticks:
        sig = detector.detect(
            f"scenario/{name}", "value", values[:tick].tolist(), ts=tick * 60_000
        )
        entry = {
            "tick": tick,
            "severity": sig.severity,
            "score": round(sig.score, 3),
            "regime": sig.labels.get("regime"),
            "mode": sig.labels.get("mode"),
            "method": sig.method,
        }
        timeline.append(entry)
        if entry["regime"] == "incident" and first_incident_tick is None:
            first_incident_tick = tick
    elapsed_s = round(time.monotonic() - t0, 1)

    false_positive_ticks = [e["tick"] for e in timeline if e["tick"] < incident_at and e["regime"] == "incident"]
    detection_latency_ticks = (first_incident_tick - incident_at) if first_incident_tick is not None else None

    return {
        "scenario": name,
        "description": description,
        "n_points": len(values),
        "incident_at": incident_at,
        "detector_epochs": epochs,
        "ticks_evaluated": ticks,
        "timeline": timeline,
        "detected": first_incident_tick is not None,
        "first_incident_tick": first_incident_tick,
        "detection_latency_ticks": detection_latency_ticks,
        "false_positive_ticks": false_positive_ticks,
        "elapsed_s": elapsed_s,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="training epochs per tick (reduced from the prod default of 30 for harness speed)",
    )
    parser.add_argument("--step", type=int, default=8, help="tick spacing (points between checkpoints)")
    parser.add_argument("--out", default=str(ROOT / "docs/evidence/signal-captures"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for name, (values, incident_at, description) in SCENARIOS.items():
        print(f"[{name}] running ({len(values)} points, epochs={args.epochs})...")
        result = run_scenario(name, values, incident_at, description, epochs=args.epochs, step=args.step)
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
        status = "DETECTED" if result["detected"] else "MISSED"
        fp = len(result["false_positive_ticks"])
        print(
            f"  {status} — latency={result['detection_latency_ticks']} ticks, "
            f"false_positives={fp}, {result['elapsed_s']}s"
        )
        summary.append({
            k: result[k]
            for k in ("scenario", "detected", "detection_latency_ticks", "false_positive_ticks", "elapsed_s")
        })

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(SCENARIOS)} capture(s) + summary.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
