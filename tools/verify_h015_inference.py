"""Empirical check (SIGNAL_VALIDATION.md follow-up): does a frozen, checkpoint-
backed inference detector catch etcd_compaction_stall (h015) where the
train-on-the-fly PatchTSTDetector/ReconstructionDetector pair does not?

Trains small forecast+reconstruction checkpoints ONLY on a long, purely-normal
version of the scenario's periodic spike pattern (never sees the incident),
then runs the frozen ForecastInferenceDetector/ReconstructionInferenceDetector
(wrapped in RegimeSwitchDetector) against the real test series, tick by tick,
the same way tools/capture_signals.py ticks the trained detectors.

Not wired into capture_signals.py's regular run: this trains a
scenario-specific checkpoint from scratch (~1-2 min on CPU) rather than
reusing a shared one, so it stays a manual, on-demand verification for now.

Usage: python -m tools.verify_h015_inference
"""
from __future__ import annotations

import logging

import numpy as np

from detection import ForecastInferenceDetector, RegimeSwitchDetector, ReconstructionInferenceDetector
from inference.config import ModelSpec
from inference.train_reference import _standardize, finetune, pretrain
from scenarios.library import etcd_compaction_stall

log = logging.getLogger(__name__)


class _Args:
    lr = 1e-3
    pretrain_epochs = 15
    finetune_epochs = 15
    batch_size = 32
    mask_ratio = 0.4
    win_stride = 2
    dset = "h015_normal_only"


def _gen_long_normal(n: int = 2000, spike_period: int = 20, spike_len: int = 3) -> np.ndarray:
    """A long run of etcd_compaction_stall's *pre-incident* pattern only —
    periodic transient spikes that always recover, no sustained shift ever."""
    rng = np.random.default_rng(99)
    values = 5.0 + rng.normal(0.0, 0.5, n)
    for i in range(n):
        if (i % spike_period) < spike_len:
            values[i] += 40.0
    return values.astype(np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    spec = ModelSpec(
        c_in=1, context_length=32, target_length=8,
        patch_len=8, stride=8, n_layers=1, d_model=16, n_heads=2, d_ff=32,
    )
    device = "cpu"
    args = _Args()

    train = _standardize(_gen_long_normal().reshape(-1, 1), n_points := 2000)
    print("=== training on long normal-only pattern (never sees the incident) ===")
    pt_path = pretrain(spec, train, args, device)
    ft_path = finetune(spec, train, pt_path, args, device)

    forecast = ForecastInferenceDetector(forecast_ckpt=ft_path, reconstruct_ckpt=pt_path, spec=spec, device=device)
    detective = ReconstructionInferenceDetector(forecast_ckpt=ft_path, reconstruct_ckpt=pt_path, spec=spec, device=device)
    detector = RegimeSwitchDetector(forecast=forecast, detective=detective, enter_after=2, exit_after=2)

    values, incident_at, desc = etcd_compaction_stall()
    print(f"\n=== running frozen checkpoint against real h015 series (incident_at={incident_at}) ===")
    first_incident = None
    false_positives = []
    for tick in range(40, len(values) + 1, 4):
        sig = detector.detect("scn", "value", values[:tick].tolist(), ts=tick * 60000)
        regime = sig.labels.get("regime")
        if tick < incident_at and regime == "incident":
            false_positives.append(tick)
        if tick >= incident_at and first_incident is None and regime == "incident":
            first_incident = tick
        print(tick, sig.severity, round(sig.score, 3), regime, sig.labels.get("mode"), sig.method)

    latency = (first_incident - incident_at) if first_incident is not None else None
    print(f"\nfirst_incident_tick={first_incident}  latency={latency}  false_positives={false_positives}")


if __name__ == "__main__":
    main()
