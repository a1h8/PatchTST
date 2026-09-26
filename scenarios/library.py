"""Synthetic time-series scenarios for h013+ (roadmap B14: "network latency,
cert expiry, etcd compaction") plus a negative control (h016).

Each scenario is a single-channel series with a known ground-truth
``incident_at`` index, designed to exercise a *specific* face of D1
(RegimeSwitchDetector): a slow, forecastable degradation should trip the
forecast/anticipation face; a break in a learned periodic/structural pattern
should trip the reconstruction/detective face. h016 is the exception —
``incident_at = n`` (never) marks a noisy-but-healthy series where the
detector must stay quiet the whole way through; see its docstring. This is
what ``tools/capture_signals.py`` measures against.

Deterministic (fixed RNG seeds) so captures are reproducible.
"""
from __future__ import annotations

import numpy as np

# name -> (values, incident_at, description)
Scenario = tuple[np.ndarray, int, str]


def h013_network_latency(n: int = 160, incident_at: int = 110, ramp_len: int = 15) -> Scenario:
    """p99 latency: flat baseline, then a slow ramp to a sustained elevated
    level (a real degrading link/route — never recovers on its own).
    Exercises: forecast/anticipation face (predictable trend breaking the
    learned flat baseline), handing off to reconstruction once the plateau
    is far out-of-distribution.
    """
    rng = np.random.default_rng(13)
    values = 20.0 + rng.normal(0.0, 1.5, n)
    for i in range(incident_at, n):
        t = min(i - incident_at, ramp_len)
        values[i] += 180.0 * (t / ramp_len)
    return values, incident_at, "p99_latency_ms — gradual network degradation, sustained after onset"


def h014_cert_renewal_stall(n: int = 160, incident_at: int = 110, period: int = 40) -> Scenario:
    """cert_days_remaining: a sawtooth (renewal resets the countdown every
    ``period`` ticks) that stalls — the expected reset stops happening and the
    countdown runs straight through zero into negative (already-expired) days.
    Exercises: reconstruction/detective face (the input breaks the learned
    periodic reset pattern, not a smooth forecastable trend).
    """
    rng = np.random.default_rng(14)
    values = np.zeros(n)
    offset_at_incident = period - (incident_at % period)
    for i in range(n):
        if i < incident_at:
            values[i] = float(period - (i % period))
        else:
            values[i] = float(offset_at_incident - (i - incident_at))
    values += rng.normal(0.0, 0.3, n)
    return values, incident_at, "cert_days_remaining — renewal stall breaks the periodic reset pattern"


def h015_etcd_compaction_stall(
    n: int = 160, incident_at: int = 110, spike_period: int = 20, spike_len: int = 3
) -> Scenario:
    """etcd request latency: periodic transient compaction spikes that fully
    recover between windows (normal) — then, from ``incident_at``, latency
    stops recovering between spikes and stays sustained-elevated.
    Exercises: reconstruction/detective face (the learned "spike then recover"
    pattern breaks; the plateau between spikes shifts).
    """
    rng = np.random.default_rng(15)
    values = 5.0 + rng.normal(0.0, 0.5, n)
    for i in range(n):
        in_spike = (i % spike_period) < spike_len
        if in_spike:
            values[i] += 40.0
        elif i >= incident_at:
            values[i] += 30.0  # sustained elevation between spikes, post-incident
    return values, incident_at, "etcd_compaction_latency_ms — post-compaction recovery stops happening"


def h016_noisy_baseline_no_incident(n: int = 160) -> Scenario:
    """pod_restart_count: noisy but genuinely healthy the whole window —
    three isolated single-tick blips (a benign restart during a routine
    rollout) that fully recover immediately, no real incident anywhere.
    Negative control, not a fourth incident shape: ``incident_at = n`` (never)
    means every evaluated tick is "pre-incident", so any tick the regime
    reads ``incident`` is a false positive — this is what
    ``enter_after``/``exit_after`` anti-flapping exists to prevent. A detector
    that is well-tuned against h013-h015 but cries wolf on ordinary noise
    is not ready to page anyone.
    """
    rng = np.random.default_rng(16)
    values = 2.0 + rng.normal(0.0, 0.4, n)
    for i in (40, 90, 130):
        values[i] += 6.0  # isolated, non-repeating, self-recovering blip
    values = np.clip(values, 0.0, None)
    return values, n, "pod_restart_count — noisy healthy baseline with benign one-off blips, never a real incident"


SCENARIOS: dict[str, Scenario] = {
    "h013_network_latency": h013_network_latency(),
    "h014_cert_renewal_stall": h014_cert_renewal_stall(),
    "h015_etcd_compaction_stall": h015_etcd_compaction_stall(),
    "h016_noisy_baseline_no_incident": h016_noisy_baseline_no_incident(),
}
