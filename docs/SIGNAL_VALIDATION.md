# Signal validation strategy — before cloud spend

The gap this closes: PatchTST's forecast/reconstruction detectors were wired
into the pipeline (M1–M4) and pushed to kube-verdict (M7,
`kubeverdict-alert` → `POST /api/v1/webhook/signal`), but never measured
against a known scenario. "It's wired" is not "it detects." Before committing
budget to a live deployment (Flink-on-K8s + object storage on a chosen
provider — see the sovereignty note in [ARCHITECTURE.md](./ARCHITECTURE.md)),
this validates the detection logic itself, entirely offline.

## Why offline first

`tools/capture_signals.py` runs the real `RegimeSwitchDetector`
(`PatchTSTDetector` forecast face + `ReconstructionDetector` detective face —
not the z-score fallback) against synthetic scenarios, locally, with no
network, no Mimir, no cloud runner. It costs nothing and answers the question
a live run cannot answer cheaply: does the detection logic actually catch
these incident shapes, and how fast? Only once these captures clear the gate
below does a live run add anything — spending on infra to validate logic that
hasn't been checked offline would be paying to discover offline-catchable
bugs.

## Scenarios

Named for the signal shape each one is, not for any consumer's incident
catalog or numbering — this suite validates PatchTST's own detection logic on
its own terms (the decoupling principle in
[ARCHITECTURE.md](./ARCHITECTURE.md) applies to tests the same way it applies
to the runtime coupling with kube-verdict). Each scenario (`scenarios/library.py`)
is a synthetic series with a known ground-truth incident onset (`incident_at`),
chosen to exercise a *specific* face of the regime-switch detector — not just
"some anomaly somewhere":

| Scenario | Shape | Exercises |
|---|---|---|
| **network_latency** | Flat baseline → slow ramp to a sustained elevated level (a real degrading link, never self-recovers) | Forecast/anticipation face: a predictable trend breaking a learned flat baseline |
| **cert_renewal_stall** | Periodic sawtooth (renewal resets the countdown) → the reset stops happening, countdown runs through zero | Reconstruction/detective face: a break in a learned *periodic* pattern, not a smooth trend |
| **etcd_compaction_stall** | Periodic transient latency spikes that fully recover → spikes stop recovering, sustained elevation between them | Reconstruction/detective face: the learned "spike then recover" pattern breaks |
| **noisy_baseline_no_incident** | Noisy-but-healthy series with isolated single-tick blips that self-recover — `incident_at` never happens | Negative control: a detector well-tuned against the other three that still cries wolf on ordinary noise is not ready to page anyone |

`noisy_baseline_no_incident` is not a fourth incident shape — it's the
false-positive check the other three don't cover on their own (they only
measure false positives *before* their own onset, never across a series with
no onset at all).

These deliberately are not point-in-time state (CrashLoopBackOff,
ImagePullBackOff, the kind of fixture a status-only check validates against)
— they're time-series patterns, the shape a temporal detector earns its place
catching something a status-only check cannot see.

## What is measured

Per scenario, ticking the detector forward over growing windows (mirroring a
periodic CronJob tick against an accumulating/rolling window):

- **`detection_latency_ticks`** — ticks between the true `incident_at` and the
  regime actually flipping to `incident` (anti-flapping debounce included:
  `enter_after=2` consecutive critical forecasts). `null` = missed entirely.
- **`false_positive_ticks`** — any tick *before* `incident_at` where the
  regime already reads `incident`.
- **`elapsed_s`** — wall-clock cost of the capture (CPU-only, reduced
  capacity — `d_model=16`, `epochs=10` vs. the prod default `d_model=32`,
  `epochs=30` — this harness measures detection *logic*, not production
  training time).

Results are frozen as versioned JSON under `docs/evidence/signal-captures/`,
one file per scenario plus `summary.json` — a captured artifact, not a
narrated claim, the same discipline kube-verdict's B13 uses for
`tests/golden/real_00N.json`.

## Go / no-go gate

Before spending on a live deployment (any provider):

1. All three incident scenarios (`network_latency`, `cert_renewal_stall`,
   `etcd_compaction_stall`) **detected** (no `null` latency).
2. **Zero false positives** pre-`incident_at` on those three, and **zero
   incidents flagged anywhere** on `noisy_baseline_no_incident` (its whole
   series counts as "pre-incident" — `detected: true` there is itself a
   failing result).
3. Detection latency is **actionable** — comfortably under the remediation
   window for that incident class (a network degradation you can reroute in
   minutes is not helped by a detector that takes hours).

Any scenario failing this gate is a detector/threshold problem to fix
*offline* (tune `enter_after`/`exit_after`, `warning`/`critical` thresholds,
`context_length`) — not a reason to reach for more infrastructure. Once all
three clear, a live run's only remaining question is deployment mechanics
(the connector, the runner, the provider), not detection quality.

## Current status (2026-09-26) — gate not yet green

`network_latency` and `cert_renewal_stall` detect cleanly (0 false
positives). **`etcd_compaction_stall` and `noisy_baseline_no_incident` do not
clear the gate yet**, and the root cause for `etcd_compaction_stall` turned
out not to be a threshold problem:

- **`etcd_compaction_stall` is missed even at full production capacity**
  (`epochs=30, d_model=32`, not just the harness's reduced `epochs=10,
  d_model=16`) — verified directly against `ReconstructionDetector` alone,
  score never leaves the 0.7-1.7 range post-incident. Tried widening entry to
  *either* face reading critical (not just forecast) — reverted: it didn't
  fix `etcd_compaction_stall` (reconstruction never reads critical there
  either) and made `noisy_baseline_no_incident` worse (more false positives,
  since detective got a second chance to misfire on the benign blips too).
- **Why, structurally:** both `PatchTSTDetector` and `ReconstructionDetector`
  train from scratch on `values[:tick]` — the growing window — on every
  call. `etcd_compaction_stall`'s incident is a *sustained* level-shift, not
  a spike. Once enough post-incident data enters that training window, the
  model **learns the new level as the new normal** and stops flagging it —
  it absorbs a persistent anomaly instead of catching it. A one-off spike
  (`network_latency`'s ramp, `cert_renewal_stall`'s stall-through-zero)
  doesn't have time to get "learned away" before it's caught; a sustained
  plateau does.
- **The likely real fix**: test with `ForecastInferenceDetector` /
  `ReconstructionInferenceDetector` (`patchtst-infer`/`reconstruction-infer`
  in `pipeline/runner.py`) instead — these load a frozen checkpoint rather
  than retraining per tick, so "normal" can't drift toward whatever the
  recent window looks like. `tools/capture_signals.py` doesn't exercise
  these yet; that's the next concrete step, not further threshold tuning on
  the train-on-the-fly detectors.
- **`noisy_baseline_no_incident` is a calibration problem, not an infra
  one:** the detector misreads isolated, unrelated benign blips as
  `critical` — it has not learned to tell "one-off noise" apart from a real
  sustained shift. This is the same axis as `etcd_compaction_stall` (both are
  properties of the trained model's behavior), just the opposite failure
  mode: one under-reacts to a real sustained incident, the other over-reacts
  to noise that isn't one.
- **Its false-positive count is also not perfectly reproducible run-to-run**
  (2 on one run, 4 on another, same code, same input data) — the scenario
  data has a fixed RNG seed, but neither detector's own torch training seeds
  its weight init, so the "Deterministic... so captures are reproducible"
  claim above only covers the input series, not the trained model. Worth a
  fixed torch seed in the harness if `noisy_baseline_no_incident` becomes the
  thing being tuned against.

## Follow-up (2026-09-27) — frozen checkpoint verified against `etcd_compaction_stall`

The "likely real fix" above is now verified, not just proposed:
`tools/verify_h015_inference.py` trains small forecast+reconstruction
checkpoints on a long (2000-tick) run of *only* the pre-incident periodic
pattern (never sees the incident), freezes them, and runs
`ForecastInferenceDetector`/`ReconstructionInferenceDetector` (wrapped in the
same `RegimeSwitchDetector`) against the real `etcd_compaction_stall` series.

- **Detects at tick ~116 (latency ~6 ticks), 0 false positives before
  `incident_at`** — versus a total miss for the train-on-the-fly pair, and
  competitive with `network_latency`'s 10-tick latency. Scores at the break
  are unambiguous (critical, 18-35 vs. a 1.8/3.0 warning/critical threshold),
  not a borderline call.
- **New, different limitation found**: the regime does not *stay* INCIDENT
  indefinitely — it drifts back to NORMAL after roughly 30-40 more ticks,
  even though the frozen model's weights never change and the incident in
  the scenario never actually ends. Root cause is `_score()`'s own rolling
  baseline (`inference_detector.py`): it's an empirical baseline computed
  from recent windows of the *same growing `v`*, not something the frozen
  model "knows" as normal. Once most of that recent history is itself
  past `incident_at`, the baseline windows are scored against the same
  frozen model as the eval window, so both come out elevated and the ratio
  normalizes back toward 1.0 — a second, different way to "dilute" a
  sustained anomaly, this time in the score's baseline math rather than the
  model's weights.
- **Net assessment**: a real, substantial improvement (correct, fast, clean
  entry) with a distinct remaining gap (sustained alerting) — not wired into
  `capture_signals.py`'s regular run since it trains its own
  scenario-specific checkpoint from scratch (~1-2 min), kept as a standalone,
  on-demand verification for now. Fixing the baseline-dilution gap would mean
  seeding `_baseline()` from a fixed reference/validation window instead of
  the live growing series — not attempted here.

## Reruns

```sh
python -m tools.capture_signals                      # defaults: epochs=10, step=8
python -m tools.capture_signals --epochs 30 --step 4  # closer to prod capacity, slower
```
