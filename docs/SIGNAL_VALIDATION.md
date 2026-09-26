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

## Scenarios (h013+, roadmap B14)

Each scenario (`scenarios/library.py`) is a synthetic series with a known
ground-truth incident onset (`incident_at`), chosen to exercise a *specific*
face of the regime-switch detector — not just "some anomaly somewhere":

| Scenario | Shape | Exercises |
|---|---|---|
| **h013 — network latency** | Flat baseline → slow ramp to a sustained elevated level (a real degrading link, never self-recovers) | Forecast/anticipation face: a predictable trend breaking a learned flat baseline |
| **h014 — cert renewal stall** | Periodic sawtooth (renewal resets the countdown) → the reset stops happening, countdown runs through zero | Reconstruction/detective face: a break in a learned *periodic* pattern, not a smooth trend |
| **h015 — etcd compaction stall** | Periodic transient latency spikes that fully recover → spikes stop recovering, sustained elevation between them | Reconstruction/detective face: the learned "spike then recover" pattern breaks |
| **h016 — noisy baseline, no incident** | Noisy-but-healthy series with isolated single-tick blips that self-recover — `incident_at` never happens | Negative control: a detector well-tuned against h013–h015 that still cries wolf on ordinary noise is not ready to page anyone |

h016 is not a fourth incident shape — it's the false-positive check the other
three don't cover on their own (they only measure false positives *before*
their own onset, never across a series with no onset at all).

These deliberately are not the same shape as h001–h010 (kube-verdict's own
K8s-status fixtures) — those are point-in-time state (CrashLoopBackOff,
ImagePullBackOff), not time-series patterns. h013+ is where a temporal
detector earns its place: a scenario a status-only check cannot see.

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

1. All three incident scenarios (h013–h015) **detected** (no `null` latency).
2. **Zero false positives** pre-`incident_at` on h013–h015, and **zero
   incidents flagged anywhere** on h016 (its whole series counts as
   "pre-incident" — `detected: true` there is itself a failing result).
3. Detection latency is **actionable** — comfortably under the remediation
   window for that incident class (a network degradation you can reroute in
   minutes is not helped by a detector that takes hours).

Any scenario failing this gate is a detector/threshold problem to fix
*offline* (tune `enter_after`/`exit_after`, `warning`/`critical` thresholds,
`context_length`) — not a reason to reach for more infrastructure. Once all
three clear, a live run's only remaining question is deployment mechanics
(the connector, the runner, the provider), not detection quality.

## Current status (2026-09-26) — gate not yet green

h013 and h014 detect cleanly (0 false positives). **h015 and h016 do not
clear the gate yet**, and the root cause for h015 turned out not to be a
threshold problem:

- **h015 is missed even at full production capacity** (`epochs=30,
  d_model=32`, not just the harness's reduced `epochs=10, d_model=16`) —
  verified directly against `ReconstructionDetector` alone, score never
  leaves the 0.7-1.7 range post-incident. Tried widening entry to *either*
  face reading critical (not just forecast) — reverted: it didn't fix h015
  (reconstruction never reads critical there either) and made h016 worse
  (more false positives, since detective got a second chance to misfire on
  the benign blips too).
- **Why, structurally:** both `PatchTSTDetector` and `ReconstructionDetector`
  train from scratch on `values[:tick]` — the growing window — on every
  call. h015's incident is a *sustained* level-shift, not a spike. Once
  enough post-incident data enters that training window, the model **learns
  the new level as the new normal** and stops flagging it — it absorbs a
  persistent anomaly instead of catching it. A one-off spike (h013's ramp,
  h014's stall-through-zero) doesn't have time to get "learned away" before
  it's caught; a sustained plateau does.
- **The likely real fix**: test with `ForecastInferenceDetector` /
  `ReconstructionInferenceDetector` (`patchtst-infer`/`reconstruction-infer`
  in `pipeline/runner.py`) instead — these load a frozen checkpoint rather
  than retraining per tick, so "normal" can't drift toward whatever the
  recent window looks like. `tools/capture_signals.py` doesn't exercise
  these yet; that's the next concrete step, not further threshold tuning on
  the train-on-the-fly detectors.
- **h016's false-positive count is not perfectly reproducible run-to-run**
  (2 on one run, 4 on another, same code, same input data) — the scenario
  data has a fixed RNG seed, but neither detector's own torch training seeds
  its weight init, so the "Deterministic... so captures are reproducible"
  claim above only covers the input series, not the trained model. Worth a
  fixed torch seed in the harness if h016 becomes the thing being tuned
  against.

## Reruns

```sh
python -m tools.capture_signals                      # defaults: epochs=10, step=8
python -m tools.capture_signals --epochs 30 --step 4  # closer to prod capacity, slower
```
