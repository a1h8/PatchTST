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
- **`elapsed_s`** — wall-clock cost of the capture (CPU-only). Runs at the
  production model capacity by default (`d_model=32`, 2 layers, `epochs=30`),
  with ticks every 5 points — the deployed `*/5` CronJob over 60 s points. An
  earlier version of this harness ran at reduced capacity (`d_model=16`,
  `epochs=10`); see the findings below for why that was not representative.

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

## How `etcd_compaction_stall` got fixed — three separate problems, not one

Chasing this one scenario surfaced three distinct problems at three
different layers. None of them were fixed by tuning one threshold harder;
kept here in the order they were found, since each changes how to read the
next.

### 1. A scoring bug that had nothing to do with `etcd_compaction_stall`

`PatchTSTDetector` scored `eval_rmse / baseline_rmse` with the baseline
taken **in-sample**, on training windows. The better the model overfits (30
epochs, prod capacity), the closer that baseline gets to 0, so *any* unseen
window scores as a huge ratio — at production capacity every scenario
flipped to `incident` on the same warm-up tick (scores up to ×4000); the
harness's earlier reduced-capacity default (`epochs=10, d_model=16`) had
been masking this the whole time. Fixed by measuring the baseline on a
held-out tail the model never trained on (`holdout_chunks`), falling back
to z-score until the series is long enough to hold one out.

### 2. Both faces are structurally blind to a *sustained* level shift

Two independent ways this showed up, and two independent mitigations —
not competing fixes, complementary layers:

- **Train-on-the-fly retrains itself out of catching it.** Both
  `PatchTSTDetector` and `ReconstructionDetector` train from scratch on
  `values[:tick]` — the growing window — on every call.
  `etcd_compaction_stall`'s incident is a sustained level-shift, not a
  spike, so once enough post-incident data enters that training window the
  model **learns the new level as the new normal** and stops flagging it. A
  one-off spike (`network_latency`'s ramp, `cert_renewal_stall`'s
  stall-through-zero) doesn't have time to get "learned away" before it's
  caught; a sustained plateau does. Global z-normalisation and each model's
  own per-window scaling compound this: they remove the level entirely, so
  the shift is only visible at the *instant* it happens — with coarse tick
  spacing (every 5 or 8 points, not every 1-2), the instant can be stepped
  over entirely and the scenario is missed outright, exactly as first
  observed.
- **Mitigation A — a third, detector-agnostic signal**
  (`detection/levelshift.py` + `RegimeSwitchDetector.level_critical`).
  Compares the recent window's median against the window's own baseline
  median, in MAD units — operates on raw values, not on either face's
  internal ratio, so it doesn't share either face's blind spot. At or above
  `level_critical` (8.0, the default) it counts as a break in NORMAL *and*
  blocks recovery in INCIDENT, so the level-blind reconstruction face can't
  end an incident while the plateau persists. `level_critical: null`
  disables it. Spot-checked (not a committed test) against legitimate
  trends — memory ramps, disk fill, daily cycles, a random walk — all
  scored under 3.2, well under the 8.0 cut.
- **Mitigation B — stop retraining at all** (verified 2026-09-27,
  `tools/verify_h015_inference.py`): freeze forecast+reconstruction
  checkpoints once, on a long pre-incident-only run, and use
  `ForecastInferenceDetector`/`ReconstructionInferenceDetector` instead of
  the train-on-the-fly pair. Detects at tick ~116 (latency ~6 ticks), 0
  false positives, scores unambiguous (critical, 18-35 vs. a 1.8/3.0
  threshold) — the model genuinely never drifts. **But this trades one
  blind spot for another**: `_score()`'s rolling baseline is still computed
  from recent windows of the same *growing* series, not something the
  frozen model "knows" as normal — so long enough after the incident, most
  of those baseline windows are themselves post-onset, both baseline and
  eval error read elevated, and the ratio normalizes back toward 1.0. The
  regime drifts back to NORMAL after ~30-40 ticks even though the incident
  never ended and the model's weights never changed (characterized as a
  fast, deterministic test in `tests/test_inference_detector.py`, no
  training needed to reproduce it). **Mitigation A's "block recovery while
  displaced" rule directly covers this gap** — the level-shift check reads
  raw values, so a diluted ratio in the detective face doesn't matter once
  level-shift is stopping the exit transition on its own. Not wired
  together yet (the inference detectors aren't in `capture_signals.py`'s
  regular run — see caveat below); worth doing before trusting sustained
  alerting on the inference path in production.

### 3. Tick spacing must match the deployment cadence

Denser ticks "detected" the scenario with *negative* latency before the
level-shift check existed — false positives on the ordinary periodic
spikes, not a real detection. Results are now reported at the actual
deployment cadence (`--step 5`, the `*/5` CronJob over 60s points), and the
three incident scenarios were swept over steps 1-8 at reduced capacity: all
detected, zero false positives, at every step — cadence sensitivity was a
real effect, not a fluke of one step value.

### Caveat: `noisy_baseline_no_incident` didn't exist when the level-shift check was written

The level-shift mitigation (and its trend spot-check above) predates this
negative-control scenario. It needs a real run against it, not an assumption
that "robust to trends" also means "robust to isolated blips" — see
`## Current status` below for the actual result.

## Reruns

```sh
python -m tools.capture_signals                              # prod capacity, step 5 (the gate)
python -m tools.capture_signals --epochs 10 --d-model 16 --layers 1 --step 8   # fast
```

Needs torch/transformers (`requirements-detection-patchtst.txt`); the
`patchtst-pipeline:torch` image (`--build-arg INSTALL_TORCH=1`) has them.

## Current status

At production capacity, step 5: h013 detected (7 ticks), h014 detected
(37 ticks), h015 detected (12 ticks), zero false positives on all three — the gate
above clears on the synthetic scenarios. Caveats: latency is in ticks of 60 s
points; h014's 37 ticks is long for anything but a slow, days-scale expiry; and
these are synthetic series — the level-shift threshold in particular has only been
spot-checked against realistic trends, not against real metric history.
