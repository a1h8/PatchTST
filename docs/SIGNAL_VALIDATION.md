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

1. All three scenarios **detected** (no `null` latency).
2. **Zero false positives** pre-`incident_at` on every scenario.
3. Detection latency is **actionable** — comfortably under the remediation
   window for that incident class (a network degradation you can reroute in
   minutes is not helped by a detector that takes hours).

Any scenario failing this gate is a detector/threshold problem to fix
*offline* (tune `enter_after`/`exit_after`, `warning`/`critical` thresholds,
`context_length`) — not a reason to reach for more infrastructure. Once all
three clear, a live run's only remaining question is deployment mechanics
(the connector, the runner, the provider), not detection quality.

## Findings from the first captures (and what changed)

The first real run (reduced capacity, ticks every 8 points) reported 2/3, with
h015 missed. Chasing that showed the harness itself was hiding worse problems:

1. **The forecast baseline was measured in-sample.** `PatchTSTDetector` scored
   `eval_rmse / baseline_rmse` with the baseline taken on training windows. The
   better the model overfits (30 epochs, or one training window early in the
   series), the closer the baseline gets to 0, so any unseen window scored as a
   huge ratio: at production capacity every scenario flipped to `incident` on
   the same warm-up tick (scores up to ×4000). The reduced-capacity run had
   masked this. The baseline is now measured on a held-out tail the model never
   trained on (`holdout_chunks`), and the detector falls back to z-score until the
   series is long enough to hold it out.
2. **Both faces are blind to a sustained level shift.** Global z-normalisation
   and the model's per-window scaling remove the level, and a forecaster
   re-trained on a window that already contains a plateau learns the new level.
   h015 (latency that stops recovering) was therefore only visible at the instant
   it starts, so detection depended on the phase of the ticks relative to the
   onset (caught with ticks every 1, 2 or 4 points, missed at 5 and 8). Denser
   ticks that "detected" it did so with negative latency — false positives on the
   periodic spikes, not a detection. `detection/levelshift.py` adds a robust
   check (median of the recent window vs the window's own baseline, in MAD
   units): at or above `level_critical` (8) it counts as a break in NORMAL and
   blocks recovery in INCIDENT, so the level-blind reconstruction face cannot end
   an incident while the plateau persists. It is enabled by default;
   `level_critical: null` turns it off. Legitimate trends (memory ramps, disk
   fill, daily cycles) scored under 3.2 in an ad-hoc spot check (not a committed test).
3. **Tick spacing matters and must match the deployment.** Results are reported
   for the CronJob cadence (`--step 5`). The same three scenarios were also swept
   over steps 1–8 at reduced capacity: all detected with zero false positives at
   every step.

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
