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

## Reruns

```sh
python -m tools.capture_signals                      # defaults: epochs=10, step=8
python -m tools.capture_signals --epochs 30 --step 4  # closer to prod capacity, slower
```
