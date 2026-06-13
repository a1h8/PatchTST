"""RegimeSwitchDetector — composes the two faces of D1 into one verdict.

A per-(entity, metric) state machine decides which face drives the signal:

    NORMAL    forecast face (anticipation)
       │  forecast severity == critical (a break: forecaster residual spikes)
       ▼
    INCIDENT  reconstruction face (detective — the clean OOD signal)
       │  reconstruction severity == normal (recovery)
       ▼
    NORMAL

Only one face runs per call (cheap). The emitted SignalRecord keeps the running
face's ``method`` (``patchtst`` / ``patchtst-recon`` / ``zscore``) and is annotated
with ``labels["mode"]`` (anticipation|detective) and ``labels["regime"]`` (the
regime after this assessment).

State note: the regime is held in a pluggable ``RegimeState`` (in-memory by
default). Within a long-lived/streaming run this works directly; for separate
batch runs (e.g. a K3s CronJob) the regime should be seeded from the knowledge
base (the last signal's ``labels["regime"]``) — a deployment follow-up.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

from kb.signal import SignalRecord

from .detector import Detector


class InMemoryRegimeState:
    """Per-key regime store. Default 'normal'."""

    def __init__(self) -> None:
        self._state: dict[tuple[str, str], str] = {}

    def get(self, key: tuple[str, str]) -> str:
        return self._state.get(key, "normal")

    def set(self, key: tuple[str, str], regime: str) -> None:
        self._state[key] = regime


@dataclass
class RegimeSwitchDetector(Detector):
    forecast: Detector       # anticipation face (NORMAL)
    detective: Detector      # reconstruction face (INCIDENT)
    state: InMemoryRegimeState = field(default_factory=InMemoryRegimeState)

    method = "regime-switch"

    def detect(
        self,
        entity_uid: str,
        metric_name: str,
        values: Sequence[float],
        ts: int,
        labels: dict | None = None,
    ) -> SignalRecord:
        key = (entity_uid, metric_name)
        regime = self.state.get(key)

        if regime == "normal":
            sig = self.forecast.detect(entity_uid, metric_name, values, ts, labels)
            mode = "anticipation"
            # a break: the forecaster residual spikes to critical → switch to
            # the detective face for the accurate read next time.
            next_regime = "incident" if sig.severity == "critical" else "normal"
        else:  # incident
            sig = self.detective.detect(entity_uid, metric_name, values, ts, labels)
            mode = "detective"
            # recovery: reconstruction error back to baseline.
            next_regime = "normal" if sig.severity == "normal" else "incident"

        self.state.set(key, next_regime)
        return replace(
            sig,
            labels={**sig.labels, "mode": mode, "regime": next_regime},
        )
