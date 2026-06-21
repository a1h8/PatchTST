"""RegimeSwitchDetector — composes the two faces of D1 into one verdict.

A per-(entity, metric) state machine decides which face drives the signal:

    NORMAL    forecast face (anticipation)
       │  forecast severity == critical, ``enter_after`` ticks running
       ▼
    INCIDENT  reconstruction face (detective — the clean OOD signal)
       │  reconstruction severity == normal, ``exit_after`` ticks running
       ▼
    NORMAL

Only one face runs per call (cheap). The emitted SignalRecord keeps the running
face's ``method`` (``patchtst`` / ``patchtst-recon`` / ``zscore``) and is annotated
with ``labels["mode"]`` (anticipation|detective) and ``labels["regime"]`` (the
regime after this assessment).

Anti-flapping (roadmap M4). Two mechanisms keep the state machine from
oscillating on a score that hovers near a threshold:

  * **Hysteresis (dead-band).** Entry and exit use *different* gates: we enter
    INCIDENT on a ``critical`` forecast, but only leave on a ``normal``
    reconstruction — the ``warning`` band in between holds the current regime.
  * **Debounce (persistence).** A transition needs ``enter_after`` consecutive
    break ticks (resp. ``exit_after`` consecutive recovery ticks); a single
    contrary tick resets the streak. ``enter_after = exit_after = 1`` reproduces
    the instantaneous switch. The streak lives in the regime state, so it
    survives across calls (and, once seeded, across batch runs).

State note: the regime is held in a pluggable state object. Within a
long-lived/streaming run the in-memory default works directly; for separate
batch runs (e.g. a K3s CronJob) pass :class:`KBSeededRegimeState`, which seeds
each key from the last persisted signal's ``labels["regime"]`` so an in-flight
incident survives across runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

from kb.signal import SignalRecord

from .detector import Detector


@dataclass(frozen=True)
class RegimeStatus:
    """A key's regime plus its debounce streak (consecutive transition-pushing ticks)."""

    regime: str = "normal"
    streak: int = 0


class InMemoryRegimeState:
    """Per-key regime store with a debounce streak. Default ('normal', 0).

    ``get``/``set`` keep the plain regime-string interface (regime only);
    ``get_status``/``set_status`` carry the streak the debounce needs.
    """

    def __init__(self) -> None:
        self._state: dict[tuple[str, str], RegimeStatus] = {}

    def get(self, key: tuple[str, str]) -> str:
        return self._state.get(key, RegimeStatus()).regime

    def set(self, key: tuple[str, str], regime: str) -> None:
        self._state[key] = RegimeStatus(regime, 0)

    def get_status(self, key: tuple[str, str]) -> RegimeStatus:
        return self._state.get(key, RegimeStatus())

    def set_status(self, key: tuple[str, str], status: RegimeStatus) -> None:
        self._state[key] = status


class KBSeededRegimeState(InMemoryRegimeState):
    """In-memory regime state seeded lazily from the knowledge base.

    Separate batch runs would each reset every key to NORMAL, losing an
    in-flight incident between runs. This seeds a key, on first access, from the
    last persisted signal's ``labels["regime"]`` (via ``store.latest``), then
    behaves exactly like the in-memory parent. ``store`` is any object exposing
    ``latest(entity, metric)`` (the KB :class:`~kb.store.SignalStore`).

    The debounce *streak* is not persisted (only the regime is, in the signal's
    labels), so a run resumes the regime with a fresh streak — transition timing
    restarts, the regime itself carries over.
    """

    _VALID = {"normal", "incident"}

    def __init__(self, store, *, default: str = "normal") -> None:
        super().__init__()
        self._store = store
        self._default = default
        self._seeded: set[tuple[str, str]] = set()

    def _seed(self, key: tuple[str, str]) -> None:
        if key in self._seeded:
            return
        self._seeded.add(key)   # mark first: a key with no record isn't re-queried
        rec = self._store.latest(key[0], key[1])
        if rec is not None:
            regime = rec.labels.get("regime", self._default)
            if regime not in self._VALID:
                regime = self._default
            self._state[key] = RegimeStatus(regime, 0)

    def get(self, key: tuple[str, str]) -> str:
        self._seed(key)
        return super().get(key)

    def get_status(self, key: tuple[str, str]) -> RegimeStatus:
        self._seed(key)
        return super().get_status(key)

    def set(self, key: tuple[str, str], regime: str) -> None:
        self._seeded.add(key)   # an explicit value supersedes any KB seed
        super().set(key, regime)

    def set_status(self, key: tuple[str, str], status: RegimeStatus) -> None:
        self._seeded.add(key)
        super().set_status(key, status)


@dataclass
class RegimeSwitchDetector(Detector):
    forecast: Detector       # anticipation face (NORMAL)
    detective: Detector      # reconstruction face (INCIDENT)
    state: InMemoryRegimeState = field(default_factory=InMemoryRegimeState)
    enter_after: int = 1     # consecutive 'critical' forecasts to enter INCIDENT
    exit_after: int = 1      # consecutive 'normal' reconstructions to leave INCIDENT

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
        status = self.state.get_status(key)

        if status.regime == "normal":
            sig = self.forecast.detect(entity_uid, metric_name, values, ts, labels)
            mode = "anticipation"
            # a break candidate: the forecaster residual spikes to critical.
            next_status = self._advance(
                status, pushing=sig.severity == "critical",
                threshold=self.enter_after, target="incident",
            )
        else:  # incident
            sig = self.detective.detect(entity_uid, metric_name, values, ts, labels)
            mode = "detective"
            # a recovery candidate: reconstruction error back to baseline.
            next_status = self._advance(
                status, pushing=sig.severity == "normal",
                threshold=self.exit_after, target="normal",
            )

        self.state.set_status(key, next_status)
        return replace(
            sig,
            labels={**sig.labels, "mode": mode, "regime": next_status.regime},
        )

    @staticmethod
    def _advance(
        status: RegimeStatus, *, pushing: bool, threshold: int, target: str
    ) -> RegimeStatus:
        """Debounced transition: count consecutive ``pushing`` ticks; switch to
        ``target`` (streak reset) once they reach ``threshold``; a non-pushing
        tick resets the streak and holds the current regime."""
        if not pushing:
            return RegimeStatus(status.regime, 0)
        streak = status.streak + 1
        if streak >= threshold:
            return RegimeStatus(target, 0)
        return RegimeStatus(status.regime, streak)
