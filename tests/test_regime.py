"""RegimeSwitchDetector tests — deterministic state machine with stub
detectors (no torch). Verifies which face runs, the NORMAL↔INCIDENT
transitions, and the mode/regime annotations."""
from detection import InMemoryRegimeState, RegimeStatus, RegimeSwitchDetector
from detection.detector import Detector
from kb.signal import SignalRecord


class FakeDetector(Detector):
    """Returns a preset severity sequence; records calls."""

    def __init__(self, method, severities):
        self.method = method
        self.severities = list(severities)
        self.calls = []
        self._i = 0

    def detect(self, entity_uid, metric_name, values, ts, labels=None):
        self.calls.append((entity_uid, metric_name))
        sev = self.severities[min(self._i, len(self.severities) - 1)]
        self._i += 1
        return SignalRecord(
            entity_uid, metric_name, ts, sev, score=1.0,
            method=self.method, labels=dict(labels or {}),
        )


def _detect(d, sev_label="cpu", ts=0):
    return d.detect("Pod/p/a", sev_label, [0.0], ts)


# --- state defaults -------------------------------------------------------

def test_inmemory_regime_state_default_and_set():
    st = InMemoryRegimeState()
    assert st.get(("e", "m")) == "normal"
    st.set(("e", "m"), "incident")
    assert st.get(("e", "m")) == "incident"


# --- the NORMAL ↔ INCIDENT cycle ------------------------------------------

def test_regime_full_cycle():
    fc = FakeDetector("patchtst", ["normal", "critical"])
    dt = FakeDetector("patchtst-recon", ["critical", "normal"])
    d = RegimeSwitchDetector(forecast=fc, detective=dt)

    # tick1: NORMAL, forecast normal → stay NORMAL (anticipation)
    s1 = _detect(d)
    assert s1.method == "patchtst"
    assert s1.labels["mode"] == "anticipation" and s1.labels["regime"] == "normal"

    # tick2: NORMAL, forecast critical → break → INCIDENT
    s2 = _detect(d)
    assert s2.method == "patchtst"
    assert s2.labels["mode"] == "anticipation" and s2.labels["regime"] == "incident"

    # tick3: INCIDENT, reconstruction critical → stay INCIDENT (detective)
    s3 = _detect(d)
    assert s3.method == "patchtst-recon"
    assert s3.labels["mode"] == "detective" and s3.labels["regime"] == "incident"

    # tick4: INCIDENT, reconstruction normal → recovery → NORMAL
    s4 = _detect(d)
    assert s4.method == "patchtst-recon"
    assert s4.labels["mode"] == "detective" and s4.labels["regime"] == "normal"

    # forecast ran in NORMAL ticks, detective in INCIDENT ticks
    assert len(fc.calls) == 2 and len(dt.calls) == 2


def test_regime_warning_in_normal_stays_normal():
    fc = FakeDetector("patchtst", ["warning"])
    dt = FakeDetector("patchtst-recon", ["normal"])
    d = RegimeSwitchDetector(forecast=fc, detective=dt)
    s = _detect(d)
    # an early WARN is anticipation, not a break — stay NORMAL
    assert s.labels["regime"] == "normal" and s.labels["mode"] == "anticipation"
    assert len(dt.calls) == 0


def test_regime_warning_in_incident_stays_incident():
    fc = FakeDetector("patchtst", ["critical"])
    dt = FakeDetector("patchtst-recon", ["warning"])
    d = RegimeSwitchDetector(forecast=fc, detective=dt)
    _detect(d)               # → INCIDENT
    s = _detect(d)           # INCIDENT, recon warning → stay INCIDENT
    assert s.labels["regime"] == "incident" and s.labels["mode"] == "detective"


def test_regime_is_independent_per_entity_metric():
    fc = FakeDetector("patchtst", ["critical", "normal"])
    dt = FakeDetector("patchtst-recon", ["normal"])
    d = RegimeSwitchDetector(forecast=fc, detective=dt)

    a = d.detect("Pod/a", "cpu", [0.0], 0)      # → INCIDENT for (Pod/a, cpu)
    b = d.detect("Pod/b", "cpu", [0.0], 0)      # independent key, still NORMAL
    assert a.labels["regime"] == "incident"
    assert b.labels["regime"] == "normal"


# --- anti-flapping: debounce on entry (enter_after) -----------------------

def _regimes(d, n):
    return [_detect(d, ts=i).labels["regime"] for i in range(n)]


def test_enter_after_needs_consecutive_criticals():
    fc = FakeDetector("patchtst", ["critical", "critical"])
    dt = FakeDetector("patchtst-recon", ["normal"])
    d = RegimeSwitchDetector(forecast=fc, detective=dt, enter_after=2)
    # one critical is not enough; the second consecutive one flips to INCIDENT
    assert _regimes(d, 2) == ["normal", "incident"]
    assert len(dt.calls) == 0   # detective never ran while still NORMAL


def test_enter_streak_resets_on_non_critical():
    fc = FakeDetector("patchtst", ["critical", "normal", "critical"])
    dt = FakeDetector("patchtst-recon", ["normal"])
    d = RegimeSwitchDetector(forecast=fc, detective=dt, enter_after=2)
    # critical, then normal (resets the streak), then a lone critical → never enters
    assert _regimes(d, 3) == ["normal", "normal", "normal"]


# --- anti-flapping: debounce on exit (exit_after) -------------------------

def test_exit_after_needs_consecutive_normals():
    fc = FakeDetector("patchtst", ["critical"])              # clamps → enters once
    dt = FakeDetector("patchtst-recon", ["normal", "normal"])
    d = RegimeSwitchDetector(forecast=fc, detective=dt, exit_after=2)
    # enter on tick0; one normal holds INCIDENT; the second consecutive recovers
    assert _regimes(d, 3) == ["incident", "incident", "normal"]


def test_exit_streak_resets_on_non_normal():
    fc = FakeDetector("patchtst", ["critical"])
    dt = FakeDetector("patchtst-recon", ["normal", "warning", "normal"])
    d = RegimeSwitchDetector(forecast=fc, detective=dt, exit_after=2)
    # enter; normal (streak 1); warning (reset); normal (streak 1) → never recovers
    assert _regimes(d, 4) == ["incident", "incident", "incident", "incident"]


# --- state carries the debounce streak ------------------------------------

def test_regime_status_default_and_backward_compat_get_set():
    st = InMemoryRegimeState()
    assert st.get_status(("e", "m")) == RegimeStatus("normal", 0)
    assert st.get(("e", "m")) == "normal"                 # string view unchanged
    st.set(("e", "m"), "incident")                        # set resets streak
    assert st.get(("e", "m")) == "incident"
    assert st.get_status(("e", "m")) == RegimeStatus("incident", 0)
    st.set_status(("e", "m"), RegimeStatus("incident", 3))
    assert st.get_status(("e", "m")).streak == 3
