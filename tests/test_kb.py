"""Knowledge-base tests — SignalRecord, SignalStore (write/query), and the
signal_history HTTP contract kube-verdict consumes."""
import pytest

from kb import SignalRecord, SignalStore, create_app


def _rec(entity, metric, ts, severity="warning", score=2.0, method="patchtst", **kw):
    return SignalRecord(
        entity_uid=entity, metric_name=metric, ts=ts,
        severity=severity, score=score, method=method, **kw
    )


# --- SignalRecord ---------------------------------------------------------

def test_signal_record_valid_and_text():
    r = _rec("Pod/prod/api", "cpu_usage", 1000, horizon="short", n_points=64)
    assert r.is_anomalous
    assert "metric=cpu_usage" in r.to_text() and "horizon=short" in r.to_text()


def test_signal_record_custom_text_preserved():
    r = _rec("Pod/p/a", "mem", 0, text="custom narrative")
    assert r.to_text() == "custom narrative"


@pytest.mark.parametrize("bad", [
    dict(entity_uid="", metric_name="m", ts=0, severity="warning", score=1.0, method="zscore"),
    dict(entity_uid="e", metric_name="m", ts=1.0, severity="warning", score=1.0, method="zscore"),
    dict(entity_uid="e", metric_name="m", ts=0, severity="boom", score=1.0, method="zscore"),
    dict(entity_uid="e", metric_name="m", ts=0, severity="normal", score=1.0, method="zscore", horizon="year"),
])
def test_signal_record_validation(bad):
    with pytest.raises(ValueError):
        SignalRecord(**bad)


def test_from_anomaly_result_duck_typed():
    class AR:  # mimics kube-verdict AnomalyResult
        entity_uid = "Pod/prod/api"
        metric_name = "cpu_usage"
        severity = "critical"
        score = 3.4
        method = "patchtst"
        horizon = "medium"
        n_points = 96
        def to_text(self):
            return "ar text"
    r = SignalRecord.from_anomaly_result(AR(), ts=1234, labels={"ns": "prod"})
    assert r.severity == "critical" and r.ts == 1234 and r.labels == {"ns": "prod"}
    assert r.to_text() == "ar text"


# --- SignalStore ----------------------------------------------------------

def test_store_query_empty_returns_list(tmp_path):
    store = SignalStore(str(tmp_path / "kb"))
    assert store.query("Pod/prod/api") == []


def test_store_write_then_query_by_entity(tmp_path):
    store = SignalStore(str(tmp_path / "kb"))
    store.write([
        _rec("Pod/prod/api", "cpu_usage", 1000),
        _rec("Pod/prod/api", "mem", 2000, severity="critical", score=3.5),
        _rec("Pod/prod/db", "cpu_usage", 1500),
    ])
    api = store.query("Pod/prod/api")
    assert [r.metric_name for r in api] == ["cpu_usage", "mem"]   # ordered by ts
    assert {r.entity_uid for r in api} == {"Pod/prod/api"}


def test_store_query_filters_metric_and_window(tmp_path):
    store = SignalStore(str(tmp_path / "kb"))
    store.write([
        _rec("Pod/prod/api", "cpu_usage", 1000),
        _rec("Pod/prod/api", "cpu_usage", 5000),
        _rec("Pod/prod/api", "cpu_usage", 9000),
        _rec("Pod/prod/api", "mem", 5000),
    ])
    by_metric = store.query("Pod/prod/api", metric="cpu_usage")
    assert [r.ts for r in by_metric] == [1000, 5000, 9000]
    windowed = store.query("Pod/prod/api", metric="cpu_usage", since=2000, until=8000)
    assert [r.ts for r in windowed] == [5000]


def test_store_write_empty_returns_none(tmp_path):
    store = SignalStore(str(tmp_path / "kb"))
    assert store.write([]) is None


def test_store_query_limit(tmp_path):
    store = SignalStore(str(tmp_path / "kb"))
    store.write([_rec("Pod/p/a", "cpu", t) for t in (1000, 2000, 3000)])
    assert [r.ts for r in store.query("Pod/p/a", limit=2)] == [1000, 2000]


def test_store_query_across_partitions_and_labels(tmp_path):
    store = SignalStore(str(tmp_path / "kb"))
    store.write([_rec("Pod/p/a", "cpu", 1000, labels={"team": "x"})])
    store.write([_rec("Pod/p/a", "cpu", 2000, labels={"team": "y"})])   # second partition
    rows = store.query("Pod/p/a")
    assert [r.ts for r in rows] == [1000, 2000]
    assert rows[0].labels == {"team": "x"} and rows[1].labels == {"team": "y"}


# --- HTTP contract --------------------------------------------------------

def test_signal_history_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    store = SignalStore(str(tmp_path / "kb"))
    store.write([
        _rec("Pod/prod/api", "cpu_usage", 1000, severity="warning"),
        _rec("Pod/prod/api", "cpu_usage", 2000, severity="critical", score=3.1),
    ])
    client = TestClient(create_app(store))

    assert client.get("/health").json() == {"status": "ok"}

    resp = client.get("/api/v1/signals/history", params={"entity": "Pod/prod/api", "metric": "cpu_usage"})
    body = resp.json()
    assert resp.status_code == 200
    assert body["count"] == 2
    assert [s["severity"] for s in body["signals"]] == ["warning", "critical"]
    assert body["signals"][0]["entity_uid"] == "Pod/prod/api"


def test_signal_history_unknown_entity_empty(tmp_path):
    from fastapi.testclient import TestClient

    store = SignalStore(str(tmp_path / "kb"))
    store.write([_rec("Pod/prod/api", "cpu", 1000)])
    client = TestClient(create_app(store))
    body = client.get("/api/v1/signals/history", params={"entity": "Pod/prod/nope"}).json()
    assert body["count"] == 0 and body["signals"] == []


# --- SPI sink façade (write path goes through the connector cycle) --------

class _SignalSource:
    """Engine-agnostic source yielding SignalRecords (duck-typed)."""

    def __init__(self, rows):
        self._rows = rows

    def read(self):
        return self._rows

    def native_beam_read(self):
        return None


def test_signal_store_sink_registered_and_conforms():
    import kb  # noqa: F401  (registers the sink)
    from connectors import available, build
    from connectors.conformance import assert_sink_contract

    assert available()["signal-store"] == "sink"
    sink = build("signal-store", root="/tmp/kb-x")
    assert_sink_contract(sink)
    assert sink.describe()["root"] == "/tmp/kb-x"


def test_signal_store_sink_full_spi_cycle(tmp_path):
    import kb  # noqa: F401
    from connectors import LocalEngine, build

    sink = build("signal-store", root=str(tmp_path / "kb"))
    rows = [_rec("Pod/p/a", "cpu", 1000), _rec("Pod/p/a", "cpu", 2000)]

    # build → Engine.run → sink.write — the SPI cycle, no standalone write
    LocalEngine().run(_SignalSource(rows), [sink])

    out = sink.store.query("Pod/p/a")
    assert [r.ts for r in out] == [1000, 2000]
