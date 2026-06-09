"""Connector SPI tests — pure-Python, no Beam install required.

Covers: pivot validation, internal registry, the conformance contract for the
built-in connectors, and the multivariate alignment that backs decision D4.
"""
import pytest

import connectors  # noqa: F401  (triggers built-in registration)
from connectors.alignment import align_group
from connectors.base import SinkConnector, SourceConnector
from connectors.conformance import (
    assert_buildable,
    assert_registered,
    assert_sink_contract,
    assert_source_contract,
)
from connectors.pivot import PivotRow
from connectors.registry import available, connector
from connectors.sinks.parquet import ParquetSink
from connectors.sources.mimir import MimirSource, query_range, to_pivot_rows


class _FakeResp:
    """Minimal urlopen() return value: a context manager exposing read()."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --- pivot schema ---------------------------------------------------------

def test_pivot_valid():
    row = PivotRow("pod-a", 1000, (1.0, 2.0), ("cpu", "mem"))
    assert row.width == 2


def test_pivot_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        PivotRow("pod-a", 1000, (1.0,), ("cpu", "mem"))


def test_pivot_rejects_float_ts():
    with pytest.raises(ValueError, match="ts must be int"):
        PivotRow("pod-a", 1000.0, (1.0,), ("cpu",))  # type: ignore[arg-type]


def test_pivot_rejects_duplicate_channels():
    with pytest.raises(ValueError, match="duplicate channels"):
        PivotRow("pod-a", 1000, (1.0, 2.0), ("cpu", "cpu"))


# --- registry -------------------------------------------------------------

def test_builtins_registered():
    reg = available()
    assert reg.get("mimir") == "source"
    assert reg.get("parquet") == "sink"


def test_build_unknown_raises():
    with pytest.raises(KeyError, match="unknown connector"):
        assert_buildable("does-not-exist")


# --- conformance ----------------------------------------------------------

def test_mimir_conforms_to_source_contract():
    assert_registered("mimir")
    src = assert_buildable(
        "mimir", endpoint="http://mimir:9009", promql="up", start=0, end=10
    )
    assert_source_contract(src)


def test_parquet_conforms_to_sink_contract():
    assert_registered("parquet")
    sink = assert_buildable("parquet", path="/tmp/out")
    assert_sink_contract(sink)


# --- multivariate alignment (D4) -----------------------------------------

def test_align_group_ffill_across_cadences():
    # cpu @ 15s, mem @ 30s on a 15s grid -> mem forward-filled into the gap.
    rows = align_group(
        "pod-a",
        {
            "cpu": [(0, 1.0), (15_000, 1.1), (30_000, 1.2)],
            "mem": [(0, 5.0), (30_000, 5.5)],
        },
        step_ms=15_000,
        fill="ffill",
    )
    assert [r.ts for r in rows] == [0, 15_000, 30_000]
    assert rows[0].channels == ("cpu", "mem")
    assert rows[1].values == (1.1, 5.0)   # mem carried forward
    assert rows[2].values == (1.2, 5.5)


def test_align_group_drop_incomplete():
    rows = align_group(
        "pod-a",
        {"cpu": [(0, 1.0), (15_000, 1.1)], "mem": [(15_000, 5.0)]},
        step_ms=15_000,
        fill="drop",
    )
    # grid bucket 0 has no mem -> dropped; only bucket 15000 is complete.
    assert [r.ts for r in rows] == [15_000]


def test_mimir_to_pivot_rows_groups_and_aligns():
    result = [
        {
            "metric": {"__name__": "cpu", "instance": "pod-a"},
            "values": [[0, "1.0"], [15, "1.1"]],
        },
        {
            "metric": {"__name__": "mem", "instance": "pod-a"},
            "values": [[0, "5.0"], [15, "5.5"]],
        },
    ]
    rows = to_pivot_rows(
        result,
        group_by=["instance"],
        channel_label="__name__",
        step_ms=15_000,
        fill="ffill",
    )
    assert {r.group_id for r in rows} == {"pod-a"}
    assert rows[0].channels == ("cpu", "mem")
    assert rows[0].values == (1.0, 5.0)


# --- pivot edge cases -----------------------------------------------------

def test_pivot_rejects_empty_group_id():
    with pytest.raises(ValueError, match="group_id must be non-empty"):
        PivotRow("", 0, (1.0,), ("cpu",))


def test_pivot_default_labels_empty():
    row = PivotRow("g", 0, (1.0,), ("cpu",))
    assert row.labels == {}


# --- registry edge cases --------------------------------------------------

def test_connector_rejects_non_connector_class():
    with pytest.raises(TypeError, match="must subclass"):

        @connector("bad")
        class _Bad:  # not a Source/Sink
            pass


def test_connector_rejects_duplicate_name():
    with pytest.raises(ValueError, match="already registered"):

        @connector("mimir")  # already taken by MimirSource
        class _Dup(SourceConnector):
            def read(self):  # pragma: no cover
                ...


def test_available_is_sorted():
    keys = list(available())
    assert keys == sorted(keys)


def test_assert_registered_missing_raises():
    with pytest.raises(AssertionError, match="not registered"):
        assert_registered("nope")


# --- alignment edge cases -------------------------------------------------

def test_align_group_empty_series():
    assert align_group("g", {}, step_ms=1000) == []


def test_align_group_zero_fill_before_first_value():
    # channel 'b' has no value at bucket 0 -> 'zero' fills 0.0 instead of skipping
    rows = align_group(
        "g",
        {"a": [(0, 1.0)], "b": [(1000, 2.0)]},
        step_ms=1000,
        fill="zero",
    )
    assert rows[0].ts == 0 and rows[0].values == (1.0, 0.0)
    assert rows[1].values == (1.0, 2.0)  # 'a' carried forward, 'b' now present


def test_align_group_ffill_skips_until_all_channels_seen():
    # 'b' only appears at bucket 2000 -> earlier buckets have no prior to ffill,
    # so they are skipped until every channel has been seen at least once.
    rows = align_group(
        "g",
        {"a": [(0, 1.0), (1000, 1.1), (2000, 1.2)], "b": [(2000, 9.0)]},
        step_ms=1000,
        fill="ffill",
    )
    assert [r.ts for r in rows] == [2000]
    assert rows[0].values == (1.2, 9.0)


def test_align_group_single_channel():
    rows = align_group("g", {"cpu": [(0, 1.0), (1000, 2.0)]}, step_ms=1000)
    assert [r.values for r in rows] == [(1.0,), (2.0,)]
    assert all(r.width == 1 for r in rows)


# --- connector describe / parquet record ----------------------------------

def test_mimir_describe():
    src = MimirSource("http://m:9009", "up", 0, 10, group_by=["pod"])
    d = src.describe()
    assert d["kind"] == "source" and d["promql"] == "up" and d["group_by"] == ["pod"]


def test_mimir_defaults_group_id_when_label_absent():
    result = [{"metric": {"__name__": "cpu"}, "values": [[0, "1.0"]]}]
    rows = to_pivot_rows(
        result, group_by=["instance"], channel_label="__name__",
        step_ms=1000, fill="ffill",
    )
    assert rows[0].group_id == "default"


def test_parquet_as_record_and_describe():
    sink = ParquetSink("/tmp/out")
    rec = ParquetSink.as_record(PivotRow("g", 5, (1.0, 2.0), ("a", "b"), {"k": "v"}))
    assert rec["group_id"] == "g" and rec["values"] == [1.0, 2.0]
    assert rec["labels"] == '{"k": "v"}'
    assert sink.describe()["path"] == "/tmp/out"


def test_sink_and_source_kinds():
    assert ParquetSink("/x").kind == "sink"
    assert MimirSource("http://m", "up", 0, 1).kind == "source"
    assert issubclass(MimirSource, SourceConnector)
    assert issubclass(ParquetSink, SinkConnector)


# --- mimir HTTP query (urlopen mocked) ------------------------------------

def test_query_range_success(monkeypatch):
    import json as _json

    payload = {
        "status": "success",
        "data": {"result": [{"metric": {}, "values": [[0, "1.0"]]}]},
    }
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["header_values"] = list(req.headers.values())
        return _FakeResp(_json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = query_range("http://mimir:9009/", "up", 0, 30, 15, tenant="team-a")

    assert out == payload["data"]["result"]
    assert "query=up" in captured["url"]
    assert "/prometheus/api/v1/query_range" in captured["url"]
    assert "team-a" in captured["header_values"]  # tenant header set


def test_query_range_no_tenant_omits_header(monkeypatch):
    import json as _json

    payload = {"status": "success", "data": {"result": []}}
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["header_values"] = list(req.headers.values())
        return _FakeResp(_json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert query_range("http://m", "up", 0, 1, 15) == []
    assert captured["header_values"] == []


def test_query_range_error_status_raises(monkeypatch):
    import json as _json

    payload = {"status": "error", "error": "bad query"}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: _FakeResp(_json.dumps(payload).encode()),
    )
    with pytest.raises(RuntimeError, match="bad query"):
        query_range("http://m", "up", 0, 1, 15)