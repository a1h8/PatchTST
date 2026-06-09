"""LocalEngine tests — pure Python, no execution engine installed.

Demonstrates the engine-agnostic core: the same connectors run with zero
third-party engine dependency. Parquet read-back uses pyarrow (the sink's own
dep), not Beam.
"""
import pyarrow.parquet as pq

from connectors import LocalEngine, build
from connectors.base import SinkConnector
from connectors.pivot import PivotRow
from connectors.sources.mimir import to_pivot_rows

_FAKE_RESULT = [
    {"metric": {"__name__": "cpu", "instance": "pod-a"}, "values": [[0, "1.0"], [15, "1.1"]]},
    {"metric": {"__name__": "mem", "instance": "pod-a"}, "values": [[0, "5.0"], [15, "5.5"]]},
]


class _MemSink(SinkConnector):
    """Zero-dependency sink that collects rows in memory."""

    def __init__(self):
        self.rows = []

    def write(self, rows):
        self.rows.extend(rows)


class _StaticSource:
    """Minimal in-memory source (duck-typed) for a dependency-free test."""

    def __init__(self, rows):
        self._rows = rows

    def read(self):
        return self._rows

    def native_beam_read(self):
        return None


def test_local_engine_no_third_party_dep():
    rows = [PivotRow("g", 0, (1.0,), ("cpu",)), PivotRow("g", 1000, (2.0,), ("cpu",))]
    sink = _MemSink()
    LocalEngine().run(_StaticSource(rows), [sink])
    assert sink.rows == rows


def test_local_engine_fans_out_to_multiple_sinks():
    rows = [PivotRow("g", 0, (1.0,), ("cpu",))]
    a, b = _MemSink(), _MemSink()
    LocalEngine().run(_StaticSource(rows), [a, b])
    assert a.rows == rows and b.rows == rows


def test_local_engine_mimir_to_parquet(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "connectors.sources.mimir.query_range", lambda *a, **k: _FAKE_RESULT
    )
    src = build("mimir", endpoint="http://m", promql="up", start=0, end=30, step_s=15)
    sink = build("parquet", path=str(tmp_path / "out"))

    LocalEngine().run(src, [sink])

    # agnostic writer produces a single file at path + suffix
    table = pq.read_table(str(tmp_path / "out.parquet")).to_pydict()
    assert table["group_id"] == ["pod-a", "pod-a"]
    assert table["values"] == [[1.0, 5.0], [1.1, 5.5]]


def test_mimir_read_is_engine_agnostic_iterable():
    # read() returns plain PivotRows, no engine types
    rows = to_pivot_rows(
        _FAKE_RESULT, group_by=["instance"], channel_label="__name__",
        step_ms=15_000, fill="ffill",
    )
    assert all(isinstance(r, PivotRow) for r in rows)
