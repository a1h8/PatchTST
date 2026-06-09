"""Integration tests — the Beam engine adapter on the DirectRunner.

Exercises BeamEngine end-to-end (read wrap + native Parquet writer + the
gather-and-call fallback for a sink without a native hook). Network is mocked;
no external services required. Skipped if apache-beam / pyarrow are absent.
"""
import glob

import pytest

beam = pytest.importorskip("apache_beam")
pq = pytest.importorskip("pyarrow.parquet")

from connectors import build  # noqa: E402
from connectors.base import SinkConnector  # noqa: E402
from connectors.engines.beam import BeamEngine  # noqa: E402

_FAKE_RESULT = [
    {"metric": {"__name__": "cpu", "instance": "pod-a"}, "values": [[0, "1.0"], [15, "1.1"]]},
    {"metric": {"__name__": "mem", "instance": "pod-a"}, "values": [[0, "5.0"], [15, "5.5"]]},
]


class _FileSink(SinkConnector):
    """Sink with NO native hook — exercises the Beam fallback path.

    Module-level (picklable) so DirectRunner can serialize it.
    """

    def __init__(self, path):
        self.path = path

    def write(self, rows):
        with open(self.path, "w") as f:
            for r in rows:
                f.write(f"{r.group_id}\n")


@pytest.fixture
def mimir_source(monkeypatch):
    monkeypatch.setattr(
        "connectors.sources.mimir.query_range", lambda *a, **k: _FAKE_RESULT
    )
    return build(
        "mimir", endpoint="http://mimir:9009", promql="up", start=0, end=30, step_s=15
    )


def test_beam_engine_mimir_to_parquet_native(mimir_source, tmp_path):
    # ParquetSink provides native_beam_write -> distributed WriteToParquet
    sink = build("parquet", path=str(tmp_path / "e2e"), num_shards=1)
    BeamEngine().run(mimir_source, [sink])

    files = sorted(glob.glob(str(tmp_path / "e2e*")))
    assert files, "native Parquet writer produced no file"
    table = pq.read_table(files[0]).to_pydict()
    assert len(table["group_id"]) == 2
    assert table["channels"][0] == ["cpu", "mem"]
    assert sorted(table["values"]) == [[1.0, 5.0], [1.1, 5.5]]


def test_beam_engine_fallback_to_agnostic_write(mimir_source, tmp_path):
    # a sink without a native hook still runs under Beam via gather-and-call
    out = tmp_path / "fallback.txt"
    BeamEngine().run(mimir_source, [_FileSink(str(out))])

    lines = out.read_text().splitlines()
    assert lines == ["pod-a", "pod-a"]
