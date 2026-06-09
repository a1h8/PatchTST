"""Integration tests — real Beam pipelines on the DirectRunner.

Exercises the beam-dependent connector paths (``MimirSource.read`` and
``ParquetSink.write``) end-to-end, including a source -> sink round trip.
Network is mocked; no external services required.

Skipped automatically if apache-beam / pyarrow are not installed.
"""
import glob
import json

import pytest

beam = pytest.importorskip("apache_beam")
pq = pytest.importorskip("pyarrow.parquet")
# aliased to avoid pytest trying to collect it as a test class
from apache_beam.testing.test_pipeline import TestPipeline as Pipeline  # noqa: E402
from apache_beam.testing.util import assert_that, equal_to  # noqa: E402

from connectors import build  # noqa: E402
from connectors.pivot import PivotRow  # noqa: E402

_FAKE_RESULT = [
    {
        "metric": {"__name__": "cpu", "instance": "pod-a"},
        "values": [[0, "1.0"], [15, "1.1"]],
    },
    {
        "metric": {"__name__": "mem", "instance": "pod-a"},
        "values": [[0, "5.0"], [15, "5.5"]],
    },
]


@pytest.fixture
def mimir_source(monkeypatch):
    monkeypatch.setattr(
        "connectors.sources.mimir.query_range", lambda *a, **k: _FAKE_RESULT
    )
    return build(
        "mimir",
        endpoint="http://mimir:9009",
        promql="up",
        start=0,
        end=30,
        step_s=15,
    )


def test_mimir_source_read_pipeline(mimir_source):
    expected = [
        PivotRow("pod-a", 0, (1.0, 5.0), ("cpu", "mem"), {"instance": "pod-a"}),
        PivotRow("pod-a", 15_000, (1.1, 5.5), ("cpu", "mem"), {"instance": "pod-a"}),
    ]
    with Pipeline() as p:
        out = p | mimir_source.read()
        assert_that(out, equal_to(expected))


def test_parquet_sink_roundtrip(tmp_path):
    sink = build("parquet", path=str(tmp_path / "out"), num_shards=1)
    rows = [
        PivotRow("pod-a", 0, (1.0, 5.0), ("cpu", "mem"), {"k": "v"}),
        PivotRow("pod-a", 15_000, (1.1, 5.5), ("cpu", "mem"), {"k": "v"}),
    ]
    with Pipeline() as p:
        p | beam.Create(rows) | sink.write()

    files = sorted(glob.glob(str(tmp_path / "out*")))
    assert files, "no parquet output written"
    table = pq.read_table(files[0]).to_pydict()
    assert table["group_id"] == ["pod-a", "pod-a"]
    assert table["values"] == [[1.0, 5.0], [1.1, 5.5]]
    assert json.loads(table["labels"][0]) == {"k": "v"}


def test_mimir_to_parquet_e2e(mimir_source, tmp_path):
    sink = build("parquet", path=str(tmp_path / "e2e"), num_shards=1)
    with Pipeline() as p:
        p | mimir_source.read() | sink.write()

    files = sorted(glob.glob(str(tmp_path / "e2e*")))
    assert files, "no parquet output written"
    table = pq.read_table(files[0]).to_pydict()
    assert len(table["group_id"]) == 2
    assert table["channels"][0] == ["cpu", "mem"]
    assert sorted(table["values"]) == [[1.0, 5.0], [1.1, 5.5]]
