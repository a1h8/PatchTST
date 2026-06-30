"""Streaming integration tests — BeamEngine windowed path on the DirectRunner.

Exercises the M5 streaming mode end-to-end with the synthetic unbounded source
(a ``TestStream`` under the hood): sliding windows, watermark-driven triggering,
per-(entity, window) detection, and the allowed-lateness gate on late data. No
external services; skipped if apache-beam is absent.
"""
import pytest

beam = pytest.importorskip("apache_beam")

from connectors import build  # noqa: E402
from connectors.base import SinkConnector  # noqa: E402
from connectors.engines.beam import BeamEngine, WindowSpec  # noqa: E402
from detection import make_detection_transform  # noqa: E402
from detection.detector import ZScoreDetector  # noqa: E402


class _CountingFileSink(SinkConnector):
    """No native hook -> exercises the streaming per-element fallback write.

    Appends one line per emitted SignalRecord so the test can count firings.
    Module-level + picklable for DirectRunner serialization.
    """

    def __init__(self, path):
        self.path = path

    def write(self, rows):
        with open(self.path, "a") as f:
            for r in rows:
                f.write(f"{r.entity_uid}|{r.metric_name}\n")


def _run(source, sink, window):
    BeamEngine(streaming=True, window=window).run(
        source,
        [sink],
        transform=make_detection_transform(ZScoreDetector()),
    )


def _lines(path):
    return path.read_text().splitlines() if path.exists() else []


def test_streaming_sliding_windows_emit_more_than_batch(tmp_path):
    """Sliding windows split the stream: more firings than a single batch pass."""
    out = tmp_path / "stream.txt"
    source = build("synthetic-stream", n_points=8, step_s=15)
    _run(source, _CountingFileSink(str(out)), WindowSpec(size_s=60, period_s=30))

    lines = _lines(out)
    # One ZScore signal per (entity, metric) per window -> several overlapping
    # windows over 8 points => strictly more than the single batch verdict.
    assert len(lines) > 1
    assert all(line == "g0|m0" for line in lines)


def test_synthetic_source_agnostic_read():
    """read() returns the full clean series, engine-free (no beam needed)."""
    src = build(
        "synthetic-stream",
        group_ids=["a", "b"],
        channels=["cpu", "mem"],
        n_points=4,
        step_s=10,
        anomaly_at=2,  # default anomaly magnitude (base * 10)
    )
    rows = src.read()
    assert len(rows) == 4 * 2  # n_points * groups
    assert {r.group_id for r in rows} == {"a", "b"}
    assert rows[0].channels == ("cpu", "mem")
    spike = [r for r in rows if r.ts == 2 * 10_000]
    assert all(v == 10.0 for r in spike for v in r.values)  # base(1.0) * 10
    assert src.describe()["type"] == "SyntheticStreamSource"


def test_streaming_early_firings_emit_speculative_panes(tmp_path):
    """early_firing_count adds speculative panes before the watermark closes."""
    watermark_only = tmp_path / "wm.txt"
    with_early = tmp_path / "early.txt"

    _run(
        build("synthetic-stream", n_points=8, step_s=15),
        _CountingFileSink(str(watermark_only)),
        WindowSpec(size_s=60, period_s=30),
    )
    _run(
        build("synthetic-stream", n_points=8, step_s=15),
        _CountingFileSink(str(with_early)),
        WindowSpec(size_s=60, period_s=30, early_firing_count=1),
    )

    # Same stream, same windows: speculative early panes => strictly more
    # firings than the watermark-only policy.
    assert len(_lines(with_early)) > len(_lines(watermark_only))


def test_windowspec_rejects_invalid_policy():
    """Bad triggering/accumulation config fails fast, not silently."""
    with pytest.raises(ValueError):
        WindowSpec(accumulation="nope")
    with pytest.raises(ValueError):
        WindowSpec(late_firing_count=0)
    with pytest.raises(ValueError):
        WindowSpec(early_firing_count=0)


def test_streaming_late_data_gated_by_allowed_lateness(tmp_path):
    """A late element fires an extra pane only when within allowed lateness."""
    dropped = tmp_path / "drop.txt"
    accepted = tmp_path / "accept.txt"

    _run(
        build("synthetic-stream", n_points=8, step_s=15, late_at=2),
        _CountingFileSink(str(dropped)),
        WindowSpec(size_s=60, period_s=30, allowed_lateness_s=0),
    )
    _run(
        build("synthetic-stream", n_points=8, step_s=15, late_at=2),
        _CountingFileSink(str(accepted)),
        WindowSpec(size_s=60, period_s=30, allowed_lateness_s=3600),
    )

    # The late point is dropped with zero lateness, admitted (extra late panes)
    # with generous lateness.
    assert len(_lines(accepted)) > len(_lines(dropped))