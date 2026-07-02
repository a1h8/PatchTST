"""M6 runner selection + throughput monitoring — config-driven Beam engine.

Covers the runner-agnostic slice validated on the DirectRunner: alias -> Beam
runner options (incl. Dataflow settings), config wiring into ``BeamEngine``, and
the throughput counters read off the ``PipelineResult``. The real Dataflow /
Flink-on-K8s submissions need external infra and are exercised out of band.
"""
import pytest

beam = pytest.importorskip("apache_beam")

from apache_beam.options.pipeline_options import (  # noqa: E402
    GoogleCloudOptions,
    StandardOptions,
)
from apache_beam.metrics.metric import MetricsFilter  # noqa: E402

from connectors import build  # noqa: E402
from connectors.base import SinkConnector  # noqa: E402
from connectors.engines.beam import BeamEngine  # noqa: E402
from connectors.engines.runner import beam_pipeline_options  # noqa: E402
from pipeline.runner import build_engine  # noqa: E402

_FAKE_RESULT = [
    {"metric": {"__name__": "cpu", "instance": "pod-a"},
     "values": [[0, "1.0"], [15, "1.1"]]},
    {"metric": {"__name__": "mem", "instance": "pod-a"},
     "values": [[0, "5.0"], [15, "5.5"]]},
]


class _NullSink(SinkConnector):
    """Discards rows; picklable for DirectRunner serialization."""

    def write(self, rows):
        for _ in rows:
            pass


# -- runner options ---------------------------------------------------------
def test_options_map_dataflow_alias_and_settings():
    opts = beam_pipeline_options(
        "dataflow",
        streaming=True,
        options={
            "project": "proj",
            "region": "europe-west1",
            "temp_location": "gs://bucket/tmp",
        },
    )
    assert opts.view_as(StandardOptions).runner == "DataflowRunner"
    assert opts.view_as(StandardOptions).streaming is True
    gco = opts.view_as(GoogleCloudOptions)
    assert gco.project == "proj"
    assert gco.region == "europe-west1"
    assert gco.temp_location == "gs://bucket/tmp"


def test_options_default_is_direct_batch():
    opts = beam_pipeline_options()
    assert opts.view_as(StandardOptions).runner == "DirectRunner"
    assert opts.view_as(StandardOptions).streaming is False


def test_options_map_portable_alias():
    """Flink-on-K8s path: portable runner + standalone job server endpoint."""
    from apache_beam.options.pipeline_options import PortableOptions

    opts = beam_pipeline_options(
        "portable",
        streaming=True,
        options={"job_endpoint": "beam-job-server.patchtst.svc:8099"},
    )
    assert opts.view_as(StandardOptions).runner == "PortableRunner"
    assert (
        opts.view_as(PortableOptions).job_endpoint
        == "beam-job-server.patchtst.svc:8099"
    )


def test_options_reject_unknown_runner():
    with pytest.raises(KeyError):
        beam_pipeline_options("spark")


def test_options_list_value_emits_repeated_flag():
    """A list setting (e.g. experiments) becomes a repeatable Beam flag."""
    from apache_beam.options.pipeline_options import DebugOptions

    opts = beam_pipeline_options(
        "dataflow",
        options={"experiments": ["use_runner_v2", "enable_streaming_engine"]},
    )
    experiments = opts.view_as(DebugOptions).experiments
    assert "use_runner_v2" in experiments
    assert "enable_streaming_engine" in experiments


# -- config wiring ----------------------------------------------------------
def test_build_engine_wires_streaming_window_and_runner():
    eng = build_engine(
        {
            "type": "beam",
            "runner": "dataflow",
            "streaming": True,
            "window": {"size_s": 30, "period_s": 15, "early_firing_count": 2},
            "options": {
                "project": "p",
                "region": "r",
                "temp_location": "gs://b/t",
            },
        }
    )
    assert isinstance(eng, BeamEngine)
    assert eng._streaming is True
    assert eng._window.size_s == 30
    assert eng._window.early_firing_count == 2
    assert eng._options.view_as(StandardOptions).runner == "DataflowRunner"


def test_build_engine_beam_defaults_to_direct_batch():
    eng = build_engine({"type": "beam"})
    assert isinstance(eng, BeamEngine)
    assert eng._streaming is False
    assert eng._options.view_as(StandardOptions).runner == "DirectRunner"


def test_shipped_dataflow_example_config_builds():
    """The committed Dataflow example stays in sync with build_engine."""
    from pathlib import Path

    from pipeline.runner import load_config

    cfg = load_config(
        str(
            Path(__file__).resolve().parent.parent
            / "config"
            / "dataflow-streaming.example.yaml"
        )
    )
    eng = build_engine(cfg["engine"])
    assert isinstance(eng, BeamEngine)
    assert eng._streaming is True
    assert eng._window.size_s == 300
    assert eng._window.early_firing_count == 1
    assert eng._options.view_as(StandardOptions).runner == "DataflowRunner"


def test_shipped_flink_example_config_builds():
    """The committed Flink-on-K8s example stays in sync with build_engine."""
    from pathlib import Path

    from apache_beam.options.pipeline_options import PortableOptions
    from pipeline.runner import load_config

    cfg = load_config(
        str(
            Path(__file__).resolve().parent.parent
            / "config"
            / "flink-streaming.example.yaml"
        )
    )
    eng = build_engine(cfg["engine"])
    assert isinstance(eng, BeamEngine)
    assert eng._streaming is True
    assert eng._options.view_as(StandardOptions).runner == "PortableRunner"
    assert (
        eng._options.view_as(PortableOptions).job_endpoint
        == "beam-job-server.patchtst.svc:8099"
    )


# -- monitoring -------------------------------------------------------------
def _counters(result):
    got = result.metrics().query(MetricsFilter().with_namespace("pipeline"))
    return {m.key.metric.name: m.committed for m in got["counters"]}


def test_throughput_counters_reported_on_result(monkeypatch):
    """rows_in / records_out land on the PipelineResult after a bounded run.

    Uses the bounded Mimir source: committed metrics aggregate reliably on the
    batch DirectRunner (streaming DirectRunner does not fully report committed
    metrics — those are read from the live job on a real runner).
    """
    monkeypatch.setattr(
        "connectors.sources.mimir.query_range", lambda *a, **k: _FAKE_RESULT
    )
    source = build(
        "mimir", endpoint="http://m", promql="up", start=0, end=30, step_s=15
    )
    result = BeamEngine().run(source, [_NullSink()])
    counters = _counters(result)
    assert counters["rows_in"] == 2  # 2 timestamps -> 2 multivariate rows
    assert counters["records_out"] == 2  # passthrough, no transform