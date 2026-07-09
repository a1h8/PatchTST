"""Offline pre-flight validation for a remote (Dataflow-first) submit — M6.

These run with no infra: they gate the config before a slow, billable job so the
boring first-submit failures (SDK/image version drift, missing options, a
local path a distributed worker cannot reach, a streaming graph that hangs the
driver) surface locally instead of on the runner.
"""
from pathlib import Path

import pytest

beam = pytest.importorskip("apache_beam")

from pipeline.preflight import (  # noqa: E402
    format_report,
    has_errors,
    preflight,
)

_ROOT = Path(__file__).resolve().parent.parent
_BEAM = beam.__version__


def _codes(issues, level=None):
    return {i.code for i in issues if level is None or i.level == level}


def _dataflow_config(**engine_over):
    """A well-formed streaming Dataflow config; override engine keys per-test."""
    engine = {
        "type": "beam",
        "runner": "dataflow",
        "streaming": True,
        "block": False,
        "window": {"size_s": 300, "period_s": 60},
        "options": {
            "project": "proj",
            "region": "europe-west1",
            "temp_location": "gs://b/tmp",
            "staging_location": "gs://b/staging",
            "sdk_container_image": f"eu-docker.pkg.dev/proj/p/worker:{_BEAM}",
            "experiments": ["use_runner_v2", "enable_streaming_engine"],
        },
    }
    # Shallow-merge overrides; a dict override replaces the whole sub-block.
    engine.update(engine_over)
    return {
        "source": {"type": "mimir", "params": {"endpoint": "http://m", "promql": "up",
                                                "start": 0, "end": 0, "step_s": 60}},
        "detector": {"type": "zscore"},
        "sinks": [{"type": "signal-store", "params": {"root": "s3://kb/kb"}}],
        "engine": engine,
    }


def test_wellformed_dataflow_config_has_no_errors():
    issues = preflight(_dataflow_config(), root=_ROOT)
    assert not has_errors(issues), format_report(issues)


def test_version_mismatch_is_error():
    cfg = _dataflow_config()
    cfg["engine"]["options"]["sdk_container_image"] = "repo/worker:0.0.0"
    issues = preflight(cfg, root=_ROOT)
    assert "beam-version-mismatch" in _codes(issues, "error")


def test_missing_required_option_is_error():
    cfg = _dataflow_config()
    del cfg["engine"]["options"]["temp_location"]
    issues = preflight(cfg, root=_ROOT)
    assert "missing-option" in _codes(issues, "error")


def test_non_gcs_location_is_error():
    cfg = _dataflow_config()
    cfg["engine"]["options"]["temp_location"] = "/local/tmp"
    issues = preflight(cfg, root=_ROOT)
    assert "non-gcs-location" in _codes(issues, "error")


def test_local_sink_root_is_error_on_remote_runner():
    cfg = _dataflow_config()
    cfg["sinks"][0]["params"]["root"] = "/var/data/kb"
    issues = preflight(cfg, root=_ROOT)
    assert "local-sink-root" in _codes(issues, "error")


def test_streaming_without_window_is_error():
    cfg = _dataflow_config()
    del cfg["engine"]["window"]
    issues = preflight(cfg, root=_ROOT)
    assert "streaming-no-window" in _codes(issues, "error")


def test_streaming_blocking_submit_is_warning():
    cfg = _dataflow_config(block=True)
    issues = preflight(cfg, root=_ROOT)
    assert "streaming-submit-blocks" in _codes(issues, "warn")
    assert not has_errors(issues)  # a warning, not a gate


def test_missing_runner_v2_experiment_is_warning():
    cfg = _dataflow_config()
    cfg["engine"]["options"]["experiments"] = ["enable_streaming_engine"]
    issues = preflight(cfg, root=_ROOT)
    assert "runner-v2-recommended" in _codes(issues, "warn")


def test_bad_detector_name_is_construction_error():
    cfg = _dataflow_config()
    cfg["detector"] = {"type": "no-such-detector"}
    issues = preflight(cfg, root=_ROOT)
    assert "detector-build" in _codes(issues, "error")


def test_local_runner_skips_remote_checks():
    """A local engine with a local sink root is fine — no distributed workers."""
    cfg = {
        "source": {"type": "mimir", "params": {"endpoint": "http://m", "promql": "up",
                                               "start": 0, "end": 0, "step_s": 60}},
        "detector": {"type": "zscore"},
        "sinks": [{"type": "signal-store", "params": {"root": "/local/kb"}}],
        "engine": {"type": "local"},
    }
    issues = preflight(cfg, root=_ROOT)
    assert not has_errors(issues), format_report(issues)


def test_shipped_dataflow_example_passes_preflight():
    """The committed example is aligned with the local beam SDK version.

    Skips only if the pinned image tag differs from the installed beam (e.g. a
    dev machine on a newer beam than the frozen 2.74.0 example) — that is the
    real drift the check exists to catch, not a test failure.
    """
    from pipeline.runner import load_config

    cfg = load_config(str(_ROOT / "config" / "dataflow-streaming.example.yaml"))
    image = cfg["engine"]["options"]["sdk_container_image"]
    if not image.endswith(f":{_BEAM}"):
        pytest.skip(f"example pinned to {image}, local beam is {_BEAM}")
    issues = preflight(cfg, root=_ROOT)
    assert not has_errors(issues), format_report(issues)
