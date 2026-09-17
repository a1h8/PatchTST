"""OTLP source (connector C2) tests.

``parse_export_request`` is pure stdlib and tested directly against the
OTLP/HTTP JSON mapping, no server involved. The embedded receiver itself is
exercised with one real HTTP round-trip against an OS-assigned ephemeral port
(``port=0``), the same "small, real, cleaned-up" spirit as binding a temp file
for a filesystem test — no mocking of ``http.server`` needed since it's stdlib
and fast.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import connectors  # noqa: F401  (triggers built-in registration)
from connectors.conformance import (
    assert_buildable,
    assert_registered,
    assert_source_contract,
)
from connectors.pivot import PivotRow
from connectors.registry import available
from connectors.sources.otlp import OTLPSource, parse_export_request

# --- registry / conformance --------------------------------------------------

def test_otlp_registered():
    assert available().get("otlp") == "source"


def test_otlp_conforms_to_source_contract():
    assert_registered("otlp")
    src = assert_buildable("otlp", host="127.0.0.1", port=0)
    assert_source_contract(src)


def test_otlp_describe_defaults():
    src = OTLPSource()
    d = src.describe()
    assert d["kind"] == "source"
    assert d["path"] == "/v1/metrics"
    assert d["group_by"] == ["service.name"]


# --- parse_export_request: OTLP/HTTP JSON mapping ----------------------------

def _export_body(*, service: str, points: list[tuple[str, int, float]]) -> dict:
    """Build one ExportMetricsServiceRequest with `points` = (metric, ts_ns, value)."""
    return {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": service}}]
                },
                "scopeMetrics": [
                    {
                        "metrics": [
                            {
                                "name": name,
                                "gauge": {
                                    "dataPoints": [
                                        {"timeUnixNano": str(ts_ns), "asDouble": value}
                                    ]
                                },
                            }
                            for name, ts_ns, value in points
                        ]
                    }
                ],
            }
        ]
    }


def test_parse_export_request_groups_by_resource_and_aligns():
    body = _export_body(
        service="svc-a",
        points=[("cpu", 15_000_000_000, 1.5), ("mem", 15_000_000_000, 5.0)],
    )
    rows = parse_export_request(body, group_by=["service.name"], step_ms=15_000, fill="ffill")

    assert len(rows) == 1
    row = rows[0]
    assert row.group_id == "svc-a"
    assert row.ts == 15_000
    assert row.channels == ("cpu", "mem")
    assert row.values == (1.5, 5.0)
    assert row.labels == {"service.name": "svc-a"}


def test_parse_export_request_sum_as_int():
    body = {
        "resourceMetrics": [
            {
                "resource": {"attributes": []},
                "scopeMetrics": [
                    {
                        "metrics": [
                            {
                                "name": "requests_total",
                                "sum": {
                                    "dataPoints": [
                                        {"timeUnixNano": "30000000000", "asInt": "42"}
                                    ]
                                },
                            }
                        ]
                    }
                ],
            }
        ]
    }
    rows = parse_export_request(body, group_by=["service.name"], step_ms=15_000, fill="ffill")
    assert rows == [
        PivotRow("default", 30_000, (42.0,), ("requests_total",), {"service.name": ""})
    ]


def test_parse_export_request_skips_unsupported_point_types():
    body = {
        "resourceMetrics": [
            {
                "resource": {"attributes": []},
                "scopeMetrics": [
                    {
                        "metrics": [
                            {
                                "name": "latency_histogram",
                                "histogram": {"dataPoints": [{"timeUnixNano": "0"}]},
                            }
                        ]
                    }
                ],
            }
        ]
    }
    assert parse_export_request(body, group_by=["service.name"], step_ms=1000, fill="ffill") == []


def test_parse_export_request_empty_body():
    assert parse_export_request({}, group_by=["service.name"], step_ms=1000, fill="ffill") == []


# --- embedded receiver: one real HTTP round-trip -----------------------------

def test_receiver_accepts_push_and_read_drains_it():
    src = OTLPSource(host="127.0.0.1", port=0, poll_timeout_s=0.3)
    try:
        src._ensure_server()
        port = src.describe()["port"]
        assert port != 0

        body = _export_body(service="svc-b", points=[("cpu", 60_000_000_000, 2.0)])
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/metrics",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            assert json.loads(resp.read()) == {}

        rows = src.read()
        assert rows == [PivotRow("svc-b", 60_000, (2.0,), ("cpu",), {"service.name": "svc-b"})]
    finally:
        src.stop()


def test_receiver_rejects_unknown_path_with_404():
    src = OTLPSource(host="127.0.0.1", port=0, path="/v1/metrics", poll_timeout_s=0.2)
    try:
        src._ensure_server()
        port = src.describe()["port"]
        req = urllib.request.Request(f"http://127.0.0.1:{port}/nope", method="POST", data=b"{}")
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected HTTPError")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        src.stop()


def test_read_returns_empty_when_nothing_pushed():
    src = OTLPSource(host="127.0.0.1", port=0, poll_timeout_s=0.2)
    try:
        assert src.read() == []
    finally:
        src.stop()
