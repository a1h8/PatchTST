"""OTLP source (connector C2) — a live push metrics receiver.

Unlike Mimir (pull, batch query) and Kafka (pull, consumer poll), OTLP is a
*push* protocol: an OpenTelemetry Collector or SDK exporter POSTs metrics to
us. This connector embeds a stdlib-only OTLP/HTTP receiver — ``http.server``,
JSON body (the OTLP/HTTP JSON mapping, no ``opentelemetry-proto``/protobuf
dependency needed) — matching the project's "stdlib-only and unit-testable
without an engine install" pattern already used by ``mimir.py``.

Alignment (decision D4): one export request commonly bundles every metric for
a resource at the same collection instant (an OTel Collector's batch
processor), so channels are grouped by resource attributes and aligned with
the *same* ``align_group`` helper Mimir's batch query uses — just applied per
push instead of per historical query result.

Only gauge/sum data points (``asDouble``/``asInt``) are supported; histogram
and summary points are skipped (documented, not silently misparsed).

**No ``native_beam_read``.** An HTTP receiver is a stateful, long-lived,
single-port process — it does not fit Beam's per-bundle, horizontally-scaled
worker model the way an unbounded Kafka *consumer* does. Feeding Beam directly
from OTLP push is not a supported shape here; front it with an OTel Collector
exporting to Kafka/Pub-Sub (this repo's Kafka source, C7) for the distributed
streaming path instead. ``read()``'s bounded drain is for the ``LocalEngine``
or a dedicated single-process ingestion daemon.
"""
from __future__ import annotations

import http.server
import json
import queue
import threading
import time
from collections import defaultdict
from typing import Any

from ..alignment import FillPolicy, align_group
from ..base import SourceConnector
from ..pivot import PivotRow
from ..registry import connector


def _attr_map(attributes: list[dict]) -> dict[str, str]:
    """OTLP ``KeyValue[]`` -> a flat ``{key: str(value)}`` map."""
    out: dict[str, str] = {}
    for attr in attributes or []:
        v = attr.get("value", {})
        val = v.get(
            "stringValue",
            v.get("intValue", v.get("doubleValue", v.get("boolValue"))),
        )
        out[attr["key"]] = "" if val is None else str(val)
    return out


def _data_point_value(dp: dict) -> float | None:
    if "asDouble" in dp:
        return float(dp["asDouble"])
    if "asInt" in dp:
        return float(dp["asInt"])
    return None  # histogram/summary points: not supported


def parse_export_request(
    body: dict,
    *,
    group_by: list[str],
    step_ms: int,
    fill: FillPolicy,
) -> list[PivotRow]:
    """Turn one OTLP/HTTP JSON ``ExportMetricsServiceRequest`` into aligned rows.

    Mirrors ``mimir.to_pivot_rows``: series are grouped by ``group_by`` resource
    attribute values (-> group_id) and channel-named by metric name, then
    aligned onto a common grid with ``align_group``.
    """
    # group_id -> channel -> [(ts_ms, value)]
    groups: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    group_labels: dict[str, dict[str, str]] = {}

    for rm in body.get("resourceMetrics", []):
        resource_attrs = _attr_map(rm.get("resource", {}).get("attributes", []))
        gid = "/".join(resource_attrs.get(k, "") for k in group_by) or "default"
        group_labels.setdefault(gid, {k: resource_attrs.get(k, "") for k in group_by})

        for sm in rm.get("scopeMetrics", []):
            for metric in sm.get("metrics", []):
                channel = metric["name"]
                points = (
                    metric.get("gauge", {}).get("dataPoints")
                    or metric.get("sum", {}).get("dataPoints")
                    or []
                )
                for dp in points:
                    value = _data_point_value(dp)
                    if value is None:
                        continue
                    ts_ms = int(dp["timeUnixNano"]) // 1_000_000
                    groups[gid][channel].append((ts_ms, value))

    rows: list[PivotRow] = []
    for gid, series_map in groups.items():
        rows.extend(
            align_group(
                gid, series_map, step_ms=step_ms, fill=fill,
                labels=group_labels.get(gid),
            )
        )
    return rows


class _Handler(http.server.BaseHTTPRequestHandler):
    """Minimal OTLP/HTTP JSON receiver: parses, queues, acks with ``{}``."""

    def do_POST(self) -> None:
        if self.path != self.server.otlp_path:  # type: ignore[attr-defined]
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw)
            rows = parse_export_request(
                body,
                group_by=self.server.group_by,  # type: ignore[attr-defined]
                step_ms=self.server.step_ms,  # type: ignore[attr-defined]
                fill=self.server.fill,  # type: ignore[attr-defined]
            )
            self.server.row_queue.put(rows)  # type: ignore[attr-defined]
            status, payload = 200, b"{}"
        except (ValueError, KeyError, TypeError):
            # ValueError: malformed JSON or a non-numeric timeUnixNano/asInt;
            # KeyError: a required field (metric name, attribute key) missing.
            status, payload = 400, b'{"error":"bad OTLP export request"}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # keep stdout clean; wire to `logging` if request-level audit is needed


@connector("otlp")
class OTLPSource(SourceConnector):
    """Live push source: an embedded OTLP/HTTP (JSON) metrics receiver."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 4318,
        path: str = "/v1/metrics",
        group_by: list[str] | None = None,
        step_ms: int = 15_000,
        fill: FillPolicy = "ffill",
        poll_timeout_s: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.path = path
        self.group_by = group_by or ["service.name"]
        self.step_ms = step_ms
        self.fill = fill
        self.poll_timeout_s = poll_timeout_s
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue[list[PivotRow]] = queue.Queue()

    def _ensure_server(self) -> None:
        if self._server is not None:
            return
        server = http.server.ThreadingHTTPServer((self.host, self.port), _Handler)
        server.otlp_path = self.path  # type: ignore[attr-defined]
        server.group_by = self.group_by  # type: ignore[attr-defined]
        server.step_ms = self.step_ms  # type: ignore[attr-defined]
        server.fill = self.fill  # type: ignore[attr-defined]
        server.row_queue = self._queue  # type: ignore[attr-defined]
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        self.port = server.server_address[1]  # resolve an OS-assigned (port=0) port

    def read(self) -> list[PivotRow]:
        # Engine-agnostic: bounded drain of whatever was pushed within
        # poll_timeout_s, starting the receiver on first call. See the module
        # docstring for why there is no native_beam_read.
        self._ensure_server()
        rows: list[PivotRow] = []
        deadline = time.monotonic() + self.poll_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                rows.extend(self._queue.get(timeout=remaining))
            except queue.Empty:
                break
        return rows

    def stop(self) -> None:
        """Shut down the embedded receiver (tests / graceful process exit)."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        assert self._thread is not None
        self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "host": self.host,
            "port": self.port,
            "path": self.path,
            "group_by": self.group_by,
        }
