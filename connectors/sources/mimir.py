"""Grafana Mimir source (connector C9a).

Reads historical metrics via Mimir's Prometheus-compatible ``query_range`` API
and emits aligned multivariate ``PivotRow`` records. This connector owns the
alignment cost (decision D4): it groups raw series by ``group_by`` labels and
aligns their channels onto a common grid before emitting.

The HTTP query is stdlib-only and unit-testable; Beam is imported lazily in
``read`` so importing this module does not require a Beam install.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections import defaultdict

from ..alignment import FillPolicy, align_group
from ..base import SourceConnector
from ..pivot import PivotRow
from ..registry import connector


def query_range(
    endpoint: str,
    promql: str,
    start: int,
    end: int,
    step_s: int,
    *,
    tenant: str | None = None,
    timeout: float = 30.0,
) -> list[dict]:
    """Call Mimir/Prometheus ``/api/v1/query_range``; return the result list.

    Args separate from the connector so this can be mocked in tests.
    """
    params = urllib.parse.urlencode(
        {"query": promql, "start": start, "end": end, "step": f"{step_s}s"}
    )
    url = f"{endpoint.rstrip('/')}/prometheus/api/v1/query_range?{params}"
    req = urllib.request.Request(url)
    if tenant:
        req.add_header("X-Scope-OrgID", tenant)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        payload = json.loads(resp.read())
    if payload.get("status") != "success":
        raise RuntimeError(f"Mimir query failed: {payload.get('error', payload)}")
    return payload["data"]["result"]


def to_pivot_rows(
    result: list[dict],
    *,
    group_by: list[str],
    channel_label: str,
    step_ms: int,
    fill: FillPolicy,
) -> list[PivotRow]:
    """Turn a Prometheus matrix result into aligned multivariate rows.

    Series are grouped by the ``group_by`` label values (-> group_id); the
    ``channel_label`` value names each channel within a group.
    """
    # group_id -> channel -> [(ts_ms, value)]
    groups: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    group_labels: dict[str, dict[str, str]] = {}
    for series in result:
        metric = series.get("metric", {})
        gid = "/".join(metric.get(k, "") for k in group_by) or "default"
        channel = metric.get(channel_label, metric.get("__name__", "value"))
        group_labels.setdefault(gid, {k: metric.get(k, "") for k in group_by})
        for ts_s, val in series.get("values", []):
            groups[gid][channel].append((int(float(ts_s) * 1000), float(val)))

    rows: list[PivotRow] = []
    for gid, series_map in groups.items():
        rows.extend(
            align_group(
                gid,
                series_map,
                step_ms=step_ms,
                fill=fill,
                labels=group_labels.get(gid),
            )
        )
    return rows


@connector("mimir")
class MimirSource(SourceConnector):
    """Historical multivariate source backed by Grafana Mimir."""

    def __init__(
        self,
        endpoint: str,
        promql: str,
        start: int,
        end: int,
        *,
        step_s: int = 15,
        group_by: list[str] | None = None,
        channel_label: str = "__name__",
        fill: FillPolicy = "ffill",
        tenant: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.promql = promql
        self.start = start
        self.end = end
        self.step_s = step_s
        self.group_by = group_by or ["instance"]
        self.channel_label = channel_label
        self.fill = fill
        self.tenant = tenant

    def _fetch_rows(self) -> list[PivotRow]:
        result = query_range(
            self.endpoint,
            self.promql,
            self.start,
            self.end,
            self.step_s,
            tenant=self.tenant,
        )
        return to_pivot_rows(
            result,
            group_by=self.group_by,
            channel_label=self.channel_label,
            step_ms=self.step_s * 1000,
            fill=self.fill,
        )

    def read(self):
        import apache_beam as beam  # lazy: keep module importable without beam

        # Batch/replay: materialize the query, then fan out as a PCollection.
        # Streaming live ingestion is a separate connector (M5).
        return beam.Create(self._fetch_rows())

    def describe(self):
        return {
            **super().describe(),
            "endpoint": self.endpoint,
            "promql": self.promql,
            "group_by": self.group_by,
            "step_s": self.step_s,
        }