"""KubeVerdict alerting sink (M7) — push notable signals to kube-verdict.

The complement to the knowledge-base store. ``signal-store`` persists *every*
``SignalRecord`` as longitudinal RCA evidence (the D7 pull path kube-verdict
queries); this sink *pushes* the anomalous ones (severity >= ``min_severity`` —
the NORMAL→INCIDENT verdicts) to kube-verdict's alert webhook as they happen. We
forward our own aggregated signals; we never run kube-verdict's own detector
(see docs/ARCHITECTURE.md, and the kube-verdict relationship note).

Outbound HTTP is stdlib ``urllib`` (consistent with the Mimir source), so the
sink adds no runtime dependency. Best-effort by default: a webhook failure is
logged, never fatal to the pipeline — the KB store stays the source of truth.
Set ``raise_on_error=True`` to surface failures instead.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Iterable

from connectors.base import SinkConnector
from connectors.registry import connector

from .signal import SEVERITIES, SignalRecord

# normal < warning < critical
_SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}

_log = logging.getLogger(__name__)


def post_alerts(
    endpoint: str,
    alerts: list[dict],
    *,
    token: str | None = None,
    timeout: float = 10.0,
) -> int:
    """POST a batch of alert dicts to the kube-verdict webhook; return status.

    Separate from the connector so it can be mocked in tests (mirrors
    ``connectors.sources.mimir.query_range``).
    """
    body = json.dumps({"alerts": alerts}).encode()
    req = urllib.request.Request(endpoint, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return getattr(resp, "status", 200)


@connector("kubeverdict-alert")
class KubeVerdictAlertSink(SinkConnector):
    """Forward anomalous ``SignalRecord``s to kube-verdict's alert webhook."""

    def __init__(
        self,
        endpoint: str,
        *,
        min_severity: str = "warning",
        token: str | None = None,
        timeout: float = 10.0,
        raise_on_error: bool = False,
    ) -> None:
        if min_severity not in SEVERITIES:
            raise ValueError(
                f"min_severity must be one of {SEVERITIES}, got {min_severity!r}"
            )
        self.endpoint = endpoint
        self.min_severity = min_severity
        self.token = token
        self.timeout = timeout
        self.raise_on_error = raise_on_error

    def _alert(self, r: SignalRecord) -> dict:
        # Shape mirrors kube-verdict's AnomalyResult so the alert speaks its language.
        return {
            "entity_uid": r.entity_uid,
            "metric_name": r.metric_name,
            "ts": r.ts,
            "severity": r.severity,
            "score": r.score,
            "method": r.method,
            "horizon": r.horizon,
            "n_points": r.n_points,
            "labels": r.labels,
            "text": r.to_text(),
        }

    def write(self, rows: Iterable[SignalRecord]) -> None:
        threshold = _SEVERITY_RANK[self.min_severity]
        alerts = [
            self._alert(r)
            for r in rows
            if _SEVERITY_RANK.get(r.severity, 0) >= threshold
        ]
        if not alerts:
            return  # nothing worth alerting on; the KB still has the full history
        try:
            post_alerts(
                self.endpoint, alerts, token=self.token, timeout=self.timeout
            )
        except Exception:
            _log.exception(
                "kube-verdict alert POST to %s failed (%d alert(s) dropped)",
                self.endpoint,
                len(alerts),
            )
            if self.raise_on_error:
                raise

    def describe(self) -> dict:
        return {
            **super().describe(),
            "endpoint": self.endpoint,
            "min_severity": self.min_severity,
        }
