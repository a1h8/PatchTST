"""SignalRecord — one aggregated signal in the knowledge base.

Schema deliberately aligned with kube-verdict's ``AnomalyResult`` so the
knowledge base speaks its language: kube-verdict's RCA (``rca/context_builder``)
queries this store by entity/metric/window as historical evidence. See
docs/ARCHITECTURE.md (D7).

This is the *aggregated output* schema, distinct from the raw-metric
``connectors.PivotRow`` input schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SEVERITIES = ("normal", "warning", "critical")
HORIZONS = ("", "short", "medium", "long")


@dataclass(frozen=True, slots=True)
class SignalRecord:
    entity_uid: str            # e.g. "Pod/prod/api-7c9"
    metric_name: str           # e.g. "cpu_usage"
    ts: int                    # epoch milliseconds the signal was observed
    severity: str              # normal | warning | critical
    score: float               # anomaly score (>= warning_threshold = anomalous)
    method: str                # "patchtst" | "zscore"
    horizon: str = ""          # short | medium | long | ""
    n_points: int = 0
    labels: dict[str, str] = field(default_factory=dict)
    text: str = ""             # narrative; embedded later for semantic lookup

    def __post_init__(self) -> None:
        if not self.entity_uid:
            raise ValueError("entity_uid must be non-empty")
        if not isinstance(self.ts, int):
            raise ValueError(f"ts must be int epoch-ms, got {type(self.ts).__name__}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")
        if self.horizon not in HORIZONS:
            raise ValueError(f"horizon must be one of {HORIZONS}, got {self.horizon!r}")

    @property
    def is_anomalous(self) -> bool:
        return self.severity != "normal"

    def to_text(self) -> str:
        """Narrative form (mirrors kube-verdict AnomalyResult.to_text) for the
        semantic/vector face of the knowledge base."""
        if self.text:
            return self.text
        horizon_part = f" horizon={self.horizon}" if self.horizon else ""
        return (
            f"signal metric={self.metric_name} entity={self.entity_uid}"
            f"{horizon_part} severity={self.severity} score={self.score:.3f} "
            f"method={self.method} n={self.n_points}"
        )

    @classmethod
    def from_anomaly_result(cls, result, *, ts: int, labels=None) -> "SignalRecord":
        """Build from a kube-verdict ``AnomalyResult`` (duck-typed)."""
        return cls(
            entity_uid=result.entity_uid,
            metric_name=result.metric_name,
            ts=ts,
            severity=result.severity,
            score=float(result.score),
            method=result.method,
            horizon=getattr(result, "horizon", "") or "",
            n_points=int(getattr(result, "n_points", 0)),
            labels=dict(labels or {}),
            text=result.to_text() if hasattr(result, "to_text") else "",
        )
