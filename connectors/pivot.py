"""Pivot schema — the only data contract the processing core understands.

Native multivariate (decision D4): a row carries an aligned vector of channel
values at a single timestamp. The source connector is responsible for producing
already-aligned vectors (see ``connectors.alignment``); the core never sees
unaligned data.

Note on the model mismatch (intentional, documented): PatchTST is
channel-independent, so it will not exploit cross-channel correlation carried in
``values``. The grouping buys batching and a group-level detection decision, not
joint modeling. See docs/ARCHITECTURE.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PivotRow:
    """One aligned multivariate sample.

    Attributes:
        group_id: stable identity of the channel group (e.g. a pod/node/service).
        ts: epoch milliseconds, snapped to the alignment grid by the source.
        values: channel values, ordered to match ``channels`` position-for-position.
        channels: ordered channel names; must be stable for a given ``group_id``.
        labels: free-form metadata (e.g. Prometheus labels), not used by the core.
    """

    group_id: str
    ts: int
    values: tuple[float, ...]
    channels: tuple[str, ...]
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must be non-empty")
        if not isinstance(self.ts, int):
            raise ValueError(f"ts must be int epoch-ms, got {type(self.ts).__name__}")
        if len(self.values) != len(self.channels):
            raise ValueError(
                f"values/channels length mismatch: "
                f"{len(self.values)} != {len(self.channels)}"
            )
        if len(set(self.channels)) != len(self.channels):
            raise ValueError(f"duplicate channels in {self.channels}")

    @property
    def width(self) -> int:
        """Number of channels (n_channels in the [n_channels, L] window)."""
        return len(self.channels)