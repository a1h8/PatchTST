"""Engine port — the boundary between the agnostic core and a runtime.

An engine knows how to execute a ``source → sinks`` flow on a concrete runtime
(plain Python, Beam, Spark/Databricks). Connectors never depend on it; it depends
on connectors. This is the "ports & adapters" hexagonal boundary that lets the
same pipeline plug into any engine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional, Sequence

from ..base import SinkConnector, SourceConnector

# An optional stage applied between read and write: rows in, records out
# (e.g. variation detection: Iterable[PivotRow] -> Iterable[SignalRecord]).
Transform = Callable[[Iterable[Any]], Iterable[Any]]


class Engine(ABC):
    """Runs a source → (transform) → sinks flow on a concrete runtime."""

    @abstractmethod
    def run(
        self,
        source: SourceConnector,
        sinks: Sequence[SinkConnector],
        transform: Optional[Transform] = None,
    ) -> None:
        """Read from ``source``, optionally ``transform``, write to every sink."""
