"""Engine port — the boundary between the agnostic core and a runtime.

An engine knows how to execute a ``source → sinks`` flow on a concrete runtime
(plain Python, Beam, Spark/Databricks). Connectors never depend on it; it depends
on connectors. This is the "ports & adapters" hexagonal boundary that lets the
same pipeline plug into any engine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..base import SinkConnector, SourceConnector


class Engine(ABC):
    """Runs a source into one or more sinks on a concrete runtime."""

    @abstractmethod
    def run(
        self, source: SourceConnector, sinks: Sequence[SinkConnector]
    ) -> None:
        """Read from ``source`` and write to every sink in ``sinks``."""
