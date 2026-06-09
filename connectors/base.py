"""Connector SPI — engine-agnostic contract (M1.5, decision D6).

A connector is defined in pure-Python, domain terms only: a source *yields*
``PivotRow`` records, a sink *consumes* them. No execution engine appears in the
contract — Beam, Spark/Databricks, or a plain-Python runner are **adapters**
behind the boundary (see ``connectors.engines``), never baked into the core.

Why this shape:
- connectors are testable and runnable without any engine installed;
- the same connector runs under Beam, Spark/Databricks, or locally;
- swapping or adding an engine touches ``engines/``, never the connectors.

**Native capability hook.** A pure-iterator source cannot express an engine's
native distributed/unbounded I/O (e.g. a Kafka streaming source, or a parallel
Parquet writer). A connector may therefore expose an engine-native override; the
engine adapter uses it when present and falls back to the agnostic iterator
otherwise. Hooks return ``None`` by default and import their engine lazily, so
the core stays engine-free.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterable

from .pivot import PivotRow

if TYPE_CHECKING:  # never imported at runtime by the core
    import apache_beam as beam


class SourceConnector(ABC):
    """Yields ``PivotRow`` records. Engine-agnostic.

    Implementations own ingestion concerns the core must not know about:
    querying the backend, and (for multivariate) temporal alignment of
    heterogeneous channels onto a common grid before emitting rows.
    """

    kind = "source"

    @abstractmethod
    def read(self) -> Iterable[PivotRow]:
        """Produce pivot rows. Pure Python — no engine types."""

    def native_beam_read(self) -> "beam.PTransform | None":
        """Optional Beam-native source (e.g. unbounded Kafka).

        Return ``None`` (default) to let the Beam adapter wrap ``read()``.
        """
        return None

    def describe(self) -> dict[str, Any]:
        """Optional human/debug description of the configured source."""
        return {"kind": self.kind, "type": type(self).__name__}


class SinkConnector(ABC):
    """Consumes records (e.g. pivot rows or detections). Engine-agnostic."""

    kind = "sink"

    @abstractmethod
    def write(self, rows: Iterable[Any]) -> None:
        """Consume an iterable of records. Pure Python — no engine types."""

    def native_beam_write(self) -> "beam.PTransform | None":
        """Optional Beam-native sink (e.g. a parallel/distributed writer).

        Return ``None`` (default) to let the Beam adapter gather-and-call
        ``write()``.
        """
        return None

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "type": type(self).__name__}
