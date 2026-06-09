"""Connector SPI — the open extension point (M1.5).

Two contracts, one pivot schema. The core (windowing -> PatchTST -> detection)
depends only on these abstractions, never on a concrete connector. Adding a
connector means implementing one of these and registering it with
``@connector`` (see ``connectors.registry``).

``apache_beam`` is imported lazily inside ``read``/``write`` implementations so
that the pure-Python parts of this package (pivot, registry, alignment,
conformance) remain importable and testable without a Beam install.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid hard beam import at module load
    import apache_beam as beam


class SourceConnector(ABC):
    """Produces ``PivotRow`` records into the pipeline.

    Implementations own ingestion concerns the core must not know about:
    querying the backend, and (for multivariate) temporal alignment of
    heterogeneous channels onto a common grid before emitting rows.
    """

    kind = "source"

    @abstractmethod
    def read(self) -> "beam.PTransform":
        """Return a PTransform: () -> PCollection[PivotRow]."""

    def describe(self) -> dict[str, Any]:
        """Optional human/debug description of the configured source."""
        return {"kind": self.kind, "type": type(self).__name__}


class SinkConnector(ABC):
    """Consumes pipeline output (detections, enriched rows) to an external store."""

    kind = "sink"

    @abstractmethod
    def write(self) -> "beam.PTransform":
        """Return a PTransform: PCollection -> writes out (returns PDone/None)."""

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "type": type(self).__name__}