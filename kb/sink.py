"""SignalStore as an SPI sink — aligns the knowledge-base WRITE path with the
connector cycle.

Without this, signals could be written two ways (the SPI *and* a standalone
``SignalStore.write``) — a divergence. This façade makes writing a signal go
through the same cycle as every other sink:

    build("signal-store", root=...) → Engine.run(source, [sink]) → sink.write(rows)

Dependency direction is correct: ``kb`` (higher level) depends on the connector
SPI, never the reverse. Registration happens when ``kb`` is imported.

Note: the knowledge-base READ path (``SignalStore.query`` / the HTTP service)
stays *outside* the SPI on purpose — it is a request/response serving concern,
not streaming dataflow.
"""
from __future__ import annotations

from typing import Iterable

from connectors.base import SinkConnector
from connectors.registry import connector

from .signal import SignalRecord
from .store import SignalStore


@connector("signal-store")
class SignalStoreSink(SinkConnector):
    """Write aggregated ``SignalRecord``s into the knowledge base via the SPI."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.store = SignalStore(root)

    def write(self, rows: Iterable[SignalRecord]) -> None:
        self.store.write(rows)

    def describe(self) -> dict:
        return {**super().describe(), "root": self.root}
