"""Connector SPI package (M1.5).

Public surface: the pivot schema, the Source/Sink contracts, and the registry.
Concrete connectors are registered by explicit import below (decision D3,
internal registry) — importing this package makes the built-in plugins
available via ``build("name", ...)``.
"""
from .base import SinkConnector, SourceConnector
from .engines import Engine, LocalEngine
from .pivot import PivotRow
from .registry import available, build, connector

# Register built-in connectors (side-effect imports).
from .sources import mimir as _mimir  # noqa: E402,F401
from .sources import synthetic as _synthetic  # noqa: E402,F401
from .sinks import parquet as _parquet  # noqa: E402,F401

__all__ = [
    "PivotRow",
    "SourceConnector",
    "SinkConnector",
    "connector",
    "build",
    "available",
    "Engine",
    "LocalEngine",
]