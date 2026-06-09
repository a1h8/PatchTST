"""Execution engines (adapters).

The core and connectors are engine-agnostic; an ``Engine`` adapter runs a
source → sinks flow on a concrete runtime. ``LocalEngine`` needs no third-party
dependency; ``BeamEngine`` runs on Apache Beam. A Spark/Databricks adapter is the
next planned engine.
"""
from .base import Engine
from .local import LocalEngine

__all__ = ["Engine", "LocalEngine"]
