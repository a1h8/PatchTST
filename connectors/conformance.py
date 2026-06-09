"""Conformance suite — the contract every connector plugin must satisfy.

Reusable assertions so a new connector PR proves it honors the SPI without
re-inventing checks. The beam-dependent parts (actually running read/write) are
exercised in integration tests; these checks validate the contract surface and
registration, and run without a Beam install.
"""
from __future__ import annotations

from .base import SinkConnector, SourceConnector
from .registry import _REGISTRY, build


def assert_registered(name: str) -> type:
    """The name resolves to a registered connector class."""
    assert name in _REGISTRY, f"{name!r} not registered"
    return _REGISTRY[name]


def assert_source_contract(instance: SourceConnector) -> None:
    """Instance honors the SourceConnector surface."""
    assert isinstance(instance, SourceConnector), "must be a SourceConnector"
    assert instance.kind == "source"
    assert callable(getattr(instance, "read", None)), "must implement read()"
    desc = instance.describe()
    assert desc["kind"] == "source" and "type" in desc


def assert_sink_contract(instance: SinkConnector) -> None:
    """Instance honors the SinkConnector surface."""
    assert isinstance(instance, SinkConnector), "must be a SinkConnector"
    assert instance.kind == "sink"
    assert callable(getattr(instance, "write", None)), "must implement write()"
    desc = instance.describe()
    assert desc["kind"] == "sink" and "type" in desc


def assert_buildable(name: str, **cfg) -> object:
    """The connector can be instantiated from config via the registry."""
    instance = build(name, **cfg)
    if instance.kind == "source":
        assert_source_contract(instance)  # type: ignore[arg-type]
    else:
        assert_sink_contract(instance)  # type: ignore[arg-type]
    return instance