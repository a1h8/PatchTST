"""Internal plugin registry (decision D3).

Connectors register themselves with ``@connector("name")`` and are discovered by
explicit import (see ``connectors/__init__.py``). No entry-points, no packaging
overhead — sufficient while connectors live in this repo. The contract is small
enough to swap for ``importlib.metadata`` entry-points later without touching
call sites.
"""
from __future__ import annotations

from typing import Callable, TypeVar

from .base import SinkConnector, SourceConnector

_REGISTRY: dict[str, type] = {}

T = TypeVar("T", bound=type)


def connector(name: str) -> Callable[[T], T]:
    """Register a Source/Sink connector class under ``name``."""

    def deco(cls: T) -> T:
        if not issubclass(cls, (SourceConnector, SinkConnector)):
            raise TypeError(
                f"{cls.__name__} must subclass SourceConnector or SinkConnector"
            )
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"connector name {name!r} already registered")
        _REGISTRY[name] = cls
        return cls

    return deco


def build(name: str, **cfg):
    """Instantiate a registered connector from config."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown connector {name!r}; available: {sorted(_REGISTRY)}"
        ) from None
    return cls(**cfg)


def available() -> dict[str, str]:
    """Map of registered name -> 'source'|'sink'."""
    return {name: cls.kind for name, cls in sorted(_REGISTRY.items())}