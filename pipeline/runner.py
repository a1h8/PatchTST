"""Config-driven pipeline runner — the runnable entrypoint.

Assembles the cycle from config: source (SPI) → detection transform → sinks
(SPI), executed by an engine. This is what a K3s CronJob invokes.

Config shape (dict / YAML / JSON):

    source:   {type: mimir, params: {...}}
    detector: {type: regime-switch,
               forecast:  {type: patchtst},
               detective: {type: reconstruction}}
    sinks:    [{type: signal-store, params: {root: ...}}]
    engine:   {type: local}        # or beam

The ``patchtst-infer`` / ``reconstruction-infer`` detectors run the load-once M1
engine instead of training on the fly; both take ``params: {forecast_ckpt,
reconstruct_ckpt, spec}`` and, sharing those checkpoints, share one loaded engine.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import kb  # noqa: F401  (registers the signal-store sink connector)
from connectors import LocalEngine, build
from connectors.engines.base import Engine
from detection import (
    Detector,
    ForecastInferenceDetector,
    PatchTSTDetector,
    ReconstructionDetector,
    ReconstructionInferenceDetector,
    RegimeSwitchDetector,
    ZScoreDetector,
    make_detection_transform,
)

_DETECTORS: dict[str, type[Detector]] = {
    "zscore": ZScoreDetector,
    "patchtst": PatchTSTDetector,
    "reconstruction": ReconstructionDetector,
    "patchtst-infer": ForecastInferenceDetector,
    "reconstruction-infer": ReconstructionInferenceDetector,
}


def build_detector(cfg: dict) -> Detector:
    """Build a Detector from config (recursive for regime-switch)."""
    kind = cfg["type"]
    if kind == "regime-switch":
        # params carry the anti-flapping knobs: enter_after / exit_after.
        return RegimeSwitchDetector(
            forecast=build_detector(cfg["forecast"]),
            detective=build_detector(cfg["detective"]),
            **cfg.get("params", {}),
        )
    try:
        cls = _DETECTORS[kind]
    except KeyError:
        raise KeyError(
            f"unknown detector {kind!r}; available: "
            f"{sorted(_DETECTORS) + ['regime-switch']}"
        ) from None
    return cls(**cfg.get("params", {}))


def build_engine(cfg: dict | None) -> Engine:
    kind = (cfg or {}).get("type", "local")
    if kind == "beam":
        from connectors.engines.beam import BeamEngine

        return BeamEngine()
    if kind == "local":
        return LocalEngine()
    raise KeyError(f"unknown engine {kind!r}; available: ['local', 'beam']")


def load_config(path: str) -> dict:
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        import yaml  # lazy: only the YAML path needs PyYAML

        return yaml.safe_load(text)
    return json.loads(text)


def run_pipeline(config: dict[str, Any], *, now_ms: int | None = None) -> None:
    """Build source/detector/sinks/engine from config and run the cycle."""
    source = build(config["source"]["type"], **config["source"].get("params", {}))
    sinks = [
        build(s["type"], **s.get("params", {})) for s in config["sinks"]
    ]
    detector = build_detector(config["detector"])
    transform = make_detection_transform(detector, now_ms=now_ms)
    engine = build_engine(config.get("engine"))
    engine.run(source, sinks, transform=transform)
