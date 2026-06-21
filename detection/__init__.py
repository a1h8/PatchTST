"""Variation detection — PivotRows → SignalRecords.

The detection stage between a source and the knowledge-base sink. Plug a
``Detector`` (ZScoreDetector now; PatchTST later) into ``make_detection_transform``
and hand it to an engine: ``Engine.run(source, sinks, transform=...)``.
"""
from .aggregate import (
    ENTITY_METRIC,
    aggregate_entity,
    aggregate_signals,
    detect_signals,
    make_detection_transform,
)
from .detector import Detector, ZScoreDetector
from .inference_detector import (
    ForecastInferenceDetector,
    ReconstructionInferenceDetector,
    clear_engine_cache,
)
from .patchtst import PatchTSTDetector
from .reconstruction import ReconstructionDetector
from .regime import InMemoryRegimeState, RegimeStatus, RegimeSwitchDetector

__all__ = [
    "Detector",
    "ZScoreDetector",
    "PatchTSTDetector",
    "ReconstructionDetector",
    "ForecastInferenceDetector",
    "ReconstructionInferenceDetector",
    "clear_engine_cache",
    "RegimeSwitchDetector",
    "InMemoryRegimeState",
    "RegimeStatus",
    "detect_signals",
    "make_detection_transform",
    "aggregate_entity",
    "aggregate_signals",
    "ENTITY_METRIC",
]
