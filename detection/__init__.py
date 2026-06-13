"""Variation detection — PivotRows → SignalRecords.

The detection stage between a source and the knowledge-base sink. Plug a
``Detector`` (ZScoreDetector now; PatchTST later) into ``make_detection_transform``
and hand it to an engine: ``Engine.run(source, sinks, transform=...)``.
"""
from .aggregate import detect_signals, make_detection_transform
from .detector import Detector, ZScoreDetector
from .patchtst import PatchTSTDetector
from .reconstruction import ReconstructionDetector
from .regime import InMemoryRegimeState, RegimeSwitchDetector

__all__ = [
    "Detector",
    "ZScoreDetector",
    "PatchTSTDetector",
    "ReconstructionDetector",
    "RegimeSwitchDetector",
    "InMemoryRegimeState",
    "detect_signals",
    "make_detection_transform",
]
