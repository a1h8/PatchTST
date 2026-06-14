"""PatchTST inference module (roadmap M1).

Decouples PatchTST from the training ``Learner``: a checkpoint-backed engine that
exposes the model's **two heads** as plain methods — ``forecast(window)`` and
``reconstruct(window)`` — with RevIN normalization and the checkpoint loaded once
per process (per Beam worker, later). No optimizer, no dataloader, no training.

The two heads come from two checkpoints sharing the PatchTST architecture but not
their weights: the *reconstruct* head from self-supervised masked pretraining, the
*forecast* head from the supervised finetune (whose backbone has diverged from the
pretrain backbone). The engine therefore holds two independent ``PatchTST``
instances — see docs/ROADMAP.md (M1) and the D1 detection faces it feeds.
"""
from __future__ import annotations

from .config import ModelSpec
from .engine import ForecastResult, PatchTSTInference, ReconstructResult

__all__ = ["ModelSpec", "PatchTSTInference", "ForecastResult", "ReconstructResult"]
