"""Architecture spec for a PatchTST checkpoint.

A checkpoint is just a ``state_dict``; to load it we must rebuild the exact module
that produced it. ``ModelSpec`` captures the architecture hyper-parameters (the
ones that change tensor shapes) so forecast and reconstruct checkpoints can be
re-instantiated and loaded. Values mirror the defaults used by the training
scripts' ``get_model`` (shared embedding, relu activation, non-residual
attention) — keep them in sync with the checkpoint you load.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Shape-defining hyper-parameters of a PatchTST model.

    ``c_in`` is the number of channels (native-multivariate, decision D4).
    ``context_length`` is the input window length; ``target_length`` the forecast
    horizon (ignored by the reconstruction head). The window fed to the engine
    must be exactly ``context_length`` long, since the prediction head's linear
    layer is sized from ``num_patch``.
    """

    c_in: int
    context_length: int
    target_length: int
    patch_len: int = 8
    stride: int = 8
    n_layers: int = 2
    d_model: int = 32
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.0
    head_dropout: float = 0.0
    shared_embedding: bool = True
    res_attention: bool = False
    activation: str = "relu"

    @property
    def num_patch(self) -> int:
        """Number of patches produced from a ``context_length`` window."""
        return (max(self.context_length, self.patch_len) - self.patch_len) // self.stride + 1

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        """Build from a plain dict (e.g. parsed YAML), ignoring unknown keys."""
        fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})
