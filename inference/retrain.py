"""Optional retraining loop (M7) — close the knowledge-base feedback.

PatchTST's forecast/reconstruction heads must learn *normal* behaviour, so the
knowledge base drives training-data selection: stretches the pipeline flagged as
anomalous (severity != ``normal`` — the NORMAL→INCIDENT verdicts) are cut out,
and the model is retrained on the remaining clean history. New checkpoints are
written under a **versioned** directory with a ``latest`` pointer the inference
detectors (``patchtst-infer`` / ``reconstruction-infer``) can follow.

The heavy training reuses :mod:`inference.train_reference` (``pretrain`` →
``finetune``); this module is the orchestration: KB-guided window selection,
assembling the clean training array, and checkpoint versioning. It is the
"optional retraining loop back to the datalake" of M7 (D5: the datalake is the
knowledge base — signal history — which here feeds back into model selection).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping

import numpy as np


def normal_keep_mask(
    store,
    entity_uid: str,
    ts_start: int,
    step_ms: int,
    n: int,
    *,
    metric: str | None = None,
    exclude_radius: int = 0,
) -> np.ndarray:
    """Boolean mask over ``n`` grid points: ``False`` where the KB flags anomaly.

    Queries the signal history for ``entity_uid`` over the grid
    ``[ts_start, ts_start + (n-1)*step_ms]`` and drops every point covered by an
    anomalous ``SignalRecord`` (``severity != "normal"``), widened by
    ``exclude_radius`` grid steps on each side so the transition around an
    incident is not learned as normal.
    """
    until = ts_start + (n - 1) * step_ms
    keep = np.ones(n, dtype=bool)
    for s in store.query(entity_uid, metric=metric, since=ts_start, until=until):
        if s.severity == "normal":
            continue
        idx = round((s.ts - ts_start) / step_ms)
        lo = max(0, idx - exclude_radius)
        hi = min(n, idx + exclude_radius + 1)
        keep[lo:hi] = False
    return keep


def longest_normal_segment(series: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Longest contiguous run of kept rows in ``series`` (``keep[i]`` True).

    Training windows must be contiguous in time, so a single clean run is used
    rather than a gappy concatenation that would straddle excised incidents.
    """
    best_lo = best_len = cur_lo = cur_len = 0
    for i, k in enumerate(keep):
        if k:
            if cur_len == 0:
                cur_lo = i
            cur_len += 1
            if cur_len > best_len:
                best_lo, best_len = cur_lo, cur_len
        else:
            cur_len = 0
    return series[best_lo : best_lo + best_len]


def select_training_series(
    store,
    entity_uid: str,
    series: np.ndarray,
    ts_start: int,
    step_ms: int,
    *,
    metric: str | None = None,
    exclude_radius: int = 0,
) -> np.ndarray:
    """Longest clean (KB-NORMAL) contiguous slice of ``series`` for the entity."""
    keep = normal_keep_mask(
        store,
        entity_uid,
        ts_start,
        step_ms,
        len(series),
        metric=metric,
        exclude_radius=exclude_radius,
    )
    return longest_normal_segment(series, keep)


def write_version(
    out_dir: str,
    *,
    forecast_ckpt: str,
    reconstruct_ckpt: str,
    meta: Mapping[str, Any] | None = None,
    version: str | None = None,
) -> dict:
    """Record a checkpoint set under ``out_dir/<version>/manifest.json``.

    Also writes ``out_dir/latest.json`` pointing at this version — the pointer
    the inference detectors resolve to pick up a freshly retrained model.
    Returns the manifest.
    """
    version = version or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    vdir = os.path.join(out_dir, version)
    os.makedirs(vdir, exist_ok=True)
    manifest = {
        "version": version,
        "created_ms": int(time.time() * 1000),
        "forecast_ckpt": forecast_ckpt,
        "reconstruct_ckpt": reconstruct_ckpt,
        **(dict(meta) if meta else {}),
    }
    with open(os.path.join(vdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out_dir, "latest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def resolve_latest(out_dir: str) -> dict | None:
    """The most recently written version manifest, or ``None`` if never trained."""
    path = os.path.join(out_dir, "latest.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def retrain(
    store,
    spec,
    series_by_entity: Mapping[str, np.ndarray],
    args,
    *,
    out_dir: str,
    ts_start: int,
    step_ms: int,
    device: str = "cpu",
    exclude_radius: int = 0,
) -> dict:
    """Retrain on KB-clean history and write a new versioned checkpoint set.

    For each entity, keep its longest KB-NORMAL contiguous slice (>= one context
    window); the slices are stacked into one training array. Reconstruction
    (``pretrain``) then forecast (``finetune``) checkpoints are produced by
    :mod:`inference.train_reference` and versioned via :func:`write_version`.

    Raises ``ValueError`` if the KB filtering leaves no usable training data.
    """
    from .train_reference import finetune, pretrain

    segments = []
    for entity_uid, series in series_by_entity.items():
        seg = select_training_series(
            store,
            entity_uid,
            np.asarray(series),
            ts_start,
            step_ms,
            exclude_radius=exclude_radius,
        )
        if len(seg) >= spec.context_length:
            segments.append(seg)

    if not segments:
        raise ValueError(
            "no clean training windows after KB filtering "
            f"(need >= {spec.context_length} contiguous NORMAL points)"
        )

    train = np.concatenate(segments, axis=0)
    reconstruct_ckpt = pretrain(spec, train, args, device)
    forecast_ckpt = finetune(spec, train, reconstruct_ckpt, args, device)
    return write_version(
        out_dir,
        forecast_ckpt=forecast_ckpt,
        reconstruct_ckpt=reconstruct_ckpt,
        meta={"n_train_rows": int(len(train)), "n_entities": len(segments)},
    )
