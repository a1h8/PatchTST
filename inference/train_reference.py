"""Produce real PatchTST checkpoints for the M1 engine (PR-C).

Two-stage PatchTST recipe on a public dataset (ETTh1 by default), mirroring the
paper and the vendored training scripts, but self-contained (no hard-coded data
paths, no GPU required — uses MPS/CPU):

    1. pretrain   — self-supervised masked-patch reconstruction  -> reconstruct head
    2. finetune   — supervised forecast, backbone transferred     -> forecast head

Both stages build the model via :class:`ModelSpec` so the saved ``state_dict``s
load straight into :class:`inference.PatchTSTInference` with the *same* spec.
Checkpoints are written under ``saved_models/`` (git-ignored — artifacts, not
source). At the end it loads them back through the engine and prints real
forecast / reconstruction evidence (incl. normal-vs-anomaly reconstruction error)
so M1 is no longer "validated on synthetic weights only".

Run:  python -m inference.train_reference            # ETTh1, sensible defaults
See inference/README.md for the args->ModelSpec contract.
"""
from __future__ import annotations

import argparse
import logging
import os
import urllib.request

import numpy as np

from .config import ModelSpec
from .engine import PatchTSTInference

log = logging.getLogger(__name__)

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
DATA_DIR = "dataset/ETT-small"
CKPT_DIR = "saved_models"

# ETTh1 standard boundaries (months * 30 days * 24 hours).
_TRAIN_END = 12 * 30 * 24
_VAL_END = _TRAIN_END + 4 * 30 * 24


def _device() -> str:  # pragma: no cover - environment-dependent
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _download_etth1() -> str:  # pragma: no cover - network I/O
    path = os.path.join(DATA_DIR, "ETTh1.csv")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        log.info("downloading ETTh1 -> %s", path)
        urllib.request.urlretrieve(ETTH1_URL, path)
    return path


def _load_channels(path: str) -> np.ndarray:
    """ETTh1 -> [N, 7] float32 (drops the date column)."""
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=range(1, 8))
    return raw.astype(np.float32)


def _standardize(data: np.ndarray, train_end: int):
    mu = data[:train_end].mean(axis=0, keepdims=True)
    sd = data[:train_end].std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (data - mu) / sd


def _windows(series: np.ndarray, length: int, stride: int) -> np.ndarray:
    """Sliding windows -> [n, length, c]."""
    idx = range(0, len(series) - length + 1, stride)
    return np.stack([series[i : i + length] for i in idx]).astype(np.float32)


def _pairs(series: np.ndarray, ctx: int, horizon: int, stride: int):
    """Context/future pairs -> ([n, ctx, c], [n, horizon, c])."""
    total = ctx + horizon
    idx = list(range(0, len(series) - total + 1, stride))
    x = np.stack([series[i : i + ctx] for i in idx]).astype(np.float32)
    y = np.stack([series[i + ctx : i + total] for i in idx]).astype(np.float32)
    return x, y


def _batches(n: int, batch_size: int, shuffle: bool = True):
    order = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        yield order[i : i + batch_size]


def pretrain(spec: ModelSpec, train: np.ndarray, args, device: str) -> str:
    """Masked-patch reconstruction; saves and returns the pretrain checkpoint."""
    import torch

    from PatchTST_self_supervised.src.callback.patch_mask import create_patch, random_masking
    from PatchTST_self_supervised.src.models.layers.revin import RevIN

    model = PatchTSTInference._build(spec, head_type="pretrain").to(device)
    revin = RevIN(spec.c_in, affine=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    wins = _windows(train, spec.context_length, args.win_stride)
    log.info("pretrain: %d windows", len(wins))
    model.train()
    for epoch in range(args.pretrain_epochs):
        total, nb = 0.0, 0
        for sel in _batches(len(wins), args.batch_size):
            xb = torch.from_numpy(wins[sel]).to(device)
            xn = revin(xb, "norm")
            patched, _ = create_patch(xn, spec.patch_len, spec.stride)
            x_masked, _, mask, _ = random_masking(patched, args.mask_ratio)
            recon = model(x_masked)
            # MSE on masked patches only (mask==1): [bs, num_patch, c]
            err = ((recon - patched) ** 2).mean(dim=-1)
            loss = (err * mask).sum() / (mask.sum() + 1e-8)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        log.info("pretrain epoch %d/%d  loss=%.4f", epoch + 1, args.pretrain_epochs, total / nb)

    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"{args.dset}_pretrain.pth")
    torch.save(model.state_dict(), path)
    log.info("saved pretrain checkpoint -> %s", path)
    return path


def _transfer_backbone(pretrain_path: str, model, device: str):
    """Copy matching non-head weights from a pretrain checkpoint into ``model``.

    A lightweight stand-in for the vendored ``transfer_weights`` (which pulls in
    sklearn via the Learner). Backbone keys are shared; head keys differ between
    the pretrain and prediction heads and are left at init.
    """
    import torch

    src = torch.load(pretrain_path, map_location=device)
    dst = model.state_dict()
    matched = 0
    for name, param in dst.items():
        if "head" in name:
            continue
        if name in src and src[name].shape == param.shape:
            param.copy_(src[name])
            matched += 1
    if matched == 0:
        raise RuntimeError("no shared backbone weights between pretrain and finetune models")
    log.info("transferred %d backbone tensors from %s", matched, pretrain_path)
    return model


def finetune(spec: ModelSpec, train: np.ndarray, pretrain_path: str, args, device: str) -> str:
    """Forecast finetune with transferred backbone; saves the forecast checkpoint."""
    import torch

    from PatchTST_self_supervised.src.models.layers.revin import RevIN

    model = PatchTSTInference._build(spec, head_type="prediction").to(device)
    model = _transfer_backbone(pretrain_path, model, device)
    revin = RevIN(spec.c_in, affine=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    x, y = _pairs(train, spec.context_length, spec.target_length, args.win_stride)
    log.info("finetune: %d pairs", len(x))
    model.train()
    for epoch in range(args.finetune_epochs):
        total, nb = 0.0, 0
        for sel in _batches(len(x), args.batch_size):
            xb = torch.from_numpy(x[sel]).to(device)
            yb = torch.from_numpy(y[sel]).to(device)
            from PatchTST_self_supervised.src.callback.patch_mask import create_patch

            xn = revin(xb, "norm")
            patched, _ = create_patch(xn, spec.patch_len, spec.stride)
            pred = revin(model(patched), "denorm")
            loss = torch.mean((pred - yb) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        log.info("finetune epoch %d/%d  loss=%.4f", epoch + 1, args.finetune_epochs, total / nb)

    path = os.path.join(CKPT_DIR, f"{args.dset}_finetune.pth")
    torch.save(model.state_dict(), path)
    log.info("saved finetune checkpoint -> %s", path)
    return path


def report(spec: ModelSpec, val: np.ndarray, fc_path: str, rc_path: str, device: str) -> None:
    """Load the checkpoints through the engine and print real evidence."""
    eng = PatchTSTInference.from_checkpoints(fc_path, rc_path, spec, device=device)

    x, y = _pairs(val, spec.context_length, spec.target_length, spec.context_length)
    fc = eng.forecast(x[: min(64, len(x))], future=y[: min(64, len(y))])

    win = _windows(val, spec.context_length, spec.context_length)[0]
    normal_err = eng.reconstruct(win).error
    # Structural anomaly: shuffle the timesteps. RevIN normalizes mean/std per
    # window, so an additive offset would be absorbed — destroying temporal order
    # is the scale-invariant anomaly a structure-learning model should miss.
    anomaly = win.copy()
    np.random.default_rng(0).shuffle(anomaly)
    anomaly_err = eng.reconstruct(anomaly).error

    print("\n=== M1 engine on REAL checkpoints ===")
    print(f"spec: {spec}")
    print(f"forecast RMSE (val, denormalized): {fc.rmse:.4f}")
    print(f"reconstruction error  normal={normal_err:.4f}  anomaly={anomaly_err:.4f}"
          f"  ratio={anomaly_err / max(normal_err, 1e-8):.2f}x")


def build_spec(args) -> ModelSpec:
    return ModelSpec(
        c_in=7,  # ETTh1 channels
        context_length=args.context_length,
        target_length=args.target_length,
        patch_len=args.patch_len,
        stride=args.stride,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
    )


def main() -> None:  # pragma: no cover - CLI entrypoint
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Train reference PatchTST checkpoints for M1")
    p.add_argument("--dset", default="etth1")
    p.add_argument("--context_length", type=int, default=336)
    p.add_argument("--target_length", type=int, default=96)
    p.add_argument("--patch_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=128)
    p.add_argument("--mask_ratio", type=float, default=0.4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--win_stride", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pretrain_epochs", type=int, default=10)
    p.add_argument("--finetune_epochs", type=int, default=10)
    args = p.parse_args()

    device = _device()
    log.info("device: %s", device)
    spec = build_spec(args)

    data = _standardize(_load_channels(_download_etth1()), _TRAIN_END)
    train, val = data[:_TRAIN_END], data[_TRAIN_END:_VAL_END]

    rc_path = pretrain(spec, train, args, device)
    fc_path = finetune(spec, train, rc_path, args, device)
    report(spec, val, fc_path, rc_path, device)


if __name__ == "__main__":  # pragma: no cover
    main()
