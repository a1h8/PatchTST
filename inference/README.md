# PatchTST inference (M1)

Checkpoint-backed PatchTST inference, decoupled from the training `Learner`. The
engine exposes the model's two heads as plain methods and loads each checkpoint
once per process:

```python
from inference import ModelSpec, PatchTSTInference

spec = ModelSpec(c_in=7, context_length=336, target_length=96,
                 patch_len=16, stride=8, n_layers=3, d_model=64, n_heads=8, d_ff=128)
eng = PatchTSTInference.from_checkpoints(
    "saved_models/etth1_finetune.pth",   # forecast head
    "saved_models/etth1_pretrain.pth",   # reconstruct head
    spec, device="cpu",
)

fc = eng.forecast(window)              # window: [context_length, c_in]
rc = eng.reconstruct(window)           # rc.error_per_channel, rc.error
```

`forecast` and `reconstruct` are two **independent** `PatchTST` instances: the
forecast head comes from the supervised finetune, the reconstruct head from the
self-supervised masked pretrain. Full finetune diverges the backbone, so they do
not share weights — see [docs/ROADMAP.md](../docs/ROADMAP.md) (M1).

## Producing checkpoints

`inference/train_reference.py` produces a reference pair on a public dataset
(ETTh1, downloaded on first run), self-contained and GPU-optional (MPS/CPU):

```bash
python -m inference.train_reference            # ETTh1, sensible defaults
python -m inference.train_reference --pretrain_epochs 20 --finetune_epochs 20 --d_model 128
```

Two stages, mirroring the PatchTST paper:

1. **pretrain** — self-supervised masked-patch reconstruction → `*_pretrain.pth`
   (backbone + pretrain head, the *reconstruct* head).
2. **finetune** — supervised forecast with the pretrain backbone transferred
   → `*_finetune.pth` (diverged backbone + prediction head, the *forecast* head).

It then loads both back through the engine and prints real evidence, e.g.:

```
forecast RMSE (val, denormalized): 0.95
reconstruction error  normal=0.87  anomaly=1.94  ratio=2.23x
```

The forecast head is a modest reference (reduced model, few epochs); scale up
`--d_model`/epochs on a GPU for quality. The reconstruction head already
separates in-distribution windows from a structural anomaly by ~2x.

## args → `ModelSpec` contract

A checkpoint is just a `state_dict`; to load it the `ModelSpec` must match the
**shape-defining** hyper-parameters used at training time, or `load_state_dict`
fails. The driver builds both stages from one `ModelSpec`, so its training args
map 1:1:

| training arg      | `ModelSpec` field   |
|-------------------|---------------------|
| (dataset channels)| `c_in`              |
| `context_length`  | `context_length`    |
| `target_length`   | `target_length`     |
| `patch_len`       | `patch_len`         |
| `stride`          | `stride`            |
| `n_layers`        | `n_layers`          |
| `d_model`         | `d_model`           |
| `n_heads`         | `n_heads`           |
| `d_ff`            | `d_ff`              |

`dropout`/`head_dropout` are not in the `state_dict` (dropout has no parameters),
so the inference spec leaves them at `0.0` regardless of the training value.

## Notes

- **Checkpoints and datasets are artifacts, not source.** `saved_models/` and
  `dataset/` are git-ignored. For shared/production use, publish the `.pth` to an
  object store (the S3/Iceberg sink is already on the roadmap) and load by path.
- **Reconstruction at inference is unmasked.** Pretraining masks 40% of patches;
  inference feeds the full window and measures how well it reconstructs — the
  error is the in-distribution score. Reported in normalized space (RevIN),
  i.e. scale-invariant.
- **Dependencies:** `torch` + `numpy` only (`requirements-inference.txt`); no
  `transformers`, unlike the on-the-fly `PatchTSTDetector`.
