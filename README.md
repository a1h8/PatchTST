# PatchTST Temporal Evidence Layer for KubeVerdict

This repository is a fork of the official PatchTST implementation:
*"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"* — ICLR 2023.

This fork adapts PatchTST into an operational **temporal evidence** pipeline for KubeVerdict.

It does **not** replace Kubernetes RCA and does **not** claim autonomous incident
prediction. It produces temporal variation signals — forecast residuals,
reconstruction errors, z-score fallbacks and regime transitions — that can
**strengthen or weaken** evidence-ranked RCA hypotheses in KubeVerdict.

> **Current status:** the operational pipeline is designed and partially
> implemented with **synthetic or fixture-based** time-series scenarios. Real
> Prometheus-backed telemetry integration is the next validation step before
> claiming production-grade temporal evidence.

**Value added in this fork:**

- connector interfaces
- detection pipeline
- inference wrappers
- reconstruction-based anomaly signals
- fallback detectors
- k3s deployment assets
- KubeVerdict integration path

How the two faces work together: a regime-switching state machine runs the
**forecast** face in the NORMAL regime (residuals surface slow saturation before
it breaks — disk / memory / quota fill, latency drift) and the **reconstruction**
face in the INCIDENT regime (reconstruction error is the clean out-of-distribution
signal during the break), with adaptive thresholds, per-entity aggregation and
anti-flapping on top.

### Repository layout

| Component | Path | Role |
|-----------|------|------|
| Connector SPI | [`connectors/`](./connectors) | engine-agnostic source/sink contracts + registry (Mimir source, signal-store sink) |
| Detection | [`detection/`](./detection) | `ZScoreDetector`, PatchTST forecast / reconstruction detectors, `RegimeSwitchDetector`, adaptive thresholds, entity aggregation |
| Inference | [`inference/`](./inference) | PatchTST inference decoupled from training: forecast + reconstruct heads, checkpoint loaded once per worker |
| Knowledge base | [`kb/`](./kb) | `SignalRecord` + Parquet/DuckDB store + `signal_history` query API (RCA evidence for KubeVerdict) |
| Pipeline | [`pipeline/`](./pipeline) | config-driven runner: source → detection → signal-store |
| Deployment | [`deploy/k3s/`](./deploy/k3s) | k3s manifests for the full simulation loop |
| Tests | [`tests/`](./tests) | contract/conformance + detection / inference / KB suites |

Design and plan: [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) ·
[`docs/ROADMAP.md`](./docs/ROADMAP.md) · [`docs/CONNECTORS.md`](./docs/CONNECTORS.md).

## Upstream attribution

This project is based on the official PatchTST implementation by Yuqi Nie et al.

- Original repository: <https://github.com/yuqinie98/PatchTST>
- Paper: *"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"* — ICLR 2023 (<https://arxiv.org/abs/2211.14730>)

The original work is licensed under **Apache-2.0**. This fork keeps the original
license ([`LICENSE`](./LICENSE)) and citation while adding operational components
for temporal incident evidence in KubeVerdict. The unmodified upstream model code
lives in [`PatchTST_supervised/`](./PatchTST_supervised) and
[`PatchTST_self_supervised/`](./PatchTST_self_supervised).

### Citation

If you use the PatchTST model, please cite the original authors:

```
@inproceedings{Yuqietal-2023-PatchTST,
  title     = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author    = {Nie, Yuqi and
               H. Nguyen, Nam and
               Sinthong, Phanwadee and
               Kalagnanam, Jayant},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```

---

# Upstream: PatchTST (ICLR 2023)

*The following is the original PatchTST project documentation, preserved from
upstream.*

This is the official implementation of PatchTST:
[A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730).

:triangular_flag_on_post: The model is included in [GluonTS](https://github.com/awslabs/gluonts). Special thanks to the contributor @[kashif](https://github.com/kashif)!

:triangular_flag_on_post: The model is included in [NeuralForecast](https://github.com/Nixtla/neuralforecast). Special thanks to the contributors @[kdgutier](https://github.com/kdgutier) and @[cchallu](https://github.com/cchallu)!

:triangular_flag_on_post: The model is included in [timeseriesAI(tsai)](https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/15_PatchTST_a_new_transformer_for_LTSF.ipynb). Special thanks to the contributor @[oguiza](https://github.com/oguiza)!

A video providing a concise overview of the paper: https://www.youtube.com/watch?v=Z3-NrohddJw

## Key Designs

:star2: **Patching**: segmentation of time series into subseries-level patches which are served as input tokens to Transformer.

:star2: **Channel-independence**: each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series.

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/model.png)

## Results

### Supervised Learning

Compared with the best results that Transformer-based models can offer, PatchTST/64 achieves an overall **21.0%** reduction on MSE and **16.7%** reduction
on MAE, while PatchTST/42 attains a overall **20.2%** reduction on MSE and **16.4%** reduction on MAE. It also outperforms other non-Transformer-based models like DLinear.

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/table3.png)

### Self-supervised Learning

We do comparison with other supervised and self-supervised models, and self-supervised PatchTST is able to outperform all the baselines.

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/table4.png)

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/table6.png)

We also test the capability of transfering the pre-trained model to downstream tasks.

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/table5.png)

## Efficiency on Long Look-back Windows

PatchTST consistently <ins>reduces the MSE scores as the look-back window increases</ins>, which confirms the model’s capability to learn from longer receptive field.

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/varying_L.png)

## Getting Started (upstream model)

The codes for supervised learning and self-supervised learning are in 2 folders: ```PatchTST_supervised``` and ```PatchTST_self_supervised```. Please choose the one that you want to work with.

### Supervised Learning

1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a seperate folder ```./dataset``` and put all the csv files in the directory.

3. Training. All the scripts are in the directory ```./scripts/PatchTST```. The default model is PatchTST/42. For example, if you want to get the multivariate forecasting results for weather dataset, just run the following command, and you can open ```./result.txt``` to see the results once the training is done:
```
sh ./scripts/PatchTST/weather.sh
```

You can adjust the hyperparameters based on your needs (e.g. different patch length, different look-back windows and prediction lengths.). We also provide codes for the baseline models.

### Self-supervised Learning

1. Follow the first 2 steps above

2. Pre-training: The scirpt patchtst_pretrain.py is to train the PatchTST/64. To run the code with a single GPU on ettm1, just run the following command
```
python patchtst_pretrain.py --dset ettm1 --mask_ratio 0.4
```
The model will be saved to the saved_model folder for the downstream tasks. There are several other parameters can be set in the patchtst_pretrain.py script.

3. Fine-tuning: The script patchtst_finetune.py is for fine-tuning step. Either linear_probing or fine-tune the entire network can be applied.
```
python patchtst_finetune.py --dset ettm1 --pretrained_model <model_name>
```

## Acknowledgement

We appreciate the following github repos very much for the valuable code base and datasets:

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/ts-kim/RevIN

https://github.com/timeseriesAI/tsai

## Contact (upstream authors)

For questions about the PatchTST model/paper: ynie@princeton.edu or nnguyen@us.ibm.com, or submit an issue.
