# OTLP demo (connector C2)

Runs the `OTLPSource` path (D2: push, not pull — an OTel Collector or SDK
exporter POSTs to us) side by side with the existing Mimir and Kafka paths:

```
live-check-otlp-pusher pod ──POST /v1/metrics──▶ pipeline-otlp Job (embedded receiver) ──▶ KB datalake
```

Unlike Kafka's seed-then-drain shape, the receiver only starts listening once
the Job calls `read()`, so the pusher has to find it mid-flight — see
`tools/live_check_otlp.py` for the retry-until-reachable handshake, rather
than assuming a fixed startup delay.

Same shared `kb-datalake` PVC and `kb` service as `deploy/k3s`.

## Run it

```
kubectl apply -k deploy/k3s   # if not already applied
python -m tools.live_check_otlp
```

## Manifests

- `10-pipeline-otlp.yaml` — ConfigMap (`source: {type: otlp}`) + Job (the
  embedded receiver) + a Service (`pipeline-otlp:4318`) so another pod can
  reach it by DNS name, since a bare Job has no stable Service of its own.
