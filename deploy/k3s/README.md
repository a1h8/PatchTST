# K3s deployment

Runs the full simulation loop on a single-node k3s cluster:

```
ingest Job ──remote-write──▶ Mimir ──PromQL──▶ pipeline CronJob ──▶ KB datalake ──▶ KB service ──▶ kube-verdict
```

| Workload          | Kind        | Role                                                        |
|-------------------|-------------|------------------------------------------------------------|
| `mimir`           | HelmChart   | Prometheus-compatible TSDB (push + query), bundled MinIO    |
| `ingest-seed`     | Job         | Generates synthetic `sim_*` metrics, remote-writes to Mimir |
| `pipeline`        | CronJob     | Reads last hour from Mimir, detects, writes signals to KB   |
| `kb`              | Deployment  | Serves `signal_history` to kube-verdict                     |
| `kb-datalake`     | PVC         | Parquet datalake shared by pipeline (write) and KB (read)   |

> Prometheus and Loki are **not** part of this stack. The ingest Job replaces a
> Prometheus scrape; Loki (logs) would only matter for log/metric correlation,
> which is a future signal source, not the current metric pipeline.

## 1. Build the image and import it into k3s

The manifests use `patchtst-pipeline:dev` with `imagePullPolicy: IfNotPresent`,
so a locally-built image works without a registry — but it must live in k3s'
containerd, not just Docker:

```sh
# from the repo root
docker build -t patchtst-pipeline:dev .

# import into k3s' containerd (single node)
docker save patchtst-pipeline:dev | sudo k3s ctr images import -
```

For the PatchTST / reconstruction detectors (pulls torch, large image):

```sh
docker build --build-arg INSTALL_TORCH=1 -t patchtst-pipeline:dev .
```

## 2. Deploy

```sh
kubectl apply -k deploy/k3s
```

Mimir comes up via k3s' helm-controller (watch `kubectl get helmchart -n kube-system`
then `kubectl get pods -n patchtst`). First start pulls the chart + images, so
give it a few minutes.

## 3. Seed data and run a detection tick

The `ingest-seed` Job runs once on apply. Re-seed any time:

```sh
kubectl -n patchtst delete job ingest-seed --ignore-not-found
kubectl apply -k deploy/k3s            # recreates the Job
```

Trigger a pipeline run without waiting for the 5-minute schedule:

```sh
kubectl -n patchtst create job --from=cronjob/pipeline pipeline-manual
kubectl -n patchtst logs job/pipeline-manual -f
```

## 4. Verify the KB serves signals

```sh
kubectl -n patchtst port-forward svc/kb 8080:80 &
curl 'http://localhost:8080/health'
curl 'http://localhost:8080/api/v1/signals/history?entity=node1/demo'
```

## Notes / caveats

- **Single-node assumptions.** The KB datalake is a `ReadWriteOnce` local-path
  PVC shared by the pipeline and KB pods — fine because both schedule on the one
  node. The KB Deployment uses `Recreate` for the same reason. Multi-node needs
  RWX storage (NFS/Longhorn) or an object-store (`root: s3://…`).
- **Mimir values track chart 5.x.** They disable zone-aware replication and set
  `replication_factor: 1` for a single node. After a chart version bump, run
  `helm template mimir grafana/mimir-distributed --version <v> -f <values>` to
  confirm the keys still resolve.
- **Gateway service name.** Apps talk to `mimir-gateway.patchtst.svc:80`. If you
  change the HelmChart release name, update the endpoints in `30-pipeline.yaml`
  and `40-ingest.yaml`. Confirm with `kubectl -n patchtst get svc | grep gateway`.
- **Rolling window.** The runner takes static `start/end`; the CronJob computes
  `now-3600 .. now` at runtime and substitutes it into the mounted config.
