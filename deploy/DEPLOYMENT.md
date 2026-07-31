# Deploying mrc-ng-server

mrc-ng-server runs as a standalone, horizontally-scalable data service that
serves MRC tomograms to Neuroglancer over the `precomputed` protocol. It is
consumed by the ai-cryoet portal: the API builds `precomputed://<this
service>/data/<mrc_path>` links, and the browser fetches chunks directly from
here.

## Topology

It deploys into the **`ai-cryoet` namespace** so it can share storage with the
portal:

- **Data** (`catalog-data-pvc`, read-only) — the NFS-backed MRC tree, declared
  by the ai-cryoet deployment and reused here by name. `MRCNG_SOURCE_ROOT` must
  equal its mountPath and the portal's `CATALOG_DATA_ROOT`.
- **Cache** (`mrc-cache-pvc`, ReadWriteMany) — the downsample pyramid cache,
  owned by this repo (`k8s/base/storage.yaml`). **Written** by the ai-cryoet
  scanner's `mrc-pyramid build` step, **read** here. Apply this repo's manifests
  before the scanner runs so the PVC exists.

TLS terminates at the OpenShift **Route** (`k8s/base/route.yaml`, edge); the pod
serves plain HTTP on :8000. The Route host is what the portal's `MRCNG_BASE_URL`
must point at: `https://<route-host>/data`.

## Build the image

```bash
docker build -f deploy/Dockerfile -t ghcr.io/janeliascicomp/mrc-ng-server:<tag> .
docker push ghcr.io/janeliascicomp/mrc-ng-server:<tag>
```

## Deploy

```bash
cp deploy/k8s/overlays/production/config.env.example \
   deploy/k8s/overlays/production/config.env
# edit config.env: MRCNG_SOURCE_ROOT, MRCNG_CACHE_ROOT, MRCNG_CORS_ORIGINS
# set the image tag + Route host in overlays/production/kustomization.yaml
kubectl apply -k deploy/k8s/overlays/production
```

`config.env` is gitignored — never commit real values.

## Scaling

The service is stateless; raise `replicas` in `k8s/base/deployment.yaml` (or
attach an HPA). The bottleneck is NFS/cephfs I/O, not CPU.
