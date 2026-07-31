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

## Release (build + push the image)

Pushing a `v*.*.*` git tag is the release trigger: the `Build Docker Image`
workflow runs the tests, then builds `deploy/Dockerfile` and pushes it to
`ghcr.io/janeliascicomp/mrc-ng-server` with semver tags (e.g. `0.1.2`, `0.1`,
`0`, `latest`). The leading `v` is stripped, so `v0.1.2` publishes image tag
`0.1.2`.

Steps to cut release `v0.1.2`:

1. **Bump `version` in `pyproject.toml`** to `0.1.2`. Nothing enforces that this
   matches the tag, so do it by hand first. It's what `importlib.metadata`
   reports at `/healthz` and in `GENERATOR_VERSION`; the image tag k8s pulls
   comes from the git tag, not here. No relock is needed — the project's own
   version isn't recorded in `pixi.lock`, so `pixi install --locked` still
   passes unchanged.

   ```bash
   # edit pyproject.toml:  version = "0.1.2"
   git add pyproject.toml
   git commit -m "Bump version to 0.1.2"
   ```

2. **Tag the commit and push the tag** — this fires the build:

   ```bash
   git tag v0.1.2
   git push origin v0.1.2
   ```

That same tag is also the ref [ai-cryoet](https://github.com/JaneliaSciComp/ai-cryoet)'s
`mrc-ng-server` git dependency should pin to (its scanner needs the
`mrc-pyramid build --from-file` option), so one tag serves both the image and
the dependency.

(To skip the manual bump and derive `version` from the tag automatically, switch
the build to the `hatch-vcs` plugin; note the Docker build excludes `.git`, so
you'd pass the version in as a build-arg rather than have hatch-vcs read git
history.)

Manual build (no CI, e.g. local testing):

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
