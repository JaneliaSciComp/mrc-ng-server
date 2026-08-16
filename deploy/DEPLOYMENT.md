# Deploying mrc-ng-server

mrc-ng-server is a standalone, horizontally-scalable data service that serves
MRC tomograms to Neuroglancer over the `precomputed` protocol. It is generic:
point it at a tree of MRC files and a pyramid cache and it serves them. A
consumer builds `precomputed://<this service>/data/<mrc_path>` links and the
browser fetches chunks directly from here.

This repo ships the **image**; each deployment supplies its own orchestration
(k8s manifests, compose file, etc.) and config. For the Janelia ai-cryoet
deployment, the manifests live alongside that portal's own deploy config in the
[ai-cryoet](https://github.com/JaneliaSciComp/ai-cryoet) repo, not here.

## Configuration

All runtime config is read from the `MRCNG_`-prefixed environment
(pydantic-settings; see `src/mrcng/server/config.py`). Two are required; the
rest have defaults.

| Env var | Required | Meaning |
|---|---|---|
| `MRCNG_SOURCE_ROOT` | yes | Root of the MRC tree. Must equal the path the pyramid cache was built against — consumers build `precomputed://` URLs from each file's path relative to this root. |
| `MRCNG_CACHE_ROOT` | yes | The pyramid cache (`mrc-pyramid build` output) this server reads. |
| `MRCNG_CORS_ORIGINS` | no (`*`) | `*` or a comma-separated list of browser origins allowed to fetch chunks. The Neuroglancer viewer fetches cross-origin, so scope this to the viewer's origin in production. |
| `MRCNG_CHUNK_SIZE` | no (`64,64,64`) | Must match what `mrc-pyramid build` used, or caches read as incompatible. |
| `MRCNG_FD_CACHE_SIZE`, `MRCNG_MAX_CONCURRENT_READS`, ... | no | Performance knobs; see `config.py`. |

The image also reads `NUM_WORKERS` (uvicorn worker count, default 1) and
`HOST`/`PORT` (default `0.0.0.0:8000`).

## Run

The container serves **plain HTTP on :8000**; terminate TLS upstream (a reverse
proxy or the cluster's Route). Supply the two paths and mount the data + cache:

```bash
docker run --rm -p 8000:8000 \
  -e MRCNG_SOURCE_ROOT=/data \
  -e MRCNG_CACHE_ROOT=/cache \
  -e MRCNG_CORS_ORIGINS=https://viewer.example.org \
  -e NUM_WORKERS=4 \
  -v /path/to/mrc/tree:/data:ro \
  -v /path/to/pyramid/cache:/cache:ro \
  ghcr.io/janeliascicomp/mrc-ng-server:latest
```

Verify: `curl http://localhost:8000/healthz` and, for a precomputed dataset,
`curl http://localhost:8000/data/<mrc_path>/info`.

## Release (build + push the image)

Pushing a `v*.*.*` git tag is the release trigger: the `Build Docker Image`
workflow runs the tests, then builds `deploy/Dockerfile` and pushes it to
`ghcr.io/janeliascicomp/mrc-ng-server` with semver tags (e.g. `0.1.2`, `0.1`,
`0`, `latest`). The leading `v` is stripped, so `v0.1.2` publishes image tag
`0.1.2`.

Steps to cut release `v0.1.2`:

1. **Bump `version` in `pyproject.toml`** to `0.1.2`. Nothing enforces that this
   matches the tag, so do it by hand first. It's what `importlib.metadata`
   reports at `/healthz` and in `GENERATOR_VERSION`; the image tag a deployment
   pulls comes from the git tag, not here. No relock is needed — the project's
   own version isn't recorded in `pixi.lock`, so `pixi install --locked` still
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

A consumer that pins mrc-ng-server as a git dependency (e.g. for the
`mrc-pyramid build --from-file` CLI) can pin the same tag, so one tag serves both
the image and the dependency.

(To skip the manual bump and derive `version` from the tag automatically, switch
the build to the `hatch-vcs` plugin; note the Docker build excludes `.git`, so
you'd pass the version in as a build-arg rather than have hatch-vcs read git
history.)

Manual build (no CI, e.g. local testing):

```bash
docker build -f deploy/Dockerfile -t ghcr.io/janeliascicomp/mrc-ng-server:<tag> .
docker push ghcr.io/janeliascicomp/mrc-ng-server:<tag>
```

## Scaling

The service is stateless; run more replicas (or raise `NUM_WORKERS`) to add
throughput. The bottleneck is storage I/O, not CPU.
