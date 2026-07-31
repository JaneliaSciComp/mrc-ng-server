# syntax=docker/dockerfile:1.7
# mrc-ng-server: serves MRC tomograms to Neuroglancer via the precomputed
# protocol. Build context: repo root.
#
# Serves plain HTTP on :8000. TLS is terminated upstream (the OpenShift Route,
# see deploy/k8s/base/route.yaml) — the pod never sees the cert. This differs
# from the `serve` pixi task, which binds HTTPS directly for local dev.

FROM ghcr.io/prefix-dev/pixi:0.73.0-bookworm-slim AS base
WORKDIR /app

# Resolve and materialize the environment from pixi.lock. The project installs
# itself editable (pyproject `mrc_ng_server = { path = ".", editable = true }`),
# so src must be present at install time and stay at the same path at runtime.
FROM base AS deps
COPY pixi.lock pyproject.toml ./
COPY src ./src
RUN pixi install --locked -e default

FROM base AS runtime
# Copy the whole resolved tree (env + editable-linked src + manifests) so the
# editable install's path link stays valid.
COPY --from=deps /app /app

# OpenShift runs the pod as an arbitrary non-root UID in gid 0. The long-running
# uvicorn task never exits, but a one-shot pixi task (e.g. `pixi run build-cache`
# as a Job) would write a task cache under /app/.pixi and fail with "Permission
# denied" unless /app is group-writable. Mirror the ai-cryoet images.
RUN chgrp -R 0 /app && chmod -R g=u /app

# MRCNG_SOURCE_ROOT must match the mountPath of the data volume and the path the
# scanner recorded on each tomogram row (the API builds precomputed:// URLs from
# mrc_path relative to that root). MRCNG_CACHE_ROOT is the shared pyramid cache
# the scanner writes and this server reads.
ENV MRCNG_SOURCE_ROOT=/groups/cryoet/cryoet/data \
    MRCNG_CACHE_ROOT=/cache \
    MRCNG_CORS_ORIGINS=*

EXPOSE 8000
CMD ["pixi", "run", "uvicorn", "mrcng.server.app:get_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
