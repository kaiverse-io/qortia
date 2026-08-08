# LiteLLM gateway with config.yaml baked in, so `docker compose up` doesn't
# need to bind-mount it. Some Docker hosts refuse bind mounts entirely
# (Docker Desktop's file-sharing allowlist not covering the workspace, for
# example) — build-context COPY has no such dependency.
FROM ghcr.io/berriai/litellm:main-v1.74.3-stable

COPY config.yaml /app/config.yaml
