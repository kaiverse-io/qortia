# Postgres + migrations baked in, so `docker compose up` doesn't need a
# docker-entrypoint-initdb.d bind mount. Some Docker hosts refuse bind
# mounts entirely (Docker Desktop's file-sharing allowlist not covering the
# workspace, for example) — build-context COPY has no such dependency.
FROM pgvector/pgvector:pg16

COPY *.sql /docker-entrypoint-initdb.d/
