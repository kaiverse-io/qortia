# Qortia application image — API (qortia.app:app) and worker (qortia-worker)
# share this image; docker-compose.yml picks the entrypoint per service.
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev

EXPOSE 8080

CMD ["uvicorn", "qortia.app:app", "--host", "0.0.0.0", "--port", "8080"]
