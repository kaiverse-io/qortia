---
audience: contributor
last_reviewed: <!-- update when reviewing -->
---

# Contributing to Qortia

## Workflow

1. Branch off `main`.
2. Make changes; run `just fmt` then `just ci` locally — CI must be green before pushing.
3. Commit with [conventional commits](https://www.conventionalcommits.org/):
   `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `perf:`.
4. Open a PR; all CI checks must pass.

## Running tests

`just ci-test` (or `uv run pytest`) runs both `tests/unit` and `tests/integration`.
Integration tests need Docker: they spin up a real `pgvector/pgvector:pg16`
container via [testcontainers](https://testcontainers-python.readthedocs.io/),
apply `migrations/V1__initial_schema.sql` against it, and exercise the actual
`qortia.app` FastAPI app over HTTP — no external services or manual setup
required on a normal Docker host (including GitHub Actions' `ubuntu-latest`
runners).

**Devcontainer-specific note:** this repo's own devcontainer uses
docker-outside-of-docker (forwards the host's `docker.sock`, sibling
containers rather than nested ones — see `.devcontainer/devcontainer.json`).
On some sandboxed/virtualized hosts, the Docker daemon reachable through that
forwarded socket lives in a different network namespace than the devcontainer
itself — only the Docker control API (via the socket) is reachable, not raw
TCP to a sibling container's published port. Symptom: integration tests fail
with `ConnectionRefusedError`/`OSError` connecting to `172.17.0.1` or
`localhost`. If you hit this:

```bash
docker network connect bridge $(hostname)   # put this container on the same network as testcontainers' siblings
TESTCONTAINERS_RYUK_DISABLED=true TESTCONTAINERS_CONNECTION_MODE=bridge_ip uv run pytest tests/integration
```

This is not needed on a normal (non-nested) Docker host or in CI — only set
these if the plain `uv run pytest tests/integration` fails with a connection
error first.

## Decisions

Significant decisions go in `docs/decisions/adrs/` as [MADR](https://adr.github.io/madr/) files.
Copy `docs/decisions/adrs/adr-000-madr-template.md`, increment the number, fill it in.

## Secrets

- Never commit secrets or credentials.
- Copy `.env.example` to `.env` (gitignored) and fill in values.
- `gitleaks` runs in pre-commit and will block accidental leaks.

## Guardrail rules

See the **Forbidden patterns** section of `AGENTS.md` — these apply to humans and agents equally.
