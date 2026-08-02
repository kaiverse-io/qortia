#!/usr/bin/env bash
# Live E2E: Postgres + mock LiteLLM + API + worker → remember → embed → recall
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PG_NAME=qortia-e2e-pg
PG_PORT=5434
MOCK_PORT=4000
API_PORT=8090
WORKDIR="${TMPDIR:-/tmp}/qortia-e2e-$$"
mkdir -p "$WORKDIR"
LOG="$WORKDIR/e2e.log"
cleanup() {
  set +e
  [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null
  [[ -n "${WORKER_PID:-}" ]] && kill "$WORKER_PID" 2>/dev/null
  [[ -n "${MOCK_PID:-}" ]] && kill "$MOCK_PID" 2>/dev/null
  docker rm -f "$PG_NAME" >/dev/null 2>&1
}
trap cleanup EXIT

echo "==> Postgres (pgvector) on :$PG_PORT"
docker rm -f "$PG_NAME" >/dev/null 2>&1 || true
docker run -d --name "$PG_NAME" \
  -e POSTGRES_DB=qortia \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=test \
  -p "${PG_PORT}:5432" \
  pgvector/pgvector:pg16 >/dev/null

for i in $(seq 1 60); do
  if docker exec "$PG_NAME" pg_isready -U postgres -d qortia >/dev/null 2>&1 \
    && docker exec "$PG_NAME" psql -U postgres -d qortia -c 'SELECT 1' >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done
docker exec "$PG_NAME" psql -U postgres -d qortia -c 'SELECT 1' >/dev/null

echo "==> Apply migrations"
# Retry once — first query after boot can race with recovery.
for attempt in 1 2 3; do
  if docker exec -i "$PG_NAME" psql -U postgres -d qortia < migrations/V1__initial_schema.sql >/dev/null; then
    break
  fi
  sleep 1
  [[ "$attempt" == "3" ]] && exit 1
done

export QORTIA_DATABASE_URL="postgresql://qortia_platform:qortia_platform@127.0.0.1:${PG_PORT}/qortia"
export QORTIA_LITELLM_URL="http://127.0.0.1:${MOCK_PORT}"
export QORTIA_LITELLM_API_KEY="sk-e2e-test"
export QORTIA_EMBEDDING_MODEL="bge-m3"
export QORTIA_EMBEDDING_DIMENSION="1024"

echo "==> Mock LiteLLM embeddings on :$MOCK_PORT"
uv run python scripts/mock_litellm_embeddings.py >"$WORKDIR/mock.log" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 40); do
  curl -sf "http://127.0.0.1:${MOCK_PORT}/health" >/dev/null && break
  sleep 0.25
done
curl -sf "http://127.0.0.1:${MOCK_PORT}/health" | tee -a "$LOG"

echo "==> Probe /embeddings (dim check)"
DIM=$(curl -sf "http://127.0.0.1:${MOCK_PORT}/embeddings" \
  -H "Authorization: Bearer $QORTIA_LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-m3","input":"dimension check"}' \
  | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data'][0]['embedding']))")
echo "mock embedding dim=$DIM"
[[ "$DIM" == "1024" ]] || { echo "FAIL: expected 1024"; exit 1; }

echo "==> Provision tenant/agent/key"
PROV=$(uv run python - <<'PY'
import asyncio, os
import asyncpg
from qortia.provisioning import create_tenant, create_agent, issue_api_key

async def main():
    pool = await asyncpg.create_pool(os.environ["QORTIA_DATABASE_URL"])
    try:
        tid = await create_tenant(pool, name="e2e")
        aid = await create_agent(pool, tid, clearance_level="internal", division="all")
        # make agent a chief for knowledge paths if needed; remember works for engineer
        key = await issue_api_key(pool, tid)
        print(f"{tid},{aid},{key}")
    finally:
        await pool.close()
asyncio.run(main())
PY
)
IFS=',' read -r TENANT_ID AGENT_ID API_KEY <<<"$PROV"
echo "tenant=$TENANT_ID agent=$AGENT_ID"

echo "==> Start API :$API_PORT"
uv run uvicorn qortia.app:app --host 127.0.0.1 --port "$API_PORT" >"$WORKDIR/api.log" 2>&1 &
API_PID=$!
for i in $(seq 1 40); do
  curl -sf "http://127.0.0.1:${API_PORT}/docs" >/dev/null && break
  sleep 0.25
done
curl -sf "http://127.0.0.1:${API_PORT}/docs" >/dev/null

echo "==> Start worker"
uv run qortia-worker --only embed >"$WORKDIR/worker.log" 2>&1 &
WORKER_PID=$!
sleep 2

echo "==> Remember episodic memory"
REM=$(curl -sf "http://127.0.0.1:${API_PORT}/v1/remember" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "X-Agent-Id: ${AGENT_ID}" \
  -H "Content-Type: application/json" \
  -d '{
    "memories": [{
      "type": "episodic",
      "content": "Qortia live E2E decided to use Postgres pgvector with BGE-M3 style embeddings for hybrid recall testing."
    }]
  }') || {
  echo "FAIL: remember HTTP error"
  tail -50 "$WORKDIR/api.log"
  exit 1
}
echo "$REM" | tee -a "$LOG"
MEM_ID=$(echo "$REM" | python3 -c "import sys,json; d=json.load(sys.stdin); ids=d.get('ids') or []; print(ids[0] if ids else '')")
[[ -n "$MEM_ID" ]] || { echo "FAIL: no memory id"; cat "$WORKDIR/api.log"; exit 1; }

echo "==> Wait for embedding worker to fill vector"
EMBEDDED=0
for i in $(seq 1 30); do
  GOT=$(docker exec "$PG_NAME" psql -U postgres -d qortia -Atc \
    "SELECT CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END FROM hindsight_memories WHERE id='${MEM_ID}'")
  if [[ "$GOT" == "1" ]]; then
    EMBEDDED=1
    break
  fi
  sleep 1
done
echo "embedded=$EMBEDDED after ${i}s"
[[ "$EMBEDDED" == "1" ]] || {
  echo "FAIL: embedding never filled"
  echo "--- worker log ---"; tail -50 "$WORKDIR/worker.log"
  echo "--- api log ---"; tail -30 "$WORKDIR/api.log"
  echo "--- mock log ---"; tail -30 "$WORKDIR/mock.log"
  exit 1
}

DIM_DB=$(docker exec "$PG_NAME" psql -U postgres -d qortia -Atc \
  "SELECT vector_dims(embedding) FROM hindsight_memories WHERE id='${MEM_ID}'")
echo "db vector_dims=$DIM_DB"
[[ "$DIM_DB" == "1024" ]] || { echo "FAIL: db dim"; exit 1; }

echo "==> Recall (vector+bm25 hybrid)"
REC=$(curl -sf "http://127.0.0.1:${API_PORT}/v1/recall" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "X-Agent-Id: ${AGENT_ID}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "pgvector BGE-M3 hybrid recall embeddings",
    "scope": "private",
    "type": "episodic"
  }') || {
  echo "FAIL: recall HTTP error"
  tail -50 "$WORKDIR/api.log"
  exit 1
}
echo "$REC" | python3 -m json.tool | head -40 | tee -a "$LOG"
HIT=$(echo "$REC" | python3 -c "import sys,json; r=json.load(sys.stdin); ids=[x.get('id') for x in r.get('results', [])]; print('1' if '$MEM_ID' in ids else '0'); print('count', len(ids))")
echo "$HIT"
echo "$HIT" | grep -q '^1' || {
  echo "FAIL: memory not in recall results"
  exit 1
}

echo ""
echo "E2E PASS: remember → worker embed (1024-dim) → recall hit"
echo "logs: $WORKDIR"
