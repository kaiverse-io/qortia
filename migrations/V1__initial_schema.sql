-- Qortia — standalone initial schema.
--
-- A fresh squashed migration, not a replay of extraction history: this repo
-- was extracted from a larger host platform whose schema FK'd qortia's memory
-- tables against that platform's own auth.tenants/auth.agents and gated org
-- reads on a full Vault/JWT/RBAC stack. Standalone, qortia owns a minimal
-- identity substrate of its own (qortia_tenants/qortia_agents/qortia_api_keys)
-- — just enough for RLS + API-key auth — not a general-purpose tenant/user
-- management system.
--
-- Run as a Postgres superuser: creates the qortia_platform role the app
-- connects as, and every RLS policy is written assuming that role name.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'qortia_platform') THEN
        CREATE ROLE qortia_platform LOGIN PASSWORD 'qortia_platform';
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO qortia_platform;

-- ── Identity substrate ───────────────────────────────────────────────────────

CREATE TABLE qortia_tenants (
    id                          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name                        TEXT,
    status                      TEXT        NOT NULL DEFAULT 'active',
    weekly_summary_last_run_at  TIMESTAMPTZ,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE qortia_agents (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID        NOT NULL REFERENCES qortia_tenants(id) ON DELETE CASCADE,
    name                TEXT,
    role                TEXT        NOT NULL DEFAULT 'engineer',
    status              TEXT        NOT NULL DEFAULT 'active',
    clearance_level     TEXT        NOT NULL DEFAULT 'internal',
    division            TEXT        NOT NULL DEFAULT 'all',
    reflection_counter  INTEGER     NOT NULL DEFAULT 0,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_qortia_agents_tenant ON qortia_agents (tenant_id);

-- API keys authenticate a *tenant*, not an individual agent — the caller
-- names which agent it's acting as via the X-Agent-Id header, and
-- qortia.auth.require_agent validates that agent belongs to the key's
-- tenant. Only key_hash is ever stored; plaintext is returned once at
-- issuance (qortia.provisioning.issue_api_key) and never persisted.
CREATE TABLE qortia_api_keys (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   UUID        NOT NULL REFERENCES qortia_tenants(id) ON DELETE CASCADE,
    key_hash    TEXT        NOT NULL UNIQUE,
    revoked_at  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_qortia_api_keys_hash ON qortia_api_keys (key_hash) WHERE revoked_at IS NULL;

-- Global (not tenant-scoped) clearance lookup — simplified from the original
-- per-tenant-customizable design. Per-tenant customization is a clean future
-- extension, not needed for RBAC-gated org reads to work correctly today.
CREATE TABLE qortia_clearance_levels (
    level_name  TEXT PRIMARY KEY,
    level_order INTEGER NOT NULL
);

INSERT INTO qortia_clearance_levels (level_name, level_order) VALUES
    ('external', 1),
    ('internal', 2),
    ('restricted', 3);

GRANT SELECT, INSERT, UPDATE, DELETE ON qortia_tenants, qortia_agents, qortia_api_keys
    TO qortia_platform;
GRANT SELECT ON qortia_clearance_levels TO qortia_platform;

-- ── Private memory ───────────────────────────────────────────────────────────

CREATE TABLE hindsight_memories (
    id                      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID        NOT NULL REFERENCES qortia_tenants(id),
    agent_id                UUID        NOT NULL REFERENCES qortia_agents(id) ON DELETE CASCADE,
    type                    TEXT        NOT NULL
        CHECK (type IN ('episodic', 'experiential', 'mental_model', 'decision', 'lesson', 'short_term')),
    content                 TEXT        NOT NULL,
    content_tsv             TSVECTOR    GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    embedding               vector(1024),
    embedding_attempts      SMALLINT    NOT NULL DEFAULT 0,
    importance              FLOAT       NOT NULL DEFAULT 0.5 CHECK (importance BETWEEN 0 AND 1),
    is_consolidated         BOOLEAN     NOT NULL DEFAULT false,
    stability_score         FLOAT       CHECK (stability_score BETWEEN 0 AND 1),
    is_graphed              BOOLEAN     NOT NULL DEFAULT false,
    tier                    TEXT        NOT NULL DEFAULT 'active' CHECK (tier IN ('active', 'archive')),
    expires_at              TIMESTAMPTZ,
    source_task_id          UUID,
    metadata                JSONB       DEFAULT '{}',
    entities                JSONB       NOT NULL DEFAULT '[]',
    recall_count            SMALLINT    NOT NULL DEFAULT 0,
    last_recalled_at        TIMESTAMPTZ,
    lang                    TEXT        NOT NULL DEFAULT 'en',
    content_hash            TEXT,
    valid_from              TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_until             TIMESTAMPTZ,
    confidence_multiplier   FLOAT       NOT NULL DEFAULT 1.0,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON hindsight_memories (tenant_id, agent_id, type);
CREATE INDEX ON hindsight_memories (tenant_id, agent_id, created_at DESC);
CREATE INDEX ON hindsight_memories (tenant_id, agent_id, importance DESC);
CREATE INDEX ON hindsight_memories USING GIN (content_tsv);
CREATE INDEX hindsight_memories_embedding_active_idx ON hindsight_memories
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE tier = 'active';
CREATE INDEX ON hindsight_memories (tenant_id, agent_id) WHERE embedding IS NULL AND embedding_attempts < 3;
CREATE INDEX ON hindsight_memories (agent_id, type, importance DESC) WHERE is_consolidated = true;
CREATE INDEX ON hindsight_memories (agent_id, created_at DESC) WHERE type = 'decision';
CREATE INDEX idx_hindsight_entities ON hindsight_memories USING GIN (entities);
CREATE INDEX ON hindsight_memories (is_graphed) WHERE is_graphed = false;
CREATE INDEX ON hindsight_memories (agent_id, type, stability_score DESC)
    WHERE is_consolidated = true AND stability_score IS NOT NULL;
CREATE INDEX idx_hindsight_tier ON hindsight_memories (agent_id, tier, created_at DESC);
CREATE INDEX idx_hindsight_expiry ON hindsight_memories (agent_id, expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_hindsight_valid_from ON hindsight_memories (agent_id, valid_from);
CREATE INDEX idx_hindsight_valid_until ON hindsight_memories (agent_id, valid_until) WHERE valid_until IS NOT NULL;

ALTER TABLE hindsight_memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE hindsight_memories FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON hindsight_memories
    FOR ALL
    USING (tenant_id::text = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id::text = current_setting('app.tenant_id', true));

CREATE POLICY agent_read_isolation ON hindsight_memories
    AS RESTRICTIVE FOR SELECT
    USING (
        current_user = 'qortia_platform'
        OR agent_id::text = current_setting('app.agent_id', true)
    );

CREATE POLICY platform_write ON hindsight_memories
    FOR ALL TO qortia_platform
    USING (true) WITH CHECK (true);

GRANT SELECT, INSERT, UPDATE, DELETE ON hindsight_memories TO qortia_platform;

-- ── Org memory ────────────────────────────────────────────────────────────────

CREATE TABLE org_memory (
    id                      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID        NOT NULL REFERENCES qortia_tenants(id),
    type                    TEXT        NOT NULL
        CHECK (type IN ('org_chart', 'process', 'handoff', 'weekly_summary', 'decision_log')),
    title                   TEXT        NOT NULL,
    content                 TEXT        NOT NULL,
    content_tsv             TSVECTOR    GENERATED ALWAYS AS (to_tsvector('simple', title || ' ' || content)) STORED,
    embedding               vector(1024),
    embedding_attempts      SMALLINT    NOT NULL DEFAULT 0,
    author_id               UUID        REFERENCES qortia_agents(id) ON DELETE SET NULL,
    metadata                JSONB       DEFAULT '{}',
    entities                JSONB       NOT NULL DEFAULT '[]',
    is_graphed              BOOLEAN     NOT NULL DEFAULT false,
    recall_count            SMALLINT    NOT NULL DEFAULT 0,
    last_recalled_at        TIMESTAMPTZ,
    lang                    TEXT        NOT NULL DEFAULT 'en',
    min_clearance           TEXT        NOT NULL DEFAULT 'internal',
    audience                TEXT[]      NOT NULL DEFAULT '{all}',
    confidence_multiplier   FLOAT       NOT NULL DEFAULT 1.0,
    valid_from              TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_until             TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON org_memory (tenant_id, type);
CREATE INDEX ON org_memory (tenant_id, created_at DESC);
CREATE UNIQUE INDEX org_memory_org_chart_per_agent ON org_memory (tenant_id, author_id) WHERE type = 'org_chart';
CREATE UNIQUE INDEX org_memory_upsertable_per_title ON org_memory (tenant_id, type, title)
    WHERE type IN ('process', 'decision_log');
CREATE INDEX ON org_memory USING GIN (content_tsv);
CREATE INDEX ON org_memory USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX ON org_memory (tenant_id) WHERE embedding IS NULL AND embedding_attempts < 3;
CREATE INDEX idx_org_memory_entities ON org_memory USING GIN (entities);
CREATE INDEX ON org_memory (is_graphed) WHERE is_graphed = false;
CREATE INDEX ON org_memory (tenant_id, min_clearance);
CREATE INDEX ON org_memory USING GIN (audience);
CREATE INDEX idx_org_memory_valid_from ON org_memory (tenant_id, valid_from);
CREATE INDEX idx_org_memory_valid_until ON org_memory (tenant_id, valid_until) WHERE valid_until IS NOT NULL;

ALTER TABLE org_memory ENABLE ROW LEVEL SECURITY;
ALTER TABLE org_memory FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_visibility_read ON org_memory
    FOR SELECT USING (
        tenant_id::text = current_setting('app.tenant_id', true)
        AND (
            coalesce(nullif(current_setting('app.memory_clearance_order', true), ''), '2')::integer
            >= (SELECT level_order FROM qortia_clearance_levels WHERE level_name = org_memory.min_clearance)
        )
        AND (
            coalesce(nullif(current_setting('app.agent_division', true), ''), 'all') = ANY(org_memory.audience)
            OR 'all' = ANY(org_memory.audience)
        )
    );

CREATE POLICY platform_write ON org_memory FOR ALL TO qortia_platform USING (true) WITH CHECK (true);

GRANT SELECT, INSERT, UPDATE, DELETE ON org_memory TO qortia_platform;

-- ── Org knowledge ─────────────────────────────────────────────────────────────

CREATE TABLE org_knowledge (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID        NOT NULL REFERENCES qortia_tenants(id),
    source_type         TEXT        NOT NULL CHECK (source_type IN ('file', 'url', 'transcript', 'note')),
    source_path         TEXT        NOT NULL,
    chunk_index         INTEGER     NOT NULL,
    content             TEXT        NOT NULL,
    content_hash        TEXT        NOT NULL,
    index_summary       TEXT,
    index_questions     TEXT,
    index_entities      TEXT,
    index_tsv           TSVECTOR    GENERATED ALWAYS AS (
                            to_tsvector('simple',
                                coalesce(index_summary, '') || ' ' ||
                                coalesce(index_questions, '') || ' ' ||
                                coalesce(index_entities, '')
                            )
                        ) STORED,
    embedding           vector(1024),
    embedding_attempts  SMALLINT    NOT NULL DEFAULT 0,
    author_id           UUID        REFERENCES qortia_agents(id) ON DELETE SET NULL,
    metadata            JSONB       DEFAULT '{}',
    recall_count        SMALLINT    NOT NULL DEFAULT 0,
    last_recalled_at    TIMESTAMPTZ,
    lang                TEXT        NOT NULL DEFAULT 'en',
    min_clearance       TEXT        NOT NULL DEFAULT 'internal',
    audience            TEXT[]      NOT NULL DEFAULT '{all}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, source_path, chunk_index)
);

CREATE INDEX ON org_knowledge (tenant_id, source_path);
CREATE INDEX ON org_knowledge (tenant_id, content_hash);
CREATE INDEX ON org_knowledge (tenant_id, created_at DESC);
CREATE INDEX ON org_knowledge USING GIN (index_tsv);
CREATE INDEX ON org_knowledge USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX ON org_knowledge (tenant_id) WHERE embedding IS NULL AND embedding_attempts < 3;
CREATE INDEX ON org_knowledge (tenant_id, min_clearance);
CREATE INDEX ON org_knowledge USING GIN (audience);

ALTER TABLE org_knowledge ENABLE ROW LEVEL SECURITY;
ALTER TABLE org_knowledge FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_visibility_read ON org_knowledge
    FOR SELECT USING (
        tenant_id::text = current_setting('app.tenant_id', true)
        AND (
            coalesce(nullif(current_setting('app.memory_clearance_order', true), ''), '2')::integer
            >= (SELECT level_order FROM qortia_clearance_levels WHERE level_name = org_knowledge.min_clearance)
        )
        AND (
            coalesce(nullif(current_setting('app.agent_division', true), ''), 'all') = ANY(org_knowledge.audience)
            OR 'all' = ANY(org_knowledge.audience)
        )
    );

CREATE POLICY platform_write ON org_knowledge FOR ALL TO qortia_platform USING (true) WITH CHECK (true);

GRANT SELECT, INSERT, UPDATE, DELETE ON org_knowledge TO qortia_platform;

-- ── Entity graph ──────────────────────────────────────────────────────────────

CREATE TABLE qortia_entities (
    id                      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID        NOT NULL REFERENCES qortia_tenants(id) ON DELETE CASCADE,
    agent_id                UUID        REFERENCES qortia_agents(id) ON DELETE CASCADE,
    entity_text             TEXT        NOT NULL,
    entity_type             TEXT        NOT NULL,
    embedding               vector(1024),
    embedding_attempts      SMALLINT    NOT NULL DEFAULT 0,
    linked_memory_ids       UUID[]      NOT NULL DEFAULT '{}',
    summary                 TEXT,
    max_clearance_order     INTEGER     NOT NULL DEFAULT 2,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX idx_qortia_entities_unique_org ON qortia_entities (tenant_id, entity_text) WHERE agent_id IS NULL;
CREATE UNIQUE INDEX idx_qortia_entities_unique_agent ON qortia_entities (tenant_id, agent_id, entity_text)
    WHERE agent_id IS NOT NULL;
CREATE INDEX ON qortia_entities USING GIN (linked_memory_ids);
CREATE INDEX ON qortia_entities USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX ON qortia_entities (tenant_id) WHERE embedding IS NULL AND embedding_attempts < 3;

ALTER TABLE qortia_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE qortia_entities FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON qortia_entities
    FOR ALL
    USING (tenant_id::text = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id::text = current_setting('app.tenant_id', true));

CREATE POLICY agent_read_isolation ON qortia_entities
    AS RESTRICTIVE FOR SELECT
    USING (
        current_user = 'qortia_platform'
        OR agent_id IS NULL
        OR agent_id::text = current_setting('app.agent_id', true)
    );

CREATE POLICY platform_write ON qortia_entities FOR ALL TO qortia_platform USING (true) WITH CHECK (true);

GRANT SELECT, INSERT, UPDATE, DELETE ON qortia_entities TO qortia_platform;

-- ── Cross-memory links ────────────────────────────────────────────────────────

CREATE TABLE memory_links (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   UUID        NOT NULL REFERENCES qortia_tenants(id) ON DELETE CASCADE,
    source_id   UUID        NOT NULL,
    target_id   UUID        NOT NULL,
    similarity  FLOAT       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_id, target_id)
);

CREATE INDEX ON memory_links (source_id);
CREATE INDEX ON memory_links (target_id);

ALTER TABLE memory_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_links FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON memory_links
    FOR ALL
    USING (tenant_id::text = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id::text = current_setting('app.tenant_id', true));

CREATE POLICY platform_write ON memory_links FOR ALL TO qortia_platform USING (true) WITH CHECK (true);

GRANT SELECT, INSERT, UPDATE, DELETE ON memory_links TO qortia_platform;

-- ── Audit trail ───────────────────────────────────────────────────────────────

CREATE TABLE memory_history (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID        NOT NULL REFERENCES qortia_tenants(id),
    agent_id        UUID        NOT NULL REFERENCES qortia_agents(id) ON DELETE CASCADE,
    operation       TEXT        NOT NULL
        CHECK (operation IN ('remember', 'remember_org', 'forget', 'knowledge_ingest', 'knowledge_delete', 'reflect')),
    target_table    TEXT        NOT NULL,
    target_id       UUID,
    content_hash    TEXT,
    metadata        JSONB       DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON memory_history (tenant_id, agent_id, created_at DESC);
CREATE INDEX ON memory_history (tenant_id, operation, created_at DESC);

ALTER TABLE memory_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_history FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON memory_history
    FOR ALL
    USING (tenant_id::text = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id::text = current_setting('app.tenant_id', true));

CREATE POLICY platform_write ON memory_history FOR ALL TO qortia_platform USING (true) WITH CHECK (true);

-- Append-only audit trail: INSERT only, deliberately no UPDATE/DELETE grant.
GRANT SELECT, INSERT ON memory_history TO qortia_platform;

-- ── ADR-125 causal tracking ───────────────────────────────────────────────────

CREATE TABLE qortia_session_reads (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID        NOT NULL REFERENCES qortia_tenants(id),
    agent_id        UUID        NOT NULL REFERENCES qortia_agents(id) ON DELETE CASCADE,
    work_order_id   UUID        NOT NULL,
    memory_id       UUID        NOT NULL,
    recalled_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_session_reads_wo ON qortia_session_reads (work_order_id);
CREATE INDEX idx_session_reads_memory ON qortia_session_reads (memory_id);
CREATE INDEX idx_session_reads_tenant ON qortia_session_reads (tenant_id, work_order_id);

ALTER TABLE qortia_session_reads ENABLE ROW LEVEL SECURITY;
ALTER TABLE qortia_session_reads FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON qortia_session_reads
    FOR ALL
    USING (tenant_id::text = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id::text = current_setting('app.tenant_id', true));

-- The legacy source of this table was missing a platform_write policy and an
-- explicit GRANT — both added here for consistency with every other table
-- rather than reproducing the gap in a fresh migration.
CREATE POLICY platform_write ON qortia_session_reads FOR ALL TO qortia_platform USING (true) WITH CHECK (true);

GRANT SELECT, INSERT, UPDATE, DELETE ON qortia_session_reads TO qortia_platform;

CREATE TABLE qortia_outcome_records (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID        NOT NULL REFERENCES qortia_tenants(id),
    agent_id        UUID        NOT NULL REFERENCES qortia_agents(id) ON DELETE CASCADE,
    work_order_id   UUID        NOT NULL UNIQUE,
    outcome         TEXT        NOT NULL CHECK (outcome IN ('SUCCESS', 'MINOR_FAILURE', 'CRITICAL_FAILURE')),
    memory_count    INTEGER     NOT NULL DEFAULT 0,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_outcome_records_tenant ON qortia_outcome_records (tenant_id);

ALTER TABLE qortia_outcome_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE qortia_outcome_records FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON qortia_outcome_records
    FOR ALL
    USING (tenant_id::text = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id::text = current_setting('app.tenant_id', true));

CREATE POLICY platform_write ON qortia_outcome_records FOR ALL TO qortia_platform USING (true) WITH CHECK (true);

GRANT SELECT, INSERT, UPDATE, DELETE ON qortia_outcome_records TO qortia_platform;
