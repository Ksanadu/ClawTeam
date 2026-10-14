-- memory_module_v2: Structured Distillation schema
-- Requires: PostgreSQL with pgvector extension

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS memory_v2;

-- 1) memory_exchanges: evidence layer + keyword corpus
DROP TABLE IF EXISTS memory_v2.memory_exchanges;
CREATE TABLE IF NOT EXISTS memory_v2.memory_exchanges (
  exchange_id      text PRIMARY KEY,

  session_id       text NOT NULL,
  ply_start        integer NOT NULL,
  ply_end          integer NOT NULL,

  verbatim_text    text NOT NULL,
  verbatim_snippet text NOT NULL,

  created_at       timestamptz NOT NULL DEFAULT now(),
  updated_at       timestamptz NOT NULL DEFAULT now(),

  message_count    integer NOT NULL DEFAULT 0,
  has_substantive_assistant boolean NOT NULL DEFAULT false
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_exchanges_session_ply
  ON memory_v2.memory_exchanges(session_id, ply_start, ply_end);

CREATE INDEX IF NOT EXISTS ix_memory_exchanges_session_id
  ON memory_v2.memory_exchanges(session_id);

CREATE INDEX IF NOT EXISTS ix_memory_exchanges_created_at
  ON memory_v2.memory_exchanges(created_at);

-- 2) memory_objects: index layer + dense retrieval
-- Replace vector(1024) with your embedding dimension if different
DROP TABLE IF EXISTS memory_v2.memory_objects;
CREATE TABLE IF NOT EXISTS memory_v2.memory_objects (
  object_id        text PRIMARY KEY,
  exchange_id      text NOT NULL REFERENCES memory_v2.memory_exchanges(exchange_id) ON DELETE CASCADE,

  session_id       text NOT NULL,
  ply_start        integer NOT NULL,
  ply_end          integer NOT NULL,

  exchange_core    text NOT NULL,
  specific_context text NOT NULL,
  distill_text     text NOT NULL,

  room_assignments jsonb NOT NULL DEFAULT '[]'::jsonb,
  files_touched    jsonb NOT NULL DEFAULT '[]'::jsonb,

  distill_provider text NOT NULL,
  distill_model    text NOT NULL,
  distilled_at     timestamptz NOT NULL DEFAULT now(),

  embedding        vector(1024)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_objects_exchange
  ON memory_v2.memory_objects(exchange_id);

CREATE INDEX IF NOT EXISTS ix_memory_objects_session_id
  ON memory_v2.memory_objects(session_id);

CREATE INDEX IF NOT EXISTS ix_memory_objects_distilled_at
  ON memory_v2.memory_objects(distilled_at);

CREATE INDEX IF NOT EXISTS ix_memory_objects_room_assignments_gin
  ON memory_v2.memory_objects USING GIN (room_assignments);

CREATE INDEX IF NOT EXISTS ix_memory_objects_files_touched_gin
  ON memory_v2.memory_objects USING GIN (files_touched);

-- Vector index (enable when data grows):
-- CREATE INDEX IF NOT EXISTS ix_memory_objects_embedding_ivfflat
--   ON memory_v2.memory_objects USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
