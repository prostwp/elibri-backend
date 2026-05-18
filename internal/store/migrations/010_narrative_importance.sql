-- Migration 010 — Narrative Radar v2: LLM importance classifier columns.
--
-- Background:
--   v1 (migration 009) gave us mention_count + growth_pct + sentiment + source
--   diversity. Trend score was a weighted sum of those four. The user
--   explicitly rejected adding rule-based weights ("UK CPI = boring → score
--   it lower") because magic-number rules don't generalise: the right answer
--   is "ask an LLM how market-moving is this headline".
--
--   This migration adds four nullable columns to narrative_mentions so the
--   classifier worker can run AFTER ingest, score each row in [0, 100], and
--   persist the result alongside a category + one-line LLM rationale. The
--   nullable design lets us deploy this migration BEFORE the classifier code
--   without breaking the worker — rows are inserted unscored, then back-
--   filled on the next pass through the classifier.
--
-- Backfill strategy:
--   The worker uses idx_narrative_mentions_pending_importance to find rows
--   where importance_score IS NULL and classify them. Once classified, the
--   index no longer covers the row — keeping the partial index small even as
--   narrative_mentions grows.
--
-- Idempotent: ADD COLUMN IF NOT EXISTS so re-running the migration is a no-op.
-- The CHECK constraint is added separately because PG doesn't have an
-- "IF NOT EXISTS" form for constraints — we wrap it in a DO block that
-- looks up pg_constraint first.

ALTER TABLE narrative_mentions
  ADD COLUMN IF NOT EXISTS importance_score        SMALLINT,
  ADD COLUMN IF NOT EXISTS importance_category     VARCHAR(32),
  ADD COLUMN IF NOT EXISTS importance_why          TEXT,
  ADD COLUMN IF NOT EXISTS importance_classified_at TIMESTAMPTZ;

-- Range CHECK on importance_score: NULL is allowed (= "not yet classified"),
-- otherwise must be in [0, 100]. The bounded range mirrors the LLM contract.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'narrative_mentions_importance_score_chk'
  ) THEN
    ALTER TABLE narrative_mentions
      ADD CONSTRAINT narrative_mentions_importance_score_chk
      CHECK (importance_score IS NULL OR (importance_score BETWEEN 0 AND 100));
  END IF;
END
$$;

-- Worker query hot-path: "find every unscored mention, oldest first".
-- Partial index with WHERE importance_score IS NULL keeps the index tiny —
-- once a row is classified it falls out of the index entirely.
CREATE INDEX IF NOT EXISTS idx_narrative_mentions_pending_importance
  ON narrative_mentions (ingested_at)
  WHERE importance_score IS NULL;
