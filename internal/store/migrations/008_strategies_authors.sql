-- 008_strategies_authors.sql — author metadata on strategies
--
-- Background:
--   Strategy rows used to be 1:1 with end-users ("my personal strategy").
--   With the V4 author pivot (project_v4_authors_real in agent memory)
--   each row now also represents a *named author* whose signals fan out
--   to a Telegram bot menu. Adding metadata so:
--     - runner can build per-author Redis Stream keys (signals:<slug>:...)
--     - Telegram bot can list authors by theme/style and route clicks
--     - frontend (later) can render an author directory page
--     - premium tier (BTC ML model) is flagged separately from human-author
--       scenarios for paywall logic.
--
-- Backwards compatibility:
--   All new columns are nullable / default-set so existing strategies
--   keep working unchanged. Author slug is unique only when not null —
--   personal strategies (no author) skip the slug check entirely.

ALTER TABLE strategies
  ADD COLUMN IF NOT EXISTS author_slug VARCHAR(64),       -- url-safe id, e.g. "gold_news", "ta_1"
  ADD COLUMN IF NOT EXISTS author_name VARCHAR(120),      -- display name, e.g. "Vanya Andreev"
  ADD COLUMN IF NOT EXISTS author_theme VARCHAR(64),      -- "gold_silver", "currencies", "indices", "oil_gas", "crypto", "multi"
  ADD COLUMN IF NOT EXISTS author_style VARCHAR(64),      -- "news", "fundamental", "technical", "astro", "levels"
  ADD COLUMN IF NOT EXISTS author_bio TEXT,               -- short blurb for bot/site display
  ADD COLUMN IF NOT EXISTS is_premium BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS author_position SMALLINT;      -- order in bot menu (1-99); NULL = hidden from menu

-- Slug must be unique among author-strategies, but personal user strategies
-- (where slug is NULL) are unaffected. Partial index supports this cleanly.
CREATE UNIQUE INDEX IF NOT EXISTS idx_strategies_author_slug_unique
  ON strategies(author_slug)
  WHERE author_slug IS NOT NULL;

-- Bot menu listing query: WHERE author_slug IS NOT NULL ORDER BY author_position.
-- Index supports both filter and order without a sort step.
CREATE INDEX IF NOT EXISTS idx_strategies_author_listing
  ON strategies(author_position, author_slug)
  WHERE author_slug IS NOT NULL AND is_active = TRUE;

-- Premium subscription field on users — needed for paywall on premium scenarios.
-- Default FALSE: existing users are non-premium, admin can flip via SQL or
-- a future /admin/users/:id/premium endpoint.
ALTER TABLE users
  ADD COLUMN IF NOT EXISTS is_premium_subscriber BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS premium_until TIMESTAMPTZ;     -- NULL = forever-trial / lifetime; otherwise expiry
