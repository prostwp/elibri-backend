-- 011 — Public scenarios for the SaaS catalog
-- Background:
--   v3 redesign turns the strategy builder into a SaaS where the home/catalog
--   shows publicly-visible scenarios from authors. Public-flagged strategies
--   render on / and /scenarios without auth; clone_count drives "Trending"
--   sort. tier on users prepares the paywall (free/pro/premium) — the actual
--   billing endpoint is a stub for now.

ALTER TABLE strategies
  ADD COLUMN IF NOT EXISTS is_public      BOOLEAN     NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS is_featured    BOOLEAN     NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS clone_count    INTEGER     NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS preview_chip   VARCHAR(120),               -- short tagline for the card
  ADD COLUMN IF NOT EXISTS published_at   TIMESTAMPTZ;                -- NULL = unpublished

ALTER TABLE users
  ADD COLUMN IF NOT EXISTS tier           VARCHAR(20) NOT NULL DEFAULT 'free';

-- Constraint after column add — separate DO block to preserve idempotency.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'users_tier_check') THEN
    ALTER TABLE users ADD CONSTRAINT users_tier_check
      CHECK (tier IN ('free','pro','premium'));
  END IF;
END $$;

-- Hot path: catalog page lists is_public=TRUE rows ordered by featured then
-- popularity then recency. Partial index keeps the unrelated 99%+ private
-- rows out of the catalog index.
CREATE INDEX IF NOT EXISTS idx_strategies_public_featured
  ON strategies(is_featured DESC, clone_count DESC, published_at DESC NULLS LAST)
  WHERE is_public = TRUE;

CREATE TABLE IF NOT EXISTS scenario_views (
  id           BIGSERIAL PRIMARY KEY,
  strategy_id  UUID        NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
  viewer_user  UUID                 REFERENCES users(id),    -- NULL for anonymous
  viewed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ip_hash      CHAR(64)                                       -- SHA256(ip+salt) for unique-counter
);
CREATE INDEX IF NOT EXISTS idx_scenario_views_sid_time
  ON scenario_views(strategy_id, viewed_at DESC);
