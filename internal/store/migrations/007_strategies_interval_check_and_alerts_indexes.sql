-- Migration 007 — strategies.interval CHECK + alerts query indexes
--
-- Two cleanups surfaced by the global review:
--
-- 1) strategies.interval had no CHECK. Nothing prevented a user from
--    saving interval='9h' or '30m' — internal/scenario/interval.go's
--    TFToPoll default branch would return 5 minutes and the bar would
--    never close (we only support 5m/15m/1h/4h/1d in production).
--    Silent misconfiguration. CHECK rejects the junk at the DB layer
--    regardless of any client-side validation that might be missed.
--
-- 2) alerts was scanned by `WHERE user_id=$1 ORDER BY created_at DESC`
--    on every /api/v1/alerts request. With the paper-trading runner
--    generating signals over weeks/months the table grows; a full
--    heap scan on each page load will bite after the first 10k rows.
--    Composite index (user_id, created_at DESC) serves the query
--    without sorting.
--
-- Both are idempotent — safe to re-run.

DO $$
BEGIN
  -- strategies.interval check (normalize existing junk first to avoid CHECK failure).
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'strategies_interval_check'
  ) THEN
    UPDATE strategies
    SET interval = '1h'
    WHERE interval IS NULL OR interval NOT IN ('5m', '15m', '1h', '4h', '1d');

    ALTER TABLE strategies
      ADD CONSTRAINT strategies_interval_check
      CHECK (interval IN ('5m', '15m', '1h', '4h', '1d'));
  END IF;
END $$;

-- alerts query index. CREATE INDEX IF NOT EXISTS is safe to retry.
CREATE INDEX IF NOT EXISTS idx_alerts_user_created
  ON alerts (user_id, created_at DESC);

-- Secondary: per-strategy filter on /api/v1/alerts?strategy_id=...
CREATE INDEX IF NOT EXISTS idx_alerts_strategy_created
  ON alerts (strategy_id, created_at DESC);
