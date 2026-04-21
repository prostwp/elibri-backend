-- Migration 006 — strategies.risk_tier CHECK constraint
--
-- Mirrors the users.risk_tier constraint from migration 004. Without this,
-- a frontend or direct SQL insert with risk_tier='yolo' or any other string
-- passes DB validation and then silently falls back to 'balanced' inside
-- ml.GetTier — giving the user a different tier than they asked for.
--
-- Idempotent: does nothing if the constraint already exists.

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'strategies_risk_tier_check'
  ) THEN
    -- Normalize any rogue existing values first (shouldn't happen in prod,
    -- but keeps the constraint-add safe if an experimental row slipped in).
    UPDATE strategies
    SET risk_tier = 'balanced'
    WHERE risk_tier NOT IN ('conservative', 'balanced', 'aggressive');

    ALTER TABLE strategies
      ADD CONSTRAINT strategies_risk_tier_check
      CHECK (risk_tier IN ('conservative', 'balanced', 'aggressive'));
  END IF;
END $$;
