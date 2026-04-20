-- Patch 2C: per-user risk tier. Drives tier-based vol gates,
-- confidence thresholds, and position-size hints in /ml/predict.
-- Existing users default to 'balanced' — matches the Pro segment in UI.

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS risk_tier VARCHAR(20) NOT NULL DEFAULT 'balanced';

-- Constraint is additive: NOT VALID avoids a long table scan on large user sets;
-- VALIDATE runs inline since the DEFAULT above guarantees compliance.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'users_risk_tier_check'
    ) THEN
        ALTER TABLE users
            ADD CONSTRAINT users_risk_tier_check
            CHECK (risk_tier IN ('conservative', 'balanced', 'aggressive'));
    END IF;
END$$;
