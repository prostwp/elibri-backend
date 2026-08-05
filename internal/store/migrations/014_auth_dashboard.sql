-- 014_auth_dashboard.sql — email verify/reset tokens, follows, consent columns.
-- Idempotent (IF NOT EXISTS everywhere) — re-runs on every boot.

-- 1. Tokens (email_verify + password_reset). Храним sha256(token) hex(64), не raw.
CREATE TABLE IF NOT EXISTS auth_tokens (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  token_hash  CHAR(64) NOT NULL,                 -- sha256 hex of the raw token
  purpose     VARCHAR(20) NOT NULL CHECK (purpose IN ('email_verify','password_reset')),
  expires_at  TIMESTAMPTZ NOT NULL,
  used_at     TIMESTAMPTZ,                        -- NULL = ещё не использован
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- Поиск при verify/reset — по hash. (token_hash не глобально-уникален: одинаковый
-- hash под разные purpose теоретически возможен, поэтому индекс, не UNIQUE.)
CREATE INDEX IF NOT EXISTS idx_auth_tokens_hash    ON auth_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_auth_tokens_user    ON auth_tokens(user_id, purpose);
CREATE INDEX IF NOT EXISTS idx_auth_tokens_expires ON auth_tokens(expires_at);

-- 2. Follows (юзер подписан на публичный сценарий = strategies.id).
CREATE TABLE IF NOT EXISTS user_follows (
  user_id     UUID NOT NULL REFERENCES users(id)      ON DELETE CASCADE,
  strategy_id UUID NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT user_follows_uq UNIQUE (user_id, strategy_id)
);
CREATE INDEX IF NOT EXISTS idx_user_follows_user     ON user_follows(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_follows_strategy ON user_follows(strategy_id);

-- 3. Consent + verify-флаг на users (идемпотентно, в стиле 005/011).
ALTER TABLE users
  ADD COLUMN IF NOT EXISTS email_verified    BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS terms_accepted_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS terms_version     VARCHAR(20);
