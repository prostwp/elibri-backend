-- Migration 005: scenarios (active flag + telemetry) and Telegram integration.
-- Backs Patch 3 — live scenario runner + Telegram alerts.

-- Users gain a linked Telegram chat_id (1:1, optional).
ALTER TABLE users
  ADD COLUMN IF NOT EXISTS telegram_chat_id BIGINT UNIQUE,
  ADD COLUMN IF NOT EXISTS telegram_linked_at TIMESTAMPTZ;

-- Short-lived codes issued by /api/v1/telegram/link.
-- User sends `/link <code>` to the bot within 10 min to complete the binding.
CREATE TABLE IF NOT EXISTS telegram_link_codes (
  code VARCHAR(8) PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tg_codes_user ON telegram_link_codes(user_id);
CREATE INDEX IF NOT EXISTS idx_tg_codes_expires ON telegram_link_codes(expires_at);

-- Strategies gain runtime fields for the live runner.
ALTER TABLE strategies
  ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS interval VARCHAR(8) NOT NULL DEFAULT '1h',
  ADD COLUMN IF NOT EXISTS risk_tier VARCHAR(20) NOT NULL DEFAULT 'balanced',
  ADD COLUMN IF NOT EXISTS last_signal_bar_time BIGINT,
  ADD COLUMN IF NOT EXISTS last_signal_direction VARCHAR(10),
  ADD COLUMN IF NOT EXISTS telegram_enabled BOOLEAN NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS auto_execute BOOLEAN NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS paused_until TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_strategies_active ON strategies(is_active) WHERE is_active = true;

-- Fired alerts: both the audit log and the delivery queue state.
CREATE TABLE IF NOT EXISTS alerts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  strategy_id UUID NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
  symbol VARCHAR(20) NOT NULL,
  interval VARCHAR(8) NOT NULL,
  direction VARCHAR(10) NOT NULL,
  label VARCHAR(30),
  confidence DECIMAL(5,2),
  entry_price DECIMAL(20,8),
  stop_loss DECIMAL(20,8),
  take_profit DECIMAL(20,8),
  position_size_usd DECIMAL(20,2),
  bar_time BIGINT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  telegram_sent_at TIMESTAMPTZ,
  telegram_message_id BIGINT,
  meta JSONB
);
CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts(user_id, created_at DESC);
-- Dedup: one alert per (scenario, bar, direction) — same bar never fires twice
-- even if the runner restarts mid-tick.
CREATE UNIQUE INDEX IF NOT EXISTS idx_alerts_dedup ON alerts(strategy_id, bar_time, direction);
