CREATE TABLE IF NOT EXISTS ml_models (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol          VARCHAR(20) NOT NULL,
    interval        VARCHAR(5)  NOT NULL DEFAULT 'D1',
    model_type      VARCHAR(20) NOT NULL DEFAULT 'gbdt',
    version         INTEGER NOT NULL DEFAULT 1,
    weights         JSONB NOT NULL,
    metrics         JSONB,
    feature_names   TEXT[] NOT NULL DEFAULT '{"rsi","macd","vol_ratio","atr","momentum","bb_pos"}',
    trained_at      TIMESTAMPTZ DEFAULT NOW(),
    is_active       BOOLEAN DEFAULT false,
    UNIQUE(symbol, interval, version)
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id        UUID REFERENCES ml_models(id),
    symbol          VARCHAR(20) NOT NULL,
    predicted_at    TIMESTAMPTZ DEFAULT NOW(),
    direction       VARCHAR(10) NOT NULL,
    confidence      DECIMAL(6,2) NOT NULL,
    price_target    DECIMAL(20,8),
    features        JSONB,
    outcome         VARCHAR(10),
    outcome_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON ml_predictions (symbol, predicted_at DESC);

CREATE TABLE IF NOT EXISTS signal_history (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL,
    strategy_id     UUID,
    symbol          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    direction       VARCHAR(10) NOT NULL,
    confidence      INTEGER NOT NULL,
    entry_price     DECIMAL(20,8),
    stop_loss       DECIMAL(20,8),
    take_profit     DECIMAL(20,8),
    mode            VARCHAR(10),
    graph_result    JSONB,
    analysis        JSONB
);

CREATE TABLE IF NOT EXISTS scan_results (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scanned_at      TIMESTAMPTZ DEFAULT NOW(),
    symbol          VARCHAR(20) NOT NULL,
    signal          VARCHAR(10) NOT NULL,
    score           INTEGER NOT NULL,
    reason          TEXT,
    volume_24h      DECIMAL(20,4),
    price_change    DECIMAL(10,4),
    scan_type       VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS user_preferences (
    user_id             UUID PRIMARY KEY,
    watchlist           TEXT[] DEFAULT '{}',
    default_pair        VARCHAR(20) DEFAULT 'SBER',
    alert_settings      JSONB DEFAULT '{}',
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);
