CREATE TABLE IF NOT EXISTS candles (
    symbol      VARCHAR(20) NOT NULL,
    interval    VARCHAR(5)  NOT NULL,
    open_time   TIMESTAMPTZ NOT NULL,
    open        DECIMAL(20,8) NOT NULL,
    high        DECIMAL(20,8) NOT NULL,
    low         DECIMAL(20,8) NOT NULL,
    close       DECIMAL(20,8) NOT NULL,
    volume      DECIMAL(20,4) NOT NULL,
    source      VARCHAR(10) NOT NULL DEFAULT 'moex',
    PRIMARY KEY (symbol, interval, open_time)
);

CREATE INDEX IF NOT EXISTS idx_candles_time ON candles (symbol, interval, open_time DESC);

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker          VARCHAR(10) PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    sector          VARCHAR(50),
    report_type     VARCHAR(10),
    pe              DECIMAL(8,2),
    pb              DECIMAL(8,2),
    ps              DECIMAL(8,2),
    ev_ebitda       DECIMAL(8,2),
    div_yield       DECIMAL(6,2),
    roe             DECIMAL(6,2),
    roa             DECIMAL(6,2),
    net_margin      DECIMAL(6,2),
    revenue         DECIMAL(12,2),
    net_income      DECIMAL(12,2),
    ebitda          DECIMAL(12,2),
    fcf             DECIMAL(12,2),
    net_debt        DECIMAL(12,2),
    net_debt_ebitda DECIMAL(8,2),
    fair_value      DECIMAL(12,2),
    current_price   DECIMAL(12,2),
    market_cap      DECIMAL(14,2),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
