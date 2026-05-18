-- Migration 009 — Narrative Radar (Илья's scenario #1)
--
-- Background:
--   The bot/site already aggregates news via internal/news/* (Reddit hot,
--   2 RSS feeds, Finnhub, AlphaVantage, LunarCrush). That pipeline is
--   in-memory only with a 5-minute cache — no history.
--
--   Narrative Radar needs *historical* mention counts to compute
--   24h-vs-prior-24h growth, stage classification (early / trending /
--   mainstream / declining), and trend score. So mentions get persisted
--   to Postgres, deduped by URL+narrative, and aggregated into snapshots
--   on each worker tick.
--
-- Two tables:
--
--   narrative_mentions
--     One row per (item URL, narrative). The same news article matches
--     multiple narratives (e.g. an article about EigenLayer matches both
--     "Restaking" and "Ethereum DeFi") — denormalised on purpose so the
--     count queries stay simple.
--
--   narrative_snapshots
--     One row per (narrative, captured_at) — the worker writes one per
--     tick. The bot/api reads MAX(captured_at) per narrative to render
--     the radar page. History is kept for the growth_pct sparkline and
--     for backtesting future tuning of the stage thresholds.
--
-- Both tables are wholly owned by Narrative Radar — no FKs to existing
-- tables (auth/strategies live in their own world). Drop+rebuild in dev
-- is safe.

CREATE TABLE IF NOT EXISTS narrative_mentions (
  id           BIGSERIAL PRIMARY KEY,
  narrative    VARCHAR(64)  NOT NULL,        -- canonical slug, e.g. "restaking"
  source       VARCHAR(32)  NOT NULL,        -- whitelisted in CHECK below
  url          TEXT         NOT NULL,
  title        TEXT         NOT NULL,
  posted_at    TIMESTAMPTZ  NOT NULL,        -- from upstream item, NOT ingestion time
  ingested_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
  sentiment    REAL         NOT NULL,        -- -1.0..+1.0, from news.scoreSentiment
  tickers      TEXT[]       NOT NULL DEFAULT '{}',  -- e.g. ['ETH','EIGEN']
  -- Same item can match multiple narratives → composite uniqueness, not URL alone.
  CONSTRAINT narrative_mentions_uq UNIQUE (narrative, url),
  -- Whitelist sources to catch typos like "Reddit" / "reddit "/"reddIt".
  -- Add new sources here as the ingestion layer grows (lunarcrush, finnhub, …).
  CONSTRAINT narrative_mentions_source_chk CHECK (source IN
    ('reddit','coindesk','cointelegraph','lunarcrush','finnhub','alphavantage'))
);

-- Hot path: "fetch all mentions of narrative N in window [now - 48h, now]".
-- Composite (narrative, posted_at) covers both filter and ORDER BY.
CREATE INDEX IF NOT EXISTS idx_narrative_mentions_time
  ON narrative_mentions (narrative, posted_at DESC);

-- Secondary: top-N global recent mentions across narratives (debug page).
CREATE INDEX IF NOT EXISTS idx_narrative_mentions_recent
  ON narrative_mentions (posted_at DESC);


CREATE TABLE IF NOT EXISTS narrative_snapshots (
  id                     BIGSERIAL PRIMARY KEY,
  narrative              VARCHAR(64) NOT NULL,
  captured_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
  mention_count_24h      INTEGER     NOT NULL,
  mention_count_prev_24h INTEGER     NOT NULL,
  -- growth_pct is capped at ±999 by the scorer; CHECK enforces the contract
  -- so a future scorer bug can't push runaway 8500.0 values into the column.
  growth_pct             REAL        NOT NULL
                          CONSTRAINT narrative_snapshots_growth_chk
                          CHECK (growth_pct BETWEEN -999.0 AND 999.0),
  trend_score            SMALLINT    NOT NULL
                          CONSTRAINT narrative_snapshots_trend_chk
                          CHECK (trend_score BETWEEN 0 AND 100),
  stage                  VARCHAR(16) NOT NULL
                          CONSTRAINT narrative_snapshots_stage_chk
                          CHECK (stage IN ('early','trending','mainstream','declining')),
  sentiment_score        REAL        NOT NULL,    -- -1.0..+1.0 (no CHECK: scorer clamps)
  sentiment_label        VARCHAR(8)  NOT NULL
                          CONSTRAINT narrative_snapshots_sentlabel_chk
                          CHECK (sentiment_label IN ('bull','neutral','bear')),
  confidence             SMALLINT    NOT NULL
                          CONSTRAINT narrative_snapshots_confidence_chk
                          CHECK (confidence BETWEEN 0 AND 100),
  related_assets         TEXT[]      NOT NULL DEFAULT '{}',
  sources_breakdown      JSONB       NOT NULL DEFAULT '{}'::jsonb,
                          -- {"reddit": 12, "coindesk": 5, ...}
  reasons                TEXT[]      NOT NULL DEFAULT '{}',  -- bullets for UI
  -- Worker writes one row per (narrative, captured_at). UNIQUE here makes
  -- a retried tick a no-op via ON CONFLICT instead of doubling the row,
  -- which would corrupt the growth_pct sparkline derived from history.
  CONSTRAINT narrative_snapshots_uq UNIQUE (narrative, captured_at)
);

-- Read path: "latest snapshot per narrative" — uses DISTINCT ON
-- (narrative) ORDER BY narrative, captured_at DESC. The composite
-- index serves it without a sort step.
CREATE INDEX IF NOT EXISTS idx_narrative_snapshots_latest
  ON narrative_snapshots (narrative, captured_at DESC);
