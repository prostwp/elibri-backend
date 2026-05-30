-- Migration 013 — Whale Flow (whale movements on/off exchanges).
--
-- Background:
--   Whale Flow tracks large transfers between whales and exchange hot-wallets.
--   Deposits TO an exchange ("inflow") signal downward pressure (bearish ▼);
--   withdrawals FROM an exchange ("outflow") signal coins leaving for cold
--   storage (bullish ▲). Source data is free public on-chain data: Etherscan
--   (ETH + stablecoins, near-live poll) and Mempool.space (BTC, live feed).
--
--   Free-tier netflow is an ESTIMATE over labeled hot-wallets only (exchanges
--   rotate hundreds of deposit addresses we can't all see) — the snapshot
--   carries partial=true + a source badge so the UI can be honest about it.
--
-- Two tables:
--
--   whale_transfers
--     One row per (chain, tx_hash, to_addr). The live feed + flow map read
--     from here; the worker also aggregates these rows into snapshots.
--
--   whale_snapshots
--     One row per (asset, captured_at) — the worker writes one per tick per
--     watched asset. The api reads the latest per asset (DISTINCT ON) to
--     render the HERO direction-verdict + netflow gauge.
--
-- Both tables are wholly owned by Whale Flow — no FKs to existing tables.
-- All IF NOT EXISTS so this is idempotent and re-runs safely on every boot.

CREATE TABLE IF NOT EXISTS whale_transfers (
  id BIGSERIAL PRIMARY KEY,
  chain VARCHAR(8) NOT NULL CHECK (chain IN ('ETH','BTC')),
  asset VARCHAR(32) NOT NULL,
  exchange VARCHAR(32) NOT NULL DEFAULT '',
  tx_hash TEXT NOT NULL,
  from_addr TEXT NOT NULL DEFAULT '',
  to_addr TEXT NOT NULL DEFAULT '',
  amount_native DOUBLE PRECISION NOT NULL DEFAULT 0,
  amount_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
  direction VARCHAR(16) NOT NULL CHECK (direction IN ('inflow','outflow','neutral')),
  transferred_at TIMESTAMPTZ NOT NULL,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT whale_transfers_uq UNIQUE (chain, tx_hash, to_addr)
);
CREATE INDEX IF NOT EXISTS idx_whale_transfers_time ON whale_transfers (transferred_at DESC);
CREATE INDEX IF NOT EXISTS idx_whale_transfers_asset ON whale_transfers (asset, transferred_at DESC);

CREATE TABLE IF NOT EXISTS whale_snapshots (
  id BIGSERIAL PRIMARY KEY,
  asset VARCHAR(32) NOT NULL,
  captured_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  net_flow_usd_24h DOUBLE PRECISION NOT NULL DEFAULT 0,
  net_flow_prev_24h DOUBLE PRECISION NOT NULL DEFAULT 0,
  flow_pct REAL NOT NULL DEFAULT 0 CHECK (flow_pct BETWEEN -999 AND 999),
  direction VARCHAR(16) NOT NULL CHECK (direction IN ('inflow','outflow','neutral')),
  inflow_usd_24h DOUBLE PRECISION NOT NULL DEFAULT 0,
  outflow_usd_24h DOUBLE PRECISION NOT NULL DEFAULT 0,
  tx_count_24h INTEGER NOT NULL DEFAULT 0,
  confidence SMALLINT NOT NULL DEFAULT 0 CHECK (confidence BETWEEN 0 AND 100),
  exchange_breakdown JSONB NOT NULL DEFAULT '{}'::jsonb,
  reasons TEXT[] NOT NULL DEFAULT '{}',
  is_new_spike BOOLEAN NOT NULL DEFAULT false,
  source VARCHAR(16) NOT NULL DEFAULT 'etherscan',
  partial BOOLEAN NOT NULL DEFAULT true,
  CONSTRAINT whale_snapshots_uq UNIQUE (asset, captured_at)
);
CREATE INDEX IF NOT EXISTS idx_whale_snapshots_latest ON whale_snapshots (asset, captured_at DESC);
