// Package whale implements the Whale Flow scoring layer: type definitions,
// pure functions, on-chain data sources, store, and worker for tracking large
// transfers between whales and exchange hot-wallets.
//
// Deposits TO an exchange ("inflow") read as downward pressure (bearish ▼);
// withdrawals FROM an exchange ("outflow") read as coins leaving for cold
// storage (bullish ▲). This package is the contract between the ingestion
// worker and the REST/UI layer — the on-disk schema in migration
// 013_whale_flow.sql is mirrored here field-for-field.
//
// This file holds types only. The math lives in scorer.go; the source
// interface in source.go; persistence in store.go.
package whale

import "time"

// Chain identifies the blockchain a transfer happened on. Values match the
// CHECK constraint on whale_transfers.chain in migration 013.
type Chain string

const (
	// ChainETH: Ethereum L1 (ETH + ERC-20 stablecoins via Etherscan).
	ChainETH Chain = "ETH"
	// ChainBTC: Bitcoin (Mempool.space live feed).
	ChainBTC Chain = "BTC"
)

// FlowDirection labels which way coins moved relative to an exchange. Values
// match the CHECK constraints on whale_transfers.direction and
// whale_snapshots.direction in migration 013 — keep these in sync with the
// SQL enum lists exactly.
type FlowDirection string

const (
	// FlowInflow: coins moved TO an exchange (deposit). Net inflow reads as a
	// bearish ▼ observation in the UI.
	FlowInflow FlowDirection = "inflow"
	// FlowOutflow: coins moved FROM an exchange (withdrawal). Net outflow
	// reads as a bullish ▲ observation in the UI.
	FlowOutflow FlowDirection = "outflow"
	// FlowNeutral: no clear net direction, or the counterparty is not a
	// labeled exchange wallet.
	FlowNeutral FlowDirection = "neutral"
)

// WhaleTransfer mirrors one row of whale_transfers (migration 013). One row
// per (chain, tx_hash, to_addr). amount_native is already divided by
// 10^decimals; amount_usd is best-effort (0 when the spot price is
// unavailable). JSON tags match the on-wire contract in spec §1.
type WhaleTransfer struct {
	Chain     Chain     `json:"chain"`
	TxHash    string    `json:"tx_hash"`
	Timestamp time.Time `json:"timestamp"`
	FromAddr  string    `json:"from_addr"`
	ToAddr    string    `json:"to_addr"`
	Asset     string    `json:"asset"`
	// AmountNative is already scaled by 10^decimals (e.g. wei/1e18 for ETH,
	// sats/1e8 for BTC, value/1e6 for USDT/USDC).
	AmountNative float64 `json:"amount_native"`
	// AmountUSD is 0 when the spot price was unavailable → UI renders "—".
	AmountUSD float64       `json:"amount_usd"`
	Direction FlowDirection `json:"direction"`
	// Exchange is the labeled counterparty ("Binance", "Coinbase", …) or ""
	// when the counterparty isn't a known hot-wallet.
	Exchange string `json:"exchange"`
}

// Snapshot mirrors one row of whale_snapshots (migration 013). The worker
// writes one Snapshot per (asset, tick); the read path returns the latest one
// per asset. JSON tags match the on-wire contract in spec §1.
type Snapshot struct {
	Asset      string    `json:"asset"`
	CapturedAt time.Time `json:"captured_at"`
	// NetFlowUSD24h = inflow_usd - outflow_usd. >0 = net inflow (bearish ▼);
	// <0 = net outflow (bullish ▲).
	NetFlowUSD24h float64 `json:"net_flow_usd_24h"`
	// NetFlowPrev24h is the same metric over the prior 24h window, used to
	// derive flow_pct.
	NetFlowPrev24h float64 `json:"net_flow_prev_24h"`
	// FlowPct is capped at ±999 by ComputeFlowPct; ±999 is the "new spike"
	// sentinel (prior window was zero).
	FlowPct       float64       `json:"flow_pct"`
	Direction     FlowDirection `json:"direction"`
	InflowUSD24h  float64       `json:"inflow_usd_24h"`
	OutflowUSD24h float64       `json:"outflow_usd_24h"`
	TxCount24h    int           `json:"tx_count_24h"`
	// Confidence in [0, 100] — data-quality score (tx volume, exchange
	// diversity). 0 = "too little data" sentinel → UI renders "—".
	Confidence        int            `json:"confidence"`
	ExchangeBreakdown map[string]int `json:"exchange_breakdown"`
	// Reasons use safe wording only (no banned words / $ amounts) so they
	// pass the frontend's isIdeaSafe guard.
	Reasons []string `json:"reasons"`
	// IsNewSpike is true when the prior 24h window had zero net flow but this
	// window has nonzero flow — the same predicate ComputeFlowPct uses to
	// return the ±999 sentinel. Frontend prefers this boolean over parsing
	// flow_pct >= 999.
	IsNewSpike bool `json:"is_new_spike"`
	// Source is the honest data-origin badge: "etherscan" | "mempool" |
	// "sample".
	Source string `json:"source"`
	// Partial is true when the netflow is an estimate over labeled wallets
	// (free-tier data), not a full exchange total.
	Partial bool `json:"partial"`
}

// WhaleTransferRef is a slim, JSON-facing shape for click-target transfer
// references — distinct from WhaleTransfer so a future endpoint can return a
// minimal payload (no chain-internal fields) without re-shaping the full
// struct. Currently the public list endpoint returns full WhaleTransfer rows;
// WhaleTransferRef exists for parity with narrative.MentionRef and for a
// future trimmed endpoint.
type WhaleTransferRef struct {
	TxHash    string        `json:"tx_hash"`
	Asset     string        `json:"asset"`
	AmountUSD float64       `json:"amount_usd"`
	Direction FlowDirection `json:"direction"`
	Exchange  string        `json:"exchange"`
	Timestamp time.Time     `json:"timestamp"`
}

// ExchangeNetflow is the result of a Netflow() query over one (chain, asset,
// window): the inflow/outflow sums plus the honesty metadata. Source providers
// return this; the worker uses RecentWhaleTransfers + the store instead for
// the actual snapshot math, but the interface keeps Netflow available for a
// future paid provider (Glassnode) that returns true totals directly.
type ExchangeNetflow struct {
	Chain      Chain         `json:"chain"`
	Asset      string        `json:"asset"`
	Exchange   string        `json:"exchange"`
	InflowUSD  float64       `json:"inflow_usd"`
	OutflowUSD float64       `json:"outflow_usd"`
	TxCount    int           `json:"tx_count"`
	Window     time.Duration `json:"window_seconds"`
	// Source is the honest data-origin badge ("etherscan"|"mempool"|"sample").
	Source string `json:"source"`
	// Partial is true when this is an estimate over labeled wallets only.
	Partial bool `json:"partial"`
}
