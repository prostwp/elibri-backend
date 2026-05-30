package whale

// source.go — the data-source interface seam for Whale Flow.
//
// This is the ONLY boundary the worker/endpoint depend on. The free-tier
// implementations (EtherscanSource for ETH+stablecoins, MempoolSource for
// BTC) satisfy WhaleFlowSource today; a paid provider (Glassnode/Nansen) can
// be swapped in at boot-time in cmd/server/main.go without touching the
// worker or handler — exactly the pattern narrative.ImportanceClassifier uses.

import (
	"context"
	"errors"
	"time"
)

// WhaleFlowSource is the read surface the worker pulls from. Implementations:
//   - EtherscanSource (ETH + ERC-20 stablecoins, near-live poll)
//   - MempoolSource (BTC, live feed)
//   - (future) Glassnode/Nansen paid providers
//
// RecentWhaleTransfers powers the live feed (#1) + flow map (#5). Netflow
// powers the netflow gauge (#2) + direction verdict (#3) — though the worker
// computes its own netflow from the persisted transfers via the store, so a
// free source can return a coarse estimate (or Partial=true) from Netflow.
type WhaleFlowSource interface {
	// Name is the honest source badge ("etherscan"|"mempool"|"sample").
	Name() string
	// Chains lists the chains this source can serve.
	Chains() []Chain
	// RecentWhaleTransfers returns recent transfers on `chain` whose USD value
	// is >= minUSD, newest-first, capped at `limit`. On a partial failure it
	// returns the rows it did parse plus the error — callers log and keep the
	// partial set (never panic).
	RecentWhaleTransfers(ctx context.Context, chain Chain, minUSD float64, limit int) ([]WhaleTransfer, error)
	// Netflow sums inflow/outflow USD over `window` for one (chain, exchange,
	// asset). exchange="" means "all known exchanges". For free sources this
	// is an estimate over labeled wallets only (ExchangeNetflow.Partial=true).
	Netflow(ctx context.Context, chain Chain, exchange, asset string, window time.Duration) (ExchangeNetflow, error)
}

// LiveStreamer is the optional streaming surface — only Mempool's BTC
// websocket implements it (ETH is poll-only). Consumers type-assert:
//
//	if ls, ok := src.(LiveStreamer); ok { ls.StreamTransfers(ctx, ...) }
//
// Kept optional so the worker's MVP loop (poll-based) doesn't depend on it.
type LiveStreamer interface {
	WhaleFlowSource
	// StreamTransfers pushes transfers to `onTransfer` until ctx is cancelled.
	StreamTransfers(ctx context.Context, chain Chain, minUSD float64, onTransfer func(WhaleTransfer)) error
}

// ExchangeRegistry maps on-chain addresses to exchange labels. The Etherscan
// source consults it to attribute a transfer (to_addr ∈ registry → inflow;
// from_addr ∈ registry → outflow). Implemented by staticRegistry (registry.go).
type ExchangeRegistry interface {
	// HotWallets returns lowercase-address → exchange-name for `chain`. The
	// returned map MUST NOT be mutated by callers (it may be shared).
	HotWallets(chain Chain) map[string]string
}

// SmartMoneyProvider is the paid-tier stub for feature #4 ("what funds are
// doing"). Labeled fund-wallet flows are a Nansen product — impossible on a
// $0 budget — so no free implementation exists; the method returns
// ErrPaidTierRequired and the UI renders a locked card. The interface is
// defined now so wiring a paid provider later is a boot-time swap, not a
// refactor.
type SmartMoneyProvider interface {
	WhaleFlowSource
	// SmartMoneyFlows returns labeled fund-wallet flows for `asset` over
	// `window`. Free implementations return ErrPaidTierRequired.
	SmartMoneyFlows(ctx context.Context, asset string, window time.Duration) ([]WhaleTransfer, error)
}

// ErrPaidTierRequired is returned by SmartMoneyProvider implementations that
// have no free data source (feature #4). The handler/UI treats it as "render
// the locked card", not as a hard error.
var ErrPaidTierRequired = errors.New("whale: smart-money flows require a paid data tier")

// Compile-time assertions: the free-tier sources satisfy WhaleFlowSource.
// Breaking either contract fails the build here rather than at the call site.
var (
	_ WhaleFlowSource  = (*EtherscanSource)(nil)
	_ WhaleFlowSource  = (*MempoolSource)(nil)
	_ ExchangeRegistry = (*staticRegistry)(nil)
)
