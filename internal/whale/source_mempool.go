package whale

// source_mempool.go — free-tier BTC whale source via Mempool.space.
//
// BTC's UTXO model + exchange address churn make labeled-wallet attribution
// unreliable, so the MVP scope (discovery §3) is: BTC is a LIVE FEED ONLY —
// large recent transactions, direction=neutral, exchange="". No netflow /
// direction verdict for BTC (Netflow returns Partial=true, empty sums).
//
// REST endpoint:
//   GET https://mempool.space/api/mempool/recent
//     → [{ "txid": "...", "fee": N, "vsize": N, "value": <sats> }, ...]
// value is total output in satoshis → BTC = value/1e8. amount_usd = BTC × spot
// (Binance BTCUSDT, cached ~60s; 0 when unavailable). Filtered by minUSD.
//
// All HTTP/parse errors return the partial set + an error to log — never panic.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

const (
	// mempoolRecentURL returns the most recent mempool transactions (a few
	// hundred). This is the live-feed source for BTC whale activity.
	mempoolRecentURL = "https://mempool.space/api/mempool/recent"

	// satsPerBTC is the sat→BTC divisor.
	satsPerBTC = 1e8
)

// MempoolSource implements WhaleFlowSource for BTC (live feed only).
type MempoolSource struct {
	// HTTP is the client used for all calls. nil → a 10s-timeout default.
	HTTP *http.Client

	// spotPrice / spotAt cache the BTC/USD spot for spotCacheTTL.
	spotPrice float64
	spotAt    time.Time
	spotMu    sync.Mutex
}

// NewMempoolSource builds a source with a default HTTP client when nil.
func NewMempoolSource(httpClient *http.Client) *MempoolSource {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 10 * time.Second}
	}
	return &MempoolSource{HTTP: httpClient}
}

// Name returns the honest source badge.
func (m *MempoolSource) Name() string { return "mempool" }

// Chains returns the chains this source serves (BTC only).
func (m *MempoolSource) Chains() []Chain { return []Chain{ChainBTC} }

func (m *MempoolSource) client() *http.Client {
	if m.HTTP != nil {
		return m.HTTP
	}
	return &http.Client{Timeout: 10 * time.Second}
}

// mempoolRecentTx is one entry from /api/mempool/recent.
type mempoolRecentTx struct {
	TxID  string  `json:"txid"`
	Fee   float64 `json:"fee"`
	VSize float64 `json:"vsize"`
	Value float64 `json:"value"` // total output, satoshis
}

// RecentWhaleTransfers returns recent large BTC transactions, newest-first,
// capped at `limit`, filtered by minUSD. Only ChainBTC is served; any other
// chain returns (nil, nil). direction=neutral, exchange="" — BTC attribution
// is out of MVP scope.
//
// Mempool.space's /recent endpoint doesn't carry a timestamp per tx (these are
// unconfirmed, just-seen txns), so we stamp Timestamp = now(). The from/to
// addresses aren't in this lightweight endpoint either — they're left empty;
// the UI's per-row explorer link uses tx_hash, which is sufficient.
func (m *MempoolSource) RecentWhaleTransfers(ctx context.Context, chain Chain, minUSD float64, limit int) ([]WhaleTransfer, error) {
	if chain != ChainBTC {
		return nil, nil
	}
	if limit <= 0 {
		return []WhaleTransfer{}, nil
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, mempoolRecentURL, nil)
	if err != nil {
		return nil, fmt.Errorf("mempool: build request: %w", err)
	}
	resp, err := m.client().Do(req)
	if err != nil {
		return nil, fmt.Errorf("mempool: http: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("mempool: http status %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20)) // cap 4 MB
	if err != nil {
		return nil, fmt.Errorf("mempool: read body: %w", err)
	}
	var txns []mempoolRecentTx
	if uerr := json.Unmarshal(body, &txns); uerr != nil {
		return nil, fmt.Errorf("mempool: decode: %w", uerr)
	}

	spot := m.btcSpotUSD(ctx) // 0 if unavailable → amount_usd stays 0
	now := time.Now().UTC()

	out := make([]WhaleTransfer, 0, len(txns))
	for _, tx := range txns {
		btc := tx.Value / satsPerBTC
		if btc <= 0 {
			continue
		}
		usd := 0.0
		if spot > 0 {
			usd = btc * spot
		}
		// minUSD filter: with no spot we can't assert the floor, so an
		// amount_usd=0 row is kept only when minUSD<=0.
		if minUSD > 0 && usd < minUSD {
			continue
		}
		out = append(out, WhaleTransfer{
			Chain:        ChainBTC,
			TxHash:       tx.TxID,
			Timestamp:    now,
			Asset:        "BTC",
			AmountNative: btc,
			AmountUSD:    usd,
			Direction:    FlowNeutral, // BTC attribution out of MVP scope
			Exchange:     "",
		})
	}

	sortTransfersDesc(out)
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

// Netflow for BTC returns an empty/neutral ExchangeNetflow with Partial=true —
// BTC netflow is NOT computed in the MVP (discovery §3). Non-BTC chains aren't
// served by this source either; both return the same Partial sentinel.
func (m *MempoolSource) Netflow(ctx context.Context, chain Chain, exchange, asset string, window time.Duration) (ExchangeNetflow, error) {
	return ExchangeNetflow{
		Chain:    chain,
		Asset:    asset,
		Exchange: exchange,
		Window:   window,
		Source:   m.Name(),
		Partial:  true,
	}, nil
}

// btcSpotUSD returns the cached BTC/USD spot, refreshing from Binance when
// stale. Returns 0 (not an error) when Binance is unavailable.
func (m *MempoolSource) btcSpotUSD(ctx context.Context) float64 {
	m.spotMu.Lock()
	if time.Since(m.spotAt) < spotCacheTTL && m.spotPrice > 0 {
		p := m.spotPrice
		m.spotMu.Unlock()
		return p
	}
	m.spotMu.Unlock()

	price := fetchBinanceSpot(ctx, m.client(), "BTCUSDT")
	if price > 0 {
		m.spotMu.Lock()
		m.spotPrice = price
		m.spotAt = time.Now()
		m.spotMu.Unlock()
	}
	return price
}
