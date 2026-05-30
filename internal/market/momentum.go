package market

import (
	"fmt"
	"strings"
	"sync"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// momentum.go — relative-strength + volume-trend computation for the Crypto
// Sentiment "liquidity" panel. The frontend already gets a coin's 1d change vs
// BTC from the live Binance ticker; this endpoint adds the multi-day picture
// (7d/30d vs BTC) and a volume-trend read that the ticker can't provide.
//
// All daily-close math reuses FetchCryptoCandlesCached (binance_cache.go), so
// repeated calls share one 30s-cached klines fetch per (symbol,interval,limit)
// and we don't add a second cache layer here — the compute itself is trivial.

// momentumDays — how many daily candles we request. The widest window is 30d,
// which needs 31 candles (today + 30 prior closes); volume-growth needs ≥9.
const momentumDays = 31

// Momentum is a coin's relative-strength vs BTC over several windows plus its
// own volume trend. Fields are pointers so an unavailable window (short
// history) is omitted from JSON rather than serialized as a misleading 0.
type Momentum struct {
	RS1d         *float64 `json:"rs_1d,omitempty"`
	RS7d         *float64 `json:"rs_7d,omitempty"`
	RS30d        *float64 `json:"rs_30d,omitempty"`
	VolGrowthPct *float64 `json:"vol_growth_pct,omitempty"`
}

// pctChangeOverDays returns the % change in closing price between the candle
// `days` positions back and the most recent candle. ok is false when there
// isn't enough history or the base close is non-positive.
//
// Anchored on the latest candle (candles[n-1]) — which is the in-progress UTC
// day — on purpose: this is a "current price vs price N days ago" read, and we
// want the freshest price. The relative-strength caller subtracts BTC's change
// computed the same way, so any in-progress-day effect is shared and cancels.
func pctChangeOverDays(candles []types.OHLCVCandle, days int) (float64, bool) {
	n := len(candles)
	if days < 1 || n < days+1 {
		return 0, false
	}
	latest := candles[n-1].Close
	base := candles[n-1-days].Close
	if base <= 0 {
		return 0, false
	}
	return (latest/base - 1) * 100, true
}

// relativeStrength is the coin's % change minus BTC's % change over the same
// window — positive means the coin outperformed BTC. Valid only when both
// sides have enough history.
func relativeStrength(sym, btc []types.OHLCVCandle, days int) (float64, bool) {
	s, okS := pctChangeOverDays(sym, days)
	b, okB := pctChangeOverDays(btc, days)
	if !okS || !okB {
		return 0, false
	}
	return s - b, true
}

// volumeGrowth compares the most recent CLOSED day's volume to the average of
// the 7 closed days before it, as a %. ok is false with <9 candles or a
// non-positive average.
//
// Anchored on candles[n-2], not candles[n-1]: the last candle is the
// in-progress UTC day whose volume is only partial, so comparing it to full
// days would always understate it. n-2 is the last fully-closed day.
func volumeGrowth(candles []types.OHLCVCandle) (float64, bool) {
	n := len(candles)
	if n < 9 {
		return 0, false
	}
	recent := candles[n-2].Volume
	var sum float64
	for i := n - 9; i < n-2; i++ { // 7 closed days before the last closed day
		sum += candles[i].Volume
	}
	avg := sum / 7
	if avg <= 0 {
		return 0, false
	}
	return (recent/avg - 1) * 100, true
}

// computeMomentum is the pure, network-free core: given a symbol's daily
// candles and BTC's, produce its Momentum. Unavailable windows stay nil.
func computeMomentum(sym, btc []types.OHLCVCandle) Momentum {
	var m Momentum
	if v, ok := relativeStrength(sym, btc, 1); ok {
		m.RS1d = &v
	}
	if v, ok := relativeStrength(sym, btc, 7); ok {
		m.RS7d = &v
	}
	if v, ok := relativeStrength(sym, btc, 30); ok {
		m.RS30d = &v
	}
	if v, ok := volumeGrowth(sym); ok {
		m.VolGrowthPct = &v
	}
	return m
}

// FetchMomentum returns the baseline asset ("BTC") and per-symbol Momentum for
// the given bare symbols (e.g. "ETH", "SOL" — no "USDT" suffix). BTC is the
// baseline, so it's skipped if present in the input. A symbol whose klines
// can't be fetched is omitted rather than failing the whole batch; only a BTC
// fetch failure (no baseline possible) returns an error.
//
// Fetches run concurrently: BTC + each symbol is one independently-cached
// klines call, so the fan-out overlaps the round-trips on a cold cache rather
// than serializing N+1 of them on a single request thread. Inputs are
// normalized + de-duplicated first so a repeated symbol costs nothing extra.
func FetchMomentum(symbols []string) (string, map[string]Momentum, error) {
	seen := make(map[string]struct{})
	var want []string
	for _, raw := range symbols {
		s := strings.ToUpper(strings.TrimSpace(raw))
		if s == "" || s == "BTC" {
			continue
		}
		if _, dup := seen[s]; dup {
			continue
		}
		seen[s] = struct{}{}
		want = append(want, s)
	}

	type result struct {
		candles []types.OHLCVCandle
		err     error
	}
	var (
		btc    []types.OHLCVCandle
		btcErr error
	)
	results := make([]result, len(want)) // each goroutine writes its own slot

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		btc, btcErr = FetchCryptoCandlesCached("BTCUSDT", "1d", momentumDays)
	}()
	for i, s := range want {
		wg.Add(1)
		go func(i int, s string) {
			defer wg.Done()
			c, err := FetchCryptoCandlesCached(s+"USDT", "1d", momentumDays)
			results[i] = result{candles: c, err: err}
		}(i, s)
	}
	wg.Wait()

	if btcErr != nil {
		return "", nil, fmt.Errorf("momentum baseline (BTC): %w", btcErr)
	}

	out := make(map[string]Momentum, len(want))
	for i, s := range want {
		if results[i].err != nil {
			continue // skip unfetchable symbols; don't sink the batch
		}
		out[s] = computeMomentum(results[i].candles, btc)
	}
	return "BTC", out, nil
}
