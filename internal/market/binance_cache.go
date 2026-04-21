package market

import (
	"fmt"
	"sync"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// FetchCryptoCandlesCached — drop-in cached wrapper around FetchCryptoCandles.
//
// Problem it solves (from Patch 3 post-rollout review): every scenario
// goroutine independently called FetchCryptoCandles on every tick, so N
// active scenarios on the same (symbol, interval) multiplied Binance
// requests by N. Binance public REST has a 1200 weight/min IP cap, and
// klines cost 1-10 each depending on limit — easy to blow past 100/min
// with a handful of scenarios.
//
// Strategy: per-key sync.Map with 30s TTL. 30s is short enough that we
// don't miss a 5m bar close (the shortest TF we support) by more than
// one tick, and long enough that 5-10 scenarios with poll_interval=60s
// share a single fetch.
//
// Not guarded against stampede on cold start — acceptable for our load
// (≤50 scenarios). Add singleflight.Group if that changes.
//
// Safe to call from any goroutine.
type candleCacheEntry struct {
	candles   []types.OHLCVCandle
	fetchedAt time.Time
}

var (
	candleCache    sync.Map // key (symbol|interval|limit) → candleCacheEntry
	candleCacheTTL = 30 * time.Second
)

func candleCacheKey(symbol, interval string, limit int) string {
	return fmt.Sprintf("%s|%s|%d", symbol, interval, limit)
}

// FetchCryptoCandlesCached — same signature as FetchCryptoCandles, transparent
// 30s TTL cache. Returns a slice that MUST NOT be mutated by callers; make a
// copy if you need to modify (all current callers read-only).
func FetchCryptoCandlesCached(symbol, interval string, limit int) ([]types.OHLCVCandle, error) {
	key := candleCacheKey(symbol, interval, limit)
	if v, ok := candleCache.Load(key); ok {
		e := v.(candleCacheEntry)
		if time.Since(e.fetchedAt) < candleCacheTTL {
			return e.candles, nil
		}
	}
	candles, err := FetchCryptoCandles(symbol, interval, limit)
	if err != nil {
		return nil, err
	}
	candleCache.Store(key, candleCacheEntry{
		candles:   candles,
		fetchedAt: time.Now(),
	})
	return candles, nil
}

// InvalidateCandleCache — test hook + graceful restart helper. Not called in
// hot path; noop-fast if cache is empty.
func InvalidateCandleCache() {
	candleCache.Range(func(k, _ interface{}) bool {
		candleCache.Delete(k)
		return true
	})
}
