package demobot

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// klineCache keeps recently fetched candles for 60s so /digest (which needs
// BTC 4h candles for momentum, trend, S/R and vol at once) and Refresh spam
// hit the sources once, not four times.
type klineCache struct {
	mu    sync.Mutex
	items map[string]klineEntry
}

type klineEntry struct {
	candles []types.OHLCVCandle
	at      time.Time
}

const klineTTL = 60 * time.Second

func newKlineCache() *klineCache {
	return &klineCache{items: map[string]klineEntry{}}
}

// cached returns fresh cached candles for key or fills the cache via load.
// Errors are never cached — the next request retries the source.
func (c *klineCache) cached(key string, load func() ([]types.OHLCVCandle, error)) ([]types.OHLCVCandle, error) {
	c.mu.Lock()
	if e, ok := c.items[key]; ok && time.Since(e.at) < klineTTL {
		c.mu.Unlock()
		return e.candles, nil
	}
	c.mu.Unlock()

	candles, err := load()
	if err != nil {
		return nil, err
	}
	if len(candles) == 0 {
		return nil, fmt.Errorf("%s: empty candle set", key)
	}
	c.mu.Lock()
	c.items[key] = klineEntry{candles: candles, at: time.Now()}
	c.mu.Unlock()
	return candles, nil
}

// fetch returns cached candles or pulls them from Binance.
func (c *klineCache) fetch(ctx context.Context, symbol, interval string, limit int) ([]types.OHLCVCandle, error) {
	key := fmt.Sprintf("binance|%s|%s|%d", symbol, interval, limit)
	return c.cached(key, func() ([]types.OHLCVCandle, error) {
		return fetchBinanceKlines(ctx, symbol, interval, limit)
	})
}

// ── Binance spot klines ──────────────────────────────────────────────────────
//
// The bot carries its own fetcher instead of internal/market's: the review
// requires context threading + body limits + non-200 draining, and the shared
// server code cannot be modified from here. Same endpoint, same parsing.

var binanceHTTP = &http.Client{Timeout: 10 * time.Second}

var binanceKlinesBase = "https://api.binance.com/api/v3/klines"

func fetchBinanceKlines(ctx context.Context, symbol, interval string, limit int) ([]types.OHLCVCandle, error) {
	u := fmt.Sprintf("%s?symbol=%s&interval=%s&limit=%d", binanceKlinesBase, symbol, interval, limit)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := binanceHTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("binance klines %s: %w", symbol, err)
	}
	defer resp.Body.Close()
	limited := io.LimitReader(resp.Body, 8<<20)
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, limited)
		return nil, fmt.Errorf("binance klines %s: HTTP %d", symbol, resp.StatusCode)
	}
	var raw [][]any
	if err := json.NewDecoder(limited).Decode(&raw); err != nil {
		_, _ = io.Copy(io.Discard, limited)
		return nil, fmt.Errorf("binance klines %s: decode: %w", symbol, err)
	}
	_, _ = io.Copy(io.Discard, limited)

	candles := make([]types.OHLCVCandle, 0, len(raw))
	for _, row := range raw {
		if len(row) < 6 {
			continue
		}
		openMs, ok := row[0].(float64)
		if !ok {
			continue
		}
		num := func(v any) float64 {
			s, ok := v.(string)
			if !ok {
				return 0
			}
			f, _ := strconv.ParseFloat(s, 64)
			return f
		}
		candles = append(candles, types.OHLCVCandle{
			Time:   int64(openMs / 1000), // ms → sec (bar OPEN time)
			Open:   num(row[1]),
			High:   num(row[2]),
			Low:    num(row[3]),
			Close:  num(row[4]),
			Volume: num(row[5]),
		})
	}
	return candles, nil
}

// ── Closed-bar discipline (adversarial review item 7) ────────────────────────

// dropUnclosedBars keeps only bars whose close time (open + interval) is at
// or before now. Signals computed on a forming bar flicker mid-bar while
// looking final, so all indicator math runs on closed bars only.
//
// Why not "drop exactly one": Binance contributes exactly one forming bar,
// but Yahoo often appends BOTH the forming hour bar and a live snapshot row
// stamped at regularMarketTime — a fixed drop-one left a future-stamped bar
// in the math (observed live: card footer 41 minutes in the future).
func dropUnclosedBars(candles []types.OHLCVCandle, interval string, now time.Time) []types.OHLCVCandle {
	if len(candles) == 0 {
		return candles
	}
	sec := intervalSeconds[interval]
	if sec == 0 { // unknown interval — conservative fallback: drop the last bar
		return candles[:len(candles)-1]
	}
	cut := now.Unix()
	i := len(candles)
	for i > 0 && candles[i-1].Time+sec > cut {
		i--
	}
	return candles[:i]
}

var intervalSeconds = map[string]int64{
	"1h": 3600,
	"4h": 14400,
	"1d": 86400,
}

// closeTimeOf is the close time of the last bar in the series (candle Time
// is the OPEN time on both sources) — the honest "UTC time of data" for a
// card. Empty series → now, so a footer never shows the zero time.
func closeTimeOf(candles []types.OHLCVCandle, interval string) time.Time {
	if len(candles) == 0 {
		return time.Now().UTC()
	}
	sec := intervalSeconds[interval]
	return time.Unix(candles[len(candles)-1].Time+sec, 0).UTC()
}

// ── Perp funding rates (Binance futures public API) ──────────────────────────
//
// The backend has no funding-rate REST endpoint (only the liquidation feed),
// so the "widest skew" read comes straight from the public premiumIndex
// endpoint — same no-key tier as the spot klines the task allows.

var fundingHTTP = &http.Client{Timeout: 8 * time.Second}

// var so tests can point it at a stub server.
var premiumIndexURL = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol="

// fetchFundingRates pulls lastFundingRate for each symbol concurrently.
// Partial success is fine; it errors only when every symbol failed.
func fetchFundingRates(ctx context.Context, symbols []string) (map[string]float64, error) {
	type res struct {
		sym  string
		rate float64
		err  error
	}
	ch := make(chan res, len(symbols))
	for _, s := range symbols {
		go func(sym string) {
			rate, err := fetchOneFundingRate(ctx, sym)
			ch <- res{sym: sym, rate: rate, err: err}
		}(s)
	}
	out := make(map[string]float64, len(symbols))
	var lastErr error
	for range symbols {
		r := <-ch
		if r.err != nil {
			lastErr = r.err
			continue
		}
		out[r.sym] = r.rate
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("funding rates unavailable: %w", lastErr)
	}
	return out, nil
}

func fetchOneFundingRate(ctx context.Context, symbol string) (float64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, premiumIndexURL+symbol, nil)
	if err != nil {
		return 0, err
	}
	resp, err := fundingHTTP.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	limited := io.LimitReader(resp.Body, 8<<20)
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, limited)
		return 0, fmt.Errorf("premiumIndex %s: HTTP %d", symbol, resp.StatusCode)
	}
	var body struct {
		LastFundingRate string `json:"lastFundingRate"`
	}
	if err := json.NewDecoder(limited).Decode(&body); err != nil {
		_, _ = io.Copy(io.Discard, limited)
		return 0, err
	}
	_, _ = io.Copy(io.Discard, limited)
	rate, err := strconv.ParseFloat(body.LastFundingRate, 64)
	if err != nil {
		return 0, fmt.Errorf("premiumIndex %s: bad rate %q", symbol, body.LastFundingRate)
	}
	return rate, nil
}
