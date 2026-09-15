package demobot

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// klineCache keeps recently fetched candles for 60s so /digest (which needs
// BTC 4h candles for momentum, trend, S/R and vol at once) and Refresh spam
// hit the sources once, not four times. Concurrent misses on one key JOIN a
// single in-flight load (singleflight, mirroring the aiMemo pattern — review
// fix 10): six-asset scans arriving together must not multiply upstream GETs.
type klineCache struct {
	mu    sync.Mutex
	items map[string]*klineEntry
}

type klineEntry struct {
	done    chan struct{} // closed when the load finished
	candles []types.OHLCVCandle
	err     error
	at      time.Time
}

const klineTTL = 60 * time.Second

func newKlineCache() *klineCache {
	return &klineCache{items: map[string]*klineEntry{}}
}

// cached returns fresh cached candles for key, joins an in-flight load for
// the same key, or starts one. Errors are shared with concurrent joiners of
// the same flight but never CACHED — the next request retries the source.
//
// It also returns the FETCH time (when the load started): the closed-bar
// cut must run against it, not against the request clock. A bar still
// forming at fetch time carries intermediate OHLC in the cache; cut by "now"
// it would pass as closed once its close time went by, with numbers that the
// next fetch replaces under the same close — a changed body under an
// unchanged Last-Modified.
func (c *klineCache) cached(key string, load func() ([]types.OHLCVCandle, error)) ([]types.OHLCVCandle, time.Time, error) {
	c.mu.Lock()
	if e, ok := c.items[key]; ok {
		select {
		case <-e.done: // finished — serve if fresh and healthy, else reload below
			if e.err == nil && time.Since(e.at) < klineTTL {
				c.mu.Unlock()
				return e.candles, e.at, nil
			}
		default: // in flight — join it outside the lock
			c.mu.Unlock()
			<-e.done
			return e.candles, e.at, e.err
		}
	}
	e := &klineEntry{done: make(chan struct{})}
	c.items[key] = e
	c.mu.Unlock()

	start := time.Now()
	candles, err := load()
	if err == nil && len(candles) == 0 {
		err = fmt.Errorf("%s: empty candle set", key)
	}
	e.candles, e.err, e.at = candles, err, start
	close(e.done)
	if err != nil {
		// Failed flights are evicted so the NEXT request retries; the joiners
		// of THIS flight already share the error (they were concurrent).
		c.mu.Lock()
		if c.items[key] == e {
			delete(c.items, key)
		}
		c.mu.Unlock()
		return nil, time.Time{}, err
	}
	return candles, start, nil
}

// binanceFetchLimit is the one window every Binance kline request asks for
// (Binance's maximum). Callers wanting fewer bars get the newest `limit` of
// them — exactly the rows a limit=`limit` request returns — so the Trend
// Agent's 1000-bar window and the other agents' 250 share ONE upstream
// request per symbol|interval within the cache TTL.
const binanceFetchLimit = 1000

// fetch returns the newest `limit` raw bars (at most binanceFetchLimit) from
// the cached superset, pulling it from Binance on a miss. The tail keeps
// capacity == length, so a caller appending to it can never write into the
// cached backing array another caller is reading.
func (c *klineCache) fetch(ctx context.Context, symbol, interval string, limit int) ([]types.OHLCVCandle, time.Time, error) {
	key := fmt.Sprintf("binance|%s|%s", symbol, interval)
	all, at, err := c.cached(key, func() ([]types.OHLCVCandle, error) {
		return fetchBinanceKlines(ctx, symbol, interval, binanceFetchLimit)
	})
	if err != nil {
		return nil, time.Time{}, err
	}
	n := len(all)
	if limit <= 0 || limit > n {
		limit = n
	}
	return all[n-limit : n : n], at, nil
}

// klinesContiguous reports whether every bar opens exactly one interval after
// the previous one — no row skipped by the parser, no hole in the source.
func klinesContiguous(candles []types.OHLCVCandle, interval string) bool {
	sec := intervalSeconds[interval]
	if sec == 0 {
		return false
	}
	for i := 1; i < len(candles); i++ {
		if candles[i].Time-candles[i-1].Time != sec {
			return false
		}
	}
	return true
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
		c := types.OHLCVCandle{
			Time:   int64(openMs / 1000), // ms → sec (bar OPEN time)
			Open:   num(row[1]),
			High:   num(row[2]),
			Low:    num(row[3]),
			Close:  num(row[4]),
			Volume: num(row[5]),
		}
		// Binance serves floats AS STRINGS and strconv accepts "NaN"/"Inf" —
		// a poisoned OHLC bar is dropped whole (review fix 4: nothing
		// non-finite may reach indicator math); a non-finite volume degrades
		// to 0, matching the null-volume FX convention.
		if !isFinite(c.Open) || !isFinite(c.High) || !isFinite(c.Low) || !isFinite(c.Close) {
			continue
		}
		if !isFinite(c.Volume) {
			c.Volume = 0
		}
		candles = append(candles, c)
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

// fetchFundingRates pulls lastFundingRate (and markPrice, when served) for
// each symbol concurrently. Partial success is fine; it errors only when every
// symbol failed — the card reports which symbols are missing.
//
// The answer carries no funding interval, so the card never labels the rate
// "/8h": it is the symbol's last funding rate.
func fetchFundingRates(ctx context.Context, symbols []string) (map[string]fundingQuote, error) {
	type res struct {
		sym string
		q   fundingQuote
		err error
	}
	ch := make(chan res, len(symbols))
	for _, s := range symbols {
		go func(sym string) {
			q, err := fetchOneFundingRate(ctx, sym)
			ch <- res{sym: sym, q: q, err: err}
		}(s)
	}
	out := make(map[string]fundingQuote, len(symbols))
	var lastErr error
	for range symbols {
		r := <-ch
		if r.err != nil {
			lastErr = r.err
			continue
		}
		out[r.sym] = r.q
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("funding rates unavailable: %w", lastErr)
	}
	return out, nil
}

// fetchOneFundingRate: a rate that is not a finite number below 100% in
// magnitude is bad data and counts as a missing symbol. A missing or bad
// markPrice only leaves the mark unknown (0).
func fetchOneFundingRate(ctx context.Context, symbol string) (fundingQuote, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, premiumIndexURL+symbol, nil)
	if err != nil {
		return fundingQuote{}, err
	}
	resp, err := fundingHTTP.Do(req)
	if err != nil {
		return fundingQuote{}, err
	}
	defer resp.Body.Close()
	limited := io.LimitReader(resp.Body, 8<<20)
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, limited)
		return fundingQuote{}, fmt.Errorf("premiumIndex %s: HTTP %d", symbol, resp.StatusCode)
	}
	var body struct {
		LastFundingRate string `json:"lastFundingRate"`
		MarkPrice       string `json:"markPrice"`
	}
	if err := json.NewDecoder(limited).Decode(&body); err != nil {
		_, _ = io.Copy(io.Discard, limited)
		return fundingQuote{}, err
	}
	_, _ = io.Copy(io.Discard, limited)
	rate, err := strconv.ParseFloat(body.LastFundingRate, 64)
	if err != nil || math.IsNaN(rate) || math.IsInf(rate, 0) || math.Abs(rate) >= 1 {
		return fundingQuote{}, fmt.Errorf("premiumIndex %s: bad rate %q", symbol, body.LastFundingRate)
	}
	q := fundingQuote{rate: rate}
	if m, err := strconv.ParseFloat(body.MarkPrice, 64); err == nil && m > 0 && !math.IsInf(m, 0) {
		q.mark = m
	}
	return q, nil
}
