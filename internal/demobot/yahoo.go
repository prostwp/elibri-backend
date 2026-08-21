package demobot

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// Yahoo Finance chart API — the FX/gold candle source. No key needed, but
// Yahoo answers 429 to requests without a User-Agent (verified live), so
// every request sends a browser-like UA.
// yahooChartBase is a var so tests can point it at a scripted server.
var yahooChartBase = "https://query1.finance.yahoo.com/v8/finance/chart/"

const (
	yahooUA       = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
	yahooRange    = "1mo" // ~500 hourly bars on a 24/5 instrument
	yahooInterval = "1h"
)

var yahooHTTP = &http.Client{Timeout: 10 * time.Second}

// yahooChartResp mirrors /v8/finance/chart. OHLCV arrays use pointers:
// Yahoo pads session gaps with nulls inside the arrays, and FX volume is
// null throughout.
type yahooChartResp struct {
	Chart struct {
		Result []struct {
			Timestamp  []int64 `json:"timestamp"`
			Indicators struct {
				Quote []struct {
					Open   []*float64 `json:"open"`
					High   []*float64 `json:"high"`
					Low    []*float64 `json:"low"`
					Close  []*float64 `json:"close"`
					Volume []*float64 `json:"volume"`
				} `json:"quote"`
			} `json:"indicators"`
		} `json:"result"`
		Error *struct {
			Code        string `json:"code"`
			Description string `json:"description"`
		} `json:"error"`
	} `json:"chart"`
}

// parseYahooChart converts a chart payload into candles, skipping
// null-padded bars (any of O/H/L/C null → bar dropped; null volume → 0).
func parseYahooChart(data []byte) ([]types.OHLCVCandle, error) {
	var resp yahooChartResp
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("yahoo chart: decode: %w", err)
	}
	if resp.Chart.Error != nil {
		return nil, fmt.Errorf("yahoo chart: %s — %s", resp.Chart.Error.Code, resp.Chart.Error.Description)
	}
	if len(resp.Chart.Result) == 0 {
		return nil, errors.New("yahoo chart: empty result")
	}
	r := resp.Chart.Result[0]
	if len(r.Indicators.Quote) == 0 {
		return nil, errors.New("yahoo chart: no quote block")
	}
	q := r.Indicators.Quote[0]

	n := len(r.Timestamp)
	for _, arr := range [][]*float64{q.Open, q.High, q.Low, q.Close} {
		if len(arr) < n {
			n = len(arr)
		}
	}
	candles := make([]types.OHLCVCandle, 0, n)
	for i := 0; i < n; i++ {
		if q.Open[i] == nil || q.High[i] == nil || q.Low[i] == nil || q.Close[i] == nil {
			continue // Yahoo null-padding — skip, never invent a bar
		}
		// Defensive finiteness guard (review fix 4). encoding/json cannot
		// actually deliver NaN/Inf (bare NaN is rejected, overflow errors the
		// whole decode), so this is unreachable via the wire today — kept as
		// a cheap belt in case the decode path ever changes.
		if !isFinite(*q.Open[i]) || !isFinite(*q.High[i]) || !isFinite(*q.Low[i]) || !isFinite(*q.Close[i]) {
			continue
		}
		var vol float64
		if i < len(q.Volume) && q.Volume[i] != nil && isFinite(*q.Volume[i]) {
			vol = *q.Volume[i]
		}
		candles = append(candles, types.OHLCVCandle{
			Time:   r.Timestamp[i],
			Open:   *q.Open[i],
			High:   *q.High[i],
			Low:    *q.Low[i],
			Close:  *q.Close[i],
			Volume: vol,
		})
	}
	if len(candles) == 0 {
		return nil, errors.New("yahoo chart: no usable bars (all null-padded)")
	}
	return candles, nil
}

// fetchYahooCandles pulls 1h/1mo candles for one Yahoo symbol (the historical
// default — FX/gold cards run on 1h natively).
func fetchYahooCandles(ctx context.Context, symbol string) ([]types.OHLCVCandle, error) {
	return fetchYahooCandlesTF(ctx, symbol, yahooInterval)
}

// fetchYahooCandlesTF pulls candles at the requested timeframe (B1):
//
//	1h → native (interval=1h&range=1mo, ~500 bars on a 24/5 instrument)
//	1d → native (interval=1d&range=1y, ~250 trading days)
//	4h → Yahoo has NO native 4h interval (verified against the v8 chart API's
//	     accepted set) — fetched as 1h and honestly aggregated: only complete
//	     4-of-4 hourly groups aligned to UTC 4h boundaries become a bar, gaps
//	     and partial groups are dropped, never padded (aggregate1hTo4h).
func fetchYahooCandlesTF(ctx context.Context, symbol, interval string) ([]types.OHLCVCandle, error) {
	fetchInterval, fetchRange := yahooInterval, yahooRange
	aggregate := false
	switch interval {
	case "", "1h":
		// defaults
	case "1d":
		fetchInterval, fetchRange = "1d", "1y"
	case "4h":
		aggregate = true // fetch 1h/1mo, merge below
	default:
		return nil, fmt.Errorf("yahoo chart %s: unsupported interval %q", symbol, interval)
	}
	candles, err := fetchYahooChart(ctx, symbol, fetchInterval, fetchRange)
	if err != nil {
		return nil, err
	}
	if aggregate {
		candles = aggregate1hTo4h(candles)
	}
	return candles, nil
}

// fetchYahooChart is the raw chart GET.
//
// Candles are accepted from HTTP 200 ONLY (adversarial review item 4): a
// 429/500 whose body happens to parse must never be treated — and cached —
// as live data. Non-200 bodies are read solely to drain the connection.
func fetchYahooChart(ctx context.Context, symbol, interval, rng string) ([]types.OHLCVCandle, error) {
	u := yahooChartBase + url.PathEscape(symbol) + "?interval=" + interval + "&range=" + rng
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", yahooUA)
	resp, err := yahooHTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("yahoo chart %s: %w", symbol, err)
	}
	defer resp.Body.Close()
	limited := io.LimitReader(resp.Body, 8<<20)
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, limited)
		return nil, fmt.Errorf("yahoo chart %s: HTTP %d", symbol, resp.StatusCode)
	}
	body, err := io.ReadAll(limited)
	if err != nil {
		return nil, fmt.Errorf("yahoo chart %s: read: %w", symbol, err)
	}
	return parseYahooChart(body)
}

// aggregate1hTo4h merges hourly bars into 4h bars aligned to UTC 4h
// boundaries (Time % 14400 == 0). Honesty rules:
//   - a 4h bar exists ONLY when all four expected hourly bars (g, g+1h,
//     g+2h, g+3h) are present — session gaps and partial edge groups are
//     dropped, never padded or half-filled;
//   - O = first bar's open, H/L = extremes, C = last bar's close, V = sum
//     (null-volume FX sums to 0, matching the 1h behavior);
//   - bars off the hour grid can never complete a group and vanish.
func aggregate1hTo4h(candles []types.OHLCVCandle) []types.OHLCVCandle {
	byGroup := make(map[int64][]types.OHLCVCandle)
	for _, c := range candles {
		if c.Time%3600 != 0 {
			continue // off the hour grid — cannot honestly slot it
		}
		g := c.Time - c.Time%14400
		byGroup[g] = append(byGroup[g], c)
	}
	out := make([]types.OHLCVCandle, 0, len(byGroup))
	for g, group := range byGroup {
		if len(group) != 4 {
			continue
		}
		sort.Slice(group, func(i, j int) bool { return group[i].Time < group[j].Time })
		complete := true
		for i, c := range group {
			if c.Time != g+int64(i)*3600 {
				complete = false
				break
			}
		}
		if !complete {
			continue
		}
		bar := types.OHLCVCandle{
			Time: g,
			Open: group[0].Open, High: group[0].High,
			Low: group[0].Low, Close: group[3].Close,
			Volume: 0,
		}
		for _, c := range group {
			if c.High > bar.High {
				bar.High = c.High
			}
			if c.Low < bar.Low {
				bar.Low = c.Low
			}
			bar.Volume += c.Volume
		}
		out = append(out, bar)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Time < out[j].Time })
	return out
}
