package demobot

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
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
		var vol float64
		if i < len(q.Volume) && q.Volume[i] != nil {
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

// fetchYahooCandles pulls 1h/1mo candles for one Yahoo symbol.
//
// Candles are accepted from HTTP 200 ONLY (adversarial review item 4): a
// 429/500 whose body happens to parse must never be treated — and cached —
// as live data. Non-200 bodies are read solely to drain the connection.
func fetchYahooCandles(ctx context.Context, symbol string) ([]types.OHLCVCandle, error) {
	u := yahooChartBase + url.PathEscape(symbol) + "?interval=" + yahooInterval + "&range=" + yahooRange
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
