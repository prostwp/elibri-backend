package demobot

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

// Fixture is the real /v8/finance/chart shape (captured live 2026-08-18),
// trimmed to 8 hourly bars with one null-padded bar at index 2 — Yahoo pads
// session gaps with nulls inside the OHLC arrays. Volume is null on FX.
func TestParseYahooChartFixture(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "yahoo_chart.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	candles, err := parseYahooChart(data)
	if err != nil {
		t.Fatalf("parseYahooChart: %v", err)
	}
	// 8 timestamps, 1 null bar skipped → 7 candles.
	if len(candles) != 7 {
		t.Fatalf("candles: got %d, want 7 (null bar must be skipped)", len(candles))
	}
	first := candles[0]
	if first.Time != 1786402800 {
		t.Errorf("first.Time: got %d, want 1786402800", first.Time)
	}
	if !almostEqual(first.Open, 1.1540, 1e-9) || !almostEqual(first.High, 1.1550, 1e-9) ||
		!almostEqual(first.Low, 1.1535, 1e-9) || !almostEqual(first.Close, 1.1546, 1e-9) {
		t.Errorf("first OHLC mismatch: %+v", first)
	}
	if first.Volume != 0 {
		t.Errorf("null volume must become 0, got %f", first.Volume)
	}
	// The bar after the skipped index-2 bar carries timestamp index 3.
	if candles[2].Time != 1786413600 {
		t.Errorf("post-gap bar time: got %d, want 1786413600", candles[2].Time)
	}
	last := candles[6]
	if last.Time != 1786428000 || !almostEqual(last.Close, 1.1583, 1e-9) {
		t.Errorf("last bar mismatch: %+v", last)
	}
	// Ascending order preserved.
	for i := 1; i < len(candles); i++ {
		if candles[i].Time <= candles[i-1].Time {
			t.Fatalf("candles not ascending at %d: %d after %d", i, candles[i].Time, candles[i-1].Time)
		}
	}
}

// Real error shape (XAUUSD=X actually returns this live — the GC=F fallback
// path depends on detecting it).
func TestParseYahooChartError(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "yahoo_error.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	if _, err := parseYahooChart(data); err == nil {
		t.Fatal("error payload must produce an error")
	}
}

// Item 4: a non-200 response must be an error even when its body happens to
// parse as a valid chart — otherwise a 429/500 snapshot gets cached as live.
func TestFetchYahooRejectsNon200(t *testing.T) {
	body, err := os.ReadFile(filepath.Join("testdata", "yahoo_chart.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var status int32 = 429
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(int(atomic.LoadInt32(&status)))
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	orig := yahooChartBase
	yahooChartBase = srv.URL + "/v8/finance/chart/"
	defer func() { yahooChartBase = orig }()

	ctx := context.Background()
	if _, err := fetchYahooCandles(ctx, "EURUSD=X"); err == nil {
		t.Fatal("HTTP 429 with a parseable body must be an error, never candles")
	}
	atomic.StoreInt32(&status, 500)
	if _, err := fetchYahooCandles(ctx, "EURUSD=X"); err == nil {
		t.Fatal("HTTP 500 must be an error")
	}
	// Positive control: same body with 200 parses fine.
	atomic.StoreInt32(&status, 200)
	candles, err := fetchYahooCandles(ctx, "EURUSD=X")
	if err != nil || len(candles) != 7 {
		t.Fatalf("200 path broke: %d candles, err=%v", len(candles), err)
	}
}

func TestParseYahooChartMalformed(t *testing.T) {
	cases := map[string]string{
		"not json":     "<html>rate limited</html>",
		"empty result": `{"chart":{"result":[],"error":null}}`,
		"no quote":     `{"chart":{"result":[{"timestamp":[1],"indicators":{"quote":[]}}],"error":null}}`,
		"all null bars": `{"chart":{"result":[{"timestamp":[1,2],"indicators":{"quote":[{` +
			`"open":[null,null],"high":[null,null],"low":[null,null],"close":[null,null],"volume":[null,null]}]}}],"error":null}}`,
	}
	for name, payload := range cases {
		if _, err := parseYahooChart([]byte(payload)); err == nil {
			t.Errorf("%s: expected error, got none", name)
		}
	}
}
