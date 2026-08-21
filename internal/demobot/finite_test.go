package demobot

// finite_test.go — review fix 4 (demobot side): non-finite floats must die at
// the candle parsers, the single choke point every card/level float derives
// from. The live vector is Binance, whose kline payload carries floats AS
// STRINGS — strconv.ParseFloat accepts "NaN"/"Inf" happily. (The Yahoo path
// cannot deliver non-finite values at all: encoding/json rejects bare NaN and
// errors on overflow, failing the whole fetch — an honest degrade; the parser
// still carries a defensive guard.)

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestFetchBinanceKlinesRejectsNonFiniteBars(t *testing.T) {
	start := time.Now().Unix() - 10*14400
	rows := [][]any{
		{float64(start) * 1000, "100", "110", "90", "105", "10"},
		{float64(start+14400) * 1000, "105", "NaN", "95", "108", "10"},   // poisoned high → bar dropped
		{float64(start+28800) * 1000, "108", "118", "Inf", "112", "10"},  // poisoned low → bar dropped
		{float64(start+43200) * 1000, "112", "120", "100", "115", "NaN"}, // poisoned volume → 0, bar kept
		{float64(start+57600) * 1000, "115", "125", "105", "118", "10"},
	}
	body, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	orig := binanceKlinesBase
	binanceKlinesBase = srv.URL
	t.Cleanup(func() { binanceKlinesBase = orig })

	candles, err := fetchBinanceKlines(t.Context(), "BTCUSDT", "4h", 10)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(candles) != 3 {
		t.Fatalf("candles = %d, want 3 (two poisoned OHLC bars skipped, NaN volume kept as 0)", len(candles))
	}
	for _, c := range candles {
		for _, v := range []float64{c.Open, c.High, c.Low, c.Close, c.Volume} {
			if !isFinite(v) {
				t.Errorf("non-finite value leaked into a candle: %+v", c)
			}
		}
	}
	var kept bool
	for _, c := range candles {
		if c.Close == 115 && c.Volume == 0 {
			kept = true
		}
	}
	if !kept {
		t.Errorf("the NaN-volume bar must survive with volume 0: %+v", candles)
	}
}
