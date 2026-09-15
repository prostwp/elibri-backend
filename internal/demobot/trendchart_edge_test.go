package demobot

// trendchart_edge_test.go — /agents/trend/chart under hostile or irregular
// input: Yahoo session/weekend gaps, several unclosed trailing rows, and
// NaN/Inf/degenerate prices. The contract: original bar-open timestamps, no
// compaction, closed bars only, and always valid JSON with finite numbers —
// an honest 503 or clean data, never a panic.

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// stubYahooRows serves a Yahoo chart JSON with EXACTLY the given rows (FX-style
// null volume) for any symbol.
func stubYahooRows(t *testing.T, ts []int64, o, h, l, c []float64) {
	t.Helper()
	payload := map[string]any{
		"chart": map[string]any{
			"result": []any{map[string]any{
				"timestamp": ts,
				"indicators": map[string]any{
					"quote": []any{map[string]any{
						"open": o, "high": h, "low": l, "close": c,
						"volume": make([]any, len(ts)),
					}},
				},
			}},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	orig := yahooChartBase
	yahooChartBase = srv.URL + "/v8/finance/chart/"
	t.Cleanup(func() { yahooChartBase = orig })
}

// ── Yahoo: weekend + session gaps, trailing live rows ────────────────────────

func TestTrendChartYahooSessionGaps(t *testing.T) {
	now := time.Now().UTC()
	day := now.Truncate(24*time.Hour).AddDate(0, 0, -28)
	for day.Weekday() != time.Monday {
		day = day.AddDate(0, 0, -1)
	}
	// Hourly bars Mon-Fri, a one-hour break at 22:00 UTC every day, weekends
	// absent — every real bar closed at least two minutes ago.
	var ts []int64
	for hr := day; hr.Unix()+3600 <= now.Unix()-120; hr = hr.Add(time.Hour) {
		if wd := hr.Weekday(); wd == time.Saturday || wd == time.Sunday || hr.Hour() == 22 {
			continue
		}
		ts = append(ts, hr.Unix())
	}
	real := len(ts)
	if real < 399 {
		t.Fatalf("fixture has %d bars, need >= 399 so all three EMAs cover the chart", real)
	}
	wiggle := []float64{0, 2, 4, 6, 4, 2, 0, -2, -4, -2}
	var o, h, l, c []float64
	for i := 0; i < real; i++ {
		p := 1.1 + float64(i)*0.00002 + wiggle[i%10]*0.0001
		o, h, l, c = append(o, p), append(h, p+0.0003), append(l, p-0.0003), append(c, p)
	}
	// Two trailing rows Yahoo really sends: the forming hour bar and the
	// regularMarketTime snapshot. Neither is a closed bar.
	for _, at := range []int64{now.Unix() - 1800, now.Unix() - 10} {
		p := c[real-1] + 0.01 // a price the chart must never show
		ts, o, h, l, c = append(ts, at), append(o, p), append(h, p+0.0003), append(l, p-0.0003), append(c, p)
	}
	stubYahooRows(t, ts, o, h, l, c)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)

	code, hdr, ch, body := getChart(t, srv.URL+"/agents/trend/chart?asset=eurusd")
	if code != http.StatusOK {
		t.Fatalf("status %d: %s", code, body)
	}
	if ch.Live != nil || ch.Source != "yahoo" || ch.Timeframe != "1h" {
		t.Errorf("live=%+v source=%q tf=%q", ch.Live, ch.Source, ch.Timeframe)
	}

	// Original stamps, no compaction: the served candles ARE the last 200 real rows.
	want := ts[real-trendChartBars : real]
	if len(ch.Candles) != len(want) {
		t.Fatalf("candles = %d, want %d", len(ch.Candles), len(want))
	}
	var sawBreak, sawWeekend bool
	for i, k := range ch.Candles {
		if k.Time != want[i] {
			t.Fatalf("candle %d time %d, want original %d (no re-stamping)", i, k.Time, want[i])
		}
		if bt := time.Unix(k.Time, 0).UTC(); bt.Minute() != 0 || bt.Second() != 0 {
			t.Errorf("candle %d time %s is not an hour-bar open in UTC", i, bt)
		}
		if i > 0 {
			gap := k.Time - ch.Candles[i-1].Time
			sawBreak = sawBreak || gap == 7200
			sawWeekend = sawWeekend || gap >= 48*3600
		}
	}
	if !sawBreak || !sawWeekend {
		t.Fatalf("window must span a session break and a weekend: break=%v weekend=%v", sawBreak, sawWeekend)
	}

	// EMA points sit on the same (gapped) times — exactly the candle times.
	for name, series := range map[string][]chartPoint{"ema20": ch.EMA20, "ema50": ch.EMA50, "ema200": ch.EMA200} {
		if len(series) != len(ch.Candles) {
			t.Errorf("%s has %d points, want %d", name, len(series), len(ch.Candles))
			continue
		}
		for i, p := range series {
			if p.Time != ch.Candles[i].Time {
				t.Errorf("%s[%d] time %d != candle time %d", name, i, p.Time, ch.Candles[i].Time)
				break
			}
		}
	}

	// data_as_of = close of the last REAL bar; the trailing rows are invisible.
	wantAsOf := time.Unix(ts[real-1]+3600, 0).UTC().Format(time.RFC3339)
	if ch.DataAsOf != wantAsOf {
		t.Errorf("data_as_of = %s, want %s", ch.DataAsOf, wantAsOf)
	}
	if pt, _ := http.ParseTime(hdr.Get("Last-Modified")); pt.UTC().Format(time.RFC3339) != wantAsOf {
		t.Errorf("Last-Modified %q != %s", hdr.Get("Last-Modified"), wantAsOf)
	}
	if ch.Price != c[real-1] {
		t.Errorf("price %v, want last real close %v", ch.Price, c[real-1])
	}
	candleAt := map[int64]bool{}
	for _, k := range ch.Candles {
		candleAt[k.Time] = true
	}
	for i, p := range ch.Pivots {
		if !candleAt[p.Time] {
			t.Errorf("pivot %d time %d is not a served candle", i, p.Time)
		}
	}
}

// ── Binance: several unclosed trailing rows ──────────────────────────────────

func TestTrendChartSeveralUnclosedRows(t *testing.T) {
	now := time.Now().Unix()
	candles := zigzagCandles(250) // all closed; the last opened 3 bars ago
	lastClosed := candles[len(candles)-1]
	for _, open := range []int64{now - 3600, now + 3*3600, now + 7*3600} {
		candles = append(candles, types.OHLCVCandle{
			Time: open, Open: 1, High: 999999, Low: 1, Close: 999999, Volume: 1,
		})
	}
	stubBinanceCandles(t, candles)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)

	code, _, ch, body := getChart(t, srv.URL+"/agents/trend/chart")
	if code != http.StatusOK {
		t.Fatalf("status %d: %s", code, body)
	}
	if len(ch.Candles) != trendChartBars {
		t.Fatalf("candles = %d, want %d", len(ch.Candles), trendChartBars)
	}
	for i, k := range ch.Candles {
		if k.Time+14400 > time.Now().Unix() {
			t.Errorf("candle %d (open %d) has not closed", i, k.Time)
		}
		if k.High == 999999 {
			t.Errorf("candle %d is one of the unclosed rows", i)
		}
	}
	if got := ch.Candles[len(ch.Candles)-1]; got.Time != lastClosed.Time || ch.Price != lastClosed.Close {
		t.Errorf("last served %d/%v, want last closed %d/%v", got.Time, ch.Price, lastClosed.Time, lastClosed.Close)
	}
	if want := time.Unix(lastClosed.Time+14400, 0).UTC().Format(time.RFC3339); ch.DataAsOf != want {
		t.Errorf("data_as_of = %s, want %s", ch.DataAsOf, want)
	}
	for name, series := range map[string][]chartPoint{"ema20": ch.EMA20, "ema50": ch.EMA50, "ema200": ch.EMA200} {
		if n := len(series); n == 0 || series[n-1].Time != lastClosed.Time {
			t.Errorf("%s must end on the last closed bar", name)
		}
	}
}

// ── NaN / Inf / degenerate prices ────────────────────────────────────────────

var nonFiniteToken = regexp.MustCompile(`\b(NaN|[+-]?Inf)\b`)

func TestTrendChartDegenerateInputs(t *testing.T) {
	poisoned := func(bars int, every int) ([]types.OHLCVCandle, map[int64]bool) {
		cs := zigzagCandles(bars)
		bad := map[int64]bool{}
		for i := range cs {
			switch {
			case i%every == 0:
				cs[i].Close = math.NaN()
			case i%every == 1:
				cs[i].High = math.Inf(1)
			case i%every == 2:
				cs[i].Low = math.Inf(-1)
			default:
				continue
			}
			bad[cs[i].Time] = true
		}
		return cs, bad
	}
	flatAt := func(bars int, p float64) []types.OHLCVCandle {
		cs := zigzagCandles(bars)
		for i := range cs {
			cs[i].Open, cs[i].High, cs[i].Low, cs[i].Close = p, p, p, p
		}
		return cs
	}
	overflow := zigzagCandles(250)
	for i := range overflow {
		// Finite inputs whose true ranges sum past MaxFloat64 inside ATR/ADX.
		overflow[i].Open, overflow[i].High, overflow[i].Low, overflow[i].Close =
			1.2e308, 1.7e308, 1e307, 1.2e308+float64(i%3)*1e306
	}
	nanRows, nanBad := poisoned(300, 20)
	tooFew, _ := poisoned(230, 10)

	cases := []struct {
		name    string
		candles []types.OHLCVCandle
		bad     map[int64]bool // row times that must never be served
		want    int            // 0 = 200 or 503 both acceptable
	}{
		{"NaN/Inf rows dropped", nanRows, nanBad, http.StatusOK},
		{"NaN/Inf rows leave too little history", tooFew, nil, http.StatusServiceUnavailable},
		{"constant price", flatAt(250, 100), nil, http.StatusOK},
		{"zero price", flatAt(250, 0), nil, 0},
		{"near-overflow prices", overflow, nil, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stubBinanceCandles(t, tc.candles)
			ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
			_, srv := newTestAPI(t, ag, true)
			code, _, body := httpGet(t, srv.URL+"/agents/trend/chart")
			var peek struct {
				State, Verdict, Reason string
			}
			_ = json.Unmarshal(body, &peek)
			t.Logf("status %d state=%q verdict=%q reason=%q", code, peek.State, peek.Verdict, peek.Reason)

			if tc.want != 0 && code != tc.want {
				t.Fatalf("status %d, want %d: %.300s", code, tc.want, body)
			}
			if code != http.StatusOK && code != http.StatusServiceUnavailable {
				t.Fatalf("status %d, want an honest 200 or 503: %.300s", code, body)
			}
			if !json.Valid(body) {
				t.Fatalf("invalid JSON: %.300s", body)
			}
			if m := nonFiniteToken.Find(body); m != nil {
				t.Fatalf("body carries non-finite token %q: %.300s", m, body)
			}
			if code == http.StatusServiceUnavailable {
				assertDegraded(t, body, "insufficient_history")
				return
			}
			var ch trendChart
			if err := json.Unmarshal(body, &ch); err != nil {
				t.Fatal(err)
			}
			if !ch.OK || len(ch.Candles) == 0 || ch.Price != ch.Candles[len(ch.Candles)-1].Close {
				t.Errorf("clean-data 200 expected: ok=%v n=%d price=%v", ch.OK, len(ch.Candles), ch.Price)
			}
			for _, k := range ch.Candles {
				if tc.bad[k.Time] {
					t.Errorf("poisoned row at %d was served", k.Time)
				}
			}
			if !strings.Contains(string(body), `"pivots":[`) {
				t.Errorf("pivots must be an array: %.200s", body)
			}
		})
	}
}

// ── /agents index lists the chart, and the example works ─────────────────────

func TestTrendChartListedInIndex(t *testing.T) {
	stubYahoo1h(t, 300)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)
	code, _, body := httpGet(t, srv.URL+"/agents")
	if code != http.StatusOK {
		t.Fatalf("/agents status %d", code)
	}
	var list struct {
		Agents []struct {
			Name     string   `json:"name"`
			Examples []string `json:"examples"`
		} `json:"agents"`
	}
	if err := json.Unmarshal(body, &list); err != nil {
		t.Fatal(err)
	}
	const example = "/agents/trend/chart?asset=eurusd"
	found := false
	for _, a := range list.Agents {
		for _, ex := range a.Examples {
			if ex == example {
				found = found || a.Name == keyTrend
				if a.Name != keyTrend {
					t.Errorf("chart example listed under %q, want trend", a.Name)
				}
			}
		}
	}
	if !found {
		t.Fatalf("/agents trend entry must list %s", example)
	}
	if code, _, body := httpGet(t, srv.URL+example); code != http.StatusOK {
		t.Errorf("listed example %s → %d: %.200s", example, code, body)
	}
}
