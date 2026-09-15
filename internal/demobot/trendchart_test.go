package demobot

// trendchart_test.go — GET /agents/trend/chart: shape, closed bars, EMA
// alignment, confirmed-only zone/invalidation, one-definition parity with
// TrendCard, pivot labelling honesty, live stream spec, error paths, and the
// route not disturbing /agents/trend. All upstreams are stubbed.

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

func getChart(t *testing.T, url string) (int, http.Header, trendChart, []byte) {
	t.Helper()
	code, hdr, body := httpGet(t, url)
	var c trendChart
	if code == http.StatusOK {
		if err := json.Unmarshal(body, &c); err != nil {
			t.Fatalf("decode %s: %v\n%s", url, err, body)
		}
	}
	return code, hdr, c, body
}

// zigzagCandles is the zigzagUp series as candles (same shape as
// stubBinanceSeries) for the pure-builder tests.
func zigzagCandles(bars int) []types.OHLCVCandle {
	start := time.Now().Unix() - int64(bars+2)*14400
	out := make([]types.OHLCVCandle, bars)
	for i := range out {
		p := zigzagUp(i)
		out[i] = types.OHLCVCandle{Time: start + int64(i)*14400, Open: p - 25, High: p + 100, Low: p - 100, Close: p, Volume: 100}
	}
	return out
}

// ── shape, ordering, EMA alignment (confirmed uptrend) ───────────────────────

func TestTrendChartShapeConfirmedUp(t *testing.T) {
	stubBinanceSeries(t, 250, zigzagUp, flatVol)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)

	code, hdr, c, body := getChart(t, srv.URL+"/agents/trend/chart")
	if code != http.StatusOK {
		t.Fatalf("status %d: %s", code, body)
	}
	// Every contract key present, nullable ones explicitly.
	for _, k := range []string{`"agent":`, `"asset":`, `"asset_key":`, `"timeframe":`, `"ok":true`, `"reason":null`,
		`"state":`, `"verdict":`, `"price":`, `"candles":`, `"ema20":`, `"ema50":`, `"ema200":`,
		`"pullback_zone":`, `"invalidation":`, `"pivots":`, `"structure":`, `"data_as_of":`,
		`"source":`, `"live":`, `"disclaimer":"Analytics, not financial advice"`} {
		if !strings.Contains(string(body), k) {
			t.Errorf("body missing %s", k)
		}
	}
	if c.Agent != "Trend Agent" || c.Asset != "BTC" || c.AssetKey != "btc" || c.Timeframe != "4h" || c.Source != "binance" {
		t.Errorf("header fields = %q %q %q %q %q", c.Agent, c.Asset, c.AssetKey, c.Timeframe, c.Source)
	}
	if c.State != trendUp || c.Structure != "hh_hl" {
		t.Fatalf("fixture must be a confirmed HH/HL uptrend, got state=%q structure=%q", c.State, c.Structure)
	}

	// Candles: last 200 of the window, oldest first, strictly 4h apart.
	if len(c.Candles) != trendChartBars {
		t.Fatalf("candles = %d, want %d", len(c.Candles), trendChartBars)
	}
	for i := 1; i < len(c.Candles); i++ {
		if c.Candles[i].Time-c.Candles[i-1].Time != 14400 {
			t.Fatalf("candle %d not 4h after previous", i)
		}
	}
	last := c.Candles[len(c.Candles)-1]
	if last.Close != c.Price {
		t.Errorf("price %v != last close %v", c.Price, last.Close)
	}
	if want := time.Unix(last.Time+14400, 0).UTC().Format(time.RFC3339); c.DataAsOf != want {
		t.Errorf("data_as_of = %s, want last open + 4h = %s", c.DataAsOf, want)
	}
	if lm := hdr.Get("Last-Modified"); lm == "" {
		t.Error("Last-Modified missing")
	} else if pt, _ := http.ParseTime(lm); pt.UTC().Format(time.RFC3339) != c.DataAsOf {
		t.Errorf("Last-Modified %s != data_as_of %s", lm, c.DataAsOf)
	}

	// EMA alignment: 250-bar window, returned from index 50. EMA20/EMA50 are
	// defined on every returned bar; EMA200 only from index 199 (51 points).
	candleAt := map[int64]bool{}
	for _, k := range c.Candles {
		candleAt[k.Time] = true
	}
	for name, series := range map[string][]chartPoint{"ema20": c.EMA20, "ema50": c.EMA50, "ema200": c.EMA200} {
		for i, p := range series {
			if !candleAt[p.Time] {
				t.Errorf("%s[%d] time %d is not a returned candle", name, i, p.Time)
			}
			if i > 0 && p.Time <= series[i-1].Time {
				t.Errorf("%s not strictly ascending at %d", name, i)
			}
		}
		if len(series) == 0 || series[len(series)-1].Time != last.Time {
			t.Errorf("%s must end on the last candle", name)
		}
	}
	if len(c.EMA20) != 200 || len(c.EMA50) != 200 || len(c.EMA200) != 51 {
		t.Errorf("EMA lengths = %d/%d/%d, want 200/200/51", len(c.EMA20), len(c.EMA50), len(c.EMA200))
	}
	if c.EMA200[0].Time != c.Candles[149].Time { // window index 199 = returned index 149
		t.Errorf("EMA200 must start at window bar 199")
	}

	// Last EMA points ARE the agent's read.
	r := trendReadOf(zigzagCandlesFromChartWindow(t, ag))
	if c.EMA20[len(c.EMA20)-1].Value != r.EMA20 || c.EMA50[len(c.EMA50)-1].Value != r.EMA50 ||
		c.EMA200[len(c.EMA200)-1].Value != r.EMA200 {
		t.Error("last EMA points differ from trendReadOf")
	}

	// Confirmed → zone and invalidation present, zone ordered.
	if c.PullbackZone == nil || c.PullbackZone.Low > c.PullbackZone.High {
		t.Errorf("zone = %+v, want present with low <= high", c.PullbackZone)
	}
	if c.Invalidation == nil || c.Invalidation.Side != "below" {
		t.Errorf("invalidation = %+v, want present, side below (uptrend)", c.Invalidation)
	}

	// Live spec for the Binance stream.
	if c.Live == nil || c.Live.Provider != "binance" || c.Live.Symbol != "BTCUSDT" || c.Live.Interval != "4h" {
		t.Errorf("live = %+v", c.Live)
	}

	// Pivots: inside the window, chronological, exactly the 4 compared ones
	// labelled (HH/HL in a clean uptrend).
	labelled := 0
	for i, p := range c.Pivots {
		if !candleAt[p.Time] {
			t.Errorf("pivot %d outside the returned window", i)
		}
		if i > 0 && p.Time < c.Pivots[i-1].Time {
			t.Errorf("pivots not chronological at %d", i)
		}
		if p.Label != "" {
			labelled++
			if (p.Kind == "high" && p.Label != "HH") || (p.Kind == "low" && p.Label != "HL") {
				t.Errorf("pivot %d kind %s labelled %s in an HH/HL read", i, p.Kind, p.Label)
			}
		}
	}
	if labelled != 4 {
		t.Errorf("labelled pivots = %d, want 4 (2 highs + 2 lows compared by hhhlStructure)", labelled)
	}

	// If-Modified-Since at data_as_of → 304.
	req, _ := http.NewRequest(http.MethodGet, srv.URL+"/agents/trend/chart", nil)
	req.Header.Set("If-Modified-Since", hdr.Get("Last-Modified"))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotModified {
		t.Errorf("If-Modified-Since = data time → %d, want 304", resp.StatusCode)
	}
}

// zigzagCandlesFromChartWindow re-fetches the agent's own window (through the
// shared cache) so parity checks run over exactly what the endpoint read.
func zigzagCandlesFromChartWindow(t *testing.T, ag *Agents) []types.OHLCVCandle {
	t.Helper()
	candles, err := ag.candlesFor(context.Background(), btcSpec)
	if err != nil {
		t.Fatal(err)
	}
	return candles
}

// ── one definition: identical to TrendCard ───────────────────────────────────

func TestTrendChartMatchesTrendCard(t *testing.T) {
	fixtures := map[string]func(t *testing.T){
		"confirmed up": func(t *testing.T) { stubBinanceSeries(t, 250, zigzagUp, flatVol) },
		"unconfirmed":  func(t *testing.T) { stubBinanceCandles(t, srCardCycleCandles(250, flatVol)) },
		"monotone":     func(t *testing.T) { stubBinanceKlines(t, 250, flatVol) },
		"downtrend": func(t *testing.T) {
			stubBinanceSeries(t, 250, func(i int) float64 { return 200000 - zigzagUp(i) }, flatVol)
		},
	}
	for name, stub := range fixtures {
		t.Run(name, func(t *testing.T) {
			stub(t)
			ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
			_, srv := newTestAPI(t, ag, true)
			code, _, c, body := getChart(t, srv.URL+"/agents/trend/chart?asset=btc")
			if code != http.StatusOK {
				t.Fatalf("status %d: %s", code, body)
			}
			card := ag.TrendCard(context.Background(), btcSpec)
			if c.State != card.State || c.Verdict != card.Verdict {
				t.Errorf("chart %q/%q != card %q/%q", c.State, c.Verdict, card.State, card.Verdict)
			}
			lv, _ := card.Levels.(TrendLevels)
			confirmed := c.State == trendUp || c.State == trendDown
			if !confirmed {
				if c.PullbackZone != nil || c.Invalidation != nil || card.Levels != nil {
					t.Errorf("unconfirmed %q must have null zone/invalidation: %+v %+v", c.State, c.PullbackZone, c.Invalidation)
				}
				return
			}
			if c.Invalidation == nil || lv.Invalidation == nil ||
				c.Invalidation.Level != *lv.Invalidation || c.Invalidation.Side != lv.InvalidationSide {
				t.Errorf("invalidation %+v != card %+v", c.Invalidation, lv)
			}
			if lv.PullbackZone == nil || c.PullbackZone == nil ||
				c.PullbackZone.Low != math.Min(lv.PullbackZone.From, lv.PullbackZone.To) ||
				c.PullbackZone.High != math.Max(lv.PullbackZone.From, lv.PullbackZone.To) {
				t.Errorf("zone %+v != card %+v", c.PullbackZone, lv.PullbackZone)
			}
		})
	}
}

// The unconfirmed fixture really is unconfirmed, and the downtrend one really
// confirms down — otherwise the parity test above proves less than it says.
func TestTrendChartFixtureStates(t *testing.T) {
	c, ok := trendChartOf(btcSpec, "btc", srCardCycleCandles(250, flatVol))
	if !ok || c.State == trendUp || c.State == trendDown {
		t.Fatalf("cycle fixture: ok=%v state=%q, want an unconfirmed state", ok, c.State)
	}
	if c.PullbackZone != nil || c.Invalidation != nil {
		t.Errorf("unconfirmed: zone/invalidation must be nil")
	}
	down := zigzagCandles(250)
	for i := range down {
		p := 200000 - zigzagUp(i)
		down[i].Open, down[i].High, down[i].Low, down[i].Close = p+25, p+100, p-100, p
	}
	d, ok := trendChartOf(btcSpec, "btc", down)
	if !ok || d.State != trendDown {
		t.Fatalf("mirrored fixture: state=%q, want down", d.State)
	}
	if d.Invalidation == nil || d.Invalidation.Side != "above" {
		t.Errorf("downtrend invalidation = %+v, want side above", d.Invalidation)
	}
	if d.PullbackZone == nil || d.PullbackZone.Low > d.PullbackZone.High {
		t.Errorf("downtrend zone = %+v", d.PullbackZone)
	}
}

// ── pivot labelling honesty ──────────────────────────────────────────────────

// An unreadable structure labels nothing, even with pivots on the chart.
func TestTrendChartPivotsUnlabelledWhenUnreadable(t *testing.T) {
	candles := zigzagCandles(250)
	highs, lows := highsLowsOf(candles)
	ps := trendChartPivots(candles, highs, lows, "", 50)
	if len(ps) == 0 {
		t.Fatal("fixture must have pivots in the window")
	}
	for i, p := range ps {
		if p.Label != "" {
			t.Errorf("pivot %d labelled %q with structure unreadable", i, p.Label)
		}
	}
}

// Monotone series: no swing points at all → structure "" and pivots [] (not
// null), the case the card words "too few swing points".
func TestTrendChartMonotoneNoPivots(t *testing.T) {
	stubBinanceKlines(t, 250, flatVol)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)
	code, _, c, body := getChart(t, srv.URL+"/agents/trend/chart")
	if code != http.StatusOK {
		t.Fatalf("status %d: %s", code, body)
	}
	if c.Structure != "" {
		t.Errorf("structure = %q, want unreadable", c.Structure)
	}
	if !strings.Contains(string(body), `"pivots":[]`) {
		t.Errorf("pivots must serialize as [], body: %.300s", body)
	}
}

// equalHighsRisingLowsCandles is a period-12 zigzag whose swing highs are ALL
// exactly 110.5 (phase 3) while swing lows rise 0.1 per cycle (phase 9, from
// 89.5). Prices never depend on the clock, so the structure read is always
// "mixed": highs tie (neither rising nor falling), lows rise. The last six
// alternating pivots of a 250-bar window are L213 H219 L225 H231 L237 H243.
func equalHighsRisingLowsCandles(bars int) []types.OHLCVCandle {
	offsets := []float64{100, 103, 106, 110, 106, 103, 100, 97, 94, 90, 94, 97}
	start := time.Now().Unix() - int64(bars+2)*14400
	out := make([]types.OHLCVCandle, bars)
	for i := range out {
		p := offsets[i%12]
		if i%12 == 9 {
			p += float64(i/12) * 0.1 // rising troughs, still below the 94 neighbours
		}
		out[i] = types.OHLCVCandle{Time: start + int64(i)*14400, Open: p, High: p + 0.5, Low: p - 0.5, Close: p, Volume: 100}
	}
	return out
}

// Equal extremes read as "mixed"; the tie comparisons carry "" — never an
// invented HH or LH — while the real comparisons in the same read (the rising
// lows) are labelled. Any other structure is a broken fixture, not a skip.
func TestTrendChartTieLabelsEmpty(t *testing.T) {
	candles := equalHighsRisingLowsCandles(250)
	if s := trendReadOf(candles).Structure; s != "mixed" {
		t.Fatalf("fixture structure = %q, want mixed (equal highs, rising lows)", s)
	}
	c, ok := trendChartOf(btcSpec, "btc", candles)
	if !ok || c.Structure != "mixed" {
		t.Fatalf("chart ok=%v structure=%q, want mixed", ok, c.Structure)
	}
	var highs, hl int
	for _, p := range c.Pivots {
		switch {
		case p.Kind == "high":
			highs++
			if p.Price != 110.5 {
				t.Fatalf("fixture high %v, want every high exactly 110.5", p.Price)
			}
			if p.Label != "" {
				t.Errorf("equal-high pivot at %d labelled %q, want \"\"", p.Time, p.Label)
			}
		case p.Label == "HL":
			hl++
		case p.Label != "":
			t.Errorf("rising low labelled %q, want HL or \"\"", p.Label)
		}
	}
	if highs < 3 {
		t.Fatalf("only %d highs in the window, the tie is not exercised", highs)
	}
	if hl != 2 {
		t.Errorf("HL labels = %d, want exactly 2 (2nd and 3rd low of the tail)", hl)
	}
}

func TestPivotLabel(t *testing.T) {
	cases := []struct {
		high        bool
		price, prev float64
		want        string
	}{
		{true, 11, 10, "HH"}, {true, 9, 10, "LH"}, {true, 10, 10, ""},
		{false, 11, 10, "HL"}, {false, 9, 10, "LL"}, {false, 10, 10, ""},
	}
	for _, tc := range cases {
		if got := pivotLabel(tc.high, tc.price, tc.prev); got != tc.want {
			t.Errorf("pivotLabel(%v,%v,%v) = %q, want %q", tc.high, tc.price, tc.prev, got, tc.want)
		}
	}
}

// ── closed bars only ─────────────────────────────────────────────────────────

func TestTrendChartClosedBarsOnly(t *testing.T) {
	now := time.Now().Unix()
	// 250 bars whose LAST bar opened 1h ago → still forming on 4h.
	start := now - 3600 - 249*14400
	candles := make([]types.OHLCVCandle, 250)
	for i := range candles {
		p := zigzagUp(i)
		candles[i] = types.OHLCVCandle{Time: start + int64(i)*14400, Open: p - 25, High: p + 100, Low: p - 100, Close: p, Volume: 100}
	}
	stubBinanceCandles(t, candles)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)
	code, _, c, body := getChart(t, srv.URL+"/agents/trend/chart")
	if code != http.StatusOK {
		t.Fatalf("status %d: %s", code, body)
	}
	last := c.Candles[len(c.Candles)-1]
	if last.Time+14400 > time.Now().Unix() {
		t.Errorf("forming bar served: last open %d", last.Time)
	}
	if last.Time != candles[248].Time {
		t.Errorf("last served bar = %d, want the last CLOSED one %d", last.Time, candles[248].Time)
	}
	if c.Price != candles[248].Close {
		t.Errorf("price %v must be the last closed close %v", c.Price, candles[248].Close)
	}
}

// ── Yahoo: no live stream ────────────────────────────────────────────────────

func TestTrendChartYahooLiveNull(t *testing.T) {
	stubYahoo1h(t, 300)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)
	code, _, c, body := getChart(t, srv.URL+"/agents/trend/chart?asset=eurusd")
	if code != http.StatusOK {
		t.Fatalf("status %d: %s", code, body)
	}
	if c.Live != nil || !strings.Contains(string(body), `"live":null`) {
		t.Errorf("Yahoo spec must serve live:null, got %+v", c.Live)
	}
	if c.Source != "yahoo" || c.Timeframe != "1h" || c.AssetKey != "eurusd" || c.Asset != "EURUSD" {
		t.Errorf("yahoo header = %q %q %q %q", c.Source, c.Timeframe, c.AssetKey, c.Asset)
	}
	// Gold resolves through its alias and keeps the honest contract label.
	g, ok := trendChartOf(xauSpec, "xauusd", zigzagCandles(250))
	if !ok || g.Live != nil || g.Asset != "GOLD · COMEX GC=F" || g.AssetKey != "xauusd" {
		t.Errorf("gold chart = ok %v live %+v asset %q key %q", ok, g.Live, g.Asset, g.AssetKey)
	}
}

// ── error paths ──────────────────────────────────────────────────────────────

func TestTrendChartErrors(t *testing.T) {
	t.Run("400 unknown asset names allowed values", func(t *testing.T) {
		_, srv := newTestAPI(t, deadAgents(t), true)
		code, _, _, body := getChart(t, srv.URL+"/agents/trend/chart?asset=doge")
		if code != http.StatusBadRequest {
			t.Fatalf("status %d", code)
		}
		for _, k := range []string{"btc", "eth", "eurusd", "gbpusd", "usdjpy", "xauusd"} {
			if !strings.Contains(string(body), k) {
				t.Errorf("400 body must name %s: %s", k, body)
			}
		}
	})
	t.Run("400 duplicate and foreign params", func(t *testing.T) {
		_, srv := newTestAPI(t, deadAgents(t), true)
		for _, q := range []string{"?asset=btc&asset=eth", "?tf=1d", "?assets=btc"} {
			if code, _, _, body := getChart(t, srv.URL+"/agents/trend/chart"+q); code != http.StatusBadRequest {
				t.Errorf("%s → %d: %s", q, code, body)
			}
		}
	})
	t.Run("503 source_offline", func(t *testing.T) {
		_, srv := newTestAPI(t, deadAgents(t), true)
		for _, asset := range []string{"btc", "eurusd"} {
			code, hdr, _, body := getChart(t, srv.URL+"/agents/trend/chart?asset="+asset)
			if code != http.StatusServiceUnavailable {
				t.Fatalf("%s: status %d", asset, code)
			}
			assertDegraded(t, body, "source_offline")
			if hdr.Get("Last-Modified") != "" {
				t.Errorf("a 503 must not carry Last-Modified")
			}
		}
	})
	t.Run("503 insufficient_history", func(t *testing.T) {
		stubBinanceKlines(t, trendMinBars-1, flatVol)
		ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
		_, srv := newTestAPI(t, ag, true)
		code, _, _, body := getChart(t, srv.URL+"/agents/trend/chart")
		if code != http.StatusServiceUnavailable {
			t.Fatalf("status %d: %s", code, body)
		}
		assertDegraded(t, body, "insufficient_history")
	})
}

func assertDegraded(t *testing.T, body []byte, reason string) {
	t.Helper()
	var e struct {
		Error  string `json:"error"`
		OK     *bool  `json:"ok"`
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(body, &e); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if e.Error == "" || e.OK == nil || *e.OK || e.Reason != reason {
		t.Errorf("degraded body = %s, want error + ok:false + reason %s", body, reason)
	}
}

// ── routing: /agents/trend unaffected ────────────────────────────────────────

func TestTrendChartRouteLeavesTrendAlone(t *testing.T) {
	stubBinanceSeries(t, 250, zigzagUp, flatVol)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)

	code, _, body := httpGet(t, srv.URL+"/agents/trend")
	if code != http.StatusOK {
		t.Fatalf("/agents/trend status %d", code)
	}
	var env testEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env.Agent != "Trend Agent" || env.CardHTML == "" || strings.Contains(string(body), `"candles"`) {
		t.Errorf("/agents/trend must still serve the card envelope: %.200s", body)
	}
	for _, p := range []string{"/agents/trend/chart/", "/agents/trend/other", "/agents/sr/chart"} {
		if code, _, _ := httpGet(t, srv.URL+p); code != http.StatusNotFound {
			t.Errorf("%s → %d, want 404", p, code)
		}
	}
}
