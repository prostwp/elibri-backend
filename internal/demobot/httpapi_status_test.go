package demobot

// httpapi_status_test.go — the machine-readable status contract (review-debt
// closure): every envelope carries ok/reason so templates branch on WHY a
// reading is absent, and level-bearing agents (trend/sr/vol) add a
// raw-precision "levels" object. Covers:
//   - the reason enum mapping and the defensive Offline collapse,
//   - one byte-exact golden with ok/reason + trend levels,
//   - ok/reason per degraded path: macro unknown (open window; the closed
//     window rides in macro_honesty_test.go), narrative below threshold,
//     offline backend (503 body + digest/top envelopes), insufficient history,
//   - levels presence (trend/sr/vol) and absence (momentum/risk/degraded),
//   - S/R raw precision: cluster means, never display-rounded labels.

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// ── reason enum: pinned mapping ──────────────────────────────────────────────

func TestCardStatusReasonEnum(t *testing.T) {
	cases := map[cardStatus]string{
		statusOK:                  "",
		statusSourceOffline:       "source_offline",
		statusInsufficientHistory: "insufficient_history",
		statusBelowThreshold:      "below_threshold",
		statusNoData:              "no_data",
		statusMarketClosed:        "market_closed",
	}
	for st, want := range cases {
		if got := st.reason(); got != want {
			t.Errorf("reason(%d): got %q, want %q", st, got, want)
		}
	}
	// Defensive collapse: an Offline card without an explicit status can never
	// serve ok=true — it reads as source_offline.
	if got := (Card{Offline: true}).effectiveStatus(); got != statusSourceOffline {
		t.Errorf("bare Offline card: got status %d, want source_offline", got)
	}
	// A specific status wins over the collapse (insufficient cards set both).
	c := Card{Offline: true, Status: statusInsufficientHistory}
	if got := c.effectiveStatus(); got != statusInsufficientHistory {
		t.Errorf("insufficient+Offline: got status %d, want insufficient_history", got)
	}
	if (Card{}).effectiveStatus() != statusOK {
		t.Error("zero-value card must read as ok")
	}
}

// ── golden: ok/reason + trend levels, byte-exact ─────────────────────────────

func TestHTTPEnvelopeTrendLevelsGolden(t *testing.T) {
	c := Card{
		Emoji:    "🟢",
		Agent:    "Trend Agent",
		Asset:    "BTC",
		Verdict:  "Confirmed UPTREND",
		Facts:    []string{"ADX(14): 31.0 (trend confirms above 25) · RSI(14): 62.0"},
		DataTime: goldenTime,
		Levels:   TrendLevels{Invalidation: 63297.5, InvalidationSide: "below"},
	}
	got, err := encodeJSON(cardEnvelope(c))
	if err != nil {
		t.Fatal(err)
	}
	want := `{"agent":"Trend Agent","asset":"BTC","ok":true,"reason":null,"verdict":"Confirmed UPTREND","semaphore":"bullish","facts":["ADX(14): 31.0 (trend confirms above 25) · RSI(14): 62.0"],"levels":{"invalidation":63297.5,"invalidation_side":"below"},"confidence":null,"ai_text":null,"data_as_of":"2026-08-18T06:00:00Z","disclaimer":"Analytics, not financial advice","card_html":"🟢 <b>Trend Agent</b> · BTC\n<b>Confirmed UPTREND</b>\n• ADX(14): 31.0 (trend confirms above 25) · RSI(14): 62.0\n\n<i>Analytics, not financial advice · AlphaVizor · 2026-08-18 06:00 UTC</i>"}`
	if g := strings.TrimRight(string(got), "\n"); g != want {
		t.Fatalf("levels golden mismatch:\ngot:  %s\nwant: %s", g, want)
	}
}

// ── S/R levels: raw precision, [] over null, no display rounding ─────────────

func TestSRPointsRawPrecision(t *testing.T) {
	pts := srPoints([]SRLevel{{Level: 63775, Raw: 63775.4167, Touches: 9}})
	if len(pts) != 1 || pts[0].Level != 63775.4167 || pts[0].Touches != 9 {
		t.Fatalf("srPoints must carry Raw at full precision: %+v", pts)
	}
	if srPoints(nil) == nil {
		t.Fatal("empty side must be a non-nil empty slice, so JSON serves []")
	}

	// FX-scale envelope: the raw cluster mean survives; neither the integer
	// Level (1) nor the %.4f display label may replace it.
	c := Card{
		Emoji: "⚪", Agent: "S/R Agent", Asset: "EURUSD",
		Verdict: "Key levels around 1.1601", DataTime: goldenTime,
		Levels: SRLevels{
			Supports:    srPoints([]SRLevel{{Level: 1, Raw: 1.15834, Touches: 5}}),
			Resistances: srPoints(nil),
		},
	}
	got, err := encodeJSON(cardEnvelope(c))
	if err != nil {
		t.Fatal(err)
	}
	body := string(got)
	if !strings.Contains(body, `"levels":{"supports":[{"level":1.15834,"touches":5,"strength":0,"weakening":false,"breaks":0,"holds":0,"last_touch":""}],"resistances":[]}`) {
		t.Errorf("sr levels must serve the raw mean and [] for the empty side, got: %s", body)
	}
	if strings.Contains(body, `"level":1,`) || strings.Contains(body, `"level":1.1583,`) {
		t.Errorf("display-rounded level leaked into the envelope: %s", body)
	}
}

// ── end-to-end levels presence: trend / vol have them, momentum / risk not ───

func TestHTTPLevelsPresenceEndToEnd(t *testing.T) {
	stubBinanceKlines(t, 250, func(i int) float64 { return 100 })
	deadYahoo(t)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)

	get := func(path string) testEnvelope {
		t.Helper()
		status, _, body := httpGet(t, srv.URL+path)
		if status != 200 {
			t.Fatalf("%s: status %d (%s)", path, status, body)
		}
		var env testEnvelope
		if err := json.Unmarshal(body, &env); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		return env
	}

	// trend: {"invalidation": raw float inside the stub's price region}.
	env := get("/agents/trend")
	if !env.OK || env.Reason != nil {
		t.Errorf("trend: ok/reason %v/%v, want true/null", env.OK, env.Reason)
	}
	if env.Levels == nil {
		t.Fatal("trend envelope must carry levels")
	}
	var tl struct {
		Invalidation *float64 `json:"invalidation"`
	}
	if err := json.Unmarshal(env.Levels, &tl); err != nil {
		t.Fatal(err)
	}
	if tl.Invalidation == nil || *tl.Invalidation <= 60000 || *tl.Invalidation >= 72500 {
		t.Errorf("trend invalidation out of the series' price region: %v", tl.Invalidation)
	}

	// vol: {"expansion_ratio": > 0} — the raw ratio, present even at ~1.0.
	env = get("/agents/vol")
	if env.Levels == nil {
		t.Fatal("vol envelope must carry levels")
	}
	var vl struct {
		Ratio *float64 `json:"expansion_ratio"`
	}
	if err := json.Unmarshal(env.Levels, &vl); err != nil {
		t.Fatal(err)
	}
	if vl.Ratio == nil || *vl.Ratio <= 0 {
		t.Errorf("vol expansion_ratio must be a positive raw float: %v", vl.Ratio)
	}

	// sr on this monotone stub: zero swing points in the window → the honest
	// degrade is a 503 insufficient_history (review fix 2), never a "real
	// reading" whose levels are empty arrays.
	srStatus, _, srBody := httpGet(t, srv.URL+"/agents/sr")
	if srStatus != 503 || !strings.Contains(string(srBody), "insufficient_history") {
		t.Errorf("sr on a swing-free series: status %d body %s, want 503 insufficient_history", srStatus, srBody)
	}

	// Agents without levels: the key is absent, never null/{}.
	for _, path := range []string{"/agents/momentum", "/agents/risk?balance=10000&risk=1&entry=64000&stop=62500"} {
		if env := get(path); env.Levels != nil {
			t.Errorf("%s: levels must be absent, got %s", path, env.Levels)
		}
	}
}

// stubBinanceKlinesWave serves `bars` closed 4h candles for any symbol on a
// 10-bar triangle wave, so swing highs/lows recur at the same two prices and
// the S/R clusterer finds real levels (the shared stubBinanceKlines trends
// monotonically — no swings by construction). The fractional base pins raw
// precision end to end.
func stubBinanceKlinesWave(t *testing.T, bars int) {
	t.Helper()
	wave := []float64{0, 200, 400, 600, 800, 1000, 800, 600, 400, 200}
	start := time.Now().Unix() - int64(bars+2)*14400
	rows := make([][]any, bars)
	for i := range rows {
		price := 60000.4 + wave[i%len(wave)]
		rows[i] = []any{
			float64(start+int64(i)*14400) * 1000,
			fmt.Sprintf("%f", price), fmt.Sprintf("%f", price+50),
			fmt.Sprintf("%f", price-50), fmt.Sprintf("%f", price),
			"100.0",
		}
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
}

func TestHTTPSRLevelsRawPrecisionEndToEnd(t *testing.T) {
	stubBinanceKlinesWave(t, 250)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)

	status, _, body := httpGet(t, srv.URL+"/agents/sr")
	if status != 200 {
		t.Fatalf("status %d (%s)", status, body)
	}
	var env testEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if !env.OK || env.Reason != nil {
		t.Fatalf("sr ok/reason: %v/%v", env.OK, env.Reason)
	}
	var sl struct {
		Supports    []SRPoint `json:"supports"`
		Resistances []SRPoint `json:"resistances"`
	}
	if err := json.Unmarshal(env.Levels, &sl); err != nil {
		t.Fatalf("levels: %v (%s)", err, env.Levels)
	}
	if len(sl.Supports) == 0 || len(sl.Resistances) == 0 {
		t.Fatalf("wave series must cluster both sides: %s", env.Levels)
	}
	// The wave's swing lows sit at 59950.4, highs at 61050.4 — the levels must
	// carry the fractional cluster mean, not the display-rounded integer the
	// facts strings show (srLevelLabel renders %d at this scale).
	if s := sl.Supports[0]; math.Abs(s.Level-59950.4) > 1e-6 || s.Touches < srStrongTouches {
		t.Errorf("support: %+v, want raw ~59950.4 with strong touches", s)
	}
	if r := sl.Resistances[0]; math.Abs(r.Level-61050.4) > 1e-6 || r.Touches < srStrongTouches {
		t.Errorf("resistance: %+v, want raw ~61050.4 with strong touches", r)
	}
	if frac := sl.Supports[0].Level - math.Trunc(sl.Supports[0].Level); frac == 0 {
		t.Error("support level lost its fractional part — display rounding leaked in")
	}
}

// ── degraded paths: ok=false with the specific reason ────────────────────────

// Macro unknown inside the open tradfin window: the feed is dark while the
// market trades → "no_data", never "market_closed". (The closed-window twin
// asserts market_closed in macro_honesty_test.go's envelope test.)
func TestHTTPStatusMacroUnknownOpenWindow(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": macroUnknownFixture(true)})
	_, srv := newTestAPI(t, ag, true)

	status, _, body := httpGet(t, srv.URL+"/agents/macro")
	if status != 200 {
		t.Fatalf("status %d (%s)", status, body)
	}
	var env testEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env.OK {
		t.Error("unknown regime is not a real reading — ok must be false")
	}
	if env.Reason == nil || *env.Reason != "no_data" {
		t.Errorf("reason: %v, want no_data (window open, feed dark)", env.Reason)
	}
	if env.Levels != nil {
		t.Errorf("macro has no levels: %s", env.Levels)
	}
}

// Narrative radar below the mention threshold: an honest 200 card, ok=false,
// reason below_threshold — templates keep the theme list but skip the score UI.
func TestHTTPStatusNewsBelowThreshold(t *testing.T) {
	thin := `{"captured_at":"2026-08-18T06:00:00Z","narratives":[
	  {"narrative":"zk","trend_score":72,"stage":"early","sentiment_label":"bull","mention_count_24h":3,"confidence":61}]}`
	ag := newStubBackend(t, map[string]string{"/api/v1/narratives": thin})
	_, srv := newTestAPI(t, ag, true)

	status, _, body := httpGet(t, srv.URL+"/agents/news")
	if status != 200 {
		t.Fatalf("status %d (%s)", status, body)
	}
	var env testEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env.OK {
		t.Error("warming-up radar must serve ok=false")
	}
	if env.Reason == nil || *env.Reason != "below_threshold" {
		t.Errorf("reason: %v, want below_threshold", env.Reason)
	}
	if !strings.Contains(env.Verdict, "warming up") {
		t.Errorf("verdict must still word the state: %q", env.Verdict)
	}
	// At the threshold the scored card returns and the status clears.
	at := `{"captured_at":"2026-08-18T06:00:00Z","narratives":[
	  {"narrative":"zk","trend_score":72,"stage":"early","sentiment_label":"bull","mention_count_24h":5,"confidence":61}]}`
	ag2 := newStubBackend(t, map[string]string{"/api/v1/narratives": at})
	_, srv2 := newTestAPI(t, ag2, true)
	status, _, body = httpGet(t, srv2.URL+"/agents/news")
	if status != 200 {
		t.Fatalf("at-threshold status %d (%s)", status, body)
	}
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if !env.OK || env.Reason != nil {
		t.Errorf("at-threshold card must be ok/null again: %v/%v", env.OK, env.Reason)
	}
}

// Offline backend: single agents keep their honest 503, and the error body
// now carries the same machine-readable pair; digest/top aggregate to a 200
// whose head (the macro fallback card) reports ok=false / source_offline.
func TestHTTPStatusOfflineBackend(t *testing.T) {
	_, srv := newTestAPI(t, deadAgents(t), true)

	status, _, body := httpGet(t, srv.URL+"/agents/whale")
	if status != 503 {
		t.Fatalf("status %d, want 503 (%s)", status, body)
	}
	var e struct {
		Error  string `json:"error"`
		OK     *bool  `json:"ok"`
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(body, &e); err != nil {
		t.Fatal(err)
	}
	if e.Error == "" || e.OK == nil || *e.OK || e.Reason != "source_offline" {
		t.Errorf("503 body must carry error + ok:false + reason:source_offline, got %s", body)
	}

	for _, path := range []string{"/agents/digest", "/agents/top"} {
		status, _, body := httpGet(t, srv.URL+path)
		if status != 200 {
			t.Fatalf("%s: status %d (%s)", path, status, body)
		}
		var env testEnvelope
		if err := json.Unmarshal(body, &env); err != nil {
			t.Fatal(err)
		}
		if env.OK {
			t.Errorf("%s: all-dead sweep must not claim a real reading", path)
		}
		if env.Reason == nil || *env.Reason != "source_offline" {
			t.Errorf("%s: reason %v, want source_offline", path, env.Reason)
		}
	}
}

// Insufficient history: the source answers but the closed-bar series is too
// short for the indicator set — a 503 whose reason is insufficient_history,
// distinct from source_offline without parsing the error text. The same
// 30-bar MONOTONE series also degrades /agents/sr: enough bars, but zero
// swing structure to read (review blocker: never "Key levels around …" over
// empty arrays).
func TestHTTPStatusInsufficientHistory(t *testing.T) {
	stubBinanceKlines(t, 30, func(i int) float64 { return 100 })
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)

	for _, path := range []string{"/agents/trend", "/agents/vol", "/agents/momentum?asset=btc"} {
		status, _, body := httpGet(t, srv.URL+path)
		if status != 503 {
			t.Errorf("%s: status %d, want 503 (%s)", path, status, body)
			continue
		}
		var e struct {
			Error  string `json:"error"`
			OK     *bool  `json:"ok"`
			Reason string `json:"reason"`
		}
		if err := json.Unmarshal(body, &e); err != nil {
			t.Fatal(err)
		}
		if e.Reason != "insufficient_history" {
			t.Errorf("%s: reason %q, want insufficient_history", path, e.Reason)
		}
		if e.OK == nil || *e.OK {
			t.Errorf("%s: ok must be false, got %s", path, body)
		}
		if !strings.Contains(e.Error, "Insufficient history") {
			t.Errorf("%s: error wording: %q", path, e.Error)
		}
	}

	status, _, body := httpGet(t, srv.URL+"/agents/sr")
	if status != 503 {
		t.Fatalf("sr on a 30-bar monotone: status %d, want 503 (%s)", status, body)
	}
	var e2 struct {
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(body, &e2); err != nil {
		t.Fatal(err)
	}
	if e2.Reason != "insufficient_history" {
		t.Errorf("sr monotone reason %q, want insufficient_history (zero swing points)", e2.Reason)
	}
}

// The funding card's partial degrade (rates dead, liquidation feed alive) is
// a 200 card whose headline reading is missing — ok=false / source_offline
// while the liq facts still render.
func TestHTTPStatusFundingRatesOffline(t *testing.T) {
	stubExternalBases(t) // rates + klines dead
	now := time.Now().UTC().Format(time.RFC3339)
	fixture := `{"captured_at":"` + now + `","feed":[
	    {"symbol":"BTCUSDT","side":"long_liq","qty":1,"price":118000,"usd_value":90000,"ts":"` + now + `"}],
	  "zones":[]}`
	ag := newStubBackend(t, map[string]string{"/api/v1/funding/liquidations": fixture})
	_, srv := newTestAPI(t, ag, true)

	status, _, body := httpGet(t, srv.URL+"/agents/funding")
	if status != 200 {
		t.Fatalf("status %d (%s)", status, body)
	}
	var env testEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env.OK {
		t.Error("rates-offline card must not claim a real funding reading")
	}
	if env.Reason == nil || *env.Reason != "source_offline" {
		t.Errorf("reason: %v, want source_offline", env.Reason)
	}
	if !strings.Contains(strings.Join(env.Facts, "|"), "Liquidations 1h:") {
		t.Errorf("liquidation facts must survive the partial degrade: %v", env.Facts)
	}
}

// The whale agent with an alive feed but no BTC snapshot: ok=false / no_data.
func TestHTTPStatusWhaleNoSnapshot(t *testing.T) {
	fixture := `{"captured_at":"2026-08-18T06:00:00Z","flows":[],"transfers":[]}`
	ag := newStubBackend(t, map[string]string{"/api/v1/whale-flow": fixture})
	_, srv := newTestAPI(t, ag, true)

	status, _, body := httpGet(t, srv.URL+"/agents/whale")
	if status != 200 {
		t.Fatalf("status %d (%s)", status, body)
	}
	var env testEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env.OK || env.Reason == nil || *env.Reason != "no_data" {
		t.Errorf("no-snapshot whale card: ok=%v reason=%v, want false/no_data", env.OK, env.Reason)
	}
}
