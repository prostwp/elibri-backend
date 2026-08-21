package demobot

// momentum_b1_test.go — B1: user-configurable momentum scan. Asset-list
// parsing against the registry, the ?tf= plumbing down to the candle fetch,
// honest Yahoo 1h→4h aggregation, HTTP validation and Telegram parity.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// ── registry: asset lists + timeframes ───────────────────────────────────────

func TestParseAssetList(t *testing.T) {
	keys, err := parseAssetList("btc,eurusd,gold")
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if strings.Join(keys, ",") != "btc,eurusd,xauusd" {
		t.Errorf("keys = %v, want [btc eurusd xauusd] (alias resolved)", keys)
	}

	// Duplicates collapse (aliases too), order of first appearance kept.
	keys, err = parseAssetList("eth, btc ,ETH,bitcoin")
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if strings.Join(keys, ",") != "eth,btc" {
		t.Errorf("dedup keys = %v, want [eth btc]", keys)
	}

	// A bad entry names itself AND the allowed registry.
	if _, err := parseAssetList("btc,doge"); err == nil || !strings.Contains(err.Error(), "doge") ||
		!strings.Contains(err.Error(), "eurusd") {
		t.Errorf("bad entry error = %v, want the entry + the allowed list", err)
	}

	// Empty entries ("btc,,eth") are rejected, not silently skipped.
	if _, err := parseAssetList("btc,,eth"); err == nil {
		t.Error("empty list entry must error")
	}

	// The whole registry is exactly 6 — fits; a 7th (duplicate-free) cannot
	// be constructed, so the cap is tested via repetition after dedup limit:
	if keys, err := parseAssetList("btc,eth,eurusd,gbpusd,usdjpy,xauusd"); err != nil || len(keys) != 6 {
		t.Errorf("full registry scan = %v/%v, want 6 keys", keys, err)
	}
}

func TestSpecWithTF(t *testing.T) {
	spec, err := specWithTF(btcSpec, "1d")
	if err != nil || spec.Interval != "1d" {
		t.Errorf("btc@1d = %+v/%v, want interval 1d", spec, err)
	}
	spec, err = specWithTF(assetTable["eurusd"], "4h")
	if err != nil || spec.Interval != "4h" {
		t.Errorf("eurusd@4h = %+v/%v, want interval 4h (aggregated at fetch)", spec, err)
	}
	if got, err := specWithTF(btcSpec, ""); err != nil || got.Interval != btcSpec.Interval {
		t.Errorf("empty tf must keep the native interval, got %+v/%v", got, err)
	}
	if _, err := specWithTF(btcSpec, "2h"); err == nil || !strings.Contains(err.Error(), "1h, 4h, 1d") {
		t.Errorf("bad tf error = %v, want the allowed list", err)
	}
}

// ── Yahoo 1h → 4h aggregation (pure) ─────────────────────────────────────────

func hourBar(t0 int64, hour int, o, h, l, c, v float64) types.OHLCVCandle {
	return types.OHLCVCandle{Time: t0 + int64(hour)*3600, Open: o, High: h, Low: l, Close: c, Volume: v}
}

func TestAggregate1hTo4h(t *testing.T) {
	t0 := int64(1755648000) // divisible by 14400 → a 4h boundary
	if t0%14400 != 0 {
		t.Fatal("fixture start must sit on a 4h boundary")
	}
	in := []types.OHLCVCandle{
		// Complete group 1: hours 0-3.
		hourBar(t0, 0, 10, 15, 9, 12, 100),
		hourBar(t0, 1, 12, 18, 11, 17, 200),
		hourBar(t0, 2, 17, 17.5, 13, 14, 50),
		hourBar(t0, 3, 14, 16, 12, 13, 150),
		// Group 2 has a gap (hour 5 missing) → dropped, never padded.
		hourBar(t0, 4, 13, 14, 12, 12.5, 10),
		hourBar(t0, 6, 12.5, 13, 12, 12.2, 10),
		hourBar(t0, 7, 12.2, 13, 12, 12.8, 10),
		// Trailing partial group (hours 8-9 only) → dropped.
		hourBar(t0, 8, 12.8, 13, 12, 12.9, 10),
		hourBar(t0, 9, 12.9, 13, 12, 12.7, 10),
	}
	out := aggregate1hTo4h(in)
	if len(out) != 1 {
		t.Fatalf("bars = %d, want 1 (only the complete group)", len(out))
	}
	b := out[0]
	if b.Time != t0 {
		t.Errorf("time = %d, want the group start %d", b.Time, t0)
	}
	if b.Open != 10 || b.Close != 13 {
		t.Errorf("open/close = %v/%v, want first open 10 / last close 13", b.Open, b.Close)
	}
	if b.High != 18 || b.Low != 9 {
		t.Errorf("high/low = %v/%v, want 18/9", b.High, b.Low)
	}
	if b.Volume != 500 {
		t.Errorf("volume = %v, want the 500 sum", b.Volume)
	}

	// Bars not aligned to the hour grid can never complete a group.
	misaligned := []types.OHLCVCandle{
		{Time: t0 + 120, Open: 1, High: 1, Low: 1, Close: 1},
		{Time: t0 + 3720, Open: 1, High: 1, Low: 1, Close: 1},
		{Time: t0 + 7320, Open: 1, High: 1, Low: 1, Close: 1},
		{Time: t0 + 10920, Open: 1, High: 1, Low: 1, Close: 1},
	}
	if got := aggregate1hTo4h(misaligned); len(got) != 0 {
		t.Errorf("misaligned bars aggregated: %v", got)
	}
	if got := aggregate1hTo4h(nil); len(got) != 0 {
		t.Errorf("nil input must aggregate to empty, got %v", got)
	}
}

// ── tf plumbing: the requested interval reaches the fetcher ──────────────────

// stubBinanceRecord serves 250 rising closed candles for ANY symbol at the
// REQUESTED interval spacing and records "symbol|interval" per request.
func stubBinanceRecord(t *testing.T) *[]string {
	t.Helper()
	var mu sync.Mutex
	calls := []string{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		mu.Lock()
		calls = append(calls, q.Get("symbol")+"|"+q.Get("interval"))
		mu.Unlock()
		sec := intervalSeconds[q.Get("interval")]
		if sec == 0 {
			sec = 14400
		}
		bars := 250
		start := time.Now().Unix() - int64(bars+2)*sec
		rows := make([][]any, bars)
		price := 60000.0
		for i := range rows {
			price += 50
			rows[i] = []any{
				float64(start+int64(i)*sec) * 1000,
				fmt.Sprintf("%f", price-25), fmt.Sprintf("%f", price+100),
				fmt.Sprintf("%f", price-100), fmt.Sprintf("%f", price),
				"100",
			}
		}
		body, _ := json.Marshal(rows)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	orig := binanceKlinesBase
	binanceKlinesBase = srv.URL
	t.Cleanup(func() { binanceKlinesBase = orig })
	return &calls
}

func TestMomentumScanTFMatrix(t *testing.T) {
	for _, tf := range []string{"1h", "4h", "1d"} {
		t.Run(tf, func(t *testing.T) {
			calls := stubBinanceRecord(t)
			ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
			c := ag.MomentumScanCard(context.Background(), []string{"btc", "eth"}, tf)
			if c.Offline {
				t.Fatalf("scan offline: %q", c.Verdict)
			}
			want := map[string]bool{"BTCUSDT|" + tf: true, "ETHUSDT|" + tf: true}
			for _, call := range *calls {
				delete(want, call)
			}
			if len(want) > 0 {
				t.Errorf("fetches missing for %v (got %v)", want, *calls)
			}
			joined := strings.Join(c.Facts, "|")
			if !strings.Contains(joined, tf+" candles") {
				t.Errorf("rule line must name the timeframe: %v", c.Facts)
			}
			if !strings.Contains(c.Verdict, "BTC:") || !strings.Contains(c.Verdict, "ETH:") {
				t.Errorf("verdict must cover both assets: %q", c.Verdict)
			}
		})
	}
}

// ── HTTP: assets/tf validation + envelope ────────────────────────────────────

func TestHTTPMomentumScanAndValidation(t *testing.T) {
	stubBinanceRecord(t)
	deadYahoo(t)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)

	// Happy path: two Binance assets at 1d.
	status, _, body := httpGet(t, srv.URL+"/agents/momentum?assets=btc,eth&tf=1d")
	if status != 200 {
		t.Fatalf("scan status %d (%s)", status, body)
	}
	var env testEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(env.Asset, "BTC") || !strings.Contains(env.Asset, "ETH") {
		t.Errorf("asset label = %q, want both scanned assets", env.Asset)
	}
	if !env.OK {
		t.Errorf("scan envelope ok = false: %s", body)
	}

	cases := []struct {
		path, wantMsg string
	}{
		{"/agents/momentum?assets=btc,doge", "unknown asset"},
		{"/agents/momentum?assets=btc,doge", "eurusd"}, // the allowed list rides in the message
		{"/agents/momentum?assets=", "empty"},
		{"/agents/momentum?tf=2h", "1h, 4h, 1d"},
		{"/agents/momentum?asset=btc&assets=btc,eth", "either"},
		{"/agents/trend?tf=1d", "does not take a ?tf= parameter"},
		{"/agents/sr?assets=btc", "does not take a ?assets= parameter"},
	}
	for _, tc := range cases {
		status, _, body := httpGet(t, srv.URL+tc.path)
		if status != 400 {
			t.Errorf("%s: status %d, want 400 (%s)", tc.path, status, body)
			continue
		}
		if !strings.Contains(string(body), tc.wantMsg) {
			t.Errorf("%s: body %s must contain %q", tc.path, body, tc.wantMsg)
		}
	}

	// tf alone re-bases the default trio (XAUUSD degrades honestly — Yahoo is
	// dead in this test — but the crypto legs still answer).
	status, _, body = httpGet(t, srv.URL+"/agents/momentum?tf=1d")
	if status != 200 {
		t.Fatalf("tf-only status %d (%s)", status, body)
	}

	// Single asset + tf still works through ?asset=.
	status, _, body = httpGet(t, srv.URL+"/agents/momentum?asset=btc&tf=1d")
	if status != 200 {
		t.Fatalf("asset+tf status %d (%s)", status, body)
	}
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(env.Facts, "|"), "1d candles") {
		t.Errorf("single-asset card must run on 1d: %v", env.Facts)
	}
}

// Over-limit scans are rejected with the cap named. The registry holds exactly
// 6 assets, so the cap can only trip on lists with repeats — which dedup would
// otherwise absorb — hence the cap is asserted directly on the parser boundary
// via a crafted long list of DISTINCT aliases mapping to distinct keys being
// impossible; instead the HTTP layer rejects >6 RAW entries before dedup.
func TestHTTPMomentumScanRawLimit(t *testing.T) {
	stubBinanceRecord(t)
	deadYahoo(t)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)
	status, _, body := httpGet(t, srv.URL+"/agents/momentum?assets=btc,btc,btc,btc,btc,btc,btc")
	if status != 400 || !strings.Contains(string(body), "6") {
		t.Errorf("7-entry list: status %d body %s, want 400 naming the 6-asset cap", status, body)
	}
}

// ── Yahoo 4h aggregation reaches the card ────────────────────────────────────

// stubYahoo1h serves a Yahoo chart JSON with `hours` hourly bars ending in the
// past (all closed), rising closes.
func stubYahoo1h(t *testing.T, hours int) {
	t.Helper()
	end := time.Now().Add(-2 * time.Hour).Truncate(4 * time.Hour) // aligned + closed
	start := end.Add(-time.Duration(hours) * time.Hour)
	ts := make([]int64, hours)
	opens := make([]float64, hours)
	highs := make([]float64, hours)
	lows := make([]float64, hours)
	closes := make([]float64, hours)
	for i := 0; i < hours; i++ {
		ts[i] = start.Unix() + int64(i)*3600
		p := 2000 + float64(i)
		opens[i], highs[i], lows[i], closes[i] = p, p+1, p-1, p+0.5
	}
	payload := map[string]any{
		"chart": map[string]any{
			"result": []any{map[string]any{
				"timestamp": ts,
				"indicators": map[string]any{
					"quote": []any{map[string]any{
						"open": opens, "high": highs, "low": lows, "close": closes,
						"volume": make([]any, hours), // FX-style null volume
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

func TestMomentumYahoo4hAggregation(t *testing.T) {
	stubYahoo1h(t, 400) // 400 hourly bars → ~100 aggregated 4h bars, plenty for RSI/MACD
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	spec, err := specWithTF(assetTable["eurusd"], "4h")
	if err != nil {
		t.Fatal(err)
	}
	c := ag.MomentumAssetCard(context.Background(), spec)
	if c.Offline {
		t.Fatalf("card offline: %q", c.Verdict)
	}
	joined := strings.Join(c.Facts, "|")
	if !strings.Contains(joined, "4h candles") {
		t.Errorf("rule line must name 4h: %v", c.Facts)
	}
	// The aggregation must be DISCLOSED — Yahoo has no native 4h.
	if !strings.Contains(joined, "aggregated from Yahoo 1h") {
		t.Errorf("aggregation note missing: %v", c.Facts)
	}
}

// ── review fixes: machine status, duplicate params, singleflight ─────────────

// stubBinanceBarsPerSymbol serves per-symbol bar counts (rising closed
// candles at the requested interval); symbols absent from the map get 250.
func stubBinanceBarsPerSymbol(t *testing.T, bars map[string]int) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		n, ok := bars[q.Get("symbol")]
		if !ok {
			n = 250
		}
		sec := intervalSeconds[q.Get("interval")]
		if sec == 0 {
			sec = 14400
		}
		start := time.Now().Unix() - int64(n+2)*sec
		rows := make([][]any, n)
		price := 60000.0
		for i := range rows {
			price += 50
			rows[i] = []any{
				float64(start+int64(i)*sec) * 1000,
				fmt.Sprintf("%f", price-25), fmt.Sprintf("%f", price+100),
				fmt.Sprintf("%f", price-100), fmt.Sprintf("%f", price),
				"100",
			}
		}
		body, _ := json.Marshal(rows)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	orig := binanceKlinesBase
	binanceKlinesBase = srv.URL
	t.Cleanup(func() { binanceKlinesBase = orig })
}

// Review fix 3: every asset fetched fine but too short → the scan is
// insufficient_history, never source_offline.
func TestMomentumScanAllInsufficient(t *testing.T) {
	stubBinanceBarsPerSymbol(t, map[string]int{"BTCUSDT": 20, "ETHUSDT": 20})
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.MomentumScanCard(context.Background(), []string{"btc", "eth"}, "")
	if c.Status != statusInsufficientHistory || !c.Offline {
		t.Errorf("status=%v offline=%v, want insufficient_history 503-path", c.Status, c.Offline)
	}

	_, srv := newTestAPI(t, ag, true)
	status, _, body := httpGet(t, srv.URL+"/agents/momentum?assets=btc,eth")
	if status != 503 {
		t.Fatalf("status %d, want 503 (%s)", status, body)
	}
	var e struct {
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(body, &e); err != nil {
		t.Fatal(err)
	}
	if e.Reason != "insufficient_history" {
		t.Errorf("reason = %q, want insufficient_history (sources answered, series too short)", e.Reason)
	}
}

// Review fix 3: mixed outcomes stay 200/ok:true (≥1 real reading) and carry a
// machine-readable per-asset results array.
func TestMomentumScanMixedResults(t *testing.T) {
	stubBinanceBarsPerSymbol(t, map[string]int{"BTCUSDT": 250, "ETHUSDT": 20})
	deadYahoo(t) // eurusd → source_offline
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)

	status, _, body := httpGet(t, srv.URL+"/agents/momentum?assets=btc,eth,eurusd")
	if status != 200 {
		t.Fatalf("status %d, want 200 (%s)", status, body)
	}
	var env testEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if !env.OK {
		t.Errorf("ok = false, want true (BTC produced a real reading)")
	}
	if len(env.Results) != 3 {
		t.Fatalf("results = %+v, want 3 entries", env.Results)
	}
	byAsset := map[string]testAssetResult{}
	for _, r := range env.Results {
		byAsset[r.Asset] = r
	}
	if r := byAsset["BTC"]; !r.OK || r.Reason != nil {
		t.Errorf("BTC result = %+v, want ok:true", r)
	}
	if r := byAsset["ETH"]; r.OK || r.Reason == nil || *r.Reason != "insufficient_history" {
		t.Errorf("ETH result = %+v, want insufficient_history", r)
	}
	if r := byAsset["EURUSD"]; r.OK || r.Reason == nil || *r.Reason != "source_offline" {
		t.Errorf("EURUSD result = %+v, want source_offline", r)
	}
}

// The default multi-asset card carries the same per-asset results array.
func TestMomentumDefaultCardResults(t *testing.T) {
	stubBinanceRecord(t)
	deadYahoo(t)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)
	status, _, body := httpGet(t, srv.URL+"/agents/momentum")
	if status != 200 {
		t.Fatalf("status %d (%s)", status, body)
	}
	var env testEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if len(env.Results) != 3 {
		t.Fatalf("results = %+v, want BTC/ETH/XAUUSD entries", env.Results)
	}
	for _, r := range env.Results {
		if r.Asset == "XAUUSD" && (r.OK || r.Reason == nil || *r.Reason != "source_offline") {
			t.Errorf("XAUUSD result = %+v, want source_offline (dead Yahoo)", r)
		}
	}
}

// Review fix 5: repeated query params are a 400, never a silent first-wins.
func TestHTTPDuplicateParams(t *testing.T) {
	stubBinanceRecord(t)
	deadYahoo(t)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)
	for _, path := range []string{
		"/agents/momentum?assets=btc&assets=eth",
		"/agents/momentum?asset=btc&asset=eth",
		"/agents/momentum?tf=1h&tf=1d",
		"/agents/trend?asset=btc&asset=eth",
	} {
		status, _, body := httpGet(t, srv.URL+path)
		if status != 400 {
			t.Errorf("%s: status %d, want 400 (%s)", path, status, body)
			continue
		}
		if !strings.Contains(string(body), "duplicate parameter") {
			t.Errorf("%s: body %s must name the duplicate parameter", path, body)
		}
	}
}

// Review fix 10: concurrent cache misses on one key produce exactly ONE
// upstream call (singleflight), and errors are joined but never cached.
func TestKlineCacheSingleflight(t *testing.T) {
	var hits int32
	block := make(chan struct{})
	cache := newKlineCache()
	load := func() ([]types.OHLCVCandle, error) {
		atomic.AddInt32(&hits, 1)
		<-block
		return []types.OHLCVCandle{{Time: 1, Close: 2}}, nil
	}

	var wg sync.WaitGroup
	errs := make([]error, 10)
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, errs[i] = cache.cached("k", load)
		}(i)
	}
	time.Sleep(50 * time.Millisecond) // let every goroutine reach the flight
	close(block)
	wg.Wait()
	if got := atomic.LoadInt32(&hits); got != 1 {
		t.Errorf("upstream calls = %d, want 1 (singleflight)", got)
	}
	for i, err := range errs {
		if err != nil {
			t.Errorf("goroutine %d: err = %v", i, err)
		}
	}

	// Errors are shared with concurrent joiners but NOT cached: the next
	// wave retries the upstream.
	var failHits int32
	failLoad := func() ([]types.OHLCVCandle, error) {
		atomic.AddInt32(&failHits, 1)
		return nil, fmt.Errorf("boom")
	}
	if _, err := cache.cached("fail", failLoad); err == nil {
		t.Fatal("want the load error")
	}
	if _, err := cache.cached("fail", failLoad); err == nil {
		t.Fatal("want the retried load error")
	}
	if got := atomic.LoadInt32(&failHits); got != 2 {
		t.Errorf("failed loads = %d, want 2 (errors never cached)", got)
	}
}

// ── Telegram parity ──────────────────────────────────────────────────────────

func TestTelegramMomentumScanParity(t *testing.T) {
	calls := stubBinanceRecord(t)
	deadYahoo(t) // the default-trio path includes XAUUSD — keep it offline-honest
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	b := newBotWithClient(NewTGClient("test-token"), ag)

	// "/momentum btc,eth 1d" — comma list + tf token.
	reply, kb := b.buildReply(context.Background(), keyMomentum, []string{"btc,eth", "1d"})
	if !strings.Contains(reply, "BTC:") || !strings.Contains(reply, "ETH:") {
		t.Errorf("reply must carry both assets:\n%s", reply)
	}
	if !strings.Contains(reply, "1d candles") {
		t.Errorf("reply must name the timeframe:\n%s", reply)
	}
	found := false
	for _, call := range *calls {
		if call == "BTCUSDT|1d" {
			found = true
		}
	}
	if !found {
		t.Errorf("1d never reached the fetcher: %v", *calls)
	}
	// Refresh must re-run the SAME scan: command + list + tf in the callback.
	if kb == nil || len(kb.InlineKeyboard) == 0 {
		t.Fatal("keyboard missing")
	}
	if got := kb.InlineKeyboard[0][0].CallbackData; got != "r|momentum btc,eth 1d" {
		t.Errorf("refresh callback = %q, want r|momentum btc,eth 1d", got)
	}

	// Bad entry answers with the allowed list, no card.
	reply, _ = b.buildReply(context.Background(), keyMomentum, []string{"btc,doge"})
	if !strings.Contains(reply, "doge") || !strings.Contains(reply, "eurusd") {
		t.Errorf("bad-entry reply must name the entry and the allowed list:\n%s", reply)
	}

	// Plain "/momentum eurusd" single-asset path stays intact (spot check via
	// tf token only): "/momentum 1d" re-bases the default trio.
	reply, _ = b.buildReply(context.Background(), keyMomentum, []string{"1d"})
	if !strings.Contains(reply, "BTC:") {
		t.Errorf("tf-only reply must still scan the default set:\n%s", reply)
	}
}
