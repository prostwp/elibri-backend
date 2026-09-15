package demobot

// trend_window_levels_test.go — two product decisions of 2026-09-15:
//
//  1. levels.invalidation / invalidation_side exist ONLY for a confirmed trend
//     (up/down); flat, grey (structure-demoted included) and conflict ship no
//     levels object at all — never a zero, never a side without a level.
//  2. The Trend Agent (card and /agents/trend/chart) reads 1000 raw Binance
//     bars (999 closed); momentum, S/R and vol keep 250 raw (249 closed). All
//     of them are served from ONE upstream request per symbol|interval.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

func trendLevelPtr(v float64) *float64 { return &v }

// levelsKeys returns the JSON keys of an envelope's levels object, or nil when
// the envelope has none.
func levelsKeys(t *testing.T, c Card) (map[string]json.RawMessage, bool) {
	t.Helper()
	b, err := json.Marshal(cardEnvelope(c))
	if err != nil {
		t.Fatal(err)
	}
	var env map[string]json.RawMessage
	if err := json.Unmarshal(b, &env); err != nil {
		t.Fatal(err)
	}
	raw, has := env["levels"]
	if !has {
		return nil, false
	}
	var lv map[string]json.RawMessage
	if err := json.Unmarshal(raw, &lv); err != nil {
		t.Fatal(err)
	}
	return lv, true
}

func sortedKeys(m map[string]json.RawMessage) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Decision 1, as a table over every state: the Go value and the JSON contract.
func TestTrendLevelsOnlyWhenConfirmed(t *testing.T) {
	cases := []struct {
		key      string
		atr      float64 // -1 = the fixture's own ATR
		wantKeys []string
		wantSide string
	}{
		{"up_inside", -1, []string{"invalidation", "invalidation_side", "pullback_zone"}, "below"},
		{"down", -1, []string{"invalidation", "invalidation_side", "pullback_zone"}, "above"},
		{"down", 0, []string{"pullback_zone"}, ""}, // confirmed, ATR = 0: zone only
		{"flat", -1, nil, ""},
		{"grey", -1, nil, ""},
		{"conflict", -1, nil, ""},
		{"demoted", -1, nil, ""}, // structure-demoted grey
		{"ema_equal", -1, nil, ""},
		{"flat", 0, nil, ""},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%s/atr=%v", tc.key, tc.atr), func(t *testing.T) {
			f := trendFixtures[tc.key]
			if tc.atr >= 0 {
				f.atr = tc.atr
			}
			c := renderFixture(f)
			lvJSON, has := levelsKeys(t, c)

			if tc.wantKeys == nil {
				if c.Levels != nil || has {
					t.Fatalf("state %q must ship no levels, got %+v / %v", c.State, c.Levels, lvJSON)
				}
				return
			}
			if !has {
				t.Fatalf("confirmed %q must ship levels", c.State)
			}
			if got := sortedKeys(lvJSON); !reflect.DeepEqual(got, tc.wantKeys) {
				t.Errorf("levels keys = %v, want %v", got, tc.wantKeys)
			}
			lv := c.Levels.(TrendLevels)
			if tc.wantSide == "" {
				if lv.Invalidation != nil || lv.InvalidationSide != "" {
					t.Errorf("no ATR → no invalidation and no side, got %+v", lv)
				}
				return
			}
			want, side := invalidationFor(c.State, f.r.EMA50, f.r.EMA200, f.atr)
			if lv.Invalidation == nil || *lv.Invalidation != want || side != tc.wantSide || lv.InvalidationSide != tc.wantSide {
				t.Errorf("invalidation %+v, want %v %s", lv, want, tc.wantSide)
			}
			var gotLevel float64
			if err := json.Unmarshal(lvJSON["invalidation"], &gotLevel); err != nil || gotLevel != want {
				t.Errorf("JSON invalidation %s, want %v", lvJSON["invalidation"], want)
			}
		})
	}
}

// trendTestSeries builds n closed 4h candles from a price path (±100 wicks).
func trendTestSeries(n int, price func(i int) float64) []types.OHLCVCandle {
	start := time.Now().Unix() - int64(n+2)*14400
	out := make([]types.OHLCVCandle, n)
	for i := range out {
		p := price(i)
		out[i] = types.OHLCVCandle{Time: start + int64(i)*14400, Open: p - 25, High: p + 100, Low: p - 100, Close: p, Volume: 100}
	}
	return out
}

// Card and chart agree on presence AND value of the invalidation (and on the
// EMAs and state behind it) for the same candles, in every state reachable
// from the fixtures.
func TestTrendCardAndChartAgreeOnLevels(t *testing.T) {
	sets := map[string][]types.OHLCVCandle{
		// 250 bars: zigzagDown turns negative past ~600, as its B3 tests assume.
		"up (zigzag)":         trendTestSeries(250, zigzagUp),
		"down (zigzag)":       trendTestSeries(250, zigzagDown),
		"up, unreadable":      hhhlUnreadableOverUptrend(250),
		"demoted (LH/LL)":     lhllOverUptrend(250),
		"unconfirmed (cycle)": srCardCycleCandles(250, flatVol),
		"flat (constant)":     trendTestSeries(250, func(int) float64 { return 60000 }),
	}
	states := map[string]bool{}
	for name, candles := range sets {
		t.Run(name, func(t *testing.T) {
			r := trendReadOf(candles)
			if !r.OK {
				t.Fatal("fixture must produce a reading")
			}
			states[r.State] = true
			card := trendCardFrom(r, btcSpec, trendTestTime)
			chart, ok := trendChartOf(btcSpec, "btc", candles)
			if !ok {
				t.Fatal("chart must build")
			}
			if chart.State != card.State {
				t.Fatalf("state chart %q != card %q", chart.State, card.State)
			}
			last := func(s []chartPoint) float64 { return s[len(s)-1].Value }
			if last(chart.EMA20) != r.EMA20 || last(chart.EMA50) != r.EMA50 || last(chart.EMA200) != r.EMA200 {
				t.Errorf("chart EMAs %v/%v/%v != read %v/%v/%v",
					last(chart.EMA20), last(chart.EMA50), last(chart.EMA200), r.EMA20, r.EMA50, r.EMA200)
			}
			lv, has := card.Levels.(TrendLevels)
			switch {
			case chart.Invalidation == nil && (!has || lv.Invalidation == nil):
				// agree: no level
			case chart.Invalidation == nil || !has || lv.Invalidation == nil:
				t.Errorf("presence differs: chart %+v, card %+v", chart.Invalidation, card.Levels)
			case chart.Invalidation.Level != *lv.Invalidation || chart.Invalidation.Side != lv.InvalidationSide:
				t.Errorf("value differs: chart %+v, card %v %s", chart.Invalidation, *lv.Invalidation, lv.InvalidationSide)
			}
			if card.State != trendUp && card.State != trendDown && card.Levels != nil {
				t.Errorf("unconfirmed %q card must have no levels", card.State)
			}
		})
	}
	for _, s := range []string{trendUp, trendDown, trendGrey} {
		if !states[s] {
			t.Errorf("fixtures never reached state %q — the agreement test proves less than it says", s)
		}
	}
}

// ── decision 2: the trend window ─────────────────────────────────────────────

// binanceWindowStub serves the newest `limit` rows of a fixed raw series (the
// way Binance does), the last row still FORMING, and counts requests per
// symbol|interval together with the limits asked for.
type binanceWindowStub struct {
	mu     sync.Mutex
	calls  map[string]int
	limits []int
	delay  time.Duration // per-response delay, so concurrent callers overlap
}

func newBinanceWindowStub(t *testing.T, raw []types.OHLCVCandle) *binanceWindowStub {
	t.Helper()
	s := &binanceWindowStub{calls: map[string]int{}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		limit, _ := strconv.Atoi(q.Get("limit"))
		s.mu.Lock()
		s.calls[q.Get("symbol")+"|"+q.Get("interval")]++
		s.limits = append(s.limits, limit)
		delay := s.delay
		s.mu.Unlock()
		time.Sleep(delay)
		rows := raw
		if limit > 0 && limit < len(rows) {
			rows = rows[len(rows)-limit:]
		}
		out := make([][]any, len(rows))
		for i, c := range rows {
			out[i] = []any{float64(c.Time) * 1000,
				fmt.Sprintf("%f", c.Open), fmt.Sprintf("%f", c.High),
				fmt.Sprintf("%f", c.Low), fmt.Sprintf("%f", c.Close), fmt.Sprintf("%f", c.Volume)}
		}
		body, _ := json.Marshal(out)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	orig := binanceKlinesBase
	binanceKlinesBase = srv.URL
	t.Cleanup(func() { binanceKlinesBase = orig })
	return s
}

// rawBinanceSeries: n raw 4h bars, a swinging rising path, the last bar
// opened an hour ago (still forming — dropped as the live forming bar is).
func rawBinanceSeries(n int) []types.OHLCVCandle {
	now := time.Now().Unix()
	start := now - 3600 - int64(n-1)*14400
	out := make([]types.OHLCVCandle, n)
	for i := range out {
		p := zigzagUp(i)
		out[i] = types.OHLCVCandle{Time: start + int64(i)*14400,
			Open: p - 25, High: p + 100 + float64(i%7)*30, Low: p - 100 - float64(i%5)*30, Close: p,
			Volume: 100 + float64(i%11)*10}
	}
	return out
}

func TestTrendWindowIs999ClosedOthers249OneRequest(t *testing.T) {
	raw := rawBinanceSeries(1200)
	stub := newBinanceWindowStub(t, raw)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	ctx := context.Background()

	trendBars, err := ag.trendCandlesFor(ctx, btcSpec)
	if err != nil {
		t.Fatal(err)
	}
	baseBars, err := ag.candlesFor(ctx, btcSpec)
	if err != nil {
		t.Fatal(err)
	}
	if len(trendBars) != 999 || len(baseBars) != 249 {
		t.Fatalf("closed bars: trend %d (want 999), others %d (want 249)", len(trendBars), len(baseBars))
	}
	// The 249 are exactly the newest closed bars of the 999 — the same bars
	// a limit=250 request returns.
	if !reflect.DeepEqual(baseBars, trendBars[len(trendBars)-249:]) {
		t.Error("the 249-bar window must be the tail of the 999-bar one")
	}

	// Every Binance consumer within the cache TTL: one upstream request.
	_ = ag.TrendCard(ctx, btcSpec)
	_ = ag.SRCard(ctx, btcSpec)
	_ = ag.VolCard(ctx, btcSpec)
	_ = ag.MomentumAssetCard(ctx, btcSpec)
	_ = ag.lastBTCClose(ctx)
	_, srv := newTestAPI(t, ag, true)
	if code, _, _, body := getChart(t, srv.URL+"/agents/trend/chart?asset=btc"); code != http.StatusOK {
		t.Fatalf("chart status %d: %s", code, body)
	}
	stub.mu.Lock()
	calls, limits := stub.calls["BTCUSDT|4h"], append([]int{}, stub.limits...)
	stub.mu.Unlock()
	if calls != 1 {
		t.Errorf("BTCUSDT|4h upstream requests = %d, want 1 (calls %v)", calls, stub.calls)
	}
	for _, l := range limits {
		if l != binanceFetchLimit {
			t.Errorf("upstream limit = %d, want %d", l, binanceFetchLimit)
		}
	}
}

// Momentum, S/R and vol produce byte-identical output to the old behaviour:
// a server that only ever returns the newest 250 raw rows (what their
// limit=250 request got) vs the new superset-and-tail path.
func TestNonTrendAgentsUnchangedByTrendWindow(t *testing.T) {
	raw := rawBinanceSeries(1200)
	// Every production consumer of the short window. gather and the showcase
	// also touch funding (live Binance futures) and FX (Yahoo): both are
	// pinned dead so the comparison is about the kline window only.
	deadYahoo(t)
	deadFunding := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "down", http.StatusInternalServerError)
	}))
	t.Cleanup(deadFunding.Close)
	origPremium := premiumIndexURL
	premiumIndexURL = deadFunding.URL + "/?symbol="
	t.Cleanup(func() { premiumIndexURL = origPremium })

	envJSON := func(c Card) string {
		b, err := encodeJSON(cardEnvelope(c))
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	render := func(ag *Agents) map[string]string {
		ctx := context.Background()
		out := map[string]string{}
		for name, c := range map[string]Card{
			"sr":               ag.SRCard(ctx, btcSpec),
			"vol":              ag.VolCard(ctx, btcSpec),
			"momentum (asset)": ag.MomentumAssetCard(ctx, btcSpec),
			"momentum (multi)": ag.MomentumCard(ctx),
			"momentum scan":    ag.MomentumScanCard(ctx, []string{"btc", "eth"}, ""),
			"momentum scan 1h": ag.MomentumScanCard(ctx, []string{"btc", "eth"}, "1h"),
		} {
			out[name] = envJSON(c)
		}
		out["lastBTCClose"] = fmt.Sprint(ag.lastBTCClose(ctx))

		// The digest sweep: its non-trend cards (trend, and so digest/top,
		// legitimately change with the new window).
		g := ag.gather(ctx)
		for _, k := range []string{keyMomentum, keySR, keyVol} {
			out["gather "+k] = envJSON(g.cards[k]) + " | " + g.cards[k].OneLiner()
		}

		// The landing showcase rows of the same agents.
		s, _ := newTestAPI(t, ag, true)
		for _, row := range s.showcase(ctx).rows() {
			if row.Slug == keyMomentum || row.Slug == keySR || row.Slug == keyVol {
				b, _ := json.Marshal(row)
				out["showcase "+row.Slug] = string(b)
			}
		}
		return out
	}

	newBinanceWindowStub(t, raw) // honours limit; new code asks for 1000
	now := render(NewAgents(NewBackendClient("http://127.0.0.1:1")))

	newBinanceWindowStub(t, raw[len(raw)-250:]) // the old limit=250 answer, whatever is asked
	before := render(NewAgents(NewBackendClient("http://127.0.0.1:1")))

	for name := range now {
		if now[name] != before[name] {
			t.Errorf("%s changed with the trend window:\n now:    %s\n before: %s", name, now[name], before[name])
		}
	}
}

// Trend (1000) and short-window (250) callers arriving TOGETHER for the same
// symbol|interval join one in-flight load: exactly one upstream request, and
// each caller still gets its own window. Meaningful under -race.
func TestTrendAndShortWindowsConcurrentOneRequest(t *testing.T) {
	stub := newBinanceWindowStub(t, rawBinanceSeries(1200))
	stub.mu.Lock()
	stub.delay = 150 * time.Millisecond
	stub.mu.Unlock()
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	ctx := context.Background()

	start := make(chan struct{})
	var wg sync.WaitGroup
	var mu sync.Mutex
	var long, short [][]types.OHLCVCandle
	get := func(f func(context.Context, assetSpec) ([]types.OHLCVCandle, error), into *[][]types.OHLCVCandle) {
		defer wg.Done()
		<-start
		b, err := f(ctx, btcSpec)
		if err != nil {
			t.Errorf("window fetch: %v", err)
			return
		}
		mu.Lock()
		*into = append(*into, b)
		mu.Unlock()
	}
	for i := 0; i < 6; i++ {
		wg.Add(2)
		go get(ag.trendCandlesFor, &long)
		go get(ag.candlesFor, &short)
	}
	for _, f := range []func(){
		func() { _ = ag.TrendCard(ctx, btcSpec) },
		func() { _ = ag.SRCard(ctx, btcSpec) },
		func() { _ = ag.MomentumAssetCard(ctx, btcSpec) },
	} {
		wg.Add(1)
		go func(f func()) {
			defer wg.Done()
			<-start
			f()
		}(f)
	}
	close(start)
	wg.Wait()

	stub.mu.Lock()
	calls := stub.calls["BTCUSDT|4h"]
	stub.mu.Unlock()
	if calls != 1 {
		t.Errorf("concurrent callers made %d upstream requests, want 1", calls)
	}
	if len(long) != 6 || len(short) != 6 {
		t.Fatalf("got %d long and %d short windows, want 6 each", len(long), len(short))
	}
	for i := range long {
		if len(long[i]) != 999 || len(short[i]) != 249 {
			t.Errorf("window sizes %d/%d, want 999/249", len(long[i]), len(short[i]))
			continue
		}
		if !reflect.DeepEqual(short[i], long[0][len(long[0])-249:]) {
			t.Error("a short window is not the tail of the long one")
		}
	}
}

// The card and the chart read the same 999 bars: state, EMAs and the
// invalidation are identical through the real handlers.
func TestTrendCardAndChartShareTheLongWindow(t *testing.T) {
	newBinanceWindowStub(t, rawBinanceSeries(1200))
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	ctx := context.Background()
	bars, err := ag.trendCandlesFor(ctx, btcSpec)
	if err != nil {
		t.Fatal(err)
	}
	r := trendReadOf(bars)
	card := ag.TrendCard(ctx, btcSpec)
	_, srv := newTestAPI(t, ag, true)
	code, _, chart, body := getChart(t, srv.URL+"/agents/trend/chart?asset=btc")
	if code != http.StatusOK {
		t.Fatalf("chart status %d: %s", code, body)
	}
	if card.State != r.State || chart.State != r.State {
		t.Errorf("states: card %q chart %q read %q", card.State, chart.State, r.State)
	}
	last := func(s []chartPoint) float64 { return s[len(s)-1].Value }
	if last(chart.EMA200) != r.EMA200 || last(chart.EMA50) != r.EMA50 || last(chart.EMA20) != r.EMA20 {
		t.Errorf("chart EMAs differ from the 999-bar read")
	}
	// EMA200 now covers every returned candle (999 bars ≫ 200 + 200).
	if len(chart.EMA200) != len(chart.Candles) {
		t.Errorf("EMA200 covers %d of %d candles", len(chart.EMA200), len(chart.Candles))
	}
	lv, _ := card.Levels.(TrendLevels)
	if (chart.Invalidation == nil) != (lv.Invalidation == nil) ||
		(chart.Invalidation != nil && (chart.Invalidation.Level != *lv.Invalidation || chart.Invalidation.Side != lv.InvalidationSide)) {
		t.Errorf("invalidation: chart %+v, card %+v", chart.Invalidation, card.Levels)
	}
}
