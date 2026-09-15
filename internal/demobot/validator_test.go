package demobot

// validator_test.go — which responses may carry Last-Modified, and what it
// must cover. Rule: a validator only where the whole body is a function of
// the stamped data; a false 304 hides a changed body from the client's cache.
//   - Stamped: single-asset Binance cards and the Binance trend chart (the
//     close of the last bar Binance has already returned the NEXT bar for,
//     closed by the fetch time, on a complete contiguous window), and
//     /showcase (composite_modtime_test.go).
//   - No validator (always 200, If-Modified-Since ignored): every Yahoo read
//     (late or revised bars, clock-driven market state), gold, fx, funding,
//     the composite momentum card, any multi-asset scan, news (hourly AI
//     idea), whale (live transfer table) and macro on every path, UNKNOWN
//     included (request-time captured_at).

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// cget is a GET with an optional If-Modified-Since: status, the Last-Modified
// header as sent ("" when absent) and the raw body.
func cget(t *testing.T, url, ims string) (int, string, []byte) {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	if ims != "" {
		req.Header.Set("If-Modified-Since", ims)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, resp.Header.Get("Last-Modified"), body
}

// stubYahooWave serves Yahoo chart JSON for any symbol: interval=1d → 300
// closed daily bars; anything else → `hours` hourly bars whose last CLOSES at
// end. Prices run a 10-bar triangle wave, so swings, ATR and EMAs all exist.
func stubYahooWave(t *testing.T, end time.Time, hours int) {
	t.Helper()
	wave := []float64{0, 2, 4, 6, 8, 10, 8, 6, 4, 2}
	chart := func(n int, openAt func(i int) int64, base, step float64) []byte {
		ts := make([]int64, n)
		o, h, l, c := make([]float64, n), make([]float64, n), make([]float64, n), make([]float64, n)
		for i := 0; i < n; i++ {
			ts[i] = openAt(i)
			p := base + wave[i%len(wave)]*step
			o[i], h[i], l[i], c[i] = p, p+step*2, p-step*2, p
		}
		b, err := json.Marshal(map[string]any{
			"chart": map[string]any{"result": []any{map[string]any{
				"timestamp": ts,
				"indicators": map[string]any{"quote": []any{map[string]any{
					"open": o, "high": h, "low": l, "close": c, "volume": make([]any, n),
				}}},
			}}},
		})
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	lastOpen := end.Add(-time.Hour).Unix()
	hourly := chart(hours, func(i int) int64 { return lastOpen - int64(hours-1-i)*3600 }, 1.1, 0.0005)
	const days = 300
	dayEnd := time.Now().UTC().Truncate(24 * time.Hour).Add(-48 * time.Hour).Unix()
	daily := chart(days, func(i int) int64 { return dayEnd - int64(days-1-i)*86400 }, 2000, 5)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Query().Get("interval") == "1d" {
			_, _ = w.Write(daily)
			return
		}
		_, _ = w.Write(hourly)
	}))
	t.Cleanup(srv.Close)
	orig := yahooChartBase
	yahooChartBase = srv.URL + "/v8/finance/chart/"
	t.Cleanup(func() { yahooChartBase = orig })
}

func liqFixture(at string) string {
	return `{"captured_at":"` + at + `","feed":[
	    {"symbol":"BTCUSDT","side":"long_liq","qty":1,"price":118000,"usd_value":90000,"ts":"` + at + `"}],
	  "zones":[]}`
}

// Every address, one snapshot. Stamped addresses answer their own stamp with
// 304 and no body; unstamped ones never send Last-Modified and answer even a
// future If-Modified-Since (which any stamp would satisfy) with 200 + body.
func TestConditionalGetByAddress(t *testing.T) {
	stubExternalBases(t)                        // funding rates dead; candles re-stubbed below
	stubBinanceKlinesWave(t, binanceFetchLimit) // full, contiguous windows
	stubYahooWave(t, time.Now().UTC().Truncate(time.Hour).Add(-2*time.Hour), 600)
	ag := newStubBackend(t, map[string]string{
		"/api/v1/macro":                macroLiveFixture,
		"/api/v1/whale-flow":           whaleLiveFixture,
		"/api/v1/narratives":           narrativesFixture,
		"/api/v1/funding/liquidations": liqFixture(time.Now().UTC().Format(time.RFC3339)),
	})
	_, srv := newTestAPI(t, ag, true)

	stamped := []string{
		"/agents/trend", "/agents/trend?asset=eth", "/agents/trend?asset=bitcoin", "/agents/sr", "/agents/vol?asset=eth",
		"/agents/momentum?asset=btc", "/agents/momentum?asset=eth&tf=1h", "/agents/momentum?assets=btc",
		"/agents/trend/chart?asset=btc",
	}
	for _, p := range stamped {
		t.Run("stamped "+p, func(t *testing.T) {
			st, lm, body := cget(t, srv.URL+p, "")
			if st != http.StatusOK || lm == "" || len(body) == 0 {
				t.Fatalf("first GET → %d, Last-Modified %q, %d bytes; want 200 with a stamp (%s)", st, lm, len(body), body)
			}
			st2, lm2, body2 := cget(t, srv.URL+p, lm)
			if st2 != http.StatusNotModified || len(body2) != 0 || lm2 != lm {
				t.Errorf("same snapshot + own stamp → %d, Last-Modified %q, %d bytes; want 304, %q, empty", st2, lm2, len(body2), lm)
			}
		})
	}

	future := httpStamp(time.Now().Add(24 * time.Hour))
	unstamped := []string{
		// Yahoo single asset, aliases included, and the Yahoo chart
		"/agents/trend?asset=eurusd", "/agents/sr?asset=gbpusd", "/agents/vol?asset=usdjpy",
		"/agents/momentum?asset=xauusd", "/agents/momentum?asset=gold&tf=4h", "/agents/momentum?assets=eurusd",
		"/agents/trend/chart?asset=eurusd",
		// composites
		"/agents/gold", "/agents/fx", "/agents/funding",
		"/agents/momentum", "/agents/momentum?tf=1h",
		"/agents/momentum?assets=btc,eth", "/agents/momentum?assets=btc,eurusd",
		// backend payloads that change under the same captured_at
		"/agents/news", "/agents/whale",
		"/agents/macro", "/agents/macro?asset=btc", "/agents/macro?asset=gold",
	}
	for _, p := range unstamped {
		t.Run("unstamped "+p, func(t *testing.T) {
			st, lm, body := cget(t, srv.URL+p, "")
			if st != http.StatusOK || lm != "" {
				t.Fatalf("first GET → %d, Last-Modified %q; want 200 and none (%s)", st, lm, body)
			}
			st2, lm2, body2 := cget(t, srv.URL+p, future)
			if st2 != http.StatusOK || lm2 != "" || len(body2) == 0 {
				t.Errorf("future If-Modified-Since → %d, Last-Modified %q, %d bytes; want 200 with the body", st2, lm2, len(body2))
			}
			// data_as_of is untouched: the body still states its data time.
			var env testEnvelope
			if err := json.Unmarshal(body2, &env); err != nil || env.DataAsOf == "" {
				t.Errorf("data_as_of %q (%v) — the data time must stay in the body", env.DataAsOf, err)
			}
		})
	}
}

// Macro on its UNKNOWN path (no lamps at all) returns before any fill step;
// the card still carries no validator on any of the three addresses.
func TestMacroUnknownHasNoValidator(t *testing.T) {
	fixture := `{"captured_at":"2026-09-15T04:44:32Z","tradfin_open":true,"lamps":[]}`
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": fixture})
	_, srv := newTestAPI(t, ag, true)
	future := time.Date(2026, 9, 16, 0, 0, 0, 0, time.UTC).Format(http.TimeFormat)
	for _, p := range []string{"/agents/macro", "/agents/macro?asset=btc", "/agents/macro?asset=gold"} {
		st, lm, body := cget(t, srv.URL+p, "")
		if st != http.StatusOK || lm != "" || !strings.Contains(string(body), "UNKNOWN") {
			t.Errorf("%s → %d, Last-Modified %q; want the 200 UNKNOWN card with no stamp (%s)", p, st, lm, body)
		}
		st2, lm2, body2 := cget(t, srv.URL+p, future)
		if st2 != http.StatusOK || lm2 != "" || len(body2) == 0 {
			t.Errorf("%s + future If-Modified-Since → %d, Last-Modified %q, %d bytes; want 200 with the body", p, st2, lm2, len(body2))
		}
	}
}

// ── Binance: which bars are closed, and which window a card reads ───────────

// seedBTC4h installs a finished, fresh cache entry for BTCUSDT 4h as if the
// candles had been fetched at fetchedAt — the cache is driven directly, so
// no clock stepping is needed around its time.Since TTL.
func seedBTC4h(ag *Agents, candles []types.OHLCVCandle, fetchedAt time.Time) {
	done := make(chan struct{})
	close(done)
	ag.klines.mu.Lock()
	ag.klines.items["binance|BTCUSDT|4h"] = &klineEntry{done: done, candles: candles, at: fetchedAt}
	ag.klines.mu.Unlock()
}

// btcSeries is n contiguous 4h bars, the last closing at lastClose, on a
// 10-bar sawtooth (swings for S/R, a live ATR, a defined RSI/MACD).
func btcSeries(n int, lastClose time.Time) []types.OHLCVCandle {
	cs := make([]types.OHLCVCandle, n)
	for i := range cs {
		p := 60000 + float64(i%10)*100
		cs[i] = types.OHLCVCandle{
			Time: lastClose.Add(-time.Duration(n-i) * 4 * time.Hour).Unix(),
			Open: p, High: p + 50, Low: p - 50, Close: p, Volume: 100,
		}
	}
	return cs
}

// withForming appends the bar that opens at lastClose and is still forming —
// how every real klines answer ends. Its prices are absurd on purpose: they
// must never reach a card.
func withForming(closed []types.OHLCVCandle, lastClose time.Time) []types.OHLCVCandle {
	forming := types.OHLCVCandle{Time: lastClose.Unix(), Open: 1, High: 99999, Low: 1, Close: 99999, Volume: 1}
	return append(append([]types.OHLCVCandle{}, closed...), forming)
}

// binancePaths is every Binance-fed stamped address with the size of the
// closed window it reads: limit−1 (999 for trend and its chart, 249 for the
// rest — see candlesWindow).
var binancePaths = []struct {
	path   string
	window int
}{
	{"/agents/trend", trendKlineLimit - 1},
	{"/agents/trend/chart?asset=btc", trendKlineLimit - 1},
	{"/agents/sr", klineLimit - 1},
	{"/agents/vol", klineLimit - 1},
	{"/agents/momentum?asset=btc", klineLimit - 1},
	{"/agents/momentum?assets=btc", klineLimit - 1},
}

// A bar is served only once Binance has returned the bar AFTER it (and it
// had closed by the fetch time). The first fetch ends with bar T carrying
// intermediate prices and no bar after it — fetched either before T closed,
// or after T closed with the REST answer still lagging. Bar T is not read
// and the stamp is T−4h. The next fetch carries T final plus the forming
// bar: T is in, the stamp is T, and it answers itself with 304.
func TestBinanceBarServedOnlyOnceTheNextBarExists(t *testing.T) {
	stubExternalBases(t) // a cache miss fails loudly; nothing reaches the network
	now := time.Now().UTC().Truncate(time.Second)
	lastClose := now.Add(-10 * time.Second) // bar T closes here
	prevClose := lastClose.Add(-4 * time.Hour)
	first := btcSeries(binanceFetchLimit, lastClose)
	first[len(first)-1].Close = 60870 // intermediate: T is the answer's last row
	final := btcSeries(binanceFetchLimit, lastClose)
	final[len(final)-1].Close = 60930
	next := withForming(final[1:], lastClose) // T final + the forming bar

	for _, sc := range []struct {
		name      string
		fetchedAt time.Time
	}{
		{"fetched before the close", now.Add(-30 * time.Second)},
		{"fetched after the close, REST lagging", now},
	} {
		for _, bp := range binancePaths {
			t.Run(sc.name+" "+bp.path, func(t *testing.T) {
				ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
				_, srv := newTestAPI(t, ag, true)

				seedBTC4h(ag, first, sc.fetchedAt)
				st, lm1, body := cget(t, srv.URL+bp.path, "")
				var env testEnvelope
				_ = json.Unmarshal(body, &env)
				if st != 200 || lm1 != httpStamp(prevClose) || env.DataAsOf != prevClose.Format(time.RFC3339) {
					t.Fatalf("T without a next bar → %d, Last-Modified %q, data_as_of %q; want the previous bar %s",
						st, lm1, env.DataAsOf, prevClose.Format(time.RFC3339))
				}

				seedBTC4h(ag, next, now)
				st, lm2, _ := cget(t, srv.URL+bp.path, lm1)
				if st != 200 || lm2 != httpStamp(lastClose) {
					t.Errorf("T final + next bar, old stamp → %d, Last-Modified %q; want 200 at T %s",
						st, lm2, httpStamp(lastClose))
				}
				if st, _, b := cget(t, srv.URL+bp.path, lm2); st != http.StatusNotModified || len(b) != 0 {
					t.Errorf("same fetch + its stamp → %d, %d bytes; want 304", st, len(b))
				}
			})
		}
	}
}

// A Binance stamp is the close of the last bar, which versions the body only
// when the window the card reads is COMPLETE (exactly limit−1 closed bars)
// and CONTIGUOUS (one interval between opens). A short window or a hole
// inside it (a row the parser skipped, or the source never sent) can change
// indicators, levels and the chart under the same last bar: those serve the
// same body with no validator.
func TestBinanceStampNeedsCompleteContiguousWindow(t *testing.T) {
	stubExternalBases(t)
	now := time.Now().UTC().Truncate(time.Second)
	lastClose := now.Add(-time.Hour) // closed well before the fetch
	closed := btcSeries(binanceFetchLimit-1, lastClose)
	longer := btcSeries(binanceFetchLimit, lastClose)
	// 999 closed rows with one hole 10 bars from the end — inside every window.
	holed := append(append([]types.OHLCVCandle{}, longer[:len(longer)-10]...), longer[len(longer)-9:]...)
	future := httpStamp(now.Add(24 * time.Hour))

	for _, bp := range binancePaths {
		for _, tc := range []struct {
			name    string
			candles []types.OHLCVCandle
			stamped bool
		}{
			{"full", withForming(closed, lastClose), true},
			{"short", withForming(closed[len(closed)-(bp.window-1):], lastClose), false}, // one closed bar short
			{"gap", withForming(holed, lastClose), false},
		} {
			t.Run(tc.name+" "+bp.path, func(t *testing.T) {
				ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
				_, srv := newTestAPI(t, ag, true)
				seedBTC4h(ag, tc.candles, now)

				st, lm, body := cget(t, srv.URL+bp.path, "")
				if st != http.StatusOK || len(body) == 0 {
					t.Fatalf("→ %d, %d bytes; want 200 with the body (%s)", st, len(body), body)
				}
				if tc.stamped {
					if lm != httpStamp(lastClose) {
						t.Errorf("full window: Last-Modified %q, want the last close %q", lm, httpStamp(lastClose))
					}
					if st, _, b := cget(t, srv.URL+bp.path, lm); st != http.StatusNotModified || len(b) != 0 {
						t.Errorf("full window + its stamp → %d, %d bytes; want 304", st, len(b))
					}
					return
				}
				if lm != "" {
					t.Errorf("%s window: Last-Modified %q, want none", tc.name, lm)
				}
				st2, lm2, body2 := cget(t, srv.URL+bp.path, future)
				if st2 != http.StatusOK || lm2 != "" || string(body2) != string(body) {
					t.Errorf("%s window + future If-Modified-Since → %d, Last-Modified %q, same body=%v; want 200, none, the same body",
						tc.name, st2, lm2, string(body2) == string(body))
				}
			})
		}
	}
}

// A fetch taken right at a close carries no forming bar (1000 closed rows,
// last close T); the next one carries 999 of the same closed bars plus the
// forming one. The extra, oldest bar of the first fetch is an extreme spike
// in both windows (index 0 for trend/chart, 750 for the 250-bar tail), so
// reading it visibly changes every body. Invariant: one stamp, one body —
// and under the rule each window is set by its last bar Binance returned a
// next bar for: T−4h for the first fetch, T for the second, no 304 between.
func TestBinanceWindowIsSetByLastClosedBar(t *testing.T) {
	stubExternalBases(t)
	now := time.Now().UTC().Truncate(time.Second)
	lastClose := now.Add(-time.Hour)
	atClose := btcSeries(binanceFetchLimit, lastClose) // 1000 closed, nothing forming
	for _, i := range []int{0, binanceFetchLimit - klineLimit} {
		atClose[i].Open, atClose[i].High, atClose[i].Low, atClose[i].Close = 1e10, 1e10, 1e10, 1e10
	}
	later := withForming(atClose[1:], lastClose) // 999 closed + forming

	for _, bp := range binancePaths {
		t.Run(bp.path, func(t *testing.T) {
			ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
			_, srv := newTestAPI(t, ag, true)

			seedBTC4h(ag, atClose, now)
			st1, lm1, body1 := cget(t, srv.URL+bp.path, "")
			seedBTC4h(ag, later, now)
			st2, lm2, body2 := cget(t, srv.URL+bp.path, "")
			if st1 != 200 || st2 != 200 || lm1 == "" || lm2 == "" {
				t.Fatalf("→ %d/%d, Last-Modified %q/%q; want two stamped 200s", st1, st2, lm1, lm2)
			}
			if lm1 == lm2 && string(body1) != string(body2) {
				t.Errorf("false 304: one stamp %q over two bodies", lm1)
			}
			if lm1 != httpStamp(lastClose.Add(-4*time.Hour)) || lm2 != httpStamp(lastClose) {
				t.Errorf("stamps %q → %q; want the last bar with a next bar: %q → %q",
					lm1, lm2, httpStamp(lastClose.Add(-4*time.Hour)), httpStamp(lastClose))
			}
			if st, _, _ := cget(t, srv.URL+bp.path, lm1); st != http.StatusOK {
				t.Errorf("second fetch + the first stamp → %d; want 200 (the window moved)", st)
			}
			if st, _, b := cget(t, srv.URL+bp.path, lm2); st != http.StatusNotModified || len(b) != 0 {
				t.Errorf("second fetch + its own stamp → %d, %d bytes; want 304", st, len(b))
			}
		})
	}
}

// A joiner of an in-flight cache load gets the OWNER's fetch time — the
// closed-bar cut of every card built from that answer runs on one instant.
func TestKlineCacheJoinerGetsOwnersFetchTime(t *testing.T) {
	cache := newKlineCache()
	release := make(chan struct{})
	var hits int32
	load := func() ([]types.OHLCVCandle, error) {
		atomic.AddInt32(&hits, 1)
		<-release
		return []types.OHLCVCandle{{Time: 1, Close: 2}}, nil
	}
	ownerAt := make(chan time.Time, 1)
	go func() { _, at, _ := cache.cached("k", load); ownerAt <- at }()
	for { // wait until the owner's flight is registered
		cache.mu.Lock()
		_, inFlight := cache.items["k"]
		cache.mu.Unlock()
		if inFlight {
			break
		}
		time.Sleep(time.Millisecond)
	}
	time.Sleep(20 * time.Millisecond) // the joiner arrives clearly later
	joinerAt := make(chan time.Time, 1)
	go func() { _, at, _ := cache.cached("k", load); joinerAt <- at }()
	time.Sleep(20 * time.Millisecond) // the joiner is parked on the flight
	close(release)
	o, j := <-ownerAt, <-joinerAt
	if got := atomic.LoadInt32(&hits); got != 1 {
		t.Fatalf("loads = %d, want 1 (the joiner must join the flight)", got)
	}
	if o.IsZero() || !o.Equal(j) {
		t.Errorf("owner fetch time %s, joiner %s — want the same instant", o, j)
	}
}

// ── Yahoo ────────────────────────────────────────────────────────────────────

// lastFridayBefore is the UTC midnight of the latest Friday on or before t.
func lastFridayBefore(t time.Time) time.Time {
	d := time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
	for d.Weekday() != time.Friday {
		d = d.AddDate(0, 0, -1)
	}
	return d
}

// The same Yahoo candles across the weekend: the market-state wording still
// follows the card's clock (on at Friday 21:00 UTC, off at Sunday 21:00 UTC),
// and at every step the card answers 200 with no Last-Modified — even to an
// If-Modified-Since that any stamp would satisfy.
func TestYahooCardsHaveNoValidatorAcrossWeekend(t *testing.T) {
	// A past weekend, so every bar is closed on the real clock too.
	fri := lastFridayBefore(time.Now().UTC().AddDate(0, 0, -7))
	sun := fri.AddDate(0, 0, 2)
	stubExternalBases(t)
	stubYahooWave(t, fri.Add(20*time.Hour), 600)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	clk := &stepClock{}
	ag.now = clk.now
	_, srv := newTestAPI(t, ag, true)
	future := httpStamp(time.Now().Add(24 * time.Hour))

	closed := func(b []byte) bool { return strings.Contains(string(b), "market closed") }
	steps := []struct {
		name   string
		at     time.Time
		closed bool
	}{
		{"Fri 20:30", fri.Add(20*time.Hour + 30*time.Minute), false},
		{"Fri 21:30", fri.Add(21*time.Hour + 30*time.Minute), true},
		{"Sun 20:30", sun.Add(20*time.Hour + 30*time.Minute), true},
		{"Sun 21:30", sun.Add(21*time.Hour + 30*time.Minute), false},
	}
	for _, p := range []string{
		"/agents/trend?asset=eurusd", "/agents/sr?asset=eurusd", "/agents/vol?asset=eurusd",
		"/agents/momentum?asset=eurusd", "/agents/momentum?assets=eurusd",
	} {
		t.Run(p, func(t *testing.T) {
			for _, s := range steps {
				clk.set(s.at)
				for _, ims := range []string{"", future} {
					st, lm, body := cget(t, srv.URL+p, ims)
					if st != http.StatusOK || lm != "" || len(body) == 0 {
						t.Errorf("%s (IMS %q) → %d, Last-Modified %q, %d bytes; want 200, none, the body",
							s.name, ims, st, lm, len(body))
						continue
					}
					if closed(body) != s.closed {
						t.Errorf("%s: market-closed wording = %v, want %v", s.name, closed(body), s.closed)
					}
				}
			}
		})
	}
}

// ── Whale ────────────────────────────────────────────────────────────────────

// whaleWindowFixture: one transfer 1s inside the snapshot's 24h, one exactly
// on the lower boundary (excluded: the window is strictly after), one 1s
// outside, and one stamped 1s AFTER captured_at (excluded: newer than the
// snapshot the card is dated by).
const whaleWindowFixture = `{"captured_at":"2026-08-18T06:00:00Z",
  "flows":[{"asset":"BTC","net_flow_usd_24h":-18400000,"direction":"outflow",
            "inflow_usd_24h":5000000,"outflow_usd_24h":23400000,"tx_count_24h":37,
            "confidence":72,"partial":false,"source":"test"}],
  "transfers":[
    {"chain":"BTC","tx_hash":"in","timestamp":"2026-08-17T06:00:01Z","asset":"BTC","amount_native":12.5,"amount_usd":800000},
    {"chain":"BTC","tx_hash":"edge","timestamp":"2026-08-17T06:00:00Z","asset":"BTC","amount_native":99,"amount_usd":6000000},
    {"chain":"BTC","tx_hash":"out","timestamp":"2026-08-17T05:59:59Z","asset":"BTC","amount_native":77,"amount_usd":5000000},
    {"chain":"BTC","tx_hash":"late","timestamp":"2026-08-18T06:00:01Z","asset":"BTC","amount_native":55,"amount_usd":9000000}]}`

// The whale top-3 window is the snapshot's own 24h up to captured_at, like
// the backend's counters — the request time never changes which transfers
// show. The card still carries no validator: the list itself comes from the
// backend's live table, which can change under the same captured_at.
func TestWhaleWindowAnchoredToSnapshot(t *testing.T) {
	captured := time.Date(2026, 8, 18, 6, 0, 0, 0, time.UTC)
	ag := newStubBackend(t, map[string]string{"/api/v1/whale-flow": whaleWindowFixture})
	clk := &stepClock{}
	ag.now = clk.now
	_, srv := newTestAPI(t, ag, true)
	future := httpStamp(captured.AddDate(1, 0, 0))

	var first []byte
	for _, at := range []time.Time{
		captured.Add(time.Hour), captured.Add(23 * time.Hour), captured.Add(30 * time.Hour), captured.AddDate(0, 1, 0),
	} {
		clk.set(at)
		st, lm, body := cget(t, srv.URL+"/agents/whale", future)
		if st != 200 || lm != "" || len(body) == 0 {
			t.Fatalf("at %s → %d, Last-Modified %q, %d bytes; want 200, none, the body", at, st, lm, len(body))
		}
		s := string(body)
		if !strings.Contains(s, "12.50 BTC") {
			t.Errorf("at %s: the transfer inside captured_at−24h (12.50 BTC) is missing: %s", at, s)
		}
		for _, gone := range []string{"99.00 BTC", "77.00 BTC", "55.00 BTC"} {
			if strings.Contains(s, gone) {
				t.Errorf("at %s: %s is outside (captured_at−24h, captured_at] and must not show: %s", at, gone, s)
			}
		}
		if first == nil {
			first = body
		} else if string(first) != s {
			t.Errorf("at %s the body changed under the same snapshot:\n%s\nvs\n%s", at, first, s)
		}
	}
}

// Without a parseable captured_at the window is the 24h up to the card's
// clock — with the same upper bound: a transfer stamped after the clock
// (a skewed source clock) is not shown either.
func TestWhaleWithoutCapturedAtHasNoValidator(t *testing.T) {
	at := time.Date(2026, 8, 18, 6, 0, 0, 0, time.UTC)
	fixture := `{"captured_at":"","flows":[{"asset":"BTC","net_flow_usd_24h":0,"direction":"neutral",
	    "tx_count_24h":2,"partial":true,"source":"test"}],
	  "transfers":[
	    {"chain":"BTC","tx_hash":"in","timestamp":"2026-08-18T05:00:00Z","asset":"BTC","amount_native":12.5,"amount_usd":800000},
	    {"chain":"BTC","tx_hash":"out","timestamp":"2026-08-17T05:00:00Z","asset":"BTC","amount_native":77,"amount_usd":5000000},
	    {"chain":"BTC","tx_hash":"future","timestamp":"2026-08-18T07:00:00Z","asset":"BTC","amount_native":55,"amount_usd":9000000}]}`
	ag := newStubBackend(t, map[string]string{"/api/v1/whale-flow": fixture})
	clk := &stepClock{t: at}
	ag.now = clk.now
	_, srv := newTestAPI(t, ag, true)

	st, lm, body := cget(t, srv.URL+"/agents/whale", httpStamp(at.Add(24*time.Hour)))
	if st != 200 || lm != "" || len(body) == 0 {
		t.Fatalf("→ %d, Last-Modified %q, %d bytes; want 200, no stamp, the body", st, lm, len(body))
	}
	s := string(body)
	if !strings.Contains(s, "12.50 BTC") || strings.Contains(s, "77.00 BTC") {
		t.Errorf("window must be the 24h up to the request clock: %s", s)
	}
	if strings.Contains(s, "55.00 BTC") {
		t.Errorf("a transfer stamped after the card's clock must not show: %s", s)
	}
}
