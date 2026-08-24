package macro

// source_test.go — offline tests for the provider order (MACRO_SOURCE_ORDER),
// the per-symbol fallback and its attribution. Every case runs against
// httptest servers; nothing here touches the network.
//
// The stooq stubs deliberately reproduce the REAL 2026-08-24 failure bodies —
// the 404 HTML page on the quote endpoint and the JavaScript anti-bot
// challenge on the daily endpoint — so the fallback is exercised against what
// actually broke, not a convenient synthetic error.

import (
	"context"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

// The verbatim shapes stooq started serving on 2026-08-24.
const (
	stooqQuote404HTML = `<meta charset=utf-8><title>Stooq</title><center style=font-family:arial;margin-top:50px>` +
		`<p><a href=/><img src=//static.stooq.com/stooq.svg height=68></a>` +
		`<p style=font-size:x-large>The page you requested does not exist<br>or has been moved`

	stooqAntiBotHTML = `<!DOCTYPE html><html><head><meta charset="utf-8">` +
		`<meta name="robots" content="noindex,nofollow"></head><body>` +
		`<noscript>This site requires JavaScript to verify your browser.</noscript>` +
		`<script nonce="WO4aNtcl7OSuTuwKBuGlqA">(async()=>{})()</script></body></html>`
)

// deadStooq serves the real broken bodies: 404 + HTML on the quote path, 200 +
// anti-bot challenge on the daily path (which is what stooq actually does — the
// daily failure is a 200, so a status check alone would not catch it).
func deadStooq(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/q/d/l/") {
			_, _ = io.WriteString(w, stooqAntiBotHTML)
			return
		}
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, stooqQuote404HTML)
	}))
}

// yahooFixtureServer serves the captured fixtures by ticker; anything else 404s.
// It records which tickers were requested.
func yahooFixtureServer(t *testing.T, byTicker map[string]string) (*httptest.Server, *[]string) {
	t.Helper()
	var hits []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ticker := strings.TrimPrefix(r.URL.Path, "/")
		hits = append(hits, ticker)
		fixture, ok := byTicker[ticker]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			_, _ = io.WriteString(w, `{"chart":{"result":null,"error":{"code":"Not Found","description":"No data found"}}}`)
			return
		}
		_, _ = w.Write(readFixture(t, fixture))
	}))
	return srv, &hits
}

// TestParseSourceOrder covers MACRO_SOURCE_ORDER parsing, including the
// operator-error paths: a typo must never leave the worker with zero providers.
func TestParseSourceOrder(t *testing.T) {
	cases := []struct {
		name        string
		in          string
		wantOrder   []string
		wantUnknown []string
	}{
		{"empty → default", "", []string{SourceStooq, SourceYahoo}, nil},
		{"default spelled out", "stooq,yahoo", []string{SourceStooq, SourceYahoo}, nil},
		{"reversed", "yahoo,stooq", []string{SourceYahoo, SourceStooq}, nil},
		{"pin one", "yahoo", []string{SourceYahoo}, nil},
		{"whitespace and case", " Stooq ,\tYAHOO ", []string{SourceStooq, SourceYahoo}, nil},
		{"duplicates collapse, first wins", "yahoo,stooq,yahoo", []string{SourceYahoo, SourceStooq}, nil},
		{"empty entries skipped", "stooq,,yahoo,", []string{SourceStooq, SourceYahoo}, nil},
		{"unknown reported, known kept", "bloomberg,yahoo", []string{SourceYahoo}, []string{"bloomberg"}},
		{"all unknown → default, still reported", "bloomberg,reuters",
			[]string{SourceStooq, SourceYahoo}, []string{"bloomberg", "reuters"}},
		{"mixed is NOT a provider", "mixed", []string{SourceStooq, SourceYahoo}, []string{"mixed"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			order, unknown := ParseSourceOrder(tc.in)
			if !reflect.DeepEqual(order, tc.wantOrder) {
				t.Errorf("order = %v, want %v", order, tc.wantOrder)
			}
			if !reflect.DeepEqual(unknown, tc.wantUnknown) {
				t.Errorf("unknown = %v, want %v", unknown, tc.wantUnknown)
			}
		})
	}
}

// TestParseSourceOrder_DefaultIsNotAliased: the fallback must hand out a COPY,
// or a caller mutating its order would corrupt the package default.
func TestParseSourceOrder_DefaultIsNotAliased(t *testing.T) {
	order, _ := ParseSourceOrder("")
	order[0] = "tampered"
	if defaultSourceOrder[0] != SourceStooq {
		t.Fatalf("defaultSourceOrder was mutated through a returned slice: %v", defaultSourceOrder)
	}
}

func TestCombineSources(t *testing.T) {
	cases := []struct{ a, b, want string }{
		{SourceStooq, SourceStooq, SourceStooq},
		{SourceYahoo, SourceYahoo, SourceYahoo},
		{SourceStooq, SourceYahoo, SourceMixed},
		{SourceYahoo, SourceStooq, SourceMixed},
		{"", SourceYahoo, ""},
		{SourceYahoo, "", ""},
		{"", "", ""},
	}
	for _, tc := range cases {
		if got := CombineSources(tc.a, tc.b); got != tc.want {
			t.Errorf("CombineSources(%q,%q) = %q, want %q", tc.a, tc.b, got, tc.want)
		}
	}
}

// TestWorkerSourceOrder_EnvHonored: MACRO_SOURCE_ORDER drives the resolved
// provider list when the Worker does not pin one.
func TestWorkerSourceOrder_EnvHonored(t *testing.T) {
	cases := []struct {
		env  string
		want []string
	}{
		{"", []string{SourceStooq, SourceYahoo}},
		{"yahoo", []string{SourceYahoo}},
		{"yahoo,stooq", []string{SourceYahoo, SourceStooq}},
		{"stooq", []string{SourceStooq}},
		{"nonsense", []string{SourceStooq, SourceYahoo}}, // typo → default, never zero providers
	}
	for _, tc := range cases {
		t.Run("MACRO_SOURCE_ORDER="+tc.env, func(t *testing.T) {
			t.Setenv(SourceOrderEnv, tc.env)
			wk := &Worker{Store: NewStore(), Logger: log.New(io.Discard, "", 0)}
			if got := sourceNames(wk.sources()); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("source order = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestWorkerSourceOrder_FieldBeatsEnv: an explicitly pinned Worker.SourceOrder
// wins over the environment (this is what keeps the rest of the suite offline).
func TestWorkerSourceOrder_FieldBeatsEnv(t *testing.T) {
	t.Setenv(SourceOrderEnv, "yahoo,stooq")
	wk := &Worker{Store: NewStore(), Logger: log.New(io.Discard, "", 0),
		SourceOrder: []string{SourceStooq}}
	if got := sourceNames(wk.sources()); !reflect.DeepEqual(got, []string{SourceStooq}) {
		t.Errorf("source order = %v, want [stooq] (the field must beat the env)", got)
	}
}

// TestWorkerFallback_StooqGarbage_YahooUsed is the headline case: stooq serves
// the real broken bodies, Yahoo serves real fixtures, and the lamp ends up with
// a Yahoo value that SAYS it is a Yahoo value.
//
// Only DX-Y.NYB and BTC-USD are stubbed on the Yahoo side, so the same cycle
// also proves the honest half: symbols nobody can answer for stay not-ok.
func TestWorkerFallback_StooqGarbage_YahooUsed(t *testing.T) {
	stooq := deadStooq(t)
	defer stooq.Close()
	yahoo, hits := yahooFixtureServer(t, map[string]string{
		"DX-Y.NYB": "yahoo_dxy_1d_3mo.json",
		"BTC-USD":  "yahoo_btcusd_1d_3mo.json",
	})
	defer yahoo.Close()

	store := NewStore()
	wk := &Worker{
		Store:          store,
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		YahooBase:      yahoo.URL + "/",
		SourceOrder:    []string{SourceStooq, SourceYahoo},
		now:            func() time.Time { return time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC) },
	}
	wk.refresh(context.Background())

	latest := store.Latest()

	// DXY: stooq had nothing, Yahoo did → an OK quote ATTRIBUTED to yahoo.
	dxy := latest[SymDXY]
	if !dxy.OK {
		t.Fatalf("DXY quote OK = false, want true (Yahoo fallback should have filled it)")
	}
	if dxy.Source != SourceYahoo {
		t.Errorf("DXY Source = %q, want %q — a fallback value must say where it came from", dxy.Source, SourceYahoo)
	}
	if !closeEnough(dxy.Price, 98.97) {
		t.Errorf("DXY Price = %v, want ~98.97 (fixture's last close)", dxy.Price)
	}
	if !closeEnough(dxy.Open, 98.856) {
		t.Errorf("DXY Open = %v, want ~98.856 (session baseline)", dxy.Open)
	}

	// BTC likewise (not a lamp, but it feeds the correlations).
	if btc := latest[SymBTC]; !btc.OK || btc.Source != SourceYahoo {
		t.Errorf("BTC quote = %+v, want OK with source yahoo", btc)
	}

	// The unstubbed symbols: BOTH sources failed → honest not-ok, no source.
	for _, sym := range []string{SymSPX, SymVIX, SymGold, SymRates} {
		q := latest[sym]
		if q.OK {
			t.Errorf("%s OK = true, want false (neither source could answer)", sym)
		}
		if q.Source != "" {
			t.Errorf("%s Source = %q, want \"\" — a lamp with no value has no provider", sym, q.Source)
		}
		if q.Price != 0 {
			t.Errorf("%s Price = %v, want 0 (never a fabricated value)", sym, q.Price)
		}
	}

	// Yahoo was consulted for every symbol (stooq answered for none of them):
	// one quote request each, plus the daily-window pass.
	seen := map[string]bool{}
	for _, h := range *hits {
		seen[h] = true
	}
	for _, sym := range allSymbols {
		ticker, ok := YahooTicker(sym)
		if !ok {
			t.Fatalf("%s has no Yahoo ticker", sym)
		}
		if !seen[ticker] {
			t.Errorf("Yahoo was never asked for %s (%s); hits = %v", sym, ticker, *hits)
		}
	}
}

// TestWorkerFallback_StooqWins_YahooUntouched: when stooq DOES answer, the
// fallback must not fire at all — no wasted request, and the value is
// attributed to stooq.
func TestWorkerFallback_StooqWins_YahooUntouched(t *testing.T) {
	stooq := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/q/d/l/") {
			_, _ = io.WriteString(w, "Date,Open,High,Low,Close,Volume\n2026-08-21,1,2,3,7580.1,9\n")
			return
		}
		sym := strings.ToUpper(r.URL.Query().Get("s"))
		_, _ = io.WriteString(w, "Symbol,Date,Time,Open,High,Low,Close,Volume\n"+
			sym+",2026-08-21,21:00:00,7579.3,7599.4,7563.6,7580.1,5498484255\n")
	}))
	defer stooq.Close()

	yahooCalled := false
	yahoo := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		yahooCalled = true
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer yahoo.Close()

	store := NewStore()
	wk := &Worker{
		Store:          store,
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		YahooBase:      yahoo.URL + "/",
		SourceOrder:    []string{SourceStooq, SourceYahoo},
		now:            func() time.Time { return time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC) },
	}
	wk.refresh(context.Background())

	if yahooCalled {
		t.Errorf("Yahoo was called even though stooq answered — the order must short-circuit")
	}
	for _, sym := range allSymbols {
		q := store.Latest()[sym]
		if !q.OK || q.Source != SourceStooq {
			t.Errorf("%s = %+v, want OK with source %q", sym, q, SourceStooq)
		}
	}
	if got := store.DailySource(SymSPX); got != SourceStooq {
		t.Errorf("DailySource(^spx) = %q, want %q", got, SourceStooq)
	}
}

// TestWorkerFallback_BothFail_LampStaysNotOK: with every provider dead the
// behaviour is EXACTLY today's — a not-ok quote, no value, no source, and a
// regime of "unknown" once the lamps are built.
func TestWorkerFallback_BothFail_LampStaysNotOK(t *testing.T) {
	stooq := deadStooq(t)
	defer stooq.Close()
	yahoo := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, "Too Many Requests")
	}))
	defer yahoo.Close()

	store := NewStore()
	wk := &Worker{
		Store:          store,
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		YahooBase:      yahoo.URL + "/",
		SourceOrder:    []string{SourceStooq, SourceYahoo},
		now:            func() time.Time { return time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC) },
	}
	wk.refresh(context.Background())

	latest := store.Latest()
	// Build the lamps with their REAL keys — a lamp keyed by symbol would carry
	// weight 0 and be skipped by Composite for the wrong reason, making the
	// assertions below pass even if the lamps had values.
	symToKey := []struct{ sym, key string }{
		{SymDXY, KeyDXY}, {SymRates, KeyRates}, {SymVIX, KeyVIX},
		{SymSPX, KeySPX}, {SymGold, KeyGold},
	}
	lamps := make([]Lamp, 0, len(symToKey))
	for _, m := range symToKey {
		q := latest[m.sym]
		if q.OK {
			t.Errorf("%s OK = true, want false (both sources dead)", m.sym)
		}
		if q.Source != "" {
			t.Errorf("%s Source = %q, want \"\"", m.sym, q.Source)
		}
		if lampWeight(m.key) == 0 {
			t.Fatalf("lamp key %q carries no composite weight — the test's key "+
				"mapping is wrong and every assertion below would be vacuous", m.key)
		}
		lamps = append(lamps, Lamp{Key: m.key}) // real key, no value → not-ok lamp
	}
	// The honesty contract downstream is untouched: no live lamps → no
	// composite, regime "unknown", and no diagnosis sentence.
	if c := Composite(lamps); c != nil {
		t.Errorf("Composite = %v, want nil with zero live lamps", *c)
	}
	if r := ClassifyRegime(nil, lamps); r != RegimeUnknown {
		t.Errorf("regime = %q, want %q", r, RegimeUnknown)
	}
	if d := BuildDiagnosis(RegimeUnknown, lamps); d != "" {
		t.Errorf("diagnosis = %q, want \"\" (no data → no sentence)", d)
	}
	if store.DailySource(SymSPX) != "" {
		t.Errorf("DailySource = %q, want \"\" with no stored history", store.DailySource(SymSPX))
	}
}

// TestWorkerFallback_BothFail_KeepsLastKnownDate: stooq's N/D rows carry the
// last session's DATE. That date is a real fact ("last time this symbol had
// data") and must survive the fallback — when Yahoo also comes up empty the
// stored quote is still not-ok but keeps as_of, exactly as before the fallback
// existed. It must NOT gain a source: there is no value to attribute.
func TestWorkerFallback_BothFail_KeepsLastKnownDate(t *testing.T) {
	stooq := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/q/d/l/") {
			_, _ = io.WriteString(w, stooqAntiBotHTML)
			return
		}
		sym := strings.ToUpper(r.URL.Query().Get("s"))
		// N/D value but a real date — stooq's weekend/holiday shape.
		_, _ = io.WriteString(w, "Symbol,Date,Time,Open,High,Low,Close,Volume\n"+
			sym+",2026-08-21,21:00:00,N/D,N/D,N/D,N/D,N/D\n")
	}))
	defer stooq.Close()
	yahoo := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer yahoo.Close()

	store := NewStore()
	wk := &Worker{
		Store:          store,
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		YahooBase:      yahoo.URL + "/",
		SourceOrder:    []string{SourceStooq, SourceYahoo},
		now:            func() time.Time { return time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC) },
	}
	wk.refresh(context.Background())

	q := store.Latest()[SymSPX]
	if q.OK {
		t.Errorf("OK = true, want false")
	}
	want := time.Date(2026, 8, 21, 21, 0, 0, 0, time.UTC)
	if !q.AsOf.Equal(want) {
		t.Errorf("AsOf = %v, want %v (stooq's N/D date must survive the fallback)", q.AsOf, want)
	}
	if q.Source != "" {
		t.Errorf("Source = %q, want \"\" (a dated N/D quote has no value to attribute)", q.Source)
	}
}

// TestWorkerDaily_YahooFeedsCorrelations: the daily history fetched from Yahoo
// lands in the store, is attributed to Yahoo, and yields a correlation with at
// least MinDailyCorrPoints overlapping daily closes.
func TestWorkerDaily_YahooFeedsCorrelations(t *testing.T) {
	stooq := deadStooq(t)
	defer stooq.Close()
	yahoo, _ := yahooFixtureServer(t, map[string]string{
		"DX-Y.NYB": "yahoo_dxy_1d_3mo.json",
		"BTC-USD":  "yahoo_btcusd_1d_3mo.json",
	})
	defer yahoo.Close()

	store := NewStore()
	wk := &Worker{
		Store:          store,
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		YahooBase:      yahoo.URL + "/",
		SourceOrder:    []string{SourceStooq, SourceYahoo},
		// Pinned clock: the fixtures were captured on 2026-08-24 and the
		// dailyMaxAgeDays guard is relative to now — without this the test
		// would silently start dropping rows as wall-clock time moved on.
		now: func() time.Time { return time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC) },
	}
	wk.refresh(context.Background())

	if n := store.DailyCount(SymBTC); n != dailyKeep {
		t.Errorf("DailyCount(btc) = %d, want %d (capped at the keep window)", n, dailyKeep)
	}
	if n := store.DailyCount(SymDXY); n != dailyKeep {
		t.Errorf("DailyCount(dxy) = %d, want %d", n, dailyKeep)
	}
	for _, sym := range []string{SymBTC, SymDXY} {
		if got := store.DailySource(sym); got != SourceYahoo {
			t.Errorf("DailySource(%s) = %q, want %q", sym, got, SourceYahoo)
		}
	}

	coef, points := store.DailyCorrelation(SymBTC, SymDXY)
	if points < MinDailyCorrPoints {
		t.Fatalf("overlapping daily closes = %d, want ≥ %d — the Yahoo daily window "+
			"must be able to feed a correlation", points, MinDailyCorrPoints)
	}
	if coef == nil {
		t.Fatalf("coef = nil with %d points, want a value", points)
	}
	if *coef < -1 || *coef > 1 {
		t.Errorf("coef = %v, outside [-1,1]", *coef)
	}
	t.Logf("btc_dxy over Yahoo daily closes: coef=%.4f points=%d", *coef, points)

	// A symbol nobody served keeps an empty history → the correlation is
	// honestly absent rather than computed off one leg.
	if c, p := store.DailyCorrelation(SymBTC, SymSPX); c != nil || p != 0 {
		t.Errorf("btc_spx = (%v, %d), want (nil, 0) — no SPX history was stored", c, p)
	}
}

// TestWorkerDaily_MixedSourcesAreAttributedMixed: two legs from different
// providers must surface as "mixed", never silently as one of them.
func TestWorkerDaily_MixedSourcesAreAttributedMixed(t *testing.T) {
	store := NewStore()
	store.SetDailyCloses(SymBTC, mkDaily(minDailyCorrPoints, func(i int) float64 { return float64(100 + i) }), SourceYahoo)
	store.SetDailyCloses(SymSPX, mkDaily(minDailyCorrPoints, func(i int) float64 { return float64(200 + 2*i) }), SourceStooq)

	if got := CombineSources(store.DailySource(SymBTC), store.DailySource(SymSPX)); got != SourceMixed {
		t.Errorf("combined source = %q, want %q", got, SourceMixed)
	}
	// And the same-provider case still names the provider.
	store.SetDailyCloses(SymSPX, mkDaily(minDailyCorrPoints, func(i int) float64 { return float64(200 + 2*i) }), SourceYahoo)
	if got := CombineSources(store.DailySource(SymBTC), store.DailySource(SymSPX)); got != SourceYahoo {
		t.Errorf("combined source = %q, want %q", got, SourceYahoo)
	}
}

// TestWorkerDaily_AgeGuardAppliesToEverySource: the dailyMaxAgeDays guard lives
// in the Worker, not in a source, so a provider that ignores the requested
// window cannot smuggle an ancient series through as "the last 30 days".
func TestWorkerDaily_AgeGuardAppliesToEverySource(t *testing.T) {
	stooq := deadStooq(t)
	defer stooq.Close()
	yahoo, _ := yahooFixtureServer(t, map[string]string{
		"DX-Y.NYB": "yahoo_dxy_1d_3mo.json",
		"BTC-USD":  "yahoo_btcusd_1d_3mo.json",
	})
	defer yahoo.Close()

	store := NewStore()
	wk := &Worker{
		Store:          store,
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		YahooBase:      yahoo.URL + "/",
		SourceOrder:    []string{SourceStooq, SourceYahoo},
		// A year after the fixtures were captured: every row is behind the
		// 90-day guard.
		now: func() time.Time { return time.Date(2027, 8, 24, 12, 0, 0, 0, time.UTC) },
	}
	wk.refresh(context.Background())

	for _, sym := range []string{SymBTC, SymDXY} {
		if n := store.DailyCount(sym); n != 0 {
			t.Errorf("DailyCount(%s) = %d, want 0 (rows older than %d days are dropped)",
				sym, n, dailyMaxAgeDays)
		}
		if got := store.DailySource(sym); got != "" {
			t.Errorf("DailySource(%s) = %q, want \"\" (nothing was stored)", sym, got)
		}
	}
}

// TestWorkerFallback_StaleQuoteRejected pins the freshness guard, which exists
// because Yahoo has no "N/D" sentinel: its chart endpoint just returns the last
// bar it holds, so a frozen feed would otherwise keep voting in the composite
// with a weeks-old close marked ok:true.
//
// The fixture's newest bar is 2026-08-24; the clock is pinned 37 days later.
// Expected: the value is REJECTED (not-ok, no source, no price) but its DATE
// survives, so the lamp still shows how stale the market went.
func TestWorkerFallback_StaleQuoteRejected(t *testing.T) {
	stooq := deadStooq(t)
	defer stooq.Close()
	yahoo, _ := yahooFixtureServer(t, map[string]string{
		"DX-Y.NYB": "yahoo_dxy_1d_3mo.json",
	})
	defer yahoo.Close()

	store := NewStore()
	wk := &Worker{
		Store:          store,
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		YahooBase:      yahoo.URL + "/",
		SourceOrder:    []string{SourceStooq, SourceYahoo},
		now:            func() time.Time { return time.Date(2026, 9, 30, 12, 0, 0, 0, time.UTC) },
	}
	wk.refresh(context.Background())

	q := store.Latest()[SymDXY]
	if q.OK {
		t.Errorf("OK = true, want false — a bar %v old must not vote in the composite",
			time.Date(2026, 9, 30, 12, 0, 0, 0, time.UTC).Sub(time.Date(2026, 8, 24, 4, 0, 0, 0, time.UTC)).Round(time.Hour))
	}
	if q.Price != 0 {
		t.Errorf("Price = %v, want 0 (a rejected value must not be stored)", q.Price)
	}
	if q.Source != "" {
		t.Errorf("Source = %q, want \"\" (no value → nothing to attribute)", q.Source)
	}
	want := time.Date(2026, 8, 24, 4, 0, 0, 0, time.UTC)
	if !q.AsOf.Equal(want) {
		t.Errorf("AsOf = %v, want %v — the last known date must survive so the "+
			"lamp can show HOW stale the market is", q.AsOf, want)
	}
}

// TestWorkerFallback_FreshQuoteAccepted is the other half of the guard: the
// same fixture read at a plausible wall-clock is accepted normally. Without
// this, a guard that rejected everything would still pass the test above.
func TestWorkerFallback_FreshQuoteAccepted(t *testing.T) {
	stooq := deadStooq(t)
	defer stooq.Close()
	yahoo, _ := yahooFixtureServer(t, map[string]string{
		"DX-Y.NYB": "yahoo_dxy_1d_3mo.json",
	})
	defer yahoo.Close()

	store := NewStore()
	wk := &Worker{
		Store:          store,
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		YahooBase:      yahoo.URL + "/",
		SourceOrder:    []string{SourceStooq, SourceYahoo},
		// A Monday morning read of a bar stamped that same day.
		now: func() time.Time { return time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC) },
	}
	wk.refresh(context.Background())

	q := store.Latest()[SymDXY]
	if !q.OK || q.Source != SourceYahoo {
		t.Fatalf("q = %+v, want OK with source yahoo", q)
	}
	if !closeEnough(q.Price, 98.97) {
		t.Errorf("Price = %v, want ~98.97", q.Price)
	}
}

// TestWorkerFallback_WeekendQuoteAccepted: the guard must never fire on a
// normal market closure. Friday's bar read on Monday morning is ~3 days old and
// is the ORDINARY weekend case for the tradfin lamps — if quoteMaxAge ever
// shrank below it, every lamp would go dark every Monday.
func TestWorkerFallback_WeekendQuoteAccepted(t *testing.T) {
	stooq := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/q/d/l/") {
			_, _ = io.WriteString(w, stooqAntiBotHTML)
			return
		}
		sym := strings.ToUpper(r.URL.Query().Get("s"))
		// Friday 2026-08-21 21:00 close.
		_, _ = io.WriteString(w, "Symbol,Date,Time,Open,High,Low,Close,Volume\n"+
			sym+",2026-08-21,21:00:00,7579.3,7599.4,7563.6,7580.1,5498484255\n")
	}))
	defer stooq.Close()

	store := NewStore()
	wk := &Worker{
		Store:          store,
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		SourceOrder:    []string{SourceStooq},
		// Monday morning — 2 days 12h after Friday's close.
		now: func() time.Time { return time.Date(2026, 8, 24, 9, 0, 0, 0, time.UTC) },
	}
	wk.refresh(context.Background())

	q := store.Latest()[SymSPX]
	if !q.OK {
		t.Fatalf("OK = false — the freshness guard fired on an ordinary weekend "+
			"(quoteMaxAge = %s must exceed a Fri→Mon gap)", quoteMaxAge)
	}
	if q.Price != 7580.1 {
		t.Errorf("Price = %v, want 7580.1", q.Price)
	}
}

// TestQuoteMaxAge_ExceedsLongestMarketClosure documents the constant's job as
// an assertion: it must clear the longest real US market break (Christmas/New
// Year, ~5 days) with margin, or lamps would blank on ordinary holidays.
func TestQuoteMaxAge_ExceedsLongestMarketClosure(t *testing.T) {
	const longestClosure = 5 * 24 * time.Hour
	if quoteMaxAge <= longestClosure {
		t.Errorf("quoteMaxAge = %s, must exceed the longest normal market closure (%s)",
			quoteMaxAge, longestClosure)
	}
}

// TestWorkerDaily_StampRequiresBTC pins the once-a-day gate: the refresh stamp
// advances ONLY when BTC stored.
//
// Every correlation is BTC↔X, so a cycle that stored SPX/gold/DXY but missed
// BTC produced no usable correlation window at all. BTC is also fetched LAST in
// allSymbols, so it is precisely the leg a budget overrun drops first. Under the
// old "any symbol stored" rule that cycle would have stamped success and parked
// all three correlations on an empty BTC leg for a full 24 hours.
func TestWorkerDaily_StampRequiresBTC(t *testing.T) {
	stooq := deadStooq(t)
	defer stooq.Close()
	// Yahoo serves DXY but NOT BTC — a partial daily cycle.
	yahoo, _ := yahooFixtureServer(t, map[string]string{
		"DX-Y.NYB": "yahoo_dxy_1d_3mo.json",
	})
	defer yahoo.Close()

	clock := time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC)
	wk := &Worker{
		Store:          NewStore(),
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		YahooBase:      yahoo.URL + "/",
		SourceOrder:    []string{SourceStooq, SourceYahoo},
		now:            func() time.Time { return clock },
	}
	wk.refreshDailyIfDue(context.Background())

	// DXY did store — so this is genuinely a PARTIAL success, not a total one.
	if n := wk.Store.DailyCount(SymDXY); n == 0 {
		t.Fatalf("DXY stored 0 rows — the test is not exercising a partial cycle")
	}
	if n := wk.Store.DailyCount(SymBTC); n != 0 {
		t.Fatalf("BTC stored %d rows, want 0 (not served)", n)
	}
	if !wk.lastDaily.IsZero() {
		t.Errorf("lastDaily = %v, want zero — the stamp must NOT advance without "+
			"BTC, or all three correlations sit on an empty leg for 24h", wk.lastDaily)
	}

	// Next tick must therefore RETRY rather than skip for a day.
	yahoo2, hits2 := yahooFixtureServer(t, map[string]string{
		"DX-Y.NYB": "yahoo_dxy_1d_3mo.json",
		"BTC-USD":  "yahoo_btcusd_1d_3mo.json",
	})
	defer yahoo2.Close()
	wk.YahooBase = yahoo2.URL + "/"
	wk.srcs, wk.srcsOnce = nil, sync.Once{} // rebuild sources against the new server
	wk.refreshDailyIfDue(context.Background())

	if len(*hits2) == 0 {
		t.Fatalf("no retry happened — the daily refresh skipped a due cycle")
	}
	if wk.Store.DailyCount(SymBTC) == 0 {
		t.Errorf("BTC still empty after the retry")
	}
	if wk.lastDaily.IsZero() {
		t.Errorf("lastDaily still zero after BTC stored — the stamp must advance now")
	}
}

// TestDailyKeepCoversTradfinSessions pins the sizing rationale for dailyKeep as
// an executable assertion, so nobody can shrink it back to 30 without this
// failing.
//
// The two legs count rows at different rates: BTC trades 7 days a week, the
// tradfin symbols 5. dailyKeep is applied per symbol in ROWS but the join
// happens on calendar DATES, so BTC's window must span enough CALENDAR days to
// cover the tradfin legs' sessions. At 30 the measured overlap was exactly 20
// against a minimum of 20 — zero margin.
func TestDailyKeepCoversTradfinSessions(t *testing.T) {
	// BTC's window in calendar days == dailyKeep (one bar per day).
	btcCalendarDays := dailyKeep
	// A 5-day-week symbol needs 7/5 calendar days per session.
	tradfinSessionsCovered := btcCalendarDays * 5 / 7
	if tradfinSessionsCovered < minDailyCorrPoints {
		t.Fatalf("dailyKeep=%d spans %d calendar days, covering only ~%d tradfin "+
			"sessions — below the %d-point minimum, so correlations would be dark",
			dailyKeep, btcCalendarDays, tradfinSessionsCovered, minDailyCorrPoints)
	}
	const wantMargin = 5
	if tradfinSessionsCovered < minDailyCorrPoints+wantMargin {
		t.Errorf("dailyKeep=%d covers ~%d tradfin sessions, only %d above the %d "+
			"minimum — a single market holiday could blank a correlation. Want ≥%d "+
			"points of margin.", dailyKeep, tradfinSessionsCovered,
			tradfinSessionsCovered-minDailyCorrPoints, minDailyCorrPoints, wantMargin)
	}
}
