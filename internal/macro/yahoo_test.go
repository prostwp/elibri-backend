package macro

// yahoo_test.go — offline tests for the Yahoo chart parser, the symbol map and
// the yahooSource. NOTHING here touches the network: the real-payload cases run
// off testdata/*.json (verbatim upstream bytes captured 2026-08-24) and the
// source cases run against httptest servers.

import (
	"context"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// readFixture loads a captured upstream payload from testdata.
func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	return b
}

// testWorker returns a silent Worker used only for its httpGet method (the
// shared httpGetter every source is constructed with).
func testWorker() *Worker { return &Worker{Logger: log.New(io.Discard, "", 0)} }

// TestParseYahooChart_RealFixture is the headline parser test: the VERBATIM
// DX-Y.NYB 1d/3mo response captured from Yahoo on 2026-08-24. It naturally
// contains both shapes the task calls out:
//
//   - NULL-PADDED BARS — 13 of the 76 slots have null OHLC (every Sunday). They
//     must be dropped, never turned into a bar.
//   - HOLIDAY GAPS — 2026-06-18 → 2026-06-22 (4 days, Juneteenth) and
//     2026-07-02 → 2026-07-06 (4 days, Independence Day) are missing entirely.
//     They must simply be absent, never interpolated or forward-filled.
func TestParseYahooChart_RealFixture(t *testing.T) {
	bars, err := ParseYahooChart(readFixture(t, "yahoo_dxy_1d_3mo.json"))
	if err != nil {
		t.Fatalf("ParseYahooChart err = %v", err)
	}

	// 76 slots, 13 null-padded → 63 usable bars.
	if len(bars) != 63 {
		t.Fatalf("len(bars) = %d, want 63 (76 slots − 13 null-padded)", len(bars))
	}

	// Null-padded Sundays must be absent.
	for _, sunday := range []string{"2026-05-31", "2026-06-07", "2026-08-23"} {
		if idx := indexOfDate(bars, sunday); idx >= 0 {
			t.Errorf("null-padded slot %s survived as bar %d (%+v)", sunday, idx, bars[idx])
		}
	}

	// Holiday gaps must stay gaps — the days simply do not exist.
	for _, holiday := range []string{"2026-06-19", "2026-07-03"} {
		if idx := indexOfDate(bars, holiday); idx >= 0 {
			t.Errorf("holiday %s was invented as bar %d (%+v)", holiday, idx, bars[idx])
		}
	}
	// …and the bars either side of a gap are both present and adjacent.
	before, after := indexOfDate(bars, "2026-06-18"), indexOfDate(bars, "2026-06-22")
	if before < 0 || after < 0 {
		t.Fatalf("gap edges missing: 2026-06-18 idx=%d, 2026-06-22 idx=%d", before, after)
	}
	if after != before+1 {
		t.Errorf("gap edges not adjacent: 2026-06-18 idx=%d, 2026-06-22 idx=%d (something was inserted between)", before, after)
	}

	// Ascending order is an invariant the quote path depends on (last = newest).
	for i := 1; i < len(bars); i++ {
		if !bars[i-1].Start.Before(bars[i].Start) {
			t.Fatalf("bars not strictly ascending at %d: %v then %v", i, bars[i-1].Start, bars[i].Start)
		}
	}

	// Last bar: the in-progress Monday session (open 98.856, close 98.97).
	last := bars[len(bars)-1]
	if last.Date != "2026-08-24" {
		t.Errorf("last bar Date = %q, want 2026-08-24", last.Date)
	}
	if !closeEnough(last.Close, 98.97) {
		t.Errorf("last bar Close = %v, want ~98.97", last.Close)
	}
	if !closeEnough(last.Open, 98.856) {
		t.Errorf("last bar Open = %v, want ~98.856", last.Open)
	}
	// ts 1787544000 = 2026-08-24T04:00:00Z (00:00 New York).
	wantStart := time.Date(2026, 8, 24, 4, 0, 0, 0, time.UTC)
	if !last.Start.Equal(wantStart) {
		t.Errorf("last bar Start = %v, want %v", last.Start, wantStart)
	}

	// The previous usable bar is Friday — the Sunday between them was padded.
	prev := bars[len(bars)-2]
	if prev.Date != "2026-08-21" {
		t.Errorf("previous bar Date = %q, want 2026-08-21 (Sunday 08-23 is null-padded)", prev.Date)
	}
	if !closeEnough(prev.Close, 98.8) {
		t.Errorf("previous bar Close = %v, want ~98.80", prev.Close)
	}
}

// TestParseYahooChart_DateIsExchangeLocal pins the session-date rule: the date
// comes from converting each bar's OWN instant into the exchange timezone.
//
// This case uses a zone where the local date and the raw UTC date DISAGREE
// (+09:00, a bar starting 08:00 local = 23:00 UTC the day before) so the rule
// cannot be satisfied by accident.
func TestParseYahooChart_DateIsExchangeLocal(t *testing.T) {
	// 2026-08-24T08:00:00+09:00 == 2026-08-23T23:00:00Z
	// (1787529600 is 2026-08-24T00:00:00Z in the BTC-USD fixture; minus 1h.)
	const ts = 1787526000
	body := `{"chart":{"result":[{"meta":{"symbol":"TEST","exchangeTimezoneName":"Asia/Tokyo","gmtoffset":32400},
		"timestamp":[` + strconv.Itoa(ts) + `],
		"indicators":{"quote":[{"open":[10],"close":[11]}]}}],"error":null}}`

	bars, err := ParseYahooChart([]byte(body))
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(bars) != 1 {
		t.Fatalf("len(bars) = %d, want 1", len(bars))
	}
	if got := bars[0].Start.UTC().Format(time.RFC3339); got != "2026-08-23T23:00:00Z" {
		t.Errorf("Start = %s, want 2026-08-23T23:00:00Z (the raw instant)", got)
	}
	if bars[0].Date != "2026-08-24" {
		t.Errorf("Date = %q, want 2026-08-24 (exchange-local session date); "+
			"raw UTC would wrongly say 2026-08-23", bars[0].Date)
	}
}

// TestParseYahooChart_DateSurvivesDST is the regression test for the bug this
// parser originally shipped with: dating bars by the payload's single
// meta.gmtoffset scalar.
//
// The fixture is a VERBATIM 1-year DX-Y.NYB response. Yahoo shifts the bar
// instants with daylight saving (166 bars at 04:00Z during EDT, 83 at 05:00Z
// during EST) while meta.gmtoffset reports only the offset AT REQUEST TIME
// (-14400, EDT). Applying that one scalar to every bar mis-dates the whole
// other regime — measured at 166 of 251 bars wrong for a request made during
// EST. Because the store keeps just the last 30 rows, that would silently slide
// the entire correlation window one day off BTC's UTC days for weeks.
//
// The assertion is against per-bar timezone truth computed independently in
// this test, so it fails if the parser ever reverts to a single offset.
func TestParseYahooChart_DateSurvivesDST(t *testing.T) {
	bars, err := ParseYahooChart(readFixture(t, "yahoo_dxy_1d_1y_dst.json"))
	if err != nil {
		t.Fatalf("ParseYahooChart err = %v", err)
	}
	if len(bars) != 251 {
		t.Fatalf("len(bars) = %d, want 251", len(bars))
	}

	ny, err := time.LoadLocation("America/New_York")
	if err != nil {
		t.Skipf("no tzdata on this host: %v", err)
	}

	// Independent truth: convert each bar's own instant in the exchange zone.
	// Also count how many bars the OLD scalar-offset rule would have got wrong,
	// so a regression reports the blast radius rather than a bare mismatch.
	scalarWrongEST := 0
	for _, b := range bars {
		want := b.Start.In(ny).Format("2006-01-02")
		if b.Date != want {
			t.Errorf("bar %s: Date = %q, want %q (per-bar timezone truth)",
				b.Start.Format(time.RFC3339), b.Date, want)
		}
		// -18000 is what meta.gmtoffset reports for a request made during EST.
		if b.Start.Add(-5*time.Hour).UTC().Format("2006-01-02") != want {
			scalarWrongEST++
		}
	}
	if scalarWrongEST == 0 {
		t.Errorf("the EST scalar offset agreed with per-bar truth on every bar — "+
			"the fixture no longer spans a DST transition and this regression "+
			"test has stopped testing anything (bars=%d)", len(bars))
	}
	t.Logf("per-bar timezone dating is correct on %d/%d bars; the old "+
		"single-gmtoffset rule would mis-date %d of them during EST",
		len(bars), len(bars), scalarWrongEST)

	// Both DST regimes really are present in the fixture.
	var edt, est int
	for _, b := range bars {
		switch b.Start.UTC().Format("15:04") {
		case "04:00":
			edt++
		case "05:00":
			est++
		}
	}
	if edt == 0 || est == 0 {
		t.Errorf("fixture spans one regime only (04:00Z=%d, 05:00Z=%d) — it must "+
			"cover both to exercise the transition", edt, est)
	}
}

// TestParseYahooChart_NoTimezoneFallsBackToUTC: a payload without
// exchangeTimezoneName must still date its bars (UTC), not fail or blank them.
// Measured to be correct for all six macro symbols in both DST regimes.
func TestParseYahooChart_NoTimezoneFallsBackToUTC(t *testing.T) {
	body := `{"chart":{"result":[{"meta":{"symbol":"TEST"},
		"timestamp":[1787529600],
		"indicators":{"quote":[{"open":[10],"close":[11]}]}}],"error":null}}`
	bars, err := ParseYahooChart([]byte(body))
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if bars[0].Date != "2026-08-24" {
		t.Errorf("Date = %q, want 2026-08-24 (UTC fallback)", bars[0].Date)
	}
}

// TestParseYahooChart_UnknownTimezoneFallsBackToUTC: a bogus zone name must not
// error the parse — it degrades to the UTC date.
func TestParseYahooChart_UnknownTimezoneFallsBackToUTC(t *testing.T) {
	body := `{"chart":{"result":[{"meta":{"symbol":"TEST","exchangeTimezoneName":"Mars/Olympus_Mons"},
		"timestamp":[1787529600],
		"indicators":{"quote":[{"open":[10],"close":[11]}]}}],"error":null}}`
	bars, err := ParseYahooChart([]byte(body))
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if bars[0].Date != "2026-08-24" {
		t.Errorf("Date = %q, want 2026-08-24 (UTC fallback on an unknown zone)", bars[0].Date)
	}
}

// TestParseYahooChart_NullOpenKeepsBar: a bar whose CLOSE is present but whose
// OPEN is null is kept with Open 0. The close is real data and the correlation
// window needs it; only the delta baseline is missing, and the caller degrades
// to the previous close (or to no delta) rather than discarding the day.
func TestParseYahooChart_NullOpenKeepsBar(t *testing.T) {
	body := `{"chart":{"result":[{"meta":{"symbol":"TEST","gmtoffset":0},
		"timestamp":[1787529600,1787616000],
		"indicators":{"quote":[{"open":[100,null],"close":[101,102]}]}}],"error":null}}`

	bars, err := ParseYahooChart([]byte(body))
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(bars) != 2 {
		t.Fatalf("len(bars) = %d, want 2 (a null OPEN must not drop the bar)", len(bars))
	}
	if bars[1].Open != 0 {
		t.Errorf("bars[1].Open = %v, want 0 (null open carries no baseline)", bars[1].Open)
	}
	if bars[1].Close != 102 {
		t.Errorf("bars[1].Close = %v, want 102", bars[1].Close)
	}
}

// TestParseYahooChart_ShortArraysNeverMispair: a truncated close array must not
// panic and must not pair a close with the wrong timestamp — the parser walks
// the SHORTEST array.
func TestParseYahooChart_ShortArraysNeverMispair(t *testing.T) {
	body := `{"chart":{"result":[{"meta":{"symbol":"TEST","gmtoffset":0},
		"timestamp":[1787529600,1787616000,1787702400],
		"indicators":{"quote":[{"open":[100],"close":[101,102]}]}}],"error":null}}`

	bars, err := ParseYahooChart([]byte(body))
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(bars) != 2 {
		t.Fatalf("len(bars) = %d, want 2 (bounded by the shortest array)", len(bars))
	}
	if bars[0].Open != 100 || bars[1].Open != 0 {
		t.Errorf("opens = %v/%v, want 100/0 (short open array → no baseline, no mispair)", bars[0].Open, bars[1].Open)
	}
}

// TestParseYahooChart_ErrorShapes: every non-payload answer must be an ERROR,
// so the worker falls through to the next provider instead of recording
// "yahoo had nothing" and going dark.
func TestParseYahooChart_ErrorShapes(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"garbage", `not json at all`},
		{"anti-bot HTML", `<!DOCTYPE html><html><head><title>Verify</title></head><body></body></html>`},
		{"chart.error set", `{"chart":{"result":null,"error":{"code":"Not Found","description":"No data found, symbol may be delisted"}}}`},
		{"empty result", `{"chart":{"result":[],"error":null}}`},
		{"no quote block", `{"chart":{"result":[{"meta":{"gmtoffset":0},"timestamp":[1787529600],"indicators":{"quote":[]}}],"error":null}}`},
		{"all bars null-padded", `{"chart":{"result":[{"meta":{"gmtoffset":0},"timestamp":[1787529600,1787616000],"indicators":{"quote":[{"open":[null,null],"close":[null,null]}]}}],"error":null}}`},
		{"no timestamps", `{"chart":{"result":[{"meta":{"gmtoffset":0},"timestamp":[],"indicators":{"quote":[{"open":[],"close":[]}]}}],"error":null}}`},
		// Non-finite guard, reachable half: encoding/json REJECTS an
		// out-of-range literal outright (verified — it errors and leaves the
		// slot at 0, it cannot deliver ±Inf), so an overflowing close fails the
		// whole decode. The isFinite() belt inside the loop is therefore
		// unreachable via the wire today and is documented as such; what must
		// hold either way is this: no panic, no bar, a clean error.
		{"overflowing close", `{"chart":{"result":[{"meta":{"gmtoffset":0},"timestamp":[1787529600],"indicators":{"quote":[{"open":[1],"close":[1e999]}]}}],"error":null}}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			bars, err := ParseYahooChart([]byte(tc.body))
			if err == nil {
				t.Fatalf("err = nil, want error (got %d bars)", len(bars))
			}
			if len(bars) != 0 {
				t.Errorf("bars = %d, want 0 on error", len(bars))
			}
		})
	}
}

// TestYahooTickerMapping is the symbol-mapping table: every canonical macro
// symbol resolves to the ticker verified live on 2026-08-24, and nothing else
// resolves at all.
func TestYahooTickerMapping(t *testing.T) {
	want := map[string]string{
		SymDXY:   "DX-Y.NYB",
		SymRates: "^TNX",
		SymVIX:   "^VIX",
		SymSPX:   "^GSPC",
		SymGold:  "GC=F",
		SymBTC:   "BTC-USD",
	}
	for sym, wantTicker := range want {
		got, ok := YahooTicker(sym)
		if !ok {
			t.Errorf("YahooTicker(%q) not mapped", sym)
			continue
		}
		if got != wantTicker {
			t.Errorf("YahooTicker(%q) = %q, want %q", sym, got, wantTicker)
		}
	}
	// Every symbol the worker fetches must be mapped — otherwise the fallback
	// silently skips a lamp.
	for _, sym := range allSymbols {
		if _, ok := YahooTicker(sym); !ok {
			t.Errorf("allSymbols entry %q has no Yahoo ticker — fallback would skip it", sym)
		}
	}
	if len(yahooTickers) != len(allSymbols) {
		t.Errorf("yahooTickers has %d entries, allSymbols has %d — the table has drifted",
			len(yahooTickers), len(allSymbols))
	}
	if _, ok := YahooTicker("nonsense"); ok {
		t.Errorf(`YahooTicker("nonsense") resolved, want not-ok`)
	}
}

// TestYahooSource_FetchQuote: the real fixture served over httptest becomes a
// Quote with the last bar's close, the bar's own open as the delta baseline,
// the bar's start instant as AsOf, and source "yahoo".
func TestYahooSource_FetchQuote(t *testing.T) {
	var gotPath, gotUA string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path + "?" + r.URL.RawQuery
		gotUA = r.Header.Get("User-Agent")
		_, _ = w.Write(readFixture(t, "yahoo_dxy_1d_3mo.json"))
	}))
	defer srv.Close()

	src := &yahooSource{base: srv.URL + "/", get: testWorker().httpGet}
	q, err := src.FetchQuote(context.Background(), SymDXY)
	if err != nil {
		t.Fatalf("FetchQuote err = %v", err)
	}
	if !q.OK {
		t.Fatalf("OK = false, want true")
	}
	if q.Symbol != SymDXY {
		t.Errorf("Symbol = %q, want %q (the CANONICAL id, not the Yahoo ticker)", q.Symbol, SymDXY)
	}
	if q.Source != SourceYahoo {
		t.Errorf("Source = %q, want %q", q.Source, SourceYahoo)
	}
	if !closeEnough(q.Price, 98.97) {
		t.Errorf("Price = %v, want ~98.97 (last bar close)", q.Price)
	}
	if !closeEnough(q.Open, 98.856) {
		t.Errorf("Open = %v, want ~98.856 (the bar's OWN open — the session-change "+
			"baseline, matching stooq's definition, not the previous close 98.80)", q.Open)
	}
	wantAsOf := time.Date(2026, 8, 24, 4, 0, 0, 0, time.UTC)
	if !q.AsOf.Equal(wantAsOf) {
		t.Errorf("AsOf = %v, want %v (the bar's start instant)", q.AsOf, wantAsOf)
	}

	// The canonical id must have been translated to the Yahoo ticker on the wire.
	if !strings.Contains(gotPath, "DX-Y.NYB") {
		t.Errorf("request path = %q, want the DX-Y.NYB ticker", gotPath)
	}
	if !strings.Contains(gotPath, "interval=1d") || !strings.Contains(gotPath, "range="+yahooQuoteRange) {
		t.Errorf("request path = %q, want interval=1d and range=%s", gotPath, yahooQuoteRange)
	}
	// Yahoo answers 429 without a browser UA — the header is load-bearing.
	if gotUA == "" {
		t.Errorf("User-Agent header missing — Yahoo answers 429 without one")
	}
}

// TestYahooSource_FetchQuote_PrevCloseFallback: when the newest bar's open is
// null, the delta baseline falls back to the PREVIOUS bar's close rather than
// leaving the lamp directionless.
func TestYahooSource_FetchQuote_PrevCloseFallback(t *testing.T) {
	body := `{"chart":{"result":[{"meta":{"symbol":"^GSPC","gmtoffset":0},
		"timestamp":[1787529600,1787616000],
		"indicators":{"quote":[{"open":[100,null],"close":[101,102]}]}}],"error":null}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	defer srv.Close()

	src := &yahooSource{base: srv.URL + "/", get: testWorker().httpGet}
	q, err := src.FetchQuote(context.Background(), SymSPX)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if q.Price != 102 {
		t.Errorf("Price = %v, want 102", q.Price)
	}
	if q.Open != 101 {
		t.Errorf("Open = %v, want 101 (previous bar's close as the fallback baseline)", q.Open)
	}
}

// TestYahooSource_FetchQuote_SingleBarNoBaseline: one bar, null open, nothing
// before it → Open stays 0 and the handler emits delta_pct null. No fabricated
// direction.
func TestYahooSource_FetchQuote_SingleBarNoBaseline(t *testing.T) {
	body := `{"chart":{"result":[{"meta":{"symbol":"^GSPC","gmtoffset":0},
		"timestamp":[1787529600],
		"indicators":{"quote":[{"open":[null],"close":[101]}]}}],"error":null}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	defer srv.Close()

	src := &yahooSource{base: srv.URL + "/", get: testWorker().httpGet}
	q, err := src.FetchQuote(context.Background(), SymSPX)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if !q.OK || q.Price != 101 {
		t.Errorf("q = %+v, want OK with Price 101", q)
	}
	if q.Open != 0 {
		t.Errorf("Open = %v, want 0 (no baseline available → delta_pct null)", q.Open)
	}
}

// TestYahooSource_FetchDaily: the fixture becomes a date-ascending daily-close
// series with the null-padded days absent, ready for the correlation window.
func TestYahooSource_FetchDaily(t *testing.T) {
	var gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.RawQuery
		_, _ = w.Write(readFixture(t, "yahoo_dxy_1d_3mo.json"))
	}))
	defer srv.Close()

	src := &yahooSource{base: srv.URL + "/", get: testWorker().httpGet}
	closes, err := src.FetchDaily(context.Background(), SymDXY, time.Now())
	if err != nil {
		t.Fatalf("FetchDaily err = %v", err)
	}
	if len(closes) != 63 {
		t.Fatalf("len(closes) = %d, want 63 (null-padded days dropped)", len(closes))
	}
	if !strings.Contains(gotQuery, "range="+yahooDailyRange) {
		t.Errorf("query = %q, want range=%s for the daily window", gotQuery, yahooDailyRange)
	}
	// Ascending, unique dates — the correlation pairs on this key, so a repeat
	// would weight one day twice.
	seen := make(map[string]bool, len(closes))
	for i, c := range closes {
		if seen[c.Date] {
			t.Fatalf("duplicate date %s at index %d", c.Date, i)
		}
		seen[c.Date] = true
		if i > 0 && closes[i-1].Date >= c.Date {
			t.Fatalf("dates not ascending at %d: %s then %s", i, closes[i-1].Date, c.Date)
		}
	}
	if closes[len(closes)-1].Date != "2026-08-24" {
		t.Errorf("last date = %s, want 2026-08-24", closes[len(closes)-1].Date)
	}
}

// TestYahooSource_FetchDaily_DedupesDate: two bars stamped with the same
// session date collapse to one (last wins).
func TestYahooSource_FetchDaily_DedupesDate(t *testing.T) {
	// Both instants fall on 2026-08-24 UTC.
	body := `{"chart":{"result":[{"meta":{"symbol":"BTC-USD","gmtoffset":0},
		"timestamp":[1787529600,1787551200],
		"indicators":{"quote":[{"open":[1,2],"close":[10,20]}]}}],"error":null}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	defer srv.Close()

	src := &yahooSource{base: srv.URL + "/", get: testWorker().httpGet}
	closes, err := src.FetchDaily(context.Background(), SymBTC, time.Now())
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(closes) != 1 {
		t.Fatalf("len(closes) = %d, want 1 (same date collapses)", len(closes))
	}
	if closes[0].Close != 20 {
		t.Errorf("Close = %v, want 20 (last bar wins)", closes[0].Close)
	}
}

// TestYahooSource_NonOKStatusRejected: a 429/500 body that happens to parse
// must never become data. Yahoo answers 429 to unfriendly clients, so this is
// the live failure mode, not a hypothetical.
func TestYahooSource_NonOKStatusRejected(t *testing.T) {
	for _, status := range []int{http.StatusTooManyRequests, http.StatusInternalServerError} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(status)
			// A perfectly valid payload behind a bad status.
			_, _ = w.Write(readFixture(t, "yahoo_dxy_1d_3mo.json"))
		}))
		src := &yahooSource{base: srv.URL + "/", get: testWorker().httpGet}
		_, err := src.FetchQuote(context.Background(), SymDXY)
		if err == nil {
			t.Errorf("status %d: err = nil, want error (a non-2xx body is never data)", status)
		}
		srv.Close()
	}
}

// TestYahooSource_UnmappedSymbol: a symbol with no ticker errors before any
// network call — it never silently fetches the wrong instrument.
func TestYahooSource_UnmappedSymbol(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("unmapped symbol must not issue a request")
	}))
	defer srv.Close()

	src := &yahooSource{base: srv.URL + "/", get: testWorker().httpGet}
	if _, err := src.FetchQuote(context.Background(), "no.such.symbol"); err == nil {
		t.Errorf("err = nil, want error for an unmapped symbol")
	}
}

// --- helpers ---

func indexOfDate(bars []yahooBar, date string) int {
	for i, b := range bars {
		if b.Date == date {
			return i
		}
	}
	return -1
}

// closeEnough compares floats at the precision the wire carries. Yahoo's chart
// arrays are float32 widened to float64, so 98.97 arrives as 98.97000122070312
// — a relative error around 1e-8 that an exact compare would fail on. The
// tolerance is on the VALUE, not on the parse: the parser copies the decoded
// number verbatim, it never rounds.
func closeEnough(got, want float64) bool {
	d := got - want
	if d < 0 {
		d = -d
	}
	return d < 1e-4
}
