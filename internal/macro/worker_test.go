package macro

// worker_test.go — offline tests for the stooq + F&G parsers and one refresh
// cycle against an httptest server (StooqBase/FngURL overridden). Mirrors
// funding/worker_test.go (ParseForceOrder) rigor.

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestParseStooqCSV(t *testing.T) {
	t.Run("valid row → OK quote", func(t *testing.T) {
		body := []byte("Symbol,Date,Time,Open,High,Low,Close,Volume\n^SPX,2026-05-29,23:00:00,7579.3,7599.4,7563.6,7580.1,5498484255\n")
		q, err := ParseStooqCSV(body)
		if err != nil {
			t.Fatalf("ParseStooqCSV err = %v", err)
		}
		if !q.OK {
			t.Fatalf("OK = false, want true")
		}
		if q.Price != 7580.1 {
			t.Errorf("Price = %v, want 7580.1", q.Price)
		}
		if q.Symbol != "^SPX" {
			t.Errorf("Symbol = %q, want ^SPX", q.Symbol)
		}
		if q.Open != 7579.3 {
			t.Errorf("Open = %v, want 7579.3 (idx 3)", q.Open)
		}
		wantTime := time.Date(2026, 5, 29, 23, 0, 0, 0, time.UTC)
		if !q.AsOf.Equal(wantTime) {
			t.Errorf("AsOf = %v, want %v", q.AsOf, wantTime)
		}
	})

	t.Run("Close=N/D → OK:false, nil err, DATE SURVIVES", func(t *testing.T) {
		body := []byte("Symbol,Date,Time,Open,High,Low,Close,Volume\n^VIX,2026-05-29,23:00:00,N/D,N/D,N/D,N/D,N/D\n")
		q, err := ParseStooqCSV(body)
		if err != nil {
			t.Fatalf("ParseStooqCSV(N/D close) err = %v, want nil", err)
		}
		if q.OK {
			t.Errorf("OK = true, want false on N/D close")
		}
		// The row still carried a date → it must ship on the N/D quote (the
		// lamp's as_of shows the last known session, not "").
		wantTime := time.Date(2026, 5, 29, 23, 0, 0, 0, time.UTC)
		if !q.AsOf.Equal(wantTime) {
			t.Errorf("AsOf = %v, want %v (date survives an N/D close)", q.AsOf, wantTime)
		}
	})

	t.Run("Close=N/D with N/D time, valid date → date-only AsOf", func(t *testing.T) {
		body := []byte("Symbol,Date,Time,Open,High,Low,Close,Volume\n^VIX,2026-05-29,N/D,N/D,N/D,N/D,N/D,N/D\n")
		q, err := ParseStooqCSV(body)
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if q.OK {
			t.Errorf("OK = true, want false")
		}
		wantTime := time.Date(2026, 5, 29, 0, 0, 0, 0, time.UTC)
		if !q.AsOf.Equal(wantTime) {
			t.Errorf("AsOf = %v, want %v (midnight of the known date)", q.AsOf, wantTime)
		}
	})

	t.Run("fully empty N/D row → OK:false, zero AsOf, no panic", func(t *testing.T) {
		// stooq emits "<sym>,N/D,N/D,N/D,N/D,N/D,N/D,N/D" for a dead symbol.
		body := []byte("Symbol,Date,Time,Open,High,Low,Close,Volume\n10USY.B,N/D,N/D,N/D,N/D,N/D,N/D,N/D\n")
		q, err := ParseStooqCSV(body)
		if err != nil {
			t.Fatalf("ParseStooqCSV(all N/D) err = %v, want nil", err)
		}
		if q.OK {
			t.Errorf("OK = true, want false on all-N/D row")
		}
		if !q.AsOf.IsZero() {
			t.Errorf("AsOf = %v, want zero (no date on the row)", q.AsOf)
		}
	})

	t.Run("unparseable close float → OK:false", func(t *testing.T) {
		body := []byte("Symbol,Date,Time,Open,High,Low,Close,Volume\nXXX,2026-05-29,23:00:00,1,2,3,notafloat,9\n")
		q, err := ParseStooqCSV(body)
		if err != nil {
			t.Fatalf("err = %v, want nil (treated as N/D)", err)
		}
		if q.OK {
			t.Errorf("OK = true, want false on bad float")
		}
	})

	t.Run("non-finite close (NaN/Inf) → OK:false, never a value", func(t *testing.T) {
		// strconv.ParseFloat happily parses "NaN"/"Inf" — the review's fix 4:
		// a non-finite quote must degrade to N/D, or a lamp goes ok:true with
		// an unserializable value.
		for _, bad := range []string{"NaN", "Inf", "-Inf", "+Inf"} {
			body := []byte("Symbol,Date,Time,Open,High,Low,Close,Volume\nXXX,2026-05-29,23:00:00,1,2,3," + bad + ",9\n")
			q, err := ParseStooqCSV(body)
			if err != nil {
				t.Fatalf("close=%s: err = %v, want nil", bad, err)
			}
			if q.OK {
				t.Errorf("close=%s: OK = true, want false (non-finite is no data)", bad)
			}
		}
	})

	t.Run("non-finite open → Open 0 (no delta), value still OK", func(t *testing.T) {
		body := []byte("Symbol,Date,Time,Open,High,Low,Close,Volume\nXXX,2026-05-29,23:00:00,NaN,2,3,4.5,9\n")
		q, err := ParseStooqCSV(body)
		if err != nil {
			t.Fatalf("err = %v", err)
		}
		if !q.OK || q.Price != 4.5 {
			t.Errorf("q = %+v, want OK 4.5", q)
		}
		if q.Open != 0 {
			t.Errorf("Open = %v, want 0 (non-finite open carries no baseline)", q.Open)
		}
	})

	t.Run("valid close, unparseable timestamp → OK, zero AsOf", func(t *testing.T) {
		body := []byte("Symbol,Date,Time,Open,High,Low,Close,Volume\nXXX,bad-date,bad-time,1,2,3,4.5,9\n")
		q, err := ParseStooqCSV(body)
		if err != nil {
			t.Fatalf("err = %v", err)
		}
		// dateStr "bad-date" is not "N/D" so it's not the sentinel; close is valid
		// → OK true, AsOf zero (the time parse fails silently).
		if !q.OK {
			t.Errorf("OK = false, want true (valid close)")
		}
		if q.Open != 1 {
			t.Errorf("Open = %v, want 1 (idx 3)", q.Open)
		}
		if !q.AsOf.IsZero() {
			t.Errorf("AsOf = %v, want zero", q.AsOf)
		}
	})

	t.Run("N/D Open, valid Close → OK with Open=0", func(t *testing.T) {
		// A real (if rare) shape: the session just opened so Open is N/D but a
		// Close print exists → lamp shows its value, the handler emits delta_pct
		// null (Open=0), no fabricated direction.
		body := []byte("Symbol,Date,Time,Open,High,Low,Close,Volume\nXXX,2026-05-29,23:00:00,N/D,N/D,N/D,4.5,9\n")
		q, err := ParseStooqCSV(body)
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if !q.OK {
			t.Errorf("OK = false, want true (valid close)")
		}
		if q.Price != 4.5 {
			t.Errorf("Price = %v, want 4.5", q.Price)
		}
		if q.Open != 0 {
			t.Errorf("Open = %v, want 0 (N/D open)", q.Open)
		}
	})

	t.Run("CRLF body parses cleanly", func(t *testing.T) {
		// stooq can serve CRLF line endings; the \r must not corrupt the last
		// field or split the row oddly.
		body := []byte("Symbol,Date,Time,Open,High,Low,Close,Volume\r\n^SPX,2026-05-29,23:00:00,7579.3,7599.4,7563.6,7580.1,5498484255\r\n")
		q, err := ParseStooqCSV(body)
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if !q.OK || q.Price != 7580.1 || q.Open != 7579.3 {
			t.Errorf("q = %+v, want OK price 7580.1 open 7579.3", q)
		}
	})

	t.Run("empty body → error", func(t *testing.T) {
		if _, err := ParseStooqCSV([]byte("")); err == nil {
			t.Errorf("err = nil, want error on empty body")
		}
	})

	t.Run("header only (no data row) → error", func(t *testing.T) {
		if _, err := ParseStooqCSV([]byte("Symbol,Date,Time,Open,High,Low,Close,Volume\n")); err == nil {
			t.Errorf("err = nil, want error on header-only body")
		}
	})

	t.Run("short row → error", func(t *testing.T) {
		body := []byte("Symbol,Date,Time\n^SPX,2026-05-29,23:00:00\n")
		if _, err := ParseStooqCSV(body); err == nil {
			t.Errorf("err = nil, want error on short row")
		}
	})
}

func TestParseFnG(t *testing.T) {
	t.Run("valid → Value/Label/OK", func(t *testing.T) {
		body := []byte(`{"data":[{"value":"23","value_classification":"Extreme Fear","timestamp":"1780099200"}]}`)
		f, err := ParseFnG(body)
		if err != nil {
			t.Fatalf("ParseFnG err = %v", err)
		}
		if !f.OK {
			t.Fatalf("OK = false, want true")
		}
		if f.Value != 23 {
			t.Errorf("Value = %d, want 23", f.Value)
		}
		if f.Label != "Extreme Fear" {
			t.Errorf("Label = %q, want Extreme Fear", f.Label)
		}
	})

	t.Run("empty data array → OK:false", func(t *testing.T) {
		f, err := ParseFnG([]byte(`{"data":[]}`))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if f.OK {
			t.Errorf("OK = true, want false on empty data")
		}
	})

	t.Run("non-integer value → OK:false", func(t *testing.T) {
		f, err := ParseFnG([]byte(`{"data":[{"value":"abc","value_classification":"?"}]}`))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if f.OK {
			t.Errorf("OK = true, want false on non-integer value")
		}
	})

	t.Run("malformed JSON → error", func(t *testing.T) {
		if _, err := ParseFnG([]byte(`{not json`)); err == nil {
			t.Errorf("err = nil, want error on malformed JSON")
		}
	})
}

// ── daily-history CSV parsing (B2) ───────────────────────────────────────────

func TestParseStooqDailyCSV(t *testing.T) {
	t.Run("valid rows → ascending closes", func(t *testing.T) {
		body := []byte("Date,Open,High,Low,Close,Volume\n" +
			"2026-08-18,117433.94,119482.98,116215.98,117294.65,123\n" +
			"2026-08-19,117294.65,118000.00,116500.00,117800.10,456\n")
		got, err := ParseStooqDailyCSV(body)
		if err != nil {
			t.Fatalf("err = %v", err)
		}
		if len(got) != 2 {
			t.Fatalf("rows = %d, want 2", len(got))
		}
		if got[0].Date != "2026-08-18" || got[0].Close != 117294.65 {
			t.Errorf("row0 = %+v", got[0])
		}
		if got[1].Date != "2026-08-19" || got[1].Close != 117800.10 {
			t.Errorf("row1 = %+v", got[1])
		}
	})

	t.Run("volume-less 5-field rows (indices) parse", func(t *testing.T) {
		body := []byte("Date,Open,High,Low,Close\n2026-08-18,7579.3,7599.4,7563.6,7580.1\n")
		got, err := ParseStooqDailyCSV(body)
		if err != nil {
			t.Fatalf("err = %v", err)
		}
		if len(got) != 1 || got[0].Close != 7580.1 {
			t.Errorf("got = %+v, want one 7580.1 row", got)
		}
	})

	t.Run("malformed and N/D rows are skipped, not fatal", func(t *testing.T) {
		body := []byte("Date,Open,High,Low,Close,Volume\n" +
			"2026-08-18,1,2,3,4.5,9\n" +
			"not-a-date,1,2,3,4.5,9\n" +
			"2026-08-19,1,2,3,N/D,9\n" +
			"2026-08-20,1,2,3,notafloat,9\n" +
			"short,row\n" +
			"2026-08-21,1,2,3,5.5,9\n")
		got, err := ParseStooqDailyCSV(body)
		if err != nil {
			t.Fatalf("err = %v", err)
		}
		if len(got) != 2 || got[0].Close != 4.5 || got[1].Close != 5.5 {
			t.Errorf("got = %+v, want the two clean rows", got)
		}
	})

	t.Run("CRLF body parses cleanly", func(t *testing.T) {
		body := []byte("Date,Open,High,Low,Close,Volume\r\n2026-08-18,1,2,3,4.5,9\r\n")
		got, err := ParseStooqDailyCSV(body)
		if err != nil || len(got) != 1 || got[0].Close != 4.5 {
			t.Errorf("got = %+v err = %v, want one 4.5 row", got, err)
		}
	})

	t.Run("non-finite closes are skipped", func(t *testing.T) {
		body := []byte("Date,Open,High,Low,Close,Volume\n" +
			"2026-08-18,1,2,3,4.5,9\n" +
			"2026-08-19,1,2,3,NaN,9\n" +
			"2026-08-20,1,2,3,Inf,9\n" +
			"2026-08-21,1,2,3,5.5,9\n")
		got, err := ParseStooqDailyCSV(body)
		if err != nil {
			t.Fatalf("err = %v", err)
		}
		if len(got) != 2 || got[0].Close != 4.5 || got[1].Close != 5.5 {
			t.Errorf("got = %+v, want the two finite rows only", got)
		}
	})

	t.Run("empty body → error", func(t *testing.T) {
		if _, err := ParseStooqDailyCSV(nil); err == nil {
			t.Error("err = nil, want error on empty body")
		}
	})

	t.Run("non-CSV body (unknown symbol page) → error", func(t *testing.T) {
		if _, err := ParseStooqDailyCSV([]byte("No data\nfor this symbol")); err == nil {
			t.Error("err = nil, want error on a non-CSV body")
		}
	})

	t.Run("header only → error", func(t *testing.T) {
		if _, err := ParseStooqDailyCSV([]byte("Date,Open,High,Low,Close,Volume\n")); err == nil {
			t.Error("err = nil, want error on header-only body")
		}
	})
}

// dailyCSVFor renders `days` daily rows ending today (UTC), close = base+i.
func dailyCSVFor(days int, base float64) string {
	var b strings.Builder
	b.WriteString("Date,Open,High,Low,Close,Volume\n")
	start := time.Now().UTC().AddDate(0, 0, -days)
	for i := 0; i < days; i++ {
		d := start.AddDate(0, 0, i).Format("2006-01-02")
		fmt.Fprintf(&b, "%s,1,2,1,%f,9\n", d, base+float64(i))
	}
	return b.String()
}

// newStooqStub serves quote CSV on /q/l/ (valid row for ^spx, N/D otherwise)
// and dailyDays of daily CSV on /q/d/l/ for any symbol, counting daily hits.
func newStooqStub(t *testing.T, dailyDays int) (*httptest.Server, *int32) {
	t.Helper()
	var dailyHits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sym := r.URL.Query().Get("s")
		w.Header().Set("Content-Type", "text/csv")
		if strings.HasPrefix(r.URL.Path, "/q/d/l") {
			atomic.AddInt32(&dailyHits, 1)
			_, _ = io.WriteString(w, dailyCSVFor(dailyDays, 100))
			return
		}
		if sym == SymSPX {
			_, _ = io.WriteString(w, "Symbol,Date,Time,Open,High,Low,Close,Volume\n^SPX,2026-05-29,23:00:00,7579.3,7599.4,7563.6,7580.1,5498484255\n")
			return
		}
		_, _ = io.WriteString(w, "Symbol,Date,Time,Open,High,Low,Close,Volume\n"+strings.ToUpper(sym)+",N/D,N/D,N/D,N/D,N/D,N/D,N/D\n")
	}))
	t.Cleanup(srv.Close)
	return srv, &dailyHits
}

// TestWorkerRefresh_OneCycle points the worker at httptest servers and verifies
// one refresh writes latest quotes + the daily histories + the F&G read. The
// stooq stub returns a valid quote row for ^spx and N/D for everything else,
// so exactly one symbol is OK; the daily endpoint serves 25 rows for everyone.
func TestWorkerRefresh_OneCycle(t *testing.T) {
	stooq, dailyHits := newStooqStub(t, 25)

	fng := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"data":[{"value":"54","value_classification":"Greed","timestamp":"1"}]}`)
	}))
	defer fng.Close()

	store := NewStore()
	wk := &Worker{
		Store:          store,
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		FngURL:         fng.URL + "/fng",
	}
	wk.refresh(context.Background())

	// latest should have all 6 symbols (^spx OK, the rest N/D).
	latest := store.Latest()
	if len(latest) != len(allSymbols) {
		t.Fatalf("latest len = %d, want %d", len(latest), len(allSymbols))
	}
	spx, ok := latest[SymSPX]
	if !ok || !spx.OK || spx.Price != 7580.1 {
		t.Errorf("SPX quote = %+v, want OK price 7580.1", spx)
	}
	vix := latest[SymVIX]
	if vix.OK {
		t.Errorf("VIX quote OK = true, want false (N/D)")
	}

	// Daily history stored for every symbol (25 rows each), one GET per symbol.
	if got := atomic.LoadInt32(dailyHits); got != int32(len(allSymbols)) {
		t.Errorf("daily GETs = %d, want %d (one per symbol)", got, len(allSymbols))
	}
	for _, sym := range allSymbols {
		if store.DailyCount(sym) != 25 {
			t.Errorf("DailyCount(%s) = %d, want 25", sym, store.DailyCount(sym))
		}
	}

	// F&G captured.
	f, has := store.FnG()
	if !has || !f.OK || f.Value != 54 {
		t.Errorf("FnG = %+v has=%v, want OK 54", f, has)
	}
}

// TestWorkerDaily_OncePerDay: the daily fetch runs on the FIRST cycle, is
// skipped while <24h old, and runs again once the clock passes 24h.
func TestWorkerDaily_OncePerDay(t *testing.T) {
	stooq, dailyHits := newStooqStub(t, 25)
	fng := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"data":[]}`)
	}))
	defer fng.Close()

	clock := time.Date(2026, 8, 20, 9, 0, 0, 0, time.UTC)
	wk := &Worker{
		Store:          NewStore(),
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		FngURL:         fng.URL + "/fng",
		now:            func() time.Time { return clock },
	}

	wk.refresh(context.Background()) // warm start → daily fetch
	if got := atomic.LoadInt32(dailyHits); got != 6 {
		t.Fatalf("daily GETs after warm start = %d, want 6", got)
	}

	clock = clock.Add(3 * time.Minute) // next tick, same day → skipped
	wk.refresh(context.Background())
	if got := atomic.LoadInt32(dailyHits); got != 6 {
		t.Errorf("daily GETs after 3min = %d, want still 6 (once a day)", got)
	}

	clock = clock.Add(25 * time.Hour) // past the 24h boundary → refetch
	wk.refresh(context.Background())
	if got := atomic.LoadInt32(dailyHits); got != 12 {
		t.Errorf("daily GETs after 25h = %d, want 12", got)
	}
}

// TestWorkerDaily_TotalFailureRetriesNextTick: when EVERY daily fetch fails
// the success stamp must not advance — the next cycle retries instead of
// going a day with an empty window.
func TestWorkerDaily_TotalFailureRetriesNextTick(t *testing.T) {
	var dailyHits int32
	stooq := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/q/d/l") {
			atomic.AddInt32(&dailyHits, 1)
			http.Error(w, "boom", http.StatusInternalServerError)
			return
		}
		_, _ = io.WriteString(w, "Symbol,Date,Time,Open,High,Low,Close,Volume\nX,N/D,N/D,N/D,N/D,N/D,N/D,N/D\n")
	}))
	defer stooq.Close()
	fng := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"data":[]}`)
	}))
	defer fng.Close()

	clock := time.Date(2026, 8, 20, 9, 0, 0, 0, time.UTC)
	wk := &Worker{
		Store:          NewStore(),
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		FngURL:         fng.URL + "/fng",
		now:            func() time.Time { return clock },
	}

	wk.refresh(context.Background())
	if got := atomic.LoadInt32(&dailyHits); got != 6 {
		t.Fatalf("daily GETs = %d, want 6", got)
	}
	clock = clock.Add(3 * time.Minute)
	wk.refresh(context.Background()) // total failure → retried immediately
	if got := atomic.LoadInt32(&dailyHits); got != 12 {
		t.Errorf("daily GETs after failed cycle = %d, want 12 (retry, not once-a-day)", got)
	}
}

// TestWorkerBudgets_SlowQuotesDoNotStarveDaily (review fix 13): the quote,
// daily and F&G phases run under SEPARATE context budgets — a hanging quote
// endpoint exhausts only its own budget, and the daily history still lands.
func TestWorkerBudgets_SlowQuotesDoNotStarveDaily(t *testing.T) {
	stooq := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/csv")
		if strings.HasPrefix(r.URL.Path, "/q/d/l") { // daily: fast
			_, _ = io.WriteString(w, dailyCSVFor(25, 100))
			return
		}
		time.Sleep(300 * time.Millisecond) // quotes: hang past their budget
		_, _ = io.WriteString(w, "Symbol,Date,Time,Open,High,Low,Close,Volume\nX,N/D,N/D,N/D,N/D,N/D,N/D,N/D\n")
	}))
	defer stooq.Close()
	fng := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"data":[{"value":"54","value_classification":"Greed","timestamp":"1"}]}`)
	}))
	defer fng.Close()

	store := NewStore()
	wk := &Worker{
		Store:          store,
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		FngURL:         fng.URL + "/fng",
		QuotesBudget:   50 * time.Millisecond, // quotes die fast…
		DailyBudget:    5 * time.Second,       // …daily still has room
		FngBudget:      5 * time.Second,
	}
	wk.refresh(context.Background())

	for _, sym := range allSymbols {
		if store.DailyCount(sym) != 25 {
			t.Errorf("DailyCount(%s) = %d, want 25 (daily budget independent of quotes)", sym, store.DailyCount(sym))
		}
	}
	if f, has := store.FnG(); !has || f.Value != 54 {
		t.Errorf("FnG = %+v/%v, want the live 54 read (own budget)", f, has)
	}
	// The quotes themselves honestly degraded to N/D sentinels.
	for _, q := range store.Latest() {
		if q.OK {
			t.Errorf("quote %s OK = true, want false (budget exceeded)", q.Symbol)
		}
	}
}

// TestWorkerDaily_AncientRowsRejected: a body whose rows are all older than the
// age guard stores nothing (correlations honestly stay "building" instead of
// being computed off years-old closes).
func TestWorkerDaily_AncientRowsRejected(t *testing.T) {
	stooq := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/q/d/l") {
			_, _ = io.WriteString(w, "Date,Open,High,Low,Close,Volume\n2005-01-03,1,2,1,4.5,9\n2005-01-04,1,2,1,4.6,9\n")
			return
		}
		_, _ = io.WriteString(w, "Symbol,Date,Time,Open,High,Low,Close,Volume\nX,N/D,N/D,N/D,N/D,N/D,N/D,N/D\n")
	}))
	defer stooq.Close()
	fng := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"data":[]}`)
	}))
	defer fng.Close()

	store := NewStore()
	wk := &Worker{
		Store:          store,
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		FngURL:         fng.URL + "/fng",
	}
	wk.refresh(context.Background())
	for _, sym := range allSymbols {
		if got := store.DailyCount(sym); got != 0 {
			t.Errorf("DailyCount(%s) = %d, want 0 (ancient rows rejected)", sym, got)
		}
	}
}

// TestWorkerRun_NilStore: Run errors immediately on a nil store.
func TestWorkerRun_NilStore(t *testing.T) {
	wk := &Worker{Logger: log.New(io.Discard, "", 0)}
	if err := wk.Run(context.Background()); err == nil {
		t.Errorf("Run(nil store) err = nil, want error")
	}
}

// TestWorkerRun_CtxCancel: Run returns the ctx error after the warm-start when
// the context is already cancelled (no panic, clean shutdown path).
func TestWorkerRun_CtxCancel(t *testing.T) {
	stooq := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "Symbol,Date,Time,Open,High,Low,Close,Volume\n"+strings.ToUpper(r.URL.Query().Get("s"))+",N/D,N/D,N/D,N/D,N/D,N/D,N/D\n")
	}))
	defer stooq.Close()
	fng := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"data":[]}`)
	}))
	defer fng.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled

	wk := &Worker{
		Store:          NewStore(),
		Logger:         log.New(io.Discard, "", 0),
		StooqBase:      stooq.URL + "/q/l/",
		StooqDailyBase: stooq.URL + "/q/d/l/",
		FngURL:         fng.URL + "/fng",
		Interval:       time.Hour,
	}
	err := wk.Run(ctx)
	if err == nil {
		t.Errorf("Run(cancelled ctx) err = nil, want ctx error")
	}
}
