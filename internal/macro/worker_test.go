package macro

// worker_test.go — offline tests for the stooq + F&G parsers and one refresh
// cycle against an httptest server (StooqBase/FngURL overridden). Mirrors
// funding/worker_test.go (ParseForceOrder) rigor.

import (
	"context"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
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

// TestWorkerRefresh_OneCycle points the worker at httptest servers and verifies
// one refresh writes latest quotes + a ring point + the F&G read. The stooq stub
// returns a valid row for ^spx and N/D for everything else, so exactly one
// symbol is OK.
func TestWorkerRefresh_OneCycle(t *testing.T) {
	stooq := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sym := r.URL.Query().Get("s")
		w.Header().Set("Content-Type", "text/csv")
		if sym == SymSPX {
			_, _ = io.WriteString(w, "Symbol,Date,Time,Open,High,Low,Close,Volume\n^SPX,2026-05-29,23:00:00,7579.3,7599.4,7563.6,7580.1,5498484255\n")
			return
		}
		// Everything else → N/D.
		_, _ = io.WriteString(w, "Symbol,Date,Time,Open,High,Low,Close,Volume\n"+strings.ToUpper(sym)+",N/D,N/D,N/D,N/D,N/D,N/D,N/D\n")
	}))
	defer stooq.Close()

	fng := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"data":[{"value":"54","value_classification":"Greed","timestamp":"1"}]}`)
	}))
	defer fng.Close()

	store := NewStore()
	wk := &Worker{
		Store:     store,
		Logger:    log.New(io.Discard, "", 0),
		StooqBase: stooq.URL + "/q/l/",
		FngURL:    fng.URL + "/fng",
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

	// Exactly one ring point written, holding just the OK symbol.
	if store.PointCount() != 1 {
		t.Errorf("PointCount = %d, want 1", store.PointCount())
	}

	// F&G captured.
	f, has := store.FnG()
	if !has || !f.OK || f.Value != 54 {
		t.Errorf("FnG = %+v has=%v, want OK 54", f, has)
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
		Store:     NewStore(),
		Logger:    log.New(io.Discard, "", 0),
		StooqBase: stooq.URL + "/q/l/",
		FngURL:    fng.URL + "/fng",
		Interval:  time.Hour,
	}
	err := wk.Run(ctx)
	if err == nil {
		t.Errorf("Run(cancelled ctx) err = nil, want ctx error")
	}
}
