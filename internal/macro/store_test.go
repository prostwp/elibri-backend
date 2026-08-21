package macro

// store_test.go — offline tests for the in-memory macro store (latest map +
// daily-close history + F&G). The correlation window is DAILY closes (B2
// rework): synthetic series with known Pearson answers, date alignment,
// the 20-point minimum and the degenerate (flat) window.

import (
	"math"
	"testing"
	"time"
)

// TestStore_SetQuoteLatestIsCopy: SetQuote stores per-symbol; Latest returns a
// copy that the caller can mutate without racing the store.
func TestStore_SetQuoteLatestIsCopy(t *testing.T) {
	s := NewStore()
	s.SetQuote(Quote{Symbol: SymSPX, Price: 7580.1, OK: true})
	s.SetQuote(Quote{Symbol: SymVIX, OK: false}) // N/D

	got := s.Latest()
	if len(got) != 2 {
		t.Fatalf("Latest len = %d, want 2", len(got))
	}
	if got[SymSPX].Price != 7580.1 || !got[SymSPX].OK {
		t.Errorf("SPX = %+v, want OK 7580.1", got[SymSPX])
	}
	if got[SymVIX].OK {
		t.Errorf("VIX OK = true, want false (N/D stored as-is)")
	}

	// Mutating the returned map must not affect the store.
	got[SymSPX] = Quote{Symbol: SymSPX, Price: 1, OK: true}
	again := s.Latest()
	if again[SymSPX].Price != 7580.1 {
		t.Errorf("store SPX price mutated to %v via returned copy", again[SymSPX].Price)
	}
}

// TestStore_SetQuoteOverwritesWithND: an N/D quote overwrites a prior valid one
// (we keep the last fact, not the last valid one).
func TestStore_SetQuoteOverwritesWithND(t *testing.T) {
	s := NewStore()
	s.SetQuote(Quote{Symbol: SymDXY, Price: 98.85, OK: true})
	s.SetQuote(Quote{Symbol: SymDXY, OK: false}) // now N/D

	q := s.Latest()[SymDXY]
	if q.OK {
		t.Errorf("DXY OK = true, want false (N/D overwrote valid)")
	}
}

// TestStore_SetQuoteCarriesLastKnownDate: a date-less overwrite (date-less N/D
// row or the worker's network-failure sentinel) keeps the previous quote's
// AsOf — "last time this symbol had data" is a fact that survives the source
// going dark. Values/OK are never carried (that would fake freshness), and an
// incoming quote WITH its own date always wins.
func TestStore_SetQuoteCarriesLastKnownDate(t *testing.T) {
	friday := time.Date(2026, 8, 14, 20, 55, 0, 0, time.UTC)
	saturday := time.Date(2026, 8, 15, 10, 0, 0, 0, time.UTC)

	s := NewStore()
	s.SetQuote(Quote{Symbol: SymSPX, Price: 7580.1, AsOf: friday, OK: true})

	// Date-less N/D overwrite → OK flips false, the Friday date survives.
	s.SetQuote(Quote{Symbol: SymSPX, OK: false})
	q := s.Latest()[SymSPX]
	if q.OK {
		t.Fatalf("OK = true, want false (N/D overwrote valid)")
	}
	if q.Price != 0 {
		t.Errorf("Price = %v, want 0 (value never carried forward)", q.Price)
	}
	if !q.AsOf.Equal(friday) {
		t.Errorf("AsOf = %v, want %v (last known date carried forward)", q.AsOf, friday)
	}

	// The carried date survives REPEATED date-less overwrites (every poll
	// cycle on a dead weekend feed).
	s.SetQuote(Quote{Symbol: SymSPX, OK: false})
	if q := s.Latest()[SymSPX]; !q.AsOf.Equal(friday) {
		t.Errorf("AsOf after 2nd N/D = %v, want %v", q.AsOf, friday)
	}

	// An N/D quote that carries its OWN date wins over the carried one.
	s.SetQuote(Quote{Symbol: SymSPX, OK: false, AsOf: saturday})
	if q := s.Latest()[SymSPX]; !q.AsOf.Equal(saturday) {
		t.Errorf("AsOf = %v, want %v (own date beats carried)", q.AsOf, saturday)
	}

	// No previous date at all → AsOf stays zero (nothing to carry).
	s.SetQuote(Quote{Symbol: SymVIX, OK: false})
	if q := s.Latest()[SymVIX]; !q.AsOf.IsZero() {
		t.Errorf("VIX AsOf = %v, want zero (never had a date)", q.AsOf)
	}
}

// ── daily-close history (B2: 20-30d correlation window) ──────────────────────

// dayKey renders day offset i as the store's UTC date key, starting 2026-07-01.
func dayKey(i int) string {
	return time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, i).Format("2006-01-02")
}

// mkDaily builds n daily closes from a value function, dates ascending.
func mkDaily(n int, val func(i int) float64) []DailyClose {
	out := make([]DailyClose, n)
	for i := range out {
		out[i] = DailyClose{Date: dayKey(i), Close: val(i)}
	}
	return out
}

// TestStore_SetDailyClosesCapAndCopy: more than dailyKeep closes → only the
// LAST dailyKeep survive; the caller's slice is copied (later mutation does
// not corrupt the store).
func TestStore_SetDailyClosesCapAndCopy(t *testing.T) {
	s := NewStore()
	in := mkDaily(dailyKeep+5, func(i int) float64 { return float64(1000 + i) })
	s.SetDailyCloses(SymBTC, in)

	if got := s.DailyCount(SymBTC); got != dailyKeep {
		t.Fatalf("DailyCount = %d, want cap %d", got, dailyKeep)
	}
	// Mutate the caller's slice — the store must be isolated.
	in[len(in)-1].Close = -1
	coef, n := s.DailyCorrelation(SymBTC, SymBTC)
	if coef == nil || *coef != 1 || n != dailyKeep {
		t.Errorf("self-correlation after caller mutation = %v/%d, want 1/%d", coef, n, dailyKeep)
	}
}

// TestStore_SetDailyClosesReplaces: a refetch REPLACES the history (the daily
// fetch always serves the full trailing window — append would duplicate days).
func TestStore_SetDailyClosesReplaces(t *testing.T) {
	s := NewStore()
	s.SetDailyCloses(SymBTC, mkDaily(25, func(i int) float64 { return float64(i) }))
	s.SetDailyCloses(SymBTC, mkDaily(10, func(i int) float64 { return float64(i) }))
	if got := s.DailyCount(SymBTC); got != 10 {
		t.Errorf("DailyCount after replace = %d, want 10", got)
	}
}

// TestStore_DailyCorrelationKnownAnswer: synthetic series with a known Pearson.
// SPX = 2×BTC + 5 → exactly +1; DXY = −BTC → exactly −1.
func TestStore_DailyCorrelationKnownAnswer(t *testing.T) {
	s := NewStore()
	s.SetDailyCloses(SymBTC, mkDaily(minDailyCorrPoints, func(i int) float64 { return float64(100 + 3*i) }))
	s.SetDailyCloses(SymSPX, mkDaily(minDailyCorrPoints, func(i int) float64 { return float64(2*(100+3*i) + 5) }))
	s.SetDailyCloses(SymDXY, mkDaily(minDailyCorrPoints, func(i int) float64 { return -float64(100 + 3*i) }))

	coef, n := s.DailyCorrelation(SymBTC, SymSPX)
	if coef == nil || math.Abs(*coef-1) > 1e-9 || n != minDailyCorrPoints {
		t.Errorf("BTC↔SPX = %v/%d, want +1/%d", coef, n, minDailyCorrPoints)
	}
	coef, n = s.DailyCorrelation(SymBTC, SymDXY)
	if coef == nil || math.Abs(*coef+1) > 1e-9 || n != minDailyCorrPoints {
		t.Errorf("BTC↔DXY = %v/%d, want −1/%d", coef, n, minDailyCorrPoints)
	}
}

// TestStore_DailyCorrelationAlignsByDate: BTC trades 7 days a week, SPX only
// weekdays — only SHARED dates pair up. BTC weekend closes carry wild values
// that would destroy the correlation if misaligned; on the shared dates the
// series are perfectly linear → +1 over exactly the shared-day count.
func TestStore_DailyCorrelationAlignsByDate(t *testing.T) {
	s := NewStore()
	base := time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC) // a Monday
	var btc, spx []DailyClose
	shared := 0
	for i := 0; i < 30; i++ {
		d := base.AddDate(0, 0, i)
		key := d.Format("2006-01-02")
		wd := d.Weekday()
		if wd == time.Saturday || wd == time.Sunday {
			btc = append(btc, DailyClose{Date: key, Close: 1e9}) // poison if misaligned
			continue
		}
		shared++
		btc = append(btc, DailyClose{Date: key, Close: float64(100 + i)})
		spx = append(spx, DailyClose{Date: key, Close: float64(3*(100+i) + 7)})
	}
	s.SetDailyCloses(SymBTC, btc)
	s.SetDailyCloses(SymSPX, spx)

	coef, n := s.DailyCorrelation(SymBTC, SymSPX)
	if n != shared {
		t.Fatalf("overlap = %d, want %d (weekday-only alignment)", n, shared)
	}
	if coef == nil || math.Abs(*coef-1) > 1e-9 {
		t.Errorf("aligned correlation = %v, want +1 (weekend poison rows excluded)", coef)
	}
}

// TestStore_DailyCorrelationBelowMin: fewer than minDailyCorrPoints overlapping
// days → nil coefficient, honest point count.
func TestStore_DailyCorrelationBelowMin(t *testing.T) {
	s := NewStore()
	n := minDailyCorrPoints - 1
	s.SetDailyCloses(SymBTC, mkDaily(n, func(i int) float64 { return float64(i) }))
	s.SetDailyCloses(SymSPX, mkDaily(n, func(i int) float64 { return float64(2 * i) }))
	coef, got := s.DailyCorrelation(SymBTC, SymSPX)
	if coef != nil {
		t.Errorf("coef = %v, want nil (%d < %d points)", *coef, n, minDailyCorrPoints)
	}
	if got != n {
		t.Errorf("points = %d, want %d", got, n)
	}
}

// TestStore_DailyCorrelationEmpty: cold store → (nil, 0).
func TestStore_DailyCorrelationEmpty(t *testing.T) {
	s := NewStore()
	if coef, n := s.DailyCorrelation(SymBTC, SymSPX); coef != nil || n != 0 {
		t.Errorf("cold store = (%v, %d), want (nil, 0)", coef, n)
	}
}

// TestStore_DailyCorrelationDegenerate: enough overlapping days but one series
// flat → nil via Pearson's zero-variance guard; points still reported.
func TestStore_DailyCorrelationDegenerate(t *testing.T) {
	s := NewStore()
	s.SetDailyCloses(SymBTC, mkDaily(minDailyCorrPoints, func(i int) float64 { return float64(i) }))
	s.SetDailyCloses(SymSPX, mkDaily(minDailyCorrPoints, func(i int) float64 { return 7580 }))
	coef, n := s.DailyCorrelation(SymBTC, SymSPX)
	if coef != nil {
		t.Errorf("coef = %v, want nil (flat series has no correlation)", *coef)
	}
	if n != minDailyCorrPoints {
		t.Errorf("points = %d, want %d", n, minDailyCorrPoints)
	}
}

// TestStore_DailyHistoryIndependentPerSymbol: histories don't bleed across
// symbols; an unknown symbol reads as empty.
func TestStore_DailyHistoryIndependentPerSymbol(t *testing.T) {
	s := NewStore()
	s.SetDailyCloses(SymBTC, mkDaily(25, func(i int) float64 { return float64(i) }))
	if got := s.DailyCount(SymSPX); got != 0 {
		t.Errorf("SPX DailyCount = %d, want 0", got)
	}
	if got := s.DailyCount(SymBTC); got != 25 {
		t.Errorf("BTC DailyCount = %d, want 25", got)
	}
}

// TestStore_FnG: SetFnG / FnG round-trip, with the has flag.
func TestStore_FnG(t *testing.T) {
	s := NewStore()
	if _, has := s.FnG(); has {
		t.Errorf("has = true on fresh store, want false")
	}
	s.SetFnG(FnG{Value: 54, Label: "Greed", OK: true})
	f, has := s.FnG()
	if !has || f.Value != 54 || f.Label != "Greed" {
		t.Errorf("FnG = %+v has=%v, want {54 Greed} true", f, has)
	}
}
