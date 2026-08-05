package macro

// store_test.go — offline tests for the in-memory macro store (latest map +
// correlation ring + F&G). Mirrors funding/store_test.go: `now` is injected so
// ring timestamps are deterministic.

import (
	"math"
	"testing"
	"time"
)

func fixedNow(t time.Time) func() time.Time {
	return func() time.Time { return t }
}

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

// TestStore_AddPointCapEvictsOldest: exceeding ringCap drops the oldest points.
func TestStore_AddPointCapEvictsOldest(t *testing.T) {
	s := NewStore()
	base := time.Date(2026, 5, 30, 12, 0, 0, 0, time.UTC)
	s.now = fixedNow(base)

	total := ringCap + 5
	for i := 0; i < total; i++ {
		s.AddPoint(map[string]float64{SymBTC: float64(1000 + i)})
	}
	if s.PointCount() != ringCap {
		t.Fatalf("PointCount = %d, want cap %d", s.PointCount(), ringCap)
	}
	// The oldest surviving point must be index `total-ringCap` (5), so its BTC
	// price is 1000+5. The OldestPrice helper reads exactly that.
	oldest, ok := s.OldestPrice(SymBTC)
	if !ok {
		t.Fatalf("OldestPrice(BTC) not found")
	}
	wantOldest := float64(1000 + (total - ringCap))
	if oldest != wantOldest {
		t.Errorf("oldest BTC price = %v, want %v (older points evicted)", oldest, wantOldest)
	}
}

// TestStore_AddPointCopiesPrices: AddPoint copies the prices map so a later
// mutation of the caller's map doesn't corrupt the ring.
func TestStore_AddPointCopiesPrices(t *testing.T) {
	s := NewStore()
	prices := map[string]float64{SymBTC: 70000}
	s.AddPoint(prices)
	prices[SymBTC] = 1 // mutate after handing over

	got, ok := s.OldestPrice(SymBTC)
	if !ok || got != 70000 {
		t.Errorf("ring BTC price = %v ok=%v, want 70000 (copy isolated mutation)", got, ok)
	}
}

// TestStore_CorrelationColdStart: an empty / under-filled ring → nil.
func TestStore_CorrelationColdStart(t *testing.T) {
	s := NewStore()
	// Empty ring.
	if c := s.Correlation(SymBTC, SymSPX); c != nil {
		t.Errorf("Correlation(empty) = %v, want nil", *c)
	}
	// Fewer than minCorrPoints overlapping points → still nil.
	for i := 0; i < minCorrPoints-1; i++ {
		s.AddPoint(map[string]float64{SymBTC: float64(i), SymSPX: float64(2 * i)})
	}
	if c := s.Correlation(SymBTC, SymSPX); c != nil {
		t.Errorf("Correlation(%d pts) = %v, want nil (<%d)", minCorrPoints-1, *c, minCorrPoints)
	}
}

// TestStore_CorrelationEnoughPoints: ≥minCorrPoints perfectly-correlated points
// → ≈+1.
func TestStore_CorrelationEnoughPoints(t *testing.T) {
	s := NewStore()
	for i := 0; i < minCorrPoints; i++ {
		// SPX = 2×BTC + a constant → perfect positive correlation. Vary i so the
		// series has non-zero variance.
		s.AddPoint(map[string]float64{SymBTC: float64(100 + i), SymSPX: float64(2*(100+i) + 5)})
	}
	c := s.Correlation(SymBTC, SymSPX)
	if c == nil {
		t.Fatalf("Correlation(%d pts) = nil, want ≈+1", minCorrPoints)
	}
	if math.Abs(*c-1) > 1e-9 {
		t.Errorf("Correlation = %v, want ≈+1", *c)
	}
}

// TestStore_CorrelationSkipsMissingSymbol: points missing one symbol are skipped;
// only overlapping points count, and if too few overlap → nil.
func TestStore_CorrelationSkipsMissingSymbol(t *testing.T) {
	s := NewStore()
	// Add many points but only a handful have BOTH symbols.
	for i := 0; i < minCorrPoints*2; i++ {
		if i%10 == 0 {
			s.AddPoint(map[string]float64{SymBTC: float64(i), SymSPX: float64(2 * i)})
		} else {
			s.AddPoint(map[string]float64{SymBTC: float64(i)}) // SPX missing
		}
	}
	// Overlapping pairs ≈ minCorrPoints*2/10 ≈ 6 < minCorrPoints → nil.
	if c := s.Correlation(SymBTC, SymSPX); c != nil {
		t.Errorf("Correlation(few overlaps) = %v, want nil", *c)
	}
}

// TestStore_CorrelationDegenerateWindow: enough overlapping points but every
// value identical (weekend Friday closes) → nil via Pearson's zero-variance
// guard.
func TestStore_CorrelationDegenerateWindow(t *testing.T) {
	s := NewStore()
	for i := 0; i < minCorrPoints; i++ {
		s.AddPoint(map[string]float64{SymBTC: 70000, SymSPX: 7580})
	}
	if c := s.Correlation(SymBTC, SymSPX); c != nil {
		t.Errorf("Correlation(flat window) = %v, want nil (degenerate)", *c)
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

// TestStore_OldestPriceEmpty: cold ring → (0,false).
func TestStore_OldestPriceEmpty(t *testing.T) {
	s := NewStore()
	if v, ok := s.OldestPrice(SymBTC); ok || v != 0 {
		t.Errorf("OldestPrice(empty) = (%v,%v), want (0,false)", v, ok)
	}
}
