package funding

// store_test.go — offline tests for the in-memory rolling window.
//
// The store's `now` field is overridden per-test so age-pruning is
// deterministic (no real clock dependence).

import (
	"testing"
	"time"
)

// fixedNow returns a now() func pinned to t for deterministic pruning tests.
func fixedNow(t time.Time) func() time.Time {
	return func() time.Time { return t }
}

// mkLiq builds a Liquidation with TS derived from ts (UTC RFC3339).
func mkLiq(symbol, side string, qty, price float64, ts time.Time) Liquidation {
	return Liquidation{
		Symbol:   symbol,
		Side:     side,
		Qty:      qty,
		Price:    price,
		USDValue: price * qty,
		TS:       ts.UTC().Format(time.RFC3339),
	}
}

// TestStore_RecentNewestFirstAndLimit: Add 3 in time order → Recent returns
// newest-first and respects the limit.
func TestStore_RecentNewestFirstAndLimit(t *testing.T) {
	base := time.Date(2026, 5, 30, 0, 0, 0, 0, time.UTC)
	s := NewStore()
	s.now = fixedNow(base.Add(1 * time.Minute)) // all events are recent

	s.Add(mkLiq("BTCUSDT", SideLongLiq, 1, 100, base.Add(0*time.Second)))  // oldest
	s.Add(mkLiq("ETHUSDT", SideShortLiq, 2, 200, base.Add(10*time.Second))) // middle
	s.Add(mkLiq("SOLUSDT", SideLongLiq, 3, 300, base.Add(20*time.Second)))  // newest

	got := s.Recent(10)
	if len(got) != 3 {
		t.Fatalf("Recent(10) len = %d, want 3", len(got))
	}
	// newest-first → SOL, ETH, BTC.
	if got[0].Symbol != "SOLUSDT" || got[1].Symbol != "ETHUSDT" || got[2].Symbol != "BTCUSDT" {
		t.Errorf("order = [%s %s %s], want [SOLUSDT ETHUSDT BTCUSDT]", got[0].Symbol, got[1].Symbol, got[2].Symbol)
	}

	// limit respected.
	got2 := s.Recent(2)
	if len(got2) != 2 {
		t.Fatalf("Recent(2) len = %d, want 2", len(got2))
	}
	if got2[0].Symbol != "SOLUSDT" || got2[1].Symbol != "ETHUSDT" {
		t.Errorf("Recent(2) = [%s %s], want [SOLUSDT ETHUSDT]", got2[0].Symbol, got2[1].Symbol)
	}

	// non-positive limit → empty non-nil slice.
	got0 := s.Recent(0)
	if got0 == nil {
		t.Errorf("Recent(0) = nil, want empty non-nil slice")
	}
	if len(got0) != 0 {
		t.Errorf("Recent(0) len = %d, want 0", len(got0))
	}
}

// TestStore_PrunesOldEntries: an entry older than the 1h window is dropped on
// read; a fresh one survives.
func TestStore_PrunesOldEntries(t *testing.T) {
	now := time.Date(2026, 5, 30, 12, 0, 0, 0, time.UTC)
	s := NewStore()
	s.now = fixedNow(now)

	// 90 min old → stale (window is 60 min).
	s.Add(mkLiq("BTCUSDT", SideLongLiq, 1, 100, now.Add(-90*time.Minute)))
	// 5 min old → fresh.
	s.Add(mkLiq("ETHUSDT", SideShortLiq, 2, 200, now.Add(-5*time.Minute)))

	got := s.Recent(10)
	if len(got) != 1 {
		t.Fatalf("Recent after prune len = %d, want 1 (stale BTC dropped)", len(got))
	}
	if got[0].Symbol != "ETHUSDT" {
		t.Errorf("survivor = %s, want ETHUSDT", got[0].Symbol)
	}
	if s.Len() != 1 {
		t.Errorf("Len after prune = %d, want 1", s.Len())
	}
}

// TestStore_ZonesAggregatesNearPrice: several same-symbol near-price events
// collapse into one zone with summed USD + count and the dominant side.
func TestStore_ZonesAggregatesNearPrice(t *testing.T) {
	now := time.Date(2026, 5, 30, 12, 0, 0, 0, time.UTC)
	s := NewStore()
	s.now = fixedNow(now)

	// Band width is per-symbol = max(price) × 0.5%. refPrice(SOL)=9914 →
	// width ≈ 49.57, so floor(9910/49.57)=floor(9912/49.57)=floor(9914/49.57)
	// = 199 → all three SOL events share one band. Prices kept within a few
	// dollars so they can't straddle a band boundary.
	ts := now.Add(-1 * time.Minute)
	s.Add(mkLiq("SOLUSDT", SideLongLiq, 10, 9910, ts))  // 99,100 USD
	s.Add(mkLiq("SOLUSDT", SideLongLiq, 10, 9912, ts))  // 99,120 USD
	s.Add(mkLiq("SOLUSDT", SideShortLiq, 1, 9914, ts))  //  9,914 USD (minority side)
	// A different symbol → its own zone, must not merge.
	s.Add(mkLiq("BTCUSDT", SideLongLiq, 1, 60000, ts))

	zones := s.Zones()
	if len(zones) != 2 {
		t.Fatalf("zones len = %d, want 2 (one SOL band + one BTC band)", len(zones))
	}

	// Zones are sorted by total USD desc → SOL band (≈208k) first.
	sol := zones[0]
	if sol.Symbol != "SOLUSDT" {
		t.Fatalf("top zone symbol = %s, want SOLUSDT", sol.Symbol)
	}
	if sol.Count != 3 {
		t.Errorf("SOL zone count = %d, want 3", sol.Count)
	}
	wantUSD := 9910.0*10 + 9912.0*10 + 9914.0*1
	if !floatNear(sol.TotalUSD, wantUSD) {
		t.Errorf("SOL zone total_usd = %v, want %v", sol.TotalUSD, wantUSD)
	}
	// Dominant side = long_liq (two long events at ~99k each ≫ one short at ~10k).
	if sol.Side != SideLongLiq {
		t.Errorf("SOL zone dominant side = %q, want %q", sol.Side, SideLongLiq)
	}
	// price_band rendered as "low-high" — for a ~9900 band (>=1000 → 0 dp).
	if sol.PriceBand == "" {
		t.Errorf("SOL zone price_band is empty, want a 'low-high' string")
	}

	// BTC second.
	if zones[1].Symbol != "BTCUSDT" {
		t.Errorf("second zone symbol = %s, want BTCUSDT", zones[1].Symbol)
	}
	if zones[1].Count != 1 {
		t.Errorf("BTC zone count = %d, want 1", zones[1].Count)
	}
}

// TestStore_ZonesEmpty: empty store → empty non-nil slice.
func TestStore_ZonesEmpty(t *testing.T) {
	s := NewStore()
	z := s.Zones()
	if z == nil {
		t.Errorf("Zones() = nil, want empty non-nil slice")
	}
	if len(z) != 0 {
		t.Errorf("Zones() len = %d, want 0", len(z))
	}
}

// TestStore_CountCapEvictsOldest: exceeding maxEvents drops the oldest so the
// hard memory bound holds.
func TestStore_CountCapEvictsOldest(t *testing.T) {
	now := time.Date(2026, 5, 30, 12, 0, 0, 0, time.UTC)
	s := NewStore()
	s.now = fixedNow(now)

	// Add maxEvents+5 fresh events; only the newest maxEvents survive. Spaced
	// 100ms apart so ALL of them (total×100ms ≈ 8.3 min) stay inside the 1h
	// window — this isolates the COUNT cap from the AGE prune. i ascending →
	// newer timestamps, so the highest-priced event is also the newest.
	total := maxEvents + 5
	for i := 0; i < total; i++ {
		offset := time.Duration(total-i) * 100 * time.Millisecond
		s.Add(mkLiq("BTCUSDT", SideLongLiq, 1, float64(1000+i), now.Add(-offset)))
	}
	if s.Len() != maxEvents {
		t.Fatalf("Len after overflow = %d, want cap %d", s.Len(), maxEvents)
	}
	// Newest event (price 1000+total-1) must still be present at the front.
	got := s.Recent(1)
	if len(got) != 1 {
		t.Fatalf("Recent(1) len = %d, want 1", len(got))
	}
	wantNewestPrice := float64(1000 + total - 1)
	if got[0].Price != wantNewestPrice {
		t.Errorf("newest price = %v, want %v", got[0].Price, wantNewestPrice)
	}
}

func floatNear(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-6
}
