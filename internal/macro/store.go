package macro

// store.go — thread-safe in-memory state for Macro Sentiment. NO database.
//
// Mirrors funding/store.go (single RWMutex, prune-on-write, copies handed out,
// NewStore presizes). There is no Postgres in the ship path — a cold restart
// loses the rolling window and the feed refills from the worker within a couple
// of hours; the frontend frames a cold correlation window as the source's
// nature, not a bug.
//
// Two independent states under one lock:
//   (a) latest map[string]Quote — the last snapshot of each of the 6 symbols.
//   (b) ring  []point           — a rolling FIFO of price snapshots, one per
//       worker cycle, used for BTC↔X correlations.
//   plus the last valid Fear & Greed read (best-effort overlay).

import (
	"sync"
	"time"
)

// RingCap is the correlation-ring capacity, exported so the handler can render
// an honest "(rolling)" suffix on the window description once the ring is full.
const RingCap = ringCap

const (
	// ringCap caps the correlation ring. At the 3-min poll cadence one point is
	// written per cycle → 64 points ≈ 3.2h of rolling window. Plenty for a
	// stable intraday Pearson, and the memory is trivial (64 × ~6 float64).
	// Oldest points drop from the front (FIFO), mirroring funding's count cap.
	ringCap = 64

	// minCorrPoints is how many overlapping (both-symbols-present) points we need
	// before computing a correlation. <24 pairs is a jumpy coefficient; 24 points
	// ≈ 1.2h at the 3-min cadence. Below it we return nil → frontend shows
	// "Building correlation window".
	minCorrPoints = 24
)

// point is one rolling snapshot: the cycle timestamp + the prices of whichever
// symbols were OK in that cycle (an N/D symbol is simply absent from the map).
type point struct {
	ts     time.Time
	prices map[string]float64
}

// Store is the in-memory Macro Sentiment state. The zero value is NOT ready —
// use NewStore (it presizes the ring). All methods are safe for concurrent use;
// the worker calls SetQuote/AddPoint/SetFnG, the handler calls
// Latest/Correlation/PointCount/FnG.
type Store struct {
	mu     sync.RWMutex
	latest map[string]Quote
	ring   []point
	fng    FnG
	hasFnG bool

	// now is injectable for deterministic tests; defaults to time.Now. Never nil
	// after NewStore.
	now func() time.Time
}

// NewStore returns an empty Store with the ring presized to ringCap.
func NewStore() *Store {
	return &Store{
		latest: make(map[string]Quote),
		ring:   make([]point, 0, ringCap),
		now:    time.Now,
	}
}

// SetQuote overwrites latest[q.Symbol] with the freshest fact for that symbol.
// An N/D quote (OK=false) overwrites too — we store the LAST fact, not the last
// VALID one. Holding a stale "last valid" value would fake freshness on a dead
// symbol; the handler reads Quote.OK and renders "—" instead.
func (s *Store) SetQuote(q Quote) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.latest[q.Symbol] = q
}

// AddPoint appends one rolling snapshot of the given prices (only OK symbols
// from this cycle) and prunes the ring to ringCap (FIFO, oldest dropped). The
// prices map is copied so the caller may mutate or retain theirs without racing
// the store.
func (s *Store) AddPoint(prices map[string]float64) {
	cp := make(map[string]float64, len(prices))
	for k, v := range prices {
		cp[k] = v
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ring = append(s.ring, point{ts: s.now(), prices: cp})
	if len(s.ring) > ringCap {
		drop := len(s.ring) - ringCap
		s.ring = append(s.ring[:0], s.ring[drop:]...)
	}
}

// SetFnG records the last valid Fear & Greed read (best-effort overlay).
func (s *Store) SetFnG(f FnG) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.fng = f
	s.hasFnG = true
}

// Latest returns a copy of the latest-per-symbol map (callers won't race the
// store).
func (s *Store) Latest() map[string]Quote {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[string]Quote, len(s.latest))
	for k, v := range s.latest {
		out[k] = v
	}
	return out
}

// Correlation computes Pearson(BTC-ish symA, symB) over the ring, using only
// points where BOTH symbols are present. <minCorrPoints overlapping points →
// nil. A degenerate window (every value identical — e.g. weekend Friday closes)
// makes Pearson return nil via its zero-variance guard, which the frontend
// renders as "Building window" (honest: a flat window has no computable
// correlation).
func (s *Store) Correlation(symA, symB string) *float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	xs := make([]float64, 0, len(s.ring))
	ys := make([]float64, 0, len(s.ring))
	for _, p := range s.ring {
		a, okA := p.prices[symA]
		b, okB := p.prices[symB]
		if !okA || !okB {
			continue
		}
		xs = append(xs, a)
		ys = append(ys, b)
	}
	if len(xs) < minCorrPoints {
		return nil
	}
	return Pearson(xs, ys)
}

// PointCount reports the current ring size. Used for the window description
// string and the tests.
func (s *Store) PointCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.ring)
}

// OldestPrice returns the price of symbol in the OLDEST ring point that contains
// it (front = oldest), and whether such a point exists. A read accessor over the
// FIFO ring used by store_test to assert eviction + copy isolation; it is not on
// the request path (the lamp delta is the session change carried on Quote.Open,
// not a ring lookup).
func (s *Store) OldestPrice(symbol string) (float64, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, p := range s.ring {
		if v, ok := p.prices[symbol]; ok {
			return v, true
		}
	}
	return 0, false
}

// FnG returns the last valid Fear & Greed read and whether one was ever set.
func (s *Store) FnG() (FnG, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.fng, s.hasFnG
}
