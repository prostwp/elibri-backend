package funding

// store.go — thread-safe in-memory rolling window for liquidation events.
//
// This is the PRIMARY (and only) store for funding — there is no Postgres in
// the ship path (spec §4: "in-memory primary; Postgres optional/phase-2"). A
// cold restart loses the window; the feed refills within minutes from the live
// WS, and the frontend copy frames a thin cold-start feed as the source's
// nature, not a bug.
//
// Retention is two-bounded:
//   - time:  entries older than `window` (1h) are pruned on every Add and read.
//   - count: the slice is capped at `maxEvents` (5000); the oldest are dropped
//            when the cap is exceeded. !forceOrder@arr is a firehose across all
//            symbols, so the count cap bounds memory even in a 1h burst.
//
// Everything is guarded by a single RWMutex. Reads (Recent/Zones) take the
// write lock anyway because they prune — the prune mutates the slice — so there
// is no read-only fast path. That is fine: the HTTP read rate (a few per
// second) is trivial next to the WS append rate, and pruning on read keeps the
// window honest without a background sweeper goroutine.

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

const (
	// window is the retention horizon. 1h matches the frontend's "last X min"
	// copy and the spec's rolling-window contract.
	window = time.Hour

	// maxEvents caps the ring regardless of age — a hard memory bound for the
	// all-symbols firehose. 5000 events ≈ a busy hour of large liquidations;
	// well within a few hundred KB.
	maxEvents = 5000

	// zoneBandFraction is the price-band width as a fraction of the band's
	// anchor price (0.5% per spec §5). Bucketing by a *fraction* (not a fixed
	// dollar width) keeps bands sensible across symbols spanning $0.50 alts to
	// $100k BTC without a per-symbol table.
	zoneBandFraction = 0.005

	// maxZones caps how many bands the endpoint returns, ranked by total USD.
	// Keeps the response bounded when the firehose spreads liquidations across
	// many symbols/prices. The frontend renders a short magnet-zone list.
	maxZones = 20
)

// Store is the in-memory rolling window. The zero value is NOT ready — use
// NewStore (it presizes the backing slice). All methods are safe for
// concurrent use; the WS worker calls Add, the HTTP handler calls Recent/Zones.
type Store struct {
	mu     sync.Mutex
	events []Liquidation // append-order (oldest first); pruned from the front
	// now is injectable for deterministic tests; defaults to time.Now. Never
	// nil after NewStore.
	now func() time.Time
}

// NewStore returns an empty rolling window with capacity presized to maxEvents.
func NewStore() *Store {
	return &Store{
		events: make([]Liquidation, 0, maxEvents),
		now:    time.Now,
	}
}

// Add inserts one liquidation, then prunes by age and count. Safe for
// concurrent callers. A zero-value or malformed event is still stored — the
// worker is responsible for only handing over events it could parse; the store
// does not second-guess (it has no symbol watchlist).
func (s *Store) Add(l Liquidation) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, l)
	s.pruneLocked()
}

// Recent returns up to `limit` liquidations, NEWEST-FIRST. A non-positive limit
// returns an empty (non-nil) slice. The result is a fresh copy — callers may
// retain or mutate it without racing the store. Prunes stale entries first so
// a quiet hour doesn't serve hour-old events.
func (s *Store) Recent(limit int) []Liquidation {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pruneLocked()

	if limit <= 0 {
		return []Liquidation{}
	}
	n := len(s.events)
	if limit > n {
		limit = n
	}
	out := make([]Liquidation, 0, limit)
	// events is oldest-first; walk from the tail to emit newest-first.
	for i := n - 1; i >= n-limit; i-- {
		out = append(out, s.events[i])
	}
	return out
}

// Zones aggregates the current window into per-symbol price bands, returns the
// top `maxZones` by total USD (desc). Each event is bucketed by
// floor(price / bandWidth) where bandWidth = price × zoneBandFraction; the
// band's dominant side is whichever side contributed more USD. Always returns a
// non-nil slice. Prunes stale entries first.
func (s *Store) Zones() []Zone {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pruneLocked()

	if len(s.events) == 0 {
		return []Zone{}
	}

	// Bands are PER SYMBOL with a fixed width and a grid ANCHORED to that
	// symbol's lowest observed price. Two design choices, both to make a tight
	// liquidation cluster collapse into ONE band deterministically:
	//   - width is derived once per symbol (from the anchor price), not per
	//     event — a per-event width lets neighbouring prices compute different
	//     band edges.
	//   - the grid origin is the symbol's MIN price, so band index is
	//     floor((price - anchor)/width). A cluster spanning < width (0.5% of
	//     price) all lands in band 0 regardless of where it sits relative to an
	//     absolute grid — this is the boundary-safety the zone test caught when
	//     a max-anchored grid put the top price exactly on an edge.
	anchorPrice := make(map[string]float64)
	for _, e := range s.events {
		if e.Price <= 0 || math.IsNaN(e.Price) || math.IsInf(e.Price, 0) {
			continue
		}
		cur, seen := anchorPrice[e.Symbol]
		if !seen || e.Price < cur {
			anchorPrice[e.Symbol] = e.Price
		}
	}

	// acc accumulates per (symbol, bandIndex). Keyed by a composite string so
	// two symbols that happen to share a band index never collide.
	type bandAcc struct {
		symbol    string
		low, high float64
		totalUSD  float64
		count     int
		longUSD   float64 // USD from long_liq events (for dominant-side tiebreak)
		shortUSD  float64
	}
	acc := make(map[string]*bandAcc)

	for _, e := range s.events {
		// A zero/negative price can't be banded (no meaningful band width);
		// skip it from zone math (it still appears in the raw feed). usd_value
		// of such an event is 0 anyway, so it would not move a zone's total.
		if e.Price <= 0 {
			continue
		}
		anchor := anchorPrice[e.Symbol]
		bandWidth := anchor * zoneBandFraction
		if bandWidth <= 0 || math.IsNaN(bandWidth) || math.IsInf(bandWidth, 0) {
			continue
		}
		// Index relative to the symbol's anchor (min price). A cluster within
		// one width lands in band 0.
		idx := math.Floor((e.Price - anchor) / bandWidth)
		low := anchor + idx*bandWidth
		high := low + bandWidth
		key := e.Symbol + "|" + strconvFormatIdx(idx)

		a := acc[key]
		if a == nil {
			a = &bandAcc{symbol: e.Symbol, low: low, high: high}
			acc[key] = a
		}
		a.totalUSD += e.USDValue
		a.count++
		switch e.Side {
		case SideLongLiq:
			a.longUSD += e.USDValue
		case SideShortLiq:
			a.shortUSD += e.USDValue
		}
	}

	zones := make([]Zone, 0, len(acc))
	for _, a := range acc {
		side := SideLongLiq
		if a.shortUSD > a.longUSD {
			side = SideShortLiq
		}
		zones = append(zones, Zone{
			Symbol:    a.symbol,
			PriceBand: formatBand(a.low, a.high),
			TotalUSD:  a.totalUSD,
			Count:     a.count,
			Side:      side,
		})
	}

	// Rank by total USD desc; ties broken by symbol then band for stable
	// (deterministic) output across requests.
	sort.Slice(zones, func(i, j int) bool {
		if zones[i].TotalUSD != zones[j].TotalUSD {
			return zones[i].TotalUSD > zones[j].TotalUSD
		}
		if zones[i].Symbol != zones[j].Symbol {
			return zones[i].Symbol < zones[j].Symbol
		}
		return zones[i].PriceBand < zones[j].PriceBand
	})

	if len(zones) > maxZones {
		zones = zones[:maxZones]
	}
	return zones
}

// Len reports the current window size (post-prune). Used by tests and the
// handler's empty-vs-populated decision.
func (s *Store) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pruneLocked()
	return len(s.events)
}

// pruneLocked drops entries older than `window`, then enforces the maxEvents
// count cap. Caller MUST hold s.mu.
//
// Age pruning parses each event's TS (RFC3339); an unparseable TS is treated as
// "keep" (we never silently delete an event we can't date — the worker always
// writes a valid ISO TS, so this is defensive only). The count cap then bounds
// the slice regardless.
func (s *Store) pruneLocked() {
	cutoff := s.now().Add(-window)

	// Drop stale from the front. events is oldest-first, but an unparseable TS
	// (kept) could sit before a parseable-stale one; to stay correct we filter
	// in place rather than assuming a clean prefix.
	keep := s.events[:0]
	for _, e := range s.events {
		t, err := time.Parse(time.RFC3339, e.TS)
		if err != nil {
			// Undatable → keep (defensive; worker writes valid TS).
			keep = append(keep, e)
			continue
		}
		if t.After(cutoff) || t.Equal(cutoff) {
			keep = append(keep, e)
		}
	}
	s.events = keep

	// Count cap: keep the newest maxEvents (drop from the front).
	if len(s.events) > maxEvents {
		drop := len(s.events) - maxEvents
		s.events = append(s.events[:0], s.events[drop:]...)
	}
}

// formatBand renders a price band as "low-high" with adaptive precision: large
// prices (BTC) show no decimals, small prices (alts) show enough digits to keep
// the band distinguishable. Mirrors the spec example "9900-9950".
func formatBand(low, high float64) string {
	prec := bandPrecision(low)
	return fmt.Sprintf("%.*f-%.*f", prec, low, prec, high)
}

// bandPrecision picks decimal places from the price magnitude so a $0.50 alt
// band doesn't collapse to "0-0" and a $100k BTC band isn't noisy.
func bandPrecision(price float64) int {
	switch {
	case price >= 1000:
		return 0
	case price >= 10:
		return 1
	case price >= 1:
		return 2
	case price >= 0.01:
		return 4
	default:
		return 6
	}
}

// strconvFormatIdx formats a band index for the map key. Kept separate (and
// integer-floored) so a float index never produces a key with locale or
// exponent surprises.
func strconvFormatIdx(idx float64) string {
	return fmt.Sprintf("%.0f", idx)
}
