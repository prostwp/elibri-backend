package macro

// store.go — thread-safe in-memory state for Macro Sentiment. NO database.
//
// Mirrors funding/store.go (single RWMutex, copies handed out, NewStore
// presizes). Two independent states under one lock:
//   (a) latest map[string]Quote      — the last snapshot of each of the 6 symbols.
//   (b) daily  map[string][]DailyClose — the last 30 DAILY closes per symbol,
//       used for the BTC↔X correlations (B2: a 20-30 trading-day window on
//       daily closes replaced the old ~3h intraday ring).
//   plus the last valid Fear & Greed read (best-effort overlay).
//
// Why in-memory is the honest option for the daily history (B2 "cheapest
// honest option" check): unlike the old intraday ring — which needed ~1.2h of
// uptime to rebuild after a restart and silently served "building" meanwhile —
// the daily window is refetchable IN FULL from stooq's daily CSV in one
// warm-start cycle. A cold restart is repaired within seconds of boot, so a
// database would add operational surface without adding truth.

import (
	"sync"
)

const (
	// dailyKeep caps the per-symbol daily history at the checklist window: the
	// last 30 daily closes (~6 trading weeks for a 5-day symbol, 30 calendar
	// days for 24/7 BTC).
	dailyKeep = 30

	// minDailyCorrPoints — a daily-close Pearson needs at least 20 overlapping
	// trading days (checklist B2: 20-30d window). Below it the correlation is
	// served absent (coef null / ok:false), never a jumpy small-sample number.
	minDailyCorrPoints = 20
)

// MinDailyCorrPoints is minDailyCorrPoints exported for the HTTP handler's
// window description ("N daily closes, min 20 for a read").
const MinDailyCorrPoints = minDailyCorrPoints

// DailyClose is one trading day's close for a symbol. Date is the UTC day key
// ("2006-01-02") straight from the stooq daily CSV — cross-symbol alignment
// happens on this key, never on wall-clock arithmetic (BTC trades 7 days a
// week, the tradfin symbols 5; only shared dates pair up).
type DailyClose struct {
	Date  string
	Close float64
}

// Store is the in-memory Macro Sentiment state. The zero value is NOT ready —
// use NewStore. All methods are safe for concurrent use; the worker calls
// SetQuote/SetDailyCloses/SetFnG, the handler calls Latest/DailyCorrelation/FnG.
type Store struct {
	mu     sync.RWMutex
	latest map[string]Quote
	daily  map[string][]DailyClose
	fng    FnG
	hasFnG bool
}

// NewStore returns an empty Store.
func NewStore() *Store {
	return &Store{
		latest: make(map[string]Quote),
		daily:  make(map[string][]DailyClose),
	}
}

// SetQuote overwrites latest[q.Symbol] with the freshest fact for that symbol.
// An N/D quote (OK=false) overwrites too — we store the LAST fact, not the last
// VALID one. Holding a stale "last valid" value would fake freshness on a dead
// symbol; the handler reads Quote.OK and renders "—" instead.
//
// One exception, dates only: when the incoming quote carries NO date (a
// date-less N/D row, or the worker's network-failure sentinel), the previous
// quote's AsOf is carried forward. The date is a different kind of fact than
// the value — "last time this symbol had data" stays true when the source goes
// dark, and the lamp's as_of should show that last known date rather than "".
// Price/OK are never carried forward (that would fake freshness).
func (s *Store) SetQuote(q Quote) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if q.AsOf.IsZero() {
		if prev, ok := s.latest[q.Symbol]; ok && !prev.AsOf.IsZero() {
			q.AsOf = prev.AsOf
		}
	}
	s.latest[q.Symbol] = q
}

// SetDailyCloses REPLACES the symbol's daily history with the given
// date-ascending series, keeping only the last dailyKeep entries. Replacement
// (not append) keeps the store idempotent across refetches: the daily fetch
// always serves the full trailing window, so appending would duplicate days.
// The slice is copied — the caller may mutate or retain theirs.
func (s *Store) SetDailyCloses(symbol string, closes []DailyClose) {
	if len(closes) > dailyKeep {
		closes = closes[len(closes)-dailyKeep:]
	}
	cp := make([]DailyClose, len(closes))
	copy(cp, closes)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.daily[symbol] = cp
}

// DailyCount reports the stored daily-history length for one symbol.
func (s *Store) DailyCount(symbol string) int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.daily[symbol])
}

// DailyCorrelation computes Pearson(symA, symB) over the daily closes, pairing
// values by calendar date (UTC day key) — only dates present in BOTH histories
// count. Returns the coefficient (nil under minDailyCorrPoints overlapping
// days, or on a degenerate zero-variance window) and the overlap count, so the
// handler can serve an honest ok:false + points instead of a fabricated number.
func (s *Store) DailyCorrelation(symA, symB string) (*float64, int) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	byDate := make(map[string]float64, len(s.daily[symA]))
	for _, d := range s.daily[symA] {
		byDate[d.Date] = d.Close
	}
	xs := make([]float64, 0, len(s.daily[symB]))
	ys := make([]float64, 0, len(s.daily[symB]))
	for _, d := range s.daily[symB] {
		a, ok := byDate[d.Date]
		if !ok {
			continue
		}
		xs = append(xs, a)
		ys = append(ys, d.Close)
	}
	if len(xs) < minDailyCorrPoints {
		return nil, len(xs)
	}
	return Pearson(xs, ys), len(xs)
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

// FnG returns the last valid Fear & Greed read and whether one was ever set.
func (s *Store) FnG() (FnG, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.fng, s.hasFnG
}
