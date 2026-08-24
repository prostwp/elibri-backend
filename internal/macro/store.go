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
// the daily window is refetchable IN FULL from the active provider's daily
// history in one warm-start cycle. A cold restart is repaired within seconds of
// boot, so a database would add operational surface without adding truth.
//
// Each symbol's daily window also records WHICH provider produced it
// (dailySrc) — see source.go. The window and its attribution are always
// replaced together, so a symbol that failed over to a different provider can
// never keep the old source name.

import (
	"sync"
)

const (
	// dailyKeep caps the per-symbol daily history.
	//
	// ⚠️ 42, NOT 30 — do not "optimize" this back down. The cap is applied per
	// symbol in ROWS, but the correlation joins on calendar DATES, and the two
	// legs count rows at different rates: BTC trades 7 days a week, the tradfin
	// symbols 5. At dailyKeep=30, BTC's 30 rows span 30 calendar days while
	// SPX's 30 rows span ~42 — so only the ~20 SPX sessions inside BTC's window
	// can pair. Measured live on 2026-08-24: btc_spx overlapped on EXACTLY 20
	// points against a minDailyCorrPoints minimum of 20. One US market holiday
	// inside the window (Labor Day, 2026-09-07) would have taken it to 19 and
	// blanked the correlation for ~30 days.
	//
	// 42 rows makes BTC's window ~42 calendar days, which covers the tradfin
	// legs' ~30 sessions: measured 29/30/30 overlapping points for
	// spx/gold/dxy — still inside the documented "20-30 daily closes" window,
	// now with ~10 points of margin above the minimum instead of zero.
	dailyKeep = 42

	// minDailyCorrPoints — a daily-close Pearson needs at least 20 overlapping
	// trading days (checklist B2: 20-30d window). Below it the correlation is
	// served absent (coef null / ok:false), never a jumpy small-sample number.
	minDailyCorrPoints = 20
)

// MinDailyCorrPoints is minDailyCorrPoints exported for the HTTP handler's
// window description ("N daily closes, min 20 for a read").
const MinDailyCorrPoints = minDailyCorrPoints

// DailyClose is one trading day's close for a symbol. Date is the SESSION day
// key ("2006-01-02") — straight from the stooq daily CSV, or derived from
// ts+gmtoffset for Yahoo bars (yahoo.go). Cross-symbol alignment happens on
// this key, never on wall-clock arithmetic (BTC trades 7 days a week, the
// tradfin symbols 5; only shared dates pair up).
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
	// dailySrc records WHICH provider produced each symbol's daily history
	// ("stooq"|"yahoo"). Kept parallel to daily rather than folded into
	// DailyClose because provenance is a property of the FETCH, not of an
	// individual row: one refresh replaces the whole window from one source.
	dailySrc map[string]string
	fng      FnG
	hasFnG   bool
}

// NewStore returns an empty Store.
func NewStore() *Store {
	return &Store{
		latest:   make(map[string]Quote),
		daily:    make(map[string][]DailyClose),
		dailySrc: make(map[string]string),
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
//
// source names the provider the window came from ("stooq"|"yahoo") and is
// replaced together with the data: history and provenance are always in sync,
// so a symbol that failed over to a different provider never keeps the old
// attribution. Callers must not pass "" for a non-empty window.
func (s *Store) SetDailyCloses(symbol string, closes []DailyClose, source string) {
	if len(closes) > dailyKeep {
		closes = closes[len(closes)-dailyKeep:]
	}
	cp := make([]DailyClose, len(closes))
	copy(cp, closes)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.daily[symbol] = cp
	s.dailySrc[symbol] = source
}

// DailySource reports which provider produced the stored daily history for one
// symbol ("stooq"|"yahoo"), or "" when the symbol has no stored history. The
// handler combines the two legs of a correlation through combineSources so a
// coefficient built from two different providers ships as "mixed" instead of
// silently claiming one of them.
func (s *Store) DailySource(symbol string) string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.dailySourceLocked(symbol)
}

// dailySourceLocked is DailySource's body; callers hold s.mu.
func (s *Store) dailySourceLocked(symbol string) string {
	if len(s.daily[symbol]) == 0 {
		return ""
	}
	return s.dailySrc[symbol]
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
	return s.dailyCorrelationLocked(symA, symB)
}

// dailyCorrelationLocked is DailyCorrelation's body; callers hold s.mu.
func (s *Store) dailyCorrelationLocked(symA, symB string) (*float64, int) {
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

// DailyCorrelationWithSource is DailyCorrelation plus the provenance of the
// read, resolved under ONE lock acquisition.
//
// Why it exists: the handler needs coefficient, overlap count and both legs'
// source names to describe a single correlation. Reading those through
// separate calls takes separate RLocks, so a daily refresh landing in between
// could pair a coefficient computed from the OLD window with the NEW window's
// source name — a torn read in exactly the field whose whole job is to say
// truthfully where a number came from. One lock, one consistent answer.
//
// The source is CombineSources over the two legs, and is "" whenever no
// coefficient was produced (nothing to attribute).
func (s *Store) DailyCorrelationWithSource(symA, symB string) (*float64, int, string) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	coef, points := s.dailyCorrelationLocked(symA, symB)
	if coef == nil {
		return nil, points, ""
	}
	return coef, points, CombineSources(s.dailySourceLocked(symA), s.dailySourceLocked(symB))
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
