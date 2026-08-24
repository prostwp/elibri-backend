package macro

// worker.go — periodic HTTP-poll loop for Macro Sentiment (mirrors
// whale.Worker; an HTTP ticker, NOT a push WS like funding).
//
// Lifecycle (mirrors whale.Worker.Run):
//   1. Run() does ONE warm-start refresh synchronously — guarantees the first
//      request to /api/v1/macro finds latest quotes per symbol (even if some
//      came back N/D).
//   2. Then loops on Interval ticks until ctx is cancelled.
//   3. Each tick is independent: a failure on one symbol (timeout, N/D) is
//      logged but never aborts the cycle. Best-effort by design — if stooq is
//      unreachable the Store stays empty → the handler serves an honest empty
//      payload and the frontend degrades on its own.
//
// There are NO LLM phases — the "AI" diagnosis is the pure compute.BuildDiagnosis
// string, not a model call.
//
// ── stooq symbol map (verbatim from dnevnik 2026-05-30 — do NOT "fix" these) ──
//
//	S&P 500 → ^spx     | VIX → vi.f      | Dollar DXY → dx.f
//	Gold    → xauusd   | US 10Y → 10yusy.b | BTC (24/7) → btcusd
//
// GOTCHA: the intuitive forms ^vix / ^dxy / ^tnx / 10usy.b all return N/D — do
// NOT use them. The working ones are vi.f / dx.f / 10yusy.b. Also: the multi-
// symbol batch (s=a,b,c) GARBLES when ^spx (a caret) is in the list, so we fetch
// ONE symbol per request (6 GETs/cycle, ~0.4s each).
//
// ── MULTI-SOURCE FALLBACK (2026-08-24) ────────────────────────────────────
//
// stooq is no longer reliable: the quote endpoint answers HTTP 404 with an
// HTML page and the daily endpoint answers with a JavaScript anti-bot
// challenge. Each symbol is therefore tried against an ORDERED list of
// providers (see source.go / stooq.go / yahoo.go) and the FIRST one that
// yields a usable row wins:
//
//	MACRO_SOURCE_ORDER=stooq,yahoo   (default) stooq first — it is the
//	                                 documented source and may recover — with
//	                                 Yahoo picking up per symbol when stooq
//	                                 yields nothing usable
//	MACRO_SOURCE_ORDER=yahoo         pin Yahoo, never touch stooq
//	MACRO_SOURCE_ORDER=yahoo,stooq   prefer Yahoo, keep stooq as the fallback
//
// Unknown names are logged and dropped; an empty/typo'd value falls back to
// the default order rather than leaving the worker with zero providers.
//
// HONESTY IS UNCHANGED. The fallback only adds attempts — it never softens a
// failure. When EVERY provider comes up empty for a symbol the store still
// receives a not-ok quote (carrying the last known date if any provider had
// one), the lamp still renders "—", and zero live lamps still classify as
// regime "unknown". What DID change is attribution: the provider that produced
// a value is recorded on the Quote and on the stored daily history, and ships
// as the additive "source" field on each lamp and correlation, so a mixed
// payload is always readable as such instead of being silently blended.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// defaultInterval is the poll cadence when Interval is zero. 6 GETs/cycle ×
	// ~0.4s ≈ 2.4s of work; the macro regime doesn't move faster (it's static on
	// weekends). 3 min → 20 cycles/h, 120 stooq GETs/h — polite to a public,
	// keyless source. (The frontend polls /api/v1/macro every 60s off the cached
	// latest, cheaper than waiting on a backend cycle.)
	defaultInterval = 3 * time.Minute

	// defaultStooqBase / defaultStooqDailyBase / defaultFngURL are the public,
	// keyless endpoints. The daily base serves historical CSV (Date,Open,High,
	// Low,Close,Volume) for the B2 correlation window.
	defaultStooqBase      = "https://stooq.com/q/l/"
	defaultStooqDailyBase = "https://stooq.com/q/d/l/"
	defaultFngURL         = "https://api.alternative.me/fng/?limit=1"

	// defaultHTTPTimeout caps a single HTTP request.
	defaultHTTPTimeout = 10 * time.Second

	// Per-phase context budgets (review fix 13): quotes, daily history and
	// F&G each run under their OWN timeout, so a hanging quote endpoint can
	// exhaust only its own phase — the daily window and the F&G overlay still
	// get their turn within the same cycle.
	// defaultQuotesBudget covers the whole quote phase; perAttemptTimeout caps
	// ONE provider call inside it. The arithmetic, with 6 symbols and the
	// default 2-provider order:
	//
	//	healthy      6 × ~0.4s (first provider answers)          ≈  2.4s
	//	stooq dead   6 × (0.3s 404 + 0.4s Yahoo)                 ≈  4.2s  (measured 2.9-5.0s live)
	//	stooq HANGS  6 × (10s timeout + 0.4s Yahoo)              ≈ 62s
	//
	// The hang case is why perAttemptTimeout exists: without it a provider that
	// never answers is bounded only by the phase budget, so ONE symbol could
	// consume it entirely and every later symbol would be cancelled before
	// being tried at all. With a 10s cap the cost is bounded per attempt and
	// the 45s budget carries ~4 symbols through a total stooq hang instead of
	// the ~2 that 20s allowed — the tail can still be cut short in that
	// worst case, but it degrades gradually and every symbol that IS reached
	// gets a real shot at the fallback. Symbols the budget does not reach store
	// an honest not-ok quote, exactly like any other failure.
	defaultQuotesBudget = 45 * time.Second

	// perAttemptTimeout caps a single provider call (see the budget note). It
	// matches defaultHTTPTimeout so a caller-supplied HTTPClient without its
	// own Timeout is still bounded.
	perAttemptTimeout  = 10 * time.Second
	defaultDailyBudget = 30 * time.Second
	defaultFngBudget   = 10 * time.Second

	// dailyRefreshEvery is the daily-history cadence: once a day per symbol
	// (checklist B2). Checked on the 3-min tick, so a due fetch lands within
	// one tick of the 24h boundary. On total failure the stamp is NOT advanced,
	// so the next tick retries instead of waiting a day on nothing.
	dailyRefreshEvery = 24 * time.Hour

	// dailyFetchCalendarDays bounds the stooq d1..d2 request window. It must be
	// wide enough to FILL dailyKeep on a 5-day-week symbol: 42 trading days
	// span ~59 calendar days before holidays, so the old 60 would have returned
	// barely 41 rows. 90 adds holiday slack, matches the Yahoo path's 3mo range
	// and the dailyMaxAgeDays guard, and still keeps the response small (~65
	// rows).
	dailyFetchCalendarDays = 90

	// quoteMaxAge is the freshness guard on a LATEST quote, and it exists
	// because the two sources fail in DIFFERENT shapes. stooq announces a dead
	// symbol explicitly (an N/D row) and the lamp goes not-ok on its own.
	// Yahoo's chart endpoint has no such sentinel: it simply returns the last
	// bar it holds, so a symbol whose feed froze would keep serving a weeks-old
	// close as ok:true — with a status, a delta and a vote in the composite.
	// That is exactly the "never overstate freshness" rule the scenario is
	// built on, so the age is checked here, uniformly, for EVERY source.
	//
	// 7 days cannot fire during any normal market closure: a weekend is 3 days
	// wall-clock from Friday's bar, a long weekend 4, and the longest US
	// holiday stretch (Christmas/New Year) about 5. Anything past a week is a
	// broken feed, not a closed market. An over-age quote is not discarded —
	// it still contributes its DATE, so the lamp keeps showing how stale the
	// market went instead of blanking to "".
	quoteMaxAge = 7 * 24 * time.Hour

	// dailyMaxAgeDays is the honesty guard on parsed daily rows: anything older
	// is dropped before storing. If stooq ever ignored d1/d2 and the body-size
	// cap truncated the RECENT tail away, the surviving ancient rows would
	// otherwise be served as "the last 30 days" — with the guard the symbol
	// degrades to an empty history (correlation ok:false) instead.
	dailyMaxAgeDays = 90

	// maxQuoteBody / maxDailyBody cap the response reads. Quote rows and the
	// F&G JSON are tiny (64 KiB is generous); the ranged daily CSV is ~45 rows
	// but gets 1 MiB of slack in case the range parameters are ignored.
	maxQuoteBody = 64 << 10
	maxDailyBody = 1 << 20
)

// allSymbols is the fetch order for one cycle. BTC is included (24/7) for the
// correlation ring even though it isn't a lamp.
var allSymbols = []string{SymSPX, SymVIX, SymDXY, SymGold, SymRates, SymBTC}

// Worker drives the periodic stooq + F&G poll cycle. Store is required; the rest
// default (Logger → log.Default, HTTPClient → 10s, Interval → 3min, URLs → the
// public endpoints). The zero value is NOT usable — Store must be set.
type Worker struct {
	Store          *Store
	Logger         *log.Logger   // nil → log.Default()
	HTTPClient     *http.Client  // nil → &http.Client{Timeout: 10s}
	Interval       time.Duration // 0 → defaultInterval
	StooqBase      string        // "" → defaultStooqBase (override for tests)
	StooqDailyBase string        // "" → defaultStooqDailyBase (override for tests)
	YahooBase      string        // "" → defaultYahooChartBase (override for tests)
	FngURL         string        // "" → defaultFngURL (override for tests)

	// SourceOrder pins the provider order for this Worker. nil → the
	// MACRO_SOURCE_ORDER env var → defaultSourceOrder. Tests set it directly
	// so they never depend on process env.
	SourceOrder []string

	// srcs is the resolved provider list, built once (sources()).
	srcs     []quoteSource
	srcsOnce sync.Once

	// Per-phase budgets; 0 → the defaults above (overridable for tests).
	QuotesBudget time.Duration
	DailyBudget  time.Duration
	FngBudget    time.Duration

	// lastDaily is the last SUCCESSFUL daily-history fetch (≥1 symbol stored).
	// Only refresh touches it, and refresh runs on Run's single goroutine — no
	// lock needed. Zero on boot → the warm-start cycle fetches immediately.
	lastDaily time.Time

	// now is injectable for deterministic once-a-day tests; nil → time.Now.
	now func() time.Time
}

// clock returns the injected clock or time.Now.
func (w *Worker) clock() time.Time {
	if w.now != nil {
		return w.now()
	}
	return time.Now()
}

// sources resolves the provider order ONCE and builds the matching
// quoteSource list. Resolution order: Worker.SourceOrder → the
// MACRO_SOURCE_ORDER env var → defaultSourceOrder. Unknown names are logged
// and dropped; a value that names no known provider degrades to the default
// rather than leaving the worker blind.
func (w *Worker) sources() []quoteSource {
	w.srcsOnce.Do(func() {
		logger := w.logger()

		order := w.SourceOrder
		if order == nil {
			var unknown []string
			order, unknown = ParseSourceOrder(os.Getenv(SourceOrderEnv))
			if len(unknown) > 0 {
				logger.Printf("macro: %s: ignoring unknown source(s) %s (known: %s, %s)",
					SourceOrderEnv, strings.Join(unknown, ","), SourceStooq, SourceYahoo)
			}
		}

		for _, name := range order {
			switch name {
			case SourceStooq:
				w.srcs = append(w.srcs, &stooqSource{
					base:      w.StooqBase,
					dailyBase: w.StooqDailyBase,
					get:       w.httpGet,
				})
			case SourceYahoo:
				w.srcs = append(w.srcs, &yahooSource{base: w.YahooBase, get: w.httpGet})
			}
		}
		logger.Printf("macro: source order = %s", strings.Join(sourceNames(w.srcs), " → "))
	})
	return w.srcs
}

// sourceNames renders a provider list for logging.
func sourceNames(srcs []quoteSource) []string {
	out := make([]string, 0, len(srcs))
	for _, s := range srcs {
		out = append(out, s.Name())
	}
	return out
}

// Run blocks until ctx is cancelled. Returns the ctx error on cancellation so
// callers can distinguish "shutdown requested" from "fatal worker bug".
//
// The warm-start refresh runs synchronously BEFORE the ticker; if it fails we
// log and continue (a flaky stooq response on boot must not crash the backend).
func (w *Worker) Run(ctx context.Context) error {
	if w.Store == nil {
		return fmt.Errorf("macro.Worker.Run: Store is nil")
	}
	interval := w.Interval
	if interval <= 0 {
		interval = defaultInterval
	}

	// Warm-start synchronously so the first /api/v1/macro request finds latest
	// quotes (refresh is best-effort — it logs its own errors, never returns one).
	w.refresh(ctx)

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			w.refresh(ctx)
		}
	}
}

// refresh runs one poll cycle: fetch the 6 symbols → SetQuote each → the daily
// history (only when due, once a day) → fetch F&G → SetFnG. Best-effort: every
// fetch/parse error is logged, never returned. (No error return at all — the
// loop is unconditional.)
func (w *Worker) refresh(ctx context.Context) {
	start := time.Now()
	logger := w.logger()

	budget := func(v, def time.Duration) time.Duration {
		if v > 0 {
			return v
		}
		return def
	}

	// ── quotes: one GET per symbol per provider (the stooq batch garbles on
	// the ^spx caret), under the QUOTES budget only (review fix 13). ──
	okCount := 0
	quotesCtx, cancelQuotes := context.WithTimeout(ctx, budget(w.QuotesBudget, defaultQuotesBudget))
	for _, sym := range allSymbols {
		q := w.fetchQuoteAnySource(quotesCtx, sym)
		if q.OK {
			// The lamp delta is the SESSION change (Close−Open), carried on the
			// quote itself.
			okCount++
		}
		w.Store.SetQuote(q)
	}
	cancelQuotes()

	// ── daily-close history for the 20-30d correlations (once a day), under
	// its OWN budget — slow quotes cannot starve it. ──
	dailyCtx, cancelDaily := context.WithTimeout(ctx, budget(w.DailyBudget, defaultDailyBudget))
	w.refreshDailyIfDue(dailyCtx)
	cancelDaily()

	// ── Fear & Greed (best-effort overlay; not folded into composite). ──
	fngCtx, cancelFng := context.WithTimeout(ctx, budget(w.FngBudget, defaultFngBudget))
	if f, err := w.fetchFnG(fngCtx); err != nil {
		logger.Printf("macro: fetch F&G failed (continuing): %v", err)
	} else if f.OK {
		w.Store.SetFnG(f)
	}
	cancelFng()

	logger.Printf("macro: refreshed %d/%d symbols, took %s",
		okCount, len(allSymbols), time.Since(start).Round(time.Millisecond))
}

// refreshDailyIfDue fetches the trailing daily-close window for every symbol
// when the last successful daily fetch is over dailyRefreshEvery old (or never
// happened). One GET per symbol, best-effort per symbol; the success stamp is
// advanced only when AT LEAST ONE symbol stored usable rows, so a fully dead
// source retries next tick instead of going dark for a day.
func (w *Worker) refreshDailyIfDue(ctx context.Context) {
	now := w.clock()
	if !w.lastDaily.IsZero() && now.Sub(w.lastDaily) < dailyRefreshEvery {
		return
	}
	logger := w.logger()
	stored := 0
	btcStored := false
	for _, sym := range allSymbols {
		closes, source := w.fetchDailyAnySource(ctx, sym, now)
		if len(closes) == 0 {
			// Every provider failed, or answered with zero usable recent rows
			// (all N/D, or ancient rows behind the age guard) — nothing honest
			// to store. The symbol keeps whatever history it already had; the
			// correlation gate (minDailyCorrPoints) decides what that is worth.
			logger.Printf("macro: daily %s — no usable recent rows from any source", sym)
			continue
		}
		w.Store.SetDailyCloses(sym, closes, source)
		stored++
		if sym == SymBTC {
			btcStored = true
		}
	}
	// The once-a-day stamp advances ONLY when BTC stored. Every correlation is
	// BTC↔X, so a cycle that missed BTC produced no usable correlation window
	// no matter how many other symbols succeeded — and BTC is fetched last, so
	// it is the leg a budget overrun drops first. Advancing on "any symbol
	// stored" would have parked all three correlations on a stale or empty BTC
	// leg for a full 24 hours; not advancing means the next 3-min tick retries.
	if btcStored {
		w.lastDaily = now
	} else {
		logger.Printf("macro: daily history NOT stamped — %s missing, every correlation "+
			"needs it; retrying next tick", SymBTC)
	}
	logger.Printf("macro: daily history refreshed for %d/%d symbols", stored, len(allSymbols))
}

// fetchQuoteAnySource tries each provider in order and returns the FIRST
// usable quote. It never returns an error — a symbol nobody can answer for is
// an honest not-ok Quote, which is exactly what the store and the lamp expect.
//
// What "usable" means: a Quote with OK true, carrying a timestamp, no older
// than quoteMaxAge. A provider that errors (transport, non-2xx, unparseable
// body), one that answers with an explicit "no data" row, and one that answers
// with a stale or undateable value all fall through to the next.
//
// LAST-KNOWN-DATE CARRY. stooq's N/D rows often still carry the last session's
// date, and that date is a real fact the lamp shows as `as_of` ("last time this
// symbol had data"). So when a provider answers not-ok BUT with a date, that
// dated quote is remembered; if every provider then comes up empty, it is what
// gets stored — preserving today's behaviour exactly. Only when nobody
// supplied even a date do we store the bare sentinel (the Store separately
// carries a previous date forward). A remembered not-ok quote NEVER carries a
// Source: it has no value to attribute.
func (w *Worker) fetchQuoteAnySource(ctx context.Context, symbol string) Quote {
	logger := w.logger()
	srcs := w.sources()
	now := w.clock()

	// remember keeps the first not-ok quote that carried a date, so a total
	// failure still ships the last known session date as the lamp's as_of.
	var dated Quote
	remember := func(q Quote) {
		if dated.AsOf.IsZero() && !q.AsOf.IsZero() {
			q.Symbol = symbol
			q.Price, q.Open, q.OK = 0, 0, false // a date is all that survives
			q.Source = ""                       // no value → nothing to attribute
			dated = q
		}
	}

	for _, src := range srcs {
		// Explicit per-attempt cap — see perAttemptTimeout. A provider that
		// hangs burns its own 10s and no more, so the phase budget is spent
		// across symbols rather than inside one of them.
		attemptCtx, cancel := context.WithTimeout(ctx, perAttemptTimeout)
		q, err := src.FetchQuote(attemptCtx, symbol)
		cancel()
		if err != nil {
			logger.Printf("macro: %s quote %s failed (continuing): %v", src.Name(), symbol, err)
			continue
		}
		if !q.OK {
			logger.Printf("macro: %s quote %s — no data row", src.Name(), symbol)
			remember(q)
			continue
		}
		// Freshness guard — see quoteMaxAge. A source with no timestamp at all
		// cannot be aged, and is rejected rather than trusted blindly.
		if q.AsOf.IsZero() {
			logger.Printf("macro: %s quote %s — value with no timestamp, rejected", src.Name(), symbol)
			continue
		}
		if age := now.Sub(q.AsOf); age > quoteMaxAge {
			logger.Printf("macro: %s quote %s — stale by %s (limit %s), rejected",
				src.Name(), symbol, age.Round(time.Hour), quoteMaxAge)
			remember(q)
			continue
		}
		q.Symbol = symbol
		if q.Source == "" {
			// Defensive: a source that forgot to stamp itself must not ship an
			// unattributed value.
			q.Source = src.Name()
		}
		return q
	}

	if !dated.AsOf.IsZero() {
		return dated
	}
	return Quote{Symbol: symbol, OK: false}
}

// fetchDailyAnySource tries each provider in order and returns the FIRST
// non-empty recent daily history plus the provider name behind it. Returns
// (nil, "") when nobody had usable rows.
//
// The dailyMaxAgeDays recency guard is applied HERE rather than inside a
// source, so every provider is held to the same rule: if a source ignored the
// requested window (or a body cap truncated the recent tail away), the
// surviving ancient rows are dropped and the symbol degrades to "no usable
// rows" instead of serving a stale window as "the last 30 days".
func (w *Worker) fetchDailyAnySource(ctx context.Context, symbol string, now time.Time) ([]DailyClose, string) {
	logger := w.logger()

	for _, src := range w.sources() {
		closes, err := src.FetchDaily(ctx, symbol, now)
		if err != nil {
			logger.Printf("macro: %s daily %s failed (continuing): %v", src.Name(), symbol, err)
			continue
		}
		recent := filterRecentCloses(closes, now)
		if len(recent) == 0 {
			logger.Printf("macro: %s daily %s returned no usable recent rows", src.Name(), symbol)
			continue
		}
		return recent, src.Name()
	}
	return nil, ""
}

// filterRecentCloses drops rows older than dailyMaxAgeDays relative to now.
// ISO dates compare lexicographically, so the cutoff is a plain string compare.
// The input slice is not reused — sources may hand out slices they retain.
func filterRecentCloses(closes []DailyClose, now time.Time) []DailyClose {
	cutoff := now.UTC().AddDate(0, 0, -dailyMaxAgeDays).Format("2006-01-02")
	recent := make([]DailyClose, 0, len(closes))
	for _, c := range closes {
		if c.Date >= cutoff {
			recent = append(recent, c)
		}
	}
	return recent
}

// fetchFnG GETs the alternative.me Fear & Greed endpoint and parses it.
func (w *Worker) fetchFnG(ctx context.Context) (FnG, error) {
	u := w.FngURL
	if u == "" {
		u = defaultFngURL
	}
	body, err := w.httpGet(ctx, u, maxQuoteBody, nil)
	if err != nil {
		return FnG{}, err
	}
	return ParseFnG(body)
}

// httpGet performs a GET and returns the body bytes (capped at maxBody),
// erroring on a non-2xx status or any transport failure. It is the single
// httpGetter shared by every quoteSource, so all providers inherit one client,
// one context and one 2xx-only rule — a 429/500 body that happens to parse can
// never be mistaken for live data.
//
// headers may be nil (stooq needs none); Yahoo passes a browser User-Agent
// because it answers 429 without one.
func (w *Worker) httpGet(ctx context.Context, u string, maxBody int64, headers map[string]string) ([]byte, error) {
	client := w.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: defaultHTTPTimeout}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	res, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		// Drain (bounded) so the keep-alive connection is reusable instead of
		// being torn down on every 404 — stooq currently 404s 6× a cycle.
		_, _ = io.Copy(io.Discard, io.LimitReader(res.Body, maxBody))
		return nil, fmt.Errorf("GET %s: status %d", u, res.StatusCode)
	}
	// io.ReadAll surfaces a mid-stream read error instead of silently truncating.
	body, err := io.ReadAll(io.LimitReader(res.Body, maxBody))
	if err != nil {
		return nil, fmt.Errorf("GET %s: read body: %w", u, err)
	}
	return body, nil
}

// logger returns the Worker's logger or log.Default() when nil.
func (w *Worker) logger() *log.Logger {
	if w.Logger != nil {
		return w.Logger
	}
	return log.Default()
}

// --- parsing (exported for worker_test.go) ---

// ParseStooqCSV parses one stooq `f=sd2t2ohlcv&e=csv` response: a header line
// plus one data row, fields in the order
//
//	Symbol,Date,Time,Open,High,Low,Close,Volume
//
// An N/D quote (Close=="N/D" or Date=="N/D", or a fully-empty
// "<sym>,N/D,N/D,…" row) returns Quote{OK:false} with a NIL error — N/D is a
// valid "no data" response, not a parse failure. A malformed Close float also
// yields OK:false (no panic). An error is returned only when the body has no
// usable data row at all.
//
// The row DATE survives an N/D close: stooq often keeps the last session's
// date on a value-less row, and that date is a real fact ("last time this
// symbol had data") — it ships as the lamp's as_of even when the value is N/D,
// so the frontend/bot can say HOW stale instead of showing nothing.
func ParseStooqCSV(data []byte) (Quote, error) {
	// Strip CR first so a CRLF body's trailing "\r" can't survive on the last
	// field (Volume) or split a line oddly; then trim surrounding whitespace.
	text := strings.ReplaceAll(string(data), "\r", "")
	text = strings.TrimSpace(text)
	if text == "" {
		return Quote{}, fmt.Errorf("ParseStooqCSV: empty body")
	}
	lines := strings.Split(text, "\n")
	if len(lines) < 2 {
		return Quote{}, fmt.Errorf("ParseStooqCSV: no data row (got %d lines)", len(lines))
	}
	// The data row is the second non-empty line.
	row := strings.TrimSpace(lines[1])
	fields := strings.Split(row, ",")
	if len(fields) < 7 {
		return Quote{}, fmt.Errorf("ParseStooqCSV: short row (%d fields)", len(fields))
	}

	symbol := strings.TrimSpace(fields[0])
	dateStr := strings.TrimSpace(fields[1])
	timeStr := strings.TrimSpace(fields[2])
	openStr := strings.TrimSpace(fields[3])
	closeStr := strings.TrimSpace(fields[6])

	q := Quote{Symbol: symbol}
	// Timestamp best-effort FIRST, so an N/D-close row still carries its date.
	// stooq is UTC. An N/D time next to a valid date degrades to midnight of
	// that date (still an honest "last known" stamp); an unparseable ts leaves
	// AsOf zero.
	if !isND(dateStr) {
		if t, terr := time.ParseInLocation("2006-01-02 15:04:05", dateStr+" "+timeStr, time.UTC); terr == nil {
			q.AsOf = t
		} else if t, terr := time.ParseInLocation("2006-01-02", dateStr, time.UTC); terr == nil {
			q.AsOf = t
		}
	}

	// N/D sentinel on the close (or a fully dated-out row) → "no data" value,
	// not an error — but the parsed date above stays on the quote.
	if isND(dateStr) || isND(closeStr) {
		return q, nil
	}

	price, perr := strconv.ParseFloat(closeStr, 64)
	if perr != nil || !isFinite(price) {
		// Unparseable OR non-finite close (strconv accepts "NaN"/"Inf"!) →
		// treat as N/D (honest "—"), never an OK quote carrying a value the
		// JSON encoder cannot serialize (review fix 4).
		return q, nil
	}

	q.Price = price
	q.OK = true
	// Open: the session baseline for the lamp delta. Same N/D + float tolerance
	// as Close — an N/D, unparseable or non-finite Open leaves Open=0 (the
	// handler then emits delta_pct:null rather than fabricating a move). A
	// valid Close with a bad Open is still an OK quote (the lamp shows its
	// value, just no direction).
	if !isND(openStr) {
		if o, oerr := strconv.ParseFloat(openStr, 64); oerr == nil && isFinite(o) {
			q.Open = o
		}
	}
	return q, nil
}

// isFinite rejects NaN and ±Inf — strconv.ParseFloat parses them happily, and
// a non-finite number must always degrade to "no data" (review fix 4).
func isFinite(f float64) bool {
	return !math.IsNaN(f) && !math.IsInf(f, 0)
}

// isND reports whether a stooq field is the "no data" sentinel.
func isND(s string) bool {
	return strings.EqualFold(strings.TrimSpace(s), "N/D")
}

// ParseStooqDailyCSV parses a stooq ranged daily-history CSV
// (https://stooq.com/q/d/l/?s=SYM&d1=…&d2=…&i=d):
//
//	Date,Open,High,Low,Close,Volume
//	2026-07-21,117433.94,119482.98,116215.98,117294.65,...
//
// one row per trading day, dates ascending. Volume is absent for some symbols
// (indices), so only Date + Close are required. Rows with an unparseable date
// or close (or the N/D sentinel) are skipped — a partial history is still an
// honest history; the store/correlation layer enforces the minimum-points
// gate. An error is returned only when the body carries no header+row
// structure at all (the "symbol unknown" plain-text response lands here).
func ParseStooqDailyCSV(data []byte) ([]DailyClose, error) {
	text := strings.ReplaceAll(string(data), "\r", "")
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, fmt.Errorf("ParseStooqDailyCSV: empty body")
	}
	lines := strings.Split(text, "\n")
	if len(lines) < 2 {
		return nil, fmt.Errorf("ParseStooqDailyCSV: no data rows (got %d lines)", len(lines))
	}
	if !strings.HasPrefix(strings.ToLower(lines[0]), "date,") {
		return nil, fmt.Errorf("ParseStooqDailyCSV: unexpected header %q", lines[0])
	}
	out := make([]DailyClose, 0, len(lines)-1)
	for _, line := range lines[1:] {
		fields := strings.Split(strings.TrimSpace(line), ",")
		if len(fields) < 5 {
			continue
		}
		dateStr := strings.TrimSpace(fields[0])
		closeStr := strings.TrimSpace(fields[4])
		if isND(dateStr) || isND(closeStr) {
			continue
		}
		if _, err := time.ParseInLocation("2006-01-02", dateStr, time.UTC); err != nil {
			continue
		}
		price, err := strconv.ParseFloat(closeStr, 64)
		if err != nil || !isFinite(price) {
			continue // non-finite closes are not data (review fix 4)
		}
		out = append(out, DailyClose{Date: dateStr, Close: price})
	}
	return out, nil
}

// fngRaw matches the alternative.me /fng response element.
//
//	{"data":[{"value":"54","value_classification":"Greed","timestamp":"…"}]}
type fngRaw struct {
	Data []struct {
		Value          string `json:"value"`
		ValueClassName string `json:"value_classification"`
	} `json:"data"`
}

// ParseFnG parses the alternative.me Fear & Greed response into an FnG. An empty
// data array → FnG{OK:false} (nil error). A non-integer value → OK:false. A hard
// JSON error is returned as err.
func ParseFnG(data []byte) (FnG, error) {
	var raw fngRaw
	if err := json.Unmarshal(data, &raw); err != nil {
		return FnG{}, err
	}
	if len(raw.Data) == 0 {
		return FnG{OK: false}, nil
	}
	d := raw.Data[0]
	v, err := strconv.Atoi(strings.TrimSpace(d.Value))
	if err != nil {
		return FnG{OK: false}, nil
	}
	return FnG{
		Value: v,
		Label: strings.TrimSpace(d.ValueClassName),
		OK:    true,
	}, nil
}
