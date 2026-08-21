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

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"strings"
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
	defaultQuotesBudget = 20 * time.Second
	defaultDailyBudget  = 30 * time.Second
	defaultFngBudget    = 10 * time.Second

	// dailyRefreshEvery is the daily-history cadence: once a day per symbol
	// (checklist B2). Checked on the 3-min tick, so a due fetch lands within
	// one tick of the 24h boundary. On total failure the stamp is NOT advanced,
	// so the next tick retries instead of waiting a day on nothing.
	dailyRefreshEvery = 24 * time.Hour

	// dailyFetchCalendarDays bounds the d1..d2 request window: 30 trading days
	// for a 5-day-week symbol span ~42 calendar days; 60 adds holiday slack
	// while keeping the response tiny (~45 rows).
	dailyFetchCalendarDays = 60

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
	FngURL         string        // "" → defaultFngURL (override for tests)

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

	// ── stooq quotes: one GET per symbol (the batch garbles on the ^spx
	// caret), under the QUOTES budget only (review fix 13). ──
	okCount := 0
	quotesCtx, cancelQuotes := context.WithTimeout(ctx, budget(w.QuotesBudget, defaultQuotesBudget))
	for _, sym := range allSymbols {
		q, err := w.fetchQuote(quotesCtx, sym)
		if err != nil {
			// Network/timeout — store an N/D quote so the lamp honestly shows "—"
			// rather than retaining a stale value, and keep going.
			logger.Printf("macro: fetch %s failed (continuing): %v", sym, err)
			w.Store.SetQuote(Quote{Symbol: sym, OK: false})
			continue
		}
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
	for _, sym := range allSymbols {
		closes, err := w.fetchDailyCloses(ctx, sym, now)
		if err != nil {
			logger.Printf("macro: daily fetch %s failed (continuing): %v", sym, err)
			continue
		}
		if len(closes) == 0 {
			// A parsable body with zero usable recent rows (all N/D, or ancient
			// rows behind the age guard) — nothing honest to store.
			logger.Printf("macro: daily fetch %s returned no usable recent rows", sym)
			continue
		}
		w.Store.SetDailyCloses(sym, closes)
		stored++
	}
	if stored > 0 {
		w.lastDaily = now
	}
	logger.Printf("macro: daily history refreshed for %d/%d symbols", stored, len(allSymbols))
}

// fetchDailyCloses GETs one symbol's ranged daily CSV and returns its recent
// closes (date-ascending). Rows older than dailyMaxAgeDays are dropped — see
// the constant for why that guard exists.
func (w *Worker) fetchDailyCloses(ctx context.Context, symbol string, now time.Time) ([]DailyClose, error) {
	base := w.StooqDailyBase
	if base == "" {
		base = defaultStooqDailyBase
	}
	d2 := now.UTC()
	d1 := d2.AddDate(0, 0, -dailyFetchCalendarDays)
	u := base + "?s=" + url.QueryEscape(symbol) +
		"&d1=" + d1.Format("20060102") + "&d2=" + d2.Format("20060102") + "&i=d"

	body, err := w.httpGet(ctx, u, maxDailyBody)
	if err != nil {
		return nil, err
	}
	closes, err := ParseStooqDailyCSV(body)
	if err != nil {
		return nil, err
	}
	cutoff := d2.AddDate(0, 0, -dailyMaxAgeDays).Format("2006-01-02")
	recent := closes[:0]
	for _, c := range closes {
		if c.Date >= cutoff { // ISO dates compare lexicographically
			recent = append(recent, c)
		}
	}
	return recent, nil
}

// fetchQuote GETs one stooq symbol and parses it into a Quote. A 200 with an N/D
// body is a successful fetch returning Quote{OK:false} (nil error) — only a
// transport / non-200 / unreadable body is an error.
func (w *Worker) fetchQuote(ctx context.Context, symbol string) (Quote, error) {
	base := w.StooqBase
	if base == "" {
		base = defaultStooqBase
	}
	u := base + "?s=" + url.QueryEscape(symbol) + "&f=sd2t2ohlcv&e=csv"

	body, err := w.httpGet(ctx, u, maxQuoteBody)
	if err != nil {
		return Quote{}, err
	}
	q, perr := ParseStooqCSV(body)
	if perr != nil {
		return Quote{}, perr
	}
	// stooq echoes the symbol uppercased (^SPX); keep the id we requested so the
	// rest of the pipeline keys on the lowercase form.
	q.Symbol = symbol
	return q, nil
}

// fetchFnG GETs the alternative.me Fear & Greed endpoint and parses it.
func (w *Worker) fetchFnG(ctx context.Context) (FnG, error) {
	u := w.FngURL
	if u == "" {
		u = defaultFngURL
	}
	body, err := w.httpGet(ctx, u, maxQuoteBody)
	if err != nil {
		return FnG{}, err
	}
	return ParseFnG(body)
}

// httpGet performs a GET and returns the body bytes (capped at maxBody),
// erroring on a non-2xx status or any transport failure.
func (w *Worker) httpGet(ctx context.Context, u string, maxBody int64) ([]byte, error) {
	client := w.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: defaultHTTPTimeout}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	res, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
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
