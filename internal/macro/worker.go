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

	// defaultStooqBase / defaultFngURL are the public, keyless endpoints.
	defaultStooqBase = "https://stooq.com/q/l/"
	defaultFngURL    = "https://api.alternative.me/fng/?limit=1"

	// defaultHTTPTimeout caps a single HTTP request.
	defaultHTTPTimeout = 10 * time.Second

	// fetcherCtxTimeout caps one full fan-out cycle (6 symbols + F&G). 30s = 3×
	// the per-request timeout — slack for the slowest while keeping the next tick
	// on schedule (mirrors whale's defaultFetcherCtxTimeout).
	fetcherCtxTimeout = 30 * time.Second
)

// allSymbols is the fetch order for one cycle. BTC is included (24/7) for the
// correlation ring even though it isn't a lamp.
var allSymbols = []string{SymSPX, SymVIX, SymDXY, SymGold, SymRates, SymBTC}

// Worker drives the periodic stooq + F&G poll cycle. Store is required; the rest
// default (Logger → log.Default, HTTPClient → 10s, Interval → 3min, URLs → the
// public endpoints). The zero value is NOT usable — Store must be set.
type Worker struct {
	Store      *Store
	Logger     *log.Logger   // nil → log.Default()
	HTTPClient *http.Client  // nil → &http.Client{Timeout: 10s}
	Interval   time.Duration // 0 → defaultInterval
	StooqBase  string        // "" → defaultStooqBase (override for tests)
	FngURL     string        // "" → defaultFngURL (override for tests)
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

// refresh runs one poll cycle: fetch the 6 symbols → SetQuote each → AddPoint
// the OK prices → fetch F&G → SetFnG. Best-effort: every fetch/parse error is
// logged, never returned. (No error return at all — the loop is unconditional.)
func (w *Worker) refresh(ctx context.Context) {
	start := time.Now()
	logger := w.logger()

	fetchCtx, cancel := context.WithTimeout(ctx, fetcherCtxTimeout)
	defer cancel()

	// ── stooq: one GET per symbol (the batch garbles on the ^spx caret). ──
	okCount := 0
	prices := make(map[string]float64, len(allSymbols))
	for _, sym := range allSymbols {
		q, err := w.fetchQuote(fetchCtx, sym)
		if err != nil {
			// Network/timeout — store an N/D quote so the lamp honestly shows "—"
			// rather than retaining a stale value, and keep going.
			logger.Printf("macro: fetch %s failed (continuing): %v", sym, err)
			w.Store.SetQuote(Quote{Symbol: sym, OK: false})
			continue
		}
		if q.OK {
			// The lamp delta is the SESSION change (Close−Open), carried on the
			// quote itself — no ring lookup needed (the ring is correlations-only).
			okCount++
			prices[sym] = q.Price
		}
		w.Store.SetQuote(q)
	}
	// One rolling point per cycle (only the OK prices of this cycle).
	w.Store.AddPoint(prices)

	// ── Fear & Greed (best-effort overlay; not folded into composite). ──
	if f, err := w.fetchFnG(fetchCtx); err != nil {
		logger.Printf("macro: fetch F&G failed (continuing): %v", err)
	} else if f.OK {
		w.Store.SetFnG(f)
	}

	logger.Printf("macro: refreshed %d/%d symbols, %d points in ring, took %s",
		okCount, len(allSymbols), w.Store.PointCount(), time.Since(start).Round(time.Millisecond))
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

	body, err := w.httpGet(ctx, u)
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
	body, err := w.httpGet(ctx, u)
	if err != nil {
		return FnG{}, err
	}
	return ParseFnG(body)
}

// httpGet performs a GET and returns the body bytes, erroring on a non-2xx
// status or any transport failure.
func (w *Worker) httpGet(ctx context.Context, u string) ([]byte, error) {
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
	// stooq rows + the F&G JSON are tiny; cap the read defensively at 64 KiB.
	// io.ReadAll surfaces a mid-stream read error instead of silently truncating.
	const maxBody = 64 << 10
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
	if perr != nil {
		// Unparseable close → treat as N/D (honest "—"), not a hard error.
		return q, nil
	}

	q.Price = price
	q.OK = true
	// Open: the session baseline for the lamp delta. Same N/D + float tolerance
	// as Close — an N/D or unparseable Open leaves Open=0 (the handler then emits
	// delta_pct:null rather than fabricating a move). A valid Close with a bad
	// Open is still an OK quote (the lamp shows its value, just no direction).
	if !isND(openStr) {
		if o, oerr := strconv.ParseFloat(openStr, 64); oerr == nil {
			q.Open = o
		}
	}
	return q, nil
}

// isND reports whether a stooq field is the "no data" sentinel.
func isND(s string) bool {
	return strings.EqualFold(strings.TrimSpace(s), "N/D")
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
