package demobot

// httpapi.go — read-only HTTP JSON API over the SAME agent card builders the
// Telegram bot uses, so the team can test agents without Telegram.
//
// Contract (docs/demobot-http.md is the user-facing reference):
//   - GET only, JSON only, CORS open (public read-only demo).
//   - Every agent endpoint answers with one envelope shape; the exact
//     Telegram HTML card rides along in card_html.
//   - Zero duplicated agent logic: handlers call the exact functions the bot
//     dispatches to (MacroCard, TrendCard, gather + renderDigestHTML, …), on
//     the same Agents instance — so the kline cache and the 5-minute AI memo
//     are shared with the Telegram path.
//   - Honesty rules carry over: a dead upstream is an explicit 503, never
//     fake numbers; degraded parts inside a 200 card say so in the facts.
//   - Machine-readable status: every envelope carries ok/reason (and the 503
//     body carries the same pair) so templates branch on WHY a reading is
//     absent; level-bearing agents add a raw-precision "levels" object.
//   - Global in-memory token bucket: 10 req/sec across all clients → 429
//     with Retry-After: 1.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"log"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	// DefaultHTTPAddr is the bind address when DEMOBOT_HTTP_ADDR is empty.
	DefaultHTTPAddr = "127.0.0.1:8090"

	httpRatePerSec = 10 // global request budget, tokens per second
	httpBurst      = 10 // bucket capacity

	disclaimerText = "Analytics, not financial advice"
)

// httpAgentNames is the full routing table in /agents listing order
// (mirrors the bot's menu grid).
var httpAgentNames = []string{
	keyDigest, keyTop, keyFX, keyMacro, keyWhale, keyFunding,
	keyMomentum, keyTrend, keySR, keyVol, keyRisk, keyNews, keyGold,
}

// assetAgents take the optional ?asset= parameter; every other agent
// rejects it so nobody mistakes a global read for an asset-specific one.
var assetAgents = map[string]bool{keyMomentum: true, keyTrend: true, keySR: true, keyVol: true}

// ── envelope ─────────────────────────────────────────────────────────────────

// httpEnvelope is the single response shape every agent endpoint serves.
type httpEnvelope struct {
	Agent string `json:"agent"`
	Asset string `json:"asset"`
	// OK is true when the agent produced a real reading. False for every
	// degraded state — Reason then names which one (the enum below), so
	// templates branch on WHY a value is absent instead of sniffing verdicts.
	OK bool `json:"ok"`
	// Reason is null when OK; otherwise one of: "market_closed",
	// "source_offline", "insufficient_history", "below_threshold", "no_data"
	// (cardStatus.reason — documented in docs/demobot-http.md).
	Reason    *string  `json:"reason"`
	Verdict   string   `json:"verdict"`
	Semaphore string   `json:"semaphore"` // bullish | bearish | neutral
	Facts     []string `json:"facts"`
	// Levels is the machine-readable numeric companion for level-bearing
	// agents: trend {"invalidation"}, sr {"supports","resistances"}, vol
	// {"expansion_ratio"} — raw precision floats. Absent for other agents.
	Levels any `json:"levels,omitempty"`
	// Results is the per-asset outcome array of momentum, fx and funding
	// cards ({"asset","ok","reason"}, plus the read itself on ok entries — see
	// AssetResult) — absent for every other agent.
	Results []AssetResult `json:"results,omitempty"`
	// Blocks is the content-ready sentence set (trend, S/R cards that show at
	// least one level, the global macro card, the single-asset momentum card,
	// vol, funding, gold) — absent for every other agent, on degraded cards and
	// on the S/R "no significant levels" finding.
	Blocks *ContentBlocks `json:"blocks,omitempty"`
	// Macro is the macro cards' machine readout (rule score, bands, per-lamp
	// contributions, freshness, Fear & Greed age) — absent for every other
	// agent and on macro cards without a reading (unknown / offline).
	Macro *MacroReadout `json:"macro,omitempty"`
	// Funding is the funding card's machine readout (additive 2026-09-15) —
	// absent for every other agent and on the all-offline funding card.
	Funding *FundingReadout `json:"funding,omitempty"`
	// Gold is the gold card's machine readout (additive 2026-09-15): per-part
	// stamps daily_as_of / price_as_of / macro_as_of, the 1h price and its
	// freshness, the day range, the macro lamp counts — absent elsewhere.
	Gold *GoldReadout `json:"gold,omitempty"`
	// Whale is the whale card's machine readout (additive 2026-09-15): state,
	// count, threshold, window, the listed transactions — absent elsewhere
	// and on the offline 503.
	Whale *WhaleReadout `json:"whale,omitempty"`
	// Narrative is the narrative radar's machine readout (additive
	// 2026-09-15): state, windows, threshold and where it is checked, the
	// listed themes, sources — absent elsewhere and on the 503.
	Narrative *NarrativeReadout `json:"narrative,omitempty"`
	// Risk is the risk calculator's machine readout (additive 2026-09-15):
	// formula, inputs, raw result in abstract units, the shown numbers,
	// applicability, calculated_at — absent elsewhere.
	Risk       *RiskReadout `json:"risk,omitempty"`
	Confidence *int         `json:"confidence"`         // 0-100, null when the source gave none; news: data quality
	AIText     *string      `json:"ai_text"`            // plain-text AI block, null when absent
	Sections   []string     `json:"sections,omitempty"` // digest only: the one-liners
	// Digest is the digest's own machine readout (digest only, additive
	// 2026-09-15): unified status, how the highlighted card was selected and
	// every section card_html renders (FX and narrative included), each with
	// its own data time.
	Digest     *DigestReadout `json:"digest,omitempty"`
	DataAsOf   string         `json:"data_as_of"` // RFC3339, same stamp as the card footer
	Disclaimer string         `json:"disclaimer"`
	CardHTML   string         `json:"card_html"` // the exact Telegram HTML card
}

// DigestReadout is the envelope's "digest" object (docs/demobot-http.md
// "Digest readout").
type DigestReadout struct {
	Status          string          `json:"status"` // live | partial | degraded — same value as /showcase digest_status
	LiveSections    int             `json:"live_sections"`
	TotalSections   int             `json:"total_sections"`   // seven digest agents + the FX block
	DegradedSources []string        `json:"degraded_sources"` // section keys, [] when none
	GeneratedAt     string          `json:"generated_at"`     // sweep time
	Selection       DigestSelection `json:"selection"`
	Sections        []DigestSection `json:"sections"` // every block below the highlighted card, render order
}

// DigestSelection says how the highlighted card was chosen.
type DigestSelection struct {
	State             string            `json:"state"`  // selected | no_highlight
	Rule              string            `json:"rule"`   // macro_risk_off | strongest_confirmed | fallback_unconfirmed | fallback_macro
	Winner            string            `json:"winner"` // agent key of the highlighted card
	Line              string            `json:"line"`   // the one-line reason shown in card_html
	HighlightOK       bool              `json:"highlight_ok"`
	HighlightReason   *string           `json:"highlight_reason"`
	HighlightDataAsOf string            `json:"highlight_data_as_of"`
	MacroRegime       string            `json:"macro_regime"`
	MacroRiskOffGate  string            `json:"macro_risk_off_gate,omitempty"` // only when the regime is risk_off
	ScalesCalibrated  bool              `json:"scales_calibrated"`             // always false for now
	Candidates        []DigestCandidate `json:"candidates"`                    // funding, momentum, trend
}

// DigestCandidate is one priority agent's standing in the sweep.
type DigestCandidate struct {
	Agent         string  `json:"agent"`
	Eligible      bool    `json:"eligible"`
	Excluded      *string `json:"excluded"`  // degraded | no_ranked_read | stale | no_data_time; null when eligible
	Confirmed     bool    `json:"confirmed"` // the agent's own rule committed to a finding
	Score         int     `json:"score"`     // 0..100, comparable only within one tier
	DataAsOf      *string `json:"data_as_of"`
	MaxAgeMinutes int     `json:"max_age_minutes"` // freshness limit for the top slot
}

// DigestSection is one rendered block below the highlighted card.
type DigestSection struct {
	Key      string   `json:"key"`   // macro | whale | funding | momentum | trend | sr | vol | fx | narrative
	Title    *string  `json:"title"` // plain header above the lines (FX), null when none
	Lines    []string `json:"lines"` // plain text, exactly as in card_html
	OK       bool     `json:"ok"`
	Reason   *string  `json:"reason"`
	DataAsOf *string  `json:"data_as_of"` // null without a data time (offline, unstamped)
}

func rfcPtr(t time.Time) *string {
	if t.IsZero() {
		return nil
	}
	s := t.UTC().Format(time.RFC3339)
	return &s
}

func reasonPtr(st cardStatus) *string {
	if st == statusOK {
		return nil
	}
	r := st.reason()
	return &r
}

// digestReadout builds the "digest" object from the same sweep, selection and
// section list the HTML renders.
func digestReadout(g gathered, p topPick, top Card) *DigestReadout {
	h := g.health()
	out := &DigestReadout{
		Status:          h.Status,
		LiveSections:    h.Live,
		TotalSections:   h.Total,
		DegradedSources: append([]string{}, h.Degraded...),
		GeneratedAt:     p.At.UTC().Format(time.RFC3339),
		Sections:        []DigestSection{},
	}
	state := "selected"
	if p.NoHighlight {
		state = "no_highlight"
	}
	sel := DigestSelection{
		State: state, Rule: p.Rule, Winner: p.Winner, Line: selectionLine(p),
		HighlightOK:       top.effectiveStatus() == statusOK,
		HighlightReason:   reasonPtr(top.effectiveStatus()),
		HighlightDataAsOf: top.DataTime.UTC().Format(time.RFC3339),
		MacroRegime:       p.MacroRegime,
		MacroRiskOffGate:  p.MacroGate,
		Candidates:        []DigestCandidate{},
	}
	for _, c := range p.Candidates {
		dc := DigestCandidate{
			Agent: c.Key, Eligible: c.Eligible, Confirmed: c.Confirmed, Score: c.Score,
			DataAsOf: rfcPtr(c.AsOf), MaxAgeMinutes: int(c.MaxAge / time.Minute),
		}
		if c.Excluded != "" {
			ex := c.Excluded
			dc.Excluded = &ex
		}
		sel.Candidates = append(sel.Candidates, dc)
	}
	out.Selection = sel
	for _, s := range digestSections(g, p.Winner, p.At) {
		ds := DigestSection{Key: s.key, OK: s.status == statusOK, Reason: reasonPtr(s.status), DataAsOf: rfcPtr(s.asOf)}
		if s.title != "" {
			t := htmlToPlain(s.title)
			ds.Title = &t
		}
		for _, l := range s.lines {
			ds.Lines = append(ds.Lines, htmlToPlain(l))
		}
		out.Sections = append(out.Sections, ds)
	}
	return out
}

// semaphoreOf maps the card emoji contract to the JSON semaphore words.
func semaphoreOf(emoji string) string {
	switch emoji {
	case emojiBull:
		return "bullish"
	case emojiBear:
		return "bearish"
	default:
		return "neutral"
	}
}

// htmlToPlain strips tags and unescapes entities — used to turn pre-rendered
// Telegram HTML fragments (AI blocks, one-liners) into JSON-friendly text.
func htmlToPlain(s string) string {
	var b strings.Builder
	inTag := false
	for _, r := range s {
		switch {
		case r == '<':
			inTag = true
		case r == '>':
			inTag = false
		case !inTag:
			b.WriteRune(r)
		}
	}
	return strings.TrimSpace(html.UnescapeString(b.String()))
}

// cardEnvelope converts one Card into the HTTP envelope. The card is the
// single source of truth — nothing here re-derives agent data: ok/reason come
// from the status the builder threaded through, levels from its computed
// structs, never from parsing rendered text.
func cardEnvelope(c Card) httpEnvelope {
	st := c.effectiveStatus()
	env := httpEnvelope{
		Agent:      c.Agent,
		Asset:      c.assetKey(),
		OK:         st == statusOK,
		Verdict:    c.Verdict,
		Semaphore:  semaphoreOf(c.Emoji),
		Facts:      append([]string{}, c.Facts...), // [] not null when empty
		Levels:     c.Levels,
		Results:    c.Results,
		Blocks:     c.Blocks,
		Macro:      c.Macro,
		Funding:    c.Funding,
		Gold:       c.Gold,
		Whale:      c.Whale,
		Narrative:  c.Narrative,
		Risk:       c.Risk,
		DataAsOf:   c.DataTime.UTC().Format(time.RFC3339),
		Disclaimer: disclaimerText,
		CardHTML:   c.RenderHTML(),
	}
	if !env.OK {
		r := st.reason()
		env.Reason = &r
	}
	if c.Confidence != nil {
		v := clampInt(*c.Confidence, 0, 100)
		env.Confidence = &v
	}
	if c.AIHTML != "" {
		txt := htmlToPlain(c.AIHTML)
		env.AIText = &txt
	}
	return env
}

// ── global rate limit: token bucket ──────────────────────────────────────────

type tokenBucket struct {
	mu     sync.Mutex
	rate   float64 // tokens per second
	burst  float64
	tokens float64
	last   time.Time
}

func newTokenBucket(rate, burst float64) *tokenBucket {
	return &tokenBucket{rate: rate, burst: burst, tokens: burst, last: time.Now()}
}

// allowAt refills by elapsed time and takes one token when available.
// Split from allow() so tests drive it with fixed clocks.
func (b *tokenBucket) allowAt(now time.Time) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if dt := now.Sub(b.last).Seconds(); dt > 0 {
		b.tokens = math.Min(b.burst, b.tokens+dt*b.rate)
		b.last = now
	}
	if b.tokens >= 1 {
		b.tokens--
		return true
	}
	return false
}

func (b *tokenBucket) allow() bool { return b.allowAt(time.Now()) }

// ── server ───────────────────────────────────────────────────────────────────

// HTTPServer is the read-only JSON API. It runs alongside Telegram long
// polling on the same Agents instance and shares its caches.
type HTTPServer struct {
	ag  *Agents
	srv *http.Server
	lim *tokenBucket
	ln  net.Listener
	// mux is the route table WITHOUT wrap() (no rate limit, no CORS): the push
	// hook (hook.go) calls it in-process so its events carry exactly the GET body.
	mux http.Handler
	// sc memoizes the whole landing-showcase sweep (60s, singleflight) so a
	// landing page render never fires twelve uncached upstream calls — see
	// showcase.go.
	sc showcaseMemo
}

// NewHTTPServer builds the read-only JSON API. A global token bucket
// (10 req/s) is on by default; set DEMOBOT_RATE_LIMIT=0 to disable it — e.g.
// a server deployment fronted by a trusted reverse proxy that does its own
// limiting. When disabled, lim is nil and wrap() skips the 429 gate entirely.
func NewHTTPServer(addr string, ag *Agents) *HTTPServer {
	s := &HTTPServer{ag: ag, lim: newTokenBucket(httpRatePerSec, httpBurst)}
	if os.Getenv("DEMOBOT_RATE_LIMIT") == "0" {
		s.lim = nil
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleRoot)
	mux.HandleFunc("/agents", s.handleList)
	mux.HandleFunc("/agents/", s.handleAgent)
	// Exact path, so it outranks the "/agents/" subtree without touching
	// handleAgent (which still 404s any other nested path) — trendchart.go.
	mux.HandleFunc("/agents/trend/chart", s.handleTrendChart)
	// Landing showcase (showcase.go): the catalog the marketing page renders
	// and the one ready-to-render "what you get" story.
	mux.HandleFunc("/showcase", s.handleShowcase)
	mux.HandleFunc("/showcase/example", s.handleShowcaseExample)
	s.mux = mux
	s.srv = &http.Server{
		Addr:              addr,
		Handler:           s.wrap(mux),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		// /agents/digest can legitimately run ~digestBudget plus one AI call;
		// the write timeout must sit safely above that, never below.
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}
	return s
}

// Start binds synchronously — a bind failure comes back to the caller so
// main can treat it as fatal — then serves in the background.
func (s *HTTPServer) Start() error {
	ln, err := net.Listen("tcp", s.srv.Addr)
	if err != nil {
		return err
	}
	s.ln = ln
	go func() {
		if err := s.srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("[demobot] http api serve error: %v", err)
		}
	}()
	log.Printf("[demobot] HTTP API listening on http://%s (read-only, see docs/demobot-http.md)", ln.Addr())
	return nil
}

// Addr reports the bound address ("" before Start) — handy with ":0".
func (s *HTTPServer) Addr() string {
	if s.ln == nil {
		return ""
	}
	return s.ln.Addr().String()
}

// Shutdown drains in-flight requests until ctx expires.
func (s *HTTPServer) Shutdown(ctx context.Context) error { return s.srv.Shutdown(ctx) }

// wrap is the outer middleware: CORS on every response (this is a public
// read-only demo), then the global rate limit, then the GET-only gate.
func (s *HTTPServer) wrap(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		if s.lim != nil && !s.lim.allow() {
			w.Header().Set("Retry-After", "1")
			writeErr(w, http.StatusTooManyRequests, "rate limit exceeded — max 10 requests/second globally, retry in 1s")
			return
		}
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			writeErr(w, http.StatusMethodNotAllowed, "GET only — this API is read-only")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// ── JSON writing ─────────────────────────────────────────────────────────────

// encodeJSON marshals without HTML escaping so card_html stays readable:
// literal angle-bracket tags instead of \uXXXX escapes. Safe: responses are
// application/json with nosniff, never served as HTML.
func encodeJSON(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// writeJSONAt is writeJSON plus the caching contract for a response whose
// content has a known data time.
//
// Last-Modified carries the SAME instant as the body's data_as_of: the close
// of the bar the reading came from, not the moment we answered. That is the
// only value a cache can act on — answering time changes on every request and
// would make every response look new.
//
// A client that sends If-Modified-Since gets 304 with no body when nothing has
// changed since. Second granularity is HTTP's, and our data times are bar
// closes on the minute, so truncation costs nothing here.
//
// Deliberately NOT applied to degraded responses: see writeCard. Caching a
// failure keeps an agent dark long after its source recovers, which is a worse
// outcome than re-asking.
func writeJSONAt(w http.ResponseWriter, r *http.Request, status int, dataTime time.Time, v any) {
	if dataTime.IsZero() {
		writeJSON(w, status, v)
		return
	}
	mod := dataTime.UTC().Truncate(time.Second)
	w.Header().Set("Last-Modified", mod.Format(http.TimeFormat))
	if notModifiedSince(r, mod) {
		// RFC 9110: a 304 carries no body and repeats the validators.
		w.WriteHeader(http.StatusNotModified)
		return
	}
	writeJSON(w, status, v)
}

// notModifiedSince reports whether the client already holds this version.
// An unparseable header is treated as absent — a malformed date must never
// suppress real data.
func notModifiedSince(r *http.Request, mod time.Time) bool {
	if r == nil {
		return false
	}
	raw := r.Header.Get("If-Modified-Since")
	if raw == "" {
		return false
	}
	since, err := http.ParseTime(raw)
	if err != nil {
		return false
	}
	return !mod.After(since.UTC())
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	b, err := encodeJSON(v)
	if err != nil { // practically unreachable: all payloads are plain structs
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"internal encoding failure"}`))
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_, _ = w.Write(b)
}

func writeErr(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

// ── handlers ─────────────────────────────────────────────────────────────────

func (s *HTTPServer) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		writeErr(w, http.StatusNotFound, "not found — GET /agents lists the available endpoints")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{
		"service":          "AlphaVizor demo bot — read-only HTTP JSON API",
		"agents":           "/agents",
		"showcase":         "/showcase",
		"showcase_example": "/showcase/example",
		"disclaimer":       disclaimerText,
	})
}

// httpAgentInfo is one /agents listing row.
type httpAgentInfo struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Assets      []string `json:"assets,omitempty"`
	Examples    []string `json:"examples"`
}

// assetKeys returns the canonical asset arguments, sorted for determinism.
func assetKeys() []string {
	keys := make([]string, 0, len(assetTable))
	for k := range assetTable {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func (s *HTTPServer) handleList(w http.ResponseWriter, _ *http.Request) {
	items := make([]httpAgentInfo, 0, len(httpAgentNames))
	for _, name := range httpAgentNames {
		info := httpAgentInfo{
			Name:        name,
			Description: howTexts[name],
			Examples:    []string{"/agents/" + name},
		}
		if assetAgents[name] {
			info.Assets = assetKeys()
			info.Examples = append(info.Examples, "/agents/"+name+"?asset=eurusd")
		}
		if name == keyTrend {
			// The chart companion (trendchart.go): same read, drawing data.
			info.Examples = append(info.Examples, "/agents/trend/chart?asset=eurusd")
		}
		if name == keyMomentum {
			// B1: user-configured scan + timeframe.
			info.Examples = append(info.Examples, "/agents/momentum?assets=btc,eurusd,gold&tf=1d")
		}
		if name == keyMacro {
			// The macro asset views (B2) — lamp re-framings, not registry assets.
			info.Assets = append([]string{}, macroAssetViews...)
			info.Examples = append(info.Examples, "/agents/macro?asset=gold")
		}
		if name == keyRisk {
			info.Examples = []string{"/agents/risk?balance=10000&risk=1&entry=64000&stop=62500"}
		}
		items = append(items, info)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"agents":     items,
		"disclaimer": disclaimerText,
	})
}

func (s *HTTPServer) handleAgent(w http.ResponseWriter, r *http.Request) {
	name := strings.ToLower(strings.TrimPrefix(r.URL.Path, "/agents/"))
	if name == "" || strings.Contains(name, "/") {
		writeErr(w, http.StatusNotFound, "not found — GET /agents lists the available agents")
		return
	}
	known := false
	for _, n := range httpAgentNames {
		if n == name {
			known = true
			break
		}
	}
	if !known {
		writeErr(w, http.StatusNotFound, fmt.Sprintf("unknown agent %q — GET /agents lists the available agents", name))
		return
	}

	q := r.URL.Query()
	// Repeated parameters are a client bug, never a silent first-wins
	// (review fix 5: ?assets=btc&assets=eth used to scan only BTC).
	for _, p := range []string{"asset", "assets", "tf"} {
		if len(q[p]) > 1 {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("duplicate parameter %q — pass it once", p))
			return
		}
	}
	asset := strings.TrimSpace(q.Get("asset"))
	if asset != "" && !assetAgents[name] && name != keyMacro {
		writeErr(w, http.StatusBadRequest, fmt.Sprintf("agent %q does not take an ?asset= parameter", name))
		return
	}
	// ?assets= (comma scan) and ?tf= are momentum-only (B1) — anywhere else
	// they would suggest a configurability the agent doesn't have.
	if name != keyMomentum {
		if _, has := q["assets"]; has {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("agent %q does not take a ?assets= parameter", name))
			return
		}
		if _, has := q["tf"]; has {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("agent %q does not take a ?tf= parameter", name))
			return
		}
	}

	ctx := r.Context()
	switch name {
	case keyMacro:
		// ?asset=btc|gold re-frames the lamps for that asset (B2); no param =
		// the global regime card. Strictly the two view keys — the macro read
		// is a lamp re-framing, not a candle asset from the trading registry.
		if asset == "" {
			card, _ := s.ag.MacroCard(ctx)
			s.writeCard(w, r, card)
			return
		}
		view := strings.ToLower(asset)
		if view != macroAssetBTC && view != macroAssetGold {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf(
				"unknown macro asset %q — allowed: %s", asset, strings.Join(macroAssetViews, ", ")))
			return
		}
		s.writeCard(w, r, s.ag.MacroAssetCard(ctx, view))
	case keyWhale:
		s.writeCard(w, r, s.ag.WhaleCard(ctx))
	case keyFunding:
		s.writeCard(w, r, s.ag.FundingCard(ctx))
	case keyFX:
		s.writeCard(w, r, s.ag.FXCard(ctx))
	case keyNews:
		s.writeCard(w, r, s.ag.NewsCard(ctx))
	case keyMomentum:
		// B1: ?assets=btc,eurusd (max 6, registry-validated) + ?tf=1h|4h|1d.
		tf := strings.ToLower(strings.TrimSpace(q.Get("tf")))
		if tf != "" && !momentumTFs[tf] {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("unknown timeframe %q — allowed: %s", tf, momentumTFList))
			return
		}
		assetsRaw, hasAssets := q["assets"]
		switch {
		case hasAssets && asset != "":
			writeErr(w, http.StatusBadRequest, "use either ?asset= (single) or ?assets= (comma scan), not both")
		case hasAssets:
			raw := strings.TrimSpace(assetsRaw[0])
			if raw == "" {
				writeErr(w, http.StatusBadRequest, "empty ?assets= — a comma list like btc,eurusd,gold")
				return
			}
			keys, err := parseAssetList(raw)
			if err != nil {
				writeErr(w, http.StatusBadRequest, err.Error())
				return
			}
			s.writeCard(w, r, s.ag.MomentumScanCard(ctx, keys, tf))
		case asset == "" && tf != "":
			// tf alone re-bases the default trio on the requested timeframe.
			s.writeCard(w, r, s.ag.MomentumScanCard(ctx, defaultMomentumKeys, tf))
		case asset == "": // multi-asset default, exactly like typing /momentum
			s.writeCard(w, r, s.ag.MomentumCard(ctx))
		default:
			spec, err := resolveAsset(asset)
			if err != nil {
				writeErr(w, http.StatusBadRequest, err.Error())
				return
			}
			spec, err = specWithTF(spec, tf)
			if err != nil {
				writeErr(w, http.StatusBadRequest, err.Error())
				return
			}
			s.writeCard(w, r, s.ag.MomentumAssetCard(ctx, spec))
		}
	case keyTrend, keySR, keyVol:
		spec, err := resolveAsset(asset) // "" resolves to BTC, same as the bot
		if err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		switch name {
		case keyTrend:
			s.writeCard(w, r, s.ag.TrendCard(ctx, spec))
		case keySR:
			s.writeCard(w, r, s.ag.SRCard(ctx, spec))
		default:
			s.writeCard(w, r, s.ag.VolCard(ctx, spec))
		}
	case keyGold:
		// Fixed asset by design: this agent IS the gold read, so it takes no
		// ?asset= (the guard above already rejects one).
		s.writeCard(w, r, s.ag.GoldCard(ctx))
	case keyRisk:
		s.handleRisk(w, r, q)
	case keyDigest:
		s.handleDigest(w, r, ctx)
	case keyTop:
		s.handleTop(w, r, ctx)
	}
}

// writeCard serves one agent card: 200 with the envelope, or an honest 503
// when the agent could not produce a verdict (source offline, insufficient
// history, radar warming up) — mirroring the card's own degraded wording.
// The 503 body carries the same machine-readable ok/reason pair as the
// envelope, so templates can branch on WHY (offline vs insufficient vs
// warming up) without parsing the error string.
func (s *HTTPServer) writeCard(w http.ResponseWriter, r *http.Request, card Card) {
	if card.Offline {
		// No Last-Modified on a failure: a cached 503 would keep the agent
		// dark for the whole window after its source came back.
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{
			"error":  fmt.Sprintf("%s: %s", card.Agent, card.Verdict),
			"ok":     false,
			"reason": card.effectiveStatus().reason(),
		})
		return
	}
	if card.noValidator {
		// Composite body: no stamp covers it (see Card.noValidator), so no
		// Last-Modified and If-Modified-Since is ignored — always 200.
		writeJSON(w, http.StatusOK, cardEnvelope(card))
		return
	}
	writeJSONAt(w, r, http.StatusOK, card.DataTime, cardEnvelope(card))
}

func (s *HTTPServer) handleRisk(w http.ResponseWriter, r *http.Request, q url.Values) {
	params := []string{"balance", "risk", "entry", "stop"}
	vals := make([]float64, len(params))
	for i, p := range params {
		raw := strings.TrimSpace(q.Get(p))
		if raw == "" {
			writeErr(w, http.StatusBadRequest,
				"missing ?"+p+"= — usage: /agents/risk?balance=10000&risk=1&entry=64000&stop=62500")
			return
		}
		v, err := parseMoney(raw) // same tolerant parsing as the /risk command
		if err != nil {
			writeErr(w, http.StatusBadRequest, p+": "+err.Error())
			return
		}
		vals[i] = v
	}
	// Same validation the card builder runs — surfaced as 400, not a 200
	// card that says "Those numbers don't work".
	// A result the card cannot print as finite numbers is the same honest 400
	// (riskPrint), never a NaN that fails JSON encoding with a 500.
	res, err := calcRisk(vals[0], vals[1], vals[2], vals[3])
	if err == nil {
		_, err = riskPrint(res)
	}
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	// No Last-Modified: the risk card's DataTime is the answering time, not a
	// data time. This result is a pure function of the query params, so a
	// validator built from "now" would never match and only add noise.
	writeJSON(w, http.StatusOK, cardEnvelope(s.ag.RiskCard(vals, false, nil)))
}

// digestAgentName / digestHeadline are the digest's own identity line, shared
// verbatim by /agents/digest and the landing showcase row so the two can
// never word the same sweep differently.
const digestAgentName = "AlphaVizor Digest"

// digestHeadline names the winner's asset in the same "Agent · Asset" form as
// the card header, because the digest envelope's own "asset" stays empty (the
// sweep covers many markets) and without it "Confirmed UPTREND" said nothing
// about WHICH market. The market-wide macro winner carries no asset and keeps
// the bare form; funding names the coin it shows (stage 1, 2026-09-15). Every card with an asset names it — since 2026-09-15
// the multi-asset momentum card too ("Momentum Agent · BTC/ETH/XAUUSD"): its
// verdict became a counter ("1 bullish (ETH) · 0 bearish · …") and no longer
// names each market.
func digestHeadline(top Card) string {
	who := top.Agent
	if top.Asset != "" {
		who += " · " + top.Asset
	}
	return "Top signal: " + who + " — " + top.Verdict
}

// ── composite content blocks (additive 2026-09-16) ───────────────────────────
//
// /agents/digest and /agents/top had no blocks at all, so the site rendered
// neither a CURRENT READING nor a SCOPE AND LIMITATIONS section for them and
// every caveat sank into the numbered facts list.
//
// Both blocks describe the SELECTION, never the market: what the sweep read,
// which card the fixed rule highlighted, and the one thing the rule cannot do
// — compare three agents whose scales are not calibrated against each other
// (the same statement howTexts[keyDigest] and selectionLine already make).
// scenarios, invalidates, why_level and regime stay empty on the digest: a
// sweep of many markets holds no single idea, level or regime.

// selectionCaveat is the limitation of a card chosen by the digest rule. It
// branches EXACTLY like selectionLine, because what the reader must not
// over-read depends on what actually put the card there:
//
//   - macro_risk_off: the gate put it there, ahead of every agent — nothing
//     was compared;
//   - strongest_confirmed with two or more CONFIRMED eligible readings: the
//     only branch where scores were compared (pickTop scores eligible AND
//     confirmed only), so the only one that says the scales are not
//     calibrated (selectionLine says so too);
//   - strongest_confirmed with one confirmed reading among several eligible:
//     the others were fresh but never competed — nothing was compared;
//   - strongest_confirmed with one eligible: the only fresh live reading —
//     nothing to compare it with;
//   - fallback_unconfirmed: nothing confirmed, the card is the fallback order's;
//   - fallback_macro: no fresh live reading at all, macro is the last resort.
//
// Agent names come from allSignalNames / scopedSignalName, which also carry
// what each ranked reading covers. Funding is ranked like the other two; it
// has no scope suffix only because it is market-wide.
func selectionCaveat(p topPick) string {
	all := allSignalNames()
	switch p.Rule {
	case ruleMacroRiskOff:
		return "Placed by the macro risk-off gate, ahead of " + all +
			"; no comparison between agents chose it, and it is a backdrop, not a finding about one market."
	case ruleConfirmed:
		// Only eligible AND confirmed candidates were scored (confirmedSplit).
		eligible, confirmed := confirmedSplit(p)
		switch {
		case len(confirmed) >= 2:
			return "Chosen by comparing the confirmed readings of " + strings.Join(confirmed, ", ") +
				"; their scores are on scales not calibrated against each other, so a higher score does not mean a stronger reading."
		case len(eligible) >= 2:
			return signalNames[p.Winner] + " is the only confirmed reading among " + strings.Join(eligible, ", ") +
				"; it was not compared with another agent."
		}
		return signalNames[p.Winner] + " is the only fresh live reading among " + all +
			"; it was not compared with another agent."
	case ruleUnconfirmed:
		return "No confirmed reading among " + all +
			"; this card is shown by the digest's fallback order, not as a finding."
	default:
		return "No fresh live reading among " + all +
			"; the macro card is shown as the last resort, not as a finding."
	}
}

// digestBlocks words one sweep: how much of it read (the same Live/Total the
// digest.status object carries), which card the rule highlighted (the headline
// the digest already prints) and how it was chosen (selectionLine, verbatim).
// nil when the highlighted card has no live reading: like every other agent,
// an ok:false envelope carries no blocks.
func digestBlocks(g gathered, p topPick, top Card) *ContentBlocks {
	if top.effectiveStatus() != statusOK {
		return nil
	}
	h := g.health()
	return &ContentBlocks{
		WhatHappened: fmt.Sprintf("%d of %d sections read. %s. %s.",
			h.Live, h.Total, strings.TrimSuffix(digestHeadlineFor(p, top), "."), selectionLine(p)),
		Limitations: selectionCaveat(p),
	}
}

// topBlocks keeps the winner card's own blocks — on /agents/top the envelope IS
// that one card — and appends the selection caveat for the rule that chose it.
// The blocks are copied first: the pointer is shared with the gathered sweep,
// and a digest rendered from the same sweep must not inherit the sentence.
// nil when the winner has no live reading or no blocks of its own: an
// ok:false card carries no blocks, and a caveat alone under an empty
// what_happened describes nothing.
func topBlocks(p topPick, c Card) *ContentBlocks {
	if c.effectiveStatus() != statusOK || c.Blocks == nil {
		return nil
	}
	// Deep copy: the slice and the invalidates pointer are the winner's own.
	b := *c.Blocks
	if c.Blocks.Scenarios != nil {
		b.Scenarios = append([]string{}, c.Blocks.Scenarios...)
	}
	if c.Blocks.Invalidates != nil {
		inv := *c.Blocks.Invalidates
		b.Invalidates = &inv
	}
	caveat := selectionCaveat(p)
	if b.Limitations == "" {
		b.Limitations = caveat
	} else {
		// Funding's and trend's limitations end without a full stop; joined as
		// is they ran into the caveat ("…other venues No confirmed reading…").
		b.Limitations = endSentence(b.Limitations) + " " + caveat
	}
	return &b
}

// handleDigest runs the exact digest sweep and serves the top card as the
// envelope head, the remaining one-liners as sections and the AI brief as
// ai_text. Partial upstream failures stay inside the 200 as honest offline
// one-liners — exactly like the Telegram digest.
func (s *HTTPServer) handleDigest(w http.ResponseWriter, r *http.Request, ctx context.Context) {
	g := s.ag.gather(ctx)
	brief := s.ag.aiBrief(ctx, g) // same aiMemo as the Telegram path
	// NO validator (zero time → no Last-Modified, If-Modified-Since ignored,
	// always 200). The digest re-sweeps on every request, so no stamp is
	// sound: a component stamp can lag a changed component (it may be neither
	// the newest nor the oldest reading), and the build time is truncated to
	// the second in the header, so two different builds inside one second
	// would share it. A validator on a per-request rebuild buys nothing.
	// data_as_of (the oldest reading) is unaffected.
	writeJSONAt(w, r, http.StatusOK, time.Time{}, digestEnvelope(g, brief))
}

// digestEnvelope is the pure half of handleDigest: one gathered sweep + the AI
// brief → the digest envelope. No network, no clock — testable on its own.
func digestEnvelope(g gathered, brief string) httpEnvelope {
	p := g.selection()
	winner, top := topSelection(g)
	env := cardEnvelope(top)
	env.Agent = digestAgentName
	env.Asset = ""
	env.Verdict = digestHeadlineFor(p, top)
	// Top-level ok/reason stay the HIGHLIGHTED card's status, exactly as
	// before (cardEnvelope above): clients gate showing verdict/semaphore on
	// them. The sweep's health lives only in the additive digest.status /
	// live_sections / degraded_sources (and /showcase digest_status);
	// digest.selection.highlight_ok/_reason repeat the top-level pair.
	env.Digest = digestReadout(g, p, top)
	// The winner's blocks are ONE agent's content sentences. Inherited here they
	// read as the DIGEST's own conclusion — a live digest said "Macro: risk-on"
	// in its sections while blocks.regime said "flat — no trend" (the BTC Trend
	// card's local regime). So they are replaced, not kept: the digest gets its
	// own pair, about the sweep and the selection (digestBlocks, 2026-09-16 —
	// before that the field was simply dropped and the site had no CURRENT
	// READING section for the digest at all). /agents/top keeps the winner's:
	// there the envelope IS that one card.
	env.Blocks = digestBlocks(g, p, top)
	env.AIText = nil
	if brief != "" {
		env.AIText = &brief
	}
	sections := []string{}
	for _, k := range digestOrder {
		if k == winner {
			continue
		}
		if c, ok := g.cards[k]; ok {
			sections = append(sections, htmlToPlain(c.OneLiner()))
		}
	}
	env.Sections = sections
	// The oldest reading the digest renders, NOT the sweep time. This used to
	// be now(), which stamped a card carrying an 84-minute-old 4h momentum
	// read as current — see oldestData in bot.go.
	env.DataAsOf = digestDataTime(g).Format(time.RFC3339)
	env.CardHTML = renderDigestHTML(g, brief)
	return env
}

// handleTop serves the single strongest signal: the winner card's envelope
// with the AI brief and why-line (when enabled) joined into ai_text, and the
// full Telegram /top message in card_html.
func (s *HTTPServer) handleTop(w http.ResponseWriter, r *http.Request, ctx context.Context) {
	g := s.ag.gather(ctx)
	winner, card := topSelection(g)
	brief, why := s.ag.aiTopTexts(ctx, winner, g) // same aiMemo as the Telegram path

	env := cardEnvelope(mergeTopWhy(card, why))
	env.Blocks = topBlocks(g.selection(), card)
	var parts []string
	if brief != "" {
		parts = append(parts, brief)
	}
	if env.AIText != nil { // card AI block (e.g. macro mood read) + why-line
		parts = append(parts, *env.AIText)
	}
	env.AIText = nil
	if len(parts) > 0 {
		joined := strings.Join(parts, "\n\n")
		env.AIText = &joined
	}
	env.CardHTML = renderTopHTML(card, brief, why)
	// NO validator (zero time → no Last-Modified, If-Modified-Since ignored,
	// always 200), like /agents/digest. The winner's stamp does not cover the
	// whole response: the AI brief and why-line are added per request (an AI
	// failure is cached only briefly, so a response without AI and the next
	// one with it would share the stamp), and the brief reads the whole sweep,
	// so a non-winning agent's change alters it without moving the winner's
	// data time. The card's own data time stays in data_as_of.
	writeJSONAt(w, r, http.StatusOK, time.Time{}, env)
}
