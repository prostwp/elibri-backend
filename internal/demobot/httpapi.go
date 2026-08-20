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
	keyMomentum, keyTrend, keySR, keyVol, keyRisk, keyNews,
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
	Levels     any      `json:"levels,omitempty"`
	Confidence *int     `json:"confidence"`         // 0-100, null when the source gave none
	AIText     *string  `json:"ai_text"`            // plain-text AI block, null when absent
	Sections   []string `json:"sections,omitempty"` // digest only: the one-liners
	DataAsOf   string   `json:"data_as_of"`         // RFC3339, same stamp as the card footer
	Disclaimer string   `json:"disclaimer"`
	CardHTML   string   `json:"card_html"` // the exact Telegram HTML card
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
		Asset:      c.Asset,
		OK:         st == statusOK,
		Verdict:    c.Verdict,
		Semaphore:  semaphoreOf(c.Emoji),
		Facts:      append([]string{}, c.Facts...), // [] not null when empty
		Levels:     c.Levels,
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
		"service":    "AlphaVizor demo bot — read-only HTTP JSON API",
		"agents":     "/agents",
		"disclaimer": disclaimerText,
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
	asset := strings.TrimSpace(q.Get("asset"))
	if asset != "" && !assetAgents[name] {
		writeErr(w, http.StatusBadRequest, fmt.Sprintf("agent %q does not take an ?asset= parameter", name))
		return
	}

	ctx := r.Context()
	switch name {
	case keyMacro:
		card, _ := s.ag.MacroCard(ctx)
		s.writeCard(w, card)
	case keyWhale:
		s.writeCard(w, s.ag.WhaleCard(ctx))
	case keyFunding:
		s.writeCard(w, s.ag.FundingCard(ctx))
	case keyFX:
		s.writeCard(w, s.ag.FXCard(ctx))
	case keyNews:
		s.writeCard(w, s.ag.NewsCard(ctx))
	case keyMomentum:
		if asset == "" { // multi-asset default, exactly like typing /momentum
			s.writeCard(w, s.ag.MomentumCard(ctx))
			return
		}
		spec, err := resolveAsset(asset)
		if err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		s.writeCard(w, s.ag.MomentumAssetCard(ctx, spec))
	case keyTrend, keySR, keyVol:
		spec, err := resolveAsset(asset) // "" resolves to BTC, same as the bot
		if err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		switch name {
		case keyTrend:
			s.writeCard(w, s.ag.TrendCard(ctx, spec))
		case keySR:
			s.writeCard(w, s.ag.SRCard(ctx, spec))
		default:
			s.writeCard(w, s.ag.VolCard(ctx, spec))
		}
	case keyRisk:
		s.handleRisk(w, q)
	case keyDigest:
		s.handleDigest(w, ctx)
	case keyTop:
		s.handleTop(w, ctx)
	}
}

// writeCard serves one agent card: 200 with the envelope, or an honest 503
// when the agent could not produce a verdict (source offline, insufficient
// history, radar warming up) — mirroring the card's own degraded wording.
// The 503 body carries the same machine-readable ok/reason pair as the
// envelope, so templates can branch on WHY (offline vs insufficient vs
// warming up) without parsing the error string.
func (s *HTTPServer) writeCard(w http.ResponseWriter, card Card) {
	if card.Offline {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{
			"error":  fmt.Sprintf("%s: %s", card.Agent, card.Verdict),
			"ok":     false,
			"reason": card.effectiveStatus().reason(),
		})
		return
	}
	writeJSON(w, http.StatusOK, cardEnvelope(card))
}

func (s *HTTPServer) handleRisk(w http.ResponseWriter, q url.Values) {
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
	if _, err := calcRisk(vals[0], vals[1], vals[2], vals[3]); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	s.writeCard(w, s.ag.RiskCard(vals, false, nil))
}

// handleDigest runs the exact digest sweep and serves the top card as the
// envelope head, the remaining one-liners as sections and the AI brief as
// ai_text. Partial upstream failures stay inside the 200 as honest offline
// one-liners — exactly like the Telegram digest.
func (s *HTTPServer) handleDigest(w http.ResponseWriter, ctx context.Context) {
	g := s.ag.gather(ctx)
	winner, top := topSelection(g)
	brief := s.ag.aiBrief(ctx, g) // same aiMemo as the Telegram path

	env := cardEnvelope(top)
	env.Agent = "AlphaVizor Digest"
	env.Asset = ""
	env.Verdict = "Top signal: " + top.Agent + " — " + top.Verdict
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
	env.DataAsOf = time.Now().UTC().Format(time.RFC3339) // digest composes at request time, like its footer
	env.CardHTML = renderDigestHTML(g, brief)
	writeJSON(w, http.StatusOK, env)
}

// handleTop serves the single strongest signal: the winner card's envelope
// with the AI brief and why-line (when enabled) joined into ai_text, and the
// full Telegram /top message in card_html.
func (s *HTTPServer) handleTop(w http.ResponseWriter, ctx context.Context) {
	g := s.ag.gather(ctx)
	winner, card := topSelection(g)
	brief, why := s.ag.aiTopTexts(ctx, winner, g) // same aiMemo as the Telegram path

	env := cardEnvelope(mergeTopWhy(card, why))
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
	writeJSON(w, http.StatusOK, env)
}
