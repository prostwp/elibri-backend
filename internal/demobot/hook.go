package demobot

// hook.go — push hook for the site backend: once per interval the demobot
// re-reads its own GET addresses in-process and POSTs an event for every
// address whose body changed since the last event sent for it.
//
// Contract (docs/demobot-http.md "Push hook" is the reference):
//   - Off unless DEMOBOT_HOOK_URL is a valid http(s) URL: no goroutine, no
//     requests.
//   - data is the GET body itself: the hook calls the API route table
//     in-process (past the rate limiter), so an event can never disagree with
//     what GET /agents/{agent} answers.
//   - "Changed" is a sha256 over the body with the fields that follow the
//     request clock masked (hookNormalize) — a new request time alone is not
//     a new reading.
//   - One attempt per event, never repeated: an attempt counts as sent
//     whatever the answer. Failures only pause the address (or, when the
//     endpoint cannot be reached at all, the whole sweep). The site
//     reconciles by GET.
//   - Last-sent state lives in process memory; after a restart every address
//     is sent again (the site answers duplicates with 200/409).

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	hookDefaultInterval       = 60 * time.Second
	hookMinInterval           = 30 * time.Second
	hookDefaultDigestInterval = 5 * time.Minute
	hookSendTimeout           = 5 * time.Second
	// hookMaxPause caps the exponential pause after failed deliveries.
	hookMaxPause = 10 * time.Minute
	// hookCallTimeout bounds one in-process GET (the digest sweep plus an AI
	// call fit well inside, like the server's WriteTimeout).
	hookCallTimeout = 60 * time.Second
	hookSnippetLen  = 300
	// hookMasked replaces a request-time value before hashing.
	hookMasked = "*"
	// hookUnavailable is the state word of a body without a real reading
	// (ok=false, the 503 degraded body included).
	hookUnavailable = "unavailable"
)

// ── targets ──────────────────────────────────────────────────────────────────

// hookTarget is one GET address the hook watches.
type hookTarget struct {
	Agent  string            // agent key, as in /agents/{agent}
	Asset  string            // upper-case asset from the address, "" for a global read
	Params map[string]string // extra query parameters (tf); none in v1
	Path   string            // request URI for the in-process router
	// slow: digest, top and news run on the digest interval, not every sweep —
	// each run reaches LLM-backed sources (digest/top: a full gather incl. the
	// backend mood read and the demobot AI; news: the backend's narrative AI
	// idea, retried on every read while its generation fails; docs "Load").
	slow bool
}

// hookTargets is the v1 address set: every agent except risk (a function of
// user input); every registry asset for the ?asset= agents; the momentum
// composite without parameters (no ?tf= / ?assets= variants in v1); the macro
// asset views; digest and top. /agents/trend/chart and /showcase are not sent.
func hookTargets() []hookTarget {
	var out []hookTarget
	add := func(agent, asset string, slow bool) {
		q := url.Values{}
		if asset != "" {
			q.Set("asset", asset)
		}
		path := "/agents/" + agent
		if len(q) > 0 {
			path += "?" + q.Encode()
		}
		out = append(out, hookTarget{
			Agent: agent, Asset: strings.ToUpper(asset), Params: map[string]string{},
			Path: path, slow: slow,
		})
	}
	for _, name := range httpAgentNames {
		switch name {
		case keyRisk:
			continue
		case keyDigest, keyTop, keyNews:
			add(name, "", true)
		case keyMacro:
			add(name, "", false)
			for _, v := range macroAssetViews {
				add(name, v, false)
			}
		case keyMomentum:
			add(name, "", false) // the composite BTC/ETH/XAUUSD card
			for _, k := range assetKeys() {
				add(name, k, false)
			}
		case keyTrend, keySR, keyVol:
			// No bare address: without ?asset= these answer BTC, already sent.
			for _, k := range assetKeys() {
				add(name, k, false)
			}
		default:
			add(name, "", false)
		}
	}
	return out
}

// paramString is the params segment of event_id: sorted k=v joined by ",".
func (t hookTarget) paramString() string {
	keys := make([]string, 0, len(t.Params))
	for k := range t.Params {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+"="+t.Params[k])
	}
	return strings.Join(parts, ",")
}

// String is for logs and tests.
func (t hookTarget) String() string { return fmt.Sprintf("%s %s", t.Agent, t.Path) }

// ── event ────────────────────────────────────────────────────────────────────

type hookEvent struct {
	EventID      string            `json:"event_id"`
	Agent        string            `json:"agent"`
	Asset        string            `json:"asset"`
	Params       map[string]string `json:"params"`
	DataAsOf     *string           `json:"data_as_of"` // null when the body has none (503)
	SentAt       string            `json:"sent_at"`
	StateChanged *hookStateChange  `json:"state_changed"`
	Data         json.RawMessage   `json:"data"`
}

type hookStateChange struct {
	From string `json:"from"`
	To   string `json:"to"`
}

func dashIfEmpty(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

// hookEventID = agent:asset:params:data_as_of:first 8 hex of the hash; an
// empty segment is "-".
func hookEventID(t hookTarget, dataAsOf, hash string) string {
	return strings.Join([]string{
		t.Agent, dashIfEmpty(t.Asset), dashIfEmpty(t.paramString()), dashIfEmpty(dataAsOf), hash[:8],
	}, ":")
}

// ── normalization ────────────────────────────────────────────────────────────

var (
	// Macro F&G fact when stale: "… stale, last update Sep 14 (52h ago) …" —
	// the age is counted to the backend's captured_at, i.e. the request time.
	hookFNGAgoRe = regexp.MustCompile(`(stale, last update [^()]*\()(?:\d+h|\d+d|>999d)( ago\))`)
	// Gold "Price now …" fact: "⚠ 7h old" — time.Since(the hourly close).
	hookGoldAgeRe = regexp.MustCompile(`⚠ \d+h old`)
)

// hookRequestStampAgents may carry a request-time stamp where a data time
// belongs (parseWhen / offlineCard / oldestData fall back to now; macro's
// captured_at is the backend's request time; a funding card inside
// digest/top). For them a stamp inside the call window is masked. Candle
// agents are left out: their stamps in a 200 are always bar closes.
var hookRequestStampAgents = map[string]bool{
	keyFunding: true, keyMacro: true, keyWhale: true, keyNews: true, keyDigest: true, keyTop: true,
}

// hookNormalizer masks the request-clock fields of one GET body.
type hookNormalizer struct {
	agent      string
	start, end time.Time // the in-process call window
}

// requestTime reports whether an RFC3339 stamp falls inside the call window —
// i.e. it was taken from a clock during this call, not from the data.
func (n hookNormalizer) requestTime(s string) bool {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return false
	}
	return !t.Before(n.start.Truncate(time.Second)) && !t.After(n.end)
}

func (n hookNormalizer) maskIfRequestTime(m map[string]any, key string) {
	if s, ok := m[key].(string); ok && n.requestTime(s) {
		m[key] = hookMasked
	}
}

// hookNormalize returns the canonical JSON the change hash is taken over:
// the body with every field that follows the request clock masked. The list
// is derived from the builders (docs/demobot-http.md "Push hook"):
//
//   - funding: data_as_of and the card_html footer stamp — always (DataTime =
//     time.Now, "as of request time");
//   - macro, whale, news, digest, top: data_as_of and the footer stamp when
//     they fall inside the call window (fallbacks to now);
//   - any "macro" object (macro cards, and digest/top with a macro winner):
//     freshness.captured_at (backend request time) and fear_greed.age_hours
//     (counted to it);
//   - digest: digest.generated_at (sweep clock); highlight_data_as_of,
//     candidates[].data_as_of and sections[].data_as_of inside the window
//     (the funding entries always are);
//   - any string: the stale-F&G "(Nh ago)" age and the gold "⚠ Nh old" age.
func hookNormalize(agent string, body []byte, start, end time.Time) ([]byte, error) {
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber() // numbers stay byte-exact through the round trip
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	if env, ok := v.(map[string]any); ok {
		hookNormalizer{agent: agent, start: start.UTC(), end: end.UTC()}.envelope(env)
	}
	return encodeJSON(hookMaskStrings(v))
}

func (n hookNormalizer) envelope(env map[string]any) {
	if s, ok := env["data_as_of"].(string); ok {
		if n.agent == keyFunding || (hookRequestStampAgents[n.agent] && n.requestTime(s)) {
			env["data_as_of"] = hookMasked
			if h, ok := env["card_html"].(string); ok {
				env["card_html"] = maskFooterStamp(h, s)
			}
		}
	}
	if m, ok := env["macro"].(map[string]any); ok {
		if fr, ok := m["freshness"].(map[string]any); ok {
			delete(fr, "captured_at")
		}
		if fg, ok := m["fear_greed"].(map[string]any); ok {
			delete(fg, "age_hours")
		}
	}
	if d, ok := env["digest"].(map[string]any); ok {
		delete(d, "generated_at")
		if sel, ok := d["selection"].(map[string]any); ok {
			n.maskIfRequestTime(sel, "highlight_data_as_of")
			for _, c := range asAnySlice(sel["candidates"]) {
				if cm, ok := c.(map[string]any); ok {
					n.maskIfRequestTime(cm, "data_as_of")
				}
			}
		}
		for _, s := range asAnySlice(d["sections"]) {
			if sm, ok := s.(map[string]any); ok {
				n.maskIfRequestTime(sm, "data_as_of")
			}
		}
	}
}

// maskFooterStamp masks the footer's "AlphaVizor · YYYY-MM-DD HH:MM UTC" —
// the same instant as data_as_of at minute resolution (Card.RenderHTML,
// renderDigestHTML).
func maskFooterStamp(cardHTML, dataAsOf string) string {
	t, err := time.Parse(time.RFC3339, dataAsOf)
	if err != nil {
		return cardHTML
	}
	stamp := "AlphaVizor · " + t.UTC().Format("2006-01-02 15:04") + " UTC"
	return strings.ReplaceAll(cardHTML, stamp, "AlphaVizor · "+hookMasked+" UTC")
}

func asAnySlice(v any) []any {
	s, _ := v.([]any)
	return s
}

func hookMaskStrings(v any) any {
	switch x := v.(type) {
	case string:
		x = hookFNGAgoRe.ReplaceAllString(x, "${1}"+hookMasked+"${2}")
		return hookGoldAgeRe.ReplaceAllString(x, "⚠ "+hookMasked+"h old")
	case map[string]any:
		for k, e := range x {
			x[k] = hookMaskStrings(e)
		}
		return x
	case []any:
		for i, e := range x {
			x[i] = hookMaskStrings(e)
		}
		return x
	}
	return v
}

// hookBodyFields reads the state and stamp of a GET body (200 envelope or
// 503 degraded body).
func hookBodyFields(status int, body []byte) (state, dataAsOf string) {
	var b struct {
		OK        *bool  `json:"ok"`
		Semaphore string `json:"semaphore"`
		DataAsOf  string `json:"data_as_of"`
	}
	_ = json.Unmarshal(body, &b)
	if status != http.StatusOK || b.OK == nil || !*b.OK {
		return hookUnavailable, b.DataAsOf
	}
	return b.Semaphore, b.DataAsOf
}

// ── configuration ────────────────────────────────────────────────────────────

type hookConfig struct {
	target         string
	interval       time.Duration
	digestInterval time.Duration
}

// hookConfigFromEnv reads DEMOBOT_HOOK_URL / DEMOBOT_HOOK_INTERVAL /
// DEMOBOT_HOOK_DIGEST_INTERVAL. ok=false when the URL is empty or not an
// http(s) URL (hook off; the invalid case is logged).
func hookConfigFromEnv(logf func(string, ...any)) (hookConfig, bool) {
	raw := strings.TrimSpace(os.Getenv("DEMOBOT_HOOK_URL"))
	if raw == "" {
		return hookConfig{}, false
	}
	u, err := url.Parse(raw)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		logf("[demobot] push hook: DEMOBOT_HOOK_URL=%q is not an http(s) URL — hook not started", raw)
		return hookConfig{}, false
	}
	cfg := hookConfig{target: raw}
	cfg.interval = hookEnvDuration(logf, "DEMOBOT_HOOK_INTERVAL", hookDefaultInterval)
	if cfg.interval < hookMinInterval {
		logf("[demobot] push hook: DEMOBOT_HOOK_INTERVAL=%s below the %s minimum — using %s", cfg.interval, hookMinInterval, hookMinInterval)
		cfg.interval = hookMinInterval
	}
	cfg.digestInterval = hookEnvDuration(logf, "DEMOBOT_HOOK_DIGEST_INTERVAL", hookDefaultDigestInterval)
	if cfg.digestInterval < cfg.interval {
		logf("[demobot] push hook: DEMOBOT_HOOK_DIGEST_INTERVAL=%s below the sweep interval — using %s", cfg.digestInterval, cfg.interval)
		cfg.digestInterval = cfg.interval
	}
	return cfg, true
}

// hookEnvDuration reads a Go duration ("90s") or plain seconds ("90"); empty
// → def; unreadable or non-positive → def, logged.
func hookEnvDuration(logf func(string, ...any), name string, def time.Duration) time.Duration {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return def
	}
	d, err := time.ParseDuration(raw)
	if err != nil {
		if n, nerr := strconv.Atoi(raw); nerr == nil {
			d, err = time.Duration(n)*time.Second, nil
		}
	}
	if err != nil || d <= 0 {
		logf("[demobot] push hook: %s=%q unreadable — using %s", name, raw, def)
		return def
	}
	return d
}

// ── the hook ─────────────────────────────────────────────────────────────────

// hookTargetState is the per-address memory.
type hookTargetState struct {
	sentHash string // hash of the last body POSTed (any answer): never POSTed again
	state    string // state of the last DELIVERED body — the state_changed baseline
	hasState bool
	fails    int // consecutive non-delivery answers (timeout, 401, 408, 5xx, other)
	skip     int // sweeps this address still sits out (per-address pause)
}

// PushHook sweeps the GET addresses and posts changed bodies to the site.
// One goroutine runs it; nothing here is shared, so no locking.
type PushHook struct {
	url            string
	interval       time.Duration
	digestInterval time.Duration // digest/top cadence, ≥ interval
	client         *http.Client
	handler        http.Handler // API routes, no rate limiter
	targets        []hookTarget
	state          map[string]*hookTargetState // by Path
	sweeps         int                         // sweeps started: digest cadence and start rotation
	netFails       int                         // consecutive failures to reach the endpoint
	pause          int                         // whole sweeps still skipped after them
	logf           func(format string, args ...any)
}

// NewPushHook builds a hook over the server's routes: one POST per changed
// address to target, a sweep every interval, digest/top every
// hookDefaultDigestInterval (never below interval).
func NewPushHook(s *HTTPServer, target string, interval time.Duration) *PushHook {
	return &PushHook{
		url:            target,
		interval:       interval,
		digestInterval: max(interval, hookDefaultDigestInterval),
		client: &http.Client{
			Timeout: hookSendTimeout,
			// A redirect is an answer to log, not a second request to send.
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		},
		handler: s.mux,
		targets: hookTargets(),
		state:   map[string]*hookTargetState{},
		logf:    log.Printf,
	}
}

// StartPushHookFromEnv starts the hook goroutine when DEMOBOT_HOOK_URL is a
// valid http(s) URL and reports whether it did. It stops with ctx.
func StartPushHookFromEnv(ctx context.Context, s *HTTPServer) bool {
	cfg, ok := hookConfigFromEnv(log.Printf)
	if !ok {
		if os.Getenv("DEMOBOT_HOOK_URL") == "" {
			log.Printf("[demobot] push hook: off (DEMOBOT_HOOK_URL empty)")
		}
		return false
	}
	h := NewPushHook(s, cfg.target, cfg.interval)
	h.digestInterval = cfg.digestInterval
	log.Printf("[demobot] push hook: on · %d addresses every %s, digest/top every %s → %s",
		len(h.targets), cfg.interval, cfg.digestInterval, cfg.target)
	go h.Run(ctx)
	return true
}

// Run sweeps immediately, then every interval, until ctx is done.
func (h *PushHook) Run(ctx context.Context) {
	tick := time.NewTicker(h.interval)
	defer tick.Stop()
	for {
		h.sweep(ctx)
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
	}
}

// digestEvery: digest/top run on every k-th sweep, k = ⌈digestInterval /
// interval⌉.
func (h *PushHook) digestEvery() int {
	return max(1, int(math.Ceil(float64(h.digestInterval)/float64(h.interval))))
}

// pauseSweeps is the number of sweeps to sit out after the n-th consecutive
// failure: 1, 3, 7, … — the next attempt no sooner than 2, 4, 8 … intervals
// later — capped so the pause never exceeds hookMaxPause.
func (h *PushHook) pauseSweeps(n int) int {
	maxSkip := max(0, int(hookMaxPause/h.interval)-1)
	if n <= 0 {
		return 0
	}
	if n > 20 {
		return maxSkip
	}
	return min((1<<n)-1, maxSkip)
}

func (h *PushHook) stateFor(t hookTarget) *hookTargetState {
	st, ok := h.state[t.Path]
	if !ok {
		st = &hookTargetState{}
		h.state[t.Path] = st
	}
	return st
}

// sweep walks the addresses sequentially (never all at once: the builders
// share the 60s candle cache, and one address at a time keeps the upstream
// load at one GET's worth per cache window). The starting address moves one
// step every sweep, so no address is always last.
func (h *PushHook) sweep(ctx context.Context) {
	n := h.sweeps
	h.sweeps++
	if h.pause > 0 {
		h.pause--
		return
	}
	every := h.digestEvery()
	k := len(h.targets)
	for i := 0; i < k; i++ {
		if ctx.Err() != nil {
			return
		}
		t := h.targets[(n+i)%k]
		st := h.stateFor(t)
		if st.skip > 0 {
			st.skip--
			continue
		}
		if t.slow && n%every != 0 {
			continue
		}
		if abort := h.process(ctx, t, st); abort {
			return
		}
	}
}

// fetch runs one GET through the route table in-process.
func (h *PushHook) fetch(ctx context.Context, t hookTarget) (int, []byte) {
	cctx, cancel := context.WithTimeout(ctx, hookCallTimeout)
	defer cancel()
	req := httptest.NewRequest(http.MethodGet, t.Path, nil).WithContext(cctx)
	rec := httptest.NewRecorder()
	h.handler.ServeHTTP(rec, req)
	return rec.Code, bytes.TrimRight(rec.Body.Bytes(), "\n")
}

// hookUnreachable reports a failure to connect at all (refused, no route,
// DNS): the endpoint is down, so the rest of the sweep would fail the same way.
func hookUnreachable(err error) bool {
	var op *net.OpError
	return errors.As(err, &op) && op.Op == "dial"
}

// process handles one address; true aborts the sweep (the endpoint cannot be
// reached — no point trying the rest now).
func (h *PushHook) process(ctx context.Context, t hookTarget, st *hookTargetState) bool {
	start := time.Now()
	status, body := h.fetch(ctx, t)
	end := time.Now()
	if status != http.StatusOK && status != http.StatusServiceUnavailable {
		// 400/404 here is a bug in the target list, never a reading.
		h.logf("[demobot] hook: GET %s answered %d — not sent", t.Path, status)
		return false
	}
	norm, err := hookNormalize(t.Agent, body, start, end)
	if err != nil {
		h.logf("[demobot] hook: GET %s body is not JSON (%v) — not sent", t.Path, err)
		return false
	}
	sum := sha256.Sum256(norm)
	hash := hex.EncodeToString(sum[:])
	if hash == st.sentHash {
		return false
	}

	state, dataAsOf := hookBodyFields(status, body)
	ev := hookEvent{
		EventID: hookEventID(t, dataAsOf, hash),
		Agent:   t.Agent,
		Asset:   t.Asset,
		Params:  t.Params,
		SentAt:  time.Now().UTC().Format(time.RFC3339),
		Data:    json.RawMessage(body),
	}
	if ev.Params == nil {
		ev.Params = map[string]string{}
	}
	if dataAsOf != "" {
		ev.DataAsOf = &dataAsOf
	}
	if st.hasState && st.state != state {
		ev.StateChanged = &hookStateChange{From: st.state, To: state}
	}
	payload, err := encodeJSON(ev) // no HTML escaping: data stays the GET bytes
	if err != nil {
		h.logf("[demobot] hook: %s encode: %v", t.Path, err)
		return false
	}

	sent := time.Now()
	code, snippet, err := h.post(ctx, payload)
	ms := time.Since(sent).Milliseconds()
	if err != nil && ctx.Err() != nil { // shutting down — not a failure of theirs
		return true
	}
	// An attempt is a send: this event_id is never POSTed again, whatever
	// the answer (no retries). The next POST for this address needs a new body.
	st.sentHash = hash
	switch {
	case err != nil && hookUnreachable(err):
		h.netFails++
		h.pause = h.pauseSweeps(h.netFails)
		h.logf("[demobot] hook POST %s event=%s → unreachable: %v (%dms); sweep stopped, next sweep in %s",
			t.Path, ev.EventID, err, ms, time.Duration(h.pause+1)*h.interval)
		return true
	case err != nil: // connected, then timed out or the exchange broke
		h.netFails = 0
		h.pauseAddress(st)
		h.logf("[demobot] hook POST %s event=%s → error %v (%dms), not delivered; address paused %s",
			t.Path, ev.EventID, err, ms, time.Duration(st.skip+1)*h.interval)
	case code == http.StatusOK || code == http.StatusCreated || code == http.StatusConflict:
		h.netFails = 0
		st.state, st.hasState, st.fails = state, true, 0
		h.logf("[demobot] hook POST %s event=%s → %d (%dms)", t.Path, ev.EventID, code, ms)
	case code == http.StatusBadRequest || code == http.StatusRequestEntityTooLarge ||
		code == http.StatusUnsupportedMediaType || code == http.StatusUnprocessableEntity:
		// Our body was refused. Not a new state_changed baseline (their
		// history lacks it); no pause — the next change is a different body.
		h.netFails = 0
		st.fails = 0
		h.logf("[demobot] hook POST %s event=%s → %d (%dms) rejected: %q", t.Path, ev.EventID, code, ms, snippet)
	default: // 401, 408, 5xx, 3xx, anything else: not delivered
		h.netFails = 0
		h.pauseAddress(st)
		h.logf("[demobot] hook POST %s event=%s → %d (%dms) not delivered; address paused %s: %q",
			t.Path, ev.EventID, code, ms, time.Duration(st.skip+1)*h.interval, snippet)
	}
	return false
}

func (h *PushHook) pauseAddress(st *hookTargetState) {
	st.fails++
	st.skip = h.pauseSweeps(st.fails)
}

// post sends one event: one attempt, hookSendTimeout, no retries. It returns
// the status and the first hookSnippetLen bytes of the answer.
func (h *PushHook) post(ctx context.Context, payload []byte) (int, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.url, bytes.NewReader(payload))
	if err != nil {
		return 0, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := h.client.Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	head, _ := io.ReadAll(io.LimitReader(resp.Body, hookSnippetLen))
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20)) // drain for reuse
	return resp.StatusCode, string(head), nil
}
