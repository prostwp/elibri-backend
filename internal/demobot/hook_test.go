package demobot

// hook_test.go — the push hook (hook.go) against a stub receiver: off without
// a valid URL, send-on-change only, state_changed, event_id, every answer
// code (no event_id is ever POSTed twice), a hung address does not starve the
// rest, refused connection, the digest cadence, hash stability under the
// request clock, data == the GET body — and the FundingCard tie-break the
// hook depends on. Every upstream is stubbed; nothing touches the network.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// ── fixtures ─────────────────────────────────────────────────────────────────

// hookSink is the stub site endpoint: records every event and answers code;
// an agent listed in hang never gets an answer (the client times out).
type hookSink struct {
	mu     sync.Mutex
	code   int
	hang   map[string]bool
	events []hookEvent
	raws   []string
	ctypes []string
	heads  []http.Header
	srv    *httptest.Server
}

func (s *hookSink) headers() []http.Header {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]http.Header(nil), s.heads...)
}

func newHookSink(t *testing.T, code int) *hookSink {
	t.Helper()
	s := &hookSink{code: code, hang: map[string]bool{}}
	s.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var ev hookEvent
		_ = json.Unmarshal(body, &ev)
		s.mu.Lock()
		s.events = append(s.events, ev)
		s.raws = append(s.raws, string(body))
		s.ctypes = append(s.ctypes, r.Header.Get("Content-Type"))
		s.heads = append(s.heads, r.Header.Clone())
		code, hang := s.code, s.hang[ev.Agent]
		s.mu.Unlock()
		if hang {
			<-r.Context().Done() // released when the client times out
			return
		}
		w.WriteHeader(code)
		_, _ = fmt.Fprintf(w, "sink answer %d", code)
	}))
	t.Cleanup(s.srv.Close)
	return s
}

func (s *hookSink) setCode(code int) {
	s.mu.Lock()
	s.code = code
	s.mu.Unlock()
}

// take returns the events received since the last take.
func (s *hookSink) take() []hookEvent {
	s.mu.Lock()
	defer s.mu.Unlock()
	ev := s.events
	s.events = nil
	return ev
}

func (s *hookSink) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.events)
}

func agentsOf(evs []hookEvent) map[string]int {
	out := map[string]int{}
	for _, e := range evs {
		out[e.Agent]++
	}
	return out
}

// testHook wraps a PushHook with captured logs and the in-process GETs it ran.
type testHook struct {
	*PushHook
	logs    []string
	fetched []string
}

func newTestHook(t *testing.T, ag *Agents, sinkURL string, targets []hookTarget) *testHook {
	t.Helper()
	h := NewPushHook(NewHTTPServer("127.0.0.1:0", ag), sinkURL, hookDefaultInterval)
	th := &testHook{PushHook: h}
	if targets != nil {
		h.targets = targets
	}
	inner := h.handler
	h.handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		th.fetched = append(th.fetched, r.URL.RequestURI())
		inner.ServeHTTP(w, r)
	})
	h.logf = func(f string, a ...any) { th.logs = append(th.logs, fmt.Sprintf(f, a...)) }
	return th
}

func (th *testHook) fetchedCount(path string) int {
	n := 0
	for _, p := range th.fetched {
		if p == path {
			n++
		}
	}
	return n
}

func mustHookTarget(t *testing.T, path string) hookTarget {
	t.Helper()
	for _, tg := range hookTargets() {
		if tg.Path == path {
			return tg
		}
	}
	t.Fatalf("no hook target %s", path)
	return hookTarget{}
}

// mutableWhaleAgents: /api/v1/whale-flow answers a swappable body/code, every
// other upstream is dead (fast, deterministic degraded cards).
func mutableWhaleAgents(t *testing.T) (*Agents, func(body string, code int)) {
	t.Helper()
	stubExternalBases(t)
	var mu sync.Mutex
	body, code := whaleLiveFixture, http.StatusOK
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/whale-flow" {
			http.NotFound(w, r)
			return
		}
		mu.Lock()
		b, c := body, code
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(c)
		_, _ = w.Write([]byte(b))
	}))
	t.Cleanup(srv.Close)
	return NewAgents(NewBackendClient(srv.URL)), func(b string, c int) {
		mu.Lock()
		body, code = b, c
		mu.Unlock()
	}
}

var (
	whaleInflowFixture  = strings.Replace(whaleLiveFixture, `"direction":"outflow"`, `"direction":"inflow"`, 1)
	whaleNumbersFixture = strings.Replace(whaleLiveFixture, `"tx_count_24h":37`, `"tx_count_24h":41`, 1)
)

// equalFundingRates is the everyday Binance picture: majors sitting exactly
// on the 0.0100% default, so the widest |rate| is a five-way tie.
var equalFundingRates = map[string]string{"BTCUSDT": "0.00010000", "ETHUSDT": "0.00010000",
	"SOLUSDT": "0.00010000", "BNBUSDT": "0.00010000", "XRPUSDT": "0.00010000"}

// liveHookAgents is every source alive and stable, funding rates tied.
func liveHookAgents(t *testing.T) *Agents { return liveHookAgentsRates(t, equalFundingRates) }

// liveHookAgentsRates is liveHookAgentsAt built on the wall clock.
func liveHookAgentsRates(t *testing.T, rates map[string]string) *Agents {
	return liveHookAgentsAt(t, rates, time.Now())
}

// hookFixtureClock is the fixture's frozen Agents clock: the wall time held
// inside its hour, [:01, :58], so a read a minute later (the 61 s step of
// TestHookHashStableAcrossCalls) stays in the same hour. Every clock-judged
// wording on these cards is hour-aligned against the fixture's hourly Yahoo
// bars: the Forex week (isForexOpen), the pairs' bar age (barMaxAge) and
// gold's (fxGoldMaxAge 3h). On the wall clock a run crossing HH:00 flipped
// gold to "no bar in the last 3h" between two reads (2026-09-15, 20:00 UTC).
// Built on the real wall it is held at most 2 min from it: the digest judges
// the wall-stamped funding read against this clock (fundingMaxAge 15 min).
// A synthetic wall (the "built at HH:59:30" subtest) sits in the previous
// hour, up to ~62 min behind the stub's real-time funding stamps: their age
// is negative, so funding stays eligible on both reads (priority.go
// rankCandidate) and the stale branch is simply not exercised there. The
// previous hour is the safe side: the next one would put the clock up to
// 58 min ahead of the wall and age funding past 15 min (for any run before
// HH:43), switching the digest branch.
func hookFixtureClock(wall time.Time) time.Time {
	h := wall.UTC().Truncate(time.Hour)
	if lo := h.Add(time.Minute); wall.Before(lo) {
		return lo
	}
	if hi := h.Add(58 * time.Minute); wall.After(hi) {
		return hi
	}
	return wall.UTC()
}

// liveHookAgentsAt: Binance and Yahoo candles, Binance funding rates per
// symbol, and a backend that stamps captured_at with ITS request time on
// macro and liquidations — exactly like the real handlers
// (macro_handlers.go, funding_handlers.go). The Agents clock is frozen at
// hookFixtureClock(wall) and the last hourly Yahoo bar closes two hours
// before its hour, so every read sees the same bar ages.
func liveHookAgentsAt(t *testing.T, rates map[string]string, wall time.Time) *Agents {
	t.Helper()
	at := hookFixtureClock(wall)
	stubExternalBases(t)
	stubBinanceKlinesWave(t, binanceFetchLimit)
	stubYahooWave(t, at.Truncate(time.Hour).Add(-2*time.Hour), 600)
	prem := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// markPrice moves on every request, as the real one does; the card body
		// must not (TestHookHashStableAcrossCalls): the cluster band below
		// holds it inside, so only the position "inside" is served.
		mark := strconv.FormatFloat(118000+float64(time.Now().UnixMicro()%1000)/10, 'f', 2, 64)
		_, _ = w.Write([]byte(`{"lastFundingRate":"` + rates[r.URL.Query().Get("symbol")] + `","markPrice":"` + mark + `"}`))
	}))
	t.Cleanup(prem.Close)
	premiumIndexURL = prem.URL + "/?symbol="

	liqAt := time.Now().UTC().Add(-10 * time.Minute).Format(time.RFC3339)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		now := time.Now().UTC().Format(time.RFC3339)
		var body string
		switch r.URL.Path {
		case "/api/v1/macro":
			body = strings.Replace(macroLiveFixture, `"captured_at":"2026-09-15T04:44:32Z"`, `"captured_at":"`+now+`"`, 1)
		case "/api/v1/whale-flow":
			body = whaleLiveFixture
		case "/api/v1/narratives":
			body = narrativesFixture
		case "/api/v1/funding/liquidations":
			body = `{"captured_at":"` + now + `","feed":[
			  {"symbol":"BTCUSDT","side":"long_liq","qty":1,"price":118000,"usd_value":90000,"ts":"` + liqAt + `"},
			  {"symbol":"ETHUSDT","side":"short_liq","qty":10,"price":4500,"usd_value":45000,"ts":"` + liqAt + `"}],
			  "zones":[{"symbol":"BTCUSDT","price_band":"117900-118100","total_usd":90000,"count":1,"side":"long_liq"}]}`
		case "/api/v1/market/momentum":
			body = `{"baseline":"BTC","items":{"ETH":{"rs_7d":-2.4,"rs_30d":5.1}}}`
		case "/api/v1/market/mood-read":
			body = `{"read":"Calm tape.","source":"alphavizor-ai"}`
		default:
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	ag := NewAgents(NewBackendClient(srv.URL))
	ag.now = func() time.Time { return at }
	return ag
}

func hashOf(t *testing.T, agent string, body []byte, start, end time.Time) string {
	t.Helper()
	norm, err := hookNormalize(agent, body, start, end)
	if err != nil {
		t.Fatalf("normalize %s: %v", agent, err)
	}
	sum := sha256.Sum256(norm)
	return hex.EncodeToString(sum[:])
}

func nextWallSecond() {
	time.Sleep(time.Until(time.Now().Truncate(time.Second).Add(time.Second + 50*time.Millisecond)))
}

// ── FundingCard tie-break ────────────────────────────────────────────────────

// Equal |rates| resolve in fundingSymbols order: one body per 20 GETs (the
// map order used to pick a random "Widest skew", and with opposite signs past
// the thresholds a random semaphore), on funding and on digest/top that
// embed it.
func TestFundingTieDeterministic(t *testing.T) {
	cases := []struct {
		name     string
		rates    map[string]string
		sym, sem string
	}{
		{"five-way tie at 0.0100%", equalFundingRates, "BTCUSDT", "neutral"},
		{"equal |rate|, BTC negative first", map[string]string{"BTCUSDT": "-0.00030000", "ETHUSDT": "0.00030000",
			"SOLUSDT": "0.00001000", "BNBUSDT": "0.00001000", "XRPUSDT": "0.00001000"}, "BTCUSDT", "bullish"},
		// Equal |rate| no longer ties (stage 1, 2026-09-15): SOL -0.03% is 3×
		// its own threshold, ETH +0.03% 1×. The old pick showed ETH.
		{"equal |rate|, SOL 3× its threshold beats ETH 1×", map[string]string{"BTCUSDT": "0.00001000", "ETHUSDT": "0.00030000",
			"SOLUSDT": "-0.00030000", "BNBUSDT": "0.00001000", "XRPUSDT": "0.00001000"}, "SOLUSDT", "bullish"},
		{"equal ratio 1.00×, BTC first", map[string]string{"BTCUSDT": "0.00030000", "ETHUSDT": "0.00001000",
			"SOLUSDT": "0.00001000", "BNBUSDT": "0.00001000", "XRPUSDT": "-0.00010000"}, "BTCUSDT", "bearish"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			th := newTestHook(t, liveHookAgentsRates(t, tc.rates), "http://127.0.0.1:1", nil)
			for _, p := range []string{"/agents/funding", "/agents/digest", "/agents/top"} {
				tg := mustHookTarget(t, p)
				hashes := map[string]bool{}
				for i := 0; i < 20; i++ {
					s := time.Now()
					_, b := th.fetch(context.Background(), tg)
					hashes[hashOf(t, tg.Agent, b, s, time.Now())] = true
					if p != "/agents/funding" {
						continue
					}
					var env testEnvelope
					if err := json.Unmarshal(b, &env); err != nil {
						t.Fatal(err)
					}
					if env.Semaphore != tc.sem || env.Asset != strings.TrimSuffix(tc.sym, "USDT") || !strings.Contains(env.CardHTML, "· "+tc.sym+"\n") {
						t.Fatalf("GET %d: semaphore %s, asset %q; want %s on %s", i, env.Semaphore, env.Asset, tc.sem, tc.sym)
					}
				}
				if len(hashes) != 1 {
					t.Errorf("%s: %d different bodies over 20 GETs of unchanged data", p, len(hashes))
				}
			}
		})
	}
}

// With tied rates and nothing else changing, a second sweep sends nothing.
func TestHookLiveTiedRatesNoRepeat(t *testing.T) {
	sink := newHookSink(t, http.StatusCreated)
	th := newTestHook(t, liveHookAgents(t), sink.srv.URL, nil)
	th.digestInterval = th.interval
	ctx := context.Background()
	for i := 0; i < 3; i++ {
		th.sweep(ctx)
		got := sink.take()
		if i == 0 && len(got) != len(hookTargets()) {
			t.Fatalf("first sweep: %d events, want %d", len(got), len(hookTargets()))
		}
		if i > 0 && len(got) != 0 {
			t.Fatalf("sweep %d repeated %v", i+1, agentsOf(got))
		}
		nextWallSecond()
	}
}

// ── configuration ────────────────────────────────────────────────────────────

func TestHookOffWithoutURL(t *testing.T) {
	t.Setenv("DEMOBOT_HOOK_URL", "")
	if StartPushHookFromEnv(context.Background(), NewHTTPServer("127.0.0.1:0", deadAgents(t))) {
		t.Fatal("hook must stay off without DEMOBOT_HOOK_URL")
	}
	t.Setenv("DEMOBOT_HOOK_URL", "   ")
	if _, ok := hookConfigFromEnv(func(string, ...any) {}); ok {
		t.Fatal("a blank URL must keep the hook off")
	}
}

func TestHookRejectsInvalidURL(t *testing.T) {
	for _, raw := range []string{"127.0.0.1:8082/internal/agents/events", "ftp://127.0.0.1/x", "http://", "not a url", "/internal/agents/events"} {
		t.Setenv("DEMOBOT_HOOK_URL", raw)
		var logs []string
		if _, ok := hookConfigFromEnv(func(f string, a ...any) { logs = append(logs, fmt.Sprintf(f, a...)) }); ok {
			t.Errorf("%q must not start the hook", raw)
		}
		if len(logs) != 1 || !strings.Contains(logs[0], "not an http(s) URL") {
			t.Errorf("%q: logs %v", raw, logs)
		}
		if StartPushHookFromEnv(context.Background(), NewHTTPServer("127.0.0.1:0", deadAgents(t))) {
			t.Errorf("%q: StartPushHookFromEnv started", raw)
		}
	}
	for _, raw := range []string{"http://127.0.0.1:8082/internal/agents/events", "https://hooks.example/x"} {
		t.Setenv("DEMOBOT_HOOK_URL", raw)
		if cfg, ok := hookConfigFromEnv(func(string, ...any) {}); !ok || cfg.target != raw {
			t.Errorf("%q must be accepted", raw)
		}
	}
}

func TestHookIntervalConfig(t *testing.T) {
	t.Setenv("DEMOBOT_HOOK_URL", "http://127.0.0.1:8082/internal/agents/events")
	cases := []struct {
		interval, digest string
		wantI, wantD     time.Duration
	}{
		{"", "", 60 * time.Second, 5 * time.Minute},
		{"90s", "", 90 * time.Second, 5 * time.Minute},
		{"45", "10m", 45 * time.Second, 10 * time.Minute},
		{"10s", "", 30 * time.Second, 5 * time.Minute},
		{"5", "junk", 30 * time.Second, 5 * time.Minute},
		{"junk", "-1m", 60 * time.Second, 5 * time.Minute},
		{"10m", "", 10 * time.Minute, 10 * time.Minute}, // digest never below the sweep
		{"2m", "1m", 2 * time.Minute, 2 * time.Minute},
	}
	for _, c := range cases {
		t.Setenv("DEMOBOT_HOOK_INTERVAL", c.interval)
		t.Setenv("DEMOBOT_HOOK_DIGEST_INTERVAL", c.digest)
		cfg, ok := hookConfigFromEnv(func(string, ...any) {})
		if !ok || cfg.interval != c.wantI || cfg.digestInterval != c.wantD {
			t.Errorf("interval %q digest %q → %s / %s ok=%v, want %s / %s",
				c.interval, c.digest, cfg.interval, cfg.digestInterval, ok, c.wantI, c.wantD)
		}
	}
}

// ── auth header ──────────────────────────────────────────────────────────────

// hookSecret is the header value in the auth tests: it must reach the wire
// and never a log line.
const hookSecret = "s3cr3t-Token_42" // no ":" — the no-colon cases rely on it

// logSink collects logf lines (the hook may log from its own goroutine).
type logSink struct {
	mu    sync.Mutex
	lines []string
}

func (l *logSink) logf(f string, a ...any) {
	l.mu.Lock()
	l.lines = append(l.lines, fmt.Sprintf(f, a...))
	l.mu.Unlock()
}

func (l *logSink) all() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.lines...)
}

func (l *logSink) assertNoSecret(t *testing.T) {
	t.Helper()
	for _, line := range l.all() {
		if strings.Contains(line, "s3cr3t") {
			t.Errorf("log line leaks the header value: %q", line)
		}
	}
}

// DEMOBOT_HOOK_HEADER from env to wire: the stub receiver gets the header with
// the right name and value on every POST, Content-Type unchanged; the start
// line names the header only; no log line (start, POST, not-delivered) holds
// the value.
func TestHookAuthHeaderDelivered(t *testing.T) {
	cases := []struct{ env, name, label string }{
		{"Authorization: Bearer " + hookSecret, "Authorization", "Authorization"},
		{"  X-Api-Key :  " + hookSecret + ":tail  ", "X-Api-Key", "X-Api-Key"},  // trimmed; ":" inside the value kept
		{"x-api-key:" + hookSecret, "x-api-key", "X-Api-Key"},                   // known name printed canonical
		{"X-Site-Sig: " + hookSecret, "X-Site-Sig", "custom header (10 chars)"}, // unknown name works, not printed
	}
	for _, c := range cases {
		for _, code := range []int{http.StatusCreated, http.StatusUnauthorized} {
			sink := newHookSink(t, code)
			t.Setenv("DEMOBOT_HOOK_URL", sink.srv.URL)
			t.Setenv("DEMOBOT_HOOK_HEADER", c.env)
			var logs logSink
			h, ok := pushHookFromEnv(NewHTTPServer("127.0.0.1:0", deadAgents(t)), logs.logf)
			if !ok {
				t.Fatalf("%q: hook not started: %v", c.env, logs.all())
			}
			h.targets = []hookTarget{mustHookTarget(t, "/agents/fx"), mustHookTarget(t, "/agents/whale")}
			h.sweep(context.Background())

			heads := sink.headers()
			if len(heads) != 2 {
				t.Fatalf("%q/%d: %d POSTs, want 2", c.env, code, len(heads))
			}
			wantValue := strings.TrimSpace(strings.SplitN(c.env, ":", 2)[1])
			for _, hd := range heads {
				if got := hd.Values(c.name); len(got) != 1 || got[0] != wantValue {
					t.Errorf("%q/%d: header %s = %q, want [%q]", c.env, code, c.name, got, wantValue)
				}
				if ct := hd.Get("Content-Type"); ct != "application/json" {
					t.Errorf("%q/%d: Content-Type %q", c.env, code, ct)
				}
			}
			lines := logs.all()
			if len(lines) == 0 || !strings.Contains(lines[0], "auth header: "+c.label+" (value hidden)") {
				t.Errorf("%q: start line %v", c.env, lines)
			}
			if code == http.StatusUnauthorized && !strings.Contains(strings.Join(lines, "\n"), "not delivered") {
				t.Errorf("%q: 401 not logged: %v", c.env, lines)
			}
			logs.assertNoSecret(t)
		}
	}
}

// Empty (or blank) DEMOBOT_HOOK_HEADER: no auth header on the wire, the hook
// runs as before, the start line says "none".
func TestHookNoAuthHeaderWhenEmpty(t *testing.T) {
	for _, env := range []string{"", "   "} {
		sink := newHookSink(t, http.StatusCreated)
		t.Setenv("DEMOBOT_HOOK_URL", sink.srv.URL)
		t.Setenv("DEMOBOT_HOOK_HEADER", env)
		var logs logSink
		h, ok := pushHookFromEnv(NewHTTPServer("127.0.0.1:0", deadAgents(t)), logs.logf)
		if !ok {
			t.Fatalf("%q: hook not started: %v", env, logs.all())
		}
		h.targets = []hookTarget{mustHookTarget(t, "/agents/fx")}
		h.sweep(context.Background())
		heads := sink.headers()
		if len(heads) != 1 {
			t.Fatalf("%q: %d POSTs, want 1", env, len(heads))
		}
		for name := range heads[0] {
			switch name {
			case "Content-Type", "Content-Length", "Accept-Encoding", "User-Agent":
			default:
				t.Errorf("%q: unexpected header %s: %q", env, name, heads[0].Values(name))
			}
		}
		if lines := logs.all(); len(lines) == 0 || !strings.HasSuffix(lines[0], "auth header: none") {
			t.Errorf("%q: start line %v", env, lines)
		}
	}
}

// A malformed DEMOBOT_HOOK_HEADER keeps the hook off (like a bad URL), with
// one log line that never carries the value — even when the variable is the
// bare secret.
func TestHookRejectsInvalidHeader(t *testing.T) {
	t.Setenv("DEMOBOT_HOOK_URL", "http://127.0.0.1:8082/internal/agents/events")
	cases := []struct{ env, why string }{
		{"Bearer " + hookSecret, `no ":"`},
		{hookSecret, `no ":"`}, // the bare token pasted without a name
		{": Bearer " + hookSecret, "empty header name"},
		{"   :" + hookSecret, "empty header name"},
		{"X Api Key: " + hookSecret, "not a valid HTTP token"},
		{"Author(ization): " + hookSecret, "not a valid HTTP token"},
		{"Authorization: Bearer " + hookSecret + "\r\nX-Evil: 1", "control character"},
		{"Authorization: Bearer " + hookSecret + "\nX-Evil: 1", "control character"},
		{"Authorization: Bearer " + hookSecret + "\rX", "control character"},
		{"X-Api-Key: " + hookSecret + "\x01x", "control character"}, // env cannot hold NUL
		{"X-Api-Key: " + hookSecret + "\x7fx", "control character"},
		{"Authorization:", "empty value"},
		{"Authorization:   ", "empty value"},
		{"content-type: " + hookSecret, "cannot be overridden"},
		{"Host: " + hookSecret, "cannot be overridden"},
	}
	for _, c := range cases {
		t.Setenv("DEMOBOT_HOOK_HEADER", c.env)
		var logs logSink
		if _, ok := hookConfigFromEnv(logs.logf); ok {
			t.Errorf("%q must not start the hook", c.env)
		}
		lines := logs.all()
		if len(lines) != 1 || !strings.Contains(lines[0], "DEMOBOT_HOOK_HEADER") ||
			!strings.Contains(lines[0], "hook not started") || !strings.Contains(lines[0], c.why) {
			t.Errorf("%q: logs %v, want one line with %q", c.env, lines, c.why)
		}
		logs.assertNoSecret(t)

		var startLogs logSink
		if _, ok := pushHookFromEnv(NewHTTPServer("127.0.0.1:0", deadAgents(t)), startLogs.logf); ok {
			t.Errorf("%q: pushHookFromEnv started", c.env)
		}
		startLogs.assertNoSecret(t)
		if StartPushHookFromEnv(context.Background(), NewHTTPServer("127.0.0.1:0", deadAgents(t))) {
			t.Errorf("%q: StartPushHookFromEnv started", c.env)
		}
	}
}

// A bare token with a ":" inside parses as "Name: value" — accepted (it is a
// valid header) or refused (empty value), but no log line may print the name,
// which is the first half of the secret.
func TestHookUnknownHeaderNameNotLogged(t *testing.T) {
	cases := []struct {
		env, name, rest string
		ok              bool
		label           string
	}{
		{"abc:def", "abc", "def", true, "custom header (3 chars)"},
		{"user:pass", "user", "pass", true, "custom header (4 chars)"},
		{"AKIA123:secretpart", "AKIA123", "secretpart", true, "custom header (7 chars)"},
		{"AKIA123:", "AKIA123", "", false, "custom header (7 chars)"},
		{"AKIA123:x\r\ny", "AKIA123", "", false, "custom header (7 chars)"},
	}
	for _, c := range cases {
		sink := newHookSink(t, http.StatusCreated)
		t.Setenv("DEMOBOT_HOOK_URL", sink.srv.URL)
		t.Setenv("DEMOBOT_HOOK_HEADER", c.env)
		var logs logSink
		h, ok := pushHookFromEnv(NewHTTPServer("127.0.0.1:0", deadAgents(t)), logs.logf)
		if ok != c.ok {
			t.Fatalf("%q: started=%v, want %v (%v)", c.env, ok, c.ok, logs.all())
		}
		if ok {
			h.targets = []hookTarget{mustHookTarget(t, "/agents/fx")}
			h.sweep(context.Background())
			if hd := sink.headers(); len(hd) != 1 || hd[0].Get(c.name) != c.rest {
				t.Errorf("%q: custom header not delivered: %v", c.env, hd)
			}
		}
		lines := logs.all()
		if len(lines) == 0 || !strings.Contains(lines[0], c.label) {
			t.Errorf("%q: first line %v, want %q", c.env, lines, c.label)
		}
		eventRe := regexp.MustCompile(`event=\S+`) // the hex hash may contain "abc"/"def" by chance
		for _, line := range lines {
			line = eventRe.ReplaceAllString(line, "event=…")
			if strings.Contains(line, c.name) || (c.rest != "" && strings.Contains(line, c.rest)) {
				t.Errorf("%q: log line leaks the secret: %q", c.env, line)
			}
		}
	}
}

// Every header the hook or net/http sets itself is refused, in any case.
func TestHookReservedHeadersRefused(t *testing.T) {
	t.Setenv("DEMOBOT_HOOK_URL", "http://127.0.0.1:8082/internal/agents/events")
	for _, name := range []string{"Content-Type", "Content-Length", "Host", "Content-Encoding",
		"Transfer-Encoding", "Connection", "Expect", "TE", "Trailer", "Upgrade", "User-Agent", "Accept-Encoding"} {
		for _, n := range []string{name, strings.ToLower(name)} {
			t.Setenv("DEMOBOT_HOOK_HEADER", n+": "+hookSecret)
			var logs logSink
			if _, ok := hookConfigFromEnv(logs.logf); ok {
				t.Errorf("%s must be refused", n)
			}
			if lines := logs.all(); len(lines) != 1 || !strings.Contains(lines[0], "cannot be overridden") {
				t.Errorf("%s: logs %v", n, lines)
			}
			logs.assertNoSecret(t)
		}
	}
	for _, name := range []string{"X-Webhook-Secret", "Api-Key", "X-Custom-Auth"} {
		t.Setenv("DEMOBOT_HOOK_HEADER", name+": "+hookSecret)
		if _, ok := hookConfigFromEnv(func(string, ...any) {}); !ok {
			t.Errorf("%s must be accepted", name)
		}
	}
}

// A site that echoes the refused header in its answer must not put the value
// into the log: the logged answer is masked — also when the value straddles
// the end of the 300-byte snippet (its first bytes inside, the rest cut), and
// when it is echoed twice with the second echo at or past the cut.
func TestHookAnswerEchoMasked(t *testing.T) {
	const prefix = "bad auth: "
	mask := strings.Repeat("*", len(hookSecret))
	pads := []int{0, hookSnippetLen - len(prefix) - 3}
	// Second echo starts at 300 - k after " again " (7 bytes): straddling the
	// cut or just past it.
	for k := -3; k <= len(hookSecret)+2; k++ {
		pads = append(pads, hookSnippetLen-k-len(prefix)-len(hookSecret)-7)
	}
	for i, pad := range pads {
		for _, code := range []int{http.StatusUnauthorized, http.StatusBadRequest} {
			twice := i >= 2
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(code)
				v := r.Header.Get("X-Api-Key")
				_, _ = fmt.Fprintf(w, "%s%s%s", strings.Repeat("x", pad), prefix, v)
				if twice {
					_, _ = fmt.Fprintf(w, " again %s tail", v)
				}
			}))
			t.Setenv("DEMOBOT_HOOK_URL", srv.URL)
			t.Setenv("DEMOBOT_HOOK_HEADER", "X-Api-Key: "+hookSecret)
			var logs logSink
			h, ok := pushHookFromEnv(NewHTTPServer("127.0.0.1:0", deadAgents(t)), logs.logf)
			if !ok {
				t.Fatalf("hook not started: %v", logs.all())
			}
			h.targets = []hookTarget{mustHookTarget(t, "/agents/fx")}
			h.sweep(context.Background())
			srv.Close()
			all := strings.Join(logs.all(), "\n")
			if !strings.Contains(all, prefix) {
				t.Errorf("pad %d, %d: answer not logged: %v", pad, code, logs.all())
			}
			if pad == 0 && !strings.Contains(all, prefix+mask) {
				t.Errorf("pad %d, %d: echoed value not masked: %v", pad, code, logs.all())
			}
			if strings.Contains(all, hookSecret[:3]) { // "s3c": not hex, cannot come from an event id
				t.Errorf("pad %d, %d: log holds part of the value: %v", pad, code, logs.all())
			}
			logs.assertNoSecret(t)
		}
	}
}

func TestParseHookHeader(t *testing.T) {
	cases := []struct{ raw, name, value string }{
		{"Authorization: Bearer xyz", "Authorization", "Bearer xyz"},
		{"X-Api-Key:xyz", "X-Api-Key", "xyz"},
		{" X-Api-Key \t:\t a:b:c ", "X-Api-Key", "a:b:c"},
		{"X-Sig: a\tb", "X-Sig", "a\tb"}, // tab is a legal value byte
		{"x~!#$%&'*+-.^_`|: v", "x~!#$%&'*+-.^_`|", "v"},
	}
	for _, c := range cases {
		n, v, err := parseHookHeader(c.raw)
		if err != nil || n != c.name || v != c.value {
			t.Errorf("%q → %q %q %v, want %q %q", c.raw, n, v, err, c.name, c.value)
		}
	}
}

// The v1 address set: no risk, no chart, no showcase, no tf/assets variants.
func TestHookTargetSet(t *testing.T) {
	targets := hookTargets()
	paths := map[string]hookTarget{}
	for _, tg := range targets {
		if _, dup := paths[tg.Path]; dup {
			t.Errorf("duplicate target %s", tg.Path)
		}
		paths[tg.Path] = tg
		if tg.Agent == keyRisk || strings.Contains(tg.Path, "chart") || strings.Contains(tg.Path, "showcase") ||
			strings.Contains(tg.Path, "tf=") || strings.Contains(tg.Path, "assets=") {
			t.Errorf("target %s must not be swept in v1", tg.Path)
		}
		if tg.slow != (tg.Agent == keyDigest || tg.Agent == keyTop || tg.Agent == keyNews) {
			t.Errorf("%s: slow=%v", tg.Path, tg.slow)
		}
	}
	want := []string{"/agents/digest", "/agents/top", "/agents/fx", "/agents/macro", "/agents/macro?asset=btc",
		"/agents/macro?asset=gold", "/agents/whale", "/agents/funding", "/agents/news", "/agents/gold", "/agents/momentum"}
	for _, a := range []string{keyMomentum, keyTrend, keySR, keyVol} {
		for _, k := range []string{"btc", "eth", "eurusd", "gbpusd", "usdjpy", "xauusd"} {
			want = append(want, "/agents/"+a+"?asset="+k)
		}
	}
	for _, p := range want {
		if _, ok := paths[p]; !ok {
			t.Errorf("missing target %s", p)
		}
	}
	if len(targets) != len(want) {
		t.Errorf("%d targets, want %d", len(targets), len(want))
	}
	if tg := paths["/agents/trend?asset=xauusd"]; tg.Asset != "XAUUSD" || tg.Agent != keyTrend {
		t.Errorf("asset label: %+v", tg)
	}
	// Every address is a real read: 200 or the honest 503, never a 4xx.
	th := newTestHook(t, deadAgents(t), "http://127.0.0.1:1", nil)
	for _, tg := range targets {
		if st, _ := th.fetch(context.Background(), tg); st != 200 && st != 503 {
			t.Errorf("%s answered %d", tg.Path, st)
		}
	}
}

// ── sending ──────────────────────────────────────────────────────────────────

// First sweep sends every address; an unchanged sweep sends nothing (with the
// request-time stamps of the degraded cards moving underneath); one changed
// source sends exactly the addresses whose body shows it.
func TestHookSendsOnlyOnChange(t *testing.T) {
	ag, setWhale := mutableWhaleAgents(t)
	sink := newHookSink(t, http.StatusCreated)
	th := newTestHook(t, ag, sink.srv.URL, nil)
	th.digestInterval = th.interval
	ctx := context.Background()

	th.sweep(ctx)
	if got := sink.take(); len(got) != len(hookTargets()) {
		t.Fatalf("first sweep sent %d events, want one per address (%d)", len(got), len(hookTargets()))
	}
	for _, l := range th.logs {
		if strings.Contains(l, "answered") {
			t.Errorf("unexpected log: %s", l)
		}
	}
	nextWallSecond()
	th.sweep(ctx)
	if got := sink.take(); len(got) != 0 {
		t.Fatalf("unchanged sweep sent %v", agentsOf(got))
	}

	setWhale(whaleInflowFixture, 200)
	th.sweep(ctx)
	sent := agentsOf(sink.take())
	if sent[keyWhale] != 1 || sent[keyDigest] != 1 {
		t.Fatalf("whale change must send whale and digest, sent %v", sent)
	}
	for a := range sent {
		if a != keyWhale && a != keyDigest && a != keyTop {
			t.Errorf("whale change sent unrelated agent %s", a)
		}
	}
}

// state_changed: null without a delivered baseline and on number-only
// changes; {from,to} on a semaphore or ok flip, measured against the last
// DELIVERED body (a 4xx-rejected one does not move the baseline).
func TestHookStateChanged(t *testing.T) {
	ag, setWhale := mutableWhaleAgents(t)
	sink := newHookSink(t, http.StatusCreated)
	th := newTestHook(t, ag, sink.srv.URL, []hookTarget{mustHookTarget(t, "/agents/whale")})
	ctx := context.Background()

	step := func(label string, want *hookStateChange) hookEvent {
		t.Helper()
		th.sweep(ctx)
		got := sink.take()
		if len(got) != 1 {
			t.Fatalf("%s: %d events, want 1", label, len(got))
		}
		ev := got[0]
		switch {
		case want == nil && ev.StateChanged != nil:
			t.Errorf("%s: state_changed %+v, want null", label, *ev.StateChanged)
		case want != nil && (ev.StateChanged == nil || *ev.StateChanged != *want):
			t.Errorf("%s: state_changed %+v, want %+v", label, ev.StateChanged, *want)
		}
		return ev
	}

	step("first delivery (no baseline)", nil)
	setWhale(whaleNumbersFixture, 200)
	step("numbers only", nil)
	setWhale(whaleInflowFixture, 200)
	step("outflow → inflow", &hookStateChange{From: "bullish", To: "bearish"})
	setWhale(`{}`, 500)
	ev := step("source down (503)", &hookStateChange{From: "bearish", To: hookUnavailable})
	if ev.DataAsOf != nil || !strings.HasPrefix(ev.EventID, "whale:-:-:-:") {
		t.Errorf("503 event: data_as_of %v, id %s", ev.DataAsOf, ev.EventID)
	}
	var body map[string]any
	if err := json.Unmarshal(ev.Data, &body); err != nil || body["ok"] != false {
		t.Errorf("503 event data must be the degraded body, got %s", ev.Data)
	}

	sink.setCode(http.StatusUnprocessableEntity)
	setWhale(whaleLiveFixture, 200)
	step("recovered, rejected by the site", &hookStateChange{From: hookUnavailable, To: "bullish"})
	sink.setCode(http.StatusCreated)
	setWhale(whaleNumbersFixture, 200)
	step("baseline is the last delivered body", &hookStateChange{From: hookUnavailable, To: "bullish"})
	th.sweep(ctx)
	if n := sink.count(); n != 0 {
		t.Errorf("unchanged after delivery: %d events", n)
	}
}

var hookIDRe = regexp.MustCompile(`^momentum:BTC:(-|tf=1d):(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ):([0-9a-f]{8})$`)

// event_id = agent:asset:params:data_as_of:hash8, and a tf variant differs
// from the default address even when both read the same bar time.
func TestHookEventShapeAndID(t *testing.T) {
	stubExternalBases(t)
	stubBinanceKlinesWave(t, binanceFetchLimit)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	sink := newHookSink(t, http.StatusCreated)
	tf := hookTarget{Agent: keyMomentum, Asset: "BTC", Params: map[string]string{"tf": "1d"}, Path: "/agents/momentum?asset=btc&tf=1d"}
	th := newTestHook(t, ag, sink.srv.URL, []hookTarget{mustHookTarget(t, "/agents/momentum?asset=btc"), tf})
	th.sweep(context.Background())

	sink.mu.Lock()
	raws, ctypes := append([]string{}, sink.raws...), append([]string{}, sink.ctypes...)
	sink.mu.Unlock()
	evs := sink.take()
	if len(evs) != 2 {
		t.Fatalf("%d events, want 2", len(evs))
	}
	params := map[string]bool{}
	for i, ev := range evs {
		m := hookIDRe.FindStringSubmatch(ev.EventID)
		if m == nil {
			t.Fatalf("event_id %q does not match agent:asset:params:data_as_of:hash8", ev.EventID)
		}
		params[m[1]] = true
		var data struct {
			DataAsOf string `json:"data_as_of"`
			OK       bool   `json:"ok"`
		}
		if err := json.Unmarshal(ev.Data, &data); err != nil || !data.OK {
			t.Fatalf("data must be the live envelope object: %s", ev.Data)
		}
		if ev.DataAsOf == nil || *ev.DataAsOf != data.DataAsOf || m[2] != data.DataAsOf {
			t.Errorf("data_as_of: event %v, id %s, body %s", ev.DataAsOf, m[2], data.DataAsOf)
		}
		if want := hashOf(t, keyMomentum, ev.Data, time.Time{}, time.Time{})[:8]; m[3] != want {
			t.Errorf("hash8 %s, want %s", m[3], want)
		}
		if _, err := time.Parse(time.RFC3339, ev.SentAt); err != nil {
			t.Errorf("sent_at %q: %v", ev.SentAt, err)
		}
		if ev.Agent != keyMomentum || ev.Asset != "BTC" {
			t.Errorf("agent/asset %q/%q", ev.Agent, ev.Asset)
		}
		if m[1] == "tf=1d" && ev.Params["tf"] != "1d" {
			t.Errorf("params %v", ev.Params)
		}
		if ctypes[i] != "application/json" {
			t.Errorf("Content-Type %q", ctypes[i])
		}
		for _, frag := range []string{`"data":{`, `"state_changed":null`} {
			if !strings.Contains(raws[i], frag) {
				t.Errorf("raw event lacks %s: %.200s", frag, raws[i])
			}
		}
		if m[1] == "-" && !strings.Contains(raws[i], `"params":{}`) {
			t.Errorf("params must be {} when none: %.300s", raws[i])
		}
	}
	if !params["-"] || !params["tf=1d"] {
		t.Errorf("default and tf=1d must differ in the params segment, got %v", params)
	}
}

// Every answer code the site listed. No event_id is ever POSTed twice: an
// attempt counts as sent. Delivered (200/201/409) and rejected
// (400/413/415/422) leave the address free, so the next change goes out on
// the next sweep; the rest pause the address (next change no sooner than 2
// sweeps later). One log line per POST, no data.
func TestHookResponseCodes(t *testing.T) {
	for _, c := range []int{200, 201, 409, 400, 413, 415, 422, 401, 408, 500} {
		t.Run(fmt.Sprint(c), func(t *testing.T) {
			ag, setWhale := mutableWhaleAgents(t)
			whale := mustHookTarget(t, "/agents/whale")
			sink := newHookSink(t, c)
			th := newTestHook(t, ag, sink.srv.URL, []hookTarget{whale})
			ctx := context.Background()
			var perSweep []int
			var ids []string
			for i := 0; i < 5; i++ {
				if i == 1 {
					setWhale(whaleNumbersFixture, 200) // a new body while paused (or not)
				}
				th.sweep(ctx)
				evs := sink.take()
				perSweep = append(perSweep, len(evs))
				for _, e := range evs {
					ids = append(ids, e.EventID)
				}
			}
			delivered := c == 200 || c == 201 || c == 409
			rejected := c == 400 || c == 413 || c == 415 || c == 422
			want := []int{1, 0, 1, 0, 0} // paused one sweep, then the new body
			if delivered || rejected {
				want = []int{1, 1, 0, 0, 0}
			}
			if fmt.Sprint(perSweep) != fmt.Sprint(want) {
				t.Errorf("sends per sweep %v, want %v", perSweep, want)
			}
			if len(ids) != 2 || ids[0] == ids[1] {
				t.Errorf("event ids %v: two different events, never a repeat", ids)
			}
			if st := th.state[whale.Path]; st.hasState != delivered {
				t.Errorf("hasState=%v, want %v (only a delivery sets the baseline)", st.hasState, delivered)
			}
			if len(th.logs) != 2 {
				t.Fatalf("%d log lines for 2 POSTs: %v", len(th.logs), th.logs)
			}
			for _, l := range th.logs {
				if !strings.Contains(l, "event=whale:") || !strings.Contains(l, fmt.Sprintf("→ %d (", c)) || !strings.Contains(l, "ms)") {
					t.Errorf("log line shape: %s", l)
				}
				if !delivered && !strings.Contains(l, fmt.Sprintf("sink answer %d", c)) {
					t.Errorf("non-delivery must log the answer head: %s", l)
				}
				if strings.Contains(l, "Whale Flow Agent") || strings.Contains(l, "card_html") {
					t.Errorf("log must not carry data: %s", l)
				}
			}
		})
	}
}

// One address whose POST hangs (5s timeout, shortened here) must not hold
// back the others: they are delivered in the same sweep, the hung event is
// never POSTed again, and the sweep is not stopped.
func TestHookHungAddressDoesNotStarve(t *testing.T) {
	ag, _ := mutableWhaleAgents(t)
	sink := newHookSink(t, http.StatusCreated)
	sink.hang[keyWhale] = true
	targets := []hookTarget{mustHookTarget(t, "/agents/whale"), mustHookTarget(t, "/agents/fx"), mustHookTarget(t, "/agents/funding")}
	th := newTestHook(t, ag, sink.srv.URL, targets)
	th.client.Timeout = 200 * time.Millisecond
	ctx := context.Background()

	th.sweep(ctx)
	if got := agentsOf(sink.take()); got[keyWhale] != 1 || got[keyFX] != 1 || got[keyFunding] != 1 {
		t.Fatalf("first sweep: %v — fx and funding must go out despite the hung whale", got)
	}
	for i := 0; i < 11; i++ {
		th.sweep(ctx)
	}
	if n := sink.count(); n != 0 {
		t.Errorf("unchanged bodies after the first sweep: %d more POSTs (the hung event_id must not repeat)", n)
	}
	if th.netFails != 0 || th.pause != 0 {
		t.Errorf("a timeout is per address, not a sweep stop: netFails %d, pause %d", th.netFails, th.pause)
	}
	if st := th.state["/agents/whale"]; st.hasState || st.fails != 1 {
		t.Errorf("whale state %+v: not delivered, one failure", *st)
	}
	if st := th.state["/agents/fx"]; !st.hasState {
		t.Error("fx must be delivered")
	}
	if !strings.Contains(th.logs[0], "error") || !strings.Contains(th.logs[0], "address paused") {
		t.Errorf("timeout log: %s", th.logs[0])
	}
}

// The start address moves one step per sweep.
func TestHookRotatesStart(t *testing.T) {
	ag, _ := mutableWhaleAgents(t)
	sink := newHookSink(t, http.StatusCreated)
	targets := []hookTarget{mustHookTarget(t, "/agents/whale"), mustHookTarget(t, "/agents/fx"), mustHookTarget(t, "/agents/funding")}
	th := newTestHook(t, ag, sink.srv.URL, targets)
	for k := 0; k < 4; k++ {
		before := len(th.fetched)
		th.sweep(context.Background())
		if first, want := th.fetched[before], targets[k%len(targets)].Path; first != want {
			t.Errorf("sweep %d starts at %s, want %s", k, first, want)
		}
	}
}

// An unreachable endpoint stops the sweep at once and pauses whole sweeps;
// the attempted event is not repeated, the untried ones go out later.
func TestHookConnectionRefused(t *testing.T) {
	ag, _ := mutableWhaleAgents(t)
	targets := []hookTarget{mustHookTarget(t, "/agents/whale"), mustHookTarget(t, "/agents/fx")}
	th := newTestHook(t, ag, "http://127.0.0.1:1/internal/agents/events", targets)
	ctx := context.Background()

	th.sweep(ctx) // n=0: whale first, refused → stop, pause 1 sweep
	if th.fetchedCount("/agents/whale") != 1 || th.fetchedCount("/agents/fx") != 0 {
		t.Errorf("the sweep must stop at the refused POST, fetched %v", th.fetched)
	}
	if len(th.logs) != 1 || !strings.Contains(th.logs[0], "unreachable") {
		t.Errorf("logs: %v", th.logs)
	}
	th.sweep(ctx) // paused
	if len(th.fetched) != 1 {
		t.Errorf("paused sweep fetched %v", th.fetched[1:])
	}
	th.sweep(ctx) // n=2: whale (same body, already attempted → no POST), fx → refused
	if len(th.logs) != 2 || !strings.Contains(th.logs[1], "event=fx:") {
		t.Errorf("third sweep must try only fx, logs %v", th.logs)
	}
	if th.state["/agents/whale"].hasState {
		t.Error("an undelivered event is not a state baseline")
	}
}

// digest, top and news run on DEMOBOT_HOOK_DIGEST_INTERVAL (default 5m →
// every 5th sweep at 60s); every other address every sweep.
func TestHookDigestCadence(t *testing.T) {
	sink := newHookSink(t, http.StatusCreated)
	th := newTestHook(t, deadAgents(t), sink.srv.URL, nil)
	for i := 0; i < 6; i++ {
		th.sweep(context.Background())
	}
	d, tp, n, w := th.fetchedCount("/agents/digest"), th.fetchedCount("/agents/top"),
		th.fetchedCount("/agents/news"), th.fetchedCount("/agents/whale")
	if d != 2 || tp != 2 || n != 2 || w != 6 {
		t.Errorf("fetched digest %d, top %d, news %d, whale %d over 6 sweeps; want 2, 2, 2, 6", d, tp, n, w)
	}
	th.digestInterval = th.interval
	if th.digestEvery() != 1 {
		t.Errorf("digest interval = sweep interval must run every sweep, got every %d", th.digestEvery())
	}
}

func TestHookRunStopsWithContext(t *testing.T) {
	sink := newHookSink(t, http.StatusCreated)
	th := newTestHook(t, deadAgents(t), sink.srv.URL, []hookTarget{mustHookTarget(t, "/agents/fx")})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { th.Run(ctx); close(done) }()
	deadline := time.Now().Add(5 * time.Second)
	for sink.count() == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	cancel()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Run did not stop with its context")
	}
	if sink.count() != 1 {
		t.Errorf("Run must sweep at start: %d events", sink.count())
	}
}

// ── what "changed" means ─────────────────────────────────────────────────────

// Two sequential reads of every address with the source data unchanged but
// the request clock moved (wall seconds for funding / the backend captured_at,
// the agents clock for the digest sweep) → the same hash. The raw bodies of
// the request-time readers really differ, so the masking is what makes them
// equal. Both reads fall in one hour of the fixture clock (hookFixtureClock)
// — also when the fixture is built at HH:59:30, the run that failed on the
// wall clock.
func TestHookHashStableAcrossCalls(t *testing.T) {
	t.Run("wall clock", func(t *testing.T) { hookHashStableAcrossCalls(t, liveHookAgents(t)) })
	t.Run("built at HH:59:30", func(t *testing.T) {
		wall := time.Now().UTC().Truncate(time.Hour).Add(-30 * time.Second)
		hookHashStableAcrossCalls(t, liveHookAgentsAt(t, equalFundingRates, wall))
	})
}

func hookHashStableAcrossCalls(t *testing.T, ag *Agents) {
	base := ag.clock()
	if base.Add(61*time.Second).Truncate(time.Hour) != base.Truncate(time.Hour) {
		t.Fatalf("fixture clock %s: the 61 s step leaves its hour", base.Format(time.RFC3339))
	}
	clk := &stepClock{t: base}
	ag.now = clk.now
	th := newTestHook(t, ag, "http://127.0.0.1:1", nil)
	ctx := context.Background()

	type read struct {
		status int
		body   []byte
		hash   string
	}
	pass := func() map[string]read {
		out := map[string]read{}
		for _, tg := range hookTargets() {
			start := time.Now()
			st, body := th.fetch(ctx, tg)
			out[tg.Path] = read{st, body, hashOf(t, tg.Agent, body, start, time.Now())}
		}
		return out
	}
	a := pass()
	nextWallSecond()
	clk.set(base.Add(61 * time.Second))
	b := pass()

	for _, tg := range hookTargets() {
		ra, rb := a[tg.Path], b[tg.Path]
		if ra.status != rb.status {
			t.Errorf("%s: status %d → %d", tg.Path, ra.status, rb.status)
		}
		if ra.hash != rb.hash {
			t.Errorf("%s: hash moved with unchanged data\nA: %s\nB: %s", tg.Path, ra.body, rb.body)
		}
	}
	for _, p := range []string{"/agents/funding", "/agents/macro", "/agents/macro?asset=gold", "/agents/digest", "/agents/trend?asset=btc", "/agents/gold", "/agents/top"} {
		if a[p].status != 200 {
			t.Errorf("%s must be live in this fixture, got %d: %s", p, a[p].status, a[p].body)
		}
	}
	// The fixture's markPrice moves on every request: the funding body must
	// carry the cluster's position (not a price or distance) and stay put.
	if fb := string(a["/agents/funding"].body); !strings.Contains(fb, "mark price is inside the band") ||
		!strings.Contains(fb, `"band_vs_mark":"inside"`) || strings.Contains(fb, "mark_price") {
		t.Errorf("/agents/funding: expected the cluster position against a moving mark price: %.800s", fb)
	}
	for _, p := range []string{"/agents/funding", "/agents/macro", "/agents/digest"} {
		if string(a[p].body) == string(b[p].body) {
			t.Errorf("%s: raw bodies are equal — the test did not move the request clock", p)
		}
	}
	// The Momentum stage-1 fields are data (bar closes, a threshold state),
	// not request time: they must be present here and left unmasked.
	for _, p := range []string{"/agents/momentum", "/agents/momentum?asset=btc", "/agents/momentum?asset=eurusd"} {
		r := a[p]
		if r.status != 200 {
			t.Errorf("%s must be live in this fixture, got %d: %s", p, r.status, r.body)
			continue
		}
		s := string(r.body)
		if p == "/agents/momentum" && !strings.Contains(s, "last bar ") { // composite fact only
			t.Errorf("%s: expected the \"last bar … UTC\" fact: %.400s", p, s)
		}
		if p == "/agents/momentum" && (!strings.Contains(s, `"freshness":"`) || !strings.Contains(s, `"results":[{"asset"`) ||
			!regexp.MustCompile(`"results":\[\{[^]]*"data_as_of":"\d{4}-`).MatchString(s)) {
			t.Errorf("%s: expected results[].data_as_of and freshness: %.600s", p, s)
		}
		norm, err := hookNormalize(keyMomentum, r.body, time.Time{}, time.Time{})
		if err != nil || string(norm) == "" || strings.Contains(string(norm), `"data_as_of":"*"`) {
			t.Errorf("%s: momentum stamps must not be masked: %s", p, norm)
		}
	}
}

// The masks cover exactly the request-clock fields: shifting them keeps the
// hash; changing a reading does not.
func TestHookNormalizeMasks(t *testing.T) {
	ag := liveHookAgents(t)
	th := newTestHook(t, ag, "http://127.0.0.1:1", nil)
	ctx := context.Background()
	decode := func(b []byte) map[string]any {
		var m map[string]any
		if err := json.Unmarshal(b, &m); err != nil {
			t.Fatal(err)
		}
		return m
	}
	encode := func(m map[string]any) []byte {
		b, err := encodeJSON(m)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	zero := time.Time{}

	// Funding: data_as_of and the footer are the request time — always masked.
	_, fb := th.fetch(ctx, mustHookTarget(t, "/agents/funding"))
	f := decode(fb)
	old := f["data_as_of"].(string)
	ot, _ := time.Parse(time.RFC3339, old)
	nt := ot.Add(2*time.Hour + 7*time.Minute)
	f["data_as_of"] = nt.Format(time.RFC3339)
	f["card_html"] = strings.Replace(f["card_html"].(string), ot.Format("2006-01-02 15:04"), nt.Format("2006-01-02 15:04"), 1)
	if hashOf(t, keyFunding, fb, zero, zero) != hashOf(t, keyFunding, encode(f), zero, zero) {
		t.Error("funding: a moved request time must not change the hash")
	}
	f["facts"].([]any)[0] = "Widest skew: BTCUSDT +0.0500%/8h (longs pay shorts)"
	if hashOf(t, keyFunding, fb, zero, zero) == hashOf(t, keyFunding, encode(f), zero, zero) {
		t.Error("funding: a changed fact must change the hash")
	}

	// Macro: captured_at and the F&G age follow the backend request.
	_, mb := th.fetch(ctx, mustHookTarget(t, "/agents/macro"))
	m := decode(mb)
	mm := m["macro"].(map[string]any)
	mm["freshness"].(map[string]any)["captured_at"] = "2030-01-01T00:00:00Z"
	if fg, ok := mm["fear_greed"].(map[string]any); ok {
		fg["age_hours"] = 123.45
	} else {
		t.Fatal("fixture must carry fear_greed")
	}
	if hashOf(t, keyMacro, mb, zero, zero) != hashOf(t, keyMacro, encode(m), zero, zero) {
		t.Error("macro: captured_at / age_hours must not change the hash")
	}
	mm["rule_score"] = 12
	if hashOf(t, keyMacro, mb, zero, zero) == hashOf(t, keyMacro, encode(m), zero, zero) {
		t.Error("macro: a changed rule score must change the hash")
	}

	// A candle stamp is data, never masked.
	_, tb := th.fetch(ctx, mustHookTarget(t, "/agents/trend?asset=btc"))
	tr := decode(tb)
	tr["data_as_of"] = "2020-01-01T00:00:00Z"
	if hashOf(t, keyTrend, tb, zero, zero) == hashOf(t, keyTrend, encode(tr), zero, zero) {
		t.Error("trend: a new bar close must change the hash")
	}

	// Known edge (docs "What counts as a change"): a real stamp that falls
	// inside the call window of a request-stamp agent is masked on that
	// sweep and not on the next — the same data then goes out twice, never
	// lost.
	T := time.Date(2026, 9, 15, 8, 0, 0, 0, time.UTC)
	body := []byte(`{"ok":true,"semaphore":"bullish","data_as_of":"2026-09-15T08:00:00Z","card_html":"x AlphaVizor · 2026-09-15 08:00 UTC"}`)
	if hashOf(t, keyTop, body, T.Add(-2*time.Second), T.Add(3*time.Second)) == hashOf(t, keyTop, body, T.Add(58*time.Second), T.Add(63*time.Second)) {
		t.Error("in-window edge: expected the documented one-time duplicate")
	}

	// Clock-counted ages inside text.
	for in, want := range map[string]string{
		"Crypto Fear & Greed 69 (Greed): stale, last update Sep 13 (52h ago) · not in the rule score": "(* ago)",
		"Crypto Fear & Greed 69 (Greed): stale, last update Sep 1 (14d ago) · not in the rule score":  "(* ago)",
	} {
		if got := hookMaskStrings(in).(string); !strings.Contains(got, want) {
			t.Errorf("mask %q → %q, want %s", in, got, want)
		}
	}
	if s := "Liquidations 1h: $90K longs vs $45K shorts"; hookMaskStrings(s) != s {
		t.Error("unrelated text must pass unchanged")
	}
}

// data is the GET body: byte-for-byte on stamped reads, and for every address
// equal under the same normalization (request-time fields differ by nature).
func TestHookDataEqualsGET(t *testing.T) {
	ag := liveHookAgents(t)
	_, srv := newTestAPI(t, ag, true)
	sink := newHookSink(t, http.StatusCreated)
	th := newTestHook(t, ag, sink.srv.URL, nil)
	start := time.Now()
	th.sweep(context.Background())
	end := time.Now()

	byPath := map[string]hookEvent{}
	for _, ev := range sink.take() {
		for _, tg := range hookTargets() {
			if tg.Agent == ev.Agent && tg.Asset == ev.Asset {
				byPath[tg.Path] = ev
			}
		}
	}
	if len(byPath) != len(hookTargets()) {
		t.Fatalf("%d events matched, want %d", len(byPath), len(hookTargets()))
	}
	exact := map[string]bool{"/agents/trend?asset=btc": true, "/agents/sr?asset=eurusd": true, "/agents/vol?asset=eth": true, "/agents/whale": true}
	for _, tg := range hookTargets() {
		ev := byPath[tg.Path]
		if len(ev.Data) == 0 || ev.Data[0] != '{' {
			t.Errorf("%s: data must be a JSON object, got %.40s", tg.Path, ev.Data)
			continue
		}
		gs := time.Now()
		_, _, body := httpGet(t, srv.URL+tg.Path)
		ge := time.Now()
		body = []byte(strings.TrimRight(string(body), "\n"))
		if exact[tg.Path] && string(ev.Data) != string(body) {
			t.Errorf("%s: data differs from GET\nhook: %s\nGET:  %s", tg.Path, ev.Data, body)
		}
		if hashOf(t, tg.Agent, ev.Data, start, end) != hashOf(t, tg.Agent, body, gs, ge) {
			t.Errorf("%s: normalized data differs from GET\nhook: %s\nGET:  %s", tg.Path, ev.Data, body)
		}
	}
}
