package demobot

// httpapi_test.go — read-only HTTP JSON API: envelope golden, routing table
// over all 12 agent names, 404/400 paths, the global rate limit and CORS.
// All upstreams are dead-stubbed (stubExternalBases via deadAgents) so every
// test is hermetic and deterministic.

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// testEnvelope mirrors httpEnvelope for decoding responses in asserts.
// Levels stays raw JSON so tests can assert exact shapes and absence.
type testEnvelope struct {
	Agent      string          `json:"agent"`
	Asset      string          `json:"asset"`
	OK         bool            `json:"ok"`
	Reason     *string         `json:"reason"`
	Verdict    string          `json:"verdict"`
	Semaphore  string          `json:"semaphore"`
	Facts      []string        `json:"facts"`
	Levels     json.RawMessage `json:"levels"`
	Confidence *int            `json:"confidence"`
	AIText     *string         `json:"ai_text"`
	Sections   []string        `json:"sections"`
	DataAsOf   string          `json:"data_as_of"`
	Disclaimer string          `json:"disclaimer"`
	CardHTML   string          `json:"card_html"`
}

// newTestAPI serves the real handler chain from an httptest server. relax
// lifts the global token bucket out of the way for tests that legitimately
// fire more than the 10-request burst.
func newTestAPI(t *testing.T, ag *Agents, relax bool) (*HTTPServer, *httptest.Server) {
	t.Helper()
	s := NewHTTPServer("127.0.0.1:0", ag)
	if relax {
		s.lim.mu.Lock()
		s.lim.rate, s.lim.burst, s.lim.tokens = 1e9, 1e9, 1e9
		s.lim.mu.Unlock()
	}
	srv := httptest.NewServer(s.srv.Handler)
	t.Cleanup(srv.Close)
	return s, srv
}

func httpGet(t *testing.T, url string) (int, http.Header, []byte) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read %s: %v", url, err)
	}
	return resp.StatusCode, resp.Header, body
}

// ── Envelope golden: one static case, byte-exact ─────────────────────────────

func TestHTTPEnvelopeGolden(t *testing.T) {
	c := Card{
		Emoji:      "🟢",
		Agent:      "Momentum Agent",
		Asset:      "BTC",
		Verdict:    "BULLISH — bulls in control",
		Facts:      []string{"RSI(14): 62.4", "MACD hist: +120.50"},
		Confidence: intPtr(60),
		AIHTML:     "<b>AI read:</b> <i>Calm &amp; greedy.</i>",
		DataTime:   goldenTime,
	}
	got, err := encodeJSON(cardEnvelope(c))
	if err != nil {
		t.Fatal(err)
	}
	want := `{"agent":"Momentum Agent","asset":"BTC","ok":true,"reason":null,"verdict":"BULLISH — bulls in control","semaphore":"bullish","facts":["RSI(14): 62.4","MACD hist: +120.50"],"confidence":60,"ai_text":"AI read: Calm & greedy.","data_as_of":"2026-08-18T06:00:00Z","disclaimer":"Analytics, not financial advice","card_html":"🟢 <b>Momentum Agent</b> · BTC\n<b>BULLISH — bulls in control</b>\n• RSI(14): 62.4\n• MACD hist: +120.50\nConfidence: ■■■□□ 60%\n<b>AI read:</b> <i>Calm &amp; greedy.</i>\n\n<i>Analytics, not financial advice · AlphaVizor · 2026-08-18 06:00 UTC</i>"}`
	if g := strings.TrimRight(string(got), "\n"); g != want {
		t.Fatalf("envelope golden mismatch:\ngot:  %s\nwant: %s", g, want)
	}
}

// Empty facts must serialize as [], never null; nil confidence/AI as null;
// ok/reason always present (true/null on a healthy card); levels omitted for
// agents without them.
func TestHTTPEnvelopeNullsAndEmptyFacts(t *testing.T) {
	c := Card{Emoji: "⚪", Agent: "X", Verdict: "v", DataTime: goldenTime}
	got, err := encodeJSON(cardEnvelope(c))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"facts":[]`, `"confidence":null`, `"ai_text":null`, `"ok":true`, `"reason":null`} {
		if !strings.Contains(string(got), want) {
			t.Errorf("envelope must contain %s, got: %s", want, got)
		}
	}
	for _, banned := range []string{`"sections"`, `"levels"`} {
		if strings.Contains(string(got), banned) {
			t.Errorf("%s must be omitted here, got: %s", banned, got)
		}
	}
}

func TestHTMLToPlain(t *testing.T) {
	cases := map[string]string{
		"<b>AI idea:</b> <i>Mentions tripled.</i>": "AI idea: Mentions tripled.",
		"⚪ <b>Whale</b> BTC: offline":              "⚪ Whale BTC: offline",
		"x &lt; y &amp; z":                         "x < y & z",
		"plain":                                    "plain",
	}
	for in, want := range cases {
		if got := htmlToPlain(in); got != want {
			t.Errorf("htmlToPlain(%q): got %q, want %q", in, got, want)
		}
	}
}

// ── Routing table: all 12 names answer through the real handler chain ────────

func TestHTTPRoutingTable(t *testing.T) {
	_, srv := newTestAPI(t, deadAgents(t), true)

	// With every upstream dead, per-agent endpoints answer an honest 503;
	// risk (no upstream) and digest/top (aggregate honestly inside) stay 200.
	cases := []struct {
		path       string
		wantStatus int
	}{
		{"/agents/macro", 503},
		{"/agents/whale", 503},
		{"/agents/funding", 503},
		{"/agents/momentum", 503},
		{"/agents/trend", 503},
		{"/agents/sr", 503},
		{"/agents/vol", 503},
		{"/agents/fx", 503},
		{"/agents/news", 503},
		{"/agents/risk?balance=10000&risk=1&entry=64000&stop=62500", 200},
		{"/agents/digest", 200},
		{"/agents/top", 200},
	}
	if len(cases) != len(httpAgentNames) {
		t.Fatalf("routing table drifted: %d cases vs %d agent names", len(cases), len(httpAgentNames))
	}
	for _, tc := range cases {
		status, header, body := httpGet(t, srv.URL+tc.path)
		if status != tc.wantStatus {
			t.Errorf("%s: status %d, want %d (body: %s)", tc.path, status, tc.wantStatus, body)
		}
		if got := header.Get("Access-Control-Allow-Origin"); got != "*" {
			t.Errorf("%s: CORS header %q, want *", tc.path, got)
		}
		if ct := header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
			t.Errorf("%s: content type %q", tc.path, ct)
		}
		var parsed map[string]any
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Errorf("%s: body is not JSON: %v", tc.path, err)
			continue
		}
		if tc.wantStatus == 503 {
			msg, _ := parsed["error"].(string)
			if msg == "" {
				t.Errorf("%s: 503 must carry an honest error, got %s", tc.path, body)
			}
		}
	}
}

func TestHTTPAgentsListComplete(t *testing.T) {
	_, srv := newTestAPI(t, deadAgents(t), true)
	status, _, body := httpGet(t, srv.URL+"/agents")
	if status != 200 {
		t.Fatalf("/agents: status %d (%s)", status, body)
	}
	var list struct {
		Agents []struct {
			Name        string   `json:"name"`
			Description string   `json:"description"`
			Assets      []string `json:"assets"`
			Examples    []string `json:"examples"`
		} `json:"agents"`
		Disclaimer string `json:"disclaimer"`
	}
	if err := json.Unmarshal(body, &list); err != nil {
		t.Fatal(err)
	}
	if len(list.Agents) != 12 {
		t.Fatalf("want 12 agents, got %d", len(list.Agents))
	}
	if list.Disclaimer != disclaimerText {
		t.Errorf("disclaimer: %q", list.Disclaimer)
	}
	wantAssets := strings.Join(assetKeys(), ",")
	for i, item := range list.Agents {
		if item.Name != httpAgentNames[i] {
			t.Errorf("agent[%d]: name %q, want %q", i, item.Name, httpAgentNames[i])
		}
		if item.Description == "" {
			t.Errorf("%s: empty description", item.Name)
		}
		if len(item.Examples) == 0 {
			t.Errorf("%s: no example URLs", item.Name)
		}
		if assetAgents[item.Name] {
			if strings.Join(item.Assets, ",") != wantAssets {
				t.Errorf("%s: assets %v, want %v", item.Name, item.Assets, assetKeys())
			}
		} else if len(item.Assets) > 0 {
			t.Errorf("%s: must not list assets, got %v", item.Name, item.Assets)
		}
	}
	// The risk example must show all four required params.
	for _, item := range list.Agents {
		if item.Name == keyRisk {
			ex := strings.Join(item.Examples, " ")
			for _, p := range []string{"balance=", "risk=", "entry=", "stop="} {
				if !strings.Contains(ex, p) {
					t.Errorf("risk examples must show %s, got %v", p, item.Examples)
				}
			}
		}
	}
}

// ── 404 / 400 paths ──────────────────────────────────────────────────────────

func TestHTTPNotFound(t *testing.T) {
	_, srv := newTestAPI(t, deadAgents(t), true)
	for _, path := range []string{"/agents/doge", "/agents/trend/extra", "/nope", "/agents/"} {
		status, header, body := httpGet(t, srv.URL+path)
		if status != 404 {
			t.Errorf("%s: status %d, want 404 (%s)", path, status, body)
		}
		if header.Get("Access-Control-Allow-Origin") != "*" {
			t.Errorf("%s: CORS header missing on error", path)
		}
		if !strings.Contains(string(body), `"error"`) {
			t.Errorf("%s: 404 body must be a JSON error, got %s", path, body)
		}
	}
	// Root index is a friendly pointer, not a 404.
	status, _, body := httpGet(t, srv.URL+"/")
	if status != 200 || !strings.Contains(string(body), "/agents") {
		t.Errorf("root index: status %d body %s", status, body)
	}
}

func TestHTTPBadArgs(t *testing.T) {
	_, srv := newTestAPI(t, deadAgents(t), true)
	cases := []struct {
		path    string
		wantMsg string
	}{
		{"/agents/trend?asset=doge", "unknown asset"},
		{"/agents/momentum?asset=doge", "unknown asset"},
		{"/agents/macro?asset=btc", "does not take an ?asset= parameter"},
		{"/agents/digest?asset=btc", "does not take an ?asset= parameter"},
		{"/agents/risk", "missing ?balance="},
		{"/agents/risk?balance=10000&risk=1&entry=64000", "missing ?stop="},
		{"/agents/risk?balance=10000&risk=1&entry=64000&stop=abc", "not a number"},
		{"/agents/risk?balance=10000&risk=1&entry=64000&stop=64000", "stop must differ from entry"},
		{"/agents/risk?balance=10000&risk=250&entry=64000&stop=62500", "risk% must be in (0, 100]"},
	}
	for _, tc := range cases {
		status, _, body := httpGet(t, srv.URL+tc.path)
		if status != 400 {
			t.Errorf("%s: status %d, want 400 (%s)", tc.path, status, body)
			continue
		}
		var e struct {
			Error string `json:"error"`
		}
		if err := json.Unmarshal(body, &e); err != nil || !strings.Contains(e.Error, tc.wantMsg) {
			t.Errorf("%s: error %q must contain %q", tc.path, e.Error, tc.wantMsg)
		}
	}
}

func TestHTTPMethodNotAllowed(t *testing.T) {
	_, srv := newTestAPI(t, deadAgents(t), true)
	resp, err := http.Post(srv.URL+"/agents", "application/json", strings.NewReader("{}"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("POST: status %d, want 405", resp.StatusCode)
	}
	if resp.Header.Get("Allow") != http.MethodGet {
		t.Errorf("405 must carry Allow: GET, got %q", resp.Header.Get("Allow"))
	}
	if resp.Header.Get("Access-Control-Allow-Origin") != "*" {
		t.Error("CORS header missing on 405")
	}
}

// ── Rate limit ───────────────────────────────────────────────────────────────

// Deterministic bucket math on a fixed clock.
func TestTokenBucketMath(t *testing.T) {
	b := newTokenBucket(10, 10)
	t0 := time.Unix(1_000_000, 0)
	b.last, b.tokens = t0, 10

	for i := 0; i < 10; i++ {
		if !b.allowAt(t0) {
			t.Fatalf("request %d within the burst must pass", i+1)
		}
	}
	if b.allowAt(t0) {
		t.Fatal("11th instant request must be limited")
	}
	if !b.allowAt(t0.Add(100 * time.Millisecond)) {
		t.Fatal("one token must refill after 100ms at 10/s")
	}
	if b.allowAt(t0.Add(100 * time.Millisecond)) {
		t.Fatal("the refilled token is spent — next must be limited")
	}
	// Refill caps at burst: after a long idle only 10 pass.
	later := t0.Add(time.Hour)
	for i := 0; i < 10; i++ {
		if !b.allowAt(later) {
			t.Fatalf("post-idle request %d must pass", i+1)
		}
	}
	if b.allowAt(later) {
		t.Fatal("burst cap must hold after long idle")
	}
	// A clock rewind must not mint tokens.
	if b.allowAt(later.Add(-time.Minute)) {
		t.Fatal("rewound clock must not refill")
	}
}

// End-to-end 429 with Retry-After through the real middleware.
func TestHTTPRateLimit429(t *testing.T) {
	s, srv := newTestAPI(t, deadAgents(t), false)
	// Tighten the bucket so the test stays deterministic even on a slow
	// runner: 2-token burst refilled at 1/s — 5 rapid requests can never
	// see more than 3 grants inside a second.
	s.lim.mu.Lock()
	s.lim.rate, s.lim.burst, s.lim.tokens = 1, 2, 2
	s.lim.mu.Unlock()

	var got429 int
	for i := 0; i < 5; i++ {
		status, header, body := httpGet(t, srv.URL+"/agents")
		if i == 0 && status != 200 {
			t.Fatalf("first request must pass, got %d", status)
		}
		if status == http.StatusTooManyRequests {
			got429++
			if header.Get("Retry-After") != "1" {
				t.Errorf("429 must carry Retry-After: 1, got %q", header.Get("Retry-After"))
			}
			if !strings.Contains(string(body), "rate limit") {
				t.Errorf("429 body must say so: %s", body)
			}
		}
	}
	if got429 < 2 {
		t.Fatalf("want at least two 429s from 5 rapid requests on a 2-token bucket, got %d", got429)
	}
}

// ── Live envelope through a 200 path (risk needs no upstream) ────────────────

func TestHTTPRiskEnvelope(t *testing.T) {
	_, srv := newTestAPI(t, deadAgents(t), true)
	status, _, body := httpGet(t, srv.URL+"/agents/risk?balance=10000&risk=1&entry=64000&stop=62500")
	if status != 200 {
		t.Fatalf("status %d (%s)", status, body)
	}
	var env testEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env.Agent != "Risk Calculator" || env.Semaphore != "neutral" {
		t.Errorf("agent/semaphore: %q/%q", env.Agent, env.Semaphore)
	}
	// Batch-2: pure sizing math — no direction labels anywhere in the envelope.
	if !strings.Contains(env.Verdict, "Position size:") {
		t.Errorf("verdict must lead with the size: %q", env.Verdict)
	}
	if strings.Contains(env.Verdict, "LONG") || strings.Contains(env.Verdict, "SHORT") {
		t.Errorf("direction label leaked into the verdict: %q", env.Verdict)
	}
	disclaimer := false
	for _, f := range env.Facts {
		if strings.Contains(f, "not a trade suggestion") {
			disclaimer = true
		}
	}
	if !disclaimer {
		t.Errorf("sizing-only disclaimer missing from facts: %v", env.Facts)
	}
	if env.Confidence != nil || env.AIText != nil {
		t.Errorf("risk card has no confidence/AI: %v %v", env.Confidence, env.AIText)
	}
	if len(env.Facts) < 3 {
		t.Errorf("facts too short: %v", env.Facts)
	}
	if env.Disclaimer != disclaimerText {
		t.Errorf("disclaimer: %q", env.Disclaimer)
	}
	if ts, err := time.Parse(time.RFC3339, env.DataAsOf); err != nil || time.Since(ts) > time.Minute {
		t.Errorf("data_as_of must be a fresh RFC3339 stamp: %q (%v)", env.DataAsOf, err)
	}
	if !strings.Contains(env.CardHTML, "<b>Risk Calculator</b>") {
		t.Errorf("card_html must be the Telegram card: %q", env.CardHTML)
	}
}

// ── Digest & top: sections, ai_text, and the SHARED AI memo ──────────────────

// The 5-minute AI memo must be one instance across Telegram and HTTP: a
// Telegram /digest followed by HTTP /agents/digest and /agents/top on the
// same market state fires exactly 2 upstream LLM calls (1 brief + 1 why),
// not 4.
func TestHTTPDigestTopEnvelopeAndSharedAIMemo(t *testing.T) {
	stub, client := newAIStub(t, func(n int, body map[string]any) (int, string) {
		msgs, _ := body["messages"].([]any)
		msg, _ := msgs[0].(map[string]any)
		content, _ := msg["content"].(string)
		if strings.Contains(content, "top signal") {
			return 200, aiEnvelope("Because macro leads while sources are down.")
		}
		return 200, aiEnvelope("Markets are quiet today.")
	})
	ag := deadAgents(t)
	ag.ai = client
	bot := &Bot{ag: ag, lastReq: map[int64]time.Time{}}

	// 1) Telegram path warms the memo.
	_ = bot.digestReply(t.Context())
	if stub.count() != 1 {
		t.Fatalf("telegram digest: want 1 upstream AI call, got %d", stub.count())
	}

	_, srv := newTestAPI(t, ag, true)

	// 2) HTTP digest — identical market state → same memo key → no new call.
	status, _, body := httpGet(t, srv.URL+"/agents/digest")
	if status != 200 {
		t.Fatalf("digest status %d (%s)", status, body)
	}
	var env testEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env.Agent != "AlphaVizor Digest" {
		t.Errorf("digest agent: %q", env.Agent)
	}
	if !strings.HasPrefix(env.Verdict, "Top signal: ") {
		t.Errorf("digest verdict must name the winner: %q", env.Verdict)
	}
	if env.AIText == nil || *env.AIText != "Markets are quiet today." {
		t.Errorf("digest ai_text must carry the brief: %v", env.AIText)
	}
	if len(env.Sections) != len(digestOrder)-1 {
		t.Errorf("sections: got %d one-liners, want %d (all digest agents minus the winner): %v",
			len(env.Sections), len(digestOrder)-1, env.Sections)
	}
	for _, sec := range env.Sections {
		if strings.ContainsAny(sec, "<>") {
			t.Errorf("sections must be plain text, got %q", sec)
		}
	}
	if !strings.Contains(env.CardHTML, "<b>AlphaVizor Digest</b>") ||
		!strings.Contains(env.CardHTML, "<b>ALPHAVIZOR AI</b>") {
		t.Errorf("digest card_html must be the full Telegram message:\n%s", env.CardHTML)
	}
	if stub.count() != 1 {
		t.Fatalf("HTTP digest must reuse the Telegram AI memo, upstream calls: %d", stub.count())
	}

	// 3) HTTP top — brief comes from the memo, the why-line is one new call.
	status, _, body = httpGet(t, srv.URL+"/agents/top")
	if status != 200 {
		t.Fatalf("top status %d (%s)", status, body)
	}
	var top testEnvelope
	if err := json.Unmarshal(body, &top); err != nil {
		t.Fatal(err)
	}
	if top.AIText == nil ||
		!strings.Contains(*top.AIText, "Markets are quiet today.") ||
		!strings.Contains(*top.AIText, "Because macro leads") {
		t.Errorf("top ai_text must join brief and why: %v", top.AIText)
	}
	if len(top.Sections) != 0 {
		t.Errorf("top has no sections, got %v", top.Sections)
	}
	if !strings.Contains(top.CardHTML, "<b>Top signal right now</b>") ||
		!strings.Contains(top.CardHTML, "<b>AI why:</b>") {
		t.Errorf("top card_html must be the full Telegram message:\n%s", top.CardHTML)
	}
	if stub.count() != 2 {
		t.Fatalf("digest+top over the shared memo must fire exactly 2 upstream calls, got %d", stub.count())
	}
}

// ── Lifecycle: bind, serve, second-bind failure, graceful shutdown ───────────

func TestHTTPServerLifecycle(t *testing.T) {
	ag := deadAgents(t)
	s1 := NewHTTPServer("127.0.0.1:0", ag)
	if err := s1.Start(); err != nil {
		t.Fatalf("bind: %v", err)
	}
	defer func() {
		ctx, cancel := t.Context(), func() {}
		defer cancel()
		_ = s1.Shutdown(ctx)
	}()
	addr := s1.Addr()
	if addr == "" {
		t.Fatal("Addr must report the bound address")
	}
	status, _, _ := httpGet(t, "http://"+addr+"/agents")
	if status != 200 {
		t.Fatalf("live server /agents: %d", status)
	}

	// Second bind on the same port must fail — main treats this as fatal.
	s2 := NewHTTPServer(addr, ag)
	if err := s2.Start(); err == nil {
		_ = s2.Shutdown(t.Context())
		t.Fatal("second bind on a taken port must error")
	}

	if err := s1.Shutdown(t.Context()); err != nil {
		t.Fatalf("graceful shutdown: %v", err)
	}
	// The listener is truly closed: a fresh server can take the port over.
	s3 := NewHTTPServer(addr, ag)
	if err := s3.Start(); err != nil {
		t.Fatalf("rebind after shutdown: %v", err)
	}
	_ = s3.Shutdown(t.Context())
}
