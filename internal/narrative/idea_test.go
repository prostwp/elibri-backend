package narrative

// idea_test.go — tests for IdeaGenerator implementations and the
// in-memory IdeaCache. Mirrors importance_test.go's structure: stub server
// for the Haiku path, table-driven for prompt shape, hermetic (no live
// network).

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ─── StaticIdeaGenerator ──────────────────────────────────────────────────

// TestStaticIdeaGenerator_ReturnsConstant pins the static fallback contract:
// always staticIdeaText, never errors on a non-cancelled ctx, ignores all
// inputs by design.
func TestStaticIdeaGenerator_ReturnsConstant(t *testing.T) {
	t.Parallel()
	g := StaticIdeaGenerator{}
	idea, err := g.GenerateIdea(context.Background(), "any", Snapshot{}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if idea != staticIdeaText {
		t.Errorf("idea = %q, want %q", idea, staticIdeaText)
	}
}

// TestStaticIdeaGenerator_HonoursCtx makes sure the static path doesn't
// silently mask a shutdown signal — useful when callers chain Generate
// inside a request hot-path.
func TestStaticIdeaGenerator_HonoursCtx(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	g := StaticIdeaGenerator{}
	_, err := g.GenerateIdea(ctx, "x", Snapshot{}, nil)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected ctx.Canceled, got %v", err)
	}
}

// ─── HaikuIdeaGenerator ───────────────────────────────────────────────────

// haikuIdeaResp is a minimal Anthropic Messages response with the LLM's
// reply as raw text (no JSON envelope to unwrap — the IdeaGenerator
// returns the text verbatim after trimming a stray markdown fence).
func haikuIdeaResp(text string) []byte {
	env := anthropicResponse{Content: []anthropicContentBlock{{Type: "text", Text: text}}}
	out, _ := json.Marshal(env)
	return out
}

// TestHaikuIdeaGenerator_HappyPath drives Generate against a stub server
// returning a valid envelope + an English paragraph. Asserts (a) the right
// headers are sent, (b) the system prompt and user message reach the
// upstream, (c) the reply is returned verbatim after trimming.
//
// The reply is intentionally NEUTRAL narrative commentary — no prices, no
// support/resistance, no trade directives — so it passes the
// bannedIdeaPattern blacklist that GenerateIdea applies post-response.
func TestHaikuIdeaGenerator_HappyPath(t *testing.T) {
	t.Parallel()

	const expectedReply = "The Solana DeFi theme is gathering attention as several ecosystem assets cluster around the conversation. Coverage spans multiple crypto outlets and the discussion is in a trending phase. Sentiment across reporting leans constructive."

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Header checks — same set as the importance Haiku client.
		if got := r.Header.Get("x-api-key"); got != "sk-test" {
			t.Errorf("x-api-key = %q, want sk-test", got)
		}
		if got := r.Header.Get("anthropic-version"); got != "2023-06-01" {
			t.Errorf("anthropic-version = %q, want 2023-06-01", got)
		}

		// Body shape check — assert the new (Phase 2) prompt contract:
		// system prompt names the FORBIDDEN vocabulary, user message
		// carries the "Narrative:" prefix and a bare ticker list (no
		// dollar prices). We don't pin the full body byte-for-byte to
		// avoid flakes from formatter drift.
		body, _ := io.ReadAll(r.Body)
		bs := string(body)
		if !strings.Contains(bs, "FORBIDDEN") {
			t.Errorf("system prompt missing FORBIDDEN section (Phase 2 contract)")
		}
		if !strings.Contains(bs, "Narrative:") {
			t.Errorf("user message missing 'Narrative:' prefix")
		}
		if !strings.Contains(bs, "JTO") {
			t.Errorf("user message missing JTO ticker (related-asset rendering broken)")
		}
		// Phase 2: dollar prices must NEVER appear in the user message —
		// they used to encourage price-specific commentary that broke
		// DoD #11.
		if strings.Contains(bs, "$2.50") || strings.Contains(bs, "$180.10") {
			t.Errorf("user message unexpectedly contains $-price (Phase 2 must omit)")
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(haikuIdeaResp(expectedReply))
	}))
	defer srv.Close()

	g := &HaikuIdeaGenerator{
		APIKey:  "sk-test",
		HTTP:    srv.Client(),
		APIBase: srv.URL,
	}
	snap := Snapshot{
		Narrative:      "solana-defi",
		TrendScore:     85,
		Stage:          StageTrending,
		SentimentLabel: SentBull,
		RelatedAssets:  []string{"JTO", "SOL", "JUP"},
		Reasons:        []string{"Stage: TRENDING", "Mentions up 35% vs prior 24h"},
	}
	prices := []TickerQuote{
		{Symbol: "JTO", Price: 2.50, ChangePct: 12.4},
		{Symbol: "SOL", Price: 180.10, ChangePct: 3.1},
		{Symbol: "JUP", Price: 0.55, ChangePct: -1.2},
	}

	idea, err := g.GenerateIdea(context.Background(), snap.Narrative, snap, prices)
	if err != nil {
		t.Fatalf("happy path error: %v", err)
	}
	if idea != expectedReply {
		t.Errorf("idea = %q, want %q", idea, expectedReply)
	}
}

// TestHaikuIdeaGenerator_PromptShape pins the user-message rendering format.
// If you change the buildIdeaUserMessage layout, this test breaks loudly so
// you can update the upstream prompt expectations in lockstep.
//
// Phase 2 (May 2026) change: the user message no longer carries $-prices
// for related assets — only the bare ticker list — so the model has no
// per-token price to anchor on. The `prices` arg is accepted by the
// function for interface symmetry but is intentionally ignored.
func TestHaikuIdeaGenerator_PromptShape(t *testing.T) {
	t.Parallel()
	snap := Snapshot{
		Narrative:      "ai-tokens",
		TrendScore:     72,
		Stage:          StageEarly,
		SentimentLabel: SentNeutral,
		RelatedAssets:  []string{"FET", "RNDR"},
		Reasons:        []string{"Only 1 source reporting"},
	}
	prices := []TickerQuote{
		{Symbol: "FET", Price: 1.23, ChangePct: 5.7},
		{Symbol: "RNDR", Price: 8.45, ChangePct: -2.3},
	}
	got := buildIdeaUserMessage(snap.Narrative, snap, prices)

	wantSubstrings := []string{
		"Narrative: ai-tokens",
		"Stage: early, Trend: 72/100, Sentiment: neutral",
		"Related assets: FET, RNDR",
		"Context: Only 1 source reporting",
	}
	for _, want := range wantSubstrings {
		if !strings.Contains(got, want) {
			t.Errorf("user message missing %q\nfull message:\n%s", want, got)
		}
	}

	// Phase 2: $-prices must NEVER make it into the user message.
	if strings.Contains(got, "$") {
		t.Errorf("user message unexpectedly contains $ price marker; Phase 2 forbids it.\nfull message:\n%s", got)
	}
	if strings.Contains(got, "1.23") || strings.Contains(got, "5.7%") {
		t.Errorf("user message unexpectedly carries per-token numeric price data.\nfull message:\n%s", got)
	}

	// Pin EN-only labels — guard against a future refactor that
	// reintroduces the Russian "Нарратив:" anchor that triggered the
	// Phase 1 language-mismatch bug.
	if strings.Contains(got, "Нарратив") {
		t.Errorf("user message contains Russian label 'Нарратив'; must be English-only.\nfull message:\n%s", got)
	}
}

// TestHaikuIdeaGenerator_PromptShape_NoPrices verifies the prompt still
// renders correctly when the price array is nil (e.g. Binance failed).
// Phase 2: the message shape is identical with or without prices —
// `prices` is ignored.
func TestHaikuIdeaGenerator_PromptShape_NoPrices(t *testing.T) {
	t.Parallel()
	snap := Snapshot{
		Narrative:      "rwa",
		TrendScore:     50,
		Stage:          StageMainstream,
		SentimentLabel: SentBear,
		RelatedAssets:  []string{"ONDO"},
	}
	got := buildIdeaUserMessage(snap.Narrative, snap, nil)

	if !strings.Contains(got, "Narrative: rwa") {
		t.Errorf("missing narrative line")
	}
	if !strings.Contains(got, "Related assets: ONDO") {
		t.Errorf("expected bare ticker fallback, got: %s", got)
	}
	// Must NOT contain ANY $ marker.
	if strings.Contains(got, "$") {
		t.Errorf("no-prices fallback unexpectedly rendered $ price line: %s", got)
	}
}

// TestHaikuIdeaGenerator_4xxNoRetry mirrors the importance classifier's
// retry policy: 4xx is auth/quota → error after one attempt, no retry.
func TestHaikuIdeaGenerator_4xxNoRetry(t *testing.T) {
	t.Parallel()

	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	g := &HaikuIdeaGenerator{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
	_, err := g.GenerateIdea(context.Background(), "x", Snapshot{}, nil)
	if err == nil {
		t.Fatal("expected error on 4xx, got nil")
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Errorf("server saw %d calls, want exactly 1 (no retry on 4xx)", got)
	}
}

// TestHaikuIdeaGenerator_5xxRetry — one 500, then 200 → success on retry.
func TestHaikuIdeaGenerator_5xxRetry(t *testing.T) {
	t.Parallel()

	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&calls, 1)
		if n == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(haikuIdeaResp("retry success"))
	}))
	defer srv.Close()

	g := &HaikuIdeaGenerator{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
	idea, err := g.GenerateIdea(context.Background(), "x", Snapshot{}, nil)
	if err != nil {
		t.Fatalf("retry path error: %v", err)
	}
	if idea != "retry success" {
		t.Errorf("idea = %q, want 'retry success'", idea)
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Errorf("server saw %d calls, want exactly 2 (initial + 1 retry)", got)
	}
}

// TestHaikuIdeaGenerator_EmptyAPIKey: the generator should refuse to send
// without a key rather than ship a request guaranteed to 401.
func TestHaikuIdeaGenerator_EmptyAPIKey(t *testing.T) {
	t.Parallel()
	g := &HaikuIdeaGenerator{APIKey: ""}
	_, err := g.GenerateIdea(context.Background(), "x", Snapshot{}, nil)
	if err == nil {
		t.Fatal("expected error on empty APIKey, got nil")
	}
}

// TestHaikuIdeaGenerator_CtxCancellation: ctx already cancelled before the
// HTTP request → ctx.Err bubbles up.
func TestHaikuIdeaGenerator_CtxCancellation(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(haikuIdeaResp("ok"))
	}))
	defer srv.Close()
	g := &HaikuIdeaGenerator{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := g.GenerateIdea(ctx, "x", Snapshot{}, nil)
	if err == nil {
		t.Fatal("expected ctx error, got nil")
	}
}

// TestHaikuIdeaGenerator_StripsMarkdownFence: the upstream occasionally
// wraps its reply in ```...``` even when told not to. cleanIdeaText
// strips it; this test pins that behaviour.
//
// Fixtures are deliberately NEUTRAL narrative-style text (no prices, no
// trader vocabulary) so they pass through bannedIdeaPattern unchanged —
// the focus of this test is fence-stripping, not blacklist behaviour
// (see TestHaikuIdeaGenerator_BlacklistRejects for that).
func TestHaikuIdeaGenerator_StripsMarkdownFence(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "fenced no language tag",
			raw:  "```\nThe restaking theme is drawing renewed coverage.\n```",
			want: "The restaking theme is drawing renewed coverage.",
		},
		{
			name: "fenced with language",
			raw:  "```text\nAI tokens cluster around recent sector reporting.\n```",
			want: "AI tokens cluster around recent sector reporting.",
		},
		{
			name: "no fence — unchanged",
			raw:  "The narrative remains in an early attention phase.",
			want: "The narrative remains in an early attention phase.",
		},
		{
			name: "trailing whitespace",
			raw:  "  Coverage breadth is gradually expanding.  \n",
			want: "Coverage breadth is gradually expanding.",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write(haikuIdeaResp(tc.raw))
			}))
			defer srv.Close()
			g := &HaikuIdeaGenerator{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
			idea, err := g.GenerateIdea(context.Background(), "x", Snapshot{}, nil)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if idea != tc.want {
				t.Errorf("idea = %q, want %q", idea, tc.want)
			}
		})
	}
}

// TestHaikuIdeaGenerator_BlacklistRejects pins the Phase 2 (May 2026)
// post-response blacklist contract: even when the upstream returns a
// well-formed paragraph, if it contains DoD-#11-banned vocabulary
// (specific prices, support/resistance, trade directives, etc.) the
// generator must return ("", nil) so the caller skips caching and the
// frontend falls back to the neutral static line.
//
// Returning err=nil is intentional — a blacklist hit is NOT a transport
// error; it's a content gate. Callers (handler) already short-circuit on
// err!=nil and would otherwise cache an empty string; nil-error lets the
// empty value flow through cleanly while keeping the cache miss for the
// next request (the handler's generateIdeaCached only caches the
// successful non-empty value).
func TestHaikuIdeaGenerator_BlacklistRejects(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		reply string // what the stub upstream returns
	}{
		{"specific dollar price", "JTO is interesting at $2.40."},
		{"support level mention", "The narrative finds support after recent reports."},
		{"resistance level mention", "Coverage is bumping into resistance."},
		{"breakout call", "A potential breakout is forming in this theme."},
		{"buy directive", "You should buy into the trend."},
		{"sell directive", "Time to sell the rumour."},
		{"entry call", "Consider an entry around current levels."},
		{"stop loss term", "Use a tight stop loss on the move."},
		{"stop-loss hyphen variant", "Tighten stop-loss management."},
		{"take profit term", "Take profit at the next milestone."},
		{"target term", "The target is the next round number."},
		{"long term", "Stay long on the theme."},
		{"short term", "Go short into the catalyst."},
		{"fade term", "Fade the move on reversal."},
		{"accumulation term", "Quiet accumulation is underway."},
		{"momentum term", "Momentum is building."},
		{"watch for phrase", "Watch for the next leg up."},
		{"traders should phrase", "Traders should be cautious here."},
		{"position term", "A long position becomes attractive."},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write(haikuIdeaResp(tc.reply))
			}))
			defer srv.Close()

			g := &HaikuIdeaGenerator{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
			idea, err := g.GenerateIdea(context.Background(), "x", Snapshot{}, nil)
			if err != nil {
				t.Fatalf("blacklist path should NOT return an error, got: %v", err)
			}
			if idea != "" {
				t.Errorf("blacklist hit should return empty string, got %q", idea)
			}
		})
	}
}

// TestHaikuIdeaGenerator_BlacklistAllowsNeutral pins the inverse contract:
// neutral narrative commentary (no banned vocabulary) flows through
// unchanged. Without this guard, an over-eager regex could swallow valid
// output and leave the dashboard permanently silent.
func TestHaikuIdeaGenerator_BlacklistAllowsNeutral(t *testing.T) {
	t.Parallel()

	allowed := []string{
		"The restaking theme is gaining traction across recent coverage.",
		"AI tokens cluster around an early-attention phase with mixed sentiment.",
		"Coverage breadth across crypto outlets is expanding around the topic.",
		"The narrative is in a trending phase with constructive overall tone.",
	}

	for _, reply := range allowed {
		t.Run(reply[:20], func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write(haikuIdeaResp(reply))
			}))
			defer srv.Close()

			g := &HaikuIdeaGenerator{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
			idea, err := g.GenerateIdea(context.Background(), "x", Snapshot{}, nil)
			if err != nil {
				t.Fatalf("unexpected error on neutral reply: %v", err)
			}
			if idea != reply {
				t.Errorf("idea = %q, want unchanged %q", idea, reply)
			}
		})
	}
}

// TestHaikuIdeaGenerator_PromptForbidsTraderVocab pins the Phase 2
// system-prompt contract — the explicit FORBIDDEN list must reach the
// upstream so the model has a fighting chance of not regressing.
func TestHaikuIdeaGenerator_PromptForbidsTraderVocab(t *testing.T) {
	t.Parallel()

	var capturedSystem string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req anthropicRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		capturedSystem = req.System
		_, _ = w.Write(haikuIdeaResp("Coverage is broadening."))
	}))
	defer srv.Close()

	g := &HaikuIdeaGenerator{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
	if _, err := g.GenerateIdea(context.Background(), "x", Snapshot{}, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Substrings the Phase 2 prompt MUST contain, by name, so a refactor
	// that drops them is loud rather than silent.
	mustContain := []string{
		"NARRATIVE",
		"FORBIDDEN",
		"support levels",
		"resistance levels",
		"breakout",
		"\"buy\"",
		"\"sell\"",
		"traders should",
		"observational",
	}
	for _, want := range mustContain {
		if !strings.Contains(capturedSystem, want) {
			t.Errorf("system prompt missing required Phase 2 substring %q\nfull prompt:\n%s", want, capturedSystem)
		}
	}

	// Temperature must be set explicitly (Phase 2): we want low-creativity
	// neutral commentary, not the API default of 1.0. The exact value
	// can drift slightly during future tuning, but it must be < 0.8 to
	// keep the model's drift into trader vocabulary in check.
	if haikuIdeaTemperature >= 0.8 {
		t.Errorf("haikuIdeaTemperature = %v, want < 0.8 to keep output observational", haikuIdeaTemperature)
	}
}

// TestHaikuIdeaGenerator_PromptIsEnglish pins the May 2026 Phase 1 fix.
// Before the fix, Haiku was returning Russian replies because the system
// prompt was Russian-tone-neutral and the user message led with the
// Russian "Нарратив:" label, so the LLM mirrored the user language.
// After the fix the system prompt MUST contain an explicit English-only
// instruction. This test captures the request body, decodes the JSON
// envelope (the same anthropicRequest shape we marshal into), and asserts
// the system field carries the expected English-only directives.
func TestHaikuIdeaGenerator_PromptIsEnglish(t *testing.T) {
	t.Parallel()

	var capturedSystem string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		// Decode just the system prompt field — keeping the anthropicRequest
		// import unused would force a parallel local struct, which is exactly
		// what the package-internal struct is for.
		var req anthropicRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		capturedSystem = req.System

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(haikuIdeaResp("ok"))
	}))
	defer srv.Close()

	g := &HaikuIdeaGenerator{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
	if _, err := g.GenerateIdea(context.Background(), "x", Snapshot{}, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if capturedSystem == "" {
		t.Fatal("system prompt was empty — handler never saw a request body")
	}

	// Case-insensitive English-mention check. Looking for any of the
	// well-known phrases keeps the test robust against minor wording
	// drift while still failing if "English" disappears entirely.
	lower := strings.ToLower(capturedSystem)
	if !strings.Contains(lower, "english") {
		t.Errorf("system prompt missing 'English' (case-insensitive). Got: %s", capturedSystem)
	}

	// Belt-and-braces: the second mention is the imperative at the end of
	// the prompt. If only one of the two ends up in the body something
	// got accidentally truncated or refactored away.
	if strings.Count(lower, "english") < 2 {
		t.Errorf("system prompt should mention 'English' at least twice (role + final imperative); got count=%d in:\n%s",
			strings.Count(lower, "english"), capturedSystem)
	}

	// The system prompt MUST NOT contain Russian text — the May 2026
	// regression we're guarding against was a Russian-leaning prompt.
	if strings.ContainsAny(capturedSystem, "абвгдеёжзийклмнопрстуфхцчшщъыьэюя") {
		t.Errorf("system prompt contains Cyrillic characters; must be English-only:\n%s", capturedSystem)
	}
}

// ─── IdeaCache ────────────────────────────────────────────────────────────

// TestIdeaCache_GetSet: round-trip — put then get returns the same value.
func TestIdeaCache_GetSet(t *testing.T) {
	t.Parallel()
	c := &IdeaCache{}
	c.Set("restaking", "2026-05-05T18:00:00Z", "JTO лидирует")
	got, ok := c.Get("restaking", "2026-05-05T18:00:00Z")
	if !ok {
		t.Fatal("expected hit, got miss")
	}
	if got != "JTO лидирует" {
		t.Errorf("got %q, want 'JTO лидирует'", got)
	}
}

// TestIdeaCache_Miss: different narrative or different hour returns miss.
func TestIdeaCache_Miss(t *testing.T) {
	t.Parallel()
	c := &IdeaCache{}
	c.Set("restaking", "2026-05-05T18:00:00Z", "x")

	if _, ok := c.Get("ai-tokens", "2026-05-05T18:00:00Z"); ok {
		t.Error("different narrative should miss")
	}
	if _, ok := c.Get("restaking", "2026-05-05T19:00:00Z"); ok {
		t.Error("different hour should miss")
	}
}

// TestIdeaCache_NilSafe: nil receiver doesn't panic.
func TestIdeaCache_NilSafe(t *testing.T) {
	t.Parallel()
	var c *IdeaCache
	c.Set("x", "y", "z") // must not panic
	if _, ok := c.Get("x", "y"); ok {
		t.Error("nil cache should always miss")
	}
}

// TestIdeaCache_Concurrent: hammer with concurrent reads + writes; race
// detector should report nothing. The sync.Map underneath is the
// concurrency contract so this is more a guard against future refactors
// that swap it out for a plain map.
func TestIdeaCache_Concurrent(t *testing.T) {
	t.Parallel()
	c := &IdeaCache{}
	const n = 50

	var wg sync.WaitGroup
	wg.Add(n * 2)

	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			c.Set("narr", "hour", "value")
			_ = i
		}(i)
		go func(i int) {
			defer wg.Done()
			_, _ = c.Get("narr", "hour")
			_ = i
		}(i)
	}
	wg.Wait()

	got, ok := c.Get("narr", "hour")
	if !ok || got != "value" {
		t.Errorf("after concurrent writes: got %q ok=%v, want 'value' true", got, ok)
	}
}

// TestIdeaCache_Overwrite: a second Set with the same key replaces the
// first value. Idempotent overwrite is the documented semantic — two
// concurrent first-cache-misses for the same key won't corrupt each other,
// just race to be the one whose value sticks.
func TestIdeaCache_Overwrite(t *testing.T) {
	t.Parallel()
	c := &IdeaCache{}
	c.Set("k", "h", "first")
	c.Set("k", "h", "second")
	got, _ := c.Get("k", "h")
	if got != "second" {
		t.Errorf("after overwrite got %q, want 'second'", got)
	}
}

// _ keeps the time import used elsewhere — guards against the linter
// complaining if we ever shorten an idea_test.go that lost the dependency.
var _ = time.Second
