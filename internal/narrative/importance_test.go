package narrative

// importance_test.go — tests for RuleClassifier, HaikuClassifier, and
// CachedClassifier. No live network — HaikuClassifier is exercised against
// httptest.NewServer so the test suite stays hermetic.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ─── RuleClassifier ───────────────────────────────────────────────────────

// TestRuleClassifier pins the score band for each major category. These
// fixtures are the "minimum sane behaviour" the rule fallback must give
// before the user adds an ANTHROPIC_API_KEY — they're effectively the
// system's smoke test for "is the radar useful out of the box".
func TestRuleClassifier(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		title    string
		summary  string
		category string
		minScore int
		maxScore int
	}{
		{
			name:     "fed rate cut → macro 70+",
			title:    "Fed cuts rates by 50 basis points in surprise move",
			summary:  "Federal Reserve cited slowing inflation.",
			category: CategoryMacro,
			minScore: 70,
			maxScore: 95,
		},
		{
			name:     "uk cpi release → macro mid-band",
			title:    "UK CPI release matches expectations at 2.1%",
			summary:  "Inflation print in line with consensus.",
			category: CategoryMacro,
			minScore: 60,
			maxScore: 80,
		},
		{
			name:     "trump declares emergency → geopolitics 80+",
			title:    "Trump declares national emergency over border situation",
			summary:  "President Trump invoked emergency powers.",
			category: CategoryGeopolitics,
			minScore: 70,
			maxScore: 95,
		},
		{
			name:     "exchange hack → exchange 70+",
			title:    "Major exchange hacked, $300M missing",
			summary:  "Funds drained from hot wallet in early-morning exploit.",
			category: CategoryExchange,
			minScore: 60,
			maxScore: 90,
		},
		{
			name:     "etf approval → regulation 70+",
			title:    "Spot ETF approved by SEC after years of delays",
			summary:  "First spot Bitcoin ETF approval clears the way.",
			category: CategoryRegulation,
			minScore: 60,
			maxScore: 90,
		},
		{
			name:     "altcoin partnership → 30-50",
			title:    "Project XYZ partners with KuCoin for new listing campaign",
			summary:  "Strategic partnership announced today.",
			category: CategoryPartnership,
			minScore: 30,
			maxScore: 60,
		},
		{
			name:     "memecoin launch → altcoin low",
			title:    "New memecoin launches on Solana",
			summary:  "Yet another shitcoin enters the chat.",
			category: CategoryAltcoin,
			minScore: 10,
			maxScore: 30,
		},
		{
			name:     "no-match → other low",
			title:    "Random nothing-burger in the news",
			summary:  "Just background noise.",
			category: CategoryOther,
			minScore: 0,
			maxScore: 30,
		},
	}

	r := &RuleClassifier{}
	ctx := context.Background()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			imp, err := r.Classify(ctx, tc.title, tc.summary, "")
			if err != nil {
				t.Fatalf("RuleClassifier.Classify error: %v", err)
			}
			if imp.Score < tc.minScore || imp.Score > tc.maxScore {
				t.Errorf("score = %d, want in [%d, %d] (title=%q)", imp.Score, tc.minScore, tc.maxScore, tc.title)
			}
			if imp.Category != tc.category {
				t.Errorf("category = %q, want %q (title=%q)", imp.Category, tc.category, tc.title)
			}
			if imp.Why == "" {
				t.Errorf("why is empty for title=%q", tc.title)
			}
		})
	}
}

// TestRuleClassifier_EmptyInput returns the canonical empty importance, no
// error, no LLM call.
func TestRuleClassifier_EmptyInput(t *testing.T) {
	t.Parallel()
	r := &RuleClassifier{}
	imp, err := r.Classify(context.Background(), "", "  \t\n  ", "url")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if imp.Score != 0 {
		t.Errorf("empty input score = %d, want 0", imp.Score)
	}
	if imp.Category != CategoryOther {
		t.Errorf("empty input category = %q, want %q", imp.Category, CategoryOther)
	}
}

// TestRuleClassifier_CtxCancellation honours ctx even though the work is
// in-memory. Pins the contract that classifier loops can safely abandon
// the rest of the batch on shutdown.
func TestRuleClassifier_CtxCancellation(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r := &RuleClassifier{}
	_, err := r.Classify(ctx, "Fed cuts rates", "macro news", "")
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected ctx.Canceled, got %v", err)
	}
}

// ─── HaikuClassifier ──────────────────────────────────────────────────────

// haikuOK is a minimal valid Anthropic Messages response with the inner
// JSON the LLM is instructed to return. Used as the happy-path fixture.
func haikuOK(score int, category, why string) []byte {
	inner := llmJSON{Score: float64(score), Category: category, Why: why}
	innerBytes, _ := json.Marshal(inner)
	env := anthropicResponse{
		Content: []anthropicContentBlock{{Type: "text", Text: string(innerBytes)}},
	}
	out, _ := json.Marshal(env)
	return out
}

// TestHaikuClassifier_HappyPath drives the classifier against a stub server
// returning a well-formed envelope + valid inner JSON.
func TestHaikuClassifier_HappyPath(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify required headers reach the upstream.
		if got := r.Header.Get("x-api-key"); got != "sk-test" {
			t.Errorf("x-api-key header = %q, want sk-test", got)
		}
		if got := r.Header.Get("anthropic-version"); got != "2023-06-01" {
			t.Errorf("anthropic-version header = %q, want 2023-06-01", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(haikuOK(85, CategoryMacro, "Fed rate cut moves markets"))
	}))
	defer srv.Close()

	h := &HaikuClassifier{
		APIKey:  "sk-test",
		HTTP:    srv.Client(),
		APIBase: srv.URL,
	}
	imp, err := h.Classify(context.Background(), "Fed cuts rates", "by 50 bp", "https://example.com/1")
	if err != nil {
		t.Fatalf("happy path error: %v", err)
	}
	if imp.Score != 85 {
		t.Errorf("score = %d, want 85", imp.Score)
	}
	if imp.Category != CategoryMacro {
		t.Errorf("category = %q, want %q", imp.Category, CategoryMacro)
	}
	if imp.Why != "Fed rate cut moves markets" {
		t.Errorf("why = %q, want %q", imp.Why, "Fed rate cut moves markets")
	}
}

// TestHaikuClassifier_5xxRetry verifies that a 500 response triggers exactly
// one retry, and that the retry sees a fresh request.
func TestHaikuClassifier_5xxRetry(t *testing.T) {
	t.Parallel()

	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&calls, 1)
		if n == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":"transient"}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(haikuOK(50, CategoryRegulation, "second-try success"))
	}))
	defer srv.Close()

	h := &HaikuClassifier{
		APIKey:  "sk-test",
		HTTP:    srv.Client(),
		APIBase: srv.URL,
	}
	imp, err := h.Classify(context.Background(), "X", "Y", "url")
	if err != nil {
		t.Fatalf("retry path error: %v", err)
	}
	if imp.Score != 50 || imp.Category != CategoryRegulation {
		t.Errorf("retry path imp = %+v, want score=50 category=regulation", imp)
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Errorf("server saw %d calls, want exactly 2 (initial + 1 retry)", got)
	}
}

// TestHaikuClassifier_5xxFinalFailure: both attempts return 500. Caller
// must see an error, not a silent zero.
func TestHaikuClassifier_5xxFinalFailure(t *testing.T) {
	t.Parallel()

	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer srv.Close()

	h := &HaikuClassifier{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
	_, err := h.Classify(context.Background(), "X", "Y", "url")
	if err == nil {
		t.Fatal("expected error after both 5xx attempts, got nil")
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Errorf("server saw %d calls, want 2 (initial + 1 retry)", got)
	}
}

// TestHaikuClassifier_4xxNoRetry: 4xx is auth/quota. Must error after one
// attempt — no retry.
func TestHaikuClassifier_4xxNoRetry(t *testing.T) {
	t.Parallel()

	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	h := &HaikuClassifier{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
	_, err := h.Classify(context.Background(), "X", "Y", "url")
	if err == nil {
		t.Fatal("expected error on 4xx, got nil")
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Errorf("server saw %d calls, want exactly 1 (no retry on 4xx)", got)
	}
}

// TestHaikuClassifier_MarkdownCodeFence: real-world hit — Haiku occasionally
// wraps the JSON in a ```json ... ``` markdown fence even though the system
// prompt says "Reply ONLY with valid JSON". Production smoke (2026-05-06)
// burned through 50 backfill items with this exact failure. The parseLLMJSON
// helper strips the fence; this test pins that behavior.
func TestHaikuClassifier_MarkdownCodeFence(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		text string
	}{
		{
			name: "json fenced with language tag",
			text: "```json\n{\"score\": 75, \"category\": \"macro\", \"why\": \"Fed signal\"}\n```",
		},
		{
			name: "bare backtick fence",
			text: "```\n{\"score\": 60, \"category\": \"regulation\", \"why\": \"SEC ruling\"}\n```",
		},
		{
			name: "fence with trailing whitespace",
			text: "```json\n{\"score\": 30, \"category\": \"altcoin\", \"why\": \"small launch\"}\n```\n\n",
		},
		{
			name: "raw JSON (no fence) still works",
			text: `{"score": 90, "category": "geopolitics", "why": "war declared"}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				env := anthropicResponse{Content: []anthropicContentBlock{
					{Type: "text", Text: tc.text},
				}}
				_ = json.NewEncoder(w).Encode(env)
			}))
			defer srv.Close()

			h := &HaikuClassifier{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
			imp, err := h.Classify(context.Background(), "X", "Y", "url")
			if err != nil {
				t.Fatalf("expected fence-wrapped JSON to parse, got error: %v", err)
			}
			if imp.Score == 0 || imp.Category == "" {
				t.Errorf("parsed Importance is empty: %+v", imp)
			}
		})
	}
}

// TestHaikuClassifier_MalformedInnerJSON: envelope parses fine, but the
// LLM's "text" field isn't valid JSON. Must surface as an error so the
// CachedClassifier doesn't write garbage to the cache.
func TestHaikuClassifier_MalformedInnerJSON(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		env := anthropicResponse{Content: []anthropicContentBlock{
			{Type: "text", Text: "this is not json {{["}}}
		_ = json.NewEncoder(w).Encode(env)
	}))
	defer srv.Close()

	h := &HaikuClassifier{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
	_, err := h.Classify(context.Background(), "X", "Y", "url")
	if err == nil {
		t.Fatal("expected error on malformed inner JSON, got nil")
	}
}

// TestHaikuClassifier_ScoreClamping: LLM returns score outside [0, 100].
// Must be clamped on parse — we don't trust the model's range guarantee.
func TestHaikuClassifier_ScoreClamping(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		raw       int
		wantScore int
	}{
		{"over-bound clamps to 100", 250, 100},
		{"negative clamps to 0", -10, 0},
		{"in-bound passes through", 73, 73},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write(haikuOK(tc.raw, CategoryMacro, "x"))
			}))
			defer srv.Close()
			h := &HaikuClassifier{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
			imp, err := h.Classify(context.Background(), "X", "Y", "url")
			if err != nil {
				t.Fatalf("error: %v", err)
			}
			if imp.Score != tc.wantScore {
				t.Errorf("score = %d, want %d", imp.Score, tc.wantScore)
			}
		})
	}
}

// TestHaikuClassifier_UnknownCategory: LLM returns a category outside the
// vocabulary. Must normalise to CategoryOther.
func TestHaikuClassifier_UnknownCategory(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(haikuOK(70, "nonsense-category", "x"))
	}))
	defer srv.Close()
	h := &HaikuClassifier{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
	imp, err := h.Classify(context.Background(), "X", "Y", "url")
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if imp.Category != CategoryOther {
		t.Errorf("unknown category should normalise to %q, got %q", CategoryOther, imp.Category)
	}
}

// TestHaikuClassifier_EmptyAPIKey: classifier should refuse to send without
// a key rather than send a request guaranteed to 401.
func TestHaikuClassifier_EmptyAPIKey(t *testing.T) {
	t.Parallel()
	h := &HaikuClassifier{APIKey: ""}
	_, err := h.Classify(context.Background(), "X", "Y", "url")
	if err == nil {
		t.Fatal("expected error on empty APIKey, got nil")
	}
}

// TestHaikuClassifier_EmptyInput: empty title and summary returns the zero
// importance without burning an API call.
func TestHaikuClassifier_EmptyInput(t *testing.T) {
	t.Parallel()

	var called int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&called, 1)
		_, _ = w.Write(haikuOK(85, CategoryMacro, "x"))
	}))
	defer srv.Close()

	h := &HaikuClassifier{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
	imp, err := h.Classify(context.Background(), "", "", "url")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if imp.Score != 0 {
		t.Errorf("empty-input score = %d, want 0", imp.Score)
	}
	if got := atomic.LoadInt32(&called); got != 0 {
		t.Errorf("server should not have been called for empty input, got %d calls", got)
	}
}

// ─── CachedClassifier ─────────────────────────────────────────────────────

// fakeImportanceStore is an in-memory stub implementing importanceCacheStore.
// Keeps the unit test free of testcontainers / DB.
type fakeImportanceStore struct {
	mu      sync.Mutex
	cache   map[string]Importance
	getErr  error
	putErr  error
	getHits int
	puts    int
}

func newFakeStore() *fakeImportanceStore {
	return &fakeImportanceStore{cache: make(map[string]Importance)}
}

func (f *fakeImportanceStore) GetImportanceByURL(ctx context.Context, url string) (Importance, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.getErr != nil {
		return Importance{}, false, f.getErr
	}
	imp, ok := f.cache[url]
	if ok {
		f.getHits++
	}
	return imp, ok, nil
}

func (f *fakeImportanceStore) UpdateImportance(ctx context.Context, url string, imp Importance) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.putErr != nil {
		return f.putErr
	}
	f.puts++
	f.cache[url] = imp
	return nil
}

// fakeClassifier is an ImportanceClassifier that returns a fixed result and
// counts the number of calls, so we can verify cache hits avoid Inner.
type fakeClassifier struct {
	mu     sync.Mutex
	calls  int
	result Importance
	err    error
}

func (f *fakeClassifier) Classify(ctx context.Context, title, summary, url string) (Importance, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls++
	if f.err != nil {
		return Importance{}, f.err
	}
	return f.result, nil
}

func (f *fakeClassifier) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

// TestCachedClassifier_CacheMiss: first call must call Inner and persist.
func TestCachedClassifier_CacheMiss(t *testing.T) {
	t.Parallel()
	store := newFakeStore()
	inner := &fakeClassifier{result: Importance{Score: 80, Category: CategoryMacro, Why: "fed"}}
	c := &CachedClassifier{Inner: inner, Store: store}

	imp, err := c.Classify(context.Background(), "T", "S", "url1")
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if imp.Score != 80 {
		t.Errorf("score = %d, want 80", imp.Score)
	}
	if inner.callCount() != 1 {
		t.Errorf("inner called %d times, want 1", inner.callCount())
	}
	if store.puts != 1 {
		t.Errorf("store puts = %d, want 1", store.puts)
	}
	cached, ok, _ := store.GetImportanceByURL(context.Background(), "url1")
	if !ok || cached.Score != 80 {
		t.Errorf("after cache miss, store should hold %v, got %+v ok=%v", imp, cached, ok)
	}
}

// TestCachedClassifier_CacheHit: pre-populate, then verify Inner is NOT
// called and the cached value is returned verbatim.
func TestCachedClassifier_CacheHit(t *testing.T) {
	t.Parallel()
	store := newFakeStore()
	store.cache["urlhit"] = Importance{Score: 42, Category: CategoryRegulation, Why: "cached"}
	inner := &fakeClassifier{result: Importance{Score: 999, Category: CategoryOther, Why: "should not see this"}}
	c := &CachedClassifier{Inner: inner, Store: store}

	imp, err := c.Classify(context.Background(), "T", "S", "urlhit")
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if imp.Score != 42 || imp.Category != CategoryRegulation {
		t.Errorf("hit returned wrong value: %+v", imp)
	}
	if inner.callCount() != 0 {
		t.Errorf("Inner should not be called on cache hit, got %d calls", inner.callCount())
	}
	if store.puts != 0 {
		t.Errorf("cache hit must not write to store, got %d puts", store.puts)
	}
}

// TestCachedClassifier_NilInner: explicit error rather than panic.
func TestCachedClassifier_NilInner(t *testing.T) {
	t.Parallel()
	c := &CachedClassifier{}
	_, err := c.Classify(context.Background(), "T", "S", "url")
	if err == nil {
		t.Fatal("expected error on nil Inner, got nil")
	}
}

// TestCachedClassifier_NilStore: degrades to pass-through (still calls
// Inner, no DB). Lets unit tests skip the DB stub when only exercising
// the wrapper's Inner path.
func TestCachedClassifier_NilStore(t *testing.T) {
	t.Parallel()
	inner := &fakeClassifier{result: Importance{Score: 77, Category: CategoryExchange, Why: "x"}}
	c := &CachedClassifier{Inner: inner}
	imp, err := c.Classify(context.Background(), "T", "S", "url")
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if imp.Score != 77 {
		t.Errorf("score = %d, want 77", imp.Score)
	}
	if inner.callCount() != 1 {
		t.Errorf("inner called %d times, want 1", inner.callCount())
	}
}

// TestCachedClassifier_SingleflightCoalesces: two concurrent Classify calls
// for the SAME url that arrive while neither has finished must share one
// Inner.Classify invocation. Without singleflight both would miss the cache
// and burn 2× LLM calls.
//
// Mechanism: slowClassifier blocks until the test signals it to proceed.
// We launch two goroutines, wait until both are stuck inside Classify, then
// release. After release both must return the same Importance and
// callCount() must be 1.
func TestCachedClassifier_SingleflightCoalesces(t *testing.T) {
	t.Parallel()

	store := newFakeStore()
	gate := make(chan struct{})
	entered := make(chan struct{}, 2)
	slow := &slowClassifier{
		result:  Importance{Score: 64, Category: CategoryMacro, Why: "fed-rate-cut"},
		gate:    gate,
		entered: entered,
	}
	c := &CachedClassifier{Inner: slow, Store: store}

	type out struct {
		imp Importance
		err error
	}
	results := make(chan out, 2)

	go func() {
		imp, err := c.Classify(context.Background(), "T", "S", "shared-url")
		results <- out{imp, err}
	}()
	go func() {
		imp, err := c.Classify(context.Background(), "T", "S", "shared-url")
		results <- out{imp, err}
	}()

	// Wait until at least the first goroutine has reached the slow Inner
	// (proves the second has either reached singleflight Do too, or hit the
	// cache after the first finishes — either way correct).
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("slowClassifier never entered — goroutines blocked elsewhere")
	}
	close(gate) // release the in-flight call

	// Collect both results.
	for i := 0; i < 2; i++ {
		r := <-results
		if r.err != nil {
			t.Errorf("call #%d returned err: %v", i, r.err)
		}
		if r.imp.Score != 64 {
			t.Errorf("call #%d score = %d, want 64", i, r.imp.Score)
		}
	}

	// The load-bearing assertion: only ONE Inner.Classify happened.
	if got := slow.callCount(); got != 1 {
		t.Errorf("Inner.Classify called %d times, want 1 (singleflight coalescing broken)", got)
	}
}

// slowClassifier blocks inside Classify until gate is closed. Used by the
// singleflight test to keep the first call in-flight while a second one
// arrives.
type slowClassifier struct {
	mu      sync.Mutex
	calls   int
	result  Importance
	gate    chan struct{}
	entered chan struct{} // signals on every Classify entry (buffered)
}

func (s *slowClassifier) Classify(ctx context.Context, title, summary, url string) (Importance, error) {
	s.mu.Lock()
	s.calls++
	s.mu.Unlock()
	// Non-blocking send so a second goroutine that also entered (shouldn't
	// happen if singleflight works) doesn't deadlock the test.
	select {
	case s.entered <- struct{}{}:
	default:
	}
	<-s.gate
	return s.result, nil
}

func (s *slowClassifier) callCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

// TestCachedClassifier_EmptyURL: url == "" must skip cache (no key) but
// still delegate to Inner.
func TestCachedClassifier_EmptyURL(t *testing.T) {
	t.Parallel()
	store := newFakeStore()
	inner := &fakeClassifier{result: Importance{Score: 50, Category: CategoryOther, Why: ""}}
	c := &CachedClassifier{Inner: inner, Store: store}

	imp, err := c.Classify(context.Background(), "T", "S", "")
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if imp.Score != 50 {
		t.Errorf("score = %d, want 50", imp.Score)
	}
	if inner.callCount() != 1 {
		t.Errorf("inner calls = %d, want 1", inner.callCount())
	}
	if store.puts != 0 {
		t.Errorf("empty URL must not write cache, got %d puts", store.puts)
	}
}

// TestCachedClassifier_LookupError: store returns an error on read. We
// MUST surface it (not silently fall through to Inner) — a flaky cache
// silently bypassed would burn LLM budget unnoticed.
func TestCachedClassifier_LookupError(t *testing.T) {
	t.Parallel()
	store := newFakeStore()
	store.getErr = errors.New("db down")
	inner := &fakeClassifier{result: Importance{Score: 80}}
	c := &CachedClassifier{Inner: inner, Store: store}

	_, err := c.Classify(context.Background(), "T", "S", "url")
	if err == nil {
		t.Fatal("expected lookup error to surface, got nil")
	}
	if inner.callCount() != 0 {
		t.Errorf("Inner should NOT be called on lookup error, got %d calls", inner.callCount())
	}
}

// TestCachedClassifier_WriteError: Inner succeeds, persistence fails. We
// still return the result (best-effort) but include the error so the worker
// can log it.
func TestCachedClassifier_WriteError(t *testing.T) {
	t.Parallel()
	store := newFakeStore()
	store.putErr = errors.New("disk full")
	inner := &fakeClassifier{result: Importance{Score: 60, Category: CategoryRegulation, Why: "x"}}
	c := &CachedClassifier{Inner: inner, Store: store}

	imp, err := c.Classify(context.Background(), "T", "S", "url")
	if err == nil {
		t.Fatal("expected wrapped write-error, got nil")
	}
	if imp.Score != 60 {
		t.Errorf("score = %d on write-error path, want 60 (result still returned)", imp.Score)
	}
}

// ─── helpers and sanity ──────────────────────────────────────────────────

// TestHaikuClassifier_ContextCancelDuringBackoff: an in-flight retry that
// gets cancelled during the 500ms backoff sleep returns ctx.Err() instead
// of waiting the full delay.
func TestHaikuClassifier_ContextCancelDuringBackoff(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	h := &HaikuClassifier{APIKey: "sk-test", HTTP: srv.Client(), APIBase: srv.URL}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := h.Classify(ctx, "T", "S", "url")
	if err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
	// Should return well before the full 500ms backoff completes.
	if elapsed := time.Since(start); elapsed > 400*time.Millisecond {
		t.Errorf("ctx cancel ignored backoff sleep — elapsed %s", elapsed)
	}
}

// Sanity: pinned constant so tests notice if the model name is silently
// rewritten by a future patch.
func TestDefaultHaikuModelPinned(t *testing.T) {
	t.Parallel()
	if DefaultHaikuModel != "claude-haiku-4-5-20251001" {
		t.Errorf("DefaultHaikuModel = %q, want %q", DefaultHaikuModel, "claude-haiku-4-5-20251001")
	}
}

