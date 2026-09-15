package demobot

// showcase_test.go — the landing showcase (showcase.go): catalog shape, the
// live/degraded classification, the 60s singleflight memo, and the worked
// example's narrative (detected → explained → data → conclusion) including
// its fallback and its honest 503. Every upstream is stubbed, so the suite is
// hermetic and deterministic.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// ── decode targets ───────────────────────────────────────────────────────────

type testShowcaseAgent struct {
	Slug       string  `json:"slug"`
	Name       string  `json:"name"`
	Category   string  `json:"category"`
	Status     string  `json:"status"`
	OK         bool    `json:"ok"`
	Reason     *string `json:"reason"`
	Headline   string  `json:"headline"`
	OneLiner   string  `json:"one_liner"`
	DataAsOf   string  `json:"data_as_of"`
	ExampleURL string  `json:"example_url"`
}

type testShowcase struct {
	GeneratedAt string              `json:"generated_at"`
	LiveCount   int                 `json:"live_count"`
	TotalCount  int                 `json:"total_count"`
	Agents      []testShowcaseAgent `json:"agents"`
}

type testShowcaseExample struct {
	GeneratedAt string          `json:"generated_at"`
	Agent       string          `json:"agent"`
	Slug        string          `json:"slug"`
	Asset       string          `json:"asset"`
	Detected    string          `json:"detected"`
	Explained   string          `json:"explained"`
	Data        []string        `json:"data"`
	Conclusion  string          `json:"conclusion"`
	Levels      json.RawMessage `json:"levels"`
	ExampleURL  string          `json:"example_url"`
	Disclaimer  string          `json:"disclaimer"`
	DataAsOf    string          `json:"data_as_of"`
}

// ── fixtures ─────────────────────────────────────────────────────────────────

// whaleLiveFixture is a real BTC flow snapshot: the whale agent is the one
// agent alive in the mixed-state tests. It sits OUTSIDE the priority trio
// (funding/momentum/trend), which is what makes it the natural fallback
// subject when a degraded macro holds the top slot.
const whaleLiveFixture = `{"captured_at":"2026-08-18T06:00:00Z",
  "flows":[{"asset":"BTC","net_flow_usd_24h":-18400000,"direction":"outflow",
            "inflow_usd_24h":5000000,"outflow_usd_24h":23400000,"tx_count_24h":37,
            "confidence":72,"partial":false,"source":"test",
            "net_flow_prev_24h":-9000000,"flow_pct":-104}],
  "transfers":[]}`

// mixedStateAgents: macro answers UNKNOWN (degraded, no_data), whale answers
// with a real reading, every other source is dead. Deviations are therefore
// empty → the priority rule falls back to the degraded macro card, which is
// exactly the state /showcase/example must fall back out of.
func mixedStateAgents(t *testing.T) *Agents {
	t.Helper()
	stubExternalBases(t)
	return newStubBackend(t, map[string]string{
		"/api/v1/macro":      macroUnknownFixture(true),
		"/api/v1/whale-flow": whaleLiveFixture,
	})
}

// newCountingBackend serves routes while counting hits per path and holding
// each response for `delay` — long enough that concurrent requests overlap
// the in-flight sweep, so a hit count of 1 proves singleflight and not just a
// fast cache fill.
func newCountingBackend(t *testing.T, routes map[string]string, delay time.Duration) (*Agents, func(string) int) {
	t.Helper()
	stubExternalBases(t)
	var mu sync.Mutex
	hits := map[string]int{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		hits[r.URL.Path]++
		mu.Unlock()
		if delay > 0 {
			time.Sleep(delay)
		}
		body, ok := routes[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return NewAgents(NewBackendClient(srv.URL)), func(p string) int {
		mu.Lock()
		defer mu.Unlock()
		return hits[p]
	}
}

func getShowcase(t *testing.T, base string) testShowcase {
	t.Helper()
	status, _, body := httpGet(t, base+"/showcase")
	if status != 200 {
		t.Fatalf("/showcase status %d: %s", status, body)
	}
	var out testShowcase
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("/showcase decode: %v (%s)", err, body)
	}
	return out
}

// ── catalog shape ────────────────────────────────────────────────────────────

// Every agent that exists is listed, in the /agents routing order, with every
// landing-facing field populated and a status of exactly live|degraded.
func TestShowcaseCatalogShape(t *testing.T) {
	_, srv := newTestAPI(t, mixedStateAgents(t), true)
	sc := getShowcase(t, srv.URL)

	if sc.TotalCount != len(httpAgentNames) || len(sc.Agents) != len(httpAgentNames) {
		t.Fatalf("total_count %d / rows %d, want %d (every agent that exists)",
			sc.TotalCount, len(sc.Agents), len(httpAgentNames))
	}
	if _, err := time.Parse(time.RFC3339, sc.GeneratedAt); err != nil {
		t.Errorf("generated_at %q is not RFC3339: %v", sc.GeneratedAt, err)
	}
	validCategory := map[string]bool{
		"crypto": true, "forex": true, "macro": true, "onchain": true,
		"derivatives": true, "news": true, "tools": true, "metals": true,
	}
	live := 0
	for i, row := range sc.Agents {
		if row.Slug != httpAgentNames[i] {
			t.Errorf("row %d slug %q, want %q (catalog keeps the /agents order)", i, row.Slug, httpAgentNames[i])
		}
		if row.Name == "" || row.Headline == "" || row.OneLiner == "" {
			t.Errorf("%s: empty name/headline/one_liner: %+v", row.Slug, row)
		}
		if strings.HasSuffix(strings.TrimSpace(row.OneLiner), ":") {
			t.Errorf("%s: dangling one_liner %q", row.Slug, row.OneLiner)
		}
		if !validCategory[row.Category] {
			t.Errorf("%s: category %q outside the landing's chip set", row.Slug, row.Category)
		}
		if want := "/agents/" + row.Slug; row.ExampleURL != want {
			t.Errorf("%s: example_url %q, want %q", row.Slug, row.ExampleURL, want)
		}
		if _, err := time.Parse(time.RFC3339, row.DataAsOf); err != nil {
			t.Errorf("%s: data_as_of %q is not RFC3339", row.Slug, row.DataAsOf)
		}
		switch row.Status {
		case showcaseLive:
			live++
			if !row.OK || row.Reason != nil {
				t.Errorf("%s: live row must be ok with null reason, got ok=%v reason=%v", row.Slug, row.OK, row.Reason)
			}
		case showcaseDegraded:
			if row.OK {
				t.Errorf("%s: degraded row must be ok=false", row.Slug)
			}
			if row.Reason == nil || *row.Reason == "" {
				t.Errorf("%s: degraded row must name its reason (honesty rule)", row.Slug)
			}
		default:
			t.Errorf("%s: status %q — only live|degraded exist on this endpoint", row.Slug, row.Status)
		}
	}
	if sc.LiveCount != live {
		t.Errorf("live_count %d, but %d rows are live", sc.LiveCount, live)
	}
}

// The name table is not allowed to drift from the builders it labels: for
// every card-backed agent the landing name IS the card's own Agent string.
// (digest/top are aggregates — their card belongs to the winning agent, so
// they carry their own identity and are excluded.)
func TestShowcaseNamesMatchBuilders(t *testing.T) {
	for _, tc := range []struct {
		name string
		ag   func(t *testing.T) *Agents
	}{
		{"all upstreams dead", deadAgents},
		{"mixed live/degraded", mixedStateAgents},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := tc.ag(t).buildShowcase(t.Context())
			for _, slug := range httpAgentNames {
				if slug == keyDigest || slug == keyTop {
					continue
				}
				c, ok := b.cards[slug]
				if !ok {
					t.Fatalf("%s: no card in the build", slug)
				}
				if showcaseNames[slug] != c.Agent {
					t.Errorf("%s: showcaseNames %q != builder %q", slug, showcaseNames[slug], c.Agent)
				}
			}
			if len(showcaseNames) != len(httpAgentNames) || len(showcaseCategories) != len(httpAgentNames) {
				t.Errorf("name/category tables must cover all %d agents, got %d/%d",
					len(httpAgentNames), len(showcaseNames), len(showcaseCategories))
			}
		})
	}
}

// ── live / degraded classification ───────────────────────────────────────────

// The whole point of the endpoint: an agent serving real data reads "live",
// a dark one reads "degraded" WITH its machine reason — and never disappears
// from the catalog.
func TestShowcaseLiveDegradedClassification(t *testing.T) {
	_, srv := newTestAPI(t, mixedStateAgents(t), true)
	sc := getShowcase(t, srv.URL)

	rows := map[string]testShowcaseAgent{}
	for _, r := range sc.Agents {
		rows[r.Slug] = r
	}
	// Live: the whale feed answered, the calculator is local arithmetic, and
	// the digest aggregates whatever is alive (whale) rather than going dark.
	for _, slug := range []string{keyWhale, keyRisk, keyDigest} {
		if rows[slug].Status != showcaseLive {
			t.Errorf("%s: status %q, want live (%+v)", slug, rows[slug].Status, rows[slug])
		}
	}
	// Degraded, each with the specific reason the card layer computed.
	wantReason := map[string]string{
		keyMacro:    "no_data",        // UNKNOWN regime inside the open window
		keyMomentum: "source_offline", // Binance dead
		keyTrend:    "source_offline",
		keySR:       "source_offline",
		keyVol:      "source_offline",
		keyFX:       "source_offline", // Yahoo dead
		keyNews:     "source_offline", // narrative radar dead
		keyFunding:  "source_offline",
	}
	for slug, reason := range wantReason {
		row := rows[slug]
		if row.Status != showcaseDegraded {
			t.Errorf("%s: status %q, want degraded", slug, row.Status)
			continue
		}
		if row.Reason == nil || *row.Reason != reason {
			t.Errorf("%s: reason %v, want %q", slug, row.Reason, reason)
		}
		if row.Headline == "" {
			t.Errorf("%s: a degraded agent still says what happened", slug)
		}
	}
	if sc.LiveCount == 0 || sc.LiveCount == sc.TotalCount {
		t.Errorf("live_count %d of %d — the mixed fixture must produce both states", sc.LiveCount, sc.TotalCount)
	}
}

// The defect this endpoint exists to fix: nothing in either payload may call
// a working agent "planned", "fictional", "coming soon" or "demo only".
func TestShowcaseNeverSaysPlanned(t *testing.T) {
	banned := []string{"planned", "fictional", "coming soon", "not implemented", "mock", "fake"}
	for _, tc := range []struct {
		name string
		ag   func(t *testing.T) *Agents
	}{
		{"mixed live/degraded", mixedStateAgents},
		{"all upstreams dead", deadAgents},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, srv := newTestAPI(t, tc.ag(t), true)
			for _, path := range []string{"/showcase", "/showcase/example"} {
				_, _, body := httpGet(t, srv.URL+path)
				low := strings.ToLower(string(body))
				for _, word := range banned {
					if strings.Contains(low, word) {
						t.Errorf("%s: response contains %q — these agents exist:\n%s", path, word, body)
					}
				}
			}
		})
	}
}

// ── memo: 60s + singleflight ─────────────────────────────────────────────────

// Ten concurrent landing renders must cost ONE sweep, not ten: the memo
// joins in-flight builds instead of starting new ones.
func TestShowcaseMemoSingleflight(t *testing.T) {
	ag, hits := newCountingBackend(t, map[string]string{
		"/api/v1/macro":      macroUnknownFixture(true),
		"/api/v1/whale-flow": whaleLiveFixture,
	}, 120*time.Millisecond)
	_, srv := newTestAPI(t, ag, true)

	const n = 10
	stamps := make([]string, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			status, _, body := httpGet(t, srv.URL+"/showcase")
			if status != 200 {
				t.Errorf("request %d: status %d", i, status)
				return
			}
			var out testShowcase
			if err := json.Unmarshal(body, &out); err != nil {
				t.Errorf("request %d decode: %v", i, err)
				return
			}
			stamps[i] = out.GeneratedAt
		}(i)
	}
	wg.Wait()

	for _, path := range []string{"/api/v1/macro", "/api/v1/whale-flow"} {
		if got := hits(path); got != 1 {
			t.Errorf("%s fetched %d times for %d concurrent renders, want 1", path, got, n)
		}
	}
	for i, s := range stamps {
		if s == "" || s != stamps[0] {
			t.Errorf("request %d generated_at %q != %q — joiners must share one sweep", i, s, stamps[0])
		}
	}
	// The example endpoint rides the same build: still one fetch per source.
	if status, _, body := httpGet(t, srv.URL+"/showcase/example"); status != 200 {
		t.Fatalf("/showcase/example status %d: %s", status, body)
	}
	if got := hits("/api/v1/whale-flow"); got != 1 {
		t.Errorf("/showcase/example re-swept upstreams: whale-flow hits %d, want 1", got)
	}
}

// A stale entry rebuilds; a sweep that found nothing alive is held for the
// SHORT window so a blip cannot freeze the landing on a blackout.
func TestShowcaseMemoTTLAndStaleRebuild(t *testing.T) {
	now := time.Now()
	live := &showcaseEntry{live: true, at: now.Add(-30 * time.Second)}
	dark := &showcaseEntry{live: false, at: now.Add(-30 * time.Second)}
	if !live.fresh(now) {
		t.Errorf("a live sweep must stay fresh for %s", showcaseTTL)
	}
	if dark.fresh(now) {
		t.Errorf("an all-dark sweep must expire after %s, not %s", showcaseDeadTTL, showcaseTTL)
	}

	ag, hits := newCountingBackend(t, map[string]string{
		"/api/v1/whale-flow": whaleLiveFixture,
	}, 0)
	s, srv := newTestAPI(t, ag, true)
	getShowcase(t, srv.URL)
	if got := hits("/api/v1/whale-flow"); got != 1 {
		t.Fatalf("first render: %d fetches, want 1", got)
	}
	getShowcase(t, srv.URL) // inside the window: served from the memo
	if got := hits("/api/v1/whale-flow"); got != 1 {
		t.Errorf("second render inside the window: %d fetches, want 1", got)
	}
	s.sc.mu.Lock() // age the entry past its TTL
	s.sc.cur.at = time.Now().Add(-2 * showcaseTTL)
	s.sc.mu.Unlock()
	getShowcase(t, srv.URL)
	if got := hits("/api/v1/whale-flow"); got != 2 {
		t.Errorf("render after expiry: %d fetches, want 2 (stale entries rebuild)", got)
	}
}

// ── the worked example ───────────────────────────────────────────────────────

// The narrative marketing asked for: every field populated, from live data.
func TestShowcaseExampleNarrativeFields(t *testing.T) {
	_, srv := newTestAPI(t, mixedStateAgents(t), true)
	status, _, body := httpGet(t, srv.URL+"/showcase/example")
	if status != 200 {
		t.Fatalf("status %d: %s", status, body)
	}
	var ex testShowcaseExample
	if err := json.Unmarshal(body, &ex); err != nil {
		t.Fatalf("decode: %v (%s)", err, body)
	}
	for name, v := range map[string]string{
		"agent": ex.Agent, "slug": ex.Slug, "detected": ex.Detected,
		"explained": ex.Explained, "conclusion": ex.Conclusion,
		"disclaimer": ex.Disclaimer, "data_as_of": ex.DataAsOf,
		"generated_at": ex.GeneratedAt, "example_url": ex.ExampleURL,
	} {
		if strings.TrimSpace(v) == "" {
			t.Errorf("%s must not be empty: %s", name, body)
		}
	}
	if len(ex.Data) == 0 || len(ex.Data) > showcaseFactsMax {
		t.Errorf("data has %d lines, want 1..%d: %v", len(ex.Data), showcaseFactsMax, ex.Data)
	}
	for _, d := range ex.Data {
		if strings.TrimSpace(d) == "" {
			t.Errorf("empty data line in %v", ex.Data)
		}
	}
	if !strings.Contains(ex.Detected, ex.Agent) {
		t.Errorf("detected %q must name the agent %q", ex.Detected, ex.Agent)
	}
	if !strings.HasSuffix(ex.Conclusion, ".") {
		t.Errorf("conclusion must be a sentence: %q", ex.Conclusion)
	}
	if ex.Disclaimer != disclaimerText {
		t.Errorf("disclaimer %q, want %q", ex.Disclaimer, disclaimerText)
	}
	if _, err := time.Parse(time.RFC3339, ex.DataAsOf); err != nil {
		t.Errorf("data_as_of %q is not RFC3339", ex.DataAsOf)
	}
}

// The advice-language rule reaches the landing too: no shouted BUY/SELL
// verdict anywhere in the story, whatever produced the text.
func TestShowcaseExampleNoAdviceLanguage(t *testing.T) {
	_, srv := newTestAPI(t, mixedStateAgents(t), true)
	_, _, body := httpGet(t, srv.URL+"/showcase/example")
	for _, banned := range []string{"BUY", "SELL"} {
		if strings.Contains(string(body), banned) {
			t.Errorf("example contains %q — analytical language only:\n%s", banned, body)
		}
	}
}

// When the priority winner is degraded the story falls back to a live agent
// instead of narrating a dead source: macro holds the top slot with UNKNOWN,
// whale is the one agent with real numbers, so whale tells the story.
func TestShowcaseExampleFallsBackWhenTopDegraded(t *testing.T) {
	ag := mixedStateAgents(t)
	b := ag.buildShowcase(t.Context())
	winner, top := topSelection(b.g)
	if winner != keyMacro || top.effectiveStatus() == statusOK {
		t.Fatalf("fixture must put a DEGRADED %s on top, got winner=%s status=%d", keyMacro, winner, top.effectiveStatus())
	}

	_, srv := newTestAPI(t, ag, true)
	status, _, body := httpGet(t, srv.URL+"/showcase/example")
	if status != 200 {
		t.Fatalf("status %d: %s", status, body)
	}
	var ex testShowcaseExample
	if err := json.Unmarshal(body, &ex); err != nil {
		t.Fatal(err)
	}
	if ex.Slug != keyWhale {
		t.Errorf("slug %q, want %q — the only agent with a real reading", ex.Slug, keyWhale)
	}
	if !strings.Contains(strings.ToLower(ex.Detected+ex.Explained), "flow") {
		t.Errorf("the story must be built from the whale card, got detected=%q explained=%q", ex.Detected, ex.Explained)
	}
}

// The fallback after a degraded winner reuses the digest's own rule for the
// trio (eligibility + confirmed tier), and a FIXED order for everything else
// — never a cross-agent "strongest" by raw Deviation.
func TestShowcaseExampleFallbackUsesDigestRule(t *testing.T) {
	degradedMacro := Card{Emoji: emojiNeutral, Agent: "Macro Agent", Verdict: "UNKNOWN", Status: statusNoData}
	fresh := goldenTime.Add(-time.Hour)
	okCard := func(agent string, dev int) Card {
		return Card{Emoji: emojiBull, Agent: agent, Verdict: agent + " reading", Short: "ok", Deviation: dev, DataTime: fresh}
	}
	build := func(cards map[string]Card) *showcaseBuild {
		return &showcaseBuild{
			// Only the degraded macro is in the sweep, so topSelection falls
			// back to it and the example has to choose among b.cards.
			g:     gathered{regime: "unknown", at: goldenTime, cards: map[string]Card{keyMacro: cards[keyMacro]}},
			cards: cards,
			at:    goldenTime,
		}
	}

	// Flat trend with a high raw ADX vs a confirmed but lower-scored
	// momentum → the confirmed one.
	flat := okCard("Trend Agent", 45)
	flat.State = trendFlat
	mom := okCard("Momentum Agent", 12)
	mom.confirmed = true
	b := build(map[string]Card{keyMacro: degradedMacro, keyTrend: flat, keyMomentum: mom, keySR: okCard("S/R Agent", 90)})
	if slug, card, ok := b.exampleCard(); !ok || slug != keyMomentum || card.Agent != "Momentum Agent" {
		t.Errorf("flat ADX 45 vs confirmed momentum 12: got %q (ok=%v), want %q", slug, ok, keyMomentum)
	}

	// A stale candidate is never picked, even confirmed and strongest.
	staleTrend := okCard("Trend Agent", 90)
	staleTrend.State, staleTrend.confirmed = trendUp, true
	staleTrend.DataTime = goldenTime.Add(-20 * time.Hour)
	b = build(map[string]Card{keyMacro: degradedMacro, keyTrend: staleTrend, keyWhale: okCard("Whale Flow Agent", 1)})
	if slug, _, ok := b.exampleCard(); !ok || slug != keyWhale {
		t.Errorf("stale trend: got %q (ok=%v), want the live whale card", slug, ok)
	}

	// Only non-trio cards live → the documented fixed order, whatever their
	// Deviation says (S/R 99 does not beat whale 1: whale comes first).
	b = build(map[string]Card{keyMacro: degradedMacro, keySR: okCard("S/R Agent", 99), keyWhale: okCard("Whale Flow Agent", 1), keyVol: okCard("Volatility Agent", 80)})
	if slug, _, ok := b.exampleCard(); !ok || slug != keyWhale {
		t.Errorf("non-trio fallback: got %q (ok=%v), want %q (exampleFallbackOrder)", slug, ok, keyWhale)
	}

	// A healthy winner is never overridden by a stronger-looking other agent.
	healthy := build(map[string]Card{keyMacro: okCard("Macro Agent", 5), keySR: okCard("S/R Agent", 99)})
	if slug, _, ok := healthy.exampleCard(); !ok || slug != keyMacro {
		t.Errorf("healthy top winner: got %q (ok=%v), want %q", slug, ok, keyMacro)
	}

	// Nothing alive → no story, and the caller must serve a 503.
	if slug, _, ok := build(map[string]Card{keyMacro: degradedMacro}).exampleCard(); ok {
		t.Errorf("all-degraded build returned %q — must refuse", slug)
	}
}

// Everything down: an honest 503 in the standard {error, ok, reason} shape —
// never a story assembled from offline cards.
func TestShowcaseExampleUnavailableWhenEverythingDown(t *testing.T) {
	_, srv := newTestAPI(t, deadAgents(t), true)
	status, _, body := httpGet(t, srv.URL+"/showcase/example")
	if status != http.StatusServiceUnavailable {
		t.Fatalf("status %d, want 503: %s", status, body)
	}
	var e struct {
		Error  string `json:"error"`
		OK     *bool  `json:"ok"`
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(body, &e); err != nil {
		t.Fatal(err)
	}
	if e.Error == "" || e.OK == nil || *e.OK || e.Reason == "" {
		t.Errorf("503 body must carry error + ok:false + reason, got %s", body)
	}
	if e.Reason != "source_offline" {
		t.Errorf("reason %q, want source_offline", e.Reason)
	}
	// The catalog still answers 200 — degraded agents stay listed.
	sc := getShowcase(t, srv.URL)
	if sc.TotalCount != len(httpAgentNames) {
		t.Errorf("catalog dropped rows in a blackout: %d of %d", sc.TotalCount, len(httpAgentNames))
	}
}

// The two worked-example constants are one number set in two forms — the
// Telegram button's strings and the calculator's floats must agree.
func TestRiskExampleValuesMatchButtonArgs(t *testing.T) {
	if len(riskExampleArgs) != len(riskExampleValues) {
		t.Fatalf("example arity: %d strings vs %d values", len(riskExampleArgs), len(riskExampleValues))
	}
	for i, s := range riskExampleArgs {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			t.Fatalf("riskExampleArgs[%d]=%q: %v", i, s, err)
		}
		if v != riskExampleValues[i] {
			t.Errorf("example arg %d: button %q != calculator %v", i, s, riskExampleValues[i])
		}
	}
}
