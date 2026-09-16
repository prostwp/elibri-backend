package demobot

// narrative_readable_test.go — Narrative Radar stage 1 (2026-09-15): the
// card's words, states, blocks and machine fields. The rules are untouched
// (5 matched items in 24h checked on the leader by the backend's score, the
// score formula, the keyword dictionary, the leader pick); every case runs
// the pure builder (newsCardFrom), so the texts are golden.

import (
	"context"
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"testing"
	"unicode/utf8"
)

// narrBelow22 is shaped like the prod radar of 2026-09-15 22:09: the leader
// by score has 1 matched item (a new theme: growth saturates the score), a
// theme with 3 sits lower, zero-item themes follow.
const narrBelow22 = `{"captured_at":"2026-09-15T22:00:00Z","narratives":[
 {"narrative":"restaking","mention_count_24h":1,"mention_count_prev_24h":0,"growth_pct":999,"trend_score":45,"stage":"early",
  "sentiment_label":"bull","confidence":40,"sources_breakdown":{"cointelegraph":1},
  "generated_idea":"Restaking chatter appeared this cycle."},
 {"narrative":"modular","mention_count_24h":1,"mention_count_prev_24h":0,"growth_pct":999,"trend_score":41,"stage":"early",
  "sentiment_label":"neutral","confidence":40,"sources_breakdown":{"coindesk":1}},
 {"narrative":"stablecoins-regulation","mention_count_24h":3,"mention_count_prev_24h":2,"growth_pct":50,"trend_score":19,"stage":"early",
  "sentiment_label":"neutral","confidence":55,"sources_breakdown":{"coindesk":2,"cointelegraph":1}},
 {"narrative":"btc-etf","mention_count_24h":0,"mention_count_prev_24h":4,"growth_pct":-100,"trend_score":0,"stage":"declining",
  "sentiment_label":"neutral","confidence":25,"sources_breakdown":{}},
 {"narrative":"rwa","mention_count_24h":0,"mention_count_prev_24h":0,"growth_pct":0,"trend_score":0,"stage":"early",
  "sentiment_label":"neutral","confidence":25,"sources_breakdown":{}}]}`

// narrScored22: the leader passes the threshold. The second theme carries an
// idea too, so the test proves the AI text comes from the leader only.
const narrScored22 = `{"captured_at":"2026-09-15T22:00:00Z","narratives":[
 {"narrative":"rwa","mention_count_24h":8,"mention_count_prev_24h":3,"growth_pct":166.66666666666669,"trend_score":72,"stage":"trending",
  "sentiment_label":"bull","confidence":61,"sources_breakdown":{"coindesk":5,"cointelegraph":3},
  "generated_idea":"Tokenized treasury coverage widened across both outlets."},
 {"narrative":"btc-etf","mention_count_24h":6,"mention_count_prev_24h":6,"growth_pct":0,"trend_score":30,"stage":"early",
  "sentiment_label":"bear","confidence":55,"sources_breakdown":{"coindesk":6},"generated_idea":"Not the leader's text."},
 {"narrative":"zk","mention_count_24h":2,"mention_count_prev_24h":0,"growth_pct":999,"trend_score":28,"stage":"early",
  "sentiment_label":"neutral","confidence":40,"sources_breakdown":{"cointelegraph":2}},
 {"narrative":"memecoins","mention_count_24h":1,"mention_count_prev_24h":1,"growth_pct":0,"trend_score":9,"stage":"early",
  "sentiment_label":"neutral","confidence":30}]}`

// narrEligible22: the leader is below the threshold while a lower-scored
// theme has 6 matched items — the rule checks the leader only (stage 3).
const narrEligible22 = `{"captured_at":"2026-09-15T22:00:00Z","narratives":[
 {"narrative":"restaking","mention_count_24h":1,"mention_count_prev_24h":0,"growth_pct":999,"trend_score":45,"stage":"early",
  "sentiment_label":"neutral","confidence":40,"sources_breakdown":{"cointelegraph":1}},
 {"narrative":"modular","mention_count_24h":1,"mention_count_prev_24h":0,"growth_pct":999,"trend_score":41,"stage":"early",
  "sentiment_label":"neutral","confidence":40,"sources_breakdown":{"coindesk":1}},
 {"narrative":"zk","mention_count_24h":2,"mention_count_prev_24h":2,"growth_pct":0,"trend_score":15,"stage":"early",
  "sentiment_label":"neutral","confidence":40,"sources_breakdown":{"coindesk":2}},
 {"narrative":"btc-etf","mention_count_24h":6,"mention_count_prev_24h":6,"growth_pct":0,"trend_score":12,"stage":"early",
  "sentiment_label":"neutral","confidence":50,"sources_breakdown":{"coindesk":6}}]}`

const narrNone22 = `{"captured_at":"2026-09-15T22:00:00Z","narratives":[
 {"narrative":"btc-etf","mention_count_24h":0,"mention_count_prev_24h":4,"growth_pct":-100,"trend_score":0,"stage":"declining","sentiment_label":"neutral","confidence":25},
 {"narrative":"rwa","mention_count_24h":0,"mention_count_prev_24h":0,"growth_pct":0,"trend_score":0,"stage":"early","sentiment_label":"neutral","confidence":25},
 {"narrative":"zk","mention_count_24h":0,"mention_count_prev_24h":0,"growth_pct":0,"trend_score":0,"stage":"early","sentiment_label":"neutral","confidence":25}]}`

const narrEmpty = `{"captured_at":"","narratives":[]}`

func narrResp(t *testing.T, body string) *NarrativesResp {
	t.Helper()
	var n NarrativesResp
	if err := json.Unmarshal([]byte(body), &n); err != nil {
		t.Fatal(err)
	}
	return &n
}

func narrCard(t *testing.T, body string) Card {
	t.Helper()
	return newsCardFrom(narrResp(t, body))
}

func eqLines(t *testing.T, what string, got, want []string) {
	t.Helper()
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Errorf("%s:\n%s\nwant\n%s", what, strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
}

// ── backend guards ───────────────────────────────────────────────────────────

// dictionarySlugs reads the canonical theme ids from the backend classifier.
func dictionarySlugs(t *testing.T) []string {
	t.Helper()
	src, err := os.ReadFile("../narrative/classifier.go")
	if err != nil {
		t.Fatal(err)
	}
	s := string(src)
	start := strings.Index(s, "var narrativeKeywords = map[string][]string{")
	if start < 0 {
		t.Fatal("narrativeKeywords not found in the backend classifier")
	}
	end := strings.Index(s[start:], "\n}\n")
	if end < 0 {
		t.Fatal("narrativeKeywords block has no end")
	}
	var out []string
	for _, m := range regexp.MustCompile(`(?m)^\t"([a-z0-9-]+)":\s*\{`).FindAllStringSubmatch(s[start:start+end], -1) {
		out = append(out, m[1])
	}
	if len(out) < 10 {
		t.Fatalf("parsed only %d themes from the dictionary: %v", len(out), out)
	}
	return out
}

// Every theme of the dictionary has a human name, and the table carries no
// theme the dictionary does not have. The machine id stays the id.
func TestNarrativeThemeNamesCoverDictionary(t *testing.T) {
	slugs := dictionarySlugs(t)
	seen := map[string]bool{}
	for _, s := range slugs {
		seen[s] = true
		name, ok := narrativeThemeNames[s]
		if !ok || strings.TrimSpace(name) == "" {
			t.Errorf("theme %q has no human name", s)
			continue
		}
		if name == s {
			t.Errorf("theme %q: the name repeats the machine id", s)
		}
		if n := utf8.RuneCountInString(name); n > narrativeNameMaxRunes {
			t.Errorf("theme %q: name %q is %d runes > %d", s, name, n, narrativeNameMaxRunes)
		}
	}
	for s := range narrativeThemeNames {
		if !seen[s] {
			t.Errorf("name table carries %q, which the dictionary does not have", s)
		}
	}
	for id, want := range map[string]string{
		"rwa": "Real-world assets (RWA)", "btc-etf": "Bitcoin ETFs and ETF issuers",
		"restaking": "Restaking and liquid staking (LST/LRT)", "btc-l2": "Bitcoin L2s, Ordinals and Runes",
		"zk":                     "Zero-knowledge (ZK) networks",
		"stablecoins-regulation": "Stablecoins and regulation",
	} {
		if got := narrativeName(id); got != want {
			t.Errorf("narrativeName(%q) = %q, want %q", id, got, want)
		}
	}
	if narrativeName("new-theme-x") != "new-theme-x" {
		t.Error("an unknown id must show as the id, never be dropped")
	}
}

// What the card says about windows, sources, ordering, the AI text and the
// score is what the backend does. A change there must fail here first.
func TestNarrativeWordingMatchesBackend(t *testing.T) {
	read := func(p string) string {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	worker := read("../narrative/worker.go")
	for _, want := range []string{
		// shown counts: last 24h; the previous 24h feed only the growth
		"since24h := now.Add(-24 * time.Hour)",
		"since48h := now.Add(-48 * time.Hour)",
		"count24, err := w.Store.CountMentions(ctx, slug, since24h, now)",
		"countPrev, err := w.Store.CountMentions(ctx, slug, since48h, since24h)",
		// three sources; per-source errors are logged, never served
		"news.FetchReddit(ctx, querySymbol, newsHoursWindow)",
		`"coindesk", "crypto", newsHoursWindow)`,
		`"cointelegraph", "crypto", newsHoursWindow)`,
		"func fetchAllSources(ctx context.Context, logger *log.Logger) []news.Item {",
		"fetch error (continuing)",
	} {
		if !strings.Contains(worker, want) {
			t.Errorf("backend worker changed (%q): update narrative_text.go and the docs", want)
		}
	}
	store := read("../narrative/store.go")
	if !strings.Contains(store, "return out[i].TrendScore > out[j].TrendScore") ||
		!strings.Contains(store, "return out[i].Narrative < out[j].Narrative") {
		t.Error("the leaderboard order changed: the card says 'ordered by activity score'")
	}
	handler := read("../api/narrative_handlers.go")
	if !strings.Contains(handler, "wrapped[0].MentionCount24h > 0") || !strings.Contains(handler, "top := &wrapped[0]") {
		t.Error("the backend attaches the idea to a different narrative now: the card reads it from the leader only")
	}
	scorer := read("../narrative/scorer.go")
	for _, re := range []string{
		`trendGrowthMaxPoints\s*=\s*35\.0`, `trendVolumeMaxPoints\s*=\s*25\.0`, `trendSentimentMaxPoints\s*=\s*15\.0`,
		`trendDiversityMaxPoints\s*=\s*10\.0`, `trendImportanceMaxPoints\s*=\s*15\.0`,
		`confVolumeMaxPoints\s*=\s*30\.0`, `confDiversityMaxPoints\s*=\s*45\.0`, `confConsistencyMaxPoints\s*=\s*25\.0`,
	} {
		if !regexp.MustCompile(re).MatchString(scorer) {
			t.Errorf("score formula changed (%s): the activity score / data quality lines name its parts", re)
		}
	}
	if newsMinMentions != 5 {
		t.Errorf("threshold %d: the rule is not changed in stage 1", newsMinMentions)
	}
	if narrativeFooter != "matched items: last 24h · growth vs previous 24h" {
		t.Errorf("footer %q", narrativeFooter)
	}
	if strings.Join(narrativeExpectedSources, ",") != "coindesk,cointelegraph,reddit" {
		t.Errorf("expected sources %v", narrativeExpectedSources)
	}
}

// ── golden cards ─────────────────────────────────────────────────────────────

func TestNarrativeGoldenBelowThreshold(t *testing.T) {
	c := narrCard(t, narrBelow22)
	if want := "Below threshold: Restaking and liquid staking (LST/LRT) ranks first, 1 matched item/24h; 5 needed"; c.Verdict != want {
		t.Errorf("verdict:\n%q\nwant\n%q", c.Verdict, want)
	}
	if c.Short != "below threshold" {
		t.Errorf("short %q", c.Short)
	}
	eqLines(t, "facts", c.Facts, []string{
		narrativeLineOrderBelow,
		"Restaking and liquid staking (LST/LRT) — 1 matched item · previous 24h: 0",
		"Modular blockchains — 1 matched item · previous 24h: 0",
		"Stablecoins and regulation — 3 matched items · previous 24h: 2",
		"Top activity score theme's matched items by source: CoinTelegraph 1",
		narrativeLineSources,
		narrativeLineMatch,
	})
	if narrativeLineOrderBelow != "Order: by activity score, not by count · scores are not shown below the threshold" {
		t.Errorf("order line %q", narrativeLineOrderBelow)
	}
	if narrativeLineSources != "Source status (which feeds answered this cycle) is not served by the backend" {
		t.Errorf("sources line %q", narrativeLineSources)
	}
	if narrativeLineMatch != "Matched item: a theme keyword in an RSS headline/summary or a Reddit title/author line; precision unmeasured" {
		t.Errorf("match line %q", narrativeLineMatch)
	}
	if c.Emoji != emojiNeutral || c.Confidence != nil || c.AIHTML != "" || c.Offline {
		t.Errorf("emoji %s confidence %v ai %q offline %v", c.Emoji, c.Confidence, c.AIHTML, c.Offline)
	}
	if c.effectiveStatus() != statusBelowThreshold {
		t.Errorf("status %v", c.effectiveStatus())
	}
	if c.SourceNote != narrativeFooter || !c.noValidator {
		t.Errorf("footer %q validator-free %v", c.SourceNote, c.noValidator)
	}
	for i, f := range c.Facts {
		if regexp.MustCompile(`^\d+\. `).MatchString(f) {
			t.Errorf("fact %d numbered below the threshold: %q", i, f)
		}
	}

	b := c.Blocks
	if b == nil {
		t.Fatal("blocks missing")
	}
	if want := "Restaking and liquid staking (LST/LRT) ranks first, 1 matched item in the 24h to Sep 15 22:00 UTC"; b.WhatHappened != want {
		t.Errorf("what_happened %q", b.WhatHappened)
	}
	if b.WhyLevel != narrativeWhyBelow || b.Invalidates != nil || b.Regime != narrativeRegimeBelow {
		t.Errorf("why %q invalidates %v regime %q", b.WhyLevel, b.Invalidates, b.Regime)
	}
	eqLines(t, "scenarios", b.Scenarios, []string{
		"If the top activity score theme reaches 5 matched items in 24h, the radar scores it",
		"If the top activity score theme stays under 5 matched items in 24h, the radar stays below threshold",
	})
	if b.Limitations != narrativeLimitations || b.Source != narrativeBlockSource {
		t.Errorf("limitations %q source %q", b.Limitations, b.Source)
	}

	ro := c.Narrative
	if ro == nil {
		t.Fatal("readout missing")
	}
	got, _ := json.Marshal(ro)
	want := `{"state":"below_threshold","window":"24h","growth_vs":"previous_24h","window_end":"2026-09-15T22:00:00Z",` +
		`"count_kind":"matched_items","threshold":5,"threshold_checked_on":"leader_by_activity_score","order":"activity_score_desc",` +
		`"leader":{"id":"restaking","name":"Restaking and liquid staking (LST/LRT)","matched_24h":1,"matched_prev_24h":0,"growth_pct":null,"new_theme":true,` +
		`"activity_score":null,"data_quality":null,"stage":null,"tone":null,"sources_with_matches":{"cointelegraph":1}},` +
		`"themes":[` +
		`{"id":"restaking","name":"Restaking and liquid staking (LST/LRT)","matched_24h":1,"matched_prev_24h":0,"growth_pct":null,"new_theme":true,"activity_score":null,"data_quality":null,"stage":null,"tone":null,"sources_with_matches":{"cointelegraph":1}},` +
		`{"id":"modular","name":"Modular blockchains","matched_24h":1,"matched_prev_24h":0,"growth_pct":null,"new_theme":true,"activity_score":null,"data_quality":null,"stage":null,"tone":null,"sources_with_matches":{"coindesk":1}},` +
		`{"id":"stablecoins-regulation","name":"Stablecoins and regulation","matched_24h":3,"matched_prev_24h":2,"growth_pct":50,"new_theme":false,"activity_score":null,"data_quality":null,"stage":null,"tone":null,"sources_with_matches":{"coindesk":2,"cointelegraph":1}}],` +
		`"eligible_not_leader":[],"themes_served":5,` +
		`"sources":{"expected":["coindesk","cointelegraph","reddit"],"status":null,"coverage":null}}`
	if string(got) != want {
		t.Errorf("readout:\n%s\nwant\n%s", got, want)
	}
}

func TestNarrativeGoldenScored(t *testing.T) {
	c := narrCard(t, narrScored22)
	if want := "Leading theme: Real-world assets (RWA) · activity score 72/100 · 8 matched items in 24h"; c.Verdict != want {
		t.Errorf("verdict:\n%q\nwant\n%q", c.Verdict, want)
	}
	if c.Short != "Real-world assets (RWA) · activity 72/100" {
		t.Errorf("short %q", c.Short)
	}
	eqLines(t, "facts", c.Facts, []string{
		"Top themes by activity score (0-100), not by count",
		"1. Real-world assets (RWA) — activity score 72 · 8 matched items · previous 24h: 3",
		"2. Bitcoin ETFs and ETF issuers — activity score 30 · 6 matched items · previous 24h: 6",
		"3. Zero-knowledge (ZK) networks — activity score 28 · 2 matched items · previous 24h: 0",
		"Stage label: trending (backend rule on 24h count and growth vs previous 24h)",
		"Tone of the leader's matched items (keyword sentiment of their text), not a price direction: positive",
		"Leader's matched items by source: CoinDesk 5 · CoinTelegraph 3",
		"Data quality 61/100 is not a probability: it counts items, sources with matches, tone agreement",
		narrativeLineSources,
		narrativeLineMatch,
	})
	if c.Emoji != emojiBull {
		t.Errorf("semaphore rule unchanged: a bull scored leader is 🟢, got %s", c.Emoji)
	}
	if c.effectiveStatus() != statusOK || c.Confidence == nil || *c.Confidence != 61 {
		t.Errorf("status %v confidence %v", c.effectiveStatus(), c.Confidence)
	}
	if want := "<b>AI comment:</b> <i>Tokenized treasury coverage widened across both outlets.</i>"; c.AIHTML != want {
		t.Errorf("ai %q", c.AIHTML)
	}
	html := c.RenderHTML()
	if !strings.Contains(html, "\nData quality: ■■■□□ 61/100\n") || strings.Contains(html, "Confidence") {
		t.Errorf("the bar must read data quality, not confidence:\n%s", html)
	}
	if !strings.Contains(html, "2026-09-15 22:00 UTC · matched items: last 24h · growth vs previous 24h</i>") {
		t.Errorf("footer:\n%s", html)
	}

	b := c.Blocks
	if b == nil {
		t.Fatal("blocks missing")
	}
	if want := "Real-world assets (RWA): 8 matched items in the 24h to Sep 15 22:00 UTC, 3 in the previous 24h"; b.WhatHappened != want {
		t.Errorf("what_happened %q", b.WhatHappened)
	}
	if want := "No price level: activity score 72/100 blends item growth, volume, tone, impact and source count"; b.WhyLevel != want {
		t.Errorf("why_level %q", b.WhyLevel)
	}
	eqLines(t, "scenarios", b.Scenarios, []string{
		"If the leading theme keeps 5+ matched items in 24h and the top score, the radar keeps scoring it",
		"If it drops under 5 matched items in 24h, the radar goes below threshold and hides scores",
	})
	if b.Invalidates == nil || *b.Invalidates != "Under 5 matched items in 24h, or another theme taking a higher activity score" {
		t.Errorf("invalidates %v", b.Invalidates)
	}
	if want := "Local news regime: one theme scored, Real-world assets (RWA); no price direction"; b.Regime != want {
		t.Errorf("regime %q", b.Regime)
	}

	ro := c.Narrative
	if ro == nil || ro.State != narrStateScored || ro.Leader == nil {
		t.Fatalf("readout %+v", ro)
	}
	l := ro.Leader
	if l.ID != "rwa" || l.ActivityScore == nil || *l.ActivityScore != 72 || l.DataQuality == nil || *l.DataQuality != 61 ||
		l.Stage == nil || *l.Stage != "trending" || l.Tone == nil || *l.Tone != "positive" ||
		l.GrowthPct == nil || *l.GrowthPct != 166.66666666666669 || l.MatchedPrev24h == nil || *l.MatchedPrev24h != 3 {
		t.Errorf("leader %+v", l)
	}
	if len(ro.Themes) != 3 || ro.Themes[1].Tone == nil || *ro.Themes[1].Tone != "negative" || ro.ThemesServed != 4 {
		t.Errorf("themes %+v served %d", ro.Themes, ro.ThemesServed)
	}
	if len(ro.EligibleNotLeader) != 0 {
		t.Errorf("eligible_not_leader is a below-threshold note: %v", ro.EligibleNotLeader)
	}

	// The AI text is the leader's or nothing: a leader without an idea never
	// borrows another theme's paragraph.
	noIdea := strings.Replace(narrScored22, `"generated_idea":"Tokenized treasury coverage widened across both outlets."`, `"generated_idea":""`, 1)
	if c2 := narrCard(t, noIdea); c2.AIHTML != "" {
		t.Errorf("AI text borrowed from another theme: %q", c2.AIHTML)
	}
}

// Below the threshold on the leader while a lower-scored theme passes it:
// the card says the rule checks the leader only, and changes nothing.
func TestNarrativeEligibleNotLeader(t *testing.T) {
	c := narrCard(t, narrEligible22)
	if c.effectiveStatus() != statusBelowThreshold {
		t.Fatalf("the leader decides (rule unchanged): status %v", c.effectiveStatus())
	}
	eqLines(t, "facts", c.Facts, []string{
		narrativeLineOrderBelow,
		"Restaking and liquid staking (LST/LRT) — 1 matched item · previous 24h: 0",
		"Modular blockchains — 1 matched item · previous 24h: 0",
		"Zero-knowledge (ZK) networks — 2 matched items · previous 24h: 2",
		"Checked on the top activity score theme only: Bitcoin ETFs and ETF issuers has 6 matched items",
		"Top activity score theme's matched items by source: CoinTelegraph 1",
		narrativeLineSources,
		narrativeLineMatch,
	})
	if ro := c.Narrative; ro == nil || strings.Join(ro.EligibleNotLeader, ",") != "btc-etf" {
		t.Errorf("eligible_not_leader %+v", ro)
	}
}

func TestNarrativeGoldenNoMatchedItems(t *testing.T) {
	c := narrCard(t, narrNone22)
	if want := "Below threshold: no theme has a matched item in the last 24h; 5 needed to score"; c.Verdict != want {
		t.Errorf("verdict %q", c.Verdict)
	}
	if c.Short != "no matched items" || c.effectiveStatus() != statusBelowThreshold || c.Offline {
		t.Errorf("short %q status %v offline %v", c.Short, c.effectiveStatus(), c.Offline)
	}
	eqLines(t, "facts", c.Facts, []string{
		"Themes served: 3 · each has 0 matched items in the last 24h",
		narrativeLineSources,
		narrativeLineMatch,
	})
	if c.Blocks == nil || c.Blocks.WhatHappened != "No theme had a matched item in the 24h to Sep 15 22:00 UTC" {
		t.Errorf("blocks %+v", c.Blocks)
	}
	if ro := c.Narrative; ro == nil || ro.State != narrStateNoMatched || ro.Leader == nil || ro.Leader.ID != "btc-etf" {
		t.Errorf("readout %+v", ro)
	}
}

func TestNarrativeNoSnapshots(t *testing.T) {
	c := narrCard(t, narrEmpty)
	if c.Verdict != "No radar snapshots from the backend yet — nothing to read" || c.Short != "no data" {
		t.Errorf("verdict %q short %q", c.Verdict, c.Short)
	}
	// The old contract stays: a 503 with reason below_threshold.
	if !c.Offline || c.effectiveStatus() != statusBelowThreshold {
		t.Errorf("offline %v status %v", c.Offline, c.effectiveStatus())
	}
	eqLines(t, "facts", c.Facts, []string{narrativeLineSources})
	if c.Blocks != nil || c.AIHTML != "" || c.Confidence != nil {
		t.Errorf("nothing to block, comment or rate: %+v", c)
	}
	if c.Narrative == nil || c.Narrative.State != narrStateNoSnapshots || c.Narrative.Leader != nil || len(c.Narrative.Themes) != 0 {
		t.Errorf("readout %+v", c.Narrative)
	}
}

// An older backend without prev counts and source breakdown: the card
// drops those parts instead of printing zeros it was not given.
func TestNarrativeOlderPayload(t *testing.T) {
	c := narrCard(t, narrativesFixture)
	if c.Facts[1] != "1. ai-agents — activity score 84 · 412 matched items" {
		t.Errorf("line without prev: %q", c.Facts[1])
	}
	for _, f := range c.Facts {
		if strings.Contains(f, "previous 24h:") || strings.HasPrefix(f, "Leader's matched items") {
			t.Errorf("printed a value the payload did not carry: %q", f)
		}
	}
	if c.Narrative.Leader.MatchedPrev24h != nil || c.Narrative.Leader.GrowthPct != nil || c.Narrative.Leader.NewTheme != nil {
		t.Errorf("prev/growth must be null: %+v", c.Narrative.Leader)
	}
}

// ── every path: lengths, words, stability ────────────────────────────────────

func narrAllCards(t *testing.T) map[string]Card {
	t.Helper()
	out := map[string]Card{
		"below": narrCard(t, narrBelow22), "scored": narrCard(t, narrScored22), "eligible": narrCard(t, narrEligible22),
		"none": narrCard(t, narrNone22), "empty": narrCard(t, narrEmpty), "older": narrCard(t, narrativesFixture),
	}
	// The longest name of the table in every state, at large counts.
	longest := ""
	for _, n := range narrativeThemeNames {
		if utf8.RuneCountInString(n) > utf8.RuneCountInString(longest) {
			longest = n
		}
	}
	var longID string
	for id, n := range narrativeThemeNames {
		if n == longest {
			longID = id
		}
	}
	for _, cnt := range []int{1, 4, 5, 999, 12345} {
		for _, second := range []int{3, 12345} {
			r := &NarrativesResp{CapturedAt: "2026-09-15T22:00:00Z", Narratives: []NarrativeSnapshot{
				{Narrative: longID, TrendScore: 100, Stage: "mainstream", SentimentLabel: "bear", MentionCount: cnt,
					MentionCountPrev24h: intPtr(99999), Confidence: 100,
					SourcesBreakdown: map[string]int{"cointelegraph": 99999, "coindesk": 99999, "reddit": 99999}},
				{Narrative: longID, TrendScore: 99, Stage: "declining", MentionCount: second, MentionCountPrev24h: intPtr(99999)},
				{Narrative: longID, TrendScore: 98, Stage: "early", MentionCount: second, MentionCountPrev24h: intPtr(0)},
			}}
			out[longID+"/"+itoa(cnt)+"/"+itoa(second)] = newsCardFrom(r)
		}
	}
	return out
}

func itoa(v int) string { b, _ := json.Marshal(v); return string(b) }

func narrTexts(c Card) []string {
	out := append([]string{c.Verdict, c.Short}, c.Facts...)
	if b := c.Blocks; b != nil {
		out = append(out, b.WhatHappened, b.WhyLevel, b.Regime, b.Limitations, b.Source)
		out = append(out, b.Scenarios...)
		if b.Invalidates != nil {
			out = append(out, *b.Invalidates)
		}
	}
	return out
}

func TestNarrativeLinesFit(t *testing.T) {
	for key, c := range narrAllCards(t) {
		for _, s := range narrTexts(c) {
			if n := utf8.RuneCountInString(s); n > narrativeFactMaxRunes {
				t.Errorf("%s: %d runes > %d: %q", key, n, narrativeFactMaxRunes, s)
			}
		}
		if n := utf8.RuneCountInString(htmlToPlain(c.OneLiner())); n > narrativeFactMaxRunes {
			t.Errorf("%s: one-liner %d runes", key, n)
		}
	}
	if n := utf8.RuneCountInString(howTexts[keyNews]); n > 200 {
		t.Errorf("how-text %d > 200 runes", n)
	}
}

// No forecast, price or trading words, and none of the old claims (48h
// window, "mentions", "trend score", "confidence", "warming up" without a
// known start time) on any path a reader sees.
func TestNarrativeNoForecastWords(t *testing.T) {
	banned := []string{"bullish", "bearish", " buy", " sell", "will ", "forecast says", "predict", "target", "pump",
		"moon", "breakout", "rally", "signal", "likely", "expect", "48h", "mention", "trend score", "confidence",
		"warming up", "top narrative", "odds", "probab"}
	for key, c := range narrAllCards(t) {
		b, _ := json.Marshal(c.Blocks)
		// The AI paragraph is the backend's text (its prompt and filter are
		// the guard there); this test checks the card's own words.
		c.AIHTML = ""
		// "not a probability" is the one allowed use: it denies the reading.
		all := strings.ToLower(c.RenderHTML() + "\n" + htmlToPlain(c.OneLiner()) + "\n" + string(b))
		all = strings.ReplaceAll(all, "is not a probability", "")
		for _, w := range banned {
			if strings.Contains(all, w) {
				t.Errorf("%s: %q must not appear:\n%s", key, w, all)
			}
		}
	}
	how := strings.ToLower(howTexts[keyNews])
	for _, w := range banned {
		if strings.Contains(how, w) {
			t.Errorf("how-text carries %q: %q", w, howTexts[keyNews])
		}
	}
	if strings.Contains(helpText, "48h") || strings.Contains(helpText, "trending crypto narratives") {
		t.Error("/help still describes the radar as a 48h trending list")
	}
}

// The body is a function of the payload: two builds are byte-identical, so
// the push hook's hash stays put while the data does.
func TestNarrativeBodyStable(t *testing.T) {
	for _, body := range []string{narrBelow22, narrScored22, narrNone22} {
		a, b := narrCard(t, body), narrCard(t, body)
		ja, _ := json.Marshal(cardEnvelope(a))
		jb, _ := json.Marshal(cardEnvelope(b))
		if string(ja) != string(jb) {
			t.Errorf("body moved between two builds:\n%s\n%s", ja, jb)
		}
	}
}

// ── HTTP, digest, showcase ───────────────────────────────────────────────────

func TestNarrativeEnvelope(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/narratives": narrBelow22})
	_, srv := newTestAPI(t, ag, true)
	status, hdr, body := httpGet(t, srv.URL+"/agents/news")
	if status != 200 {
		t.Fatalf("status %d: %s", status, body)
	}
	if hdr.Get("Last-Modified") != "" {
		t.Errorf("/agents/news must stay without Last-Modified, got %q", hdr.Get("Last-Modified"))
	}
	var env map[string]any
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env["ok"] != false || env["reason"] != "below_threshold" || env["semaphore"] != "neutral" {
		t.Errorf("ok %v reason %v semaphore %v", env["ok"], env["reason"], env["semaphore"])
	}
	ro, ok := env["narrative"].(map[string]any)
	if !ok || ro["state"] != "below_threshold" || ro["threshold"] != float64(5) {
		t.Errorf("narrative readout %v", env["narrative"])
	}
	blocks, ok := env["blocks"].(map[string]any)
	if !ok || blocks["source"] != narrativeBlockSource {
		t.Errorf("blocks %v", env["blocks"])
	}
	if env["confidence"] != nil || env["ai_text"] != nil {
		t.Errorf("no data quality and no AI text below the threshold: %v %v", env["confidence"], env["ai_text"])
	}

	ag2 := newStubBackend(t, map[string]string{"/api/v1/narratives": narrScored22})
	_, srv2 := newTestAPI(t, ag2, true)
	_, _, body2 := httpGet(t, srv2.URL+"/agents/news")
	var env2 map[string]any
	if err := json.Unmarshal(body2, &env2); err != nil {
		t.Fatal(err)
	}
	// The envelope's confidence value stays (additive-only contract); the
	// docs name it data quality for this agent.
	if env2["ok"] != true || env2["confidence"] != float64(61) || env2["semaphore"] != "bullish" {
		t.Errorf("scored: ok %v confidence %v semaphore %v", env2["ok"], env2["confidence"], env2["semaphore"])
	}
	if env2["ai_text"] != "AI comment: Tokenized treasury coverage widened across both outlets." {
		t.Errorf("ai_text %v", env2["ai_text"])
	}
}

func TestNarrativeDigestLine(t *testing.T) {
	stubExternalBases(t)
	ag := newStubBackend(t, map[string]string{"/api/v1/narratives": narrScored22})
	g := ag.gather(context.Background())
	want := "📖 <b>Narrative</b>: Real-world assets (RWA) (activity score 72, 8 matched items/24h)"
	found := false
	for _, ex := range g.extras {
		if ex == want {
			found = true
		}
		if strings.Contains(ex, "📖") && (strings.Contains(ex, "score 72)") || strings.Contains(ex, "trending")) {
			t.Errorf("old digest wording: %q", ex)
		}
	}
	if !found {
		t.Errorf("digest extra: %v, want %q", g.extras, want)
	}
	if got := narrativeAILine(*g.topNarr); got != "Real-world assets (RWA) (stage trending, activity score 72, 8 matched items/24h)" {
		t.Errorf("AI payload line %q", got)
	}
}

// The semaphore rule is the pre-stage-1 one (b39650c NewsCard): a scored
// leader's sentiment_label colors it (bull 🟢, bear 🔴, else ⚪); no
// snapshots and below the threshold are ⚪. Honesty is in the words (the
// tone line), not the color. The showcase conclusion carries no direction
// on a colored card.
func TestNarrativeSemaphoreMatchesBase(t *testing.T) {
	bear := strings.Replace(narrScored22, `"trend_score":72,"stage":"trending",
  "sentiment_label":"bull"`, `"trend_score":72,"stage":"trending",
  "sentiment_label":"bear"`, 1)
	neutral := strings.Replace(narrScored22, `"trend_score":72,"stage":"trending",
  "sentiment_label":"bull"`, `"trend_score":72,"stage":"trending",
  "sentiment_label":"neutral"`, 1)
	unlabeled := strings.Replace(narrScored22, `"trend_score":72,"stage":"trending",
  "sentiment_label":"bull"`, `"trend_score":72,"stage":"trending",
  "sentiment_label":""`, 1)
	if bear == narrScored22 || neutral == narrScored22 || unlabeled == narrScored22 {
		t.Fatal("fixture replace did not apply")
	}
	for name, tc := range map[string]struct {
		body string
		want string
	}{
		"scored bull": {narrScored22, emojiBull}, "scored bear": {bear, emojiBear},
		"scored neutral": {neutral, emojiNeutral}, "scored unlabeled": {unlabeled, emojiNeutral},
		"older bull":   {narrativesFixture, emojiBull},
		"below bull":   {narrBelow22, emojiNeutral}, // leader is bull, but below the threshold
		"eligible":     {narrEligible22, emojiNeutral},
		"no matched":   {narrNone22, emojiNeutral},
		"no snapshots": {narrEmpty, emojiNeutral},
	} {
		c := narrCard(t, tc.body)
		if c.Emoji != tc.want {
			t.Errorf("%s: emoji %s, want %s (base rule)", name, c.Emoji, tc.want)
		}
		if c.Emoji != emojiNeutral {
			found := false
			for _, f := range c.Facts {
				if strings.HasPrefix(f, "Tone of the leader's matched items (keyword sentiment of their text), not a price direction:") {
					found = true
				}
			}
			if !found {
				t.Errorf("%s: a colored card must say the color is headline tone: %v", name, c.Facts)
			}
			s := strings.ToLower(conclusionFor(c))
			for _, w := range []string{"bullish", "bearish", "lean", "neutral reading", " up", " down"} {
				if strings.Contains(s, w) {
					t.Errorf("%s: showcase conclusion carries %q on a %s card: %q", name, w, c.Emoji, s)
				}
			}
		}
	}
	if c := narrCard(t, bear); c.Emoji != emojiBear || !strings.HasSuffix(c.Facts[5], ": negative") {
		t.Errorf("bear tone line: %v", c.Facts)
	}
	if off := offlineCard("Narrative Radar", "Narrative", "", keyNews, howTexts[keyNews]); off.Emoji != emojiNeutral {
		t.Errorf("offline: %s", off.Emoji)
	}
}

// new_theme is computed from the counts: the backend does not store
// is_new_theme (LatestSnapshots does not select it) and serves false on
// every row. The fixtures are prod-shaped, without the field.
func TestNarrativeNewThemeFromCounts(t *testing.T) {
	if strings.Contains(narrBelow22, "is_new_theme") {
		t.Fatal("fixture must be shaped like the prod answer (no is_new_theme)")
	}
	th := narrCard(t, narrBelow22).Narrative.Themes
	if th[0].NewTheme == nil || !*th[0].NewTheme {
		t.Errorf("restaking prev 0 → cur 1 is a new theme: %v", th[0].NewTheme)
	}
	if th[2].NewTheme == nil || *th[2].NewTheme {
		t.Errorf("stablecoins prev 2 is not new: %v", th[2].NewTheme)
	}
	// The prod wire value false is ignored.
	prod := strings.Replace(narrBelow22, `"growth_pct":999,"trend_score":45`, `"growth_pct":999,"is_new_theme":false,"trend_score":45`, 1)
	if prod == narrBelow22 {
		t.Fatal("replace did not apply")
	}
	if v := narrCard(t, prod).Narrative.Leader.NewTheme; v == nil || !*v {
		t.Errorf("served is_new_theme=false must not override the counts: %v", v)
	}
	if v := narrCard(t, narrNone22).Narrative.Themes[1].NewTheme; v == nil || *v {
		t.Errorf("0 → 0 is not new: %v", v)
	}
	store, err := os.ReadFile("../narrative/store.go")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(store), "is_new_theme") {
		t.Error("the backend now stores is_new_theme: read it instead and update the docs")
	}
}

// The showcase's explained line and first data line (the fallback path has
// no AI) are the leader's reading, never the list header.
func TestNarrativeShowcaseExampleLines(t *testing.T) {
	for name, tc := range map[string]struct{ body, want string }{
		"scored": {narrScored22, "1. Real-world assets (RWA) — activity score 72 · 8 matched items · previous 24h: 3"},
		"below":  {narrBelow22, "Restaking and liquid staking (LST/LRT) — 1 matched item · previous 24h: 0"},
		"none":   {narrNone22, "Themes served: 3 · each has 0 matched items in the last 24h"},
	} {
		c := narrCard(t, tc.body)
		if got := strongestFact(c); got != tc.want+"." {
			t.Errorf("%s: explained %q, want %q", name, got, tc.want+".")
		}
		d := exampleFacts(c)
		if len(d) == 0 || d[0] != tc.want {
			t.Errorf("%s: data %v, want first %q", name, d, tc.want)
		}
		for _, f := range d {
			if narrativeHeaderLine(f) {
				t.Errorf("%s: header in the example data: %v", name, d)
			}
		}
	}
}

func TestNarrativeShowcaseConclusion(t *testing.T) {
	for key, c := range map[string]Card{
		"scored": narrCard(t, narrScored22), "below": narrCard(t, narrBelow22), "none": narrCard(t, narrNone22),
		"empty":   narrCard(t, narrEmpty),
		"offline": offlineCard("Narrative Radar", "Narrative", "", keyNews, howTexts[keyNews]),
	} {
		s := conclusionFor(c)
		low := strings.ToLower(s)
		for _, w := range []string{"bullish", "bearish", "neutral reading", "leans", "level structure"} {
			if strings.Contains(low, w) {
				t.Errorf("%s: conclusion carries %q: %q", key, w, s)
			}
		}
		if !strings.HasSuffix(s, ".") {
			t.Errorf("%s: not a sentence: %q", key, s)
		}
	}
	if got := conclusionFor(narrCard(t, narrScored22)); got !=
		"This is a news-activity reading, not a forecast: Real-world assets (RWA) has 8 matched items in 24h; it says nothing about where price goes." {
		t.Errorf("scored conclusion %q", got)
	}
	if got := conclusionFor(narrCard(t, narrBelow22)); got !=
		"This is a news-activity reading, not a forecast: the radar is below threshold; the top activity score theme has 1 of the 5 matched items needed in 24h." {
		t.Errorf("below conclusion %q", got)
	}
	// The example's top-up line names data quality, never "confidence".
	for _, f := range exampleFacts(Card{Command: keyNews, Facts: []string{"a"}, Confidence: intPtr(61), confLabel: narrativeConfLabel}) {
		if strings.Contains(f, "Confidence") {
			t.Errorf("example fact %q", f)
		}
	}
}
