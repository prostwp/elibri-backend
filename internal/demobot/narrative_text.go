package demobot

// narrative_text.go — everything the Narrative Radar card SAYS (stage 1,
// 2026-09-15).
//
// The rules are unchanged: the backend matches CoinDesk and CoinTelegraph RSS
// items plus Reddit posts (when reachable) against a keyword dictionary per
// theme, counts matched items over the last 24h and over the 24h before it
// (growth only), scores each theme 0-100 and serves the themes sorted by
// that score (narrative/store.go LatestSnapshots). The card checks the
// threshold (newsMinMentions, 5 matched items in 24h) on the LEADER by score
// only — a lower-scored theme with more items does not change that (stage 3
// reviews the rule). What changed is only what the card claims:
//
//   - the window: the shown counts are the last 24h; the previous 24h feed only
//     the growth ("48h mention window" was wrong);
//   - "mentions" → "matched items": a match is a keyword hit in an RSS
//     headline or summary, or in a Reddit post's title or author line (the
//     backend matches title + summary; a Reddit summary is the author and
//     engagement line, so "u/<author>" can match), and the dictionary's
//     precision has not been measured;
//   - "trend score" → "activity score": the formula rewards positive tone and
//     gives negative tone nothing, so it is not a trend strength;
//   - "confidence" → "data quality" with its basis (items, sources with
//     matches, tone agreement), never a probability;
//   - below the threshold the themes are not numbered and the order is named
//     (activity score, not count); "warming up" is gone: the backend serves
//     no start time, so the card cannot know it is in its first hours;
//   - source status: the backend logs per-source fetch errors and serves none
//     (narrative/worker.go fetchAllSources), so the card says so; the
//     sources with matched items per theme are served and shown;
//   - human theme names (narrativeThemeNames), the machine id stays the id;
//   - the AI text is the leader's own paragraph, and only on a scored card;
//   - the semaphore rule is the old one (a scored leader's tone: bull 🟢,
//     bear 🔴, else ⚪; every other path ⚪); the words say what it is: the
//     tone of the leader's matched items, not a price direction;
//   - no price, forecast or trading words; every line fits
//     narrativeFactMaxRunes.

import (
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	narrativeFactMaxRunes = 110
	narrativeNameMaxRunes = 45
	// newsMinMentions is the radar's silence threshold, unchanged: below this
	// many matched items in 24h for the LEADER by activity score the card
	// shows no score, stage, tone, data quality or AI text. Name kept for the
	// digest (bot.go), which applies the same threshold.
	newsMinMentions = 5
	// narrativeListN: themes listed on the card, in the backend's order.
	narrativeListN = 3
	// narrativeConfLabel replaces "Confidence" on the bar (Card.confLabel).
	narrativeConfLabel = "Data quality"
)

// Narrative machine states (narrative.state).
const (
	narrStateScored      = "scored"           // the leader has ≥ 5 matched items in 24h
	narrStateBelow       = "below_threshold"  // the leader has 1-4
	narrStateNoMatched   = "no_matched_items" // snapshots served, every theme at 0
	narrStateNoSnapshots = "no_snapshots"     // the backend served no theme at all (503)
)

// narrativeExpectedSources are the feeds the backend worker polls
// (narrative/worker.go fetchAllSources; pinned by
// TestNarrativeWordingMatchesBackend).
var narrativeExpectedSources = []string{"coindesk", "cointelegraph", "reddit"}

var narrativeSourceNames = map[string]string{
	"coindesk": "CoinDesk", "cointelegraph": "CoinTelegraph", "reddit": "Reddit",
}

// narrativeThemeNames maps every theme id of the backend dictionary
// (narrative/classifier.go narrativeKeywords) to a reader's name. The names
// follow the keyword lists: "stablecoins-regulation" matches any USDT, USDC,
// Tether or Circle item, not only regulation news; "btc-etf" also matches
// BlackRock, Grayscale and Fidelity bitcoin items; "zk" also matches Aleo and
// Mina, which are not rollups; "restaking" also matches liquid staking (lst,
// lrt) and Ethena; "btc-l2" also matches Ordinals, Runes and BRC-20. A test
// fails when the
// dictionary gains a theme without a name here.
var narrativeThemeNames = map[string]string{
	"restaking":              "Restaking and liquid staking (LST/LRT)",
	"ai-tokens":              "AI tokens",
	"rwa":                    "Real-world assets (RWA)",
	"memecoins":              "Memecoins",
	"btc-l2":                 "Bitcoin L2s, Ordinals and Runes",
	"solana-defi":            "Solana DeFi",
	"modular":                "Modular blockchains",
	"zk":                     "Zero-knowledge (ZK) networks",
	"btc-etf":                "Bitcoin ETFs and ETF issuers",
	"bitcoin-halving":        "Bitcoin halving and mining",
	"stablecoins-regulation": "Stablecoins and regulation",
	"gaming-metaverse":       "Web3 gaming and metaverse",
	"depin":                  "Decentralized physical infrastructure (DePIN)",
	"oracles":                "Oracles and data feeds",
	"perp-dex":               "Perpetual DEXs",
	"cosmos-appchains":       "Cosmos and appchains",
}

// narrativeName is the reader's name of a theme id; an id the table does not
// know shows as itself.
func narrativeName(id string) string {
	if n, ok := narrativeThemeNames[id]; ok {
		return n
	}
	return id
}

const (
	narrativeFooter          = "matched items: last 24h · growth vs previous 24h"
	narrativeLineOrderBelow  = "Order: by activity score, not by count · scores are not shown below the threshold"
	narrativeLineOrderScored = "Top themes by activity score (0-100), not by count"
	narrativeLineSources     = "Source status (which feeds answered this cycle) is not served by the backend"
	narrativeLineMatch       = "Matched item: a theme keyword in an RSS headline/summary or a Reddit title/author line; precision unmeasured"
	narrativeRegimeQuiet     = "Local news regime: quiet, no theme scored; no price direction"
	narrativeLimitations     = "Keyword matches, precision not measured; source status not served; tone is not direction"
	narrativeBlockSource     = "CoinDesk and CoinTelegraph RSS plus Reddit when reachable, read by the AlphaVizor backend"
)

var (
	narrativeWhyBelow = fmt.Sprintf("No price level: %d matched items in 24h is the radar's threshold, checked on the leader only",
		newsMinMentions)
	narrativeScenariosBelow = []string{
		fmt.Sprintf("If the leading theme by activity score reaches %d matched items in 24h, the radar scores it", newsMinMentions),
		fmt.Sprintf("If the leader stays under %d matched items in 24h, the radar stays below threshold", newsMinMentions),
	}
	narrativeScenariosScored = []string{
		fmt.Sprintf("If the leading theme keeps %d+ matched items in 24h and the top score, the radar keeps scoring it", newsMinMentions),
		fmt.Sprintf("If it drops under %d matched items in 24h, the radar goes below threshold and hides scores", newsMinMentions),
	}
	narrativeInvalidates = fmt.Sprintf("Under %d matched items in 24h, or another theme taking a higher activity score", newsMinMentions)
)

// NarrativeReadout is the envelope's "narrative" object (additive
// 2026-09-15; docs/demobot-http.md "Narrative card").
type NarrativeReadout struct {
	State              string           `json:"state"`     // scored | below_threshold | no_matched_items | no_snapshots
	Window             string           `json:"window"`    // always "24h": the shown counts
	GrowthVs           string           `json:"growth_vs"` // always "previous_24h": used only for growth
	WindowEnd          *string          `json:"window_end"`
	CountKind          string           `json:"count_kind"` // always "matched_items"
	Threshold          int              `json:"threshold"`
	ThresholdCheckedOn string           `json:"threshold_checked_on"` // always "leader_by_activity_score"
	Order              string           `json:"order"`                // always "activity_score_desc"
	Leader             *NarrativeTheme  `json:"leader"`               // null on no_snapshots
	Themes             []NarrativeTheme `json:"themes"`               // the listed themes, served order; [] when none
	// EligibleNotLeader: below the threshold only — themes at ≥ threshold
	// that rank under the leader (the rule checks the leader only). [] else.
	EligibleNotLeader []string         `json:"eligible_not_leader"`
	ThemesServed      int              `json:"themes_served"`
	Sources           NarrativeSources `json:"sources"`
}

// NarrativeTheme is one theme at raw values. Score, data quality, stage and
// tone are null below the threshold, as on the card.
type NarrativeTheme struct {
	ID             string   `json:"id"`
	Name           string   `json:"name"`
	Matched24h     int      `json:"matched_24h"`
	MatchedPrev24h *int     `json:"matched_prev_24h"` // null when the payload carries none
	GrowthPct      *float64 `json:"growth_pct"`       // null when the previous 24h had 0 (or is missing)
	// NewTheme is computed here as the backend defines it (matched_prev_24h
	// == 0 && matched_24h > 0): the backend does not store is_new_theme, so
	// it always serves false. Null when the payload carries no prev count.
	NewTheme           *bool          `json:"new_theme"`
	ActivityScore      *int           `json:"activity_score"`
	DataQuality        *int           `json:"data_quality"`
	Stage              *string        `json:"stage"`
	Tone               *string        `json:"tone"`                 // positive | neutral | negative (headline keyword sentiment)
	SourcesWithMatches map[string]int `json:"sources_with_matches"` // null when not served
}

// NarrativeSources: the polled feeds; per-cycle status and coverage are not
// served by the backend, so both are always null.
type NarrativeSources struct {
	Expected []string `json:"expected"`
	Status   any      `json:"status"`
	Coverage any      `json:"coverage"`
}

// matchedItems: "1 matched item", "3 matched items".
func matchedItems(n int) string {
	if n == 1 {
		return "1 matched item"
	}
	return fmt.Sprintf("%d matched items", n)
}

// narrativeToneLine names what the semaphore color is: the keyword sentiment
// of the leader's matched items (RSS headline + description, Reddit title +
// body — news/rss.go, news/reddit.go), never a price direction.
func narrativeToneLine(tone string) string {
	return "Tone of the leader's matched items (keyword sentiment of their text), not a price direction: " + tone
}

// narrativeHeaderLine: the list headers are layout, not a reading — the
// showcase skips them when it picks the card's leading fact.
func narrativeHeaderLine(f string) bool {
	return f == narrativeLineOrderBelow || f == narrativeLineOrderScored
}

func narrativeTone(label string) string {
	switch label {
	case "bull":
		return "positive"
	case "bear":
		return "negative"
	case "neutral":
		return "neutral"
	}
	return ""
}

// narrativeFit cuts a single line at the budget.
func narrativeFit(s string) string {
	if r := []rune(s); len(r) > narrativeFactMaxRunes {
		return string(r[:narrativeFactMaxRunes-1]) + "…"
	}
	return s
}

// narrativeFirstFit returns the first candidate within the budget, else the
// last one cut.
func narrativeFirstFit(cands ...string) string {
	for _, c := range cands {
		if utf8.RuneCountInString(c) <= narrativeFactMaxRunes {
			return c
		}
	}
	return narrativeFit(cands[len(cands)-1])
}

func narrativeThemeOf(s NarrativeSnapshot, scored bool) NarrativeTheme {
	t := NarrativeTheme{ID: s.Narrative, Name: narrativeName(s.Narrative), Matched24h: s.MentionCount,
		SourcesWithMatches: s.SourcesBreakdown}
	if s.MentionCountPrev24h != nil {
		p := *s.MentionCountPrev24h
		t.MatchedPrev24h = &p
		nt := p == 0 && s.MentionCount > 0 // the backend's IsNewTheme predicate
		t.NewTheme = &nt
		if p > 0 && s.GrowthPct != nil {
			g := *s.GrowthPct
			t.GrowthPct = &g
		}
	}
	if scored {
		score, dq := s.TrendScore, s.Confidence
		t.ActivityScore, t.DataQuality = &score, &dq
		if s.Stage != "" {
			st := s.Stage
			t.Stage = &st
		}
		if tone := narrativeTone(s.SentimentLabel); tone != "" {
			t.Tone = &tone
		}
	}
	return t
}

// narrativePrev is " · previous 24h: N", or "" when the payload has no count.
func narrativePrev(s NarrativeSnapshot) string {
	if s.MentionCountPrev24h == nil {
		return ""
	}
	return fmt.Sprintf(" · previous 24h: %d", *s.MentionCountPrev24h)
}

// narrativeSourcesLine lists the leader's matched items by source, the
// largest first; "" when the payload has no breakdown.
func narrativeSourcesLine(s NarrativeSnapshot) string {
	if len(s.SourcesBreakdown) == 0 {
		return ""
	}
	keys := make([]string, 0, len(s.SourcesBreakdown))
	for k := range s.SourcesBreakdown {
		keys = append(keys, k)
	}
	name := func(k string) string {
		if n, ok := narrativeSourceNames[k]; ok {
			return n
		}
		return k
	}
	sort.Slice(keys, func(i, j int) bool {
		if s.SourcesBreakdown[keys[i]] != s.SourcesBreakdown[keys[j]] {
			return s.SourcesBreakdown[keys[i]] > s.SourcesBreakdown[keys[j]]
		}
		return name(keys[i]) < name(keys[j])
	})
	parts := make([]string, len(keys))
	for i, k := range keys {
		parts[i] = fmt.Sprintf("%s %d", name(k), s.SourcesBreakdown[k])
	}
	return narrativeFit("Leader's matched items by source: " + strings.Join(parts, " · "))
}

// newsCardFrom builds the card from one backend answer. Pure: every time on
// it comes from the payload, so the body is stable while the data is.
func newsCardFrom(n *NarrativesResp) Card {
	c := Card{
		Emoji:      emojiNeutral,
		Agent:      "Narrative Radar",
		ShortName:  "Narrative",
		Command:    keyNews,
		HowItWorks: howTexts[keyNews],
		DataTime:   parseWhen(n.CapturedAt),
		SourceNote: narrativeFooter,
		// The AI text is generated and cached by the backend per narrative
		// and hour, failures uncached — it can appear or change under the
		// same captured_at. No validator.
		noValidator: true,
	}
	ro := &NarrativeReadout{Window: "24h", GrowthVs: "previous_24h", CountKind: "matched_items",
		Threshold: newsMinMentions, ThresholdCheckedOn: "leader_by_activity_score", Order: "activity_score_desc",
		Themes: []NarrativeTheme{}, EligibleNotLeader: []string{}, ThemesServed: len(n.Narratives),
		Sources: NarrativeSources{Expected: append([]string{}, narrativeExpectedSources...)}}
	c.Narrative = ro
	window := "in the last 24h"
	if end, err := time.Parse(time.RFC3339, n.CapturedAt); err == nil {
		s := end.UTC().Format(time.RFC3339)
		ro.WindowEnd = &s
		window = "in the 24h to " + whaleClock(end)
	}

	if len(n.Narratives) == 0 {
		ro.State = narrStateNoSnapshots
		c.Verdict = "No radar snapshots from the backend yet — nothing to read"
		c.Short = "no data"
		c.Offline = true // "not enough data" is not a signal
		c.Status = statusBelowThreshold
		c.Facts = []string{narrativeLineSources}
		return c
	}

	top := n.Narratives[0]
	scored := top.MentionCount >= newsMinMentions
	listed := n.Narratives
	if len(listed) > narrativeListN {
		listed = listed[:narrativeListN]
	}
	for _, s := range listed {
		ro.Themes = append(ro.Themes, narrativeThemeOf(s, scored))
	}
	leader := narrativeThemeOf(top, scored)
	ro.Leader = &leader
	name, items := narrativeName(top.Narrative), matchedItems(top.MentionCount)

	if !scored {
		c.Status = statusBelowThreshold
		anyMatched := false
		for _, s := range n.Narratives {
			if s.MentionCount > 0 {
				anyMatched = true
				break
			}
		}
		if !anyMatched {
			ro.State = narrStateNoMatched
			c.Verdict = fmt.Sprintf("Below threshold: no theme has a matched item in the last 24h; %d needed to score", newsMinMentions)
			c.Short = "no matched items"
			c.Facts = []string{
				fmt.Sprintf("Themes served: %d · each has 0 matched items in the last 24h", len(n.Narratives)),
				narrativeLineSources, narrativeLineMatch,
			}
			c.Blocks = narrativeBlocksBelow(narrativeFit("No theme had a matched item " + window))
			return c
		}
		ro.State = narrStateBelow
		c.Verdict = narrativeFirstFit(
			fmt.Sprintf("Below threshold: %s leads by activity score with %s in 24h; %d needed to score", name, items, newsMinMentions),
			fmt.Sprintf("Below threshold: %s leads with %s/24h; %d needed", name, items, newsMinMentions))
		c.Short = "below threshold"
		// Names and matched items only, unnumbered: nothing below the
		// threshold is a ranking or a finding.
		c.Facts = append(c.Facts, narrativeLineOrderBelow)
		for _, s := range listed {
			c.Facts = append(c.Facts, narrativeFit(narrativeName(s.Narrative)+" — "+matchedItems(s.MentionCount)+narrativePrev(s)))
		}
		for _, s := range n.Narratives[1:] {
			if s.MentionCount >= newsMinMentions {
				ro.EligibleNotLeader = append(ro.EligibleNotLeader, s.Narrative)
			}
		}
		if len(ro.EligibleNotLeader) > 0 {
			var e NarrativeSnapshot
			for _, s := range n.Narratives[1:] {
				if s.Narrative == ro.EligibleNotLeader[0] {
					e = s
					break
				}
			}
			en, ei := narrativeName(e.Narrative), matchedItems(e.MentionCount)
			c.Facts = append(c.Facts, narrativeFirstFit(
				fmt.Sprintf("Threshold is checked on the leader only: %s has %s but a lower activity score", en, ei),
				fmt.Sprintf("Checked on the leader only: %s has %s, lower activity score", en, ei),
				fmt.Sprintf("Leader only is checked: %s has %s", en, ei)))
		}
		if l := narrativeSourcesLine(top); l != "" {
			c.Facts = append(c.Facts, l)
		}
		c.Facts = append(c.Facts, narrativeLineSources, narrativeLineMatch)
		c.Blocks = narrativeBlocksBelow(narrativeFirstFit(
			fmt.Sprintf("%s leads by activity score with %s %s", name, items, window),
			fmt.Sprintf("%s leads with %s %s", name, items, window),
			fmt.Sprintf("%s leads by activity score with %s in 24h", name, items),
			fmt.Sprintf("%s leads with %s in 24h", name, items)))
		// No data quality, no AI text: nothing below the threshold is a finding.
		return c
	}

	ro.State = narrStateScored
	// Semaphore rule unchanged from the pre-stage-1 card (the hook's
	// state_changed and the site's card color depend on it): the scored
	// leader's tone colors it, every other path stays neutral.
	switch top.SentimentLabel {
	case "bull":
		c.Emoji = emojiBull
	case "bear":
		c.Emoji = emojiBear
	default:
		c.Emoji = emojiNeutral
	}
	c.Verdict = narrativeFirstFit(
		fmt.Sprintf("Leading theme: %s · activity score %d/100 · %s in 24h", name, top.TrendScore, items),
		fmt.Sprintf("Leading theme: %s · activity score %d/100 · %s/24h", name, top.TrendScore, items),
		fmt.Sprintf("Leading theme: %s · activity score %d/100", name, top.TrendScore))
	c.Short = narrativeFirstFit(fmt.Sprintf("%s · activity %d/100", name, top.TrendScore))
	c.Facts = append(c.Facts, narrativeLineOrderScored)
	for i, s := range listed {
		sn, si := narrativeName(s.Narrative), matchedItems(s.MentionCount)
		c.Facts = append(c.Facts, narrativeFirstFit(
			fmt.Sprintf("%d. %s — activity score %d · %s%s", i+1, sn, s.TrendScore, si, narrativePrev(s)),
			fmt.Sprintf("%d. %s — activity score %d · %s", i+1, sn, s.TrendScore, si)))
	}
	if top.Stage != "" {
		c.Facts = append(c.Facts, narrativeFit("Stage label: "+top.Stage+" (backend rule on 24h count and growth vs previous 24h)"))
	}
	if tone := narrativeTone(top.SentimentLabel); tone != "" {
		c.Facts = append(c.Facts, narrativeToneLine(tone))
	}
	if l := narrativeSourcesLine(top); l != "" {
		c.Facts = append(c.Facts, l)
	}
	if top.Confidence > 0 {
		conf := top.Confidence
		c.Confidence = &conf
		c.confLabel = narrativeConfLabel
		c.Facts = append(c.Facts, fmt.Sprintf(
			"Data quality %d/100 is not a probability: it counts items, sources with matches, tone agreement", conf))
	}
	c.Facts = append(c.Facts, narrativeLineSources, narrativeLineMatch)
	// The AI text is the leader's own paragraph or nothing: the backend
	// attaches it to the top narrative only, and a version skew must never
	// print another theme's text under this verdict.
	if idea := strings.TrimSpace(top.GeneratedIdea); idea != "" {
		c.AIHTML = "<b>AI comment:</b> <i>" + esc(truncateAtSentence(idea, 500)) + "</i>"
	}

	what := []string{}
	if top.MentionCountPrev24h != nil {
		what = append(what, fmt.Sprintf("%s: %s %s, %d in the previous 24h", name, items, window, *top.MentionCountPrev24h))
	}
	what = append(what, fmt.Sprintf("%s: %s %s", name, items, window), fmt.Sprintf("%s: %s in 24h", name, items))
	inv := narrativeInvalidates
	c.Blocks = &ContentBlocks{
		WhatHappened: narrativeFirstFit(what...),
		WhyLevel: narrativeFit(fmt.Sprintf(
			"No price level: activity score %d/100 blends item growth, volume, tone, impact and source count", top.TrendScore)),
		Scenarios:   append([]string{}, narrativeScenariosScored...),
		Invalidates: &inv,
		Regime: narrativeFirstFit(
			fmt.Sprintf("Local news regime: one theme scored, %s; no price direction", name),
			"Local news regime: one theme scored; no price direction"),
		Limitations: narrativeLimitations,
		Source:      narrativeBlockSource,
	}
	return c
}

// narrativeBlocksBelow: below the threshold nothing is scored, so there is
// nothing to invalidate; the scenarios are the radar's own state changes.
func narrativeBlocksBelow(what string) *ContentBlocks {
	return &ContentBlocks{
		WhatHappened: what,
		WhyLevel:     narrativeWhyBelow,
		Scenarios:    append([]string{}, narrativeScenariosBelow...),
		Regime:       narrativeRegimeQuiet,
		Limitations:  narrativeLimitations,
		Source:       narrativeBlockSource,
	}
}

// narrativeDigestLine is the digest's 📖 context line (scored leader only).
func narrativeDigestLine(s NarrativeSnapshot) string {
	return fmt.Sprintf("📖 <b>Narrative</b>: %s (activity score %d, %s/24h)",
		esc(narrativeName(s.Narrative)), s.TrendScore, matchedItems(s.MentionCount))
}

// narrativeAILine is the digest AI payload's top_narrative value.
func narrativeAILine(s NarrativeSnapshot) string {
	return fmt.Sprintf("%s (stage %s, activity score %d, %s/24h)",
		narrativeName(s.Narrative), s.Stage, s.TrendScore, matchedItems(s.MentionCount))
}

// narrativeConclusion is the /showcase/example conclusion of a narrative
// card: news activity, never a direction. The generic sentence read the old
// sentiment-colored semaphore as "a bullish reading on the broader market".
func narrativeConclusion(c Card) string {
	ro := c.Narrative
	if ro == nil {
		return "The radar's source is offline right now, so there is nothing to read."
	}
	switch ro.State {
	case narrStateNoSnapshots:
		return "The radar has no snapshots yet, so there is nothing to read."
	case narrStateScored:
		if ro.Leader != nil {
			return fmt.Sprintf("This is a news-activity reading, not a forecast: %s has %s in 24h; "+
				"it says nothing about where price goes.", ro.Leader.Name, matchedItems(ro.Leader.Matched24h))
		}
	case narrStateNoMatched:
		return "This is a news-activity reading, not a forecast: no theme has a matched item in 24h, so none is scored."
	}
	have := 0
	if ro.Leader != nil {
		have = ro.Leader.Matched24h
	}
	return fmt.Sprintf("This is a news-activity reading, not a forecast: no theme is scored; "+
		"the leader has %d of the %d matched items needed in 24h.", have, ro.Threshold)
}
