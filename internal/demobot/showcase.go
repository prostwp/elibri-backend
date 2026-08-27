package demobot

// showcase.go — the landing's catalog source and its one worked example.
//
// Why this file exists (marketing review of the team landing, 2026-08): the
// page read the /agents listing as a menu of IDEAS — it surfaced one agent as
// a real-data demo and labelled the rest "planned/fictional", while twelve
// agents were serving live data the whole time. What the landing was missing
// is a source of truth that answers, per agent: did this produce a real
// reading just now, and what did it say. Plus one concrete "here is what you
// actually get" story instead of a feature list.
//
// Contract:
//   - GET /showcase — every agent that EXISTS, each with status live |
//     degraded. There is no "planned" state here and never will be: this
//     endpoint only lists agents whose builder actually ran. A degraded agent
//     STAYS in the list with its machine-readable reason, so the landing can
//     choose to show or hide it — the card layer's honesty rule, one level up.
//   - GET /showcase/example — one ready-to-render story in the narrative
//     marketing asked for: detected → explained → data → conclusion.
//   - ZERO duplicated agent logic. Both endpoints go through the exact
//     builders the bot and /agents/* dispatch to, over ONE gather sweep — the
//     same one /digest runs — and pick the story agent with the same
//     topSelection rule /top uses.
//   - One sweep per minute: the whole build is memoized for 60s with
//     singleflight (the aiMemo pattern), so ten landing renders cost one
//     sweep and /showcase/example rides the build /showcase just made. A
//     landing page can never fire twelve uncached upstream calls.

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

const (
	// showcaseTTL is the memo window for a sweep that produced at least one
	// real reading. showcaseDeadTTL is the shorter window for an all-dark
	// sweep: a landing that went "everything degraded" during a blip must
	// recover in seconds, not sit on a cached blackout for a full minute
	// (same reasoning as aiCacheTTL vs aiFailureTTL).
	showcaseTTL     = 60 * time.Second
	showcaseDeadTTL = 15 * time.Second

	// showcaseFactsMax caps the example's data block — the landing renders
	// three or four lines, not a whole card.
	showcaseFactsMax = 4

	showcaseLive     = "live"
	showcaseDegraded = "degraded"
)

// showcaseNames is the landing-facing name of each agent. For the ten
// card-backed agents it is the builder's own Agent string (pinned by
// TestShowcaseNamesMatchBuilders, so the table cannot drift); digest and top
// are aggregates whose card belongs to the winning agent, so they carry their
// own identity.
var showcaseNames = map[string]string{
	keyDigest:   digestAgentName,
	keyTop:      "Top Signal",
	keyFX:       "FX Agent",
	keyMacro:    "Macro Agent",
	keyWhale:    "Whale Flow Agent",
	keyFunding:  "Funding Agent",
	keyMomentum: "Momentum Agent",
	keyTrend:    "Trend Agent",
	keySR:       "S/R Agent",
	keyVol:      "Volatility Agent",
	keyRisk:     "Risk Calculator",
	keyNews:     "Narrative Radar",
	keyGold:     "Gold Agent",
}

// showcaseCategories is the landing's filter chip per agent. "tools" holds
// the three that are not a single-market read: the two aggregators and the
// calculator. Every other agent sits in the market its data comes from.
var showcaseCategories = map[string]string{
	keyDigest:   "tools",
	keyTop:      "tools",
	keyRisk:     "tools",
	keyFX:       "forex",
	keyMacro:    "macro",
	keyWhale:    "onchain",
	keyFunding:  "derivatives",
	keyNews:     "news",
	keyMomentum: "crypto", // default read is BTC/ETH/XAUUSD; ?asset= widens it
	keyTrend:    "crypto", // default asset btc
	keySR:       "crypto",
	keyVol:      "crypto",
	keyGold:     "metals",
}

// exampleOrder is the fallback preference when the /top winner is degraded:
// the priority trio first (reusing signalOrder, so the funding > momentum >
// trend tie-break can never drift from priority.go), then the remaining
// single-agent reads. digest/top are excluded — an aggregate is not one
// agent's story — and so is risk, which is arithmetic on user numbers, not a
// market observation with a "detected" moment.
var exampleOrder = append(append([]string{}, signalOrder...),
	keyMacro, keyWhale, keySR, keyVol, keyFX, keyNews)

// ── one memoized sweep ───────────────────────────────────────────────────────

// showcaseBuild is one sweep behind both landing endpoints: the gather the
// digest runs, plus the cards gather does not produce (narrative radar) or
// does not assemble (the fx overview, built from gather's own reads), plus
// the two aggregate rows derived from the sweep.
type showcaseBuild struct {
	g     gathered
	cards map[string]Card // slug → the card that agent serves right now
	at    time.Time       // sweep time, served as generated_at
}

// buildShowcase runs every agent's real builder concurrently over one sweep.
// Upstream cost vs a plain /digest: exactly one extra GET (the narrative
// radar, which gather reads only as a single top line) — fx reuses gather's
// reads, risk is local arithmetic, digest and top are derived.
func (a *Agents) buildShowcase(ctx context.Context) *showcaseBuild {
	var g gathered
	var news, gold Card
	var wg sync.WaitGroup
	wg.Add(3)
	go func() { defer wg.Done(); g = a.gather(ctx) }()
	go func() { defer wg.Done(); news = a.NewsCard(ctx) }()
	// The gold agent is not part of gather (the digest trio is crypto-only,
	// see priority.go), so the landing sweeps it alongside the radar.
	go func() { defer wg.Done(); gold = a.GoldCard(ctx) }()
	wg.Wait()

	cards := make(map[string]Card, len(httpAgentNames))
	for k, c := range g.cards { // macro, whale, funding, momentum, trend, sr, vol
		cards[k] = c
	}
	cards[keyFX] = fxCardFromReads(g.fx)
	cards[keyNews] = news
	cards[keyGold] = gold
	cards[keyRisk] = a.RiskCard(riskExampleValues, true, nil)
	_, top := topSelection(g)
	cards[keyDigest] = top // the digest's head IS the winner card
	cards[keyTop] = top
	return &showcaseBuild{g: g, cards: cards, at: time.Now().UTC()}
}

// showcaseEntry / showcaseMemo mirror aiMemo: a fresh result is served from
// cache, concurrent callers JOIN the in-flight sweep instead of starting
// their own, and only the starter runs the build.
type showcaseEntry struct {
	done  chan struct{}
	build *showcaseBuild
	live  bool // the sweep produced at least one real reading
	at    time.Time
}

func (e *showcaseEntry) fresh(now time.Time) bool {
	ttl := showcaseTTL
	if !e.live {
		ttl = showcaseDeadTTL
	}
	return now.Sub(e.at) < ttl
}

type showcaseMemo struct {
	mu  sync.Mutex
	cur *showcaseEntry
}

// do returns the fresh cached sweep, joins an in-flight one, or runs gen
// exactly once and stores the result.
func (m *showcaseMemo) do(gen func() *showcaseBuild) *showcaseBuild {
	m.mu.Lock()
	if e := m.cur; e != nil {
		select {
		case <-e.done: // finished — serve if still fresh, else rebuild below
			if e.fresh(time.Now()) {
				m.mu.Unlock()
				return e.build
			}
		default: // in flight — join it outside the lock
			m.mu.Unlock()
			<-e.done
			return e.build
		}
	}
	e := &showcaseEntry{done: make(chan struct{})}
	m.cur = e
	m.mu.Unlock()

	e.build = gen()
	e.live = e.build.marketLive()
	e.at = time.Now()
	close(e.done)
	return e.build
}

// showcase serves the memoized sweep. The build runs on a context detached
// from the request: it is shared by every joiner, so a client that
// disconnects mid-sweep must not poison the memo with offline cards for the
// rest of the TTL. gather keeps its own digestBudget on top.
func (s *HTTPServer) showcase(ctx context.Context) *showcaseBuild {
	detached := context.WithoutCancel(ctx)
	return s.sc.do(func() *showcaseBuild { return s.ag.buildShowcase(detached) })
}

// ── GET /showcase ────────────────────────────────────────────────────────────

// showcaseAgent is one catalog row. Status is live | degraded — NEVER
// "planned": every row here is an agent whose builder just ran.
type showcaseAgent struct {
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

type showcaseResp struct {
	GeneratedAt string          `json:"generated_at"`
	LiveCount   int             `json:"live_count"`
	TotalCount  int             `json:"total_count"`
	Agents      []showcaseAgent `json:"agents"`
}

// status is the machine state of one showcase row. Ten agents answer with
// their card's own status; the digest is special-cased because it aggregates
// (it renders whatever is alive and labels the rest offline, so it is only
// degraded when the WHOLE sweep is dark).
func (b *showcaseBuild) status(slug string) cardStatus {
	if slug == keyDigest {
		firstDegraded := statusSourceOffline
		found := false
		for _, k := range digestOrder {
			c, ok := b.g.cards[k]
			if !ok {
				continue
			}
			st := c.effectiveStatus()
			if st == statusOK {
				return statusOK
			}
			if !found {
				firstDegraded, found = st, true
			}
		}
		return firstDegraded
	}
	return b.cards[slug].effectiveStatus()
}

// row renders one catalog entry from the card its agent actually produced.
func (b *showcaseBuild) row(slug string) showcaseAgent {
	c := b.cards[slug]
	st := b.status(slug)
	row := showcaseAgent{
		Slug:       slug,
		Name:       showcaseNames[slug],
		Category:   showcaseCategories[slug],
		Status:     showcaseLive,
		OK:         st == statusOK,
		Headline:   c.Verdict,
		OneLiner:   showcaseOneLiner(showcaseNames[slug], c),
		DataAsOf:   c.DataTime.UTC().Format(time.RFC3339),
		ExampleURL: "/agents/" + slug,
	}
	if slug == keyDigest {
		// The digest composes at request time and speaks for the sweep, not
		// for one card — same wording and same stamp /agents/digest serves.
		row.Headline = digestHeadline(c)
		row.DataAsOf = b.at.Format(time.RFC3339)
	}
	if !row.OK {
		row.Status = showcaseDegraded
		r := st.reason()
		row.Reason = &r
	}
	return row
}

// rows renders the catalog in the /agents listing order (the bot's menu grid).
func (b *showcaseBuild) rows() []showcaseAgent {
	out := make([]showcaseAgent, 0, len(httpAgentNames))
	for _, slug := range httpAgentNames {
		out = append(out, b.row(slug))
	}
	return out
}

func (b *showcaseBuild) liveCount() int {
	n := 0
	for _, slug := range httpAgentNames {
		if b.status(slug) == statusOK {
			n++
		}
	}
	return n
}

// marketLive reports whether any agent BACKED BY AN UPSTREAM produced a real
// reading. The risk calculator is local arithmetic — it is live even in a
// total blackout, so counting it here would mask one and hold a dark sweep
// for the full TTL. This is the flag that picks showcaseTTL vs
// showcaseDeadTTL, nothing else.
func (b *showcaseBuild) marketLive() bool {
	for _, slug := range httpAgentNames {
		if slug == keyRisk {
			continue
		}
		if b.status(slug) == statusOK {
			return true
		}
	}
	return false
}

// showcaseOneLiner is the digest-style one-liner, with a fallback for agents
// that never appear in a digest and therefore carry no Short (the risk
// calculator): the verdict stands in, so the landing can never render a
// dangling "⚪ Risk: ".
func showcaseOneLiner(name string, c Card) string {
	if strings.TrimSpace(c.Short) != "" {
		return htmlToPlain(c.OneLiner())
	}
	line := c.Emoji + " " + name
	if c.Asset != "" {
		line += " " + c.Asset
	}
	return strings.TrimSpace(line + ": " + c.Verdict)
}

func (s *HTTPServer) handleShowcase(w http.ResponseWriter, r *http.Request) {
	b := s.showcase(r.Context())
	rows := b.rows()
	writeJSON(w, http.StatusOK, showcaseResp{
		// The sweep time, not the request time: with the 60s memo a landing
		// render can legitimately serve a payload up to a minute old, and
		// saying so is the honest version of a cache.
		GeneratedAt: b.at.Format(time.RFC3339),
		LiveCount:   b.liveCount(), // same b.status() source as the rows
		TotalCount:  len(rows),
		Agents:      rows,
	})
}

// ── GET /showcase/example ────────────────────────────────────────────────────

// showcaseExampleResp is one ready-to-render story: what the agent noticed,
// why it matters, the live numbers behind it, and what it means — the exact
// narrative the landing asked for, in analytical language only.
type showcaseExampleResp struct {
	GeneratedAt string   `json:"generated_at"`
	Agent       string   `json:"agent"`
	Slug        string   `json:"slug"`
	Asset       string   `json:"asset"`
	Detected    string   `json:"detected"`
	Explained   string   `json:"explained"`
	Data        []string `json:"data"`
	Conclusion  string   `json:"conclusion"`
	Levels      any      `json:"levels,omitempty"`
	ExampleURL  string   `json:"example_url"`
	Disclaimer  string   `json:"disclaimer"`
	DataAsOf    string   `json:"data_as_of"`
}

// exampleCard picks the story agent: the /top winner (same deterministic
// priority rule, via topSelection — never a second copy of it) when it
// produced a real reading, otherwise the strongest ok agent in exampleOrder.
// "Strongest" is the same deviation-from-neutral the priority rule ranks on;
// strict > while scanning keeps the earlier agent on ties, so the pick is
// deterministic. ok=false means every candidate is degraded → the caller
// serves an honest 503 rather than a story built on a dead source.
func (b *showcaseBuild) exampleCard() (slug string, card Card, ok bool) {
	winner, top := topSelection(b.g)
	if top.effectiveStatus() == statusOK {
		return winner, top, true
	}
	best, bestDev := "", -1
	for _, k := range exampleOrder {
		c, exists := b.cards[k]
		if !exists || c.effectiveStatus() != statusOK {
			continue
		}
		if c.Deviation > bestDev {
			best, bestDev = k, c.Deviation
		}
	}
	if best == "" {
		return "", Card{}, false
	}
	return best, b.cards[best], true
}

// detectedSentence is what the agent noticed, derived from the card's own
// verdict — never a re-interpretation of it.
func detectedSentence(c Card) string {
	s := c.Agent
	if c.Asset != "" {
		s += " on " + c.Asset
	}
	return endSentence(s + " — " + c.Verdict)
}

// strongestFact is the card's leading fact — the builders order facts
// most-important-first. The weekend banner is skipped: it is context about
// the clock, not the reading it sits above.
func strongestFact(c Card) string {
	for _, f := range c.Facts {
		if f == fxClosedBanner {
			continue
		}
		if strings.TrimSpace(f) != "" {
			return endSentence(f)
		}
	}
	if c.HowItWorks != "" { // factless card: explain the method, never invent a fact
		return c.HowItWorks
	}
	return endSentence(c.Verdict)
}

// exampleFacts is the data block: the card's own live fact lines, capped.
// When a card carries fewer than three, the top-up comes only from values it
// already computed (confidence, source note) — nothing is invented to reach
// a nicer-looking three.
func exampleFacts(c Card) []string {
	out := []string{}
	for _, f := range c.Facts {
		if len(out) == showcaseFactsMax {
			break
		}
		if strings.TrimSpace(f) != "" {
			out = append(out, f)
		}
	}
	if len(out) < 3 && c.Confidence != nil {
		out = append(out, fmt.Sprintf("Confidence: %d%%", clampInt(*c.Confidence, 0, 100)))
	}
	if len(out) < 3 && c.SourceNote != "" {
		out = append(out, "Source: "+c.SourceNote)
	}
	if len(out) == 0 {
		out = append(out, endSentence(c.Verdict))
	}
	return out
}

// conclusionFor words what the reading means for a trader. Analytical
// language only — a tilt that holds while its inputs hold, never an
// instruction to enter or exit (the same rule sanitizeAdviceLanguage
// enforces on model output).
func conclusionFor(c Card) string {
	subject := c.Asset
	if subject == "" {
		subject = "the broader market"
	}
	var s string
	switch semaphoreOf(c.Emoji) {
	case "bullish":
		s = fmt.Sprintf("For a trader this is a bullish reading on %s: the numbers above lean up, and the read holds only for as long as they do.", subject)
	case "bearish":
		s = fmt.Sprintf("For a trader this is a bearish reading on %s: the numbers above lean down, and the read holds only for as long as they do.", subject)
	default:
		s = fmt.Sprintf("For a trader this is a neutral reading on %s: nothing in the numbers above leans either way, so the level structure matters more than direction right now.", subject)
	}
	// A trend card already carries the price at which its own structure
	// breaks — the honest edge of the statement, so it belongs in it.
	switch tl := c.Levels.(type) {
	case TrendLevels:
		s += fmt.Sprintf(" The structure this read describes breaks on a close %s %s.", tl.InvalidationSide, trimFloat(tl.Invalidation))
	case *TrendLevels:
		if tl != nil {
			s += fmt.Sprintf(" The structure this read describes breaks on a close %s %s.", tl.InvalidationSide, trimFloat(tl.Invalidation))
		}
	}
	return s
}

// endSentence closes a fragment with a period unless it already ends in
// terminal punctuation.
func endSentence(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return s
	}
	if strings.HasSuffix(s, ".") || strings.HasSuffix(s, "!") || strings.HasSuffix(s, "?") {
		return s
	}
	return s + "."
}

func (s *HTTPServer) handleShowcaseExample(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	b := s.showcase(ctx)
	winner, topCard := topSelection(b.g)
	slug, card, ok := b.exampleCard()
	if !ok {
		// Nothing real to show. The story is the one thing we refuse to fake,
		// so this is the standard honest 503 with the machine reason of the
		// agent that would have told it.
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{
			"error":  "no agent produced a real reading right now — every data source is degraded",
			"ok":     false,
			"reason": topCard.effectiveStatus().reason(),
		})
		return
	}

	explained := ""
	if slug == winner {
		// The SAME aiMemo key /top uses ("why|<winner>"): inside the 5-minute
		// window this costs nothing, and it never opens a new prompt kind.
		// On the fallback path there is deliberately no AI call — a new key
		// would be new LLM spend for a decoration.
		explained = sanitizeAdviceLanguage(s.ag.aiTopWhy(ctx, winner, b.g))
	}
	if strings.TrimSpace(explained) == "" {
		explained = strongestFact(card)
	}

	writeJSON(w, http.StatusOK, showcaseExampleResp{
		GeneratedAt: b.at.Format(time.RFC3339),
		Agent:       card.Agent,
		Slug:        slug,
		Asset:       card.Asset,
		Detected:    detectedSentence(card),
		Explained:   explained,
		Data:        exampleFacts(card),
		Conclusion:  conclusionFor(card),
		Levels:      card.Levels,
		ExampleURL:  "/agents/" + slug,
		Disclaimer:  disclaimerText,
		DataAsOf:    card.DataTime.UTC().Format(time.RFC3339),
	})
}
