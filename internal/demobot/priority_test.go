package demobot

// priority_test.go — the digest ranking rule (priority.go): eligibility
// (status, freshness), the confirmed tier, tie orders, the macro risk-off
// gate, the no_highlight state, the unified digest status and JSON == HTML.
// Edge cases from Digest_план.md stage 1, item 8.

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// rankAt is the sweep time of every hand-built sweep below; cards stamped
// rankFresh are one hour old (fresh for every source except funding, which
// is stamped rankAt itself).
var (
	rankAt    = time.Date(2026, 9, 15, 12, 0, 0, 0, time.UTC)
	rankFresh = rankAt.Add(-time.Hour)
)

// fundingCard is a REAL funding card in the 2026-09-15 format
// (fundingCardFrom): BTCUSDT at rate, the other majors at 0, an empty
// liquidation window, stamped rankAt. BTC is the shown coin whatever the
// rate (the others score 0, ties go to BTC), so Deviation is
// fundingDeviation(rate).
func fundingCard(rate float64) Card {
	q := map[string]fundingQuote{}
	for _, s := range fundingSymbols {
		q[s] = fundingQuote{}
	}
	q["BTCUSDT"] = fundingQuote{rate: rate}
	return fundingCardFrom(q, nil, &FundingResp{}, nil, rankAt)
}

// momentumCard is a composite momentum card in the 2026-09-15 format: the
// verdict (and one-liner) is the counter, coloured by the BTC/ETH reads.
func momentumCard(dev int, confirmed bool) Card {
	verdict, emoji := "0 bullish · 0 bearish · 3 not confirmed", emojiNeutral
	if confirmed {
		verdict, emoji = "1 bullish (BTC) · 0 bearish · 2 not confirmed", emojiBull
	}
	return Card{Agent: "Momentum Agent", ShortName: "Momentum", Asset: "BTC/ETH/XAUUSD", Command: keyMomentum,
		Verdict: verdict, Short: verdict, Emoji: emoji,
		DataTime: rankFresh, Deviation: dev, confirmed: confirmed}
}

func trendCard(state string, adx int) Card {
	c := Card{Agent: "Trend Agent", ShortName: "Trend", Asset: "BTC", Command: keyTrend,
		Verdict: "Flat · no trend to read", Short: "flat — no trend", Emoji: emojiNeutral,
		State: state, DataTime: rankFresh, Deviation: adx}
	if state == trendUp || state == trendDown {
		c.confirmed, c.Deviation, c.Verdict = true, clampInt(adx*2, 0, 100), "Confirmed UPTREND"
	}
	return c
}

func macroOK(regime string) Card {
	return Card{Agent: "Macro Agent", ShortName: "Macro", Command: keyMacro,
		Verdict: "RISK-ON", Short: "risk-on", Emoji: emojiBull, DataTime: rankFresh}
}

func sweep(regime string, cards map[string]Card) gathered {
	if _, ok := cards[keyMacro]; !ok {
		cards[keyMacro] = macroOK(regime)
	}
	return gathered{cards: cards, regime: regime, at: rankAt}
}

// ── the three verified ranking defects ───────────────────────────────────────

// Neutral 0/0/17: funding exactly balanced, momentum neutral (RSI far from 50
// but MACD disagreeing), trend flat with ADX 17. Live 2026-09-15 this made
// "Top signal: Trend Agent · BTC — Flat · no trend to read". Nothing is
// confirmed → no_highlight; the flat trend does not win on its raw ADX, and
// the balanced funding does not win the tie.
func TestRankNeutralZeroZeroSeventeen(t *testing.T) {
	g := sweep("risk_on", map[string]Card{
		keyFunding:  fundingCard(0),
		keyMomentum: momentumCard(0, false),
		keyTrend:    trendCard(trendFlat, 17),
	})
	p := g.selection()
	if !p.NoHighlight || p.Rule != ruleUnconfirmed {
		t.Fatalf("all-neutral sweep: rule %q no_highlight=%v, want %q/true", p.Rule, p.NoHighlight, ruleUnconfirmed)
	}
	if p.Winner == keyTrend {
		t.Error("a flat trend must not win on raw ADX")
	}
	if p.Winner == keyFunding {
		t.Error("a balanced funding must not win a tie")
	}
	if p.Winner != keyMomentum {
		t.Errorf("tier-2 tie order momentum > trend > funding: got %q", p.Winner)
	}
	for _, c := range p.Candidates {
		if c.Key == keyTrend && c.Score != 0 {
			t.Errorf("unconfirmed trend ranks with score %d, want 0", c.Score)
		}
	}
	// Display unchanged while the flag is off: the headline stays "Top signal".
	_, top := topSelection(g)
	if h := digestHeadlineFor(p, top); !strings.HasPrefix(h, "Top signal: ") {
		t.Errorf("headline must stay as is until the product decision: %q", h)
	}
	if l := selectionLine(p); !strings.HasPrefix(l, "No confirmed reading among") {
		t.Errorf("selection line must say nothing was confirmed: %q", l)
	}
}

// A neutral momentum (verdict neutral, RSI 80 with a falling MACD) scores 0,
// so a confirmed trend and even a balanced funding with a rate beat it.
func TestMomentumRankScoreOnlyWhenConfirmed(t *testing.T) {
	cases := []struct {
		r    momentumRead
		want int
	}{
		{momentumRead{verdict: "neutral", rsi: 80, hist: -5}, 0}, // RSI/MACD conflict
		{momentumRead{verdict: "neutral", rsi: 52, hist: 5}, 0},  // RSI in the dead band
		{momentumRead{verdict: momentumVerdict(60, 1), rsi: 60, hist: 1}, 20},
		{momentumRead{verdict: momentumVerdict(30, -1), rsi: 30, hist: -1}, 40},
		{momentumRead{verdict: momentumVerdict(55, 0.1), rsi: 55, hist: 0.1}, 10}, // on the threshold
	}
	for _, tc := range cases {
		if got := momentumRankScore(tc.r); got != tc.want {
			t.Errorf("momentumRankScore(%s RSI %.0f hist %+g) = %d, want %d", tc.r.verdict, tc.r.rsi, tc.r.hist, got, tc.want)
		}
	}
}

// Funding is scored against the threshold of its own side, so both
// thresholds score the same 30, and the positive side keeps its old values.
func TestFundingDeviationOwnSideThreshold(t *testing.T) {
	cases := []struct {
		rate float64
		want int
	}{
		{0, 0},
		{0.00029, 29},  // +0.029% balanced — unchanged from the old scale
		{0.0003, 30},   // longs threshold
		{0.0010, 100},  // old full scale
		{-0.0001, 30},  // shorts threshold — was 10 on the symmetric scale
		{-0.00005, 15}, // halfway to the shorts threshold
		{-0.0005, 100}, // clamped
	}
	for _, tc := range cases {
		if got := fundingDeviation(tc.rate); got != tc.want {
			t.Errorf("fundingDeviation(%+.5f) = %d, want %d", tc.rate, got, tc.want)
		}
	}
	// −0.010% "shorts crowded" (confirmed) outranks +0.029% "balanced".
	crowded, balanced := fundingCard(-0.0001), fundingCard(0.00029)
	if !crowded.confirmed || balanced.confirmed {
		t.Fatal("fixture: −0.010% must be crowded, +0.029% balanced")
	}
	g := sweep("risk_on", map[string]Card{keyFunding: crowded, keyMomentum: momentumCard(0, false), keyTrend: trendCard(trendFlat, 29)})
	if p := g.selection(); p.Winner != keyFunding || p.Rule != ruleConfirmed {
		t.Errorf("crowded shorts: got %q/%q, want funding/%s", p.Winner, p.Rule, ruleConfirmed)
	}
}

// An unconfirmed trend never outranks a confirmed reading, whatever its ADX.
func TestUnconfirmedTrendNeverOutranksConfirmed(t *testing.T) {
	for _, st := range []string{trendFlat, trendGrey, trendConflict} {
		g := sweep("risk_on", map[string]Card{
			keyFunding:  fundingCard(0.0001),
			keyMomentum: momentumCard(10, true), // the weakest confirmed momentum
			keyTrend:    trendCard(st, 49),
		})
		if p := g.selection(); p.Winner != keyMomentum {
			t.Errorf("%s trend ADX 49 vs confirmed momentum 10: winner %q, want momentum", st, p.Winner)
		}
	}
}

// ── ties ─────────────────────────────────────────────────────────────────────

func TestRankTieCases(t *testing.T) {
	cases := []struct {
		name  string
		cards map[string]Card
		want  string
	}{
		{"confirmed three-way tie → funding first", map[string]Card{
			keyFunding: fundingCard(0.0003), keyMomentum: momentumCard(30, true), keyTrend: trendCard(trendUp, 15)}, keyFunding},
		{"confirmed momentum = trend → momentum", map[string]Card{
			keyFunding: fundingCard(0.0001), keyMomentum: momentumCard(50, true), keyTrend: trendCard(trendUp, 25)}, keyMomentum},
		{"balanced funding 0 vs neutral momentum 0 → momentum", map[string]Card{
			keyFunding: fundingCard(0), keyMomentum: momentumCard(0, false)}, keyMomentum},
		{"balanced funding 0 vs flat trend → trend", map[string]Card{
			keyFunding: fundingCard(0), keyTrend: trendCard(trendFlat, 40)}, keyTrend},
		{"balanced funding with a real score wins tier 2 on score", map[string]Card{
			keyFunding: fundingCard(0.0001), keyMomentum: momentumCard(0, false), keyTrend: trendCard(trendFlat, 40)}, keyFunding},
	}
	for _, tc := range cases {
		if got := sweep("mixed", tc.cards).selection().Winner; got != tc.want {
			t.Errorf("%s: got %q, want %q", tc.name, got, tc.want)
		}
	}
}

// ── eligibility: status and freshness ────────────────────────────────────────

func TestTopMaxAgeConstants(t *testing.T) {
	if topMaxAge(keyFunding) != 15*time.Minute {
		t.Errorf("funding max age %s", topMaxAge(keyFunding))
	}
	for _, k := range []string{keyMomentum, keyTrend} {
		if topMaxAge(k) != 8*time.Hour {
			t.Errorf("%s max age %s, want 8h (two 4h bars)", k, topMaxAge(k))
		}
	}
	if barMaxAge("1h") != 2*time.Hour {
		t.Errorf("1h bars: %s", barMaxAge("1h"))
	}
	// Boundary: exactly the limit is fresh, one second more is stale.
	c := trendCard(trendUp, 40)
	c.DataTime = rankAt.Add(-8 * time.Hour)
	if cand := rankCandidate(keyTrend, c, rankAt); !cand.Eligible {
		t.Errorf("age == limit must be eligible: %+v", cand)
	}
	c.DataTime = c.DataTime.Add(-time.Second)
	if cand := rankCandidate(keyTrend, c, rankAt); cand.Eligible || cand.Excluded != excludedStale {
		t.Errorf("age > limit must be stale: %+v", cand)
	}
}

// A stale winner: the strongest confirmed reading is 20h old (a 4h read that
// missed four bars) — it is excluded and the fresh confirmed reading wins.
func TestStaleWinnerExcluded(t *testing.T) {
	stale := trendCard(trendUp, 45) // score 90
	stale.DataTime = rankAt.Add(-20 * time.Hour)
	g := sweep("risk_on", map[string]Card{
		keyFunding: fundingCard(0.0003), keyMomentum: momentumCard(0, false), keyTrend: stale,
	})
	p := g.selection()
	if p.Winner != keyFunding {
		t.Fatalf("stale trend must not take the slot: got %q", p.Winner)
	}
	for _, c := range p.Candidates {
		if c.Key == keyTrend && (c.Eligible || c.Excluded != excludedStale) {
			t.Errorf("trend candidate: %+v, want excluded stale", c)
		}
	}
	// A funding card replayed 20 minutes later is stale too.
	old := fundingCard(0.0005)
	old.DataTime = rankAt.Add(-20 * time.Minute)
	if cand := rankCandidate(keyFunding, old, rankAt); cand.Excluded != excludedStale {
		t.Errorf("20-minute-old funding: %+v", cand)
	}
	// A live card without a data time has an unknown age: never trusted.
	blank := momentumCard(40, true)
	blank.DataTime = time.Time{}
	if cand := rankCandidate(keyMomentum, blank, rankAt); cand.Excluded != excludedNoDataTime {
		t.Errorf("no data time: %+v", cand)
	}
}

// Momentum's freshness is its ranked (Binance) bar, not the gold bar that can
// hold the card's DataTime back over a weekend.
func TestMomentumFreshnessUsesRankedBars(t *testing.T) {
	c := momentumCard(30, true)
	c.DataTime = rankAt.Add(-60 * time.Hour) // gold, Friday close
	c.rankAsOf = rankFresh
	if cand := rankCandidate(keyMomentum, c, rankAt); !cand.Eligible {
		t.Errorf("fresh crypto reads behind a stale gold bar must stay eligible: %+v", cand)
	}
}

// Degraded cards never compete (a funding card with its rate source down
// renders liquidation facts but produced no reading).
func TestDegradedCandidateExcluded(t *testing.T) {
	off := fundingCard(0.0009)
	off.Status = statusSourceOffline
	cand := rankCandidate(keyFunding, off, rankAt)
	if cand.Eligible || cand.Excluded != excludedDegraded || cand.Score != 0 || cand.Confirmed {
		t.Errorf("degraded funding: %+v", cand)
	}
}

// ── all three offline while the rest is live ─────────────────────────────────

func TestTrioOfflineOthersLive(t *testing.T) {
	g := sweep("risk_on", map[string]Card{
		keyFunding:  offlineCard("Funding Agent", "Funding", "", keyFunding, ""),
		keyMomentum: offlineCard("Momentum Agent", "Momentum", "BTC/ETH/XAUUSD", keyMomentum, ""),
		keyTrend:    offlineCard("Trend Agent", "Trend", "BTC", keyTrend, ""),
		keyWhale:    {Agent: "Whale Flow Agent", ShortName: "Whale", Asset: "BTC", Short: "outflow", Emoji: emojiBull, DataTime: rankFresh},
	})
	p := g.selection()
	if p.Winner != keyMacro || p.Rule != ruleFallbackMacro || !p.NoHighlight {
		t.Fatalf("trio offline: %q/%q no_highlight=%v, want macro/%s/true", p.Winner, p.Rule, p.NoHighlight, ruleFallbackMacro)
	}
	h := g.health()
	if h.Status != digestPartial {
		t.Errorf("health %q, want partial (macro and whale live)", h.Status)
	}
	env := digestEnvelope(g, "")
	// Top-level ok/reason = the highlighted (live macro) card; health only in digest.*.
	if !env.OK || env.Reason != nil || env.Digest.Status != digestPartial {
		t.Errorf("digest envelope: ok=%v reason=%v status=%q, want true/null/partial", env.OK, env.Reason, env.Digest.Status)
	}
	b := &showcaseBuild{g: g, cards: g.cards, at: rankAt}
	if b.status(keyDigest) != statusOK || b.row(keyDigest).DigestStatus != digestPartial {
		t.Errorf("showcase digest row disagrees: %+v", b.row(keyDigest))
	}
}

// Top-level ok/reason are the HIGHLIGHTED card's, exactly as before the
// digest readout existed (clients gate verdict/semaphore on them): a degraded
// fallback winner with live sections is ok:false with the card's reason,
// while the sweep's health (partial here) lives only in digest.status.
func TestDigestTopLevelOKIsHighlightedCard(t *testing.T) {
	unknown := Card{Agent: "Macro Agent", ShortName: "Macro", Command: keyMacro, Verdict: "UNKNOWN",
		Short: "unknown (no data)", Emoji: emojiNeutral, Status: statusNoData, DataTime: rankFresh}
	g := sweep("unknown", map[string]Card{
		keyMacro:    unknown,
		keyFunding:  offlineCard("Funding Agent", "Funding", "", keyFunding, ""),
		keyMomentum: offlineCard("Momentum Agent", "Momentum", "BTC/ETH/XAUUSD", keyMomentum, ""),
		keyTrend:    offlineCard("Trend Agent", "Trend", "BTC", keyTrend, ""),
		keyWhale:    {Agent: "Whale Flow Agent", ShortName: "Whale", Asset: "BTC", Short: "outflow", Emoji: emojiBull, DataTime: rankFresh},
	})
	env := digestEnvelope(g, "")
	if env.OK || env.Reason == nil || *env.Reason != "no_data" {
		t.Fatalf("top-level: ok=%v reason=%v, want false/no_data (the macro card's own status)", env.OK, env.Reason)
	}
	want := cardEnvelope(g.cards[keyMacro])
	if env.OK != want.OK || *env.Reason != *want.Reason {
		t.Errorf("top-level pair differs from the highlighted card's envelope: %v/%v vs %v/%v", env.OK, *env.Reason, want.OK, *want.Reason)
	}
	sel := env.Digest.Selection
	if sel.HighlightOK != env.OK || sel.HighlightReason == nil || *sel.HighlightReason != *env.Reason {
		t.Errorf("highlight_ok/_reason must repeat the top-level pair: %v/%v", sel.HighlightOK, sel.HighlightReason)
	}
	if env.Digest.Status != digestPartial {
		t.Errorf("digest.status %q, want partial (whale live)", env.Digest.Status)
	}
	// All sections live and a live winner → ok:true, status live.
	all := richSweep()
	all.cards[keySR] = all.cards[keyVol]
	all.fx = []fxRead{{Pair: "EURUSD", OK: true, Dir: "up", CloseAt: rankFresh}}
	if env := digestEnvelope(all, ""); !env.OK || env.Reason != nil || env.Digest.Status != digestLive {
		t.Errorf("all live: ok=%v reason=%v status=%q", env.OK, env.Reason, env.Digest.Status)
	}
}

// ── macro risk-off gate ──────────────────────────────────────────────────────

func riskOffMacro(t *testing.T) Card {
	t.Helper()
	c, regime := macroCardFrom(macroRespOf(t, goldBullFixture()))
	if regime != "risk_off" || c.Macro == nil || c.Macro.RuleScore == nil ||
		c.Macro.VotingLamps < c.Macro.MinVotingLamps || c.effectiveStatus() != statusOK {
		t.Fatalf("fixture must be a fully lit risk_off card: regime %q readout %+v", regime, c.Macro)
	}
	return c
}

func TestMacroRiskOffGate(t *testing.T) {
	base := riskOffMacro(t)
	// funding 90 > trend 40: the crypto winner whenever macro is gated out.
	confirmed := map[string]Card{keyFunding: fundingCard(0.0009), keyTrend: trendCard(trendUp, 20)}
	run := func(name string, macro Card, at time.Time, wantGate, wantWinner string) {
		t.Helper()
		cards := map[string]Card{keyMacro: macro}
		for k, v := range confirmed {
			v.DataTime = at // fresh relative to this sweep
			cards[k] = v
		}
		g := gathered{cards: cards, regime: "risk_off", at: at}
		p := g.selection()
		if p.MacroGate != wantGate || p.Winner != wantWinner {
			t.Errorf("%s: gate %q winner %q, want %q/%q", name, p.MacroGate, p.Winner, wantGate, wantWinner)
		}
	}
	fresh := base.DataTime.Add(2 * time.Hour)
	run("fresh, fully lit", base, fresh, gateEligible, keyMacro)
	run("stale lamps", base, base.DataTime.Add(macroRiskOffMaxAge+time.Minute), gateStale, keyFunding)

	partial := base
	ro := *base.Macro
	ro.VotingLamps = ro.MinVotingLamps - 1
	partial.Macro = &ro
	run("too few voting lamps", partial, fresh, gatePartialLamps, keyFunding)

	noScore := base
	ro2 := *base.Macro
	ro2.RuleScore = nil
	noScore.Macro = &ro2
	run("no rule score", noScore, fresh, gatePartialLamps, keyFunding)

	degraded := base
	degraded.Status = statusNoData
	run("degraded card", degraded, fresh, gateDegraded, keyFunding)

	// Unknown regime never wins while anything else is eligible.
	g := sweep("unknown", map[string]Card{keyTrend: trendCard(trendFlat, 5)})
	if p := g.selection(); p.Winner != keyTrend || p.MacroGate != "" {
		t.Errorf("unknown regime: winner %q gate %q", p.Winner, p.MacroGate)
	}
}

// ── status: one value across endpoints ───────────────────────────────────────

type testDigestReadout struct {
	Status          string   `json:"status"`
	LiveSections    int      `json:"live_sections"`
	TotalSections   int      `json:"total_sections"`
	DegradedSources []string `json:"degraded_sources"`
	GeneratedAt     string   `json:"generated_at"`
	Selection       struct {
		State      string `json:"state"`
		Rule       string `json:"rule"`
		Winner     string `json:"winner"`
		Line       string `json:"line"`
		Candidates []struct {
			Agent    string  `json:"agent"`
			Eligible bool    `json:"eligible"`
			Excluded *string `json:"excluded"`
		} `json:"candidates"`
	} `json:"selection"`
	Sections []struct {
		Key      string   `json:"key"`
		Title    *string  `json:"title"`
		Lines    []string `json:"lines"`
		OK       bool     `json:"ok"`
		DataAsOf *string  `json:"data_as_of"`
	} `json:"sections"`
}

func TestDigestStatusSameAcrossEndpoints(t *testing.T) {
	for _, tc := range []struct {
		name   string
		ag     func(t *testing.T) *Agents
		status string
		ok     bool // top-level ok = the highlighted card (macro fallback here), not the sweep
	}{
		{"all upstreams dead", deadAgents, digestDegraded, false},
		{"whale live, rest dead", mixedStateAgents, digestPartial, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, srv := newTestAPI(t, tc.ag(t), true)
			_, _, body := httpGet(t, srv.URL+"/agents/digest")
			var env struct {
				OK     bool               `json:"ok"`
				Reason *string            `json:"reason"`
				Digest *testDigestReadout `json:"digest"`
			}
			if err := json.Unmarshal(body, &env); err != nil || env.Digest == nil {
				t.Fatalf("decode digest: %v (%s)", err, body)
			}
			if env.Digest.Status != tc.status || env.OK != tc.ok {
				t.Errorf("/agents/digest: status %q ok=%v, want %q/%v", env.Digest.Status, env.OK, tc.status, tc.ok)
			}
			if tc.ok != (env.Reason == nil) {
				t.Errorf("/agents/digest reason %v with ok=%v", env.Reason, env.OK)
			}
			var row struct {
				Status       string  `json:"status"`
				OK           bool    `json:"ok"`
				Reason       *string `json:"reason"`
				DigestStatus string  `json:"digest_status"`
			}
			_, _, sb := httpGet(t, srv.URL+"/showcase")
			var sc struct {
				Agents []json.RawMessage `json:"agents"`
			}
			if err := json.Unmarshal(sb, &sc); err != nil {
				t.Fatal(err)
			}
			for _, raw := range sc.Agents {
				var probe struct {
					Slug string `json:"slug"`
				}
				_ = json.Unmarshal(raw, &probe)
				if probe.Slug == keyDigest {
					_ = json.Unmarshal(raw, &row)
				}
			}
			// The unified status is the same value on both endpoints; the
			// showcase row's live|degraded follows it (partial reads live).
			if row.DigestStatus != env.Digest.Status {
				t.Errorf("/showcase digest_status %q != /agents/digest digest.status %q", row.DigestStatus, env.Digest.Status)
			}
			if row.OK != (env.Digest.Status != digestDegraded) {
				t.Errorf("/showcase digest row ok=%v with digest.status %q", row.OK, env.Digest.Status)
			}
		})
	}
}

// ── JSON sections == HTML sections ───────────────────────────────────────────

func richSweep() gathered {
	fxAt := rankAt.Add(-30 * time.Minute)
	return gathered{
		regime: "risk_on", at: rankAt,
		cards: map[string]Card{
			keyMacro:    macroOK("risk_on"),
			keyWhale:    {Agent: "Whale Flow Agent", ShortName: "Whale", Asset: "BTC", Short: "outflow <$18M>", Emoji: emojiBull, DataTime: rankFresh},
			keyFunding:  fundingCard(0.0004),
			keyMomentum: momentumCard(0, false),
			keyTrend:    trendCard(trendFlat, 17),
			keySR:       offlineCard("S/R Agent", "S/R", "BTC", keySR, ""),
			keyVol:      {Agent: "Volatility Agent", ShortName: "Vol", Asset: "BTC", Short: "normal", Emoji: emojiNeutral, DataTime: rankFresh},
		},
		fx: []fxRead{
			{Pair: "EURUSD", OK: true, Dir: "up", RSI: 55.5, CloseAt: fxAt},
			{Pair: "GBPUSD", OK: false},
		},
		fxAnyOK: true,
		extras:  []string{"📖 <b>Narrative</b>: AI &amp; agents (trending, score 71)"},
		narrAt:  rankAt.Add(-10 * time.Minute),
	}
}

// Every line the HTML renders below the highlighted card is in the JSON
// sections, in the same order, and nothing else: FX and the narrative are no
// longer HTML-only.
func TestDigestJSONSectionsEqualHTML(t *testing.T) {
	g := richSweep()
	env := digestEnvelope(g, "")
	raw, _ := json.Marshal(env.Digest)
	var d testDigestReadout
	if err := json.Unmarshal(raw, &d); err != nil {
		t.Fatal(err)
	}
	var fromJSON []string
	keys := map[string]bool{}
	for _, s := range d.Sections {
		keys[s.Key] = true
		if s.Title != nil {
			fromJSON = append(fromJSON, *s.Title)
		}
		fromJSON = append(fromJSON, s.Lines...)
	}
	for _, k := range []string{keyFX, "narrative", keyMacro, keyWhale, keyTrend} {
		if !keys[k] {
			t.Errorf("JSON sections miss %q: %v", k, keys)
		}
	}
	if keys[d.Selection.Winner] {
		t.Errorf("the highlighted card (%s) heads the envelope, not a section", d.Selection.Winner)
	}
	// HTML below "Everything else" and above the footer.
	plain := htmlToPlain(env.CardHTML)
	_, rest, ok := strings.Cut(plain, "Everything else\n")
	if !ok {
		t.Fatalf("no Everything else block:\n%s", plain)
	}
	rest, _, _ = strings.Cut(rest, "\nAnalytics, not financial advice")
	var fromHTML []string
	for _, l := range strings.Split(rest, "\n") {
		if strings.TrimSpace(l) != "" {
			fromHTML = append(fromHTML, l)
		}
	}
	if strings.Join(fromHTML, "\n") != strings.Join(fromJSON, "\n") {
		t.Errorf("JSON sections != HTML sections\nJSON:\n%s\nHTML:\n%s", strings.Join(fromJSON, "\n"), strings.Join(fromHTML, "\n"))
	}
	// Each section carries its own data time; offline ones carry none.
	for _, s := range d.Sections {
		switch s.Key {
		case keySR:
			if s.OK || s.DataAsOf != nil {
				t.Errorf("offline S/R section: ok=%v data_as_of=%v", s.OK, s.DataAsOf)
			}
		case keyFX:
			if s.DataAsOf == nil || *s.DataAsOf != "2026-09-15T11:30:00Z" {
				t.Errorf("fx data_as_of %v, want the oldest OK pair", s.DataAsOf)
			}
		case "narrative":
			if s.DataAsOf == nil || *s.DataAsOf != "2026-09-15T11:50:00Z" {
				t.Errorf("narrative data_as_of %v", s.DataAsOf)
			}
		default:
			if s.DataAsOf == nil {
				t.Errorf("%s: live section without data_as_of", s.Key)
			}
		}
	}
	// The legacy one-liner list is unchanged (additive contract).
	if len(env.Sections) != len(digestOrder)-1 {
		t.Errorf("legacy sections: %d, want %d", len(env.Sections), len(digestOrder)-1)
	}
	// The selection line is in the HTML and in the JSON.
	if d.Selection.Line == "" || !strings.Contains(plain, d.Selection.Line) {
		t.Errorf("selection line %q missing from card_html", d.Selection.Line)
	}
	if d.GeneratedAt != "2026-09-15T12:00:00Z" {
		t.Errorf("generated_at %q, want the sweep time", d.GeneratedAt)
	}
}

// ── the selection line ───────────────────────────────────────────────────────

func TestSelectionLineShortAndHonest(t *testing.T) {
	// ruleConfirmed only ever scores eligible AND confirmed candidates, so the
	// comparison fixture is confirmed; mixedField is the review case of
	// 2026-09-16 — three fresh readings, one confirmed, nothing compared.
	all := []topCandidate{
		{Key: keyFunding, Eligible: true, Confirmed: true}, {Key: keyMomentum, Eligible: true, Confirmed: true},
		{Key: keyTrend, Eligible: true, Confirmed: true},
	}
	one := []topCandidate{{Key: keyTrend, Eligible: true, Confirmed: true}, {Key: keyFunding, Excluded: excludedStale}}
	mixedField := []topCandidate{
		{Key: keyFunding, Eligible: true}, {Key: keyMomentum, Eligible: true}, {Key: keyTrend, Eligible: true, Confirmed: true},
	}
	picks := []topPick{
		{Rule: ruleMacroRiskOff, Winner: keyMacro},
		{Rule: ruleConfirmed, Winner: keyFunding, Candidates: all},
		{Rule: ruleConfirmed, Winner: keyTrend, Candidates: one},
		{Rule: ruleUnconfirmed, Winner: keyMomentum, NoHighlight: true},
		{Rule: ruleFallbackMacro, Winner: keyMacro, NoHighlight: true},
		{Rule: ruleConfirmed, Winner: keyTrend, Candidates: mixedField},
	}
	for _, p := range picks {
		l := selectionLine(p)
		if n := utf8.RuneCountInString(l); n > 110 || n == 0 {
			t.Errorf("%s: %d chars: %q", p.Rule, n, l)
		}
		if sanitizeAdviceLanguage(l) != l {
			t.Errorf("%s: advice language in %q", p.Rule, l)
		}
	}
	if l := selectionLine(picks[1]); !strings.Contains(l, "not calibrated") {
		t.Errorf("confirmed pick must say the scales are not calibrated: %q", l)
	}
	if l := selectionLine(picks[2]); !strings.HasPrefix(l, "Trend is the only") {
		t.Errorf("single eligible: %q", l)
	}
	// One confirmed among three fresh readings: no scores were compared, so no
	// "Selected among" the unconfirmed ones and no calibration claim.
	want := "Trend is the only confirmed reading among Funding, Momentum (BTC/ETH), Trend (BTC); selected by rule"
	if l := selectionLine(picks[5]); l != want {
		t.Errorf("one confirmed among eligible:\n got %q\nwant %q", l, want)
	}
}

// ── no_highlight display flag ────────────────────────────────────────────────

func TestNoHighlightFlag(t *testing.T) {
	g := sweep("risk_on", map[string]Card{keyFunding: fundingCard(0), keyMomentum: momentumCard(0, false), keyTrend: trendCard(trendFlat, 17)})
	if digestShowNoHighlight {
		t.Fatal("digestShowNoHighlight must default to off until the product decision")
	}
	env := digestEnvelope(g, "")
	if !strings.HasPrefix(env.Verdict, "Top signal: ") || env.Digest.Selection.State != "no_highlight" {
		t.Errorf("flag off: verdict %q state %q", env.Verdict, env.Digest.Selection.State)
	}
	digestShowNoHighlight = true
	t.Cleanup(func() { digestShowNoHighlight = false })
	env = digestEnvelope(g, "")
	if !strings.HasPrefix(env.Verdict, "No highlighted reading") || !strings.Contains(env.CardHTML, "No highlighted reading") {
		t.Errorf("flag on: verdict %q", env.Verdict)
	}
	// A confirmed pick is never affected by the flag.
	g2 := sweep("risk_on", map[string]Card{keyTrend: trendCard(trendUp, 40)})
	if v := digestEnvelope(g2, "").Verdict; !strings.HasPrefix(v, "Top signal: ") {
		t.Errorf("confirmed pick with the flag on: %q", v)
	}
}
