package demobot

// composite_blocks_test.go — the content blocks of the COMPOSITE cards
// (2026-09-16): FX, the multi-asset Momentum card, the digest and /agents/top,
// plus the limitations lifted onto trend, S/R and macro.
//
// Live on 2026-09-16 none of the four composites filled `blocks`, so the site
// rendered neither a CURRENT READING nor a SCOPE AND LIMITATIONS section for
// them: the FX page was a bare coverage header over a numbered list in which
// the column header read as "clue No. 01" and the order note as "clue No. 02".
//
// The rules of the agents are untouched. What these tests pin is what the new
// sentences may and may not say.

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

// blockForbidden: the words a block may never use — a recommendation, a
// forecast, or a ranking claim over instruments the card refuses to rank.
// Whole words on the lower-cased sentence.
var blockForbidden = []string{
	"buy", "sell", "entry", "target", "forecast", "predict", "expect", "likely",
	"probably", "should", "recommend", "advice", "leader", "best", "strongest",
	"weakest", "winner", "outperform",
}

// assertNoForbidden flags a forbidden word used as a CLAIM. A negated one is
// the opposite of a claim and is exactly what these blocks exist to say
// ("A backdrop, not a forecast"), so "not a <word>" / "not <word>" is allowed.
func assertNoForbidden(t *testing.T, where, s string) {
	t.Helper()
	low := " " + strings.ToLower(s) + " "
	for _, w := range blockForbidden {
		for _, sep := range []string{" ", ",", ".", ";", ":", ")"} {
			needle := " " + w + sep
			for i := 0; ; {
				j := strings.Index(low[i:], needle)
				if j < 0 {
					break
				}
				at := i + j
				head := low[:at+1]
				if !strings.HasSuffix(head, "not a ") && !strings.HasSuffix(head, "not ") &&
					!strings.HasSuffix(head, "never ") {
					t.Errorf("%s uses %q: %q", where, w, s)
				}
				i = at + 1
			}
		}
	}
}

// ── FX ───────────────────────────────────────────────────────────────────────

// The FX card reads four instruments and refuses to add them up. Its blocks
// therefore state the COVERAGE and that this is a comparison — and nothing
// else: no instrument named, no scenario, nothing to invalidate.
func TestFXBlocksAreCoverageOnly(t *testing.T) {
	c := fxCardFromReads(liveFXReads(), fxNow)
	b := c.Blocks
	if b == nil {
		t.Fatal("the FX card must carry blocks")
	}
	if b.WhatHappened == "" || b.Limitations == "" {
		t.Fatalf("both blocks must be filled: %+v", b)
	}
	want := "3 pairs and gold read side by side on closed 1h bars; " + fxOrderClause + "."
	if b.WhatHappened != want {
		t.Errorf("what_happened:\n got %q\nwant %q", b.WhatHappened, want)
	}
	if b.Limitations != fxLimitations {
		t.Errorf("limitations: %q", b.Limitations)
	}
	// An overview of four instruments holds no single idea, level or regime.
	if b.Scenarios != nil {
		t.Errorf("the FX card has no scenario: %v", b.Scenarios)
	}
	if b.Invalidates != nil {
		t.Errorf("the FX card has nothing to invalidate: %v", *b.Invalidates)
	}
	for name, f := range map[string]string{"why_level": b.WhyLevel, "regime": b.Regime, "context": b.Context, "source": b.Source} {
		if f != "" {
			t.Errorf("the FX card has no %s: %q", name, f)
		}
	}
	// No instrument is singled out: naming one in a coverage sentence is the
	// ranking claim fxOrderNote exists to refuse.
	for _, key := range fxPairs {
		if name := assetTable[key].Display; strings.Contains(b.WhatHappened, name) {
			t.Errorf("what_happened names an instrument (%s): %q", name, b.WhatHappened)
		}
	}
}

// Stage 3 (a dollar normalisation or a dollar-strength verdict) must not
// arrive through the blocks either — the same detector the rows face.
func TestFXBlocksCarryNoDollarClaim(t *testing.T) {
	for name, reads := range fxTextFixtures() {
		for _, now := range []time.Time{fxNow, fxWeekend} {
			b := fxCardFromReads(reads, now).Blocks
			if b == nil {
				continue
			}
			for _, l := range []string{b.WhatHappened, b.Limitations} {
				if fxStage3Claim(l) {
					t.Errorf("%s @%s: %q is a stage-3 claim", name, now.Format("Mon 15:04"), l)
				}
				assertNoForbidden(t, name+" fx block", l)
			}
			if b.Scenarios != nil || b.Invalidates != nil {
				t.Errorf("%s: the FX overview must offer neither scenarios nor an invalidation: %+v", name, b)
			}
		}
	}
}

// Coverage is worded from the reads with the header's own count
// (fxCoverageOf): the block's read list is the header's, word for word.
func TestFXBlocksFollowCoverage(t *testing.T) {
	short := func(k string) fxRead {
		return fxRead{Pair: assetTable[k].Display, spec: assetTable[k], Insufficient: true}
	}
	dead := func(k string) fxRead { return fxRead{Pair: assetTable[k].Display, spec: assetTable[k]} }
	live := liveFXReads()
	order := "; " + fxOrderClause + "."
	cases := []struct {
		name  string
		reads []fxRead
		want  string
	}{
		{"pairs and gold", live, "3 pairs and gold read side by side on closed 1h bars" + order},
		{"gold with dead pairs", []fxRead{dead("eurusd"), dead("gbpusd"), dead("usdjpy"), live[3]},
			"0 of 3 pairs and gold read side by side on closed 1h bars" + order},
		{"partial pairs", []fxRead{live[0], short("gbpusd"), dead("usdjpy"), live[3]},
			"1 of 3 pairs and gold read side by side on closed 1h bars" + order},
		{"one pair alone", []fxRead{live[0]}, "1 pair read side by side on closed 1h bars."},
	}
	for _, tc := range cases {
		c := fxCardFromReads(tc.reads, fxNow)
		if c.Blocks == nil {
			t.Errorf("%s: no blocks", tc.name)
			continue
		}
		if c.Blocks.WhatHappened != tc.want {
			t.Errorf("%s:\n got %q\nwant %q", tc.name, c.Blocks.WhatHappened, tc.want)
		}
		// The block's read list is the header's read list.
		head := strings.SplitN(c.Short, " · ", 2)[0]
		if blockRead := strings.SplitN(c.Blocks.WhatHappened, " side by side", 2)[0]; strings.ReplaceAll(head, " + ", " and ") != blockRead {
			t.Errorf("%s: header says %q, block says %q", tc.name, head, blockRead)
		}
	}
	for _, reads := range [][]fxRead{
		{short("eurusd"), short("xauusd")},
		{dead("eurusd"), dead("xauusd")},
	} {
		if b := fxCardFromReads(reads, fxNow).Blocks; b != nil {
			t.Errorf("a card with no reading must carry no blocks: %+v", b)
		}
	}
}

// Review 2026-09-16, point 2: gold with no recent bar (the COMEX break, the
// weekend) is taken out of the header's "read" list and named on its own. The
// block counted gold by r.OK alone and said "3 pairs and gold read" beside a
// header saying "gold: no recent bar". Both now come from one count.
func TestFXBlocksGoldWithoutRecentBarMatchesHeader(t *testing.T) {
	reads := liveFXReads()
	reads[3].CloseAt = fxAt(15, 5) // gold bar 3h30 old at fxNow
	c := fxCardFromReads(reads, fxNow)
	if !strings.Contains(c.Verdict, "gold: no recent bar") {
		t.Fatalf("fixture: the header must name stale gold: %q", c.Verdict)
	}
	b := c.Blocks
	if b == nil {
		t.Fatal("pairs read: blocks expected")
	}
	if strings.Contains(b.WhatHappened, "and gold read") {
		t.Errorf("stale gold is not read by the header's count: %q", b.WhatHappened)
	}
	want := "3 pairs read side by side on closed 1h bars; gold has no recent bar; " + fxOrderClause + "."
	if b.WhatHappened != want {
		t.Errorf("what_happened:\n got %q\nwant %q", b.WhatHappened, want)
	}
	if fxStage3Claim(b.WhatHappened) {
		t.Errorf("stage-3 claim: %q", b.WhatHappened)
	}
}

// Review 2026-09-16, point 5: "ordered by the size of the 24h move" dropped
// half of the order note and was wrong after a session gap, where a row shows
// its change since a named close. The clause now names what the rows are
// actually ordered by, on every path.
func TestFXBlocksOrderClauseTrueAfterAGap(t *testing.T) {
	gap := liveFXReads()
	gap[0].SinceClose, gap[0].RefAt = true, fxAt(11, 22)
	c := fxCardFromReads(gap, fxNow)
	found := false
	for _, f := range c.Facts {
		found = found || f == fxGapNote
	}
	if !found {
		t.Fatal("fixture: the gap note must be on the card")
	}
	w := c.Blocks.WhatHappened
	if strings.Contains(w, "24h move") {
		t.Errorf("after a gap a row's move is not a 24h move: %q", w)
	}
	for _, part := range []string{"the move each row shows", "rows without a fresh bar last", "not by importance"} {
		if !strings.Contains(w, part) {
			t.Errorf("the order clause must say %q, like the order note: %q", part, w)
		}
	}
}

// ── Momentum (the multi-asset card) ──────────────────────────────────────────

func momentumOverviewCard(t *testing.T, assets []momentumAsset) Card {
	t.Helper()
	c := Card{Agent: "Momentum Agent", DataTime: momNow}
	composeMomentum(&c, assets, "", momNow)
	return c
}

// The composite card's verdict is a counter, so its blocks describe the
// counter — never one asset's reading, which would read as the whole card's
// verdict. The counter in the block and the counter in the verdict come from
// one tally (momentumTallyOf), so they cannot disagree.
func TestMomentumOverviewBlocksAreTheCounter(t *testing.T) {
	bar := time.Date(2026, 9, 15, 8, 0, 0, 0, time.UTC)
	c := momentumOverviewCard(t, []momentumAsset{
		{spec: btcSpec, read: momRead(btcSpec, 61.2, 120.5, bar), status: statusOK},
		{spec: assetTable["eth"], read: momRead(assetTable["eth"], 48.0, -3.2, bar), status: statusOK},
		{spec: xauSpec, read: momRead(xauSpec, 50.0, 0, bar), status: statusOK},
	})
	b := c.Blocks
	if b == nil {
		t.Fatal("the multi-asset momentum card must carry blocks")
	}
	// Same counts as the verdict "1 bullish (BTC) · 0 bearish · 2 not confirmed".
	want := "1 bullish (BTC), 0 bearish, 2 not confirmed, each asset on its own timeframe."
	if b.WhatHappened != want {
		t.Errorf("what_happened:\n got %q\nwant %q", b.WhatHappened, want)
	}
	for _, n := range []string{"1 bullish", "0 bearish", "2 not confirmed"} {
		if !strings.Contains(c.Verdict, n) || !strings.Contains(b.WhatHappened, n) {
			t.Errorf("the counter must be the same in the verdict %q and the block %q", c.Verdict, b.WhatHappened)
		}
	}
	if want := momentumColourLine + ". " + momentumMACDLimitation; b.Limitations != want {
		t.Errorf("limitations:\n got %q\nwant %q", b.Limitations, want)
	}
	// The colour rule and the histogram's units — both already on the card.
	for _, want := range []string{"BTC/ETH only", "price units"} {
		if !strings.Contains(b.Limitations, want) {
			t.Errorf("limitations must state %q: %q", want, b.Limitations)
		}
	}
	// No single reading up here: those live per asset in results[].blocks.
	if b.WhyLevel != "" || b.Regime != "" || b.Scenarios != nil || b.Invalidates != nil {
		t.Errorf("the composite card holds no single reading: %+v", b)
	}
	if len(c.Results) != 3 || c.Results[0].Blocks == nil {
		t.Fatalf("per-asset blocks must stay in results[]: %d results", len(c.Results))
	}
	assertNoForbidden(t, "momentum block", b.WhatHappened)
	assertNoForbidden(t, "momentum limitations", b.Limitations)
}

// A shared timeframe is named; an unavailable asset is counted; a sweep with
// no reading at all has no counter to describe.
func TestMomentumOverviewBlocksShapes(t *testing.T) {
	bar := time.Date(2026, 9, 15, 8, 0, 0, 0, time.UTC)
	eth := assetTable["eth"]
	shared := momentumOverviewCard(t, []momentumAsset{
		{spec: btcSpec, read: momRead(btcSpec, 61.2, 120.5, bar), status: statusOK},
		{spec: eth, read: momRead(eth, 40.0, -9.0, bar), status: statusOK},
		{spec: xauSpec, status: statusSourceOffline},
	})
	want := "1 bullish (BTC), 1 bearish (ETH), 0 not confirmed, 1 unavailable on closed 4h candles."
	if shared.Blocks == nil || shared.Blocks.WhatHappened != want {
		t.Errorf("what_happened: %+v, want %q", shared.Blocks, want)
	}
	if b := momentumOverviewBlocks([]momentumAsset{
		{spec: btcSpec, status: statusSourceOffline},
		{spec: eth, status: statusInsufficientHistory},
	}); b != nil {
		t.Errorf("no reading, no counter to describe: %+v", b)
	}
}

// rsi_shown is the string the card prints (rsiShown), beside the untouched raw
// value: a consumer rounding `rsi` itself printed 60.9 under text saying 60.8.
// The rounding rule — away from the threshold — is unchanged.
func TestMomentumResultCarriesRSIShown(t *testing.T) {
	bar := time.Date(2026, 9, 15, 8, 0, 0, 0, time.UTC)
	for _, rsi := range []float64{60.89, 54.96, 45.04, 50, 70.5, 29.99} {
		r := momRead(btcSpec, rsi, 1.0, bar)
		res := momentumResult(btcSpec, r, momentumOnTime)
		if res.RSIShown != rsiShown(rsi) {
			t.Errorf("rsi %g: rsi_shown %q, card prints %q", rsi, res.RSIShown, rsiShown(rsi))
		}
		if res.RSI == nil || *res.RSI != rsi {
			t.Errorf("rsi %g: the raw value must stay untouched, got %v", rsi, res.RSI)
		}
		// The printed value never crosses a threshold the raw value is inside.
		if rsi < momentumBullRSI && res.RSIShown >= "55.0" && res.RSIShown < "6" {
			t.Errorf("rsi %g printed %q — past the 55 threshold", rsi, res.RSIShown)
		}
	}
}

// ── digest / top ─────────────────────────────────────────────────────────────

func compositeSweep(t *testing.T) gathered {
	t.Helper()
	at := time.Date(2026, 9, 15, 0, 0, 0, 0, time.UTC)
	trend := Card{
		Agent: "Trend Agent", ShortName: "Trend", Asset: "BTC", AssetKey: "btc",
		Command: keyTrend, Verdict: "Confirmed UPTREND · 4h", Short: "confirmed uptrend",
		Emoji: emojiBull, State: trendUp, Deviation: 90, DataTime: at, confirmed: true,
		Blocks: &ContentBlocks{WhatHappened: "Confirmed uptrend on 4h.", Limitations: "Card caveat."},
	}
	macro := Card{
		Agent: "Macro Agent", ShortName: "Macro", Command: keyMacro,
		Verdict: "RISK-ON — tradfin lamps lean into risk", Short: "risk-on",
		Emoji: emojiBull, Deviation: 10, DataTime: at,
	}
	return gathered{cards: map[string]Card{keyTrend: trend, keyMacro: macro}, regime: "risk_on", at: at.Add(time.Hour)}
}

// The digest's own blocks describe the SWEEP and the SELECTION: how much read,
// which card the fixed rule highlighted, how it was chosen — and the caveat of
// the rule that chose it.
func TestDigestBlocksDescribeTheSelection(t *testing.T) {
	g := compositeSweep(t)
	p := g.selection()
	_, top := topSelection(g)
	b := digestBlocks(g, p, top)
	if b == nil {
		t.Fatal("a live highlight must carry digest blocks")
	}
	for _, want := range []string{"sections read", digestHeadlineFor(p, top), selectionLine(p)} {
		if !strings.Contains(b.WhatHappened, strings.TrimSuffix(want, ".")) {
			t.Errorf("what_happened must contain %q: %q", want, b.WhatHappened)
		}
	}
	if b.Limitations != selectionCaveat(p) {
		t.Errorf("limitations %q, want the rule's caveat %q", b.Limitations, selectionCaveat(p))
	}
	assertNoForbidden(t, "digest block", b.WhatHappened)
	assertNoForbidden(t, "digest limitations", b.Limitations)
}

// candidate is one topCandidate for the rule table below.
func candidate(key string, eligible, confirmed bool, score int) topCandidate {
	c := topCandidate{Key: key, Eligible: eligible, Confirmed: confirmed, Score: score}
	if !eligible {
		c.Excluded = excludedStale
	}
	return c
}

// Review 2026-09-16, point 1: the caveat used to be one constant for every
// rule. It said "the score that selected this card" where no score selected
// anything (a gate, a fallback order, no candidate at all), and "only
// Momentum's BTC/ETH reads and Trend's BTC read are ranked" — Funding is ranked
// too (signalOrder), it only has no scope suffix because it is market-wide.
// The caveat now branches exactly like selectionLine.
func TestSelectionCaveatFollowsTheRule(t *testing.T) {
	all := allSignalNames()
	cases := []struct {
		name       string
		p          topPick
		scoreChose bool
		want       string
	}{
		{"macro risk-off gate", topPick{Winner: keyMacro, Rule: ruleMacroRiskOff, MacroGate: gateEligible,
			Candidates: []topCandidate{candidate(keyFunding, true, true, 80), candidate(keyMomentum, true, true, 60)}}, false,
			"Placed by the macro risk-off gate, ahead of " + all + "; no comparison between agents chose it, and it is a backdrop, not a finding about one market."},
		{"confirmed, two eligible", topPick{Winner: keyFunding, Rule: ruleConfirmed,
			Candidates: []topCandidate{candidate(keyFunding, true, true, 80), candidate(keyMomentum, true, true, 60), candidate(keyTrend, false, false, 0)}}, true,
			"Chosen by comparing the confirmed readings of Funding, Momentum (BTC/ETH); their scores are on scales not calibrated against each other, so a higher score does not mean a stronger reading."},
		{"confirmed, one eligible", topPick{Winner: keyTrend, Rule: ruleConfirmed,
			Candidates: []topCandidate{candidate(keyFunding, false, false, 0), candidate(keyMomentum, false, false, 0), candidate(keyTrend, true, true, 70)}}, false,
			"Trend is the only fresh live reading among " + all + "; it was not compared with another agent."},
		// Review round 2: Eligible and Confirmed DIVERGE. pickTop scores only
		// eligible AND confirmed candidates, so fresh unconfirmed readings were
		// never compared.
		{"confirmed, one confirmed among three eligible", topPick{Winner: keyTrend, Rule: ruleConfirmed,
			Candidates: []topCandidate{candidate(keyFunding, true, false, 40), candidate(keyMomentum, true, false, 0), candidate(keyTrend, true, true, 70)}}, false,
			"Trend is the only confirmed reading among Funding, Momentum (BTC/ETH), Trend (BTC); it was not compared with another agent."},
		{"confirmed, two confirmed among three eligible", topPick{Winner: keyTrend, Rule: ruleConfirmed,
			Candidates: []topCandidate{candidate(keyFunding, true, false, 40), candidate(keyMomentum, true, true, 30), candidate(keyTrend, true, true, 70)}}, true,
			"Chosen by comparing the confirmed readings of Momentum (BTC/ETH), Trend (BTC); their scores are on scales not calibrated against each other, so a higher score does not mean a stronger reading."},
		{"fallback unconfirmed", topPick{Winner: keyMomentum, Rule: ruleUnconfirmed, NoHighlight: true,
			Candidates: []topCandidate{candidate(keyFunding, true, false, 10), candidate(keyMomentum, true, false, 0)}}, false,
			"No confirmed reading among " + all + "; this card is shown by the digest's fallback order, not as a finding."},
		{"fallback macro", topPick{Winner: keyMacro, Rule: ruleFallbackMacro, NoHighlight: true,
			Candidates: []topCandidate{candidate(keyFunding, false, false, 0)}}, false,
			"No fresh live reading among " + all + "; the macro card is shown as the last resort, not as a finding."},
	}
	for _, tc := range cases {
		got := selectionCaveat(tc.p)
		if got != tc.want {
			t.Errorf("%s:\n got %q\nwant %q", tc.name, got, tc.want)
		}
		low := strings.ToLower(got)
		if !tc.scoreChose && strings.Contains(low, "score") {
			t.Errorf("%s: no score chose this card, the caveat must not say one did: %q", tc.name, got)
		}
		if tc.scoreChose && !strings.Contains(low, "not calibrated") {
			t.Errorf("%s: scores were compared, the caveat must say the scales are not calibrated: %q", tc.name, got)
		}
		if !tc.scoreChose && strings.Contains(low, "calibrated") {
			t.Errorf("%s: nothing was compared, so calibration is not the caveat: %q", tc.name, got)
		}
		// Funding is ranked like the other two: nothing may say only the
		// scoped agents are ranked.
		if strings.Contains(low, "are ranked") || strings.Contains(low, "only momentum") {
			t.Errorf("%s: implies Funding is not ranked: %q", tc.name, got)
		}
		// Agent names come from the scoped registry, never hardcoded.
		for _, k := range signalOrder {
			if strings.Contains(got, signalNames[k]+"'s") {
				t.Errorf("%s: hardcoded agent wording %q", tc.name, got)
			}
		}
		// Branches mirror selectionLine one for one.
		if (strings.Contains(selectionLine(tc.p), "not calibrated")) != tc.scoreChose {
			t.Errorf("%s: selectionLine and the caveat disagree on whether scores were compared", tc.name)
		}
		assertNoForbidden(t, tc.name, got)
	}
}

// The digest and /agents/top carry the SAME caveat for the same sweep.
func TestDigestAndTopShareTheRuleCaveat(t *testing.T) {
	g := compositeSweep(t)
	p := g.selection()
	_, top := topSelection(g)
	d, tb := digestBlocks(g, p, top), topBlocks(p, top)
	if d == nil || tb == nil {
		t.Fatal("a live winner must give both envelopes blocks")
	}
	if !strings.HasSuffix(tb.Limitations, selectionCaveat(p)) || d.Limitations != selectionCaveat(p) {
		t.Errorf("digest %q / top %q must carry the rule's caveat %q", d.Limitations, tb.Limitations, selectionCaveat(p))
	}
}

// /agents/top IS the winner card, so it keeps that card's own sentences and
// only adds the selection caveat — by value, never through the shared pointer
// the same sweep hands the digest.
func TestTopBlocksKeepTheWinnerAndAddTheCaveat(t *testing.T) {
	g := compositeSweep(t)
	p := g.selection()
	_, top := topSelection(g)
	before := *top.Blocks
	b := topBlocks(p, top)
	if b == nil || b == top.Blocks {
		t.Fatalf("topBlocks must return a copy of the winner's blocks, got %p (card %p)", b, top.Blocks)
	}
	if !reflect.DeepEqual(*top.Blocks, before) {
		t.Errorf("the winner card's blocks were mutated: %+v", *top.Blocks)
	}
	if b.WhatHappened != before.WhatHappened {
		t.Errorf("the winner's reading must survive: %q", b.WhatHappened)
	}
	if b.Limitations != before.Limitations+" "+selectionCaveat(p) {
		t.Errorf("limitations: %q", b.Limitations)
	}
	bare := top
	bare.Blocks = &ContentBlocks{WhatHappened: "x"}
	if got := topBlocks(p, bare); got == nil || got.Limitations != selectionCaveat(p) {
		t.Errorf("a winner without limitations of its own gets the caveat alone: %+v", got)
	}
}

// Review 2026-09-16, point 4: no live reading, no blocks. Every other agent
// serves none on ok:false and the docs promise it; the composites used to
// serve {what_happened:"", limitations:<caveat>} for a dead winner.
func TestCompositeBlocksAbsentWithoutALiveReading(t *testing.T) {
	g := compositeSweep(t)
	p := g.selection()
	_, top := topSelection(g)

	dead := top
	dead.Offline = true
	dead.Status = statusSourceOffline
	if dead.effectiveStatus() == statusOK {
		t.Fatal("fixture: the dead winner must not be ok")
	}
	if b := topBlocks(p, dead); b != nil {
		t.Errorf("/agents/top with an ok:false winner must carry no blocks: %+v", b)
	}
	if b := digestBlocks(g, p, dead); b != nil {
		t.Errorf("the digest with an ok:false highlight must carry no blocks: %+v", b)
	}
	// A live winner without blocks of its own: a caveat alone under an empty
	// what_happened describes nothing, so no blocks either.
	none := top
	none.Blocks = nil
	if b := topBlocks(p, none); b != nil {
		t.Errorf("a winner without blocks must not get a caveat-only pair: %+v", b)
	}
}

// ── limitations lifted onto trend, S/R and macro ─────────────────────────────

// Their caveats used to sink into the numbered facts list because the block
// was empty. The block now carries the card's OWN line, verbatim — nothing new
// is claimed.
func TestTrendSRMacroLimitationsComeFromTheCard(t *testing.T) {
	up := trendView{r: trendRead{State: trendUp, ADX: 31.4, EMA20: 4400, EMA50: 4380, EMA200: 4200, Last: 4450, ATR: 60}, tf: "4h", atr: 60, inv: 4320, invSide: "below"}
	if got := up.blocks("confirmed uptrend").Limitations; got != up.holdsLine() {
		t.Errorf("confirmed trend limitations %q, card line %q", got, up.holdsLine())
	}
	flat := trendView{r: trendRead{State: trendFlat, ADX: 12.0, EMA20: 4400, EMA50: 4380, EMA200: 4200, Last: 4450}, tf: "4h"}
	if got := flat.blocks("flat — no trend").Limitations; got != flat.confirmLine() {
		t.Errorf("unconfirmed trend limitations %q, card line %q", got, flat.confirmLine())
	}
	// Both wordings are already facts[] of the same card.
	for _, v := range []trendView{up, flat} {
		lim := v.blocks("x").Limitations
		if lim == "" {
			t.Fatal("trend limitations must be filled")
		}
		found := false
		for _, f := range v.facts() {
			found = found || f == lim
		}
		if !found {
			t.Errorf("trend limitations %q is not one of the card's own facts %v", lim, v.facts())
		}
		assertNoForbidden(t, "trend limitations", lim)
	}
}

func TestSRLimitationsIsTheWindowLine(t *testing.T) {
	sup := []SRLevel{{Level: 2400, Raw: 2400, Touches: 7, Strength: 7}}
	res := []SRLevel{{Level: 2531, Raw: 2531, Touches: 7, Strength: 7}}
	c := srCardFrom(btcSpec, sup, res, 2516.4, 200, time.Date(2026, 9, 15, 8, 0, 0, 0, time.UTC), nil)
	if c.Blocks == nil || c.Blocks.Limitations == "" {
		t.Fatal("the S/R card must carry limitations")
	}
	found := false
	for _, f := range c.Facts {
		found = found || f == c.Blocks.Limitations
	}
	if !found {
		t.Errorf("S/R limitations %q is not one of the card's own facts %v", c.Blocks.Limitations, c.Facts)
	}
	assertNoForbidden(t, "sr limitations", c.Blocks.Limitations)
}

// Review 2026-09-16, point 3: the macro limitation is assembled from phrases
// the card prints elsewhere (howTexts, btcContext, goldContext) — it is not a
// verbatim fact line — and it must not qualify a gold score the card does not
// have: when goldContext prints "no read", the gold clause is dropped.
func TestMacroLimitationsStatesTheBackdrop(t *testing.T) {
	// Assembled, not verbatim: each phrase exists on the card, the sentence
	// does not.
	c, _ := macroCardFrom(macroRespOf(t, macroLiveFixture))
	if c.Blocks == nil {
		t.Fatal("fixture: the global card must carry blocks")
	}
	lim := c.Blocks.Limitations
	for _, src := range []struct{ phrase, where string }{
		{"a backdrop, not a forecast", strings.ToLower(howTexts[keyMacro])},
		{"btc direction is not inferred", strings.ToLower(strings.Join(c.Facts, "\n"))},
		{"experimental model, own weights", strings.ToLower(strings.Join(c.Facts, "\n"))},
	} {
		if !strings.Contains(src.where, src.phrase) {
			t.Errorf("source phrase %q is no longer on the card", src.phrase)
		}
	}
	for _, f := range c.Facts {
		if f == lim {
			t.Errorf("the comment says assembled, not verbatim — update it if this became a fact line: %q", lim)
		}
	}
	// Gold scored in the live fixture: the clause is there.
	if !strings.Contains(c.Facts[6], "gold score") {
		t.Fatalf("fixture: the live card must have a gold score: %q", c.Facts[6])
	}
	if !strings.Contains(lim, "gold score") {
		t.Errorf("a scored gold backdrop keeps its clause: %q", lim)
	}
	// No gold read: no clause about a gold score.
	noGold := macroView{gold: modelRead{m: goldModel}}
	withGold := macroView{gold: modelRead{m: goldModel, score: new(int)}}
	if got := noGold.limitations(); strings.Contains(strings.ToLower(got), "gold") {
		t.Errorf("gold has no read, the caveat must not mention a gold score: %q", got)
	}
	if got, want := noGold.limitations(), "A backdrop, not a forecast: BTC direction is not inferred from the regime."; got != want {
		t.Errorf("no gold read:\n got %q\nwant %q", got, want)
	}
	if got := withGold.limitations(); !strings.Contains(got, "experimental model") {
		t.Errorf("scored gold: %q", got)
	}
	assertNoForbidden(t, "macro limitations", noGold.limitations())
	assertNoForbidden(t, "macro limitations", withGold.limitations())
}

// Review round 2, point 1 (also for the selection line): an eligible but
// unconfirmed candidate never competed, so neither text may list it as
// compared.
func TestSelectionTextsListOnlyWhatWasCompared(t *testing.T) {
	p := topPick{Winner: keyTrend, Rule: ruleConfirmed, Candidates: []topCandidate{
		candidate(keyFunding, true, false, 40), candidate(keyMomentum, true, true, 30), candidate(keyTrend, true, true, 70),
	}}
	for name, text := range map[string]string{"selection line": selectionLine(p), "caveat": selectionCaveat(p)} {
		if strings.Contains(text, "Funding") {
			t.Errorf("%s lists unconfirmed Funding as compared: %q", name, text)
		}
		if !strings.Contains(text, "not calibrated") {
			t.Errorf("%s: two confirmed readings were compared: %q", name, text)
		}
	}
}

// Review round 2, point 2: funding's and trend's limitations end without a
// full stop, and the appended caveat ran straight into them.
func TestTopBlocksEndTheWinnerSentenceBeforeTheCaveat(t *testing.T) {
	g := compositeSweep(t)
	p := g.selection()
	_, top := topSelection(g)
	caveat := selectionCaveat(p)
	for _, tc := range []struct{ own, want string }{
		{"Last funding rate only; positions on other venues", "Last funding rate only; positions on other venues. " + caveat},
		{"Card caveat.", "Card caveat. " + caveat},
		{"Is it a question?", "Is it a question? " + caveat},
		{"  trailing space  ", "trailing space. " + caveat},
	} {
		w := top
		w.Blocks = &ContentBlocks{WhatHappened: "x", Limitations: tc.own}
		got := topBlocks(p, w)
		if got == nil || got.Limitations != tc.want {
			t.Errorf("own %q:\n got %+v\nwant %q", tc.own, got, tc.want)
			continue
		}
		if strings.Contains(got.Limitations, "..") {
			t.Errorf("double full stop: %q", got.Limitations)
		}
	}
}

// Review round 2, point 5: the copy is deep — the invalidates pointer and the
// scenarios slice are the copy's own, so writing through them cannot reach the
// winner card the digest renders from the same sweep.
func TestTopBlocksCopyIsDeep(t *testing.T) {
	g := compositeSweep(t)
	p := g.selection()
	_, top := topSelection(g)
	inv := "A closed 4h candle below 4320 invalidates the uptrend idea"
	top.Blocks = &ContentBlocks{WhatHappened: "x", Invalidates: &inv, Scenarios: []string{"a", "b"}}
	b := topBlocks(p, top)
	if b.Invalidates == nil || *b.Invalidates != inv {
		t.Fatalf("invalidates must survive the copy: %v", b.Invalidates)
	}
	if b.Invalidates == top.Blocks.Invalidates {
		t.Error("invalidates pointer is shared with the winner card")
	}
	*b.Invalidates = "changed"
	b.Scenarios[0] = "changed"
	if *top.Blocks.Invalidates != inv || top.Blocks.Scenarios[0] != "a" {
		t.Errorf("writing through the copy changed the winner: %q / %v", *top.Blocks.Invalidates, top.Blocks.Scenarios)
	}
}

// Review round 2, point 3: the colour caveat follows what the card is coloured
// by. A scan with no BTC/ETH (?assets=gold,eurusd) is coloured by the assets it
// reads — "Colour follows BTC/ETH only" there is false.
func TestMomentumOverviewColourCaveatFollowsTheCard(t *testing.T) {
	bar := time.Date(2026, 9, 15, 8, 0, 0, 0, time.UTC)
	eur, eth := assetTable["eurusd"], assetTable["eth"]
	ok := func(spec assetSpec, rsi, hist float64) momentumAsset {
		return momentumAsset{spec: spec, read: momRead(spec, rsi, hist, bar), status: statusOK}
	}
	cases := []struct {
		name   string
		assets []momentumAsset
		colour string
	}{
		{"no BTC/ETH requested (gold, eurusd)", []momentumAsset{ok(xauSpec, 60, 1), ok(eur, 40, -1)},
			"No BTC/ETH is on this card, so the colour follows every asset read"},
		{"BTC/ETH requested, none read", []momentumAsset{{spec: btcSpec, status: statusSourceOffline}, ok(xauSpec, 60, 1)},
			momentumColourNoneLine},
		{"BTC/ETH beside other reads", []momentumAsset{ok(btcSpec, 60, 1), ok(xauSpec, 40, -1)},
			momentumColourLine},
		{"BTC/ETH only", []momentumAsset{ok(btcSpec, 60, 1), ok(eth, 50, 0)},
			"Colour follows the BTC/ETH reads, the ones the digest ranks"},
	}
	for _, tc := range cases {
		c := momentumOverviewCard(t, tc.assets)
		if c.Blocks == nil {
			t.Errorf("%s: no blocks", tc.name)
			continue
		}
		want := tc.colour + ". " + momentumMACDLimitation
		if c.Blocks.Limitations != want {
			t.Errorf("%s:\n got %q\nwant %q", tc.name, c.Blocks.Limitations, want)
		}
		// The caveat agrees with the colour line the card itself prints, where
		// it prints one.
		for _, f := range c.Facts {
			if strings.HasPrefix(f, "Colour follows") && !strings.HasPrefix(c.Blocks.Limitations, f) {
				t.Errorf("%s: card says %q, block says %q", tc.name, f, c.Blocks.Limitations)
			}
		}
		assertNoForbidden(t, tc.name, c.Blocks.Limitations)
	}
	// The FX/gold scan is really coloured by its reads (the header's fallback),
	// so the caveat above is the true one.
	one := momentumOverviewCard(t, []momentumAsset{ok(xauSpec, 60, 1)})
	if one.Emoji != emojiBull {
		t.Errorf("a gold-only scan is coloured by gold (header fallback), got %s", one.Emoji)
	}
}
