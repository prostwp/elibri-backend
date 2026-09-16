package demobot

// momentum_text.go — everything the Momentum card SAYS.
//
// Presentation only. The rule is momentumVerdict (agents.go): RSI ≥ 55 with a
// MACD histogram above 0 = bullish, RSI ≤ 45 with it below 0 = bearish, else
// neutral. Nothing here changes a verdict, a semaphore of a single asset, or
// the digest inputs (confirmed / Deviation / rankAsOf).
//
// One enumeration by construction: the two conditions are listed in exactly
// ONE place — momentumChecks, evaluated with the rule's own thresholds. The
// checklist, the neutral reason, the "turns bullish / bearish" lines and the
// scenarios all derive from it; all ✓ toward a direction is exactly that
// direction's verdict (tested).
//
// Honesty rules for every line below:
//   - analytics only: no advice, no targets, no price levels, no "price will";
//     Momentum has no price level at all — its thresholds are indicator values;
//   - the raw MACD histogram is in price units and cannot be compared across
//     assets, so the card prints only its sign; the raw value is in results[];
//   - the RSI is printed through rsiShown only, so a printed value never sits
//     on the other side of a threshold from the real one;
//   - "market closed" only when the fixed weekend window (isForexOpen) says so;
//     a Yahoo bar older than two bars otherwise reads "data delayed" (Binance
//     is on time by its source contract, see momentumFreshness);
//   - the multi-asset colour follows the BTC/ETH reads only (the ones the
//     digest ranks); every asset is counted;
//   - volume and ETH-vs-BTC are context and say so on the line;
//   - every line — verdict, facts, each content block — fits
//     momentumFactMaxRunes.

import (
	"fmt"
	"math"
	"regexp"
	"strings"
	"time"
)

// momentumFactMaxRunes is the readability budget for one line of momentum text.
const momentumFactMaxRunes = 110

// The rule's thresholds — the single definition, read by momentumVerdict.
const (
	momentumBullRSI = 55.0
	momentumBearRSI = 45.0
)

const (
	momentumBullish = "bullish"
	momentumBearish = "bearish"
)

// Machine states served as results[].state. The top contract stays
// bullish | bearish | neutral (results[].verdict, semaphore); the state says
// WHICH neutral.
const (
	momStateBullish  = "confirmed_bullish"
	momStateBearish  = "confirmed_bearish"
	momStateConflict = "conflict"     // RSI past one threshold, MACD on the other side
	momStateBelow55  = "rsi_below_55" // MACD above 0, RSI inside 45–55
	momStateAbove45  = "rsi_above_45" // MACD below 0, RSI inside 45–55
	momStateMACDZero = "macd_at_zero" // RSI past a threshold, histogram exactly 0
	momStateZone     = "neutral_zone" // RSI inside 45–55, histogram exactly 0
)

// Freshness values served as results[].freshness.
const (
	momentumOnTime       = "on_time"       // no missing bar the rule can detect
	momentumMarketClosed = "market_closed" // Yahoo asset inside the fixed weekend window
	momentumDataDelayed  = "data_delayed"  // last closed bar older than two bars while the source should trade
)

// momentumRuleLine is the overview's one statement of what turns a reading.
const momentumRuleLine = "Bullish needs RSI ≥ 55 and MACD histogram above 0; bearish needs RSI ≤ 45 and below 0"

// momentumWhyLevel: Momentum reads indicator thresholds, never a price level.
const momentumWhyLevel = "No price level: 55 and 45 are the RSI(14) thresholds, 0 is the MACD histogram line"

// rsiShown is the only way momentum text prints RSI: one decimal, rounded
// toward 50. Above 50 the value is floored, below 50 it is ceiled, so a value
// under 55 never prints at or above 55 and a value over 45 never prints at or
// below 45 (at %.1f, 54.96 used to print "55.0 < 55"). The printed value is
// never more than 0.1 from the real one.
func rsiShown(v float64) string {
	var s float64
	if v < 50 {
		s = math.Ceil(v*10) / 10
		if v > momentumBearRSI && s <= momentumBearRSI { // float rounding at the edge
			s = momentumBearRSI + 0.1
		}
	} else {
		s = math.Floor(v*10) / 10
		if v < momentumBullRSI && s >= momentumBullRSI {
			s = momentumBullRSI - 0.1
		}
	}
	return fmt.Sprintf("%.1f", s)
}

// macdSide is the histogram as the card shows it: its sign only.
func macdSide(hist float64) string {
	switch {
	case hist > 0:
		return "above 0"
	case hist < 0:
		return "below 0"
	}
	return "at 0"
}

// momentumLean is the direction the checklist is evaluated toward: the side
// RSI is past, else the side of the MACD histogram; "" when RSI is inside
// 45–55 and the histogram is exactly 0.
func momentumLean(rsi, hist float64) string {
	switch {
	case rsi >= momentumBullRSI:
		return momentumBullish
	case rsi <= momentumBearRSI:
		return momentumBearish
	case hist > 0:
		return momentumBullish
	case hist < 0:
		return momentumBearish
	}
	return ""
}

// momentumState names which of the rule's outcomes the read is.
func momentumState(rsi, hist float64) string {
	switch momentumVerdict(rsi, hist) {
	case momentumBullish:
		return momStateBullish
	case momentumBearish:
		return momStateBearish
	}
	switch {
	case rsi >= momentumBullRSI && hist < 0, rsi <= momentumBearRSI && hist > 0:
		return momStateConflict
	case rsi >= momentumBullRSI || rsi <= momentumBearRSI: // histogram exactly 0
		return momStateMACDZero
	case hist > 0:
		return momStateBelow55
	case hist < 0:
		return momStateAbove45
	}
	return momStateZone
}

// momentumWhy is the one-line reason for the state, from the same numbers.
func momentumWhy(rsi, hist float64) string {
	switch momentumState(rsi, hist) {
	case momStateBullish, momStateBearish:
		return "RSI and MACD agree"
	case momStateConflict:
		if rsi >= momentumBullRSI {
			return "conflict: RSI up, MACD down"
		}
		return "conflict: RSI down, MACD up"
	case momStateMACDZero:
		if rsi >= momentumBullRSI {
			return "RSI up, MACD histogram at 0"
		}
		return "RSI down, MACD histogram at 0"
	case momStateBelow55:
		return "RSI below the 55 threshold"
	case momStateAbove45:
		return "RSI above the 45 threshold"
	}
	return "RSI in the neutral zone 45–55"
}

// momentumWord is the verdict as the card text says it; "neutral" reads
// "not confirmed" (the JSON verdict and semaphore keep "neutral").
func momentumWord(verdict string) string {
	if verdict == momentumBullish || verdict == momentumBearish {
		return verdict
	}
	return "not confirmed"
}

// momentumCheck is one condition of the rule, evaluated toward a direction.
type momentumCheck struct {
	name  string // "RSI" | "MACD", for "now ✗: …" lists
	label string // checklist text, without the mark
	ok    bool
}

func (c momentumCheck) String() string {
	if c.ok {
		return c.label + " ✓"
	}
	return c.label + " ✗"
}

// momentumChecks is THE enumeration of the rule's two conditions toward dir.
// dir "" (RSI inside 45–55, histogram exactly 0) fails both directions: the
// checklist then says exactly that.
func momentumChecks(rsi, hist float64, dir string) []momentumCheck {
	shown := rsiShown(rsi)
	switch dir {
	case momentumBullish:
		ok := rsi >= momentumBullRSI
		cmp := "<"
		if ok {
			cmp = "≥"
		}
		return []momentumCheck{
			{name: "RSI", label: fmt.Sprintf("RSI %s %s %g", shown, cmp, momentumBullRSI), ok: ok},
			{name: "MACD", label: "MACD histogram " + macdSide(hist), ok: hist > 0},
		}
	case momentumBearish:
		ok := rsi <= momentumBearRSI
		cmp := ">"
		if ok {
			cmp = "≤"
		}
		return []momentumCheck{
			{name: "RSI", label: fmt.Sprintf("RSI %s %s %g", shown, cmp, momentumBearRSI), ok: ok},
			{name: "MACD", label: "MACD histogram " + macdSide(hist), ok: hist < 0},
		}
	}
	return []momentumCheck{
		{name: "RSI", label: fmt.Sprintf("RSI %s in %g–%g", shown, momentumBearRSI, momentumBullRSI)},
		{name: "MACD", label: "MACD histogram " + macdSide(hist)},
	}
}

// momentumChecklist renders the checks toward the lean.
func momentumChecklist(rsi, hist float64) string {
	checks := momentumChecks(rsi, hist, momentumLean(rsi, hist))
	parts := make([]string, len(checks))
	for i, c := range checks {
		parts[i] = c.String()
	}
	return strings.Join(parts, " · ")
}

// momentumDirOrder lists the lean first (bullish first when there is none).
func momentumDirOrder(rsi, hist float64) []string {
	if momentumLean(rsi, hist) == momentumBearish {
		return []string{momentumBearish, momentumBullish}
	}
	return []string{momentumBullish, momentumBearish}
}

// momentumTurnLine — what keeps (confirmed) or turns (otherwise) a reading
// toward dir, naming the ✗ conditions.
func momentumTurnLine(rsi, hist float64, dir string) string {
	rsiCond, macdCond := fmt.Sprintf("RSI ≥ %g", momentumBullRSI), "the MACD histogram is above 0"
	if dir == momentumBearish {
		rsiCond, macdCond = fmt.Sprintf("RSI ≤ %g", momentumBearRSI), "the MACD histogram is below 0"
	}
	if momentumVerdict(rsi, hist) == dir {
		return fmt.Sprintf("Stays %s while %s and %s; any ✗ turns it neutral", dir, rsiCond, macdCond)
	}
	var failing []string
	for _, c := range momentumChecks(rsi, hist, dir) {
		if !c.ok {
			failing = append(failing, c.name)
		}
	}
	return fmt.Sprintf("Turns %s when %s and %s (now ✗: %s)", dir, rsiCond, macdCond, strings.Join(failing, ", "))
}

// momentumScenarios — two "If …, the reading …" state transitions of the
// rule. Never a price direction: the reading changes, not the market.
func momentumScenarios(rsi, hist float64) []string {
	switch momentumVerdict(rsi, hist) {
	case momentumBullish:
		return []string{
			fmt.Sprintf("If RSI holds ≥ %g and the MACD histogram stays above 0, the reading stays bullish", momentumBullRSI),
			fmt.Sprintf("If RSI drops below %g or the MACD histogram falls to 0 or below, the reading turns neutral", momentumBullRSI),
		}
	case momentumBearish:
		return []string{
			fmt.Sprintf("If RSI holds ≤ %g and the MACD histogram stays below 0, the reading stays bearish", momentumBearRSI),
			fmt.Sprintf("If RSI rises above %g or the MACD histogram rises to 0 or above, the reading turns neutral", momentumBearRSI),
		}
	}
	out := make([]string, 0, 2)
	for _, d := range momentumDirOrder(rsi, hist) {
		c := momentumChecks(rsi, hist, d)
		out = append(out, fmt.Sprintf("If %s and %s, the reading turns %s", rsiEvent(d, c[0].ok), macdEvent(d, c[1].ok), d))
	}
	return out
}

func rsiEvent(dir string, holds bool) string {
	switch {
	case dir == momentumBullish && holds:
		return fmt.Sprintf("RSI holds ≥ %g", momentumBullRSI)
	case dir == momentumBullish:
		return fmt.Sprintf("RSI rises to %g or above", momentumBullRSI)
	case holds:
		return fmt.Sprintf("RSI holds ≤ %g", momentumBearRSI)
	}
	return fmt.Sprintf("RSI falls to %g or below", momentumBearRSI)
}

func macdEvent(dir string, holds bool) string {
	side := "above"
	if dir == momentumBearish {
		side = "below"
	}
	if holds {
		return "the MACD histogram stays " + side + " 0"
	}
	return "the MACD histogram turns " + side + " 0"
}

// momentumBlocks is the content-ready form of one asset's read (same
// ContentBlocks contract as trend / S/R): why_level explains the thresholds
// (there is no price level), invalidates says what ends a CONFIRMED reading
// (null otherwise), regime is the asset's LOCAL momentum state — not a market
// regime.
func momentumBlocks(label, tf string, rsi, hist float64) *ContentBlocks {
	v := momentumVerdict(rsi, hist)
	word, why, tfw := momentumWord(v), momentumWhy(rsi, hist), candleWord(tf)
	b := &ContentBlocks{
		WhatHappened: fmt.Sprintf("%s %s reads %s: %s (RSI %s, MACD histogram %s).",
			label, tfw, word, why, rsiShown(rsi), macdSide(hist)),
		WhyLevel:  momentumWhyLevel,
		Scenarios: momentumScenarios(rsi, hist),
		Regime:    fmt.Sprintf("Local momentum · %s %s · %s: %s", label, tfw, word, why),
	}
	var ends string
	switch v {
	case momentumBullish:
		ends = fmt.Sprintf("A closed %s candle with RSI below %g or the MACD histogram at or below 0 ends the bullish reading", tfw, momentumBullRSI)
	case momentumBearish:
		ends = fmt.Sprintf("A closed %s candle with RSI above %g or the MACD histogram at or above 0 ends the bearish reading", tfw, momentumBearRSI)
	}
	if ends != "" {
		b.Invalidates = &ends
	}
	return b
}

// momentumFreshness judges one asset's last closed bar at the card's clock.
//
// Binance is on_time by the source's own contract, not by the clock: the
// kline cache never serves an entry older than klineTTL (errors are not
// cached), and a klines answer always ends with the bar still forming, which
// candlesWindow drops — so the last closed bar is at most one bar (+60 s) old
// whenever a read exists. A clock test here could only fire on a feed that
// stopped advancing its forming bar, and it would make the stamped
// single-asset Binance card depend on the clock (caching is out of scope).
//
// Yahoo (FX and gold) is judged by the clock:
//   - inside the fixed weekend window (isForexOpen: Friday 21:00 to Sunday
//     21:00 UTC) it is market_closed — the only "closed" the rule knows; gold
//     (COMEX) shares that window, see docs for the limits;
//   - otherwise a bar older than two bars (barMaxAge, the digest's own stale
//     bound) is data_delayed — unless the market was still closed two bars
//     ago: right after the reopen a missing bar is not yet due;
//   - everything else is on_time.
func momentumFreshness(source, interval string, closeAt, now time.Time) string {
	if source != srcYahoo {
		return momentumOnTime
	}
	if !isForexOpen(now) {
		return momentumMarketClosed
	}
	bound := barMaxAge(interval)
	if closeAt.IsZero() || bound <= 0 || now.Sub(closeAt) <= bound || !isForexOpen(now.Add(-bound)) {
		return momentumOnTime
	}
	return momentumDataDelayed
}

func momentumFreshWords(fresh string) string {
	switch fresh {
	case momentumMarketClosed:
		return "market closed"
	case momentumDataDelayed:
		return "data delayed"
	}
	return ""
}

// momentumBarTime prints a bar close time: "Sep 15 08:00 UTC".
func momentumBarTime(t time.Time) string {
	return t.UTC().Format("Jan 2 15:04") + " UTC"
}

// ── multi-asset composite (overview and scans) ───────────────────────────────

// momentumAsset is one asset of a multi-asset card: its spec, its read (valid
// only when status is statusOK) and its outcome.
type momentumAsset struct {
	spec   assetSpec
	read   momentumRead
	status cardStatus
}

// momentumStateLine — "BTC · 4h: not confirmed — RSI below the 55 threshold".
func momentumStateLine(label string, r momentumRead) string {
	return fmt.Sprintf("%s · %s: %s — %s", label, candleWord(r.interval), momentumWord(r.verdict), momentumWhy(r.rsi, r.hist))
}

// momentumCheckLine — the asset's checklist, its bar time and freshness.
func momentumCheckLine(label string, r momentumRead, fresh string) string {
	s := fmt.Sprintf("%s: %s · last bar %s", label, momentumChecklist(r.rsi, r.hist), momentumBarTime(r.closeAt))
	if w := momentumFreshWords(fresh); w != "" {
		s += " · " + w
	}
	return s
}

// momentumFailLine states an asset that produced no reading.
func momentumFailLine(a momentumAsset) string {
	what := "data unavailable right now"
	if a.status == statusInsufficientHistory {
		what = "insufficient history for RSI/MACD"
	}
	return fmt.Sprintf("%s · %s: %s", a.spec.Display, candleWord(a.spec.Interval), what)
}

// momentumCountName is the asset's name inside the header counter: the card
// line keeps the full "GOLD · COMEX GC=F", the counter says "GOLD" so its own
// " · " separators stay unambiguous.
func momentumCountName(spec assetSpec) string {
	if spec.isGold() {
		return "GOLD"
	}
	return spec.Display
}

// momentumHeader is the overview/scan verdict — a counter, never one asset's
// verdict — and the semaphore derived from it:
//
//	"1 bullish (BTC) · 0 bearish · 2 not confirmed[ · N unavailable][ · 4h]"
//
// The timeframe rides in the header whenever every read shares it (each line
// names its own anyway). The counter covers every asset.
//
// Semaphore — the same counter rule over the RANKED reads only (Binance:
// BTC/ETH, the reads the digest ranks and `confirmed` comes from): bullish
// when at least one reads bullish and none bearish, bearish mirrored, neutral
// otherwise. So a colour never sits beside the digest's "no confirmed
// reading" because of a gold or FX read. The fallback "colour by all reads"
// applies only when the REQUEST holds no BTC/ETH asset at all (an FX/gold
// scan, which the digest never sweeps); decided on the requested specs, not
// on what read: requested BTC/ETH that produced no reading leave the card
// neutral (the default trio with dead crypto and a live gold read).
func momentumHeader(assets []momentumAsset) (verdict, emoji string) {
	var bull, bear []string
	neutral, failed := 0, 0
	rankedBull, rankedBear := 0, 0
	rankedAsked := false
	tf, sameTF := "", true
	for _, a := range assets {
		isRanked := a.spec.Source == srcBinance
		rankedAsked = rankedAsked || isRanked
		if a.status != statusOK {
			failed++
			continue
		}
		switch a.read.verdict {
		case momentumBullish:
			bull = append(bull, momentumCountName(a.spec))
			if isRanked {
				rankedBull++
			}
		case momentumBearish:
			bear = append(bear, momentumCountName(a.spec))
			if isRanked {
				rankedBear++
			}
		default:
			neutral++
		}
		if tf == "" {
			tf = a.read.interval
		} else if a.read.interval != tf {
			sameTF = false
		}
	}
	part := func(names []string, word string) string {
		s := fmt.Sprintf("%d %s", len(names), word)
		if len(names) > 0 {
			s += " (" + strings.Join(names, ", ") + ")"
		}
		return s
	}
	verdict = part(bull, momentumBullish) + " · " + part(bear, momentumBearish) + fmt.Sprintf(" · %d not confirmed", neutral)
	if failed > 0 {
		verdict += fmt.Sprintf(" · %d unavailable", failed)
	}
	if sameTF && tf != "" {
		verdict += " · " + tf
	}
	cb, cr := rankedBull, rankedBear
	if !rankedAsked {
		cb, cr = len(bull), len(bear)
	}
	switch {
	case cb > 0 && cr == 0:
		emoji = emojiBull
	case cr > 0 && cb == 0:
		emoji = emojiBear
	default:
		emoji = emojiNeutral
	}
	return verdict, emoji
}

// momentumColourLine is added when a card mixes ranked (BTC/ETH) and other
// read assets, so a counted gold/FX reading is not mistaken for the colour.
const momentumColourLine = "Colour follows BTC/ETH only, the reads the digest ranks; other assets are only counted"

// momentumColourNoneLine is added when BTC/ETH were requested but none of
// them produced a reading: the card stays neutral whatever its other reads say.
const momentumColourNoneLine = "Colour follows BTC/ETH only, the reads the digest ranks; no BTC/ETH read is available"

// momentumResult is one OK asset's machine entry: the reading at raw
// precision, its state and reason, timeframe, bar time, freshness and blocks.
func momentumResult(spec assetSpec, r momentumRead, fresh string) AssetResult {
	res := assetResult(spec.Display, statusOK)
	rsi, hist := r.rsi, r.hist
	res.Timeframe = r.interval
	if !r.closeAt.IsZero() {
		res.DataAsOf = r.closeAt.UTC().Format(time.RFC3339)
	}
	res.Freshness = fresh
	res.Verdict = r.verdict
	res.State = momentumState(rsi, hist)
	res.Why = momentumWhy(rsi, hist)
	res.RSI, res.MACDHistogram = &rsi, &hist
	res.Blocks = momentumBlocks(spec.Display, r.interval, rsi, hist)
	return res
}

// composeMomentum writes the reading part of a multi-asset card (the
// overview and every scan) from at least one OK asset: two lines per read
// asset (state + reason, checklist + bar time + freshness), one line per
// failed asset, the rule line, the shared-timeframe line (tf != ""), the
// results array, the counter header, its semaphore and the digest inputs.
//
// The digest inputs keep their meaning exactly: Deviation and confirmed come
// from the ranked (Binance) reads only, rankAsOf is their oldest bar;
// DataTime narrows to the oldest bar on the card (caller seeds it).
func composeMomentum(c *Card, assets []momentumAsset, tf string, now time.Time) {
	maxDev := 0
	anyOK := false
	for _, a := range assets {
		if a.status != statusOK {
			c.Facts = append(c.Facts, momentumFailLine(a))
			c.Results = append(c.Results, assetResult(a.spec.Display, a.status))
			continue
		}
		anyOK = true
		r := a.read
		fresh := momentumFreshness(a.spec.Source, r.interval, r.closeAt, now)
		c.Facts = append(c.Facts, momentumStateLine(a.spec.Display, r), momentumCheckLine(a.spec.Display, r, fresh))
		c.Results = append(c.Results, momentumResult(a.spec, r, fresh))
		if !r.closeAt.IsZero() && r.closeAt.Before(c.DataTime) {
			c.DataTime = r.closeAt
		}
		// Deviation drives the /digest priority rule and is CRYPTO-ONLY in
		// v1 — FX reads never push momentum to the top slot (see priority.go).
		// Only a confirmed read scores (momentumRankScore). The ranking's
		// freshness is the oldest RANKED (Binance) bar, not gold's.
		if a.spec.Source == srcBinance {
			if d := momentumRankScore(r); d > maxDev {
				maxDev = d
			}
			if r.verdict == momentumBullish || r.verdict == momentumBearish {
				c.confirmed = true
			}
			if !r.closeAt.IsZero() && (c.rankAsOf.IsZero() || r.closeAt.Before(c.rankAsOf)) {
				c.rankAsOf = r.closeAt
			}
		}
	}
	if anyOK {
		c.Facts = append(c.Facts, momentumRuleLine)
		// The colour follows BTC/ETH only (momentumHeader): a card mixing them
		// with other read assets says so, and so does one whose requested
		// BTC/ETH produced no reading (it stays neutral).
		asked, ranked, other := false, false, false
		for _, a := range assets {
			isRanked := a.spec.Source == srcBinance
			asked = asked || isRanked
			if a.status == statusOK {
				ranked = ranked || isRanked
				other = other || !isRanked
			}
		}
		switch {
		case ranked && other:
			c.Facts = append(c.Facts, momentumColourLine)
		case asked && !ranked:
			c.Facts = append(c.Facts, momentumColourNoneLine)
		}
		// Nothing the digest ranks (BTC/ETH) read: rankCandidate excludes the
		// card as no_ranked_read.
		c.noRankedRead = !ranked
		if tf != "" {
			c.Facts = append(c.Facts, "All assets read on closed "+tf+" candles")
		}
	}
	c.Deviation = maxDev
	c.Verdict, c.Emoji = momentumHeader(assets)
	c.Short = c.Verdict
}

// momentumDegraded fills a multi-asset card that produced no reading at all:
// one line and one results entry per asset.
func momentumDegraded(c *Card, assets []momentumAsset) {
	for _, a := range assets {
		c.Facts = append(c.Facts, momentumFailLine(a))
		c.Results = append(c.Results, assetResult(a.spec.Display, a.status))
	}
}

// ── context lines (not part of the reading) ──────────────────────────────────

const momentumContextPrefix = "Context, not part of the reading: "

// momentumVolumeContext — the last closed bar's volume against its 20-bar
// average. Not in the rule; no interpretation bands are claimed.
func momentumVolumeContext(what string, ratio float64) string {
	return fmt.Sprintf("%s%s %.2f× its 20-bar average", momentumContextPrefix, what, ratio)
}

// ppShown prints a return gap in percentage points, bounded to six runes.
func ppShown(v float64) string {
	switch {
	case v >= 999.95:
		return ">+999"
	case v <= -999.95:
		return "<-999"
	}
	return fmt.Sprintf("%+.1f", v)
}

// momentumRSLead is the fixed head of the RS context line, up to its first
// number. The hook finds the line by it and masks the numbers after it before
// hashing (hookMomentumRSRe) — they follow live prices, not the card's bar —
// so builder and hook read one constant and cannot drift apart.
const momentumRSLead = momentumContextPrefix + "ETH return minus BTC return incl. today, "

// momentumRSContext — ETH's return minus BTC's return (backend
// /api/v1/market/momentum; anchored on the current, still-forming UTC day on
// both sides). A return gap, not ETH's own move. "" when no window is served.
//
// The anchor makes this the one number on the card that follows LIVE prices
// while the card's data_as_of is the oldest CLOSED bar. It is labelled
// context for that reason, and the push hook does not count a moved digit
// here as a new reading (hook.go, hookMomentumRSRe).
func momentumRSContext(item MomentumItem) string {
	var parts []string
	if item.RS7D != nil {
		parts = append(parts, "7d "+ppShown(*item.RS7D)+" pp")
	}
	if item.RS30D != nil {
		parts = append(parts, "30d "+ppShown(*item.RS30D)+" pp")
	}
	if len(parts) == 0 {
		return ""
	}
	return momentumRSLead + strings.Join(parts, " · ")
}

// momentumLastBarRe matches the " · last bar Sep 15 08:00 UTC" segment that
// momentumCheckLine writes (momentumBarTime's "Jan 2 15:04 UTC" layout).
var momentumLastBarRe = regexp.MustCompile(` · last bar [A-Z][a-z]{2} \d{1,2} \d{2}:\d{2} UTC`)

// momentumAIFacts is the card's facts as the AI payload takes them: without
// the per-asset bar times and without the RS context line. The payload must
// carry no time stamps (its hash is the 5-minute AI cache key, and the model
// needs no clock); the freshness words ("market closed", "data delayed") stay
// — they are not stamps.
//
// The RS line goes for BOTH of those reasons. It is the one number on this
// card that is not as of the card's stamp (momentumRSContext: the backend
// anchors both sides on the still-forming UTC day), so a brief quoting it
// would attribute a live price to a closed-bar reading. And it follows those
// prices at 0.1 pp, which would move the payload hash — the cache key — on
// most sweeps: two LLM calls where one answer would do, and a digest and a
// top of the SAME sweep narrating the same market in two different texts.
func momentumAIFacts(facts []string) []string {
	out := make([]string, 0, len(facts))
	for _, f := range facts {
		if strings.HasPrefix(f, momentumRSLead) {
			continue
		}
		out = append(out, momentumLastBarRe.ReplaceAllString(f, ""))
	}
	return out
}
