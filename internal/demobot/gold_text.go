package demobot

// gold_text.go — everything the Gold card SAYS (stage 1, 2026-09-15).
//
// Presentation only. The rules live elsewhere and are untouched: the regime is
// the Trend Agent's state machine on daily bars, the day range is
// goldDayLevelsOf (dayUnwindCap), the nearest levels are the S/R clusterizer,
// the invalidation level is invalidationFor, the macro backdrop is the gold
// lamp model, volatility is the ATR agent.
//
// Line order: regime and where the last closed 1h price sits → the day range
// and where it comes from → two conditional day scenarios → what invalidates
// a CONFIRMED regime → background (macro, S/R, volatility). The source is the
// footer.
//
// Honesty rules for every line below:
//   - the price is the close of the last CLOSED 1h bar and is named that way,
//     with its full date and time — never "now";
//   - a stale price is flagged by a fixed threshold (goldPriceStale), never by
//     a running age: the body must not change from one request to the next
//     while the data stays the same (push hook);
//   - scenarios classify the day on a daily close; no targets, no
//     probabilities, no "will";
//   - invalidation exists only for a confirmed regime, named with its
//     direction and on a closed 1d candle — the level is the EMA cluster ±
//     1 ATR, not swing structure;
//   - every line (verdict, facts, each blocks field) fits goldFactMaxRunes.

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// goldFactMaxRunes is the readability budget for one line of gold text.
const goldFactMaxRunes = 110

// goldLimitations is the blocks.limitations sentence: the instrument and what
// the agent does not do.
const goldLimitations = "COMEX GC=F futures, not spot XAUUSD; describes the period, not a forecast of the day"

// goldNoPriceLine is the price slot when the intraday series is unavailable.
const goldNoPriceLine = "Intraday price feed is down: the last 1h close cannot be placed against the day range"

// Machine values of gold.price_freshness.
const (
	goldPriceOnTime = "on_time"
	goldPriceStaleV = "stale"
)

// goldInputs is everything the card is built from — gathered by GoldCard, so
// goldCardFrom stays pure (no network, no clock but in.now).
type goldInputs struct {
	trend     Card // TrendCard on goldDailySpec: State, Verdict, Short, Levels, Blocks
	levels    goldDayLevels
	dailyAsOf time.Time // close of the last closed daily bar
	px        float64   // close of the last closed 1h bar
	pxAt      time.Time // its close time
	hasPx     bool
	macro     Card     // MacroAssetCard(gold): State "" = no read
	sup, res  *SRLevel // nearest clustered support below / resistance above px
	vol       string   // volShortLine, "" when the ATR read is unavailable
	now       time.Time
}

// goldCardFrom is the pure half of GoldCard.
//
// Conflict priority is fixed (spec section 6), first rule wins:
//  1. no price          → no direction, whatever else is alive
//  2. regime unconfirmed → context and levels, no direction, however loud macro is
//  3. regime confirmed, macro against → regime plus an explicit conflict line
//  4. regime confirmed, macro agreeing or absent → regime plain
func goldCardFrom(in goldInputs) Card {
	trend := in.trend
	c := Card{
		Agent:      "Gold Agent",
		ShortName:  "Gold",
		Asset:      goldDailySpec.Display,
		AssetKey:   goldDailySpec.Key,
		Command:    keyGold,
		HowItWorks: goldHow,
		// data_as_of stays the daily close: the regime and the day range —
		// the headline — come from it. The other parts carry their own
		// stamps in the gold readout (daily_as_of, price_as_of, macro_as_of).
		DataTime:   in.dailyAsOf,
		State:      trend.State,
		SourceNote: goldSourceNote,
		// The body reads the daily bars, the hourly price and its age against
		// the clock, the macro payload and the weekend clock; no one stamp
		// versions it — no validator (docs "Caching").
		noValidator: true,
	}

	directional := trend.State == trendUp || trend.State == trendDown
	confirmed := in.hasPx && directional
	stale := in.hasPx && in.now.Sub(in.pxAt) > goldPriceStale
	pos := ""
	if in.hasPx {
		pos = in.levels.positionOf(in.px)
	}
	against := (trend.State == trendUp && in.macro.State == goldPressure) ||
		(trend.State == trendDown && in.macro.State == goldSupport)

	// The verdict DESCRIBES the regime; it does not call the day. The Этап 5
	// run found no next-day edge in the regime read, so no "bias".
	switch {
	case !in.hasPx:
		c.Emoji = emojiNeutral
		c.Verdict = "Intraday price unavailable — no direction claimed"
		c.Short = "no intraday price"
		c.Offline = true
		c.Status = statusSourceOffline
	case !confirmed:
		c.Emoji = emojiNeutral
		c.Verdict = "Daily regime: not confirmed — no direction claimed"
		c.Short = "regime unconfirmed"
	case trend.State == trendUp:
		c.Emoji, c.Short = emojiBull, "regime uptrend"
		c.Verdict = "Daily regime: confirmed UPTREND"
	default:
		c.Emoji, c.Short = emojiBear, "regime downtrend"
		c.Verdict = "Daily regime: confirmed DOWNTREND"
	}

	// 1. Regime and where the price sits. A confirmed regime is fully stated
	// by the verdict; an unconfirmed one carries its reason. Without a price
	// a confirmed regime is reported, but not as a direction.
	switch {
	case !in.hasPx && directional:
		c.Facts = append(c.Facts, "Daily regime reads "+lowerFirst(trend.Verdict)+"; no direction stated without a 1h price")
	case !confirmed:
		c.Facts = append(c.Facts, "Regime: "+lowerFirst(trend.Verdict))
	}
	if in.hasPx {
		c.Facts = append(c.Facts, goldPriceLine(in.px, in.pxAt, pos, stale))
	} else {
		c.Facts = append(c.Facts, goldNoPriceLine)
	}

	// 2-3. The day range, where it comes from, and the two scenarios.
	c.Facts = append(c.Facts, in.levels.rangeLine())
	scenarios := in.levels.scenarios(in.px, in.hasPx)
	c.Facts = append(c.Facts, scenarios...)

	// 4. What invalidates the regime — confirmed states only: under a header
	// that claims no direction there is nothing to invalidate.
	inv := ""
	if lv, ok := trend.Levels.(TrendLevels); ok && confirmed && lv.Invalidation != nil && *lv.Invalidation > 0 {
		inv = goldInvalidationLine(trend.State, *lv.Invalidation, lv.InvalidationSide, in.px)
		c.Facts = append(c.Facts, inv)
		c.Levels = lv
	}

	// 5. Background: macro (an absent read is stated), S/R, volatility.
	c.Facts = append(c.Facts, goldMacroLine(in.macro))
	// Spec rule 3 only: with no price rule 1 wins (no direction), and a
	// conflict with a direction the card does not claim is not stated.
	if against && confirmed { // the regime plus an explicit conflict line
		c.Facts = append(c.Facts, "Macro backdrop conflicts with the daily "+dirNoun(trend.State)+" reading; both stand as read")
	}
	if in.hasPx {
		if l := goldKeyLevelsLine(in.sup, in.res); l != "" {
			c.Facts = append(c.Facts, l)
		}
	}
	if in.vol != "" {
		c.Facts = append(c.Facts, "Volatility: "+in.vol)
	}

	// The weekend notice leads, as on every Yahoo card. KNOWN LIMITATION:
	// weekend only — COMEX holidays and the daily break are not modelled (no
	// calendar source), so the price line's own date and the stale flag are
	// what show a reader how old the number is.
	if !isForexOpen(in.now) {
		c.Facts = append([]string{goldClosedBanner}, c.Facts...)
	}

	c.Gold = goldReadoutOf(in, confirmed, stale, pos)
	if in.hasPx { // a degraded card carries no blocks
		c.Blocks = goldBlocksOf(in, scenarios, inv, pos, stale)
	}
	return c
}

// goldPriceLine: "Last closed 1h price 4354.90 at 2026-09-15 03:00 UTC —
// inside the day range", with "stale (over 6h old)" past goldPriceStale.
func goldPriceLine(px float64, at time.Time, pos string, stale bool) string {
	s := fmt.Sprintf("Last closed 1h price %s at %s UTC", goldPx(px), at.UTC().Format("2006-01-02 15:04"))
	var tail []string
	if stale {
		tail = append(tail, fmt.Sprintf("stale (over %.0fh old)", goldPriceStale.Hours()))
	}
	if pos != "" {
		tail = append(tail, pos+" the day range")
	}
	if len(tail) > 0 {
		s += " — " + strings.Join(tail, ", ")
	}
	return s
}

// candleDate is the date of the bar the range came from.
func (d goldDayLevels) candleDate() string {
	return time.Unix(d.BarTime, 0).UTC().Format("2006-01-02")
}

// rangeLine is the day range with where it comes from: the last closed daily
// candle, or the enclosing one when the later days sat inside it.
func (d goldDayLevels) rangeLine() string {
	if !d.Defined {
		return fmt.Sprintf("Day range undefined: more than %d nested inside days, no day levels to give", dayUnwindCap)
	}
	rng := goldPx(d.Low) + " – " + goldPx(d.High)
	if d.Unwound == 0 {
		return fmt.Sprintf("Day range %s: high/low of the closed %s candle of %s", rng, goldDailySpec.Interval, d.candleDate())
	}
	return fmt.Sprintf("Day range %s: high/low of the %s candle of %s (%s after it)",
		rng, goldDailySpec.Interval, d.candleDate(), plural(d.Unwound, "inside day", "inside days"))
}

// scenarios are the two conditional sentences: a daily close beyond an edge
// CLASSIFIES the day as a break — a description of the close, not a promise of
// movement. When the last 1h close is already beyond an edge, that side says
// so: the side is taken intraday, the close is not in yet. An undefined range
// has no scenarios ([] — never null in JSON).
func (d goldDayLevels) scenarios(px float64, hasPx bool) []string {
	if !d.Defined {
		return []string{}
	}
	up := fmt.Sprintf("A daily close above %s classifies the day as an upside break", goldPx(d.High))
	down := fmt.Sprintf("A daily close below %s classifies the day as a downside break", goldPx(d.Low))
	if hasPx {
		switch d.positionOf(px) {
		case dayAbove:
			up += "; the last 1h close is already above it"
		case dayBelow:
			down += "; the last 1h close is already below it"
		}
	}
	return []string{up, down}
}

// goldInvalidationLine words the trend card's own level with its direction:
// an uptrend is invalidated below the EMA cluster, a downtrend above it. It is
// only built for a confirmed regime, i.e. with a 1h price (px).
//
// The line COMPARES a daily close with the level, so the level prints through
// goldPx (tick resolution): trimFloat printed 4293.40 as "4293", and a close of
// 4293.20 — above the printed number — would already invalidate by the rule.
//
// Like the day scenarios, it says when the last 1h close is already beyond
// the level (strictly: at the level is not beyond). That tail replaces the
// "(1 ATR … the EMA cluster)" note, which does not fit beside it within
// goldFactMaxRunes; the level's origin is in the docs and in levels.
func goldInvalidationLine(state string, inv float64, side string, px float64) string {
	s := fmt.Sprintf("A closed %s candle %s %s invalidates the daily %s reading",
		goldDailySpec.Interval, side, goldPx(inv), dirNoun(state))
	// Compared as PRINTED (goldPx, then parsed back): two numbers that print the
	// same are never worded as one beyond the other — math.Round and "%.2f"
	// disagree on exact half-cents, so the comparison uses the print itself.
	pc, _ := strconv.ParseFloat(goldPx(px), 64)
	ic, _ := strconv.ParseFloat(goldPx(inv), 64)
	if (side == "below" && pc < ic) || (side == "above" && pc > ic) {
		return s + "; last 1h close already " + side + " it" // worst case 106 runes
	}
	prep := "under"
	if side == "above" {
		prep = "over"
	}
	return s + " (1 ATR " + prep + " the EMA cluster)"
}

// goldLampCounts counts the VOTING lamps of the gold macro model by their
// contribution for gold. nil without a readout. The gold lamp itself never
// votes and is not counted.
func goldLampCounts(m *MacroReadout) *GoldMacroLamps {
	if m == nil {
		return nil
	}
	n := &GoldMacroLamps{}
	for _, l := range m.Lamps {
		if !l.Voting {
			continue
		}
		switch l.Contribution {
		case contribPositive:
			n.For++
		case contribNeutral:
			n.Neutral++
		case contribNegative:
			n.Against++
		}
	}
	return n
}

// goldMacroAsOf is the oldest as_of among the voting lamps — the stalest input
// of the macro read on this card. nil when none is parseable. Taken from the
// lamps, never from the backend's captured_at (its request time).
func goldMacroAsOf(m *MacroReadout) *string {
	if m == nil {
		return nil
	}
	var oldest time.Time
	for _, l := range m.Lamps {
		if !l.Voting {
			continue
		}
		if t, ok := lampStamp(l.AsOf); ok && (oldest.IsZero() || t.Before(oldest)) {
			oldest = t
		}
	}
	if oldest.IsZero() {
		return nil
	}
	s := oldest.Format(time.RFC3339)
	return &s
}

// goldMacroLine is the macro backdrop with its basis. The word is chosen by
// the gold model's WEIGHTED score bands (DXY, rates, VIX and SPX carry
// different weights), not by a lamp majority — so the line prints that score,
// taken from the macro card's own readout (never recomputed), and the lamp
// counts only as its composition. With only DXY for gold the score is 40 and
// the word "mixed", beside 3 lamps against. The middle band is worded "mixed",
// as on both macro cards.
func goldMacroLine(m Card) string {
	if m.State == "" { // source offline, unknown, or too few voting lamps
		return "Macro backdrop: no gold read available"
	}
	word := m.State
	if word == goldNeutral {
		word = "mixed"
	}
	s := "Macro backdrop: " + word + " for gold"
	if m.Macro != nil && m.Macro.RuleScore != nil {
		s += fmt.Sprintf(" — weighted gold score %d/100", *m.Macro.RuleScore)
	}
	if n := goldLampCounts(m.Macro); n != nil {
		s += fmt.Sprintf(" (lamps: %d for, %d neutral, %d against)", n.For, n.Neutral, n.Against)
	}
	return s
}

// goldKeyLevelsLine is the nearest clustered support and resistance with the
// S/R card's class words ("single swing, 1 pivot"), so the two cards never
// describe one level two ways. "" when nothing clustered on either side.
func goldKeyLevelsLine(sup, res *SRLevel) string {
	switch {
	case sup == nil && res == nil:
		return ""
	case res == nil:
		return fmt.Sprintf("Nearest levels: support %s (%s), nothing clustered above", goldPx(sup.Raw), classPivots(*sup))
	case sup == nil:
		return fmt.Sprintf("Nearest levels: resistance %s (%s), nothing clustered below", goldPx(res.Raw), classPivots(*res))
	}
	return fmt.Sprintf("Nearest levels: support %s (%s) · resistance %s (%s)",
		goldPx(sup.Raw), classPivots(*sup), goldPx(res.Raw), classPivots(*res))
}

// goldBlocksOf is the content-ready form. what_happened is a SNAPSHOT: the
// agent keeps no previous state, so it cannot say what changed — the sentence
// says so itself.
func goldBlocksOf(in goldInputs, scenarios []string, inv, pos string, stale bool) *ContentBlocks {
	tf := goldDailySpec.Interval
	what := fmt.Sprintf("Snapshot, not an event: %s regime %s; last 1h close %s", tf, in.trend.Short, goldPx(in.px))
	if pos != "" {
		what += " " + pos + " the day range"
	} else {
		what += ", no day range defined"
	}
	if stale {
		what += " (stale)"
	}
	regime := fmt.Sprintf("Local gold regime: %s · %s", in.trend.Short, tf)
	if in.trend.Blocks != nil && in.trend.Blocks.Regime != "" {
		regime = "Local gold regime: " + in.trend.Blocks.Regime
	}
	b := &ContentBlocks{
		WhatHappened: what,
		WhyLevel:     in.levels.whyLevel(),
		Scenarios:    scenarios,
		Regime:       regime,
		Limitations:  goldLimitations,
	}
	if inv != "" {
		s := inv
		b.Invalidates = &s
	}
	return b
}

// whyLevel says what the card's levels are made of.
func (d goldDayLevels) whyLevel() string {
	const sr = "; S/R = swing-pivot cluster means"
	switch {
	case !d.Defined:
		return fmt.Sprintf("No day range: more than %d nested inside days", dayUnwindCap) + sr
	case d.Unwound == 0:
		return fmt.Sprintf("Day range = high/low of the closed %s candle of %s", goldDailySpec.Interval, d.candleDate()) + sr
	}
	return fmt.Sprintf("Day range = high/low of the %s candle of %s (%s after it)",
		goldDailySpec.Interval, d.candleDate(), plural(d.Unwound, "inside day", "inside days")) + sr
}

// goldReadoutOf is the machine readout (envelope "gold"): per-part stamps and
// the values the card places against each other, at raw precision.
func goldReadoutOf(in goldInputs, confirmed, stale bool, pos string) *GoldReadout {
	g := &GoldReadout{
		Regime:    in.trend.State,
		Confirmed: confirmed,
		DailyAsOf: in.dailyAsOf.UTC().Format(time.RFC3339),
	}
	if in.hasPx {
		at := in.pxAt.UTC().Format(time.RFC3339)
		fr := goldPriceOnTime
		if stale {
			fr = goldPriceStaleV
		}
		px := in.px
		g.PriceAsOf, g.PriceFreshness, g.Price = &at, &fr, &px
		if pos != "" {
			p := pos
			g.PricePosition = &p
		}
	}
	if d := in.levels; d.Defined {
		g.DayRange = &GoldDayRange{High: d.High, Low: d.Low, CandleDate: d.candleDate(), InsideDaysAfter: d.Unwound}
	}
	if in.macro.State != "" {
		st := in.macro.State
		g.MacroBackdrop = &st
		g.MacroLamps = goldLampCounts(in.macro.Macro)
		g.MacroAsOf = goldMacroAsOf(in.macro.Macro)
	}
	return g
}
