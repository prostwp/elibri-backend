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
// a CONFIRMED regime → background (macro, S/R, volatility) → the one-line
// disclosure that levels and EMAs run on spliced contracts (goldSplicedLine)
// → the setup structure. The source is the footer. In a contract roll window
// (gold_roll.go) a roll line follows the price line and nothing places the 1h
// price against daily data.
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

// goldSplicedLine is the one-sentence disclosure that every level-bearing read
// on this card runs on the continuous GC=F series, which is not adjusted at
// contract rolls: a level set before the current contract sits where the
// previous contract printed it. The size of the shift is not on the card (it
// changes with every roll); docs/demobot-http.md carries the measured numbers
// with their source.
const goldSplicedLine = "S/R, EMAs and regime use spliced GC=F contracts; levels older than the current contract are shifted by rolls"

// goldNoIdeaRoll: in a roll window the trigger (a day range edge) and the
// reference level are prices of one contract and the 1h price of another, so
// the card names no structure between them.
const goldNoIdeaRoll = "No setup structure during a contract roll: the 1h price and the day range are on different contracts"

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
	// roll: which contract each GC=F bar is on (gold_roll.go). In a window
	// sup/res were picked against the last daily close, not px (GoldCard).
	roll GoldRoll
}

// inRollWindow reports whether the card's 1h price and its daily data are
// prices of two different contracts — whichever of the two is the later one
// (gold.roll.reason says). Only an ESTABLISHED window counts: an unknown
// state reads as the card did before the check (see goldCardFrom).
func (in goldInputs) inRollWindow() bool {
	return in.hasPx && in.roll.State == goldRollWindow &&
		in.roll.DailyContract != nil && in.roll.HourlyContract != nil &&
		in.roll.CurrentContract != nil && in.roll.NextContract != nil
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
	// Contract roll window (gold_roll.go): the 1h price is a price of one
	// contract, the day range, levels and EMAs of another. Nothing below
	// places the one against the other — not the price line, not the
	// scenario and invalidation tails, not the nearest levels, not the
	// structure. The regime itself is read on daily bars alone and stands.
	//
	// An UNKNOWN roll state reads exactly as before the check existed. It is
	// dominated by causes unrelated to a roll (a failed or refused contract
	// request, the last bar before a weekend printing apart from its
	// contract), while a roll window is about two trading days in two months;
	// withholding the price placement on every such hour would drop a
	// comparison that is right almost always, to guard hours where the check
	// did establish nothing. gold.roll.state says "unknown", so a machine
	// reader is not told "no roll" either.
	//
	// Out of a window the day range is placed against the 1h price even when
	// it was unwound to an earlier daily bar (up to dayUnwindCap inside days):
	// a roll between that bar and the last closed one would make it a range
	// of the old contract. Checked on the research's saved 2-year GC=F daily
	// series (gold_roll_data_test.go): of 502 daily points, 62 took the range
	// from an earlier bar, and in none of them did a roll — 3 established, 7
	// candidate dates — fall between the two. It stays as it is; the test
	// pins the count.
	window := in.inRollWindow()
	pos := ""
	if in.hasPx && !window {
		pos = in.levels.positionOf(in.px)
	}
	against := (trend.State == trendUp && in.macro.State == goldPressure) ||
		(trend.State == trendDown && in.macro.State == goldSupport)

	// The verdict DESCRIBES the regime; it does not call the day. The Этап 5
	// run found no next-day edge the sample could detect, so no "bias".
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
	switch {
	case window:
		c.Facts = append(c.Facts, goldRollPriceLine(in.px, in.pxAt, stale, *in.roll.HourlyContract), goldRollLine(in.roll))
	case in.hasPx:
		c.Facts = append(c.Facts, goldPriceLine(in.px, in.pxAt, pos, stale))
	default:
		c.Facts = append(c.Facts, goldNoPriceLine)
	}

	// 2-3. The day range, where it comes from, and the two scenarios. In a
	// roll window the scenarios lose their "already above/below" tail: that
	// tail places the 1h price against the range.
	c.Facts = append(c.Facts, in.levels.rangeLine())
	scenarios := in.levels.scenarios(in.px, in.hasPx && !window)
	c.Facts = append(c.Facts, scenarios...)

	// 4. What invalidates the regime — confirmed states only: under a header
	// that claims no direction there is nothing to invalidate.
	inv, invLevel, invSide := "", (*float64)(nil), ""
	if lv, ok := trend.Levels.(TrendLevels); ok && confirmed && lv.Invalidation != nil && *lv.Invalidation > 0 {
		inv = goldInvalidationLine(trend.State, *lv.Invalidation, lv.InvalidationSide, in.px)
		if window { // the "already beyond" tail compares the 1h price with a daily level
			inv = goldInvalidationPlain(trend.State, *lv.Invalidation, lv.InvalidationSide)
		}
		invLevel, invSide = lv.Invalidation, lv.InvalidationSide
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
		levels := goldKeyLevelsLine(in.sup, in.res)
		if window {
			levels = goldKeyLevelsLineFrom(goldLevelsFromDailyClose, in.sup, in.res)
		}
		if levels != "" {
			c.Facts = append(c.Facts, levels)
		}
	}
	if in.vol != "" {
		c.Facts = append(c.Facts, "Volatility: "+in.vol)
	}
	c.Facts = append(c.Facts, goldSplicedLine)

	// 6. The setup structure closes the card: the same numbers said as one
	// shape, plus the line that says what the history run measured. It is
	// added AFTER the facts above, never instead of any of them.
	idea, ideaLines := goldIdeaOf(in, confirmed, invLevel, invSide)
	c.Facts = append(c.Facts, ideaLines...)

	// The weekend notice leads, as on every Yahoo card. KNOWN LIMITATION:
	// weekend only — COMEX holidays and the daily break are not modelled (no
	// calendar source), so the price line's own date and the stale flag are
	// what show a reader how old the number is.
	if !isForexOpen(in.now) {
		c.Facts = append([]string{goldClosedBanner}, c.Facts...)
	}

	c.Gold = goldReadoutOf(in, confirmed, stale, pos)
	c.Gold.Idea = idea // null whenever the card named no structure
	if in.hasPx {      // a degraded card carries no blocks
		c.Blocks = goldBlocksOf(in, scenarios, inv, pos, stale)
	}
	return c
}

// goldRollPriceLine is the price line in a roll window: the price, its time
// and the contract it is on — and no place against the day range, which is a
// range of the other contract. Worst case 94 runes (five-digit price, stale).
func goldRollPriceLine(px float64, at time.Time, stale bool, contract string) string {
	s := fmt.Sprintf("Last closed 1h price %s at %s UTC — ", goldPx(px), at.UTC().Format("2006-01-02 15:04"))
	if stale {
		s += fmt.Sprintf("stale (over %.0fh old), ", goldPriceStale.Hours())
	}
	return s + "on contract " + contract
}

// goldRollLine is the one fact that a contract roll is under way: from which
// contract to which, which series is on which, and that the price is not
// placed. Contract codes only — no spread, no date that would go stale on the
// card. 95 runes: every code is five characters.
func goldRollLine(r GoldRoll) string {
	return fmt.Sprintf("Contract roll %s → %s: 1h price on %s, day range on %s; price not placed against it",
		*r.CurrentContract, *r.NextContract, *r.HourlyContract, *r.DailyContract)
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
	if goldBeyond(px, inv, side) {
		return goldInvalidationHead(state, inv, side) + "; last 1h close already " + side + " it" // worst case 106 runes
	}
	return goldInvalidationPlain(state, inv, side)
}

// goldInvalidationHead is the level and what it invalidates.
func goldInvalidationHead(state string, inv float64, side string) string {
	return fmt.Sprintf("A closed %s candle %s %s invalidates the daily %s reading",
		goldDailySpec.Interval, side, goldPx(inv), dirNoun(state))
}

// goldInvalidationPlain is the line without any word about the 1h price —
// the form a roll window uses whatever that price is.
func goldInvalidationPlain(state string, inv float64, side string) string {
	prep := "under"
	if side == "above" {
		prep = "over"
	}
	return goldInvalidationHead(state, inv, side) + " (1 ATR " + prep + " the EMA cluster)"
}

// goldBeyond reports whether a price is STRICTLY beyond a level on the given
// side. Compared as PRINTED (goldPx, then parsed back): two numbers that print
// the same are never worded — or counted — as one beyond the other, because
// math.Round and "%.2f" disagree on exact half-cents. Sitting exactly on a
// level is not beyond it, as everywhere else on this card.
func goldBeyond(px, level float64, side string) bool {
	pc, _ := strconv.ParseFloat(goldPx(px), 64)
	lc, _ := strconv.ParseFloat(goldPx(level), 64)
	return (side == dayBelow && pc < lc) || (side == dayAbove && pc > lc)
}

// ── the setup structure (stage 2, 2026-09-16) ────────────────────────────────
//
// Nothing below is a new rule or a new number. The trigger is the day range
// edge the scenarios already classify against, the invalidation level is the
// trend card's own (EMA cluster ± 1 ATR), the reference level is the S/R
// clusterizer's nearest level on the other side of price. The block only says
// which level stands against which — the SHAPE of the setup.
//
// It is deliberately not a call to act. The Этап 5 history run found no edge
// this sample could detect in this read (Отчёт_прогона_золотой_агент.md) — a
// limit of the measurement, not a finding about the market — so the block
// carries that sentence at that strength on the card itself, and the
// banned-word test keeps trade vocabulary out of every path.

// Machine values of gold.idea.state.
const (
	goldIdeaArmed      = "armed"                // no level taken by the last 1h close
	goldIdeaTriggerHit = "trigger_reached"      // that close is already beyond the trigger
	goldIdeaInvalidHit = "invalidation_reached" // that close is already beyond the invalidation level
)

// Machine values of gold.idea.*.basis — where each level comes from.
const (
	goldBasisRangeHigh = "day_range_high"
	goldBasisRangeLow  = "day_range_low"
	goldBasisEMAATR    = "ema_cluster_atr"
)

// goldIdeaDisclaimer states the measurement the block rests on, and states it
// as the WEAK claim on purpose: the ten-year run (Отчёт_прогона_золотой_агент,
// section 7) resolves effects of about 12 pp and larger, so what it found is
// "no edge this sample could detect" — never a proof that no edge exists.
//
// "…showed no edge over the baseline" was the earlier wording and is wrong in
// the flattering direction: it turns a limit of the measurement into a
// finding about the market. The per-half numbers and the resolution limit are
// in the docs; the card carries the claim only at the strength it was earned.
const goldIdeaDisclaimer = "Structure, not a forecast: 10 years of history showed no edge this sample could detect"

// Why the card names no structure. An unconfirmed regime never grows one out
// of thin air: rule 1 and 2 of the conflict priority hold here too.
const (
	goldNoIdeaUnconfirmed = "No setup structure without a confirmed regime and a last closed 1h price"
	goldNoIdeaNoRange     = "No setup structure: no day range to trigger against"
	goldNoIdeaNoInv       = "No setup structure: this regime read carries no invalidation level"
)

// goldIdeaOf builds the structure and the lines that word it. inv/invSide are
// the trend card's own invalidation level, already filtered by goldCardFrom to
// a confirmed regime — passed in rather than recomputed so the block and the
// invalidation line can never print two different numbers.
func goldIdeaOf(in goldInputs, confirmed bool, inv *float64, invSide string) (*GoldIdea, []string) {
	switch {
	case !confirmed:
		return nil, []string{goldNoIdeaUnconfirmed}
	case in.inRollWindow():
		return nil, []string{goldNoIdeaRoll}
	case !in.levels.Defined:
		return nil, []string{goldNoIdeaNoRange}
	case inv == nil || *inv <= 0 || invSide == "":
		return nil, []string{goldNoIdeaNoInv}
	}
	// The trigger is the edge on the side the regime reads; the reference
	// level is then the nearest cluster on the other side of price. Confirmed
	// means trendUp or trendDown, so the two branches are the whole space.
	trigger := GoldIdeaLevel{Level: in.levels.Low, Side: dayBelow, Basis: goldBasisRangeLow}
	edge, ref, kind, refSide := "day range low", in.res, "resistance", dayAbove
	if in.trend.State == trendUp {
		trigger = GoldIdeaLevel{Level: in.levels.High, Side: dayAbove, Basis: goldBasisRangeHigh}
		edge, ref, kind, refSide = "day range high", in.sup, "support", dayBelow
	}
	idea := &GoldIdea{
		State:        goldIdeaArmed,
		Trigger:      trigger,
		Invalidation: GoldIdeaLevel{Level: *inv, Side: invSide, Basis: goldBasisEMAATR},
	}
	// The state only mirrors what the card already says elsewhere, and each
	// branch borrows that line's OWN comparison, so the two can never disagree:
	// the invalidation level at the printed tick (goldBeyond, like the
	// invalidation tail), the trigger raw (positionOf, like the stage-1
	// scenario tail — a close 0.004 above an edge that prints identically is
	// already above it there, and is trigger_reached here). Invalidation is
	// checked first: it ends the reading the trigger belongs to.
	switch {
	case goldBeyond(in.px, *inv, invSide):
		idea.State = goldIdeaInvalidHit
	case in.levels.positionOf(in.px) == trigger.Side:
		idea.State = goldIdeaTriggerHit
	}

	// The reference level is the nearest cluster on the other side OF PRICE, so
	// the words say "below price" / "above price" and never "opposite the
	// trigger": once price has taken the trigger edge, the nearest cluster on
	// its far side can sit beyond the trigger too (uptrend, range 4293.00 –
	// 4396.80, last close 4400.10, support 4398.00 — above the trigger). An
	// absent level is stated, never padded with the day range, which is a
	// different thing measured a different way.
	second := fmt.Sprintf("Invalidated by a closed %s candle %s %s; ", goldDailySpec.Interval, invSide, goldPx(*inv))
	if ref == nil {
		second += "nothing clustered " + refSide + " price"
	} else {
		idea.ReferenceLevel = &GoldIdeaRef{Level: ref.Raw, Kind: kind, Class: srClassKey(srClassOf(ref.Touches))}
		second += "nearest level " + refSide + " price: " + kind + " " + goldPx(ref.Raw) // worst case 96 runes
	}
	return idea, []string{
		fmt.Sprintf("Setup structure: a daily close %s %s (%s) is the trigger", trigger.Side, goldPx(trigger.Level), edge),
		second,
		goldIdeaDisclaimer,
	}
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
	return goldKeyLevelsLineFrom("Nearest levels: ", sup, res)
}

// goldLevelsFromDailyClose heads the levels line in a roll window, where the
// levels are the nearest to the last daily close — a price of the same
// contract as every level — rather than to the 1h price of the next one.
// Worst case 109 runes (five-digit levels, two-digit pivot counts).
const goldLevelsFromDailyClose = "Nearest to 1d close: "

// goldKeyLevelsLineFrom is goldKeyLevelsLine under a given head.
func goldKeyLevelsLineFrom(head string, sup, res *SRLevel) string {
	switch {
	case sup == nil && res == nil:
		return ""
	case res == nil:
		return fmt.Sprintf("%ssupport %s (%s), nothing clustered above", head, goldPx(sup.Raw), classPivots(*sup))
	case sup == nil:
		return fmt.Sprintf("%sresistance %s (%s), nothing clustered below", head, goldPx(res.Raw), classPivots(*res))
	}
	return fmt.Sprintf("%ssupport %s (%s) · resistance %s (%s)",
		head, goldPx(sup.Raw), classPivots(*sup), goldPx(res.Raw), classPivots(*res))
}

// goldBlocksOf is the content-ready form. what_happened is a SNAPSHOT: the
// agent keeps no previous state, so it cannot say what changed — the sentence
// says so itself.
func goldBlocksOf(in goldInputs, scenarios []string, inv, pos string, stale bool) *ContentBlocks {
	tf := goldDailySpec.Interval
	what := fmt.Sprintf("Snapshot, not an event: %s regime %s; last 1h close %s", tf, in.trend.Short, goldPx(in.px))
	if in.inRollWindow() {
		what += " on " + *in.roll.HourlyContract + " (contract roll)"
	} else if pos != "" {
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
	roll := in.roll
	if roll.State == "" { // a builder that did not run the check established nothing
		r := goldRollNotChecked
		roll = GoldRoll{State: goldRollUnknown, Reason: &r}
	}
	g.Roll = &roll
	if in.macro.State != "" {
		st := in.macro.State
		g.MacroBackdrop = &st
		g.MacroLamps = goldLampCounts(in.macro.Macro)
		g.MacroAsOf = goldMacroAsOf(in.macro.Macro)
	}
	return g
}
