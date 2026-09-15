package demobot

// trend_text.go — everything the Trend card SAYS.
//
// Presentation only. The state machine (classifyTrend, the structure gate in
// structureDemotion, the ADX thresholds) lives in agents.go and is the single
// definition of every rule.
//
// One enumeration by construction: the confirm conditions are listed in
// exactly ONE place — checks(), rendered as the "Why:" checklist, always the
// same four items, each evaluated by the rule's own functions. Every other
// sentence (hold line, confirm line, scenarios, why_level, landing
// conclusion) REFERS to the checklist ("all four conditions") and may name
// failing items, but never re-lists a subset of conditions as if it were the
// rule. A partial list is how earlier wording promised confirmations the gate
// would refuse.
//
// Honesty rules for every line below:
//   - analytics only: no advice, no targets, no probabilities, no "price will";
//   - every number is one the read computed, printed through the same helper
//     wherever it appears (adxShown for ADX, trimFloat for prices);
//   - a resulting CONFIRMED state is named only together with "all four
//     conditions"; fall-back scenarios name only states the rule fully
//     determines (ADX alone decides flat vs grey);
//   - every line — facts, content blocks, landing conclusion — fits
//     trendFactMaxRunes.

import (
	"fmt"
	"math"
	"strings"
)

// trendFactMaxRunes is the readability budget for one line of trend text.
const trendFactMaxRunes = 110

// adxShown is the only way trend text prints ADX: floored to one decimal.
//
// The text asserts comparisons against integer thresholds ("< 25", "≥ 25"),
// so the number is rounded DOWN, never to nearest: at %.0f an ADX of 19.6
// printed "ADX 20 < 20", and simply adding a decimal only moves the problem
// (19.99 → "20.0 < 20"). Flooring is provably safe in both directions for an
// integer threshold — a value below it never prints at or above it, and a
// value at or above it never prints below it.
func adxShown(adx float64) string {
	return fmt.Sprintf("%.1f", math.Floor(adx*10)/10)
}

// candleWord is the timeframe as it sits in front of "candle"/"closes".
func candleWord(tf string) string {
	if tf == "" {
		return "agent-timeframe"
	}
	return tf
}

// pctAway is |level − price| as a percentage of price, one decimal; distances
// under 0.05% print "<0.1%" rather than a "0.0%" that reads as "at the level".
//
// Bounded by construction so no line can blow the length budget: anything
// from 999.95% up (where "%.1f" would print "1000.0") prints ">999%", so the
// widest output is six characters and never exponent notation. Real series
// never get near it; a tiny price against a huge EMA would.
func pctAway(price, level float64) string {
	if price == 0 {
		return "n/a"
	}
	p := math.Abs(level-price) / math.Abs(price) * 100
	switch {
	case math.IsNaN(p):
		return "n/a"
	case p < 0.05:
		return "<0.1%"
	case p >= 999.95: // includes +Inf
		return ">999%"
	}
	return fmt.Sprintf("%.1f%%", p)
}

// signedPct is the distance from price to a level with its direction:
// "+0.6%" (level above price), "-2.1%" (below), "<0.1%" (practically at it),
// ">+999%" / ">-999%" beyond the bound.
func signedPct(price, level float64) string {
	s := pctAway(price, level)
	if strings.HasPrefix(s, "<") || s == "n/a" {
		return s
	}
	sign := "-"
	if level > price {
		sign = "+"
	}
	if strings.HasPrefix(s, ">") {
		return ">" + sign + s[1:]
	}
	return sign + s
}

// sideOf words where price sits against a level.
func sideOf(price, level float64) string {
	switch {
	case price > level:
		return "above"
	case price < level:
		return "below"
	default:
		return "at"
	}
}

// relTo: "0.9% below EMA50 4367", or "at EMA50 4367".
func relTo(price, level float64, name string) string {
	side := sideOf(price, level)
	if side == "at" {
		return fmt.Sprintf("at %s %s", name, trimFloat(level))
	}
	return fmt.Sprintf("%s %s %s %s", pctAway(price, level), side, name, trimFloat(level))
}

// conflictReason is the conflict verdict's tail: ADX is high enough, but the
// EMA conditions do not line up (the checklist names which one).
func conflictReason(r trendRead) string {
	if r.emaDirection() == "" {
		return "EMA50 equals EMA200, no direction"
	}
	return fmt.Sprintf("ADX ≥ %d but the EMA conditions disagree", trendADXConfirm)
}

// emaDirection is the direction the EMA alignment points to — the only one
// the state machine could confirm — or "" when EMA50 equals EMA200.
func (r trendRead) emaDirection() string {
	switch {
	case r.EMA50 > r.EMA200:
		return trendUp
	case r.EMA50 < r.EMA200:
		return trendDown
	}
	return ""
}

// adxConfirms asks the state machine: at this ADX, is the reading past the
// grey zone (every answer but flat/grey needs ADX ≥ trendADXConfirm)?
func (r trendRead) adxConfirms() bool {
	s := classifyTrend(r.ADX, r.EMA50, r.EMA200, r.Last)
	return s != trendFlat && s != trendGrey
}

// trendView is a read plus what the card needs to word it.
type trendView struct {
	r       trendRead
	tf      string  // agent timeframe (spec.Interval)
	atr     float64 // ATR(14) at the last closed bar; 0 = unavailable
	inv     float64 // invalidation level (valid when atr > 0)
	invSide string  // "above" | "below"
}

func dirWord(dir string) string {
	if dir == trendUp {
		return "an UPTREND"
	}
	return "a DOWNTREND"
}

func dirNoun(dir string) string {
	if dir == trendUp {
		return "uptrend"
	}
	return "downtrend"
}

// trendCheck is one confirm condition of the rule, evaluated toward a
// direction.
type trendCheck struct {
	name  string // short name for "now ✗: …" lists
	label string // checklist text, without the mark
	ok    bool
	note  string // after the mark, e.g. "(not determined)"
}

func (c trendCheck) String() string {
	mark := "✗"
	if c.ok {
		mark = "✓"
	}
	s := c.label + " " + mark
	if c.note != "" {
		s += " " + c.note
	}
	return s
}

// checks is THE enumeration of the confirm conditions, always the same four
// items: the inputs of classifyTrend (ADX, EMA50 vs EMA200, close vs EMA50)
// plus the structure gate (structureDemotion), evaluated toward dir by those
// same functions. All four ✓ is exactly "the rule confirms dir".
//
// dir == "" (EMA50 equals EMA200): there is no direction, the rule cannot
// confirm anything, so the three direction-bound items are ✗ — still listed,
// so the checklist never changes shape.
//
// An unreadable structure is "not against" by the gate's own definition — it
// never demotes — so it is ✓, noted as not determined. A failing structure
// says why, without naming pivot patterns.
func (v trendView) checks(dir string) []trendCheck {
	r := v.r
	adxCmp := "<"
	if r.adxConfirms() {
		adxCmp = "≥"
	}
	adx := trendCheck{name: "ADX", label: fmt.Sprintf("ADX %s %s %d", adxShown(r.ADX), adxCmp, trendADXConfirm), ok: r.adxConfirms()}
	if dir == "" {
		return []trendCheck{adx,
			{name: "EMA alignment", label: "EMA50 = EMA200", note: "(no direction)"},
			{name: "close vs EMA50", label: "close vs EMA50"},
			{name: "structure", label: "structure not against"},
		}
	}
	cmp := ">"
	if dir == trendDown {
		cmp = "<"
	}
	structure := trendCheck{name: "structure", label: "structure not against",
		ok: structureDemotion(dir, r.Structure) == ""}
	switch {
	case r.Structure == "":
		structure.note = "(not determined)"
	case !structure.ok && r.Structure == "mixed":
		structure.note = "(swings not aligned)"
	case !structure.ok:
		structure.note = "(runs against the trend)"
	}
	return []trendCheck{adx,
		{name: "EMA alignment", label: "EMA50 " + cmp + " EMA200", ok: r.emaDirection() == dir},
		{name: "close vs EMA50", label: "close " + cmp + " EMA50",
			ok: classifyTrend(trendADXConfirm, r.EMA50, r.EMA200, r.Last) == dir},
		structure,
	}
}

// failing names the checklist items that are ✗ toward dir.
func (v trendView) failing(dir string) []string {
	var out []string
	for _, c := range v.checks(dir) {
		if !c.ok {
			out = append(out, c.name)
		}
	}
	return out
}

// whyLine renders the checklist toward the EMA lean — for every state, flat
// and EMA-equal included (the only place the ADX number is printed).
func (v trendView) whyLine() string {
	parts := make([]string, 0, 4)
	for _, c := range v.checks(v.r.emaDirection()) {
		parts = append(parts, c.String())
	}
	return "Why: " + strings.Join(parts, " · ")
}

// positionLine — where price is. Confirmed states place it against the
// pullback zone (below / inside / above), everything else against the two
// EMAs the rule reads.
func (v trendView) positionLine() string {
	r := v.r
	px := trimFloat(r.Last)
	if z := pullbackZoneFor(r.State, r.EMA20, r.EMA50); z != nil {
		lo, hi := math.Min(z.From, z.To), math.Max(z.From, z.To)
		band := trimFloat(lo) + "–" + trimFloat(hi)
		switch {
		case r.Last < lo:
			return fmt.Sprintf("Price %s — %s below the pullback zone %s", px, pctAway(r.Last, lo), band)
		case r.Last > hi:
			return fmt.Sprintf("Price %s — %s above the pullback zone %s", px, pctAway(r.Last, hi), band)
		default:
			return fmt.Sprintf("Price %s — inside the pullback zone %s", px, band)
		}
	}
	return fmt.Sprintf("Price %s — %s and %s", px, relTo(r.Last, r.EMA50, "EMA50"), relTo(r.Last, r.EMA200, "EMA200"))
}

// holdsLine — a confirmed reading keeps confirmation only while the whole
// checklist stays ✓. Losing any item WITHDRAWS it (conflict or grey); the ATR
// level on the next line is a different thing — it INVALIDATES the idea.
func (v trendView) holdsLine() string {
	return "Confirmation holds while all four conditions stay ✓; any ✗ withdraws it"
}

// confirmLine — for an unconfirmed state: what confirmation takes (the whole
// checklist) and which items are ✗ now.
func (v trendView) confirmLine() string {
	dir := v.r.emaDirection()
	if dir == "" {
		return "No direction to confirm while EMA50 equals EMA200"
	}
	s := fmt.Sprintf("Confirms as %s only when all four conditions are ✓", dirWord(dir))
	if f := v.failing(dir); len(f) > 0 {
		s += " (now ✗: " + strings.Join(f, ", ") + ")"
	}
	return s
}

// facts is the card body: where price is → what keeps or changes the
// reading → the checklist.
func (v trendView) facts() []string {
	r := v.r
	lines := []string{v.positionLine()}
	if r.Confirmed() {
		lines = append(lines, v.holdsLine())
		if v.atr > 0 {
			lines = append(lines, trendInvalidationFact(r.State, v.inv, v.invSide, v.tf, r.Last))
		}
	} else {
		lines = append(lines, v.confirmLine())
	}
	return append(lines, v.whyLine())
}

// scenarios — two "If …, the reading …" statements. The first names a
// confirmed state only with the whole checklist; the second is a transition
// the rule fully determines from ADX alone (or the ATR invalidation).
func (v trendView) scenarios() []string {
	r := v.r
	tf := candleWord(v.tf)
	if r.Confirmed() {
		hold := fmt.Sprintf("If all four conditions stay ✓, the reading stays a confirmed %s", dirNoun(r.State))
		if v.atr > 0 {
			prep := "under"
			if v.invSide == "above" {
				prep = "over"
			}
			return []string{hold, fmt.Sprintf("If a %s candle closes %s %s, the reading is invalidated as a %s (1 ATR %s the EMA cluster)",
				tf, v.invSide, trimFloat(v.inv), dirNoun(r.State), prep)}
		}
		return []string{hold, "If any condition turns ✗, the reading loses confirmation"}
	}
	first := "If EMA50 and EMA200 separate and all four conditions turn ✓, the reading confirms in that direction"
	if dir := r.emaDirection(); dir != "" {
		art := "a"
		if dir == trendUp {
			art = "an"
		}
		first = fmt.Sprintf("If all four conditions turn ✓ for %s %s, the reading confirms as %s %s", art, dirNoun(dir), art, dirNoun(dir))
	}
	var fallback string
	switch {
	case r.State == trendFlat:
		fallback = fmt.Sprintf("If ADX stays below %d, the reading stays flat", trendADXRead)
	case !r.adxConfirms(): // grey by ADX
		fallback = fmt.Sprintf("If ADX falls below %d, the reading returns to flat", trendADXRead)
	default: // conflict, or grey by the structure gate: ADX is at 25+
		fallback = fmt.Sprintf("If ADX falls below %d, the reading turns grey (flat below %d)", trendADXConfirm, trendADXRead)
	}
	return []string{first, fallback}
}

// whyLevel explains the one level the card leans on, and that losing any
// checklist item withdraws confirmation before price ever gets there.
func (v trendView) whyLevel() string {
	r := v.r
	if r.Confirmed() && v.atr > 0 {
		fn, op := "max", "+"
		if v.invSide == "below" {
			fn, op = "min", "−"
		}
		return fmt.Sprintf("%s = %s(EMA50, EMA200) %s 1 ATR(14) on a closed %s candle; a checklist ✗ withdraws confirmation sooner",
			trimFloat(v.inv), fn, op, candleWord(v.tf))
	}
	if r.Confirmed() {
		return "No invalidation level: ATR(14) unavailable; a checklist ✗ withdraws confirmation"
	}
	// An unconfirmed trend has no invalidation level anywhere — not on the
	// card, not in levels (decided 2026-09-15).
	return "No invalidation level: the trend is not confirmed. Confirmation needs all four checklist conditions"
}

// neutralConclusion is the landing-page sentence for an unconfirmed trend
// card ("" when confirmed). A flat card has no trend at all, so it is not
// called an unconfirmed trend; the others name the ✗ checklist items.
func (v trendView) neutralConclusion(subject string) string {
	r := v.r
	switch dir := r.emaDirection(); {
	case r.Confirmed():
		return ""
	case r.State == trendFlat:
		return fmt.Sprintf("For a trader there is no trend to read on %s: ADX %s is under %d.", subject, adxShown(r.ADX), trendADXRead)
	case dir == "":
		return fmt.Sprintf("For a trader this is an unconfirmed trend on %s: EMA50 equals EMA200, no direction to confirm.", subject)
	default:
		return fmt.Sprintf("For a trader this is an unconfirmed trend on %s; failing: %s.", subject, strings.Join(v.failing(dir), ", "))
	}
}

// blocks is the content-ready form of the same card (additive JSON field).
func (v trendView) blocks(short string) *ContentBlocks {
	head := short
	if head != "" {
		head = strings.ToUpper(head[:1]) + head[1:]
	}
	b := &ContentBlocks{
		WhatHappened: fmt.Sprintf("%s on %s: %s.", head, candleWord(v.tf), lowerFirst(v.positionLine())),
		WhyLevel:     v.whyLevel(),
		Scenarios:    v.scenarios(),
		Regime:       fmt.Sprintf("%s · %s · ADX %s", short, candleWord(v.tf), adxShown(v.r.ADX)),
	}
	if v.r.Confirmed() && v.atr > 0 {
		s := fmt.Sprintf("A closed %s candle %s %s invalidates the %s idea",
			candleWord(v.tf), v.invSide, trimFloat(v.inv), dirNoun(v.r.State))
		b.Invalidates = &s
	}
	return b
}
