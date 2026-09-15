package demobot

// gold_readable_test.go — Gold stage 1 (2026-09-15): the card's wording,
// order, blocks and machine fields. Rules are untouched; every case runs the
// pure builder (goldCardFrom) at a fixed clock, so the texts are golden.

import (
	"encoding/json"
	"regexp"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

var goldNow = time.Date(2026, 9, 15, 5, 30, 0, 0, time.UTC) // a Tuesday

func goldTrendCard(state string) Card {
	r := trendRead{OK: true, State: state, Raw: state, ADX: 22.2, EMA20: 4340, EMA50: 4367, EMA200: 4433, Last: 4354.9, ATR: 60}
	switch state {
	case trendUp:
		r.ADX, r.EMA20, r.EMA50, r.EMA200 = 31.4, 4340, 4300, 4100
	case trendDown:
		r.ADX, r.EMA20, r.EMA50, r.EMA200 = 33.7, 4370, 4400, 4600
	case trendFlat:
		r.ADX = 17.3
	case trendConflict:
		r.ADX, r.EMA50, r.EMA200 = 28.1, 4367, 4300
	}
	return trendCardFrom(r, goldDailySpec, time.Date(2026, 9, 15, 4, 0, 0, 0, time.UTC))
}

func goldMacroCard(state string, pos, neu, neg int) Card {
	var lamps []MacroLampReadout
	add := func(n int, word string) {
		for i := 0; i < n; i++ {
			lamps = append(lamps, MacroLampReadout{Key: word, Contribution: word, Voting: true,
				AsOf: time.Date(2026, 9, 14, 13, 30+len(lamps), 0, 0, time.UTC).Format(time.RFC3339)})
		}
	}
	add(pos, contribPositive)
	add(neu, contribNeutral)
	add(neg, contribNegative)
	// The gold lamp itself never votes: it must not be counted or dated.
	lamps = append(lamps, MacroLampReadout{Key: "gold", AsOf: "2026-09-01T00:00:00Z"})
	return Card{State: state, Macro: &MacroReadout{Lamps: lamps}}
}

func goldFixture(state string) goldInputs {
	sup, res := SRLevel{Raw: 4329.2, Touches: 1}, SRLevel{Raw: 4364.5, Touches: 2}
	return goldInputs{
		trend: goldTrendCard(state),
		levels: goldDayLevels{High: 4396.8, Low: 4293.0, Defined: true,
			BarTime: time.Date(2026, 9, 14, 4, 0, 0, 0, time.UTC).Unix()},
		dailyAsOf: time.Date(2026, 9, 15, 4, 0, 0, 0, time.UTC),
		px:        4354.9,
		pxAt:      time.Date(2026, 9, 15, 3, 0, 0, 0, time.UTC),
		hasPx:     true,
		macro:     goldMacroCard(goldNeutral, 1, 1, 2),
		sup:       &sup,
		res:       &res,
		vol:       volShortLine(1.086, goldDailySpec.Interval),
		now:       goldNow,
	}
}

// ── golden texts ─────────────────────────────────────────────────────────────

func TestGoldCardGoldenTexts(t *testing.T) {
	macroConflictUp := goldFixture(trendUp)
	macroConflictUp.macro = goldMacroCard(goldPressure, 1, 0, 3)

	cases := map[string]struct {
		in      goldInputs
		verdict string
		facts   []string
	}{
		"confirmed up": {goldFixture(trendUp), "Daily regime: confirmed UPTREND", []string{
			"Last closed 1h price 4354.90 at 2026-09-15 03:00 UTC — inside the day range",
			"Day range 4293.00 – 4396.80: high/low of the closed 1d candle of 2026-09-14",
			"A daily close above 4396.80 classifies the day as an upside break",
			"A daily close below 4293.00 classifies the day as a downside break",
			"A closed 1d candle below 4040.00 invalidates the daily uptrend reading (1 ATR under the EMA cluster)",
			"Macro backdrop: mixed for gold (lamps: 1 for, 1 neutral, 2 against)",
			"Nearest levels: support 4329.20 (single swing, 1 pivot) · resistance 4364.50 (candidate, 2 pivots)",
			"Volatility: normal · 1d · ATR 1.086× its 30-bar baseline",
		}},
		"confirmed down": {goldFixture(trendDown), "Daily regime: confirmed DOWNTREND", []string{
			"Last closed 1h price 4354.90 at 2026-09-15 03:00 UTC — inside the day range",
			"Day range 4293.00 – 4396.80: high/low of the closed 1d candle of 2026-09-14",
			"A daily close above 4396.80 classifies the day as an upside break",
			"A daily close below 4293.00 classifies the day as a downside break",
			"A closed 1d candle above 4660.00 invalidates the daily downtrend reading (1 ATR over the EMA cluster)",
			"Macro backdrop: mixed for gold (lamps: 1 for, 1 neutral, 2 against)",
			"Nearest levels: support 4329.20 (single swing, 1 pivot) · resistance 4364.50 (candidate, 2 pivots)",
			"Volatility: normal · 1d · ATR 1.086× its 30-bar baseline",
		}},
		"not confirmed": {goldFixture(trendGrey), "Daily regime: not confirmed — no direction claimed", []string{
			"Regime: grey zone · 1d — trend forming, not confirmed",
			"Last closed 1h price 4354.90 at 2026-09-15 03:00 UTC — inside the day range",
			"Day range 4293.00 – 4396.80: high/low of the closed 1d candle of 2026-09-14",
			"A daily close above 4396.80 classifies the day as an upside break",
			"A daily close below 4293.00 classifies the day as a downside break",
			"Macro backdrop: mixed for gold (lamps: 1 for, 1 neutral, 2 against)",
			"Nearest levels: support 4329.20 (single swing, 1 pivot) · resistance 4364.50 (candidate, 2 pivots)",
			"Volatility: normal · 1d · ATR 1.086× its 30-bar baseline",
		}},
		"macro against the regime": {macroConflictUp, "Daily regime: confirmed UPTREND", []string{
			"Last closed 1h price 4354.90 at 2026-09-15 03:00 UTC — inside the day range",
			"Day range 4293.00 – 4396.80: high/low of the closed 1d candle of 2026-09-14",
			"A daily close above 4396.80 classifies the day as an upside break",
			"A daily close below 4293.00 classifies the day as a downside break",
			"A closed 1d candle below 4040.00 invalidates the daily uptrend reading (1 ATR under the EMA cluster)",
			"Macro backdrop: pressure for gold (lamps: 1 for, 0 neutral, 3 against)",
			"Macro backdrop conflicts with the daily uptrend reading; both stand as read",
			"Nearest levels: support 4329.20 (single swing, 1 pivot) · resistance 4364.50 (candidate, 2 pivots)",
			"Volatility: normal · 1d · ATR 1.086× its 30-bar baseline",
		}},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			c := goldCardFrom(tc.in)
			if c.Verdict != tc.verdict {
				t.Errorf("verdict = %q, want %q", c.Verdict, tc.verdict)
			}
			if strings.Join(c.Facts, "\n") != strings.Join(tc.facts, "\n") {
				t.Errorf("facts:\n%s\nwant:\n%s", strings.Join(c.Facts, "\n"), strings.Join(tc.facts, "\n"))
			}
		})
	}
}

// A price older than goldPriceStale is called stale — by a fixed threshold,
// never by a running age ("7h old") that would change the body every hour.
func TestGoldCardStalePrice(t *testing.T) {
	in := goldFixture(trendGrey)
	in.pxAt = time.Date(2026, 9, 14, 20, 0, 0, 0, time.UTC) // 9.5h before goldNow
	c := goldCardFrom(in)
	want := "Last closed 1h price 4354.90 at 2026-09-14 20:00 UTC — stale (over 6h old), inside the day range"
	if c.Facts[1] != want {
		t.Errorf("stale line = %q, want %q", c.Facts[1], want)
	}
	// The fixed threshold "over 6h old" is allowed; any other age is not.
	body := strings.ReplaceAll(strings.Join(c.Facts, "\n"), "stale (over 6h old)", "")
	if regexp.MustCompile(`\d+(\.\d+)?h old|ago`).MatchString(body) {
		t.Errorf("a running age must not reach the body: %v", c.Facts)
	}
	if g := c.Gold; g == nil || g.PriceFreshness == nil || *g.PriceFreshness != "stale" {
		t.Errorf("price_freshness must be stale: %+v", g)
	}
	// Exactly at the threshold it is not stale yet.
	in.pxAt = goldNow.Add(-goldPriceStale)
	if c := goldCardFrom(in); strings.Contains(c.Facts[1], "stale") || *c.Gold.PriceFreshness != "on_time" {
		t.Errorf("at exactly 6h: %q / %v", c.Facts[1], *c.Gold.PriceFreshness)
	}
}

// Price already beyond an edge intraday: the scenario on that side says so,
// the other stays plain — and the two mirror each other exactly.
func TestGoldScenariosSymmetry(t *testing.T) {
	mirror := strings.NewReplacer("above", "below", "below", "above", "upside", "downside", "downside", "upside",
		"4396.80", "4293.00", "4293.00", "4396.80", "an upside", "a downside", "a downside", "an upside")

	plain := goldFixture(trendGrey)
	c := goldCardFrom(plain)
	up, down := c.Facts[3], c.Facts[4]
	if mirror.Replace(up) != down || mirror.Replace(down) != up {
		t.Errorf("scenarios do not mirror:\n%q\n%q", up, down)
	}

	above := goldFixture(trendGrey)
	above.px = 4400.1
	below := goldFixture(trendGrey)
	below.px = 4290.0
	ca, cb := goldCardFrom(above), goldCardFrom(below)
	if want := "A daily close above 4396.80 classifies the day as an upside break; the last 1h close is already above it"; ca.Facts[3] != want {
		t.Errorf("above: %q, want %q", ca.Facts[3], want)
	}
	if want := "A daily close below 4293.00 classifies the day as a downside break; the last 1h close is already below it"; cb.Facts[4] != want {
		t.Errorf("below: %q, want %q", cb.Facts[4], want)
	}
	if ca.Facts[4] != down || cb.Facts[3] != up {
		t.Errorf("the untouched side must stay plain: %q / %q", ca.Facts[4], cb.Facts[3])
	}
	if mirror.Replace(ca.Facts[3]) != cb.Facts[4] {
		t.Errorf("already-beyond lines do not mirror:\n%q\n%q", ca.Facts[3], cb.Facts[4])
	}
	if !strings.HasSuffix(ca.Facts[1], "— above the day range") || !strings.HasSuffix(cb.Facts[1], "— below the day range") {
		t.Errorf("price lines: %q / %q", ca.Facts[1], cb.Facts[1])
	}
}

// Invalidation follows the direction: uptrend below, downtrend above — and
// only when the regime is confirmed and a price is there.
func TestGoldInvalidationSymmetry(t *testing.T) {
	inv := func(c Card) string {
		for _, f := range c.Facts {
			if strings.Contains(f, "invalidat") {
				return f
			}
		}
		return ""
	}
	up, down := inv(goldCardFrom(goldFixture(trendUp))), inv(goldCardFrom(goldFixture(trendDown)))
	if !strings.Contains(up, " below ") || !strings.Contains(up, "uptrend") || strings.Contains(up, "above") {
		t.Errorf("uptrend invalidation: %q", up)
	}
	if !strings.Contains(down, " above ") || !strings.Contains(down, "downtrend") || strings.Contains(down, "below") {
		t.Errorf("downtrend invalidation: %q", down)
	}
	flip := strings.NewReplacer("below", "above", "above", "below", "uptrend", "downtrend", "downtrend", "uptrend",
		"under", "over", "over", "under", "4040", "4660", "4660", "4040")
	if flip.Replace(up) != down {
		t.Errorf("invalidation lines do not mirror:\n%q\n%q", up, down)
	}
	for _, st := range []string{trendGrey, trendFlat, trendConflict} {
		c := goldCardFrom(goldFixture(st))
		if l := inv(c); l != "" {
			t.Errorf("%s: nothing to invalidate without a confirmed regime, got %q", st, l)
		}
		if c.Blocks == nil || c.Blocks.Invalidates != nil || c.Levels != nil {
			t.Errorf("%s: invalidates must be null and no levels: %+v %+v", st, c.Blocks, c.Levels)
		}
	}
	noPx := goldFixture(trendUp)
	noPx.hasPx = false
	if l := inv(goldCardFrom(noPx)); l != "" {
		t.Errorf("no price, no direction, nothing to invalidate: %q", l)
	}
}

// ── degraded paths ───────────────────────────────────────────────────────────

func TestGoldCardNoIntradayPrice(t *testing.T) {
	in := goldFixture(trendUp)
	in.hasPx, in.px, in.pxAt = false, 0, time.Time{}
	c := goldCardFrom(in)
	if c.Verdict != "Intraday price unavailable — no direction claimed" || c.effectiveStatus() == statusOK || c.Emoji != emojiNeutral {
		t.Errorf("verdict %q status %v emoji %q", c.Verdict, c.effectiveStatus(), c.Emoji)
	}
	want := []string{
		"Daily regime reads confirmed UPTREND · 1d; no direction stated without a 1h price",
		"Intraday price feed is down: the last 1h close cannot be placed against the day range",
		"Day range 4293.00 – 4396.80: high/low of the closed 1d candle of 2026-09-14",
		"A daily close above 4396.80 classifies the day as an upside break",
		"A daily close below 4293.00 classifies the day as a downside break",
		"Macro backdrop: mixed for gold (lamps: 1 for, 1 neutral, 2 against)",
		"Volatility: normal · 1d · ATR 1.086× its 30-bar baseline",
	}
	if strings.Join(c.Facts, "\n") != strings.Join(want, "\n") {
		t.Errorf("facts:\n%s\nwant:\n%s", strings.Join(c.Facts, "\n"), strings.Join(want, "\n"))
	}
	if c.Blocks != nil {
		t.Errorf("a degraded card carries no blocks: %+v", c.Blocks)
	}
	grey := goldFixture(trendGrey)
	grey.hasPx = false
	if f := goldCardFrom(grey).Facts[0]; f != "Regime: grey zone · 1d — trend forming, not confirmed" {
		t.Errorf("unconfirmed without price keeps its regime line: %q", f)
	}
}

func TestGoldCardNoMacroAndUndefinedRange(t *testing.T) {
	in := goldFixture(trendGrey)
	in.macro = Card{}
	in.levels = goldDayLevels{}
	in.sup, in.res = nil, nil
	c := goldCardFrom(in)
	want := []string{
		"Regime: grey zone · 1d — trend forming, not confirmed",
		"Last closed 1h price 4354.90 at 2026-09-15 03:00 UTC",
		"Day range undefined: more than 3 nested inside days, no day levels to give",
		"Macro backdrop: no gold read available",
		"Volatility: normal · 1d · ATR 1.086× its 30-bar baseline",
	}
	if strings.Join(c.Facts, "\n") != strings.Join(want, "\n") {
		t.Errorf("facts:\n%s\nwant:\n%s", strings.Join(c.Facts, "\n"), strings.Join(want, "\n"))
	}
	g := c.Gold
	if g == nil || g.MacroAsOf != nil || g.MacroBackdrop != nil || g.MacroLamps != nil || g.DayRange != nil || g.PricePosition != nil {
		t.Errorf("absent parts must be null: %+v", g)
	}
	if c.Blocks == nil || len(c.Blocks.Scenarios) != 0 || c.Blocks.Scenarios == nil {
		t.Errorf("no range → scenarios [] (not null): %+v", c.Blocks)
	}
}

func TestGoldCardInsideDayRangeLine(t *testing.T) {
	for n, want := range map[int]string{
		1: "Day range 4293.00 – 4396.80: high/low of the 1d candle of 2026-09-14 (1 inside day after it)",
		3: "Day range 4293.00 – 4396.80: high/low of the 1d candle of 2026-09-14 (3 inside days after it)",
	} {
		in := goldFixture(trendGrey)
		in.levels.Unwound = n
		if got := goldCardFrom(in).Facts[2]; got != want {
			t.Errorf("unwound %d: %q, want %q", n, got, want)
		}
	}
}

func TestGoldCardWeekendBannerLeads(t *testing.T) {
	in := goldFixture(trendGrey)
	in.now = time.Date(2026, 9, 19, 12, 0, 0, 0, time.UTC) // Saturday
	in.pxAt = time.Date(2026, 9, 18, 21, 0, 0, 0, time.UTC)
	c := goldCardFrom(in)
	if c.Facts[0] != goldClosedBanner {
		t.Errorf("banner first: %v", c.Facts)
	}
	if !strings.Contains(c.Facts[2], "stale (over 6h old)") {
		t.Errorf("weekend price is stale by age, not 'now': %q", c.Facts[2])
	}
}

// ── blocks ───────────────────────────────────────────────────────────────────

func TestGoldBlocks(t *testing.T) {
	b := goldCardFrom(goldFixture(trendUp)).Blocks
	if b == nil {
		t.Fatal("no blocks")
	}
	check := func(field, got, want string) {
		t.Helper()
		if got != want {
			t.Errorf("%s = %q, want %q", field, got, want)
		}
	}
	check("what_happened", b.WhatHappened, "Snapshot, not an event: 1d regime confirmed uptrend; last 1h close 4354.90 inside the day range")
	check("why_level", b.WhyLevel, "Day range = high/low of the closed 1d candle of 2026-09-14; S/R = swing-pivot cluster means")
	check("regime", b.Regime, "Local gold regime: confirmed uptrend · 1d · ADX 31.4")
	check("limitations", b.Limitations, "COMEX GC=F futures, not spot XAUUSD; describes the period, not a forecast of the day")
	if len(b.Scenarios) != 2 || b.Scenarios[0] != "A daily close above 4396.80 classifies the day as an upside break" ||
		b.Scenarios[1] != "A daily close below 4293.00 classifies the day as a downside break" {
		t.Errorf("scenarios: %q", b.Scenarios)
	}
	if b.Invalidates == nil || *b.Invalidates != "A closed 1d candle below 4040.00 invalidates the daily uptrend reading (1 ATR under the EMA cluster)" {
		t.Errorf("invalidates: %v", b.Invalidates)
	}
	stale := goldFixture(trendDown)
	stale.pxAt = goldNow.Add(-7 * time.Hour)
	check("what_happened (stale)", goldCardFrom(stale).Blocks.WhatHappened,
		"Snapshot, not an event: 1d regime confirmed downtrend; last 1h close 4354.90 inside the day range (stale)")
}

// ── machine fields ───────────────────────────────────────────────────────────

func TestGoldReadoutAndEnvelope(t *testing.T) {
	c := goldCardFrom(goldFixture(trendUp))
	if !c.noValidator {
		t.Error("gold must carry no validator: its body follows the clock (stale, weekend)")
	}
	env := cardEnvelope(c)
	raw, err := json.Marshal(env)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	if m["asset"] != "XAUUSD" || m["data_as_of"] != "2026-09-15T04:00:00Z" {
		t.Errorf("asset/data_as_of: %v %v", m["asset"], m["data_as_of"])
	}
	g, ok := m["gold"].(map[string]any)
	if !ok {
		t.Fatalf("no gold object: %s", raw)
	}
	want := map[string]any{
		"regime": "up", "confirmed": true,
		"daily_as_of": "2026-09-15T04:00:00Z", "price_as_of": "2026-09-15T03:00:00Z",
		"price_freshness": "on_time", "price": 4354.9, "price_position": "inside",
		"macro_as_of": "2026-09-14T13:30:00Z", "macro_backdrop": "neutral",
	}
	for k, v := range want {
		if g[k] != v {
			t.Errorf("gold.%s = %v, want %v", k, g[k], v)
		}
	}
	dr, _ := g["day_range"].(map[string]any)
	if dr["high"] != 4396.8 || dr["low"] != 4293.0 || dr["candle_date"] != "2026-09-14" || dr["inside_days_after"] != 0.0 {
		t.Errorf("day_range: %v", dr)
	}
	ml, _ := g["macro_lamps"].(map[string]any)
	if ml["for"] != 1.0 || ml["neutral"] != 1.0 || ml["against"] != 2.0 {
		t.Errorf("macro_lamps: %v", ml)
	}
	if _, ok := m["blocks"].(map[string]any); !ok {
		t.Errorf("blocks missing: %s", raw)
	}
	// Other agents do not grow a gold key.
	if b, _ := json.Marshal(cardEnvelope(goldTrendCard(trendUp))); strings.Contains(string(b), `"gold"`) {
		t.Errorf("trend envelope carries gold: %s", b)
	}
}

// ── wording rules on every path ──────────────────────────────────────────────

func goldAllCards() map[string]Card {
	out := map[string]Card{}
	for _, st := range []string{trendUp, trendDown, trendGrey, trendFlat, trendConflict} {
		for _, pos := range []float64{4354.9, 4400.1, 4290} {
			in := goldFixture(st)
			in.px = pos
			out[st+"/"+goldPx(pos)] = goldCardFrom(in)
			in.pxAt = goldNow.Add(-9 * time.Hour)
			out[st+"/stale/"+goldPx(pos)] = goldCardFrom(in)
			in.hasPx = false
			out[st+"/noprice"] = goldCardFrom(in)
		}
		in := goldFixture(st)
		in.macro, in.levels, in.sup, in.res, in.vol = Card{}, goldDayLevels{}, nil, nil, ""
		out[st+"/bare"] = goldCardFrom(in)
		in = goldFixture(st)
		in.now = time.Date(2026, 9, 19, 12, 0, 0, 0, time.UTC)
		out[st+"/weekend"] = goldCardFrom(in)
		in = goldFixture(st)
		in.macro = goldMacroCard(goldPressure, 1, 1, 2)
		out[st+"/pressure"] = goldCardFrom(in)
		in.macro = goldMacroCard(goldSupport, 2, 1, 1)
		out[st+"/support"] = goldCardFrom(in)
	}
	// Worst case for length: five-digit prices, established levels with
	// two-digit pivot counts, nested inside days, the conflict wording, stale.
	for _, st := range []string{trendUp, trendDown, trendConflict} {
		in := goldFixture(st)
		in.levels = goldDayLevels{High: 12345.6, Low: 10234.5, Defined: true, Unwound: 3, BarTime: in.levels.BarTime}
		in.px = 99999.9
		in.pxAt = goldNow.Add(-30 * time.Hour)
		sup, res := SRLevel{Raw: 99888.8, Touches: 12}, SRLevel{Raw: 99999.95, Touches: 11}
		in.sup, in.res = &sup, &res
		in.macro = goldMacroCard(goldPressure, 4, 4, 4)
		if st == trendDown {
			in.macro = goldMacroCard(goldSupport, 4, 4, 4)
		}
		in.vol = volShortLine(12.3456, goldDailySpec.Interval)
		if m := in.macro.Macro; m != nil {
			score := 100
			m.RuleScore = &score
		}
		out["worst/"+st] = goldCardFrom(in)
		in.px = 1000.1
		out["worst-below/"+st] = goldCardFrom(in)
		// Five-digit invalidation level with the "already beyond" tail.
		if lv, ok := in.trend.Levels.(TrendLevels); ok {
			lvl := 99999.99
			if st == trendDown {
				lvl, in.px = 10000.01, 99999.9
			}
			lv.Invalidation = &lvl
			in.trend.Levels = lv
			out["worst-inv-tail/"+st] = goldCardFrom(in)
		}
	}
	return out
}

func goldLines(c Card) []string {
	lines := append([]string{c.Verdict}, c.Facts...)
	if b := c.Blocks; b != nil {
		lines = append(lines, b.WhatHappened, b.WhyLevel, b.Regime, b.Limitations)
		lines = append(lines, b.Scenarios...)
		if b.Invalidates != nil {
			lines = append(lines, *b.Invalidates)
		}
	}
	return lines
}

func TestGoldTextFitsOneLine(t *testing.T) {
	for key, c := range goldAllCards() {
		for _, l := range goldLines(c) {
			if n := utf8.RuneCountInString(l); n > goldFactMaxRunes {
				t.Errorf("%s: %d runes > %d: %q", key, n, goldFactMaxRunes, l)
			}
		}
	}
}

func TestGoldTextBannedWords(t *testing.T) {
	banned := []string{
		"Price now", "Day turns", "turns up", "turns down", "structure is broken", "regime above is broken",
		" will ", "expect", "likely", "probab", "target", "bias", "Bias", "should", "advised", "recommend",
		"BUY", "SELL", "lamps split", "swing pivots)", "1 swing pivot",
	}
	nowWord := regexp.MustCompile(`(?i)\bnow\b|\d+h old|\bago\b`)
	for key, c := range goldAllCards() {
		b, err := json.Marshal(c.Blocks)
		if err != nil {
			t.Fatal(err)
		}
		all := c.RenderHTML() + "\n" + string(b)
		for _, w := range banned {
			if strings.Contains(all, w) {
				t.Errorf("%s: %q must not appear:\n%s", key, w, all)
			}
		}
		// The fixed threshold "over 6h old" is allowed; a running age is not.
		text := strings.ReplaceAll(strings.Join(goldLines(c), "\n"), "stale (over 6h old)", "")
		if m := nowWord.FindString(text); m != "" {
			t.Errorf("%s: %q — the price is the last closed 1h bar, never 'now', and no running age:\n%s", key, m, all)
		}
	}
}

func TestGoldHowText(t *testing.T) {
	h := howTexts[keyGold]
	if n := utf8.RuneCountInString(h); n > 200 {
		t.Errorf("how-text %d > 200 runes: %q", n, h)
	}
	for _, want := range []string{"GC=F", "closed 1h", "not a forecast", "enclosing day after inside days"} {
		if !strings.Contains(h, want) {
			t.Errorf("how-text lacks %q: %q", want, h)
		}
	}
}

// The invalidation line COMPARES a daily close with the level, so the level
// prints at tick resolution (goldPx), like every other compared price. With
// trimFloat a level of 4293.40 printed "below 4293", while a close of 4293.20
// — above the printed number — already invalidates under the rule.
func TestGoldInvalidationPrintsTickPrecision(t *testing.T) {
	for _, tc := range []struct {
		state, side string
		level       float64
		want        string
	}{
		{trendUp, "below", 4293.4, "A closed 1d candle below 4293.40 invalidates the daily uptrend reading (1 ATR under the EMA cluster)"},
		{trendDown, "above", 4396.75, "A closed 1d candle above 4396.75 invalidates the daily downtrend reading (1 ATR over the EMA cluster)"},
	} {
		in := goldFixture(tc.state)
		lvl := tc.level
		lv := in.trend.Levels.(TrendLevels)
		lv.Invalidation, lv.InvalidationSide = &lvl, tc.side
		in.trend.Levels = lv
		c := goldCardFrom(in)
		found := false
		for _, f := range c.Facts {
			if f == tc.want {
				found = true
			}
		}
		if !found {
			t.Errorf("%s: no line %q in %v", tc.state, tc.want, c.Facts)
		}
		if c.Blocks == nil || c.Blocks.Invalidates == nil || *c.Blocks.Invalidates != tc.want {
			t.Errorf("%s: blocks.invalidates must equal the card line: %v", tc.state, c.Blocks)
		}
		got, ok := c.Levels.(TrendLevels)
		if !ok || got.Invalidation == nil || *got.Invalidation != tc.level || got.InvalidationSide != tc.side {
			t.Errorf("%s: levels must carry the same raw level and side: %+v", tc.state, c.Levels)
		}
	}
}

// Review 2026-09-15: the macro word comes from the WEIGHTED gold score
// (DXY 40, rates 25, VIX 25, SPX 10), not from a lamp majority. With only DXY
// for gold the score is 40 — "mixed" — beside 3 lamps against; the line must
// show the score the word was chosen by, taken from the macro card itself.
func TestGoldMacroLineShowsWeightedScore(t *testing.T) {
	v := func(f float64) *float64 { return &f }
	at := "2026-09-14T20:00:00Z"
	m := &MacroResp{Regime: "mixed", TradfinOpen: true, CapturedAt: "2026-09-15T05:30:00Z", Lamps: []MacroLamp{
		{Key: "dxy", Label: "DXY", Value: v(97.1), OK: true, Status: "tailwind", AsOf: at, Source: "yahoo"},     // for gold
		{Key: "rates", Label: "US 10Y", Value: v(4.2), OK: true, Status: "headwind", AsOf: at, Source: "yahoo"}, // against
		{Key: "vix", Label: "VIX", Value: v(14), OK: true, Status: "tailwind", AsOf: at, Source: "yahoo"},       // inverted: against
		{Key: "spx", Label: "S&P 500", Value: v(6500), OK: true, Status: "tailwind", AsOf: at, Source: "yahoo"}, // inverted: against
	}}
	mc := macroAssetCardFrom(m, macroAssetGold)
	if mc.State != goldNeutral || mc.Macro == nil || mc.Macro.RuleScore == nil || *mc.Macro.RuleScore != 40 {
		t.Fatalf("precondition: the gold model must read mixed at 40: state %q readout %+v", mc.State, mc.Macro)
	}
	in := goldFixture(trendGrey)
	in.macro = mc
	want := "Macro backdrop: mixed for gold — weighted gold score 40/100 (lamps: 1 for, 0 neutral, 3 against)"
	found := false
	for _, f := range goldCardFrom(in).Facts {
		if f == want {
			found = true
		}
	}
	if !found {
		t.Errorf("no line %q in %v", want, goldCardFrom(in).Facts)
	}
}

// Review 2026-09-15: like the day scenarios, the invalidation line says when
// the last 1h close is already beyond the level — same text in blocks.
// Spec rule 1 before rule 3: without a 1h price the card claims no direction,
// so no conflict with a direction is stated — only with a price it is.
func TestGoldConflictLineOnlyWhenConfirmed(t *testing.T) {
	has := func(c Card) bool {
		for _, f := range c.Facts {
			if strings.Contains(f, "conflicts with the daily") {
				return true
			}
		}
		return false
	}
	in := goldFixture(trendUp)
	in.macro = goldMacroCard(goldPressure, 0, 1, 3)
	if !has(goldCardFrom(in)) {
		t.Errorf("confirmed uptrend + pressure: conflict line missing")
	}
	in.hasPx = false
	if c := goldCardFrom(in); has(c) {
		t.Errorf("no price: conflict line must not appear: %v", c.Facts)
	}
}

// The "already beyond" tail compares at the printed tick: a level and a close
// that both print 4354.90 are never worded as one beyond the other.
func TestGoldInvalidationTailAtPrintedTick(t *testing.T) {
	for _, tc := range []struct {
		state, side string
		level       float64
	}{
		{trendUp, "below", 4354.904},   // raw close 4354.90 < level, prints equal
		{trendDown, "above", 4354.896}, // raw close 4354.90 > level, prints equal
		{trendUp, "below", 4354.905},   // exact half-cent: math.Round and %.2f disagree
	} {
		in := goldFixture(tc.state) // last 1h close 4354.90
		lvl := tc.level
		lv := in.trend.Levels.(TrendLevels)
		lv.Invalidation, lv.InvalidationSide = &lvl, tc.side
		in.trend.Levels = lv
		c := goldCardFrom(in)
		if c.Blocks == nil || c.Blocks.Invalidates == nil {
			t.Fatalf("%s: no invalidation", tc.state)
		}
		inv := *c.Blocks.Invalidates
		if strings.Contains(inv, "already") || !strings.Contains(inv, "4354.90 invalidates") {
			t.Errorf("%s: %q", tc.state, inv)
		}
	}
}

func TestGoldInvalidationAlreadyBeyond(t *testing.T) {
	for _, tc := range []struct {
		state, side string
		level       float64
		want        string
	}{
		{trendUp, "below", 4360, "A closed 1d candle below 4360.00 invalidates the daily uptrend reading; last 1h close already below it"},
		{trendDown, "above", 4350, "A closed 1d candle above 4350.00 invalidates the daily downtrend reading; last 1h close already above it"},
	} {
		in := goldFixture(tc.state) // last 1h close 4354.90
		lvl := tc.level
		lv := in.trend.Levels.(TrendLevels)
		lv.Invalidation, lv.InvalidationSide = &lvl, tc.side
		in.trend.Levels = lv
		c := goldCardFrom(in)
		if c.Blocks == nil || c.Blocks.Invalidates == nil || *c.Blocks.Invalidates != tc.want {
			t.Errorf("%s: blocks.invalidates = %v, want %q", tc.state, c.Blocks, tc.want)
		}
		found := false
		for _, f := range c.Facts {
			found = found || f == tc.want
		}
		if !found {
			t.Errorf("%s: no line %q in %v", tc.state, tc.want, c.Facts)
		}
		// Exactly at the level is not beyond it: the plain line stays.
		in.px = tc.level
		for _, f := range goldCardFrom(in).Facts {
			if strings.Contains(f, "invalidates") && strings.Contains(f, "already") {
				t.Errorf("%s: at the level the tail must not appear: %q", tc.state, f)
			}
		}
	}
}
