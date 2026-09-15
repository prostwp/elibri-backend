package demobot

// trend_readable_test.go — the Trend card as a reader sees it: one fixture per
// state rendered through the pure builder (trendCardFrom), plus the wording
// rules. The rules themselves (thresholds, gate) are pinned elsewhere; nothing
// here may change them. Central invariant: the confirm conditions are listed
// in ONE place (the four-item checklist) and every other sentence refers to it.

import (
	"context"
	"encoding/json"
	"math"
	"regexp"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/prostwp/elibri-backend/pkg/types"
)

var (
	trendTestTime = time.Date(2026, 9, 15, 0, 0, 0, 0, time.UTC)
	eurSpec1h     = assetSpec{Display: "EURUSD", Interval: "1h"}
	goldSpec1h    = assetSpec{Display: "GOLD · COMEX GC=F", Key: "XAUUSD", Interval: "1h"}
	btcSpec4h     = assetSpec{Display: "BTC", Interval: "4h"}
)

type trendFixture struct {
	name string
	spec assetSpec
	r    trendRead
	atr  float64
}

func demoted(dir, structure string) string { return structureDemotion(dir, structure) }

// One fixture per state. Numbers are shaped on the live 2026-09-15 cards.
var trendFixtures = map[string]trendFixture{
	"down": {"confirmed down (EURUSD, below the zone)", eurSpec1h, trendRead{OK: true,
		State: trendDown, Raw: trendDown, ADX: 46.58,
		EMA20: 1.15592, EMA50: 1.15801, EMA200: 1.16093, Last: 1.1551}, 0.001},
	"up_inside": {"confirmed up, inside the zone", btcSpec4h, trendRead{OK: true,
		State: trendUp, Raw: trendUp, Structure: "hh_hl", ADX: 31.2,
		EMA20: 64000, EMA50: 63000, EMA200: 60000, Last: 63500}, 800},
	"up_above": {"confirmed up, above the zone", btcSpec4h, trendRead{OK: true,
		State: trendUp, Raw: trendUp, Structure: "hh_hl", ADX: 31.2,
		EMA20: 64000, EMA50: 63000, EMA200: 60000, Last: 65000}, 800},
	"grey": {"grey by ADX (gold)", goldSpec1h, trendRead{OK: true,
		State: trendGrey, Raw: trendGrey, ADX: 22.27,
		EMA20: 4340, EMA50: 4367, EMA200: 4433, Last: 4329}, 12},
	"flat": {"flat (BTC)", btcSpec4h, trendRead{OK: true,
		State: trendFlat, Raw: trendFlat, ADX: 17.39,
		EMA20: 115500, EMA50: 115000, EMA200: 110000, Last: 116000}, 900},
	"flat_usdjpy": {"flat at 19.09 (the 19.0/19.1 defect)", assetSpec{Display: "USDJPY", Interval: "1h"}, trendRead{OK: true,
		State: trendFlat, Raw: trendFlat, ADX: 19.09,
		EMA20: 147.30, EMA50: 147.10, EMA200: 146.80, Last: 147.02}, 0.12},
	"flat_mixed": {"flat, price across EMA50, structure mixed (live USDJPY shape)", assetSpec{Display: "USDJPY", Interval: "1h"}, trendRead{OK: true,
		State: trendFlat, Raw: trendFlat, Structure: "mixed", ADX: 19.05,
		EMA20: 154.30, EMA50: 154.18, EMA200: 155.08, Last: 154.42}, 0.12},
	"conflict": {"conflict: price above EMA50 under bearish EMAs", eurSpec1h, trendRead{OK: true,
		State: trendConflict, Raw: trendConflict, ADX: 30.14,
		EMA20: 1.1590, EMA50: 1.1580, EMA200: 1.1646, Last: 1.1600}, 0.001},
	"demoted": {"EMA/ADX up, LH/LL structure → grey", btcSpec4h, trendRead{OK: true,
		State: trendGrey, Raw: trendUp, Structure: "lh_ll", StructureDemoted: demoted(trendUp, "lh_ll"),
		ADX: 58.8, EMA20: 63900, EMA50: 63200, EMA200: 60000, Last: 64500}, 800},
	"ema_equal": {"EMA50 == EMA200 at ADX 31 (no direction)", btcSpec4h, trendRead{OK: true,
		State: trendConflict, Raw: trendConflict, ADX: 31,
		EMA20: 101, EMA50: 100, EMA200: 100, Last: 102}, 1},
	"ema_equal_flat": {"EMA50 == EMA200, flat", btcSpec4h, trendRead{OK: true,
		State: trendFlat, Raw: trendFlat, ADX: 12,
		EMA20: 101, EMA50: 100, EMA200: 100, Last: 99}, 1},
}

func renderFixture(f trendFixture) Card {
	r := f.r
	r.ATR = f.atr
	return trendCardFrom(r, f.spec, trendTestTime)
}

// Non-finite indicator values must never reach a card. Prices near the
// float64 ceiling make the ADX sums overflow (NaN ADX → the state machine
// fell through to "conflict"), and a worse variant produced a confirmed trend
// with a +Inf invalidation that failed JSON encoding with a 500.
func TestTrendReadRejectsNonFinite(t *testing.T) {
	bars := 250
	start := time.Now().Unix() - int64(bars+2)*14400
	candles := make([]types.OHLCVCandle, bars)
	for i := range candles {
		p := 1e307
		if i%2 == 1 {
			p = 1.7e308
		}
		candles[i] = types.OHLCVCandle{Time: start + int64(i)*14400, Open: p, High: p, Low: p, Close: p, Volume: 1}
	}
	if r := trendReadOf(candles); r.OK {
		t.Fatalf("overflowing series must not produce a reading: %+v", r)
	}

	stubBinanceCandles(t, candles)
	c := NewAgents(NewBackendClient("http://127.0.0.1:1")).TrendCard(context.Background(), btcSpec)
	if cardEnvelope(c).OK || c.State != "" {
		t.Errorf("card must degrade, got state %q verdict %q", c.State, c.Verdict)
	}
	b, err := json.Marshal(cardEnvelope(c))
	if err != nil {
		t.Fatalf("degraded card must still encode: %v", err)
	}
	for _, bad := range []string{"NaN", "Inf"} {
		if strings.Contains(string(b), bad) {
			t.Errorf("%s leaked into the envelope: %s", bad, b)
		}
	}

	// The guard itself, value by value.
	ok := trendRead{OK: true, ADX: 30, RSI: 50, EMA20: 1, EMA50: 1, EMA200: 1, Last: 1, ATR: 0.1}
	if !trendReadFinite(ok) {
		t.Fatal("finite read rejected")
	}
	for name, mut := range map[string]func(*trendRead){
		"ADX NaN":    func(r *trendRead) { r.ADX = math.NaN() },
		"RSI NaN":    func(r *trendRead) { r.RSI = math.NaN() },
		"EMA20 Inf":  func(r *trendRead) { r.EMA20 = math.Inf(1) },
		"EMA50 -Inf": func(r *trendRead) { r.EMA50 = math.Inf(-1) },
		"EMA200 NaN": func(r *trendRead) { r.EMA200 = math.NaN() },
		"Last Inf":   func(r *trendRead) { r.Last = math.Inf(1) },
		"ATR NaN":    func(r *trendRead) { r.ATR = math.NaN() },
		// finite inputs, overflowing level: a downtrend's max(EMA) + ATR = +Inf
		"level overflow": func(r *trendRead) {
			r.State, r.EMA50, r.EMA200, r.ATR = trendDown, 1.7e308, 1.6e308, 1.7e308
		},
	} {
		r := ok
		mut(&r)
		if trendReadFinite(r) {
			t.Errorf("%s: must be rejected", name)
		}
	}
}

// The confirmed downtrend, byte for byte.
func TestTrendCardGoldenConfirmedDown(t *testing.T) {
	got := renderFixture(trendFixtures["down"]).RenderHTML()
	want := "🔴 <b>Trend Agent</b> · EURUSD\n" +
		"<b>Confirmed DOWNTREND · 1h</b>\n" +
		"• Price 1.1551 — 0.1% below the pullback zone 1.1559–1.1580\n" +
		"• Confirmation holds while all four conditions stay ✓; any ✗ withdraws it\n" +
		"• Invalidated by a closed 1h candle above 1.1619 (+0.6%, 1 ATR over the EMA cluster)\n" +
		"• Why: ADX 46.5 ≥ 25 ✓ · EMA50 &lt; EMA200 ✓ · close &lt; EMA50 ✓ · structure not against ✓ (not determined)\n" +
		"\n<i>Analytics, not financial advice · AlphaVizor · 2026-09-15 00:00 UTC</i>"
	if got != want {
		t.Errorf("confirmed down card:\n got: %q\nwant: %q", got, want)
	}
}

// The grey gold card: no invalidation line (the old "reference: below 4348"
// was already crossed at 4329); what confirmation takes, and what is ✗.
func TestTrendCardGoldenGreyGold(t *testing.T) {
	got := renderFixture(trendFixtures["grey"]).RenderHTML()
	want := "⚪ <b>Trend Agent</b> · GOLD · COMEX GC=F\n" +
		"<b>Grey zone · 1h — trend forming, not confirmed</b>\n" +
		"• Price 4329 — 0.9% below EMA50 4367 and 2.4% below EMA200 4433\n" +
		"• Confirms as a DOWNTREND only when all four conditions are ✓ (now ✗: ADX)\n" +
		"• Why: ADX 22.2 &lt; 25 ✗ · EMA50 &lt; EMA200 ✓ · close &lt; EMA50 ✓ · structure not against ✓ (not determined)\n" +
		"\n<i>Analytics, not financial advice · AlphaVizor · 2026-09-15 00:00 UTC</i>"
	if got != want {
		t.Errorf("grey gold card:\n got: %q\nwant: %q", got, want)
	}
}

func TestTrendCardPerState(t *testing.T) {
	cases := []struct {
		key     string
		verdict string
		facts   []string
	}{
		{"flat", "Flat · 4h — no trend to read (ADX under 20)", []string{
			"Price 116000 — 0.9% above EMA50 115000 and 5.2% above EMA200 110000",
			"Confirms as an UPTREND only when all four conditions are ✓ (now ✗: ADX)",
			"Why: ADX 17.3 < 25 ✗ · EMA50 > EMA200 ✓ · close > EMA50 ✓ · structure not against ✓ (not determined)",
		}},
		{"flat_mixed", "Flat · 1h — no trend to read (ADX under 20)", []string{
			"Price 154.42 — 0.2% above EMA50 154.18 and 0.4% below EMA200 155.08",
			"Confirms as a DOWNTREND only when all four conditions are ✓ (now ✗: ADX, close vs EMA50, structure)",
			"Why: ADX 19.0 < 25 ✗ · EMA50 < EMA200 ✓ · close < EMA50 ✗ · structure not against ✗ (swings not aligned)",
		}},
		{"up_inside", "Confirmed UPTREND · 4h", []string{
			"Price 63500 — inside the pullback zone 63000–64000",
			"Confirmation holds while all four conditions stay ✓; any ✗ withdraws it",
			"Invalidated by a closed 4h candle below 59200 (-6.8%, 1 ATR under the EMA cluster)",
			"Why: ADX 31.2 ≥ 25 ✓ · EMA50 > EMA200 ✓ · close > EMA50 ✓ · structure not against ✓",
		}},
		{"up_above", "Confirmed UPTREND · 4h", []string{
			"Price 65000 — 1.5% above the pullback zone 63000–64000",
			"Confirmation holds while all four conditions stay ✓; any ✗ withdraws it",
			"Invalidated by a closed 4h candle below 59200 (-8.9%, 1 ATR under the EMA cluster)",
			"Why: ADX 31.2 ≥ 25 ✓ · EMA50 > EMA200 ✓ · close > EMA50 ✓ · structure not against ✓",
		}},
		{"conflict", "Indicator conflict · 1h — ADX ≥ 25 but the EMA conditions disagree", []string{
			"Price 1.1600 — 0.2% above EMA50 1.1580 and 0.4% below EMA200 1.1646",
			"Confirms as a DOWNTREND only when all four conditions are ✓ (now ✗: close vs EMA50)",
			"Why: ADX 30.1 ≥ 25 ✓ · EMA50 < EMA200 ✓ · close < EMA50 ✗ · structure not against ✓ (not determined)",
		}},
		{"demoted", "Grey zone · 4h — not confirmed: swing structure against the trend", []string{
			"Price 64500 — 2.0% above EMA50 63200 and 7.0% above EMA200 60000",
			"Confirms as an UPTREND only when all four conditions are ✓ (now ✗: structure)",
			"Why: ADX 58.8 ≥ 25 ✓ · EMA50 > EMA200 ✓ · close > EMA50 ✓ · structure not against ✗ (runs against the trend)",
		}},
		{"ema_equal", "Indicator conflict · 4h — EMA50 equals EMA200, no direction", []string{
			"Price 102.00 — 2.0% above EMA50 100.00 and 2.0% above EMA200 100.00",
			"No direction to confirm while EMA50 equals EMA200",
			"Why: ADX 31.0 ≥ 25 ✓ · EMA50 = EMA200 ✗ (no direction) · close vs EMA50 ✗ · structure not against ✗",
		}},
	}
	for _, tc := range cases {
		t.Run(tc.key, func(t *testing.T) {
			c := renderFixture(trendFixtures[tc.key])
			if c.Verdict != tc.verdict {
				t.Errorf("verdict:\n got %q\nwant %q", c.Verdict, tc.verdict)
			}
			if strings.Join(c.Facts, "\n") != strings.Join(tc.facts, "\n") {
				t.Errorf("facts:\n got %q\nwant %q", c.Facts, tc.facts)
			}
		})
	}
}

// The checklist IS the rule: always exactly four items — the inputs of
// classifyTrend + structureDemotion — each ✓/✗ matching the rule's own
// evaluation, and "all four ✓" holds exactly when the rule confirms the
// direction. Driven over every combination of the inputs, EMA50 == EMA200
// included.
func TestTrendChecklistMatchesRule(t *testing.T) {
	type ema struct{ e50, e200 float64 }
	against := map[string]string{trendUp: "lh_ll", trendDown: "hh_hl"}
	for _, adx := range []float64{12, 19.99, 20, 24.96, 25, 40} {
		for _, e := range []ema{{105, 100}, {95, 100}, {100, 100}} {
			// 95 and 105 equal a directional EMA50, 100 the equal-EMA case:
			// the strict < / > edges of the rule are exercised.
			for _, last := range []float64{90, 95, 99, 100, 101, 105, 110} {
				for _, structure := range []string{"", "hh_hl", "lh_ll", "mixed"} {
					raw := classifyTrend(adx, e.e50, e.e200, last)
					state := raw
					if structureDemotion(state, structure) != "" {
						state = trendGrey
					}
					r := trendRead{OK: true, State: state, Raw: raw, Structure: structure,
						StructureDemoted: structureDemotion(raw, structure),
						ADX:              adx, EMA20: e.e50, EMA50: e.e50, EMA200: e.e200, Last: last}
					v := trendView{r: r, tf: "4h"}
					dir := r.emaDirection()
					checks := v.checks(dir)
					if len(checks) != 4 {
						t.Fatalf("checklist must always have 4 items, got %d (ema %v)", len(checks), e)
					}
					want := []bool{
						adx >= 25,
						(dir == trendUp && e.e50 > e.e200) || (dir == trendDown && e.e50 < e.e200),
						(dir == trendUp && last > e.e50) || (dir == trendDown && last < e.e50),
						dir != "" && structure != against[dir] && structure != "mixed",
					}
					all := true
					for i, c := range checks {
						if c.ok != want[i] {
							t.Errorf("adx %v ema %v last %v %q: item %q ok=%v, want %v", adx, e, last, structure, c.label, c.ok, want[i])
						}
						all = all && c.ok
					}
					if all != (dir != "" && state == dir) {
						t.Errorf("adx %v ema %v last %v %q: all ✓ = %v but rule state = %q (dir %q)", adx, e, last, structure, all, state, dir)
					}
					// A failing structure says why, without pattern tokens.
					if dir != "" && !checks[3].ok {
						wantNote := "(runs against the trend)"
						if structure == "mixed" {
							wantNote = "(swings not aligned)"
						}
						if checks[3].note != wantNote {
							t.Errorf("structure %q toward %s: note %q, want %q", structure, dir, checks[3].note, wantNote)
						}
					}
					// The rendered line carries exactly these four marks.
					line := v.whyLine()
					if n := strings.Count(line, "✓") + strings.Count(line, "✗"); n != 4 {
						t.Errorf("why line must carry 4 marks, got %d: %q", n, line)
					}
				}
			}
		}
	}
}

// confirmClaim matches a sentence that names a resulting CONFIRMED state.
var confirmClaim = regexp.MustCompile(`(?i)\bconfirms (as|in)\b|\bconfirmed (up|down)trend\b`)

// allTrendCards renders every fixture — confirmed, grey, flat, conflict,
// structure-demoted and EMA-equal.
func allTrendCards() map[string]Card {
	out := map[string]Card{}
	for k, f := range trendFixtures {
		out[k] = renderFixture(f)
	}
	return out
}

// No sentence may promise a confirmed state off a partial condition set: a
// confirmation is named only together with the whole checklist.
func TestTrendTextNeverPromisesPartialConfirmation(t *testing.T) {
	for key, c := range allTrendCards() {
		lines := append(append([]string{}, c.Facts...), c.Blocks.Scenarios...)
		lines = append(lines, c.Blocks.WhyLevel)
		for _, l := range lines {
			if confirmClaim.MatchString(l) && !strings.Contains(l, "all four conditions") {
				t.Errorf("%s: names a confirmed state without the full checklist: %q", key, l)
			}
		}
		// Conditions are enumerated only in the checklist: no other line
		// mentions EMA200 as a condition or a structure token.
		for _, l := range c.Facts {
			if strings.HasPrefix(l, "Why: ") || strings.HasPrefix(l, "Price ") {
				continue
			}
			// "(now ✗: ADX, structure)" names failing checklist ITEMS by their
			// short names — a reference to the checklist, not a re-listing.
			l, _, _ = strings.Cut(l, " (now ✗: ")
			for _, tok := range []string{"EMA200", "structure", "ADX ≥", "ADX <"} {
				if strings.Contains(l, tok) && !strings.Contains(l, "EMA50 equals EMA200") {
					t.Errorf("%s: %q re-lists condition %q outside the checklist", key, l, tok)
				}
			}
		}
	}
}

// Every scenario is an "If …, the reading …" statement, EMA-equal included.
func TestTrendScenariosForm(t *testing.T) {
	for key, c := range allTrendCards() {
		s := c.Blocks.Scenarios
		if len(s) != 2 {
			t.Errorf("%s: want exactly two scenarios, got %q", key, s)
		}
		for _, sc := range s {
			if !strings.HasPrefix(sc, "If ") || !strings.Contains(sc, "the reading") {
				t.Errorf("%s: scenario not in the \"If …, the reading …\" form: %q", key, sc)
			}
		}
	}
	for _, key := range []string{"ema_equal", "ema_equal_flat"} {
		if s := allTrendCards()[key].Blocks.Scenarios[0]; s != "If EMA50 and EMA200 separate and all four conditions turn ✓, the reading confirms in that direction" {
			t.Errorf("%s scenario: %q", key, s)
		}
	}
}

// trendWorstCases are the widest inputs the text can meet: the longest asset
// label, seven-digit prices, ADX 100.0, the longest structure reason, the
// longest failing list, big distances.
func trendWorstCases() map[string]Card {
	big := func(state, raw, structure string, adx, e20, e50, e200, last float64) Card {
		r := trendRead{OK: true, State: state, Raw: raw, Structure: structure,
			StructureDemoted: structureDemotion(raw, structure),
			ADX:              adx, EMA20: e20, EMA50: e50, EMA200: e200, Last: last, ATR: 99999}
		return trendCardFrom(r, goldSpec1h, trendTestTime)
	}
	return map[string]Card{
		"up, not determined":   big(trendUp, trendUp, "", 100, 1234000, 1230000, 1100000, 1300000),
		"down, far below zone": big(trendDown, trendDown, "", 100, 1234000, 1240000, 1300000, 1000000),
		"demoted, against":     big(trendGrey, trendUp, "lh_ll", 100, 1234000, 1230000, 1100000, 1300000),
		"demoted, mixed":       big(trendGrey, trendUp, "mixed", 100, 1234000, 1230000, 1100000, 1300000),
		"flat, all failing":    big(trendFlat, trendFlat, "mixed", 19.99, 1234000, 1230000, 1300000, 1234567),
		"grey, not determined": big(trendGrey, trendGrey, "", 24.99, 1234000, 1230000, 1100000, 1000000),
		"conflict, against":    big(trendConflict, trendConflict, "hh_hl", 100, 1234000, 1230000, 1300000, 1234567),
		"ema equal":            big(trendConflict, trendConflict, "", 100, 1234000, 1230000, 1230000, 1234567),
		"ema equal, flat":      big(trendFlat, trendFlat, "", 12, 1234000, 1230000, 1230000, 1234567),
	}
}

// Readability: every line of trend text fits trendFactMaxRunes — the card's
// facts, every content block and the landing conclusion — on today's
// fixtures and on the worst-case inputs.
func TestTrendTextFitsOneLine(t *testing.T) {
	cards := allTrendCards()
	for k, c := range trendWorstCases() {
		cards["worst: "+k] = c
	}
	for key, c := range cards {
		lines := append([]string{}, c.Facts...)
		b := c.Blocks
		lines = append(lines, b.WhatHappened, b.WhyLevel, b.Regime, c.trendConclusion)
		lines = append(lines, b.Scenarios...)
		if b.Invalidates != nil {
			lines = append(lines, *b.Invalidates)
		}
		for _, l := range lines {
			if n := utf8.RuneCountInString(l); n > trendFactMaxRunes {
				t.Errorf("%s: %d chars (max %d): %q", key, n, trendFactMaxRunes, l)
			}
		}
	}
}

// A.1: one ADX value per card, floored — the USDJPY card printed 19.0 in the
// verdict and 19.1 in the facts. It now lives only in the checklist.
func TestTrendCardPrintsADXOnce(t *testing.T) {
	adxNum := regexp.MustCompile(`ADX (\d+\.\d)`)
	for key, f := range trendFixtures {
		html := renderFixture(f).RenderHTML()
		m := adxNum.FindAllStringSubmatch(html, -1)
		if len(m) != 1 {
			t.Errorf("%s: ADX value printed %d times, want exactly 1:\n%s", key, len(m), html)
			continue
		}
		if m[0][1] != adxShown(f.r.ADX) {
			t.Errorf("%s: printed ADX %s, want floored %s", key, m[0][1], adxShown(f.r.ADX))
		}
	}
	if got := renderFixture(trendFixtures["flat_usdjpy"]).Facts[2]; !strings.HasPrefix(got, "Why: ADX 19.0 < 25 ✗") {
		t.Errorf("usdjpy checklist: %q", got)
	}
}

func TestADXShownFloorsBothWays(t *testing.T) {
	for in, want := range map[float64]string{19.99: "19.9", 19.09: "19.0", 24.96: "24.9", 25: "25.0", 25.04: "25.0", 46.58: "46.5"} {
		if got := adxShown(in); got != want {
			t.Errorf("adxShown(%v) = %q, want %q", in, got, want)
		}
	}
}

// Unconfirmed cards carry no invalidation anywhere — nothing to invalidate:
// no card line, no blocks.invalidates, no levels object (decided 2026-09-15).
func TestTrendCardNoInvalidationWhenUnconfirmed(t *testing.T) {
	for key, f := range trendFixtures {
		c := renderFixture(f)
		if c.State == trendUp || c.State == trendDown {
			continue
		}
		for _, l := range c.Facts {
			if strings.Contains(strings.ToLower(l), "invalidat") {
				t.Errorf("%s: unconfirmed card must not word an invalidation: %q", key, l)
			}
		}
		if c.Blocks == nil || c.Blocks.Invalidates != nil {
			t.Errorf("%s: blocks.invalidates must be null, got %+v", key, c.Blocks)
		}
		if c.Levels != nil {
			t.Errorf("%s: an unconfirmed card ships no levels at all, got %+v", key, c.Levels)
		}
		if c.Blocks.WhyLevel != "No invalidation level: the trend is not confirmed. Confirmation needs all four checklist conditions" {
			t.Errorf("%s: why_level: %q", key, c.Blocks.WhyLevel)
		}
	}
}

// Wording rules on every card and every content block.
func TestTrendCardWordingRules(t *testing.T) {
	banned := []string{
		"above 25",                // the rule confirms AT 25
		"structure is broken",     // the level is EMA ± ATR, not swing structure
		"RSI",                     // not part of the rule
		"HH/HL", "LH/LL", "mixed", // no structure claims; the checklist says "not against"
		"should", "advised", "recommend", "avoid", "stand aside",
		" will ", "target", "probab", "likely", "BUY", "SELL",
	}
	cards := allTrendCards()
	for k, c := range trendWorstCases() {
		cards["worst: "+k] = c
	}
	for key, c := range cards {
		b, err := json.Marshal(c.Blocks)
		if err != nil {
			t.Fatal(err)
		}
		all := c.RenderHTML() + "\n" + string(b) + "\n" + c.trendConclusion
		for _, w := range banned {
			if strings.Contains(all, w) {
				t.Errorf("%s: %q must not appear:\n%s", key, w, all)
			}
		}
		if c.State == trendUp || c.State == trendDown {
			if !strings.Contains(all, "a closed ") || !strings.Contains(all, " candle") {
				t.Errorf("%s: the level must be checked on a closed candle:\n%s", key, all)
			}
		}
	}
}

// The content blocks: withdrawal vs invalidation worded apart, and the field
// exists on trend envelopes only.
func TestTrendBlocks(t *testing.T) {
	b := renderFixture(trendFixtures["down"]).Blocks
	wantScen := []string{
		"If all four conditions stay ✓, the reading stays a confirmed downtrend",
		"If a 1h candle closes above 1.1619, the reading is invalidated as a downtrend (1 ATR over the EMA cluster)",
	}
	if strings.Join(b.Scenarios, "|") != strings.Join(wantScen, "|") {
		t.Errorf("scenarios:\n got %q\nwant %q", b.Scenarios, wantScen)
	}
	if b.Invalidates == nil || *b.Invalidates != "A closed 1h candle above 1.1619 invalidates the downtrend idea" {
		t.Errorf("invalidates: %v", b.Invalidates)
	}
	if b.WhyLevel != "1.1619 = max(EMA50, EMA200) + 1 ATR(14) on a closed 1h candle; a checklist ✗ withdraws confirmation sooner" {
		t.Errorf("why_level: %q", b.WhyLevel)
	}
	if w := renderFixture(trendFixtures["up_inside"]).Blocks.WhyLevel; !strings.HasPrefix(w, "59200 = min(EMA50, EMA200) − 1 ATR(14) on a closed 4h candle") {
		t.Errorf("uptrend why_level: %q", w)
	}
	if b.WhatHappened != "Confirmed downtrend on 1h: price 1.1551 — 0.1% below the pullback zone 1.1559–1.1580." {
		t.Errorf("what_happened: %q", b.WhatHappened)
	}
	if b.Regime != "confirmed downtrend · 1h · ADX 46.5" {
		t.Errorf("regime: %q", b.Regime)
	}
	for key, want := range map[string][]string{
		"grey":     {"If all four conditions turn ✓ for a downtrend, the reading confirms as a downtrend", "If ADX falls below 20, the reading returns to flat"},
		"flat":     {"If all four conditions turn ✓ for an uptrend, the reading confirms as an uptrend", "If ADX stays below 20, the reading stays flat"},
		"conflict": {"If all four conditions turn ✓ for a downtrend, the reading confirms as a downtrend", "If ADX falls below 25, the reading turns grey (flat below 20)"},
		"demoted":  {"If all four conditions turn ✓ for an uptrend, the reading confirms as an uptrend", "If ADX falls below 25, the reading turns grey (flat below 20)"},
	} {
		if got := renderFixture(trendFixtures[key]).Blocks.Scenarios; strings.Join(got, "|") != strings.Join(want, "|") {
			t.Errorf("%s scenarios:\n got %q\nwant %q", key, got, want)
		}
	}

	envJSON := func(c Card) string {
		b, err := json.Marshal(cardEnvelope(c))
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	if s := envJSON(renderFixture(trendFixtures["grey"])); !strings.Contains(s, `"blocks":{"what_happened":`) || !strings.Contains(s, `"invalidates":null`) {
		t.Errorf("grey trend envelope blocks: %s", s)
	}
	if s := envJSON(Card{Agent: "Momentum Agent", Asset: "BTC", Verdict: "BULLISH"}); strings.Contains(s, "blocks") {
		t.Errorf("non-trend envelope must omit blocks: %s", s)
	}
	if s := envJSON(insufficientCard(btcSpec, "Trend Agent", "Trend", keyTrend, "", "EMA200/ADX(14)")); strings.Contains(s, "blocks") {
		t.Errorf("degraded trend envelope must omit blocks: %s", s)
	}
}

// Landing conclusion: an invalidation only for confirmed trends; flat is "no
// trend to read", never an "unconfirmed trend"; the rest name the ✗ checklist
// items instead of saying "nothing leans".
func TestShowcaseConclusionForTrend(t *testing.T) {
	cards := allTrendCards()
	for key, want := range map[string]string{
		"flat":      "For a trader there is no trend to read on BTC: ADX 17.3 is under 20.",
		"grey":      "For a trader this is an unconfirmed trend on GOLD · COMEX GC=F; failing: ADX.",
		"demoted":   "For a trader this is an unconfirmed trend on BTC; failing: structure.",
		"conflict":  "For a trader this is an unconfirmed trend on EURUSD; failing: close vs EMA50.",
		"ema_equal": "For a trader this is an unconfirmed trend on BTC: EMA50 equals EMA200, no direction to confirm.",
	} {
		if got := conclusionFor(cards[key]); got != want {
			t.Errorf("%s conclusion:\n got %q\nwant %q", key, got, want)
		}
	}
	if s := conclusionFor(cards["down"]); !strings.HasSuffix(s, " A closed 1h candle above 1.1619 invalidates the downtrend idea.") {
		t.Errorf("confirmed conclusion: %q", s)
	}
	if s := conclusionFor(Card{Asset: "BTC", Emoji: emojiNeutral}); !strings.Contains(s, "nothing in the numbers above leans either way") {
		t.Errorf("non-trend neutral conclusion changed: %q", s)
	}
}

// The gold agent embeds the trend verdict mid-sentence (lowerFirst); it must
// stay a readable clause in every state.
func TestGoldEmbedsTrendVerdictReadably(t *testing.T) {
	for key, want := range map[string]string{
		"flat":     "flat · 1d — no trend to read (ADX under 20)",
		"grey":     "grey zone · 1d — trend forming, not confirmed",
		"conflict": "indicator conflict · 1d — ADX ≥ 25 but the EMA conditions disagree",
		"demoted":  "grey zone · 1d — not confirmed: swing structure against the trend",
		"down":     "confirmed DOWNTREND · 1d",
	} {
		if got := lowerFirst(trendVerdict(trendFixtures[key].r, goldDailySpec.Interval)); got != want {
			t.Errorf("%s: %q, want %q", key, got, want)
		}
	}
}
