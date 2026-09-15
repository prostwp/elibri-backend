package demobot

// trend_text_property_test.go — the trend text's length and number-format
// guarantees must hold for ANY finite reading, not only for the fixtures:
// every fact, every content block and the landing conclusion stay within
// trendFactMaxRunes, and no "NaN", "Inf" or exponent notation ever reaches a
// reader. Inputs are pathological on purpose (prices from 1e-6 to 1e7, EMAs
// orders of magnitude apart, ADX 0–100, EMA50 == EMA200, zero ATR).

import (
	"encoding/json"
	"math"
	"math/rand"
	"regexp"
	"testing"
	"unicode/utf8"
)

var badNumberText = regexp.MustCompile(`NaN|[+-]?Inf\b|\de[+-]?\d`)

// trendTexts is every reader-facing string one card produces.
func trendTexts(c Card) []string {
	out := append([]string{c.Verdict}, c.Facts...)
	if b := c.Blocks; b != nil {
		out = append(out, b.WhatHappened, b.WhyLevel, b.Regime)
		out = append(out, b.Scenarios...)
		if b.Invalidates != nil {
			out = append(out, *b.Invalidates)
		}
	}
	if c.trendConclusion != "" {
		out = append(out, c.trendConclusion)
	}
	return out
}

func checkTrendTexts(t *testing.T, label string, r trendRead) {
	t.Helper()
	c := trendCardFrom(r, goldSpec1h, trendTestTime) // longest asset label
	for _, s := range trendTexts(c) {
		if n := utf8.RuneCountInString(s); n > trendFactMaxRunes {
			t.Errorf("%s: %d chars (max %d): %q", label, n, trendFactMaxRunes, s)
		}
		if badNumberText.MatchString(s) {
			t.Errorf("%s: non-finite or exponent number in text: %q", label, s)
		}
	}
	if _, err := json.Marshal(cardEnvelope(c)); err != nil {
		t.Errorf("%s: envelope does not encode: %v", label, err)
	}
}

// readFor builds a read the way trendReadOf does: state from the rule, then
// the structure gate.
func readFor(adx, e20, e50, e200, last, atr float64, structure string) trendRead {
	raw := classifyTrend(adx, e50, e200, last)
	state := raw
	demotedBy := structureDemotion(raw, structure)
	if demotedBy != "" {
		state = trendGrey
	}
	return trendRead{OK: true, State: state, Raw: raw, Structure: structure, StructureDemoted: demotedBy,
		ADX: adx, EMA20: e20, EMA50: e50, EMA200: e200, Last: last, ATR: atr}
}

// TestTrendTextBoundedForRealisticPrices: the ≤110-char budget is guaranteed
// for prices, EMAs and ATR in 1e-6…1e7 — about 100× above any asset the agent
// serves (BTC ~1e5). It is NOT a guarantee for arbitrary finite float64: prices
// print in full, so a 1e100 level would widen the line. Out-of-range inputs are
// not a live-feed scenario; if an asset above 1e7 is ever added, extend here.
func TestTrendTextBoundedForRealisticPrices(t *testing.T) {
	structures := []string{"", "hh_hl", "lh_ll", "mixed"}

	// Hand-picked extremes first: the widest percentages and numbers.
	// (Within this domain the widest printed number is a nine-character
	// level such as "-20000000": EMA 1e7 minus an ATR of 2e7.)
	t.Run("extremes", func(t *testing.T) {
		for _, e := range []struct{ last, e50, e200 float64 }{
			{1e-6, 1e7, 1e7 / 2}, {1e-6, 1e7 / 2, 1e7}, {1e7, 1e-6, 2e-6}, {1e7, 2e-6, 1e-6},
			{1e-6, 1e-6, 1e7}, {1e7, 1e7, 1e-6}, {1e-6, 5e6, 5e6}, {9999999, 1e-6, 1e-6},
		} {
			for _, adx := range []float64{0, 19.99, 24.99, 25, 100} {
				for _, s := range structures {
					for _, atr := range []float64{0, 1e-9, 1e7} {
						checkTrendTexts(t, "extreme", readFor(adx, e.e50*1.01, e.e50, e.e200, e.last, atr, s))
					}
				}
			}
		}

	})

	// Then random pathological readings, fixed seed so failures reproduce.
	t.Run("random", func(t *testing.T) {
		rng := rand.New(rand.NewSource(20260915))
		logU := func(lo, hi float64) float64 {
			return math.Exp(math.Log(lo) + rng.Float64()*(math.Log(hi)-math.Log(lo)))
		}
		for i := 0; i < 20000; i++ {
			last := logU(1e-6, 1e7)
			e50 := logU(1e-6, 1e7)
			e200 := logU(1e-6, 1e7)
			if rng.Intn(5) == 0 {
				e200 = e50 // the EMA-equal edge
			}
			atr := 0.0
			if rng.Intn(6) != 0 {
				atr = logU(1e-9, 1e7)
			}
			adx := rng.Float64() * 100
			r := readFor(adx, logU(1e-6, 1e7), e50, e200, last, atr, structures[rng.Intn(len(structures))])
			if !trendReadFinite(r) {
				continue // trendReadOf would refuse this read before any text exists
			}
			checkTrendTexts(t, "random", r)
			if t.Failed() {
				t.Fatalf("stopping at iteration %d: last=%g ema50=%g ema200=%g atr=%g adx=%g", i, last, e50, e200, atr, adx)
			}
		}
	})
}

func TestPctAwayBounded(t *testing.T) {
	for _, c := range []struct {
		price, level float64
		want, signed string
	}{
		{100, 100.01, "<0.1%", "<0.1%"},
		{100, 150, "50.0%", "+50.0%"},
		{100, 1099.9, "999.9%", "+999.9%"},
		{100, 1100, ">999%", ">+999%"},
		{1e-6, 1e7, ">999%", ">+999%"},
		{1e7, 1e-6, "100.0%", "-100.0%"},
		{-5, 1e7, ">999%", ">+999%"},
		{0, 5, "n/a", "n/a"},
	} {
		if got := pctAway(c.price, c.level); got != c.want {
			t.Errorf("pctAway(%g, %g) = %q, want %q", c.price, c.level, got, c.want)
		}
		if got := signedPct(c.price, c.level); got != c.signed {
			t.Errorf("signedPct(%g, %g) = %q, want %q", c.price, c.level, got, c.signed)
		}
	}
}
