package demobot

// sr_text_property_test.go — the S/R text's length and number-format
// guarantees must hold for any realistic read, not only the fixtures: every
// verdict, fact, short and content block stays within srFactMaxRunes, and no
// "NaN", "Inf" or exponent notation reaches a reader.
//
// Domain (stated, not arbitrary float64): unlisted assets with prices from
// 1e-3 to 1e7 and levels 0.2×…5× price (a window's clusters live inside the
// window's range); served assets at 0.1×…10× their live price; up to 999
// pivots and 999 reactions / 999 breaks per level.

import (
	"encoding/json"
	"math"
	"math/rand"
	"testing"
	"time"
	"unicode/utf8"
)

func checkSRTexts(t *testing.T, label string, c Card) {
	t.Helper()
	for _, s := range srTexts(c) {
		if n := utf8.RuneCountInString(s); n > srFactMaxRunes {
			t.Errorf("%s: %d chars (max %d): %q", label, n, srFactMaxRunes, s)
		}
		if badNumberText.MatchString(s) {
			t.Errorf("%s: non-finite or exponent number in text: %q", label, s)
		}
	}
	if n := utf8.RuneCountInString(c.HowItWorks); n > 200 {
		t.Errorf("how-it-works %d chars (max 200)", n)
	}
	if _, err := json.Marshal(cardEnvelope(c)); err != nil {
		t.Errorf("%s: envelope does not encode: %v", label, err)
	}
}

func TestSRTextBoundedForRealisticPrices(t *testing.T) {
	rng := rand.New(rand.NewSource(20260915))
	logU := func(lo, hi float64) float64 {
		return math.Exp(math.Log(lo) + rng.Float64()*(math.Log(hi)-math.Log(lo)))
	}
	live := map[string]float64{"btc": 78000, "eth": 2500, "eurusd": 1.15, "gbpusd": 1.35, "usdjpy": 155, "xauusd": 4344}
	unlisted := assetSpec{Display: "XYZ", Interval: "4h"}
	mkLevel := func(raw float64) SRLevel {
		l := SRLevel{Raw: raw, Level: int(math.Round(raw)),
			Touches: 1 + rng.Intn(999), Holds: rng.Intn(1000), Breaks: rng.Intn(1000),
			Weakening: rng.Intn(2) == 0}
		if rng.Intn(4) != 0 {
			l.LastTouch = time.Date(2026, time.Month(1+rng.Intn(12)), 1+rng.Intn(28), rng.Intn(24), 0, 0, 0, time.UTC)
		}
		if rng.Intn(5) == 0 {
			l.Touches = srStrongTouches // threshold edge
		}
		return l
	}
	for i := 0; i < 20000; i++ {
		spec, last := unlisted, logU(1e-3, 1e7)
		if rng.Intn(2) == 0 {
			key := []string{"btc", "eth", "eurusd", "gbpusd", "usdjpy", "xauusd"}[rng.Intn(6)]
			spec, last = assetTable[key], live[key]*logU(0.1, 10)
		}
		var sup, res []SRLevel
		for n := rng.Intn(4); n > 0; n-- {
			sup = append(sup, mkLevel(last*logU(0.2, 0.9999999)))
		}
		for n := rng.Intn(4); n > 0; n-- {
			res = append(res, mkLevel(last*logU(1.0000001, 5)))
		}
		if len(sup)+len(res) == 0 {
			continue // SRCard serves the no-levels finding instead
		}
		checkSRTexts(t, spec.Display, srCardFrom(spec, sup, res, last, 1+rng.Intn(1000), trendTestTime, srFullVolOf(sup, res)))
		if t.Failed() {
			t.Fatalf("stopping at iteration %d: spec=%s last=%g sup=%+v res=%+v", i, spec.Display, last, sup, res)
		}
	}
}
