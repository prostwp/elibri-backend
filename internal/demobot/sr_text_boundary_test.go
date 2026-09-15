package demobot

// sr_text_boundary_test.go — the S/R card's hard edges, run through the FULL
// card paths (srCardOf, srCardFrom, srNoLevelsCard, the insufficient and
// offline builders, the FX decoration) rather than sampled:
//   - the volume line only names a level whose six compared pivots all carry
//     volume, while levels[].weakening stays exactly what the rule computes;
//   - every visible line stays within srFactMaxRunes at worst-case inputs.

import (
	"math"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// srSevenPivotCandles is a period-12 zigzag around 2500 on 4h bars: swing
// highs of exactly 2560.3 at i%12==3 (seven of them, i = 3 … 75) and swing
// lows of 2440.3 at i%12==9 (seven, i = 9 … 81); the last close is 2518, so
// the high cluster is the resistance and the low cluster the support. The
// resistance pivots carry 500, 500, 500, 100 and then lastThree; every other
// bar 100.
func srSevenPivotCandles(lastThree [3]float64) []types.OHLCVCandle {
	offs := []float64{0, 18, 36, 54, 36, 18, 0, -18, -36, -54, -36, -18}
	pivotVol := []float64{500, 500, 500, 100, lastThree[0], lastThree[1], lastThree[2]}
	start := time.Date(2026, 8, 1, 0, 0, 0, 0, time.UTC).Unix()
	out := make([]types.OHLCVCandle, 86)
	for i := range out {
		p := 2500 + offs[i%12]
		v := 100.0
		if i%12 == 3 {
			v = pivotVol[i/12]
		}
		out[i] = types.OHLCVCandle{Time: start + int64(i)*14400, Open: p - 3, High: p + 6.3, Low: p - 5.7, Close: p, Volume: v}
	}
	return out
}

var srWeekday = time.Date(2026, 9, 15, 12, 0, 0, 0, time.UTC) // Tuesday: FX open
var srSaturday = time.Date(2026, 9, 12, 12, 0, 0, 0, time.UTC)

func TestSRVolumeLineNeedsAllSixComparedPivots(t *testing.T) {
	eth := assetTable["eth"]
	for _, tc := range []struct {
		name      string
		lastThree [3]float64
		wantLine  bool
	}{
		// Two of the last three pivots without volume: the rule's gate (volume
		// on ≥ half the pivots: 5 of 7) passes and averages the zeros as real
		// zero volume → weakening:true, but the card must not state it.
		{"two zeros in the last three", [3]float64{300, 0, 0}, false},
		// Same fade with all six compared pivots carrying volume → stated.
		{"all six carry volume", [3]float64{300, 300, 300}, true},
	} {
		candles := srSevenPivotCandles(tc.lastThree)
		_, ruleRes := supportResistance(candles, srWing, srTolPct)
		if len(ruleRes) != 1 || ruleRes[0].Touches != 7 || !ruleRes[0].Weakening {
			t.Fatalf("%s: fixture must give one established 7-pivot resistance the rule flags: %+v", tc.name, ruleRes)
		}
		c := srCardOf(eth, candles, srWeekday)
		lv := c.Levels.(SRLevels)
		if len(lv.Resistances) != 1 {
			t.Fatalf("%s: resistances %+v", tc.name, lv.Resistances)
		}
		p := lv.Resistances[0]
		if p.Class != "established" || p.Touches != 7 || p.Weakening != ruleRes[0].Weakening {
			t.Errorf("%s: levels[].weakening must stay the rule's %v: %+v", tc.name, ruleRes[0].Weakening, p)
		}
		joined := strings.Join(c.Facts, "\n")
		has := strings.Contains(joined, "lower volume")
		if has != tc.wantLine {
			t.Errorf("%s: volume line present=%v, want %v:\n%s", tc.name, has, tc.wantLine, joined)
		}
		if tc.wantLine && !strings.Contains(joined, "Last 3 pivots on lower volume than first 3: 2560.3") {
			t.Errorf("%s: volume line wording:\n%s", tc.name, joined)
		}
	}
}

// ── length budget at worst-case inputs ───────────────────────────────────────

// srVisibleLines is every line a reader can see on one S/R card: the header
// (agent · longest asset label), verdict, short, each fact, each blocks field
// and the footer source note.
func srVisibleLines(c Card) []string {
	head := c.Emoji + " " + c.Agent
	if c.Asset != "" {
		head += " · " + c.Asset
	}
	out := append([]string{head}, srTexts(c)...)
	if c.SourceNote != "" {
		out = append(out, c.SourceNote)
	}
	return out
}

func checkSRBudget(t *testing.T, label string, c Card) {
	t.Helper()
	for _, s := range srVisibleLines(c) {
		if n := utf8.RuneCountInString(s); n > srFactMaxRunes {
			t.Errorf("%s: %d chars (max %d): %q", label, n, srFactMaxRunes, s)
		}
		if badNumberText.MatchString(s) {
			t.Errorf("%s: non-finite or exponent number: %q", label, s)
		}
	}
	// The how-it-works text is the [ℹ️ How it works] callback alert, not a
	// card line: its budget is Telegram's 200-char cap.
	if n := utf8.RuneCountInString(c.HowItWorks); n > 200 {
		t.Errorf("%s: how-it-works %d chars (max 200)", label, n)
	}
}

// srWorst is a level with every counter at 9999, the flag on, and the widest
// last-touch date ("Sep 14": month abbreviations are all three letters).
func srWorst(raw float64) SRLevel {
	return SRLevel{Level: int(math.Round(raw)), Raw: raw, Touches: 9999, Strength: 9999,
		Holds: 9999, Breaks: 9999, Weakening: true, LastTouch: time.Date(2026, 9, 14, 0, 0, 0, 0, time.UTC)}
}

// withDecoration returns the card as served: FX/gold cards get the Yahoo note
// and, on a weekend, the closed banner as the first fact.
func withDecoration(t *testing.T, spec assetSpec, c Card, now time.Time) Card {
	t.Helper()
	if spec.Source != srcYahoo {
		return c
	}
	decorateFXAt(&c, now)
	if !isForexOpen(now) && (len(c.Facts) == 0 || c.Facts[0] != fxClosedBanner) {
		t.Errorf("%s: closed banner must be the first fact: %v", spec.Display, c.Facts)
	}
	if c.SourceNote == "" {
		t.Errorf("%s: yahoo source note missing", spec.Display)
	}
	return c
}

// TestSRCardBoundaryEveryPath: every card path at the widest inputs a served
// asset can produce — 7-significant-digit prices at the asset's own precision
// (BTC 9999999, ETH/gold 999999.9, USDJPY 99999.99, EURUSD/GBPUSD 999.9999),
// the widest distance that range allows (+900.0% / -90.0%) and "at price",
// 9999 pivots / reactions / breaks / window candles / swing points, all six
// shown levels on the volume line, the longest asset label (gold), the FX
// weekend banner and the Yahoo source note.
func TestSRCardBoundaryEveryPath(t *testing.T) {
	const n = 9999
	for key, spec := range assetTable {
		dec := srAssetDecimals[strings.ToUpper(key)]
		step := math.Pow(10, -float64(dec))
		max7 := math.Pow(10, float64(7-dec)) - step // 9999999 · 999999.9 · 999.9999
		min7 := math.Pow(10, float64(6-dec))        // 1000000 · 100000.0 · 100.0000
		mid := (max7 + min7) / 2
		top := []SRLevel{srWorst(max7), srWorst(max7 - step), srWorst(max7 - 2*step)}
		bottom := []SRLevel{srWorst(min7), srWorst(min7 + step), srWorst(min7 + 2*step)}
		full := srFullVolOf(top, bottom)

		cards := map[string]Card{
			"both sides":                      srCardFrom(spec, bottom, top, mid, n, trendTestTime, full),
			"resistance only, +900%":          srCardFrom(spec, nil, top, min7-step/4, n, trendTestTime, full),
			"support only, -90%":              srCardFrom(spec, bottom, nil, max7+step/4, n, trendTestTime, full),
			"at price (printed equal)":        srCardFrom(spec, bottom, top, max7-2*step-step/4, n, trendTestTime, full),
			"no significant levels":           srNoLevelsCard(spec, n, max7, n, trendTestTime),
			"insufficient history (short)":    srCardOf(spec, flatCandles(10, max7), srSaturday),
			"insufficient history (no swing)": srCardOf(spec, flatCandles(60, max7), srSaturday),
			"offline":                         assetOffline(spec, "S/R Agent", "S/R", keySR, howTexts[keySR]),
		}
		for label, c := range cards {
			for _, now := range []time.Time{srWeekday, srSaturday} {
				if strings.HasPrefix(label, "insufficient") || label == "offline" {
					checkSRBudget(t, spec.Display+" / "+label, c)
					continue
				}
				checkSRBudget(t, spec.Display+" / "+label+" / "+now.Weekday().String(), withDecoration(t, spec, c, now))
			}
		}
		// The volume line really carries all six levels at this width.
		six := false
		for _, f := range srCardFrom(spec, bottom, top, mid, n, trendTestTime, full).Facts {
			if strings.HasPrefix(f, "Last 3 pivots on lower volume than first 3: ") && strings.Count(f, ", ") == 5 {
				six = true
			}
		}
		if !six {
			t.Errorf("%s: volume line with six levels missing", spec.Display)
		}
		// And the full path from candles, decorated on a weekend.
		checkSRBudget(t, spec.Display+" / candles", srCardOf(spec, scaledZigzag(max7/2), srSaturday))
	}
}

func flatCandles(bars int, price float64) []types.OHLCVCandle {
	start := time.Date(2026, 8, 1, 0, 0, 0, 0, time.UTC).Unix()
	out := make([]types.OHLCVCandle, bars)
	for i := range out {
		out[i] = types.OHLCVCandle{Time: start + int64(i)*3600, Open: price, High: price, Low: price, Close: price}
	}
	return out
}

// scaledZigzag is srSevenPivotCandles' shape at another price scale.
func scaledZigzag(base float64) []types.OHLCVCandle {
	out := srSevenPivotCandles([3]float64{300, 300, 300})
	k := base / 2500
	for i := range out {
		out[i].Open *= k
		out[i].High *= k
		out[i].Low *= k
		out[i].Close *= k
	}
	return out
}
