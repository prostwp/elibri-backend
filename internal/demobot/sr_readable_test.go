package demobot

// sr_readable_test.go — the S/R card as a reader sees it, rendered through the
// pure builder (srCardFrom) on fixtures shaped on the six live cards of
// 2026-09-15. Rules are pinned elsewhere (sr_b4_test.go); nothing here may
// change them.

import (
	"encoding/json"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

func srAt(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		panic(err)
	}
	return t
}

func lvl(raw float64, touches int, strength float64, holds, breaks int, last string, weak bool) SRLevel {
	return SRLevel{Level: int(raw + 0.5), Raw: raw, Touches: touches, Strength: strength,
		Holds: holds, Breaks: breaks, LastTouch: srAt(last), Weakening: weak}
}

type srFixture struct {
	key      string
	last     float64
	bars     int
	sup, res []SRLevel // strength order, as supportResistance returns them
}

// srLiveFixtures — the live 2026-09-15 cards (sr_live.txt), levels in strength
// order exactly as the JSON served them.
var srLiveFixtures = []srFixture{
	{"btc", 78189, 249,
		[]SRLevel{lvl(76406.66375, 8, 11, 1, 2, "2026-09-14T00:00:00Z", false), lvl(64032.58857, 7, 9, 2, 0, "2026-08-19T04:00:00Z", true), lvl(65228.404, 5, 6.5, 2, 0, "2026-08-18T12:00:00Z", false)},
		[]SRLevel{lvl(79345.75, 4, 5, 6, 4, "2026-09-09T04:00:00Z", false), lvl(79950, 3, 4.5, 2, 3, "2026-09-11T12:00:00Z", false), lvl(78722.71667, 3, 4.5, 7, 6, "2026-09-14T12:00:00Z", false)}},
	{"eth", 2516.37, 249,
		[]SRLevel{lvl(1892.13857, 7, 8, 4, 2, "2026-08-18T00:00:00Z", true), lvl(1862.36667, 6, 8, 1, 0, "2026-08-16T20:00:00Z", false), lvl(2436.364, 5, 7.5, 5, 3, "2026-09-11T12:00:00Z", false)},
		[]SRLevel{lvl(2530.96143, 7, 10.5, 5, 1, "2026-09-14T04:00:00Z", true), lvl(2546.48333, 3, 4.5, 1, 1, "2026-09-14T16:00:00Z", false), lvl(2665.99, 1, 1.5, 0, 0, "2026-09-11T12:00:00Z", false)}},
	{"eurusd", 1.15421, 503, nil,
		[]SRLevel{lvl(1.16221, 64, 64, 8, 2, "2026-09-14T16:00:00Z", false), lvl(1.16939, 10, 10, 3, 3, "2026-08-26T08:00:00Z", false)}},
	{"gbpusd", 1.34871, 503, nil,
		[]SRLevel{lvl(1.35357, 75, 75, 9, 11, "2026-09-14T20:00:00Z", false), lvl(1.36414, 22, 22, 9, 3, "2026-08-26T12:00:00Z", false)}},
	{"usdjpy", 154.74, 500,
		[]SRLevel{lvl(153.71174, 22, 22, 2, 4, "2026-09-14T18:00:00Z", false)},
		[]SRLevel{lvl(159.18851, 59, 59, 6, 3, "2026-09-03T00:00:00Z", false), lvl(154.938, 8, 8, 1, 0, "2026-09-14T13:00:00Z", false), lvl(156.289, 5, 5, 1, 2, "2026-09-07T00:00:00Z", false)}},
	{"xauusd", 4344.2, 459,
		[]SRLevel{lvl(4340.81424, 7, 9, 3, 0, "2026-09-15T01:00:00Z", false), lvl(4293, 1, 1.5, 0, 0, "2026-09-14T13:00:00Z", false)},
		[]SRLevel{lvl(4676.97147, 14, 19.5, 2, 4, "2026-08-28T14:00:00Z", false), lvl(4468.1928, 14, 18.5, 5, 8, "2026-09-10T06:00:00Z", false), lvl(4429.75379, 13, 16.5, 4, 3, "2026-09-11T12:00:00Z", false)}},
}

func (f srFixture) card() Card {
	return srCardFrom(assetTable[f.key], f.sup, f.res, f.last, f.bars, trendTestTime, srFullVolOf(f.sup, f.res))
}

// srFullVolOf marks every given level as having volume on all six compared
// pivots — what the live Binance cards had when they printed the volume line.
func srFullVolOf(sides ...[]SRLevel) map[float64]bool {
	m := make(map[float64]bool)
	for _, side := range sides {
		for _, l := range side {
			m[l.Raw] = true
		}
	}
	return m
}

// srTexts is every reader-facing string one S/R card produces.
func srTexts(c Card) []string {
	out := append([]string{c.Verdict, c.Short}, c.Facts...)
	if b := c.Blocks; b != nil {
		// limitations is in the list since 2026-09-21: it used to be the
		// method line, covered transitively through facts[]; now it is its
		// own sentence and the site's most prominent box, so it must sit
		// under the same budget, banned-word and number checks as the rest.
		out = append(out, b.WhatHappened, b.WhyLevel, b.Regime, b.Limitations)
		out = append(out, b.Scenarios...)
		if b.Invalidates != nil {
			out = append(out, *b.Invalidates)
		}
	}
	return out
}

func TestSRPrecisionPerInstrument(t *testing.T) {
	want := map[string]int{"btc": 0, "eth": 1, "eurusd": 4, "gbpusd": 4, "usdjpy": 2, "xauusd": 1}
	for key, spec := range assetTable {
		d, ok := want[key]
		if !ok {
			t.Errorf("asset %q has no S/R precision pinned — add it to srAssetDecimals and this table", key)
			continue
		}
		if got := srDecimals(spec, 1); got != d {
			t.Errorf("%s: %d decimals, want %d", key, got, d)
		}
	}
	// The live defect: USDJPY 154.938 printed "155", 153.71174 printed "154".
	jpy := srPx{srDecimals(assetTable["usdjpy"], 154.74)}
	for raw, s := range map[float64]string{154.938: "154.94", 153.71174: "153.71", 156.289: "156.29"} {
		if got := jpy.str(raw); got != s {
			t.Errorf("USDJPY %v printed %q, want %q", raw, got, s)
		}
	}
	// Unlisted assets fall back to five significant digits, never a bare
	// integer at 100-999 the way the old label did.
	unk := assetSpec{Display: "XYZ", Interval: "4h"}
	for ref, d := range map[float64]int{154.938: 2, 78189: 0, 2516: 1, 1.1542: 4, 0.0012: 7, 1e7: 0} {
		if got := srDecimals(unk, ref); got != d {
			t.Errorf("fallback decimals for %v = %d, want %d", ref, got, d)
		}
	}
}

// The printed distance must be reproducible from the printed numbers.
var srLevelLineRe = regexp.MustCompile(`^(Resistance|Support) ([0-9.]+) \(([^)]*)\) · `)
var srHeadRe = regexp.MustCompile(`^Price ([0-9.]+) — (?:([<0-9.]+%) (above|below)|at) nearest shown (support|resistance) ([0-9.]+) \(`)

func TestSRPrintedNumbersReproduceTheDistance(t *testing.T) {
	for _, f := range srLiveFixtures {
		c := f.card()
		hm := srHeadRe.FindStringSubmatch(c.Verdict)
		if hm == nil {
			t.Fatalf("%s: headline shape: %q", f.key, c.Verdict)
		}
		price, _ := strconv.ParseFloat(hm[1], 64)
		if hm[2] != "" {
			lv, _ := strconv.ParseFloat(hm[5], 64)
			if got := pctAway(price, lv); got != hm[2] {
				t.Errorf("%s: headline says %s, printed numbers give %s: %q", f.key, hm[2], got, c.Verdict)
			}
		}
		for _, fact := range c.Facts {
			m := srLevelLineRe.FindStringSubmatch(fact)
			if m == nil {
				continue
			}
			lv, _ := strconv.ParseFloat(m[2], 64)
			want := signedPct(price, lv)
			if price == lv {
				want = "at price"
			}
			if m[3] != want {
				t.Errorf("%s: line says %q, printed numbers give %q: %q", f.key, m[3], want, fact)
			}
		}
	}
}

func TestSRLiveCardsRead(t *testing.T) {
	// With "tests:" every live verdict drops the pivot count to fit its
	// "On <tf>: …." sentence; GBPUSD (111 runes with the class) drops the
	// class too. The pivot count and class stay on the level's fact line.
	want := map[string]string{
		"btc":    "Price 78189 — 0.7% below nearest shown resistance 78723 (candidate; tests: 7 reactions / 6 breaks)",
		"eth":    "Price 2516.4 — 0.6% below nearest shown resistance 2531.0 (established; tests: 5 reactions / 1 break)",
		"eurusd": "Price 1.1542 — 0.7% below nearest shown resistance 1.1622 (established; tests: 8 reactions / 2 breaks)",
		"gbpusd": "Price 1.3487 — 0.4% below nearest shown resistance 1.3536 (tests: 9 reactions / 11 breaks)",
		"usdjpy": "Price 154.74 — 0.1% below nearest shown resistance 154.94 (established; tests: 1 reaction / 0 breaks)",
		"xauusd": "Price 4344.2 — 0.1% above nearest shown support 4340.8 (established; tests: 3 reactions / 0 breaks)",
	}
	for _, f := range srLiveFixtures {
		c := f.card()
		if c.Verdict != want[f.key] {
			t.Errorf("%s verdict:\n got %q\nwant %q", f.key, c.Verdict, want[f.key])
		}
		for _, s := range srTexts(c) {
			if n := utf8.RuneCountInString(s); n > srFactMaxRunes {
				t.Errorf("%s: %d chars (max %d): %q", f.key, n, srFactMaxRunes, s)
			}
			low := strings.ToLower(s)
			for _, banned := range []string{"held", "weakening", "key levels", "probab", "will ", "target", " buy", " sell", "odds"} {
				if strings.Contains(low, banned) {
					t.Errorf("%s: banned wording %q in %q", f.key, banned, s)
				}
			}
			if regexp.MustCompile(`\b[RS][1-3]\b`).MatchString(s) {
				t.Errorf("%s: R1/S1-style label in %q", f.key, s)
			}
		}
	}
}

func TestSRLevelLinesAndClasses(t *testing.T) {
	c := srLiveFixtures[1].card() // ETH
	joined := strings.Join(c.Facts, "\n")
	for _, want := range []string{
		"Resistance 2531.0 (+0.6%) · established, 7 pivots · 5 reactions / 1 break · last touch Sep 14",
		"Resistance 2546.5 (+1.2%) · candidate, 3 pivots · 1 reaction / 1 break · last touch Sep 14",
		"Resistance 2666.0 (+5.9%) · single swing, 1 pivot · no resolved tests · last touch Sep 11",
		"Support 2436.4 (-3.2%) · candidate, 5 pivots · 5 reactions / 3 breaks · last touch Sep 11",
		"Last 3 pivots on lower volume than first 3: 2531.0, 1892.1",
		"Window: 249 closed 4h candles · test = a close within 0.25 ATR of a level, resolved within 3 candles",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing line %q in\n%s", want, joined)
		}
	}
	// Nearest-first on each side: resistances 2531 < 2546 < 2666, supports
	// 2436 > 1892 > 1862 (the JSON keeps strength order).
	var order []string
	for _, f := range c.Facts {
		if m := srLevelLineRe.FindStringSubmatch(f); m != nil {
			order = append(order, m[2])
		}
	}
	if got := strings.Join(order, " "); got != "2531.0 2546.5 2666.0 2436.4 1892.1 1862.4" {
		t.Errorf("text order = %s", got)
	}
	for touches, class := range map[int]string{1: "single swing", 2: "candidate", 6: "candidate", 7: "established", 75: "established"} {
		if got := srClassOf(touches); got != class {
			t.Errorf("class(%d) = %q, want %q", touches, got, class)
		}
	}
}

func TestSREmptySideSaidPlainly(t *testing.T) {
	c := srLiveFixtures[2].card() // EURUSD at the window low
	joined := strings.Join(c.Facts, "\n")
	if !strings.Contains(joined, "No clustered support below price in this 503-candle 1h window") {
		t.Errorf("empty support side not stated: %s", joined)
	}
	if c.Short != "nearest shown R 1.1622 · none below" {
		t.Errorf("short = %q", c.Short)
	}
	if !strings.HasPrefix(c.Blocks.Regime, "Resistance only, none below price") {
		t.Errorf("regime = %q", c.Blocks.Regime)
	}
}

// When price and the nearest level print as the same number, the card says
// "at", never a distance next to two equal numbers.
func TestSRAtLevelWhenPrintedEqual(t *testing.T) {
	f := srLiveFixtures[4] // USDJPY; 154.938 prints 154.94
	f.last = 154.936       // prints 154.94 too, still below the raw mean
	c := f.card()
	if !strings.HasPrefix(c.Verdict, "Price 154.94 — at nearest shown resistance 154.94 (established; tests: 1 reaction / 0 breaks)") {
		t.Errorf("verdict = %q", c.Verdict)
	}
	if !strings.Contains(strings.Join(c.Facts, "\n"), "Resistance 154.94 (at price)") {
		t.Errorf("level line must say at price: %v", c.Facts)
	}
}

// display_rank follows the text, strength_rank the JSON array — both 1-based
// permutations, and they agree with what the card prints.
func TestSRRanksMatchTextAndJSON(t *testing.T) {
	for _, f := range srLiveFixtures {
		c := f.card()
		lv := c.Levels.(SRLevels)
		var textRes, textSup []string
		for _, fact := range c.Facts {
			if m := srLevelLineRe.FindStringSubmatch(fact); m != nil {
				if m[1] == "Resistance" {
					textRes = append(textRes, m[2])
				} else {
					textSup = append(textSup, m[2])
				}
			}
		}
		for side, pair := range map[string]struct {
			pts  []SRPoint
			text []string
			src  []SRLevel
		}{"res": {lv.Resistances, textRes, f.res}, "sup": {lv.Supports, textSup, f.sup}} {
			if len(pair.pts) != len(pair.src) {
				t.Fatalf("%s %s: %d points for %d levels", f.key, side, len(pair.pts), len(pair.src))
			}
			byDisplay := append([]SRPoint(nil), pair.pts...)
			sort.Slice(byDisplay, func(i, j int) bool { return byDisplay[i].DisplayRank < byDisplay[j].DisplayRank })
			for i, p := range pair.pts {
				if p.StrengthRank != i+1 {
					t.Errorf("%s %s[%d]: strength_rank %d", f.key, side, i, p.StrengthRank)
				}
				if p.Level != pair.src[i].Raw {
					t.Errorf("%s %s[%d]: JSON order is not strength order", f.key, side, i)
				}
				if p.Class != srClassKey(srClassOf(p.Touches)) {
					t.Errorf("%s %s[%d]: class %q", f.key, side, i, p.Class)
				}
				if byDisplay[i].DisplayRank != i+1 {
					t.Errorf("%s %s: display ranks are not 1..n: %+v", f.key, side, pair.pts)
				}
				if byDisplay[i].Label != pair.text[i] {
					t.Errorf("%s %s: display_rank %d is %s, card line %d is %s", f.key, side, i+1, byDisplay[i].Label, i+1, pair.text[i])
				}
			}
		}
	}
}

func TestSRBlocksContract(t *testing.T) {
	for _, f := range srLiveFixtures {
		c := f.card()
		b := c.Blocks
		if b == nil {
			t.Fatalf("%s: no blocks", f.key)
		}
		if len(b.Scenarios) != 2 {
			t.Fatalf("%s: %d scenarios, want exactly 2", f.key, len(b.Scenarios))
		}
		for _, s := range b.Scenarios {
			if !strings.HasPrefix(s, "If a ") || !strings.Contains(s, ", the level ") {
				t.Errorf("%s: scenario shape %q", f.key, s)
			}
			if strings.Contains(s, "reading adds") || strings.Contains(s, "reaction") {
				t.Errorf("%s: scenario describes the agent's counter, not a market event: %q", f.key, s)
			}
		}
		// The side comes from the fixture, independently of the card: the shown
		// level nearest the last close, resistance above price, support below.
		noun, toward, away := "support", "below", "above"
		best := -1.0
		for _, l := range f.sup {
			if d := f.last - l.Raw; best < 0 || d < best {
				best = d
			}
		}
		for _, l := range f.res {
			if d := l.Raw - f.last; best < 0 || d < best {
				best, noun, toward, away = d, "resistance", "above", "below"
			}
		}
		if want := "exits its band " + away + ", the level holds as " + noun; !strings.HasSuffix(b.Scenarios[0], want) {
			t.Errorf("%s: hold scenario must end %q (level is %s now): %q", f.key, want, noun, b.Scenarios[0])
		}
		if want := "candle closes " + toward + " "; !strings.Contains(b.Scenarios[1], want) ||
			!strings.HasSuffix(b.Scenarios[1], "'s band, the level is broken and moves "+away+" price") {
			t.Errorf("%s: break scenario for a %s must close %s it and move it %s price: %q", f.key, noun, toward, away, b.Scenarios[1])
		}
		if b.Invalidates == nil || !strings.HasPrefix(*b.Invalidates, "A closed "+assetTable[f.key].Interval+" candle ") {
			t.Errorf("%s: invalidates %v", f.key, b.Invalidates)
		}
		if b.WhatHappened == "" || b.WhyLevel == "" || b.Regime == "" {
			t.Errorf("%s: empty block field: %+v", f.key, b)
		}
		raw, err := json.Marshal(cardEnvelope(c))
		if err != nil || !strings.Contains(string(raw), `"blocks":{"what_happened"`) {
			t.Errorf("%s: envelope blocks missing (%v)", f.key, err)
		}
	}
	// ETH, spelled out.
	b := srLiveFixtures[1].card().Blocks
	for got, want := range map[string]string{
		b.WhatHappened: "On 4h: price 2516.4 — 0.6% below nearest shown resistance 2531.0 (established; tests: 5 reactions / 1 break).",
		b.WhyLevel:     "2531.0 = mean of 7 pivots · 5 reactions / 1 break in 6 resolved tests · last touch Sep 14",
		b.Scenarios[0]: "If a 4h close tests 2531.0 and a close within 3 candles exits its band below, the level holds as resistance",
		b.Scenarios[1]: "If a 4h candle closes above 2531.0's band, the level is broken and moves below price",
		*b.Invalidates: "A closed 4h candle above 2531.0 puts it below price: it no longer reads as resistance",
		b.Regime:       "Nearest shown on each side: resistance +0.6% · support -3.2% · 4h",
	} {
		if got != want {
			t.Errorf("\n got %q\nwant %q", got, want)
		}
	}
}
