package demobot

// readable_texts_test.go — three cards whose WORDS led a reader to the wrong
// conclusion on correct data (the 2026-09-16 reader review). Text only: the
// rules, thresholds, windows, order and selection are pinned elsewhere.
//   - S/R: a class word ("established" = many pivots) never stands without
//     the level's test history, and one fact gives the nearest shown distance
//     on each side;
//   - Macro: no lamp is hidden behind "· N more", and the factor lines and
//     the gold line say which score each one is about;
//   - News: below the threshold the text names the theme with the highest
//     activity score, never "leads", and never claims that no theme reached
//     the threshold (only that theme is checked).

import (
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"
)

// ── S/R ──────────────────────────────────────────────────────────────────────

// srFactWith is the fact line starting with prefix ("" when absent).
func srFactWith(c Card, prefix string) string {
	for _, f := range c.Facts {
		if strings.HasPrefix(f, prefix) {
			return f
		}
	}
	return ""
}

// The live BTC case of 2026-09-16: an established resistance (8 pivots) with
// 1 reaction and 4 breaks, and the nearest shown support 13.8% below.
func TestSRVerdictCarriesTestHistory(t *testing.T) {
	res := []SRLevel{lvl(76407, 8, 11, 1, 4, "2026-09-15T00:00:00Z", false)}
	sup := []SRLevel{lvl(65311, 7, 9, 2, 0, "2026-08-19T04:00:00Z", false)}
	c := srCardFrom(btcSpec, sup, res, 75767, 249, trendTestTime, nil)

	// The pivot count form is 109 runes, but its "On 4h: …." sentence is 117:
	// both drop the pivot count together.
	if want := "Price 75767 — 0.8% below nearest shown resistance 76407 (established; tests: 1 reaction / 4 breaks)"; c.Verdict != want {
		t.Errorf("verdict:\n got %q\nwant %q", c.Verdict, want)
	}
	if want := "On 4h: price 75767 — 0.8% below nearest shown resistance 76407 (established; tests: 1 reaction / 4 breaks)."; c.Blocks.WhatHappened != want {
		t.Errorf("what_happened:\n got %q\nwant %q", c.Blocks.WhatHappened, want)
	}
	// After the level lines, right before the window line.
	n := len(c.Facts)
	if want := "Nearest shown levels: resistance +0.8% · support -13.8%"; n < 2 || c.Facts[n-2] != want || !strings.HasPrefix(c.Facts[n-1], "Window: ") {
		t.Errorf("nearest-sides line must sit right before the window line:\n%v", c.Facts)
	}
	if !strings.Contains(strings.Join(c.Facts, "\n"), "Resistance 76407 (+0.8%) · established, 8 pivots · 1 reaction / 4 breaks") {
		t.Errorf("level line: %v", c.Facts)
	}
	if p := c.Levels.(SRLevels).Resistances[0]; p.Class != "established" || p.Holds != 1 || p.Breaks != 4 {
		t.Errorf("levels json: %+v", p)
	}
	for _, s := range []string{c.Verdict, c.Blocks.WhatHappened, srFactWith(c, srNearestSidesPrefix)} {
		low := strings.ToLower(s)
		for _, banned := range []string{"held", "strong", "weak", "empty", "thin", "far", "close to", "no support", "the nearest"} {
			if strings.Contains(low, banned) {
				t.Errorf("word %q in %q", banned, s)
			}
		}
	}
}

// Verdict and what_happened always carry the same form (the same words after
// "On <tf>: " and the case of the first letter).
func TestSRVerdictAndWhatHappenedAlike(t *testing.T) {
	for _, f := range srLiveFixtures {
		c := f.card()
		want := "On " + candleWord(assetTable[f.key].Interval) + ": " + lowerFirst(c.Verdict) + "."
		if c.Blocks.WhatHappened != want {
			t.Errorf("%s:\n verdict %q\n what    %q", f.key, c.Verdict, c.Blocks.WhatHappened)
		}
	}
}

func TestSRNearestSidesOneSideMissing(t *testing.T) {
	lv := []SRLevel{lvl(1.16221, 64, 64, 8, 2, "2026-09-14T16:00:00Z", false)}
	eur := assetTable["eurusd"]
	c := srCardFrom(eur, nil, lv, 1.15421, 503, trendTestTime, nil)
	if want := "Nearest shown levels: resistance +0.7% · no support shown below price"; srFactWith(c, srNearestSidesPrefix) != want {
		t.Errorf("resistance only: %v", c.Facts)
	}
	sup := []SRLevel{lvl(1.14, 5, 5, 0, 3, "2026-09-14T16:00:00Z", false)}
	c = srCardFrom(eur, sup, nil, 1.15421, 503, trendTestTime, nil)
	if want := "Nearest shown levels: no resistance shown above price · support -1.2%"; srFactWith(c, srNearestSidesPrefix) != want {
		t.Errorf("support only: %v", c.Facts)
	}
	if want := "Price 1.1542 — 1.2% above nearest shown support 1.1400 (candidate; tests: 0 reactions / 3 breaks)"; c.Verdict != want {
		t.Errorf("verdict:\n got %q\nwant %q", c.Verdict, want)
	}
	usd := assetTable["usdjpy"]
	c = srCardFrom(usd, nil, []SRLevel{lvl(154.938, 8, 8, 1, 0, "2026-09-14T13:00:00Z", false)}, 154.936, 500, trendTestTime, nil)
	if want := "Nearest shown levels: resistance at price · no support shown below price"; srFactWith(c, srNearestSidesPrefix) != want {
		t.Errorf("at price: %v", c.Facts)
	}
}

func TestSRVerdictNoResolvedTests(t *testing.T) {
	res := []SRLevel{lvl(2600, 9, 9, 0, 0, "2026-09-14T00:00:00Z", false)}
	c := srCardFrom(assetTable["eth"], nil, res, 2500, 249, trendTestTime, nil)
	if want := "Price 2500.0 — 4.0% below nearest shown resistance 2600.0 (established, 9 pivots; no resolved tests)"; c.Verdict != want {
		t.Errorf("verdict:\n got %q\nwant %q", c.Verdict, want)
	}
	for _, s := range []string{c.Verdict, c.Blocks.WhatHappened} {
		if strings.Contains(s, "0 reactions") || strings.Contains(s, "0 breaks") || strings.Contains(s, "tests: no") {
			t.Errorf("zeros or a doubled label instead of words: %q", s)
		}
		if !strings.Contains(s, "no resolved tests") {
			t.Errorf("missing test history: %q", s)
		}
	}
}

// At every width the boundary tests use: every line fits, and the class word
// never appears in the verdict or what_happened without the test history.
func TestSRClassNeverWithoutHistory(t *testing.T) {
	for key, spec := range assetTable {
		dec := srAssetDecimals[strings.ToUpper(key)]
		step := 1.0
		for i := 0; i < dec; i++ {
			step /= 10
		}
		for _, tc := range []struct {
			name     string
			sup, res []SRLevel
			last     float64
		}{
			{"worst both", []SRLevel{srWorst(100)}, []SRLevel{srWorst(9000)}, 5000},
			{"worst far", nil, []SRLevel{srWorst(9999999 * step)}, 1000000 * step},
			{"worst no tests", []SRLevel{{Raw: 100 * step, Touches: 9999}}, nil, 9999999 * step},
			{"worst at price", []SRLevel{srWorst(5000)}, nil, 5000 + step/4},
		} {
			c := srCardFrom(spec, tc.sup, tc.res, tc.last, 9999, trendTestTime, nil)
			for _, s := range append([]string{c.Verdict, c.Blocks.WhatHappened}, c.Facts...) {
				if n := utf8.RuneCountInString(s); n > srFactMaxRunes {
					t.Errorf("%s/%s: %d runes: %q", key, tc.name, n, s)
				}
			}
			for _, s := range []string{c.Verdict, c.Blocks.WhatHappened} {
				class := strings.Contains(s, "established") || strings.Contains(s, "candidate") || strings.Contains(s, "single swing")
				history := strings.Contains(s, "tests: ") || strings.Contains(s, "no resolved tests")
				if class && !history {
					t.Errorf("%s/%s: class without history: %q", key, tc.name, s)
				}
			}
		}
	}
}

// srShowcaseBase is strongestFact / exampleFacts for these cards as served
// on 5572bf0, before the nearest-sides line existed: the landing must quote
// them byte for byte.
var srShowcaseBase = map[string]struct {
	explained string
	data      []string
}{
	"btc":      {"Resistance 78723 (+0.7%) · candidate, 3 pivots · 7 reactions / 6 breaks · last touch Sep 14.", []string{"Resistance 78723 (+0.7%) · candidate, 3 pivots · 7 reactions / 6 breaks · last touch Sep 14", "Resistance 79346 (+1.5%) · candidate, 4 pivots · 6 reactions / 4 breaks · last touch Sep 9", "Resistance 79950 (+2.3%) · candidate, 3 pivots · 2 reactions / 3 breaks · last touch Sep 11", "Support 76407 (-2.3%) · established, 8 pivots · 1 reaction / 2 breaks · last touch Sep 14"}},
	"eth":      {"Resistance 2531.0 (+0.6%) · established, 7 pivots · 5 reactions / 1 break · last touch Sep 14.", []string{"Resistance 2531.0 (+0.6%) · established, 7 pivots · 5 reactions / 1 break · last touch Sep 14", "Resistance 2546.5 (+1.2%) · candidate, 3 pivots · 1 reaction / 1 break · last touch Sep 14", "Resistance 2666.0 (+5.9%) · single swing, 1 pivot · no resolved tests · last touch Sep 11", "Support 2436.4 (-3.2%) · candidate, 5 pivots · 5 reactions / 3 breaks · last touch Sep 11"}},
	"eurusd":   {"Resistance 1.1622 (+0.7%) · established, 64 pivots · 8 reactions / 2 breaks · last touch Sep 14.", []string{"Resistance 1.1622 (+0.7%) · established, 64 pivots · 8 reactions / 2 breaks · last touch Sep 14", "Resistance 1.1694 (+1.3%) · established, 10 pivots · 3 reactions / 3 breaks · last touch Aug 26", "No clustered support below price in this 503-candle 1h window", "Window: 503 closed 1h candles · test = a close within 0.25 ATR of a level, resolved within 3 candles"}},
	"gbpusd":   {"Resistance 1.3536 (+0.4%) · established, 75 pivots · 9 reactions / 11 breaks · last touch Sep 14.", []string{"Resistance 1.3536 (+0.4%) · established, 75 pivots · 9 reactions / 11 breaks · last touch Sep 14", "Resistance 1.3641 (+1.1%) · established, 22 pivots · 9 reactions / 3 breaks · last touch Aug 26", "No clustered support below price in this 503-candle 1h window", "Window: 503 closed 1h candles · test = a close within 0.25 ATR of a level, resolved within 3 candles"}},
	"usdjpy":   {"Resistance 154.94 (+0.1%) · established, 8 pivots · 1 reaction / 0 breaks · last touch Sep 14.", []string{"Resistance 154.94 (+0.1%) · established, 8 pivots · 1 reaction / 0 breaks · last touch Sep 14", "Resistance 156.29 (+1.0%) · candidate, 5 pivots · 1 reaction / 2 breaks · last touch Sep 7", "Resistance 159.19 (+2.9%) · established, 59 pivots · 6 reactions / 3 breaks · last touch Sep 3", "Support 153.71 (-0.7%) · established, 22 pivots · 2 reactions / 4 breaks · last touch Sep 14"}},
	"xauusd":   {"Resistance 4429.8 (+2.0%) · established, 13 pivots · 4 reactions / 3 breaks · last touch Sep 11.", []string{"Resistance 4429.8 (+2.0%) · established, 13 pivots · 4 reactions / 3 breaks · last touch Sep 11", "Resistance 4468.2 (+2.9%) · established, 14 pivots · 5 reactions / 8 breaks · last touch Sep 10", "Resistance 4677.0 (+7.7%) · established, 14 pivots · 2 reactions / 4 breaks · last touch Aug 28", "Support 4340.8 (-0.1%) · established, 7 pivots · 3 reactions / 0 breaks · last touch Sep 15"}},
	"btc-case": {"Resistance 76407 (+0.8%) · established, 8 pivots · 1 reaction / 4 breaks · last touch Sep 15.", []string{"Resistance 76407 (+0.8%) · established, 8 pivots · 1 reaction / 4 breaks · last touch Sep 15", "Support 65311 (-13.8%) · established, 7 pivots · 2 reactions / 0 breaks · last touch Aug 19", "Window: 249 closed 4h candles · test = a close within 0.25 ATR of a level, resolved within 3 candles"}},
	"one-res":  {"Resistance 1.1622 (+0.7%) · established, 64 pivots · 8 reactions / 2 breaks · last touch Sep 14.", []string{"Resistance 1.1622 (+0.7%) · established, 64 pivots · 8 reactions / 2 breaks · last touch Sep 14", "No clustered support below price in this 503-candle 1h window", "Window: 503 closed 1h candles · test = a close within 0.25 ATR of a level, resolved within 3 candles"}},
}

func TestSRShowcaseUnchangedFromBase(t *testing.T) {
	cards := map[string]Card{}
	for _, f := range srLiveFixtures {
		cards[f.key] = f.card()
	}
	cards["btc-case"] = srCardFrom(btcSpec, []SRLevel{lvl(65311, 7, 9, 2, 0, "2026-08-19T04:00:00Z", false)}, []SRLevel{lvl(76407, 8, 11, 1, 4, "2026-09-15T00:00:00Z", false)}, 75767, 249, trendTestTime, nil)
	cards["one-res"] = srCardFrom(assetTable["eurusd"], nil, []SRLevel{lvl(1.16221, 64, 64, 8, 2, "2026-09-14T16:00:00Z", false)}, 1.15421, 503, trendTestTime, nil)
	for key, want := range srShowcaseBase {
		c, ok := cards[key]
		if !ok {
			t.Fatalf("no card %s", key)
		}
		if got := strongestFact(c); got != want.explained {
			t.Errorf("%s explained:\n got %q\nwant %q", key, got, want.explained)
		}
		if got := exampleFacts(c); strings.Join(got, "\n") != strings.Join(want.data, "\n") {
			t.Errorf("%s data:\n got %q\nwant %q", key, got, want.data)
		}
	}
}

// ── Macro ────────────────────────────────────────────────────────────────────

// macroAllSame builds a payload with every lamp on one side at the widest
// printable numbers, so the side cannot fit one line.
func macroAllSame(status string) string {
	delta := map[string]float64{"dxy": 999.99, "rates": 999.99, "spx": -999.99, "gold": 999.99}
	if status == "tailwind" {
		delta = map[string]float64{"dxy": -999.99, "rates": -999.99, "spx": 999.99, "gold": -999.99}
	}
	vix := 99.99
	if status == "tailwind" {
		vix = 10.01
	}
	composite := 0
	regime := "risk_off"
	if status == "tailwind" {
		composite, regime = 100, "risk_on"
	}
	var lamps []string
	for _, k := range []string{"dxy", "rates", "vix", "spx", "gold"} {
		v, d := 1234567.0, 0.0
		if k == "vix" {
			v = vix
		} else {
			d = delta[k]
		}
		lamps = append(lamps, fmt.Sprintf(`{"key":%q,"label":%q,"value":%v,"ok":true,"delta_pct":%v,"status":%q,"as_of":"2026-09-15T04:00:00Z","source":"yahoo"}`,
			k, k, v, d, status))
	}
	return fmt.Sprintf(`{"regime":%q,"composite":%d,"tradfin_market_open":true,"tradfin_ok":true,"tradfin_as_of":"2026-09-15T04:00:00Z","captured_at":"2026-09-15T04:44:32Z","lamps":[%s]}`,
		regime, composite, strings.Join(lamps, ","))
}

func TestMacroFactorLinesShowEveryLamp(t *testing.T) {
	for _, tc := range []struct{ status, word string }{{"headwind", "Negative"}, {"tailwind", "Positive"}} {
		c, _ := macroCardFrom(macroRespOf(t, macroAllSame(tc.status)))
		head := tc.word + " for rule score"
		var side []string
		for _, f := range c.Facts {
			if n := utf8.RuneCountInString(f); n > macroFactMaxRunes {
				t.Errorf("%s: %d runes: %q", tc.status, n, f)
			}
			if strings.HasPrefix(f, head+": ") || strings.HasPrefix(f, head+" (cont.): ") {
				side = append(side, f)
			}
			if strings.Contains(f, " more") {
				t.Errorf("%s: a lamp hidden behind a count: %q", tc.status, f)
			}
		}
		if len(side) < 2 {
			t.Errorf("%s: fixture must wrap to a second line of the same side: %v", tc.status, c.Facts)
		}
		joined := strings.Join(side, "\n")
		for _, name := range []string{"DXY", "US 10Y", "VIX", "S&P 500", "Gold"} {
			if strings.Count(joined, name+" ") != 1 {
				t.Errorf("%s: lamp %s must appear exactly once in %q", tc.status, name, joined)
			}
		}
	}
}

// macroSideFixture: n lamps on the negative side (the rest neutral), at the
// widest printable numbers.
func macroSideFixture(n int) string {
	keys := []string{"dxy", "vix", "spx", "rates", "gold"}
	var lamps []string
	for i, k := range keys {
		status, d, v := "neutral", 0.2, 1234567.0
		if k == "spx" {
			d = -0.2
		}
		if k == "vix" {
			v = 20
		}
		if i < n {
			status = "headwind"
			d = 999.99
			if k == "spx" {
				d = -999.99
			}
			if k == "vix" {
				v = 99.99
			}
		}
		lamps = append(lamps, fmt.Sprintf(`{"key":%q,"label":%q,"value":%v,"ok":true,"delta_pct":%v,"status":%q,"as_of":"2026-09-15T04:00:00Z","source":"yahoo"}`, k, k, v, d, status))
	}
	return fmt.Sprintf(`{"regime":"mixed","tradfin_market_open":true,"tradfin_ok":true,"tradfin_as_of":"2026-09-15T04:00:00Z","captured_at":"2026-09-15T04:44:32Z","lamps":[%s]}`, strings.Join(lamps, ","))
}

// The showcase folds a factor side into one line: the lamps it no longer shows
// are counted ("· +N more on the card"), never dropped silently, and "what
// holds the regime" stays in the capped block.
func TestMacroShowcaseCountsFoldedLamps(t *testing.T) {
	for n := 1; n <= 5; n++ {
		m := macroRespOf(t, macroSideFixture(n))
		m.Composite = riskModel.read(m.Lamps).score // contributions print only when they reproduce it
		c, _ := macroCardFrom(m)
		var onCard int
		for _, f := range c.Facts {
			if _, _, parts, ok := macroFactorHead(f); ok && strings.HasPrefix(f, "Negative") {
				onCard += len(parts)
			}
		}
		data := exampleFacts(c)
		var side string
		for _, f := range data {
			if strings.HasPrefix(f, "Negative for rule score") {
				side = f
			}
			if utf8.RuneCountInString(f) > macroFactMaxRunes {
				t.Errorf("%d: %d runes: %q", n, utf8.RuneCountInString(f), f)
			}
			if strings.Contains(f, macroContMark) {
				t.Errorf("%d: continuation line quoted: %q", n, f)
			}
		}
		_, _, shown, _ := macroFactorHead(strings.SplitN(side, " · +", 2)[0])
		more := 0
		if i := strings.Index(side, " · +"); i > 0 {
			fmt.Sscanf(side[i+len(" · +"):], "%d", &more)
			if !strings.HasSuffix(side, fmt.Sprintf(" · +%d more on the card", more)) {
				t.Errorf("%d: tail %q", n, side)
			}
		}
		if len(shown)+more != n || onCard != n {
			t.Errorf("%d lamps: card %d, showcase shows %d + %d more: %q", n, onCard, len(shown), more, side)
		}
		if !strings.Contains(strings.Join(data, "\n"), " while the rule score ") && !strings.Contains(strings.Join(data, "\n"), "No rule score") {
			t.Errorf("%d: hold line out of the block: %q", n, data)
		}
		t.Logf("%d lamps: %q", n, data)
	}
}

// The live 2026-09-15 payload: US 10Y (third positive) used to be "· 1 more".
// The gold line names its model as separate from the rule score the factor
// lines are about.
func TestMacroLiveNoHiddenLampAndTwoScoresNamed(t *testing.T) {
	c, _ := macroCardFrom(macroRespOf(t, macroLiveFixture))
	if c.Facts[0] != "Positive for rule score: VIX 17.10 (<18) → +12.5 · S&P 500 +0.11% (rose) → +12.5 · US 10Y -0.36% (fell) → +7.5" {
		t.Errorf("positive line: %q", c.Facts[0])
	}
	if d := exampleFacts(c); len(d) == 0 || d[0] != c.Facts[0] {
		t.Errorf("showcase must quote the live side whole: %q", d)
	}
	var gold string
	for _, f := range c.Facts {
		if strings.HasPrefix(f, "Gold macro backdrop:") {
			gold = f
		}
	}
	if !strings.Contains(gold, "separate experimental model") {
		t.Errorf("gold line must name its model as separate: %q", gold)
	}
}

// ── News ─────────────────────────────────────────────────────────────────────

// narrBelowTexts is every reader-facing text of a below-threshold card: the
// verdict, facts, all blocks and the showcase conclusion.
func narrBelowTexts(c Card) []string {
	out := append([]string{c.Verdict, c.Short}, c.Facts...)
	if b := c.Blocks; b != nil {
		out = append(out, b.WhatHappened, b.WhyLevel, b.Regime, b.Limitations, b.Source)
		out = append(out, b.Scenarios...)
		if b.Invalidates != nil {
			out = append(out, *b.Invalidates)
		}
	}
	return append(out, conclusionFor(c), c.HowItWorks)
}

// The threshold is checked on the top activity score theme only: a theme
// with 6 matched items below it in the order must not be denied by the text,
// and the checked theme is named one way everywhere.
func TestNarrativeBelowThresholdDoesNotDenyALowerTheme(t *testing.T) {
	body := `{"captured_at":"2026-09-15T22:00:00Z","narratives":[
	 {"narrative":"zk","mention_count_24h":2,"mention_count_prev_24h":0,"trend_score":60,"stage":"early","sentiment_label":"bull","confidence":40,"sources_breakdown":{"coindesk":2}},
	 {"narrative":"rwa","mention_count_24h":6,"mention_count_prev_24h":5,"trend_score":30,"stage":"early","sentiment_label":"neutral","confidence":40,"sources_breakdown":{"coindesk":6}}]}`
	for label, c := range map[string]Card{"zk/rwa": narrCard(t, body), "eligible22": narrCard(t, narrEligible22), "below22": narrCard(t, narrBelow22)} {
		if c.effectiveStatus() != statusBelowThreshold {
			t.Fatalf("%s: status %v: the rule checks the first theme only", label, c.effectiveStatus())
		}
		for _, s := range narrBelowTexts(c) {
			low := strings.ToLower(s)
			for _, bad := range []string{"leader", "leads", "leading", "quiet", "no theme", "none of the themes", "every theme", "all themes"} {
				if strings.Contains(low, bad) {
					t.Errorf("%s: %q in %q", label, bad, s)
				}
			}
			if n := utf8.RuneCountInString(s); n > narrativeFactMaxRunes && s != conclusionFor(c) && s != c.HowItWorks {
				t.Errorf("%s: %d runes: %q", label, n, s)
			}
		}
	}
	c := narrCard(t, body)
	if want := "Below threshold: Zero-knowledge (ZK) networks ranks first, 2 matched items/24h; 5 needed"; c.Verdict != want {
		t.Errorf("verdict:\n got %q\nwant %q", c.Verdict, want)
	}
	if want := "Zero-knowledge (ZK) networks ranks first, 2 matched items in the 24h to Sep 15 22:00 UTC"; c.Blocks.WhatHappened != want {
		t.Errorf("what_happened:\n got %q\nwant %q", c.Blocks.WhatHappened, want)
	}
	if want := "Checked on the top activity score theme only: Real-world assets (RWA) has 6 matched items, lower activity score"; !strings.Contains(strings.Join(c.Facts, "\n"), "Real-world assets (RWA) has 6 matched items") {
		t.Errorf("the lower theme with 6 items must stay named (%q): %v", want, c.Facts)
	}
	if want := "News radar below threshold: the top activity score theme has under 5 matched items in 24h; no price direction"; c.Blocks.Regime != want {
		t.Errorf("regime %q", c.Blocks.Regime)
	}
}

// One verdict form at every length: every name of the theme table stays whole
// (no "…", no dropped parenthesis) at 1, 4, 99 and 999 matched items, with
// the count and the threshold; an unknown long name is cut, never the rest.
func TestNarrativeBelowThresholdOneForm(t *testing.T) {
	const window = "in the 24h to Sep 15 22:00 UTC"
	for id, name := range narrativeThemeNames {
		for _, n := range []int{1, 4, 99, 999} {
			items := matchedItems(n)
			v := narrativeBelowVerdict(name, n)
			if want := "Below threshold: " + name + " ranks first, " + items + "/24h; 5 needed"; v != want {
				t.Errorf("%s/%d verdict:\n got %q\nwant %q", id, n, v, want)
			}
			w := narrativeBelowWhat(name, n, window)
			if want := name + " ranks first, " + items + " " + window; w != want {
				t.Errorf("%s/%d what_happened:\n got %q\nwant %q", id, n, w, want)
			}
			for _, s := range []string{v, w} {
				if k := utf8.RuneCountInString(s); k > narrativeFactMaxRunes {
					t.Errorf("%s/%d: %d runes: %q", id, n, k, s)
				}
			}
		}
	}
	long := strings.Repeat("x", 200)
	for _, n := range []int{1, 4, 999} {
		v, w := narrativeBelowVerdict(long, n), narrativeBelowWhat(long, n, window)
		if !strings.HasSuffix(v, "… ranks first, "+matchedItems(n)+"/24h; 5 needed") || utf8.RuneCountInString(v) > narrativeFactMaxRunes {
			t.Errorf("long/%d verdict %q", n, v)
		}
		if !strings.HasSuffix(w, "… ranks first, "+matchedItems(n)+" "+window) || utf8.RuneCountInString(w) > narrativeFactMaxRunes {
			t.Errorf("long/%d what_happened %q", n, w)
		}
	}
	// A parenthesis goes whole before any cut.
	if got := narrativeWithName("Restaking and liquid staking (LST/LRT)", func(n string) string { return n + strings.Repeat("-", 75) }); got != "Restaking and liquid staking"+strings.Repeat("-", 75) {
		t.Errorf("parenthesis first: %q", got)
	}
	// Through the card, at the live counts.
	for n := 1; n < newsMinMentions; n++ {
		c := narrCard(t, fmt.Sprintf(`{"captured_at":"2026-09-15T22:00:00Z","narratives":[{"narrative":"depin","mention_count_24h":%d,"trend_score":50}]}`, n))
		if want := fmt.Sprintf("Below threshold: Decentralized physical infrastructure (DePIN) ranks first, %s/24h; 5 needed", matchedItems(n)); c.Verdict != want {
			t.Errorf("DePIN/%d verdict %q", n, c.Verdict)
		}
	}
}

// No matched items anywhere: no line singles out a theme.
func TestNarrativeNoMatchedNamesNoTheme(t *testing.T) {
	c := narrCard(t, narrNone22)
	b := c.Blocks
	if b.WhyLevel != "No price level: 5 matched items in 24h is the radar's threshold; no theme has a matched item" {
		t.Errorf("why %q", b.WhyLevel)
	}
	if b.Regime != "News radar below threshold: no theme has a matched item in 24h; no price direction" {
		t.Errorf("regime %q", b.Regime)
	}
	eqLines(t, "scenarios", b.Scenarios, []string{
		"If a theme with the top activity score reaches 5 matched items in 24h, the radar scores it",
		"Until a theme with the top activity score has 5 matched items in 24h, the radar stays below threshold",
	})
	texts := narrBelowTexts(c)
	for _, s := range texts[:len(texts)-1] { // the how-it-works text describes the method in every state
		low := strings.ToLower(s)
		for _, bad := range []string{"the top activity score theme", "first-ranked theme", "leader", "leads", "quiet"} {
			if strings.Contains(low, bad) {
				t.Errorf("%q in %q", bad, s)
			}
		}
	}
}
