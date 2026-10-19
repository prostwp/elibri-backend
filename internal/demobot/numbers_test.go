package demobot

// numbers_test.go — the numeric fields added 2026-10-19 say exactly what the
// card's text prints: each one, formatted the way the text formats it, is
// found in the text it belongs to. The fields are read from the SAME
// variables the text is built from, so these tests pin that no later edit
// lets the two drift apart.

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"
)

// numEnvelope is the part of an envelope these tests read.
type numEnvelope struct {
	Agent    string            `json:"agent"`
	OK       bool              `json:"ok"`
	Verdict  string            `json:"verdict"`
	Facts    []string          `json:"facts"`
	Levels   json.RawMessage   `json:"levels"`
	Results  []AssetResult     `json:"results"`
	Blocks   *ContentBlocks    `json:"blocks"`
	Trend    *TrendReadout     `json:"trend"`
	Momentum *MomentumReadout  `json:"momentum"`
	Funding  *FundingReadout   `json:"funding"`
	Gold     *GoldReadout      `json:"gold"`
	Macro    *MacroReadout     `json:"macro"`
	Whale    *WhaleReadout     `json:"whale"`
	FX       *FXReadout        `json:"fx"`
	Digest   *numDigestReadout `json:"digest"`
}

type numDigestReadout struct {
	Sections []DigestSection `json:"sections"`
}

func decodeNum(t *testing.T, b []byte) numEnvelope {
	t.Helper()
	var e numEnvelope
	if err := json.Unmarshal(b, &e); err != nil {
		t.Fatalf("decode: %v\n%s", err, b)
	}
	return e
}

func (e numEnvelope) text() string { return strings.Join(e.Facts, "\n") + "\n" + e.Verdict }

// pct1 prints a magnitude the way pctAway does.
func pct1(v float64) string {
	a := math.Abs(v)
	if a < 0.05 {
		return "<0.1%"
	}
	return fmt.Sprintf("%.1f%%", a)
}

// contractAll is every body the harness produces, served and built.
func contractAll(t *testing.T) []contractBody {
	t.Helper()
	return append(contractServe(t), contractBuilderCards(t)...)
}

func TestNumbersTrendMatchText(t *testing.T) {
	seen := map[string]bool{}
	for _, b := range contractAll(t) {
		e := decodeNum(t, b.body)
		if e.Agent != "Trend Agent" || !e.OK || strings.Contains(b.path, "/chart") {
			continue // the chart address has its own shape (trendchart.go)
		}
		tr := e.Trend
		if tr == nil {
			t.Errorf("%s: an ok trend card must carry trend", b.path)
			continue
		}
		seen[tr.State] = true
		txt := e.text()
		if !strings.Contains(txt, "ADX "+fmt.Sprintf("%.1f", tr.ADXShown)) || fmt.Sprintf("%.1f", tr.ADXShown) != adxShown(tr.ADX) {
			t.Errorf("%s: adx_shown %v not the printed ADX", b.path, tr.ADXShown)
		}
		if tr.ADXThreshold != 25 || tr.ADXFlatThreshold != 20 {
			t.Errorf("%s: thresholds %d/%d", b.path, tr.ADXThreshold, tr.ADXFlatThreshold)
		}
		if !strings.Contains(txt, "Price "+trimFloat(tr.Price)) {
			t.Errorf("%s: price %v not printed as %s", b.path, tr.Price, trimFloat(tr.Price))
		}
		confirmed := tr.State == trendUp || tr.State == trendDown
		if !confirmed {
			for name, d := range map[string]float64{"EMA50": tr.EMA50DistancePct, "EMA200": tr.EMA200DistancePct} {
				side := "above"
				if d < 0 {
					side = "below"
				}
				if d != 0 && !strings.Contains(txt, pct1(d)+" "+side+" "+name) {
					t.Errorf("%s: %s distance %v not printed as %q", b.path, name, d, pct1(d)+" "+side+" "+name)
				}
			}
			if len(e.Levels) != 0 {
				t.Errorf("%s: unconfirmed trend carries levels", b.path)
			}
			continue
		}
		var lv TrendLevels
		if err := json.Unmarshal(e.Levels, &lv); err != nil {
			t.Fatal(err)
		}
		switch d := lv.DistanceToPullbackPct; {
		case d == nil:
			t.Errorf("%s: confirmed card without distance_to_pullback_pct", b.path)
		case lv.PullbackPosition == "inside":
			if *d != 0 || !strings.Contains(txt, "inside the pullback zone") {
				t.Errorf("%s: inside but distance %v", b.path, *d)
			}
		default:
			if (*d > 0) != (lv.PullbackPosition == "above") ||
				!strings.Contains(txt, pct1(*d)+" "+lv.PullbackPosition+" the pullback zone") {
				t.Errorf("%s: pullback %s %v not as printed", b.path, lv.PullbackPosition, *d)
			}
		}
		if lv.Invalidation != nil {
			want := signedPct(tr.Price, *lv.Invalidation)
			got := pct1(*lv.InvalidationDistancePct)
			if !strings.HasPrefix(got, "<") {
				got = map[bool]string{true: "+", false: "-"}[*lv.InvalidationDistancePct > 0] + got
			}
			if got != want || !strings.Contains(txt, "("+want+", 1 ATR") || *lv.InvalidationATRMultiple != 1 {
				t.Errorf("%s: invalidation distance %v prints %q, card %q", b.path, *lv.InvalidationDistancePct, got, want)
			}
		}
	}
	for _, st := range []string{trendUp, trendDown, trendFlat, trendGrey, trendConflict} {
		if !seen[st] {
			t.Errorf("no %s trend card in the harness", st)
		}
	}
}

func TestNumbersSRMatchText(t *testing.T) {
	n := 0
	for _, b := range contractAll(t) {
		e := decodeNum(t, b.body)
		if e.Agent != "S/R Agent" || !e.OK {
			continue
		}
		var lv SRLevels
		if err := json.Unmarshal(e.Levels, &lv); err != nil {
			t.Fatal(err)
		}
		txt := e.text()
		if !strings.Contains(txt, lv.PriceLabel) || lv.TestBandATR != 0.25 || lv.TestResolveBars != 3 ||
			!strings.Contains(txt, fmt.Sprintf("Window: %d closed %s candles", lv.WindowBars, lv.Timeframe)) {
			t.Errorf("%s: head numbers not as printed: %+v", b.path, lv)
		}
		check := func(title string, pts []SRPoint, nearest *float64) {
			if len(pts) == 0 {
				if nearest != nil {
					t.Errorf("%s: %s nearest without a level", b.path, title)
				}
				return
			}
			best := pts[0]
			for _, p := range pts {
				n++
				d := "at price"
				if p.DistancePct != 0 {
					d = pct1(p.DistancePct)
					if !strings.HasPrefix(d, "<") {
						d = map[bool]string{true: "+", false: "-"}[p.DistancePct > 0] + d
					}
				}
				if line := title + " " + p.Label + " (" + d + ")"; !strings.Contains(txt, line) {
					t.Errorf("%s: %q not on the card", b.path, line)
				}
				if p.ResolvedTests != p.Holds+p.Breaks {
					t.Errorf("%s: resolved_tests %d", b.path, p.ResolvedTests)
				}
				if p.DisplayRank == 1 {
					best = p
				}
			}
			if nearest == nil || *nearest != best.DistancePct {
				t.Errorf("%s: %s nearest %v, want the display-rank-1 distance %v", b.path, title, nearest, best.DistancePct)
			}
		}
		check("Resistance", lv.Resistances, lv.NearestResistancePct)
		check("Support", lv.Supports, lv.NearestSupportPct)
	}
	if n == 0 {
		t.Fatal("no S/R level in the harness")
	}
}

func TestNumbersVolMatchText(t *testing.T) {
	for _, r := range []float64{0.5, 0.8, 0.80001, 0.8994, 0.97, 1, 1.0049, 1.105, 1.2499, 1.25, 3.7, 150} {
		c := volCardFrom(btcSpec, volRead{atr: r, baseline: 1, price: 100, interval: "4h"}, time.Time{})
		lv := c.Levels.(VolLevels)
		if fmt.Sprintf("%.3f", lv.RatioShown) != volRatioShown(r) && r < 99 {
			t.Errorf("ratio %v: ratio_shown %v, card %s", r, lv.RatioShown, volRatioShown(r))
		}
		want := "within 1%"
		if d := lv.DistancePct; d > 999 {
			want = ">999% above" // the text's cap; the number is not capped
		} else if d != 0 {
			want = fmt.Sprintf("%d%% %s", int(math.Abs(float64(d))), map[bool]string{true: "above", false: "below"}[d > 0])
		}
		if !strings.Contains(c.Verdict, want) || lv.BaselineBars != 30 {
			t.Errorf("ratio %v: distance_pct %d, verdict %q", r, lv.DistancePct, c.Verdict)
		}
	}
}

func TestNumbersFundingMatchText(t *testing.T) {
	cases := map[string]map[string]fundingQuote{
		"within": {"BTCUSDT": {rate: 0.0001}, "ETHUSDT": {rate: 0.00007219}, "SOLUSDT": {rate: 0.00009364},
			"BNBUSDT": {rate: 0.00008777}, "XRPUSDT": {rate: 0.00000055}},
		"long":  {"BTCUSDT": {rate: 0.00045}, "ETHUSDT": {rate: 0.0001}, "SOLUSDT": {rate: -0.00002}},
		"exact": {"BTCUSDT": {rate: fundingLongsCrowded}, "ETHUSDT": {rate: 0.0001}, "SOLUSDT": {rate: 0.0001}},
		"tiny":  {"BTCUSDT": {rate: -0.0001000049}, "ETHUSDT": {rate: 0.0001}, "SOLUSDT": {rate: 0.0001}},
		"short": {"BTCUSDT": {rate: -0.00031}, "ETHUSDT": {rate: 0.0001}, "SOLUSDT": {rate: 0.0001}},
		"part":  {"BTCUSDT": {rate: 0.0001}},
	}
	now := time.Date(2026, 10, 19, 11, 0, 0, 0, time.UTC)
	for name, q := range cases {
		c := fundingCardFrom(q, nil, nil, nil, now)
		ro, txt := c.Funding, strings.Join(c.Facts, "\n")
		if len(ro.Rates) != len(q) {
			t.Errorf("%s: %d rates, %d received", name, len(ro.Rates), len(q))
		}
		for _, r := range ro.Rates {
			if s := fundingPct(int64(math.Round(r.RatePct * 10000))); !strings.Contains(txt, r.Symbol+" "+s) &&
				!strings.Contains(txt, "rate: "+s) && !strings.Contains(txt, "< "+s+" <") {
				t.Errorf("%s: %s rate_pct %v (%s) not on the card", name, r.Symbol, r.RatePct, s)
			}
		}
		pp := func(v float64) string { return fundingPP(int64(math.Round(v * 10000))) }
		switch {
		case name == "part":
			if ro.NearestThreshold != nil || ro.PastThresholdByPP != nil {
				t.Errorf("part: distance fields without a reading")
			}
		case ro.NearestThreshold != nil:
			if !strings.Contains(txt, pp(*ro.NearestThresholdDistancePP)) || ro.PastThresholdByPP != nil {
				t.Errorf("%s: nearest %s %v not printed", name, *ro.NearestThreshold, *ro.NearestThresholdDistancePP)
			}
		case name == "exact":
			if *ro.PastThresholdByPP != 0 || !strings.Contains(txt, "Exactly at") {
				t.Errorf("exact: past_threshold_by_pp %v", *ro.PastThresholdByPP)
			}
		case name == "tiny":
			if v := *ro.PastThresholdByPP; v <= 0 || v >= 0.0001 || !strings.Contains(txt, "<0.0001 pp") {
				t.Errorf("tiny: past_threshold_by_pp %v", v)
			}
		default:
			if !strings.Contains(txt, "by "+pp(*ro.PastThresholdByPP)) {
				t.Errorf("%s: past_threshold_by_pp %v not printed", name, *ro.PastThresholdByPP)
			}
		}
		for _, r := range c.Results {
			if r.OK && r.Selected != nil && *r.Selected {
				if s := fmt.Sprintf("%.2f×", *r.RatioToThresholdShown); !strings.Contains(txt, s) {
					t.Errorf("%s: ratio shown %v not printed", name, *r.RatioToThresholdShown)
				}
			}
		}
	}
}

func TestNumbersMomentumMatchText(t *testing.T) {
	n := 0
	for _, b := range contractAll(t) {
		e := decodeNum(t, b.body)
		if e.Agent != "Momentum Agent" || !e.OK {
			continue
		}
		m := e.Momentum
		if m == nil || m.Thresholds != (MomentumThresholds{RSIBullish: 55, RSIBearish: 45, MACDZero: 0}) || m.VolumeAverageBars != 20 {
			t.Errorf("%s: momentum readout %+v", b.path, m)
			continue
		}
		n++
		txt := e.text()
		if m.Counts != nil {
			c := m.Counts
			head := fmt.Sprintf("%d bullish", c.Bullish)
			if c.Bullish > 0 {
				head += " (" + strings.Join(c.BullishAssets, ", ") + ")"
			}
			head += fmt.Sprintf(" · %d bearish", c.Bearish)
			if c.Bearish > 0 {
				head += " (" + strings.Join(c.BearishAssets, ", ") + ")"
			}
			head += fmt.Sprintf(" · %d not confirmed", c.NotConfirmed)
			if !strings.HasPrefix(e.Verdict, head) {
				t.Errorf("%s: counts %+v, verdict %q", b.path, c, e.Verdict)
			}
		}
		if v := m.VolumeRatio; v != nil && !strings.Contains(txt, fmt.Sprintf("%.2f× its 20-bar average", *v)) {
			t.Errorf("%s: volume_ratio %v not printed", b.path, *v)
		}
		if v := m.ETHMinusBTC7dPP; v != nil && !strings.Contains(txt, "7d "+ppShown(*v)+" pp") {
			t.Errorf("%s: 7d RS %v not printed", b.path, *v)
		}
		for _, r := range e.Results {
			if r.OK && (r.RSIShownValue == nil || fmt.Sprintf("%.1f", *r.RSIShownValue) != r.RSIShown) {
				t.Errorf("%s: rsi_shown_value %v vs rsi_shown %q", b.path, r.RSIShownValue, r.RSIShown)
			}
		}
	}
	if n == 0 {
		t.Fatal("no momentum card in the harness")
	}
}

func TestNumbersMacroRuleNumbers(t *testing.T) {
	for _, key := range []string{"vix", "dxy", "rates", "gold", "spx"} {
		for _, r := range []MacroLampRule{riskRule(key), goldRule(key)} {
			check := func(when string, above, below *float64) {
				switch {
				case when == "":
					if above != nil || below != nil {
						t.Errorf("%s: number without a condition", key)
					}
				case above != nil:
					if below != nil || when != "> "+fmtNum(*above) {
						t.Errorf("%s: %q vs above %v", key, when, *above)
					}
				case below != nil:
					if when != "< "+fmtNum(*below) {
						t.Errorf("%s: %q vs below %v", key, when, *below)
					}
				default:
					t.Errorf("%s: condition %q without a number", key, when)
				}
			}
			check(r.PositiveWhen, r.PositiveAbove, r.PositiveBelow)
			check(r.NegativeWhen, r.NegativeAbove, r.NegativeBelow)
		}
	}
	for _, b := range contractAll(t) {
		e := decodeNum(t, b.body)
		if e.Macro == nil || e.Agent != "Macro Agent" {
			continue
		}
		m := e.Macro
		if m.NeutralScore != 50 || m.TotalLamps != len(m.Lamps) ||
			!strings.Contains(e.text(), fmt.Sprintf("%d of %d lamps live", m.LiveLamps, m.TotalLamps)) {
			t.Errorf("%s: neutral %d total %d", b.path, m.NeutralScore, m.TotalLamps)
		}
		if e.Blocks != nil && e.Blocks.WhatHappened != "" && strings.Contains(e.Blocks.WhatHappened, "positive,") &&
			!strings.Contains(e.Blocks.WhatHappened, fmt.Sprintf("%d positive, %d negative, %d neutral",
				m.Contributions.Positive, m.Contributions.Negative, m.Contributions.Neutral)) {
			t.Errorf("%s: contributions %+v vs %q", b.path, m.Contributions, e.Blocks.WhatHappened)
		}
	}
}

func TestNumbersGoldWhaleFXDigest(t *testing.T) {
	var gold, whale, fx, digest int
	for _, b := range contractAll(t) {
		e := decodeNum(t, b.body)
		txt := e.text()
		if g := e.Gold; g != nil && e.Agent == "Gold Agent" {
			gold++
			if nl := g.NearestLevels; nl != nil {
				for kind, l := range map[string]*GoldLevelRef{"support": nl.Support, "resistance": nl.Resistance} {
					if l != nil && !strings.Contains(txt, fmt.Sprintf("%s %s (%s, %s)", kind, l.Label,
						strings.ReplaceAll(l.Class, "_", " "), plural(l.Pivots, "pivot", "pivots"))) {
						t.Errorf("%s: %s %+v not printed", b.path, kind, *l)
					}
				}
			}
			if v := g.Volatility; v != nil && !strings.Contains(txt, fmt.Sprintf("ATR %.3f× its %d-bar baseline", v.ATRRatioShown, v.BaselineBars)) {
				t.Errorf("%s: volatility %+v not printed", b.path, *v)
			}
			if g.ADXShown != nil && e.Blocks != nil && !strings.Contains(e.Blocks.Regime, fmt.Sprintf("ADX %.1f", *g.ADXShown)) {
				t.Errorf("%s: adx_shown %v vs %q", b.path, *g.ADXShown, e.Blocks.Regime)
			}
		}
		if w := e.Whale; w != nil && w.Prior24hNetFlowUSD != nil {
			whale++
			line := "prior 24h net flow: " + usd(*w.Prior24hNetFlowUSD)
			if w.Prior24hChangePct != nil {
				line += fmt.Sprintf(" → %+.0f%% change", *w.Prior24hChangePct)
			}
			if !strings.Contains(strings.ToLower(txt), strings.ToLower(line)) {
				t.Errorf("%s: %q not on the card", b.path, line)
			}
		}
		if f := e.FX; f != nil {
			fx++
			if !strings.Contains(e.Verdict, fxPairsWord(f.PairsRead, f.PairsTotal)) && f.PairsTotal > 0 {
				t.Errorf("%s: fx counts %+v vs %q", b.path, *f, e.Verdict)
			}
		}
		if e.Digest != nil {
			digest++
			for _, s := range e.Digest.Sections {
				if !s.OK {
					continue
				}
				switch s.Key {
				case keySR, keyVol:
					if s.Levels == nil {
						t.Errorf("%s: section %s without levels", b.path, s.Key)
					}
				case keyMomentum, keyFunding, keyFX:
					if len(s.Results) == 0 {
						t.Errorf("%s: section %s without results", b.path, s.Key)
					}
					for _, r := range s.Results {
						if r.Blocks != nil {
							t.Errorf("%s: section %s results carry blocks", b.path, s.Key)
						}
					}
				case "narrative":
					if s.Theme == nil || !strings.Contains(strings.Join(s.Lines, " "), fmt.Sprintf("activity score %d", s.Theme.ActivityScore)) {
						t.Errorf("%s: narrative theme %+v vs %v", b.path, s.Theme, s.Lines)
					}
				}
			}
		}
	}
	if gold == 0 || whale == 0 || fx == 0 || digest == 0 {
		t.Fatalf("harness coverage: gold %d whale %d fx %d digest %d", gold, whale, fx, digest)
	}
}

// The RS numbers in machine form follow live prices like the line they
// mirror: a moved number is not a new reading, a window going away is.
func TestHookMasksMomentumRSNumbers(t *testing.T) {
	body := func(rs7 string) []byte {
		return []byte(`{"agent":"Momentum Agent","ok":true,"momentum":{"eth_minus_btc_7d_pp":` + rs7 +
			`,"eth_minus_btc_30d_pp":-9.4},"digest":{"sections":[{"momentum":{"eth_minus_btc_7d_pp":` + rs7 + `}}]}}`)
	}
	norm := func(b []byte) string {
		n, err := hookNormalize(keyMomentum, b, time.Time{}, time.Time{})
		if err != nil {
			t.Fatal(err)
		}
		return string(n)
	}
	if norm(body("-1.8")) != norm(body("-1.9")) {
		t.Error("a moved RS number changed the hash")
	}
	if norm(body("-1.8")) == norm(body("null")) {
		t.Error("an RS window going away must change the hash")
	}
	if strings.Contains(norm(body("-1.8")), "-1.8") || strings.Contains(norm(body("-1.8")), "-9.4") {
		t.Errorf("RS numbers left in the hash: %s", norm(body("-1.8")))
	}
}
