package demobot

// vol_readable_test.go — Volatility stage 1 (2026-09-15): a readable, honest
// card with the rule UNCHANGED (ATR(14) against the mean of its previous 30
// values; ≤ 0.80 compressed, ≥ 1.25 expanding, else normal). Golden texts, the
// printed ratio never crossing a threshold, precision per asset, content
// blocks, the machine fields in levels and the 110-character line budget on
// every path (Binance, Yahoo, weekend, insufficient, offline, digest line).

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

var (
	volEURUSD = assetTable["eurusd"]
	volUSDJPY = assetTable["usdjpy"]
	volETH    = assetTable["eth"]
	volAt     = time.Date(2026, 9, 15, 8, 0, 0, 0, time.UTC)
)

// ── the rule is unchanged ────────────────────────────────────────────────────

func TestVolRuleUnchanged(t *testing.T) {
	if volCompressedAt != 0.80 || volExpandingAt != 1.25 {
		t.Fatalf("thresholds moved: %v / %v", volCompressedAt, volExpandingAt)
	}
	cases := []struct {
		r    float64
		want string
	}{
		{0.7995, volCompressed}, {0.8, volCompressed}, {math.Nextafter(0.8, 1), volNormal},
		{0.8004, volNormal}, {1.0, volNormal}, {1.2495, volNormal},
		{math.Nextafter(1.25, 0), volNormal}, {1.25, volExpanding}, {1.3, volExpanding},
	}
	for _, tc := range cases {
		if got := volState(tc.r); got != tc.want {
			t.Errorf("volState(%v) = %q, want %q", tc.r, got, tc.want)
		}
	}
}

// ── the printed ratio ────────────────────────────────────────────────────────

func TestVolRatioShownGolden(t *testing.T) {
	cases := map[float64]string{
		0.7995: "0.800", 0.8: "0.800", 0.8004: "0.801", 0.804: "0.804",
		0.877838: "0.878", 1.0: "1.000", 1.246: "1.246", 1.2495: "1.249",
		1.25: "1.250", 1.311792: "1.311", 0: "0.000",
	}
	for r, want := range cases {
		if got := volRatioShown(r); got != want {
			t.Errorf("volRatioShown(%v) = %q, want %q", r, got, want)
		}
	}
}

// A printed ratio never lands on the other side of 0.80 or 1.25 from the real
// one, and is never more than 0.001 away (1.246 at %.2f used to print 1.25
// under a normal state).
func TestVolRatioShownNeverCrossesThreshold(t *testing.T) {
	var vals []float64
	for v := 0.0; v <= 3; v += 0.00001 {
		vals = append(vals, v)
	}
	for _, th := range []float64{volCompressedAt, 1, volExpandingAt} {
		for d := -0.002; d <= 0.002; d += 0.0001 {
			vals = append(vals, th+d)
		}
		vals = append(vals, math.Nextafter(th, 0), th, math.Nextafter(th, 10))
	}
	for _, v := range vals {
		p, err := strconv.ParseFloat(volRatioShown(v), 64)
		if err != nil {
			t.Fatalf("volRatioShown(%v) = %q", v, volRatioShown(v))
		}
		if volState(v) != volState(p) {
			t.Errorf("volRatioShown(%v) = %v reads %s, the rule says %s", v, p, volState(p), volState(v))
		}
		if math.Abs(p-v) > 0.001+1e-9 {
			t.Errorf("volRatioShown(%v) = %v is more than 0.001 away", v, p)
		}
	}
}

// The verdict's percentage is the printed ratio's distance from 1, truncated:
// it never reaches "20% below" inside the range or "25% above" below 1.25.
func TestVolPercentAgreesWithState(t *testing.T) {
	cases := map[float64]string{
		0.8004: "19% below its 30-bar baseline", 0.8: "20% below its 30-bar baseline",
		0.7995: "20% below its 30-bar baseline", 1.2495: "24% above its 30-bar baseline",
		1.25: "25% above its 30-bar baseline", 1.0: "within 1% of its 30-bar baseline",
		0.9915: "within 1% of its 30-bar baseline", 1.0099: "within 1% of its 30-bar baseline",
		1.011: "1% above its 30-bar baseline", 0.877838: "12% below its 30-bar baseline",
		500: ">999% above its 30-bar baseline",
	}
	for r, want := range cases {
		if got := volDistance(r); got != want {
			t.Errorf("volDistance(%v) = %q, want %q", r, got, want)
		}
	}
}

// ── the card ─────────────────────────────────────────────────────────────────

func volCard(spec assetSpec, atr, baseline, price float64) Card {
	return volCardFrom(spec, volRead{atr: atr, baseline: baseline, price: price, interval: spec.Interval}, volAt)
}

func TestVolCardGoldenBTCElevated(t *testing.T) {
	c := volCard(btcSpec, 1250.4, 953.2, 76406.6)
	if c.Verdict != "ELEVATED · 4h — ATR 31% above its 30-bar baseline" {
		t.Errorf("verdict: %q", c.Verdict)
	}
	want := []string{
		"ATR/baseline check: 1.311 ≥ 1.25 → elevated",
		"ATR(14): 1250.4, 1.64% of price · baseline = mean of the previous 30 ATR(14) values",
		"Agent's own thresholds, not a market benchmark: ≤ 0.80 compressed, ≥ 1.25 elevated, between them normal",
		"Read on closed 4h candles; measures how far price moves per candle, not direction or breakout",
	}
	if strings.Join(c.Facts, "\n") != strings.Join(want, "\n") {
		t.Errorf("facts:\n got %q\nwant %q", c.Facts, want)
	}
	if c.Short != "elevated · 4h · ATR 1.311× its 30-bar baseline" {
		t.Errorf("short: %q", c.Short)
	}
	if c.State != volExpanding || c.Emoji != emojiNeutral || c.Status != statusOK || !c.DataTime.Equal(volAt) {
		t.Errorf("state %q emoji %q status %v time %v", c.State, c.Emoji, c.Status, c.DataTime)
	}
}

func TestVolCardGoldenEURUSDWithinRange(t *testing.T) {
	c := volCard(volEURUSD, 0.000812, 0.000925, 1.17325)
	if c.Verdict != "NORMAL · 1h — ATR 12% below its 30-bar baseline" {
		t.Errorf("verdict: %q", c.Verdict)
	}
	want := []string{
		"ATR/baseline check: 0.80 < 0.878 < 1.25 → normal",
		"ATR(14): 0.00081, 0.07% of price · baseline = mean of the previous 30 ATR(14) values",
		"Agent's own thresholds, not a market benchmark: ≤ 0.80 compressed, ≥ 1.25 elevated, between them normal",
		"Read on closed 1h candles; measures how far price moves per candle, not direction or breakout",
	}
	if strings.Join(c.Facts, "\n") != strings.Join(want, "\n") {
		t.Errorf("facts:\n got %q\nwant %q", c.Facts, want)
	}
	if c.Short != "normal · 1h · ATR 0.878× its 30-bar baseline" {
		t.Errorf("short: %q", c.Short)
	}
	if c.State != volNormal {
		t.Errorf("state %q", c.State)
	}
}

func TestVolCardGoldenGoldCompressed(t *testing.T) {
	c := volCard(xauSpec, 14.2, 18.4, 3650.5)
	if c.Verdict != "COMPRESSED · 1h — ATR 22% below its 30-bar baseline" {
		t.Errorf("verdict: %q", c.Verdict)
	}
	if c.Facts[0] != "ATR/baseline check: 0.772 ≤ 0.80 → compressed" {
		t.Errorf("check: %q", c.Facts[0])
	}
	if c.Facts[1] != "ATR(14): 14.20, 0.39% of price · baseline = mean of the previous 30 ATR(14) values" {
		t.Errorf("atr line: %q", c.Facts[1])
	}
	if c.Asset != "GOLD · COMEX GC=F" || c.assetKey() != "XAUUSD" {
		t.Errorf("asset %q key %q", c.Asset, c.assetKey())
	}
}

// Boundary reads: the check line and the verdict never disagree with the rule.
func TestVolCardBoundaries(t *testing.T) {
	cases := []struct {
		ratio   float64
		verdict string
		check   string
	}{
		{0.8, "COMPRESSED · 4h — ATR 20% below its 30-bar baseline", "ATR/baseline check: 0.800 ≤ 0.80 → compressed"},
		{0.7995, "COMPRESSED · 4h — ATR 20% below its 30-bar baseline", "ATR/baseline check: 0.800 ≤ 0.80 → compressed"},
		{0.8004, "NORMAL · 4h — ATR 19% below its 30-bar baseline", "ATR/baseline check: 0.80 < 0.801 < 1.25 → normal"},
		{1.2495, "NORMAL · 4h — ATR 24% above its 30-bar baseline", "ATR/baseline check: 0.80 < 1.249 < 1.25 → normal"},
		{1.246, "NORMAL · 4h — ATR 24% above its 30-bar baseline", "ATR/baseline check: 0.80 < 1.246 < 1.25 → normal"},
		{1.25, "ELEVATED · 4h — ATR 25% above its 30-bar baseline", "ATR/baseline check: 1.250 ≥ 1.25 → elevated"},
	}
	for _, tc := range cases {
		c := volCard(btcSpec, tc.ratio, 1, 100)
		if c.Verdict != tc.verdict || c.Facts[0] != tc.check {
			t.Errorf("ratio %v:\n got %q | %q\nwant %q | %q", tc.ratio, c.Verdict, c.Facts[0], tc.verdict, tc.check)
		}
		if c.State != volState(tc.ratio) {
			t.Errorf("ratio %v: state %q, rule %q", tc.ratio, c.State, volState(tc.ratio))
		}
	}
}

// ATR prints one decimal finer than the card's price precision: FX 5, JPY 3,
// gold 2, ETH 2, BTC 1.
func TestVolATRPrecisionPerAsset(t *testing.T) {
	cases := []struct {
		spec assetSpec
		atr  float64
		px   float64
		want string
	}{
		{btcSpec, 812.46, 76000, "ATR(14): 812.5,"},
		{volETH, 41.268, 3500, "ATR(14): 41.27,"},
		{volEURUSD, 0.000812, 1.17, "ATR(14): 0.00081,"},
		{assetTable["gbpusd"], 0.00123, 1.35, "ATR(14): 0.00123,"},
		{volUSDJPY, 0.1234, 147.5, "ATR(14): 0.123,"},
		{xauSpec, 18.694, 3650, "ATR(14): 18.69,"},
	}
	for _, tc := range cases {
		c := volCard(tc.spec, tc.atr, tc.atr, tc.px)
		if !strings.HasPrefix(c.Facts[1], tc.want) {
			t.Errorf("%s: %q, want prefix %q", tc.spec.Display, c.Facts[1], tc.want)
		}
	}
}

// The old card printed ATR 0.0008 beside an average 0.0009 and a 0.92× ratio:
// the reader got ~0.89×. The card now prints no second absolute number, so no
// pair of printed numbers can give another ratio than the printed one.
func TestVolCardPrintsOneAbsoluteNumber(t *testing.T) {
	c := volCard(volEURUSD, 0.000812, 0.000925, 1.17325)
	all := strings.Join(c.Facts, "\n") + "\n" + c.Verdict
	if strings.Contains(all, "0.00092") || strings.Contains(all, "0.0009") {
		t.Errorf("the baseline must not be printed in absolute units: %q", all)
	}
}

// No forecast, no dynamics, no market-norm claim, no direction.
func TestVolCardWordingRules(t *testing.T) {
	banned := []string{"often follows", "widening", "expanding", "EXPANDING", "expansion signal",
		"range", "RANGE", "breakout conditions", "will ", "likely", "probab"}
	for _, r := range []float64{0.5, 0.8, 0.9, 1.0, 1.2, 1.25, 2} {
		c := volCard(btcSpec, r, 1, 100)
		for _, s := range volTexts(c) {
			for _, b := range banned {
				if strings.Contains(s, b) {
					t.Errorf("ratio %v: %q contains %q", r, s, b)
				}
			}
			rest := strings.ReplaceAll(s, "normal", "") // no "norm" beside "normal"; thresholds are "not a market benchmark"
			if strings.Contains(rest, "norm") {
				t.Errorf("ratio %v: %q names a norm", r, s)
			}
		}
	}
}

// volTexts is every human sentence of a card: verdict, short, facts, blocks.
func volTexts(c Card) []string {
	out := append([]string{c.Verdict, c.Short}, c.Facts...)
	if b := c.Blocks; b != nil {
		out = append(out, b.WhatHappened, b.WhyLevel, b.Regime, b.StateChanges, b.Limitations)
		out = append(out, b.Scenarios...)
	}
	return out
}

// ── content blocks ───────────────────────────────────────────────────────────

func TestVolBlocksGolden(t *testing.T) {
	b := volCard(btcSpec, 1.05, 1, 76000).Blocks
	if b == nil {
		t.Fatal("vol card must carry blocks")
	}
	if b.WhatHappened != "BTC 4h ATR(14) is 5% above its 30-bar baseline (ratio 1.050): normal." {
		t.Errorf("what_happened: %q", b.WhatHappened)
	}
	if b.WhyLevel != "No price level: 0.80 and 1.25 are the agent's own ATR/baseline thresholds, not a market benchmark" {
		t.Errorf("why_level: %q", b.WhyLevel)
	}
	wantSc := []string{
		"If a closed 4h candle puts the ratio at 0.80 or below, the state turns compressed",
		"If a closed 4h candle puts the ratio at 1.25 or above, the state turns elevated",
	}
	if strings.Join(b.Scenarios, "\n") != strings.Join(wantSc, "\n") {
		t.Errorf("scenarios: %q", b.Scenarios)
	}
	if b.Invalidates != nil {
		t.Errorf("volatility has no directional idea to invalidate: %q", *b.Invalidates)
	}
	if b.StateChanges != "A closed 4h candle with the ratio at 0.80 or below (compressed) or 1.25 or above (elevated) changes the state" {
		t.Errorf("state_changes_when: %q", b.StateChanges)
	}
	if b.Regime != "Local amplitude regime · BTC 4h · normal: ATR 1.050× its 30-bar baseline" {
		t.Errorf("regime: %q", b.Regime)
	}
	if b.Limitations != "Measures how far price moves per candle, not its direction; it does not confirm a breakout" {
		t.Errorf("limitations: %q", b.Limitations)
	}
}

func TestVolBlocksPerState(t *testing.T) {
	up := volCard(btcSpec, 1.3, 1, 100).Blocks
	if up.Scenarios[0] != "If closed 4h candles keep the ratio at 1.25 or above, the state stays elevated" ||
		up.Scenarios[1] != "If a closed 4h candle puts the ratio below 1.25, the state turns normal (compressed at 0.80 or below)" {
		t.Errorf("elevated scenarios: %q", up.Scenarios)
	}
	if up.StateChanges != "A closed 4h candle with the ratio below 1.25 ends the elevated state" {
		t.Errorf("elevated state_changes_when: %q", up.StateChanges)
	}
	down := volCard(volEURUSD, 0.7, 1, 1.1).Blocks
	if down.Scenarios[0] != "If closed 1h candles keep the ratio at 0.80 or below, the state stays compressed" ||
		down.Scenarios[1] != "If a closed 1h candle puts the ratio above 0.80, the state turns normal (elevated at 1.25 or above)" {
		t.Errorf("compressed scenarios: %q", down.Scenarios)
	}
	if down.StateChanges != "A closed 1h candle with the ratio above 0.80 ends the compressed state" {
		t.Errorf("compressed state_changes_when: %q", down.StateChanges)
	}
}

// ── machine fields ───────────────────────────────────────────────────────────

func TestVolLevelsJSON(t *testing.T) {
	c := volCard(volEURUSD, 0.000812, 0.000925, 1.17325)
	raw, err := json.Marshal(cardEnvelope(c))
	if err != nil {
		t.Fatal(err)
	}
	var env struct {
		Levels map[string]any `json:"levels"`
		Blocks map[string]any `json:"blocks"`
	}
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatal(err)
	}
	l := env.Levels
	atr, base := 0.000812, 0.000925 // variables: a constant quotient is folded at exact precision
	ratio := atr / base
	if l["expansion_ratio"] != ratio || l["ratio"] != ratio {
		t.Errorf("ratio fields must be the raw ratio %v: %v / %v", ratio, l["expansion_ratio"], l["ratio"])
	}
	if l["state"] != "normal" || l["timeframe"] != "1h" || l["atr"] != 0.000812 || l["baseline"] != 0.000925 {
		t.Errorf("levels: %v", l)
	}
	px := 1.17325
	if got, want := l["atr_pct"], atr/px*100; got != want {
		t.Errorf("atr_pct %v, want raw %v", got, want)
	}
	th, ok := l["thresholds"].(map[string]any)
	if !ok || th["compressed"] != 0.8 || th["expanding"] != 1.25 {
		t.Errorf("thresholds: %v", l["thresholds"])
	}
	if env.Blocks == nil || env.Blocks["invalidates"] != nil || env.Blocks["state_changes_when"] == nil || env.Blocks["limitations"] == nil {
		t.Errorf("blocks: %v", env.Blocks)
	}
	// Other agents' blocks do not grow the vol-only keys.
	other, _ := json.Marshal(ContentBlocks{WhatHappened: "x", Scenarios: []string{}})
	if strings.Contains(string(other), "state_changes_when") || strings.Contains(string(other), "limitations") {
		t.Errorf("vol-only block keys leak into other agents: %s", other)
	}
}

// ── line budget on every path ────────────────────────────────────────────────

func volAssertLines(t *testing.T, name string, c Card) {
	t.Helper()
	lines := append([]string{c.Verdict, htmlToPlain(c.OneLiner())}, c.Facts...)
	if b := c.Blocks; b != nil {
		lines = append(lines, b.WhatHappened, b.WhyLevel, b.Regime, b.StateChanges, b.Limitations)
		lines = append(lines, b.Scenarios...)
	}
	for _, l := range lines {
		if n := utf8.RuneCountInString(l); n > volFactMaxRunes {
			t.Errorf("%s: %d runes > %d: %q", name, n, volFactMaxRunes, l)
		}
	}
	content := 0
	for _, f := range c.Facts {
		if f != fxClosedBanner {
			content++
		}
	}
	if content > 4 {
		t.Errorf("%s: %d content lines > 4: %q", name, content, c.Facts)
	}
}

func TestVolLinesFitEveryBuilderPath(t *testing.T) {
	specs := []assetSpec{btcSpec, volETH, volEURUSD, assetTable["gbpusd"], volUSDJPY, xauSpec, goldDailySpec}
	ratios := []float64{0, 0.001, 0.5, 0.7995, 0.8, 0.8004, 1, 1.2495, 1.25, 3, 99.9, 1e6, 1e12}
	sat := time.Date(2026, 9, 19, 12, 0, 0, 0, time.UTC) // Saturday: FX closed
	for _, s := range specs {
		for _, r := range ratios {
			for _, px := range []float64{0.0001, 1.1, 150, 76000, 1e9} {
				c := volCard(s, r*px/100, px/100, px)
				volAssertLines(t, s.Display, c)
				if s.Source == srcYahoo {
					decorateFXAt(&c, sat)
					volAssertLines(t, s.Display+" weekend", c)
					if c.Facts[0] != fxClosedBanner {
						t.Errorf("%s weekend: banner missing: %q", s.Display, c.Facts)
					}
				}
			}
		}
	}
}

// The real VolCard on Binance, Yahoo (open and weekend), short history and a
// dead source.
func TestVolCardPathsEndToEnd(t *testing.T) {
	ctx := context.Background()

	t.Run("binance live", func(t *testing.T) {
		stubExternalBases(t)
		stubBinanceKlinesWave(t, binanceFetchLimit)
		ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
		c := ag.VolCard(ctx, btcSpec)
		if c.Status != statusOK || c.Blocks == nil || !strings.Contains(c.Verdict, " · 4h — ATR ") {
			t.Fatalf("binance card: status %v verdict %q", c.Status, c.Verdict)
		}
		volAssertLines(t, "binance", c)
		lv, ok := c.Levels.(VolLevels)
		if !ok || lv.Ratio != lv.ExpansionRatio || lv.Ratio != lv.ATR/lv.Baseline || lv.Timeframe != "4h" || lv.State != c.State {
			t.Errorf("levels: %+v", c.Levels)
		}
		// The body of a stamped Binance card never follows the request clock.
		ag.now = func() time.Time { return time.Now().Add(72 * time.Hour) }
		c2 := ag.VolCard(ctx, btcSpec)
		a, _ := json.Marshal(cardEnvelope(c))
		b, _ := json.Marshal(cardEnvelope(c2))
		if stripFooter(string(a)) != stripFooter(string(b)) || !c.DataTime.Equal(c2.DataTime) {
			t.Errorf("binance vol body moved with the clock:\n%s\n%s", a, b)
		}
	})

	t.Run("yahoo weekend", func(t *testing.T) {
		fri := lastFridayBefore(time.Now().UTC().AddDate(0, 0, -7))
		stubExternalBases(t)
		stubYahooWave(t, fri.Add(20*time.Hour), 600)
		ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
		ag.now = func() time.Time { return fri.Add(26 * time.Hour) }
		c := ag.VolCard(ctx, volEURUSD)
		if c.Status != statusOK || c.Facts[0] != fxClosedBanner || !c.noValidator || c.SourceNote == "" {
			t.Fatalf("yahoo weekend: status %v facts %q noValidator %v", c.Status, c.Facts, c.noValidator)
		}
		volAssertLines(t, "yahoo weekend", c)
		ag.now = func() time.Time { return fri.Add(-24 * time.Hour) }
		open := ag.VolCard(ctx, volEURUSD)
		if open.Facts[0] == fxClosedBanner {
			t.Errorf("banner on an open market: %q", open.Facts)
		}
		volAssertLines(t, "yahoo open", open)
	})

	t.Run("insufficient", func(t *testing.T) {
		stubExternalBases(t)
		stubBinanceKlinesWave(t, 40)
		ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
		c := ag.VolCard(ctx, btcSpec)
		if c.Status != statusInsufficientHistory || c.Blocks != nil || c.Levels != nil {
			t.Fatalf("short history: status %v blocks %v levels %v", c.Status, c.Blocks, c.Levels)
		}
		volAssertLines(t, "insufficient", c)
	})

	t.Run("offline", func(t *testing.T) {
		stubExternalBases(t)
		ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
		for _, s := range []assetSpec{btcSpec, volEURUSD, xauSpec} {
			c := ag.VolCard(ctx, s)
			if c.Status != statusSourceOffline || c.Blocks != nil || c.Levels != nil {
				t.Errorf("%s offline: status %v", s.Display, c.Status)
			}
			volAssertLines(t, s.Display+" offline", c)
		}
	})
}

// A zero baseline and non-finite input degrade; they never become a reading.
func TestVolDegenerateInputsDegrade(t *testing.T) {
	for name, r := range map[string]volRead{
		"zero baseline": {atr: 1, baseline: 0, price: 100, interval: "4h"},
		"nan atr":       {atr: math.NaN(), baseline: 1, price: 100, interval: "4h"},
		"inf baseline":  {atr: 1, baseline: math.Inf(1), price: 100, interval: "4h"},
		"zero price":    {atr: 1, baseline: 1, price: 0, interval: "4h"},
	} {
		if r.valid() {
			t.Errorf("%s: must not be a valid read", name)
		}
	}
	if !(volRead{atr: 1, baseline: 1, price: 100, interval: "4h"}).valid() {
		t.Error("a plain read must be valid")
	}
}

// The invalid-read path through the real VolCard: each degenerate window
// degrades to insufficient_history with its own reason, never a reading.
func TestVolCardInvalidReadsDegrade(t *testing.T) {
	cases := []struct {
		name   string
		bar    func(i int) (h, l, c float64)
		reason string
	}{
		{"flat zero baseline", func(int) (float64, float64, float64) { return 60000, 60000, 60000 }, volReasonFlatZero},
		{"overflowing prices", func(int) (float64, float64, float64) { return 1e308, -1e308, 60000 }, volReasonNonFinite},
		{"non-positive close", func(i int) (float64, float64, float64) {
			p := -100 + float64(i%10)*10
			return p + 50, p - 50, p
		}, volReasonNoPrice},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stubExternalBases(t)
			now := time.Now().UTC().Truncate(time.Second)
			lastClose := now.Add(-time.Hour)
			series := btcSeries(binanceFetchLimit, lastClose)
			for i := range series {
				h, l, c := tc.bar(i)
				series[i].Open, series[i].High, series[i].Low, series[i].Close = c, h, l, c
			}
			ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
			seedBTC4h(ag, withForming(series[1:], lastClose), now)
			c := ag.VolCard(context.Background(), btcSpec)
			if c.Status != statusInsufficientHistory || c.Levels != nil || c.Blocks != nil {
				t.Fatalf("status %v levels %v blocks %v; want insufficient_history and no readout", c.Status, c.Levels, c.Blocks)
			}
			if len(c.Facts) != 1 || c.Facts[0] != "insufficient history for "+tc.reason {
				t.Errorf("reason: %q, want %q", c.Facts, tc.reason)
			}
			volAssertLines(t, tc.name, c)
			if _, err := json.Marshal(cardEnvelope(c)); err != nil {
				t.Errorf("envelope must marshal: %v", err)
			}
		})
	}
}

// The AI payload carries the card's state word, never the machine
// "expanding"; confirmation still follows the machine state.
func TestVolAIPayloadStateWord(t *testing.T) {
	for state, word := range map[string]string{volExpanding: "elevated", volNormal: "normal", volCompressed: "compressed"} {
		g := fakeGathered()
		c := volCard(btcSpec, 1, 1, 100)
		c.State = state
		g.cards[keyVol] = c
		var p struct {
			Agents map[string]struct {
				State    string `json:"state"`
				Withheld bool   `json:"confirmation_withheld"`
			} `json:"agents"`
		}
		raw := aiPayload(g)
		if err := json.Unmarshal([]byte(raw), &p); err != nil {
			t.Fatalf("payload: %v\n%s", err, raw)
		}
		v := p.Agents[keyVol]
		if v.State != word || strings.Contains(raw, `"expanding"`) {
			t.Errorf("%s: payload state %q, want %q (%s)", state, v.State, word, raw)
		}
		if v.Withheld != (state != volExpanding) {
			t.Errorf("%s: confirmation_withheld %v — stateConfirms must keep reading the machine state", state, v.Withheld)
		}
	}
}

// The Gold card's volatility line is the Volatility card's own short line:
// human word, ratio toward 1 at three decimals, timeframe, baseline.
func TestVolGoldLineWords(t *testing.T) {
	cases := map[float64]string{
		1.246:  "normal · 1d · ATR 1.246× its 30-bar baseline",
		0.8004: "normal · 1d · ATR 0.801× its 30-bar baseline",
		1.25:   "elevated · 1d · ATR 1.250× its 30-bar baseline",
		0.8:    "compressed · 1d · ATR 0.800× its 30-bar baseline",
	}
	for r, want := range cases {
		if got := volShortLine(r, goldDailySpec.Interval); got != want {
			t.Errorf("volShortLine(%v) = %q, want %q", r, got, want)
		}
	}
	golden, err := os.ReadFile(filepath.Join("testdata", "gold_sr_golden.txt"))
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range strings.Split(string(golden), "\n") {
		if !strings.HasPrefix(l, "fact: Volatility:") {
			continue
		}
		if !strings.Contains(l, " · 1d · ATR ") || strings.Contains(l, "average") || utf8.RuneCountInString(strings.TrimPrefix(l, "fact: ")) > volFactMaxRunes {
			t.Errorf("gold volatility line: %q", l)
		}
	}
}

// ── how-it-works and the digest line ─────────────────────────────────────────

func TestVolHowText(t *testing.T) {
	h := howTexts[keyVol]
	if n := utf8.RuneCountInString(h); n > 200 {
		t.Errorf("how-text %d > 200 runes: %q", n, h)
	}
	for _, want := range []string{"0.80", "1.25", "closed", "not a market benchmark", "not direction"} {
		if !strings.Contains(h, want) {
			t.Errorf("how-text lacks %q: %q", want, h)
		}
	}
	if strings.Contains(h, "expanding") {
		t.Errorf("how-text still says expanding: %q", h)
	}
}

func TestVolDigestOneLiner(t *testing.T) {
	c := volCard(btcSpec, 1.05, 1, 76000)
	if got := htmlToPlain(c.OneLiner()); got != "⚪ Volatility BTC: normal · 4h · ATR 1.050× its 30-bar baseline" {
		t.Errorf("one-liner: %q", got)
	}
}
