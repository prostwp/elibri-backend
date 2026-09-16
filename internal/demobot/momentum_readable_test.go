package demobot

// momentum_readable_test.go — Momentum stage 1 (2026-09-15): a readable,
// honest card with the rule UNCHANGED. Golden texts for the neutral reasons,
// the two-condition checklist, the overview counter and its semaphore rule,
// per-asset freshness, content blocks, the results[] JSON fields and the
// 110-character line budget on every path.

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// Tuesday 2026-09-15 09:30 UTC — FX week open.
var momNow = time.Date(2026, 9, 15, 9, 30, 0, 0, time.UTC)

func momRead(spec assetSpec, rsi, hist float64, closeAt time.Time) momentumRead {
	return momentumRead{
		name: spec.Display, verdict: momentumVerdict(rsi, hist), rsi: rsi, hist: hist,
		source: spec.Source, interval: spec.Interval, closeAt: closeAt,
	}
}

// ── neutral with a reason ────────────────────────────────────────────────────

func TestMomentumWhyAndState(t *testing.T) {
	cases := []struct {
		rsi, hist float64
		state     string
		why       string
	}{
		{54.7, 217.3, "rsi_below_55", "RSI below the 55 threshold"},     // BTC 4h live
		{61.3, -16.3, "conflict", "conflict: RSI up, MACD down"},        // ETH 1d live
		{37.9, 0.0001284, "conflict", "conflict: RSI down, MACD up"},    // EURUSD 1h live
		{50.0, -0.000874, "rsi_above_45", "RSI above the 45 threshold"}, // EURUSD 1d live
		{50.0, 0, "neutral_zone", "RSI in the neutral zone 45–55"},      // both sides fail
		{60, 0, "macd_at_zero", "RSI up, MACD histogram at 0"},          // RSI past 55, MACD flat
		{40, 0, "macd_at_zero", "RSI down, MACD histogram at 0"},        // RSI past 45, MACD flat
		{62, 5, "confirmed_bullish", "RSI and MACD agree"},              // rule: bullish
		{30, -1, "confirmed_bearish", "RSI and MACD agree"},             // rule: bearish
		{55, 0.1, "confirmed_bullish", "RSI and MACD agree"},            // on the threshold
		{45, -0.1, "confirmed_bearish", "RSI and MACD agree"},           // on the threshold
	}
	for _, tc := range cases {
		if got := momentumState(tc.rsi, tc.hist); got != tc.state {
			t.Errorf("state(%v, %v) = %q, want %q", tc.rsi, tc.hist, got, tc.state)
		}
		if got := momentumWhy(tc.rsi, tc.hist); got != tc.why {
			t.Errorf("why(%v, %v) = %q, want %q", tc.rsi, tc.hist, got, tc.why)
		}
	}
	// The reason never contradicts the rule: confirmed_* ⇔ bullish/bearish.
	for rsi := 0.0; rsi <= 100; rsi += 0.5 {
		for _, h := range []float64{-1, 0, 1} {
			st, v := momentumState(rsi, h), momentumVerdict(rsi, h)
			if (st == "confirmed_bullish") != (v == "bullish") || (st == "confirmed_bearish") != (v == "bearish") {
				t.Errorf("RSI %v hist %v: state %q vs verdict %q", rsi, h, st, v)
			}
		}
	}
}

// ── the checklist ────────────────────────────────────────────────────────────

func TestMomentumChecklistGolden(t *testing.T) {
	cases := []struct {
		rsi, hist float64
		want      string
	}{
		{54.7, 217.3, "RSI 54.7 < 55 ✗ · MACD histogram above 0 ✓"},
		{61.3, -16.3, "RSI 61.3 ≥ 55 ✓ · MACD histogram below 0 ✗"},
		{37.9, 0.0001284, "RSI 37.9 ≤ 45 ✓ · MACD histogram above 0 ✗"},
		{50.0, -0.000874, "RSI 50.0 > 45 ✗ · MACD histogram below 0 ✓"},
		{50.0, 0, "RSI 50.0 in 45–55 ✗ · MACD histogram at 0 ✗"},
		{62, 5, "RSI 62.0 ≥ 55 ✓ · MACD histogram above 0 ✓"},
		{30, -1, "RSI 30.0 ≤ 45 ✓ · MACD histogram below 0 ✓"},
	}
	for _, tc := range cases {
		if got := momentumChecklist(tc.rsi, tc.hist); got != tc.want {
			t.Errorf("checklist(%v, %v):\n got %q\nwant %q", tc.rsi, tc.hist, got, tc.want)
		}
	}
}

// All ✓ toward a direction is exactly the rule's verdict for it — the
// checklist is one enumeration of the rule, never a paraphrase.
func TestMomentumChecklistMatchesRule(t *testing.T) {
	for rsi := 0.0; rsi <= 100; rsi += 0.25 {
		for _, h := range []float64{-1e9, -1e-9, 0, 1e-9, 1e9} {
			for _, d := range []string{"bullish", "bearish"} {
				all := true
				for _, c := range momentumChecks(rsi, h, d) {
					all = all && c.ok
				}
				if all != (momentumVerdict(rsi, h) == d) {
					t.Errorf("RSI %v hist %v toward %s: all ✓ = %v, verdict %q", rsi, h, d, all, momentumVerdict(rsi, h))
				}
			}
		}
	}
}

// A printed RSI never lands on the wrong side of a threshold: 54.96 must not
// print "55.0 < 55", 45.04 must not print "45.0 > 45".
func TestRSIShownNeverCrossesThreshold(t *testing.T) {
	vals := []float64{}
	for v := 0.0; v <= 100; v += 0.001 {
		vals = append(vals, v)
	}
	for _, th := range []float64{45, 50, 55} {
		vals = append(vals, math.Nextafter(th, 0), th, math.Nextafter(th, 100), th-0.04, th+0.04, th-0.05, th+0.05)
	}
	for _, v := range vals {
		p, err := strconv.ParseFloat(rsiShown(v), 64)
		if err != nil {
			t.Fatalf("rsiShown(%v) = %q", v, rsiShown(v))
		}
		if (v >= 55) != (p >= 55) || (v <= 45) != (p <= 45) {
			t.Errorf("rsiShown(%v) = %v crosses a threshold", v, p)
		}
		if math.Abs(p-v) > 0.1+1e-9 {
			t.Errorf("rsiShown(%v) = %v is more than 0.1 away", v, p)
		}
	}
}

// ── the single-asset card ────────────────────────────────────────────────────

func TestMomentumAssetCardGolden(t *testing.T) {
	r := momRead(btcSpec, 54.7, 217.3, time.Date(2026, 9, 15, 8, 0, 0, 0, time.UTC))
	c := momentumAssetCardFrom(btcSpec, r, 0.96, true, momNow)
	if c.Verdict != "NOT CONFIRMED · 4h — RSI below the 55 threshold" {
		t.Errorf("verdict: %q", c.Verdict)
	}
	want := []string{
		"Why: RSI 54.7 < 55 ✗ · MACD histogram above 0 ✓",
		"Turns bullish when RSI ≥ 55 and the MACD histogram is above 0 (now ✗: RSI)",
		"Turns bearish when RSI ≤ 45 and the MACD histogram is below 0 (now ✗: RSI, MACD)",
		"Read on closed 4h candles: RSI(14) and the MACD(12,26,9) histogram",
		"Context, not part of the reading: volume 0.96× its 20-bar average (4h)",
	}
	if strings.Join(c.Facts, "\n") != strings.Join(want, "\n") {
		t.Errorf("facts:\n got %q\nwant %q", c.Facts, want)
	}
	if c.Emoji != emojiNeutral || c.Short != "neutral" || c.confirmed || c.Deviation != 0 {
		t.Errorf("contract: emoji %q short %q confirmed %v dev %d", c.Emoji, c.Short, c.confirmed, c.Deviation)
	}

	// Confirmed: the direction line says what keeps it; bearish comes second.
	r = momRead(btcSpec, 62, 120.5, r.closeAt)
	c = momentumAssetCardFrom(btcSpec, r, 0, false, momNow)
	if c.Verdict != "BULLISH · 4h — RSI and MACD agree" {
		t.Errorf("confirmed verdict: %q", c.Verdict)
	}
	if c.Facts[1] != "Stays bullish while RSI ≥ 55 and the MACD histogram is above 0; any ✗ turns it neutral" {
		t.Errorf("hold line: %q", c.Facts[1])
	}
	if c.Facts[2] != "Turns bearish when RSI ≤ 45 and the MACD histogram is below 0 (now ✗: RSI, MACD)" {
		t.Errorf("other direction: %q", c.Facts[2])
	}
	if c.Emoji != emojiBull || c.Short != "bullish" || !c.confirmed || c.Deviation != 24 {
		t.Errorf("confirmed contract: emoji %q short %q confirmed %v dev %d", c.Emoji, c.Short, c.confirmed, c.Deviation)
	}

	// A bearish lean lists bearish first.
	eur := assetTable["eurusd"]
	r = momRead(eur, 37.9, 0.0001284, time.Date(2026, 9, 15, 9, 0, 0, 0, time.UTC))
	c = momentumAssetCardFrom(eur, r, 0, false, momNow)
	if c.Verdict != "NOT CONFIRMED · 1h — conflict: RSI down, MACD up" {
		t.Errorf("eurusd verdict: %q", c.Verdict)
	}
	if !strings.HasPrefix(c.Facts[1], "Turns bearish when") || !strings.HasPrefix(c.Facts[2], "Turns bullish when") {
		t.Errorf("bearish lean must come first: %q", c.Facts)
	}
}

// The raw MACD histogram (price units — incomparable across assets) never
// reaches the card; only its sign does. The raw value lives in results[].
func TestMomentumCardShowsHistogramSignOnly(t *testing.T) {
	r := momRead(btcSpec, 54.7, 217.3456, time.Date(2026, 9, 15, 8, 0, 0, 0, time.UTC))
	c := momentumAssetCardFrom(btcSpec, r, 0, false, momNow)
	if html := c.RenderHTML(); strings.Contains(html, "217") {
		t.Errorf("raw histogram leaked into the card:\n%s", html)
	}
	if len(c.Results) != 1 || c.Results[0].MACDHistogram == nil || *c.Results[0].MACDHistogram != 217.3456 {
		t.Fatalf("results[0] must carry the raw histogram: %+v", c.Results)
	}
}

// ── the overview / scan composite ────────────────────────────────────────────

func defaultTrio(btc, eth, gold momentumRead) []momentumAsset {
	return []momentumAsset{
		{spec: btcSpec, read: btc, status: statusOK},
		{spec: assetTable["eth"], read: eth, status: statusOK},
		{spec: xauSpec, read: gold, status: statusOK},
	}
}

func TestMomentumOverviewGolden(t *testing.T) {
	bar4h := time.Date(2026, 9, 15, 8, 0, 0, 0, time.UTC)
	bar1h := time.Date(2026, 9, 15, 9, 0, 0, 0, time.UTC)
	assets := defaultTrio(
		momRead(btcSpec, 54.7, 217.3, bar4h),
		momRead(assetTable["eth"], 61.3, -16.3, bar4h),
		momRead(xauSpec, 37.9, 3.66, bar1h),
	)
	c := Card{DataTime: momNow}
	composeMomentum(&c, assets, "", momNow)
	if c.Verdict != "0 bullish · 0 bearish · 3 not confirmed" {
		t.Errorf("header: %q", c.Verdict)
	}
	want := []string{
		"BTC · 4h: not confirmed — RSI below the 55 threshold",
		"BTC: RSI 54.7 < 55 ✗ · MACD histogram above 0 ✓ · last bar Sep 15 08:00 UTC",
		"ETH · 4h: not confirmed — conflict: RSI up, MACD down",
		"ETH: RSI 61.3 ≥ 55 ✓ · MACD histogram below 0 ✗ · last bar Sep 15 08:00 UTC",
		"GOLD · COMEX GC=F · 1h: not confirmed — conflict: RSI down, MACD up",
		"GOLD · COMEX GC=F: RSI 37.9 ≤ 45 ✓ · MACD histogram above 0 ✗ · last bar Sep 15 09:00 UTC",
		"Bullish needs RSI ≥ 55 and MACD histogram above 0; bearish needs RSI ≤ 45 and below 0",
		"Colour follows BTC/ETH only, the reads the digest ranks; other assets are only counted",
	}
	if strings.Join(c.Facts, "\n") != strings.Join(want, "\n") {
		t.Errorf("facts:\n got %q\nwant %q", c.Facts, want)
	}
	if c.Emoji != emojiNeutral || c.Short != c.Verdict {
		t.Errorf("emoji %q short %q", c.Emoji, c.Short)
	}
	if !c.DataTime.Equal(bar4h) {
		t.Errorf("data time = oldest bar: %v", c.DataTime)
	}
	for _, raw := range []string{"217", "16.3", "3.66"} {
		if strings.Contains(c.RenderHTML(), raw) {
			t.Errorf("raw histogram %s on the card", raw)
		}
	}
}

// The overview's colour comes from the counter, never from the first asset:
// green/red only when every confirmed reading points the same way.
func TestMomentumOverviewSemaphoreFromCounter(t *testing.T) {
	bar := time.Date(2026, 9, 15, 8, 0, 0, 0, time.UTC)
	gold := momRead(xauSpec, 50, 0.5, bar.Add(time.Hour))
	cases := []struct {
		name       string
		btc, eth   [2]float64
		wantHeader string
		wantEmoji  string
	}{
		{"BTC neutral, ETH bullish", [2]float64{50, 1}, [2]float64{60, 1},
			"1 bullish (ETH) · 0 bearish · 2 not confirmed", emojiBull},
		{"BTC bullish, ETH bearish", [2]float64{60, 1}, [2]float64{40, -1},
			"1 bullish (BTC) · 1 bearish (ETH) · 1 not confirmed", emojiNeutral},
		{"both bearish", [2]float64{40, -1}, [2]float64{30, -2},
			"0 bullish · 2 bearish (BTC, ETH) · 1 not confirmed", emojiBear},
	}
	for _, tc := range cases {
		c := Card{DataTime: momNow}
		composeMomentum(&c, defaultTrio(
			momRead(btcSpec, tc.btc[0], tc.btc[1], bar),
			momRead(assetTable["eth"], tc.eth[0], tc.eth[1], bar), gold), "", momNow)
		if c.Verdict != tc.wantHeader || c.Emoji != tc.wantEmoji {
			t.Errorf("%s: header %q emoji %q, want %q %q", tc.name, c.Verdict, c.Emoji, tc.wantHeader, tc.wantEmoji)
		}
	}

	// A shared timeframe rides in the header; failures are counted.
	c := Card{DataTime: momNow}
	composeMomentum(&c, []momentumAsset{
		{spec: btcSpec, read: momRead(btcSpec, 50, 1, bar), status: statusOK},
		{spec: assetTable["eth"], read: momRead(assetTable["eth"], 50, 1, bar), status: statusOK},
		{spec: xauSpec, status: statusSourceOffline},
	}, "", momNow)
	if c.Verdict != "0 bullish · 0 bearish · 2 not confirmed · 1 unavailable · 4h" {
		t.Errorf("header with a failure: %q", c.Verdict)
	}
	if !strings.Contains(strings.Join(c.Facts, "|"), "GOLD · COMEX GC=F · 1h: data unavailable right now") {
		t.Errorf("failed asset must be stated: %q", c.Facts)
	}
}

// Digest inputs are untouched by the new presentation: confirmed, Deviation
// and rankAsOf come from the ranked (Binance) reads only, as before.
func TestMomentumCompositeDigestInputsUnchanged(t *testing.T) {
	btcBar := time.Date(2026, 9, 15, 4, 0, 0, 0, time.UTC)
	ethBar := time.Date(2026, 9, 15, 8, 0, 0, 0, time.UTC)
	goldBar := time.Date(2026, 9, 15, 1, 0, 0, 0, time.UTC)

	c := Card{DataTime: momNow}
	composeMomentum(&c, defaultTrio(
		momRead(btcSpec, 75, -3, btcBar),              // conflict: neutral, scores 0
		momRead(assetTable["eth"], 60, 2, ethBar),     // confirmed bullish: 20
		momRead(xauSpec, 90, 5, goldBar)), "", momNow) // gold never ranks
	if !c.confirmed || c.Deviation != 20 || !c.rankAsOf.Equal(btcBar) || !c.DataTime.Equal(goldBar) {
		t.Errorf("confirmed %v dev %d rankAsOf %v dataTime %v", c.confirmed, c.Deviation, c.rankAsOf, c.DataTime)
	}

	c = Card{DataTime: momNow}
	composeMomentum(&c, defaultTrio(
		momRead(btcSpec, 75, -3, btcBar),
		momRead(assetTable["eth"], 50, 2, ethBar),
		momRead(xauSpec, 90, 5, goldBar)), "", momNow)
	if c.confirmed || c.Deviation != 0 {
		t.Errorf("a confirmed gold read must not make the composite confirmed: %v / %d", c.confirmed, c.Deviation)
	}
}

// ── freshness per asset ──────────────────────────────────────────────────────

func TestMomentumFreshness(t *testing.T) {
	sat := time.Date(2026, 9, 19, 12, 0, 0, 0, time.UTC)      // FX weekend window
	sunOpen := time.Date(2026, 9, 20, 21, 30, 0, 0, time.UTC) // just reopened
	friClose := time.Date(2026, 9, 18, 20, 0, 0, 0, time.UTC)
	cases := []struct {
		name         string
		source, tf   string
		closeAt, now time.Time
		want         string
	}{
		{"binance on time", srcBinance, "4h", momNow.Add(-90 * time.Minute), momNow, momentumOnTime},
		// Binance is on time by the source's contract (cache ≤60s, the answer
		// always ends at the forming bar) — never by the clock.
		{"binance old bar is a stub artifact", srcBinance, "4h", momNow.Add(-12 * time.Hour), momNow, momentumOnTime},
		{"binance weekend is 24/7", srcBinance, "1h", sat.Add(-5 * time.Hour), sat, momentumOnTime},
		{"yahoo weekend", srcYahoo, "1h", friClose, sat, momentumMarketClosed},
		{"yahoo just reopened", srcYahoo, "1h", friClose, sunOpen, momentumOnTime},
		{"yahoo midweek stale bar", srcYahoo, "1h", momNow.Add(-5 * time.Hour), momNow, momentumDataDelayed},
		{"yahoo 1d monday morning", srcYahoo, "1d", time.Date(2026, 9, 19, 0, 0, 0, 0, time.UTC), time.Date(2026, 9, 21, 10, 0, 0, 0, time.UTC), momentumOnTime},
	}
	for _, tc := range cases {
		if got := momentumFreshness(tc.source, tc.tf, tc.closeAt, tc.now); got != tc.want {
			t.Errorf("%s: %q, want %q", tc.name, got, tc.want)
		}
	}

	// On the card: "market closed" only from the weekend rule, "data delayed"
	// otherwise.
	gold := momRead(xauSpec, 50, 1, friClose)
	c := Card{DataTime: sat}
	composeMomentum(&c, []momentumAsset{{spec: xauSpec, read: gold, status: statusOK}}, "", sat)
	if !strings.HasSuffix(c.Facts[1], " · last bar Sep 18 20:00 UTC · market closed") {
		t.Errorf("weekend gold line: %q", c.Facts[1])
	}
	eur := assetTable["eurusd"]
	late := momRead(eur, 50, 1, momNow.Add(-5*time.Hour)) // midweek, open market
	c = Card{DataTime: momNow}
	composeMomentum(&c, []momentumAsset{{spec: eur, read: late, status: statusOK}}, "", momNow)
	if !strings.HasSuffix(c.Facts[1], " · data delayed") || strings.Contains(c.Facts[1], "market closed") {
		t.Errorf("late EURUSD line: %q", c.Facts[1])
	}
	single := momentumAssetCardFrom(eur, late, 0, false, momNow)
	if !single.noValidator {
		t.Error("a data-delayed card depends on the clock: it must serve no validator")
	}
	if !strings.Contains(strings.Join(single.Facts, "|"), "Data delayed: the last closed 1h bar (Sep 15 04:30 UTC) is over 2 bars old") {
		t.Errorf("single card delayed line: %q", single.Facts)
	}
	// A Binance card never carries clock-derived wording, so its body stays a
	// function of its bars and keeps the validator whatever the bar's age.
	oldBTC := momentumAssetCardFrom(btcSpec, momRead(btcSpec, 50, 1, momNow.Add(-12*time.Hour)), 0, false, momNow)
	if oldBTC.noValidator || strings.Contains(strings.Join(oldBTC.Facts, "|"), "delayed") {
		t.Errorf("Binance card must not depend on the clock: noValidator %v facts %q", oldBTC.noValidator, oldBTC.Facts)
	}
}

// ── context lines ────────────────────────────────────────────────────────────

func TestMomentumContextLines(t *testing.T) {
	if got := momentumVolumeContext("BTC 4h volume", 0.9612); got != "Context, not part of the reading: BTC 4h volume 0.96× its 20-bar average" {
		t.Errorf("volume: %q", got)
	}
	a, b := -2.4, 5.1
	if got := momentumRSContext(MomentumItem{RS7D: &a, RS30D: &b}); got != "Context, not part of the reading: ETH return minus BTC return incl. today, 7d -2.4 pp · 30d +5.1 pp" {
		t.Errorf("rs: %q", got)
	}
	if got := momentumRSContext(MomentumItem{}); got != "" {
		t.Errorf("no windows → no line: %q", got)
	}
	huge := 123456.7
	if got := momentumRSContext(MomentumItem{RS7D: &huge, RS30D: &huge}); utf8.RuneCountInString(got) > momentumFactMaxRunes {
		t.Errorf("rs line over budget: %q", got)
	}
}

// ── content blocks ───────────────────────────────────────────────────────────

func TestMomentumBlocks(t *testing.T) {
	b := momentumBlocks("BTC", "4h", 54.7, 217.3)
	if b.WhatHappened != "BTC 4h reads not confirmed: RSI below the 55 threshold (RSI 54.7, MACD histogram above 0)." {
		t.Errorf("what_happened: %q", b.WhatHappened)
	}
	if b.WhyLevel != "No price level: 55 and 45 are the RSI(14) thresholds, 0 is the MACD histogram line" {
		t.Errorf("why_level: %q", b.WhyLevel)
	}
	wantScen := []string{
		"If RSI rises to 55 or above and the MACD histogram stays above 0, the reading turns bullish",
		"If RSI falls to 45 or below and the MACD histogram turns below 0, the reading turns bearish",
	}
	if strings.Join(b.Scenarios, "|") != strings.Join(wantScen, "|") {
		t.Errorf("scenarios:\n got %q\nwant %q", b.Scenarios, wantScen)
	}
	if b.Invalidates != nil {
		t.Errorf("unconfirmed reading has nothing to end: %q", *b.Invalidates)
	}
	if b.Regime != "Local momentum · BTC 4h · not confirmed: RSI below the 55 threshold" {
		t.Errorf("regime: %q", b.Regime)
	}

	b = momentumBlocks("ETH", "1d", 61.3, -16.3) // conflict: RSI side first
	wantScen = []string{
		"If RSI holds ≥ 55 and the MACD histogram turns above 0, the reading turns bullish",
		"If RSI falls to 45 or below and the MACD histogram stays below 0, the reading turns bearish",
	}
	if strings.Join(b.Scenarios, "|") != strings.Join(wantScen, "|") {
		t.Errorf("conflict scenarios:\n got %q\nwant %q", b.Scenarios, wantScen)
	}

	b = momentumBlocks("BTC", "4h", 62, 5)
	wantScen = []string{
		"If RSI holds ≥ 55 and the MACD histogram stays above 0, the reading stays bullish",
		"If RSI drops below 55 or the MACD histogram falls to 0 or below, the reading turns neutral",
	}
	if strings.Join(b.Scenarios, "|") != strings.Join(wantScen, "|") {
		t.Errorf("confirmed scenarios:\n got %q\nwant %q", b.Scenarios, wantScen)
	}
	if b.Invalidates == nil || *b.Invalidates != "A closed 4h candle with RSI below 55 or the MACD histogram at or below 0 ends the bullish reading" {
		t.Errorf("invalidates: %v", b.Invalidates)
	}
	b = momentumBlocks("BTC", "4h", 30, -5)
	if b.Invalidates == nil || *b.Invalidates != "A closed 4h candle with RSI above 45 or the MACD histogram at or above 0 ends the bearish reading" {
		t.Errorf("bearish invalidates: %v", b.Invalidates)
	}

	// No price talk: momentum has no level, target or price direction.
	for _, s := range append(append([]string{}, b.Scenarios...), b.WhatHappened, b.WhyLevel, b.Regime) {
		low := strings.ToLower(s)
		for _, bad := range []string{"target", "price will", "buy", "sell", "stop"} {
			if strings.Contains(low, bad) {
				t.Errorf("block %q contains %q", s, bad)
			}
		}
	}
}

// ── JSON ─────────────────────────────────────────────────────────────────────

func TestMomentumResultsJSON(t *testing.T) {
	bar := time.Date(2026, 9, 15, 8, 0, 0, 0, time.UTC)
	c := Card{Agent: "Momentum Agent", DataTime: momNow}
	composeMomentum(&c, []momentumAsset{
		{spec: btcSpec, read: momRead(btcSpec, 54.7, 217.3456789, bar), status: statusOK},
		{spec: assetTable["eth"], status: statusInsufficientHistory},
	}, "", momNow)
	raw, err := json.Marshal(cardEnvelope(c))
	if err != nil {
		t.Fatal(err)
	}
	var env struct {
		Results []map[string]any `json:"results"`
		Blocks  map[string]any   `json:"blocks"`
	}
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatal(err)
	}
	if len(env.Results) != 2 {
		t.Fatalf("results %d, want 2", len(env.Results))
	}
	// The composite card's OWN blocks (2026-09-16): the counter and the scope,
	// never one asset's reading — a per-asset reading lives in results[].blocks
	// and would read as the whole card's verdict up here.
	if env.Blocks == nil {
		t.Fatal("the composite card must carry its own blocks")
	}
	if got := env.Blocks["what_happened"]; got != "0 bullish, 0 bearish, 1 not confirmed, 1 unavailable on closed 4h candles." {
		t.Errorf("blocks.what_happened = %v", got)
	}
	if env.Blocks["limitations"] != "Colour follows the BTC/ETH reads, the ones the digest ranks. "+momentumMACDLimitation {
		t.Errorf("blocks.limitations = %v", env.Blocks["limitations"])
	}
	for _, k := range []string{"why_level", "regime"} {
		if env.Blocks[k] != "" {
			t.Errorf("the composite card has no single %s: %v", k, env.Blocks[k])
		}
	}
	if env.Blocks["scenarios"] != nil || env.Blocks["invalidates"] != nil {
		t.Errorf("the composite card has no idea to run forward or invalidate: %v", env.Blocks)
	}
	btc := env.Results[0]
	for k, want := range map[string]any{
		"asset": "BTC", "ok": true, "timeframe": "4h", "data_as_of": "2026-09-15T08:00:00Z",
		"freshness": "on_time", "verdict": "neutral", "state": "rsi_below_55",
		"why": "RSI below the 55 threshold", "rsi": 54.7, "macd_histogram": 217.3456789,
		"rsi_shown": "54.7",
	} {
		if fmt.Sprint(btc[k]) != fmt.Sprint(want) {
			t.Errorf("results[0].%s = %v, want %v", k, btc[k], want)
		}
	}
	blocks, ok := btc["blocks"].(map[string]any)
	if !ok || blocks["what_happened"] == nil || len(blocks["scenarios"].([]any)) != 2 {
		t.Errorf("results[0].blocks: %v", btc["blocks"])
	}
	eth := env.Results[1]
	for _, k := range []string{"timeframe", "data_as_of", "freshness", "verdict", "state", "why", "rsi", "macd_histogram", "blocks"} {
		if _, has := eth[k]; has {
			t.Errorf("a degraded entry carries no reading, got %s = %v", k, eth[k])
		}
	}

	// A zero histogram is a value, not an absence.
	c = Card{DataTime: momNow}
	composeMomentum(&c, []momentumAsset{{spec: btcSpec, read: momRead(btcSpec, 50, 0, bar), status: statusOK}}, "", momNow)
	raw, _ = json.Marshal(cardEnvelope(c))
	if !strings.Contains(string(raw), `"macd_histogram":0`) {
		t.Errorf("zero histogram must serialize: %s", raw)
	}

	// Single-asset card: top-level blocks plus one results entry.
	single := momentumAssetCardFrom(btcSpec, momRead(btcSpec, 62, 5, bar), 0, false, momNow)
	raw, _ = json.Marshal(cardEnvelope(single))
	if !strings.Contains(string(raw), `"blocks":{"what_happened":"BTC 4h reads bullish`) || !strings.Contains(string(raw), `"results":[{"asset":"BTC","ok":true`) {
		t.Errorf("single-asset envelope: %s", raw)
	}
}

// Non-finite indicator output degrades instead of reaching the JSON encoder.
func TestMomentumReadRejectsNonFinite(t *testing.T) {
	candles := srCardCycleCandles(250, func(int) float64 { return 100 })
	candles[len(candles)-1].Close = math.Inf(1)
	if _, err := momentumReadFromCandles(btcSpec, candles); err == nil {
		t.Error("an Inf close must degrade to insufficient history")
	}
}

// ── the 110-character budget on every path ──────────────────────────────────

func TestMomentumTextFitsOneLine(t *testing.T) {
	check := func(where, s string) {
		if n := utf8.RuneCountInString(s); n > momentumFactMaxRunes {
			t.Errorf("%s: %d chars (max %d): %q", where, n, momentumFactMaxRunes, s)
		}
	}
	checkCard := func(where string, c Card) {
		check(where+" verdict", c.Verdict)
		for _, f := range c.Facts {
			check(where+" fact", f)
		}
		blocks := []*ContentBlocks{c.Blocks}
		for _, r := range c.Results {
			blocks = append(blocks, r.Blocks)
			check(where+" why", r.Why)
		}
		for _, b := range blocks {
			if b == nil {
				continue
			}
			for _, s := range append([]string{b.WhatHappened, b.WhyLevel, b.Regime}, b.Scenarios...) {
				check(where+" block", s)
			}
			if b.Invalidates != nil {
				check(where+" block", *b.Invalidates)
			}
		}
	}
	rsis := []float64{0, 44.96, 45, 45.04, 50, 54.96, 55, 55.04, 100}
	hists := []float64{-1e12, -1e-12, 0, 1e-12, 1e12}
	nows := []time.Time{momNow, time.Date(2026, 9, 19, 12, 0, 0, 0, time.UTC)}
	keys := []string{"btc", "eth", "eurusd", "gbpusd", "usdjpy", "xauusd"}
	for _, tf := range []string{"1h", "4h", "1d"} {
		for _, now := range nows {
			for _, rsi := range rsis {
				for _, h := range hists {
					var assets []momentumAsset
					for _, k := range keys {
						spec := assetTable[k]
						spec.Interval = tf
						r := momRead(spec, rsi, h, now.Add(-72*time.Hour)) // late: longest suffix
						assets = append(assets, momentumAsset{spec: spec, read: r, status: statusOK})
						checkCard(fmt.Sprintf("single %s %s", k, tf), momentumAssetCardFrom(spec, r, 123456.78, true, now))
					}
					c := Card{DataTime: now}
					composeMomentum(&c, assets, tf, now)
					checkCard("scan "+tf, c)
					c = Card{DataTime: now}
					composeMomentum(&c, assets, "", now)
					checkCard("scan native", c)
				}
			}
		}
	}
	// Every failure shape of a scan, too.
	var failed []momentumAsset
	for _, k := range keys {
		failed = append(failed, momentumAsset{spec: assetTable[k], status: statusInsufficientHistory})
	}
	failed[0].status = statusOK
	failed[0].read = momRead(btcSpec, 50, 1, momNow)
	c := Card{DataTime: momNow}
	composeMomentum(&c, failed, "", momNow)
	checkCard("failures", c)
}
