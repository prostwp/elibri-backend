package demobot

// funding_readable_test.go — Funding stage 1 (2026-09-15): the coin is picked
// against the threshold of ITS OWN side, named in the header, printed beside
// both thresholds; coverage, an honest empty liquidation window, no forecast
// words, no "/8h", event times instead of ages, content blocks, machine
// fields, and the 110-character line budget on every path. The thresholds
// (+0.03% / -0.01%) are unchanged.

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// fundingStub serves premiumIndex per symbol (a symbol absent from rates
// answers 500) and, when liqBody is not "", the liquidation route.
func fundingStub(t *testing.T, rates, marks map[string]string, liqBody string) *Agents {
	t.Helper()
	stubExternalBases(t)
	prem := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sym := r.URL.Query().Get("symbol")
		rate, ok := rates[sym]
		if !ok {
			http.Error(w, "down", http.StatusInternalServerError)
			return
		}
		body := `{"symbol":"` + sym + `","lastFundingRate":"` + rate + `"`
		if m, ok := marks[sym]; ok {
			body += `,"markPrice":"` + m + `"`
		}
		_, _ = w.Write([]byte(body + `}`))
	}))
	t.Cleanup(prem.Close)
	premiumIndexURL = prem.URL + "/?symbol="
	routes := map[string]string{}
	if liqBody != "" {
		routes["/api/v1/funding/liquidations"] = liqBody
	}
	return newStubBackend(t, routes)
}

const emptyLiq = `{"captured_at":"2026-09-15T16:40:00Z","feed":[],"zones":[]}`

// liveRates1640 is the production snapshot of 2026-09-15 16:40 UTC.
var liveRates1640 = map[string]string{"BTCUSDT": "0.00009700", "ETHUSDT": "-0.00004900",
	"SOLUSDT": "-0.00006200", "BNBUSDT": "0.00003800", "XRPUSDT": "-0.00003100"}

func fundingEnv(t *testing.T, c Card) map[string]any {
	t.Helper()
	b, err := json.Marshal(cardEnvelope(c))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func wantFacts(t *testing.T, got, want []string) {
	t.Helper()
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Errorf("facts:\ngot:\n%s\nwant:\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
}

// ── golden: the live 16:40 snapshot picks SOLUSDT, not BTCUSDT ───────────────

func TestFundingGoldenLiveSnapshot(t *testing.T) {
	c := fundingStub(t, liveRates1640, nil, emptyLiq).FundingCard(context.Background())
	if c.Asset != "SOLUSDT" || c.Emoji != emojiNeutral {
		t.Fatalf("asset %q emoji %q, want SOLUSDT neutral (0.62 of its threshold beats BTC 0.32)", c.Asset, c.Emoji)
	}
	if c.Verdict != "Funding within the agent's thresholds" {
		t.Errorf("verdict %q", c.Verdict)
	}
	wantFacts(t, c.Facts, []string{
		"Last funding rate: -0.0100% < -0.0062% < +0.0300% → within thresholds",
		"Nearest threshold: short -0.0100%, 0.0038 pp away",
		"Why SOLUSDT: largest ratio to its own side's threshold of 5 majors (0.62×, rate ÷ threshold)",
		"Other majors: BTCUSDT +0.0097% · ETHUSDT -0.0049% · BNBUSDT +0.0038% · XRPUSDT -0.0031%",
		"Coverage: 5/5 Binance majors",
		"Liquidations: no events in the current 1h window",
	})
	if !strings.HasPrefix(c.RenderHTML(), "⚪ <b>Funding Agent</b> · SOLUSDT\n") {
		t.Errorf("header must name the coin:\n%s", c.RenderHTML())
	}
	if got := htmlToPlain(c.OneLiner()); got != "⚪ Funding SOLUSDT: within thresholds · -0.0062%" {
		t.Errorf("digest one-liner %q", got)
	}
	if c.Deviation != fundingDeviation(-0.000062) {
		t.Errorf("digest score %d must be computed on the shown coin (%d)", c.Deviation, fundingDeviation(-0.000062))
	}
	if env := fundingEnv(t, c); env["asset"] != "SOL" {
		t.Errorf("envelope asset %v", env["asset"])
	}
}

// ── golden: the plan's bug example — XRP past its threshold is not hidden ────

func TestFundingGoldenPlanBugExample(t *testing.T) {
	rates := map[string]string{"BTCUSDT": "0.00020000", "ETHUSDT": "0.00001000", "SOLUSDT": "0.00001000",
		"BNBUSDT": "0.00001000", "XRPUSDT": "-0.00015000"}
	c := fundingStub(t, rates, nil, emptyLiq).FundingCard(context.Background())
	if c.Asset != "XRPUSDT" || c.Emoji != emojiBull || !c.confirmed {
		t.Fatalf("asset %q emoji %q confirmed %v: XRP -0.015%% is past -0.01%%, BTC +0.020%% is not past +0.03%%",
			c.Asset, c.Emoji, c.confirmed)
	}
	if c.Verdict != "Negative funding below threshold — shorts pay an elevated rate" {
		t.Errorf("verdict %q", c.Verdict)
	}
	wantFacts(t, c.Facts, []string{
		"Last funding rate: -0.0150% ≤ -0.0100% (short threshold) → shorts pay an elevated rate",
		"Past the short threshold by 0.0050 pp · long threshold +0.0300%",
		"Why XRPUSDT: furthest past its own side's threshold (1.50×, rate ÷ threshold) · 1 of 5 past a threshold",
		"Other majors: BTCUSDT +0.0200% · ETHUSDT +0.0010% · SOLUSDT +0.0010% · BNBUSDT +0.0010%",
		"Coverage: 5/5 Binance majors",
		"Liquidations: no events in the current 1h window",
	})
	if c.Deviation != fundingDeviation(-0.00015) {
		t.Errorf("digest score %d, want %d (the shown coin)", c.Deviation, fundingDeviation(-0.00015))
	}
}

func TestFundingGoldenPositiveCrossed(t *testing.T) {
	rates := map[string]string{"BTCUSDT": "0.00041000", "ETHUSDT": "0.00035000", "SOLUSDT": "-0.00012000",
		"BNBUSDT": "0.00001000", "XRPUSDT": "0.00001000"}
	c := fundingStub(t, rates, nil, emptyLiq).FundingCard(context.Background())
	// BTC 1.36×, ETH 1.16×, SOL 1.20× → BTC.
	if c.Asset != "BTCUSDT" || c.Emoji != emojiBear || !c.confirmed {
		t.Fatalf("asset %q emoji %q confirmed %v", c.Asset, c.Emoji, c.confirmed)
	}
	if c.Verdict != "Positive funding above threshold — longs pay an elevated rate" {
		t.Errorf("verdict %q", c.Verdict)
	}
	wantFacts(t, c.Facts[:3], []string{
		"Last funding rate: +0.0410% ≥ +0.0300% (long threshold) → longs pay an elevated rate",
		"Past the long threshold by 0.0110 pp · short threshold -0.0100%",
		"Why BTCUSDT: furthest past its own side's threshold (1.36×, rate ÷ threshold) · 3 of 5 past a threshold",
	})
}

// ── selection: ties resolve in fundingSymbols order ──────────────────────────

func TestFundingSelectionTies(t *testing.T) {
	zero := "0.00000000"
	cases := []struct {
		name  string
		rates map[string]string
		sym   string
		emoji string
	}{
		{"five-way tie at +0.0100%", equalFundingRates, "BTCUSDT", emojiNeutral},
		{"all zero", map[string]string{"BTCUSDT": zero, "ETHUSDT": zero, "SOLUSDT": zero, "BNBUSDT": zero, "XRPUSDT": zero},
			"BTCUSDT", emojiNeutral},
		{"equal ratio 0.10 of opposite sides, ETH first", map[string]string{"BTCUSDT": zero, "ETHUSDT": "0.00003000",
			"SOLUSDT": "-0.00001000", "BNBUSDT": zero, "XRPUSDT": zero}, "ETHUSDT", emojiNeutral},
		{"equal ratio 0.10 of opposite sides, SOL negative first", map[string]string{"BTCUSDT": zero, "ETHUSDT": zero,
			"SOLUSDT": "-0.00001000", "BNBUSDT": "0.00003000", "XRPUSDT": zero}, "SOLUSDT", emojiNeutral},
		{"both exactly on their thresholds (1.00×), BTC positive first", map[string]string{"BTCUSDT": "0.00030000",
			"ETHUSDT": zero, "SOLUSDT": zero, "BNBUSDT": zero, "XRPUSDT": "-0.00010000"}, "BTCUSDT", emojiBear},
		{"both exactly on their thresholds (1.00×), ETH negative first", map[string]string{"BTCUSDT": zero,
			"ETHUSDT": "-0.00010000", "SOLUSDT": zero, "BNBUSDT": "0.00030000", "XRPUSDT": zero}, "ETHUSDT", emojiBull},
		{"equal |rate| 0.03%: the negative one is 3× its threshold", map[string]string{"BTCUSDT": "0.00001000",
			"ETHUSDT": "0.00030000", "SOLUSDT": "-0.00030000", "BNBUSDT": "0.00001000", "XRPUSDT": "0.00001000"},
			"SOLUSDT", emojiBull},
		{"crossed beats a larger |rate| that is not", map[string]string{"BTCUSDT": "0.00029999", "ETHUSDT": zero,
			"SOLUSDT": zero, "BNBUSDT": zero, "XRPUSDT": "-0.00010000"}, "XRPUSDT", emojiBull},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ag := fundingStub(t, tc.rates, nil, emptyLiq)
			for i := 0; i < 10; i++ {
				c := ag.FundingCard(context.Background())
				if c.Asset != tc.sym || c.Emoji != tc.emoji {
					t.Fatalf("read %d: %s %s, want %s %s", i, c.Asset, c.Emoji, tc.sym, tc.emoji)
				}
			}
		})
	}
}

// ── boundaries: exactly +0.03%, exactly -0.01%, 0, just inside ──────────────

func TestFundingBoundaries(t *testing.T) {
	zero := "0.00000000"
	one := func(btc string) map[string]string {
		return map[string]string{"BTCUSDT": btc, "ETHUSDT": zero, "SOLUSDT": zero, "BNBUSDT": zero, "XRPUSDT": zero}
	}
	cases := []struct {
		btc, emoji, check, dist string
	}{
		{"0.00030000", emojiBear, "Last funding rate: +0.0300% ≥ +0.0300% (long threshold) → longs pay an elevated rate",
			"Exactly at the long threshold +0.0300% · short threshold -0.0100%"},
		{"-0.00010000", emojiBull, "Last funding rate: -0.0100% ≤ -0.0100% (short threshold) → shorts pay an elevated rate",
			"Exactly at the short threshold -0.0100% · long threshold +0.0300%"},
		{zero, emojiNeutral, "Last funding rate: -0.0100% < 0.0000% < +0.0300% → within thresholds",
			"Nearest threshold: short -0.0100%, 0.0100 pp away"},
		{"0.00010000", emojiNeutral, "Last funding rate: -0.0100% < +0.0100% < +0.0300% → within thresholds",
			"Nearest threshold: equidistant, 0.0200 pp from both"},
		{"0.00029999", emojiNeutral, "Last funding rate: -0.0100% < +0.0299% < +0.0300% → within thresholds",
			"Nearest threshold: long +0.0300%, 0.0001 pp away"},
		{"-0.00009999", emojiNeutral, "Last funding rate: -0.0100% < -0.0099% < +0.0300% → within thresholds",
			"Nearest threshold: short -0.0100%, 0.0001 pp away"},
		{"0.00030001", emojiBear, "Last funding rate: +0.0300% ≥ +0.0300% (long threshold) → longs pay an elevated rate",
			"Past the long threshold by <0.0001 pp · short threshold -0.0100%"},
		{"-0.00010001", emojiBull, "Last funding rate: -0.0100% ≤ -0.0100% (short threshold) → shorts pay an elevated rate",
			"Past the short threshold by <0.0001 pp · long threshold +0.0300%"},
	}
	for _, tc := range cases {
		t.Run(tc.btc, func(t *testing.T) {
			c := fundingStub(t, one(tc.btc), nil, emptyLiq).FundingCard(context.Background())
			if c.Asset != "BTCUSDT" || c.Emoji != tc.emoji {
				t.Fatalf("asset %q emoji %q, want BTCUSDT %s", c.Asset, c.Emoji, tc.emoji)
			}
			if c.Facts[0] != tc.check || c.Facts[1] != tc.dist {
				t.Errorf("got\n%s\n%s\nwant\n%s\n%s", c.Facts[0], c.Facts[1], tc.check, tc.dist)
			}
		})
	}
}

// ── coverage ────────────────────────────────────────────────────────────────

func TestFundingCoverageMissingOne(t *testing.T) {
	rates := map[string]string{}
	for k, v := range liveRates1640 {
		if k != "BNBUSDT" {
			rates[k] = v
		}
	}
	c := fundingStub(t, rates, nil, emptyLiq).FundingCard(context.Background())
	if c.Asset != "SOLUSDT" || c.effectiveStatus() != statusOK {
		t.Fatalf("asset %q status %v", c.Asset, c.effectiveStatus())
	}
	joined := strings.Join(c.Facts, "\n")
	for _, want := range []string{
		"Coverage: 4/5 Binance majors · missing BNBUSDT",
		"Why SOLUSDT: largest ratio to its own side's threshold of 4 majors (0.62×, rate ÷ threshold)",
		"Other majors: BTCUSDT +0.0097% · ETHUSDT -0.0049% · XRPUSDT -0.0031%",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in\n%s", want, joined)
		}
	}
}

// Fewer than 3 of 5 majors → partial: no verdict, no coin, ok=false.
func TestFundingCoveragePartial(t *testing.T) {
	rates := map[string]string{"BTCUSDT": "0.00041000", "ETHUSDT": "-0.00004900"}
	c := fundingStub(t, rates, nil, emptyLiq).FundingCard(context.Background())
	if c.Asset != "" || c.Emoji != emojiNeutral || c.confirmed || c.Deviation != 0 {
		t.Fatalf("partial must be neutral, unconfirmed, unscored and name no coin: %q %q %v %d",
			c.Asset, c.Emoji, c.confirmed, c.Deviation)
	}
	if c.effectiveStatus() != statusSourceOffline {
		t.Errorf("partial status %v, want source_offline", c.effectiveStatus())
	}
	if c.Verdict != "Partial funding data — 2/5 Binance majors answered, no verdict" || c.Short != "partial data (2/5)" {
		t.Errorf("verdict %q short %q", c.Verdict, c.Short)
	}
	wantFacts(t, c.Facts, []string{
		"Rates received: BTCUSDT +0.0410% (past the long threshold) · ETHUSDT -0.0049%",
		"Thresholds: long +0.0300%, short -0.0100% · fewer than 3 of 5 majors, no comparison",
		"Coverage: 2/5 Binance majors · missing SOLUSDT, BNBUSDT, XRPUSDT",
		"Liquidations: no events in the current 1h window",
	})
}

// ── liquidations ────────────────────────────────────────────────────────────

func TestFundingLiquidationsEmptyWindow(t *testing.T) {
	old := time.Now().UTC().Add(-2 * time.Hour).Format(time.RFC3339)
	body := `{"captured_at":"x","feed":[{"symbol":"BTCUSDT","side":"long_liq","qty":1,"price":1,"usd_value":5000,"ts":"` + old + `"}],"zones":[]}`
	for _, liq := range []string{emptyLiq, body} {
		c := fundingStub(t, liveRates1640, nil, liq).FundingCard(context.Background())
		if last := c.Facts[len(c.Facts)-1]; last != "Liquidations: no events in the current 1h window" {
			t.Errorf("empty window: %q", last)
		}
		if strings.Contains(strings.Join(c.Facts, "|"), "quiet") || strings.Contains(strings.Join(c.Facts, "|"), "recently") {
			t.Errorf("empty window must not claim a quiet feed: %v", c.Facts)
		}
	}
}

func TestFundingLiquidationsAndCluster(t *testing.T) {
	now := time.Now().UTC()
	t1, t2 := now.Add(-40*time.Minute).Truncate(time.Second), now.Add(-12*time.Minute).Truncate(time.Second)
	body := `{"captured_at":"x","feed":[
	  {"symbol":"BTCUSDT","side":"long_liq","qty":1,"price":63300,"usd_value":90000,"ts":"` + t2.Format(time.RFC3339) + `"},
	  {"symbol":"ETHUSDT","side":"short_liq","qty":10,"price":3400,"usd_value":30000,"ts":"` + t2.Format(time.RFC3339) + `"},
	  {"symbol":"BTCUSDT","side":"long_liq","qty":1,"price":63150,"usd_value":60000,"ts":"` + t1.Format(time.RFC3339) + `"}],
	  "zones":[
	    {"symbol":"BTCUSDT","price_band":"70000-70350","total_usd":900000,"count":20,"side":"short_liq"},
	    {"symbol":"BTCUSDT","price_band":"63100-63400","total_usd":412000,"count":9,"side":"long_liq"}]}`
	marks := map[string]string{"BTCUSDT": "63900.00"}
	c := fundingStub(t, liveRates1640, marks, body).FundingCard(context.Background())
	n := len(c.Facts)
	// The largest BTC band by USD, not the one nearest the mark (63100-63400):
	// the pick uses no price. The served page holds none of its events, so no
	// "last event" time; the mark only places it (band above).
	wantFacts(t, c.Facts[n-3:], []string{
		"Liquidations, last 1h: 3 events · $180.0K · long liqs $150.0K (83%) · short liqs $30.0K",
		"Observed liquidation cluster, last 1h: BTCUSDT 70000-70350 · $900.0K · 20 events",
		"Cluster: short liqs lead by USD · band above the BTCUSDT mark price",
	})
	for _, f := range c.Facts {
		for _, bad := range []string{"Magnet", "magnet", " ago", "pain"} {
			if strings.Contains(f, bad) {
				t.Errorf("fact %q carries %q", f, bad)
			}
		}
	}
}

// ── wording ─────────────────────────────────────────────────────────────────

func TestFundingNoForecastWords(t *testing.T) {
	sets := []map[string]string{liveRates1640,
		{"BTCUSDT": "0.00041000", "ETHUSDT": "0", "SOLUSDT": "0", "BNBUSDT": "0", "XRPUSDT": "0"},
		{"BTCUSDT": "-0.00041000", "ETHUSDT": "0", "SOLUSDT": "0", "BNBUSDT": "0", "XRPUSDT": "0"}}
	for _, rates := range sets {
		c := fundingStub(t, rates, nil, emptyLiq).FundingCard(context.Background())
		all := strings.Join(append([]string{c.Verdict, c.Short, c.HowItWorks}, c.Facts...), "\n")
		for _, bad := range []string{"squeeze", "crowd", "punish", "Magnet", "magnet", "fuel", "/8h", "balanced"} {
			if strings.Contains(all, bad) {
				t.Errorf("%q in:\n%s", bad, all)
			}
		}
	}
}

// ── pure builder ─────────────────────────────────────────────────────────────

var fundAt = time.Date(2026, 9, 15, 16, 40, 0, 0, time.UTC)

func quotesOf(rates map[string]float64) map[string]fundingQuote {
	out := map[string]fundingQuote{}
	for k, v := range rates {
		out[k] = fundingQuote{rate: v}
	}
	return out
}

var liveFloat1640 = map[string]float64{"BTCUSDT": 0.000097, "ETHUSDT": -0.000049, "SOLUSDT": -0.000062,
	"BNBUSDT": 0.000038, "XRPUSDT": -0.000031}

var planBugFloat = map[string]float64{"BTCUSDT": 0.0002, "ETHUSDT": 0.00001, "SOLUSDT": 0.00001,
	"BNBUSDT": 0.00001, "XRPUSDT": -0.00015}

func liqOf(events ...Liquidation) *FundingResp { return &FundingResp{Feed: events, Zones: []LiqZone{}} }

func TestFundingRuleUnchanged(t *testing.T) {
	if fundingLongsCrowded != 0.0003 || fundingShortsCrowded != -0.0001 {
		t.Fatalf("thresholds moved: %v / %v", fundingLongsCrowded, fundingShortsCrowded)
	}
}

// A printed rate never sits on the other side of a threshold from the raw one.
func TestFundingUnitsNeverCrossThreshold(t *testing.T) {
	var vals []float64
	for k := int64(-40000); k <= 40000; k++ {
		vals = append(vals, float64(k)/1e8)
	}
	for _, th := range []float64{fundingLongsCrowded, fundingShortsCrowded, 0} {
		for d := -3e-9; d <= 3e-9; d += 1e-10 {
			vals = append(vals, th+d)
		}
	}
	for _, v := range vals {
		u := fundingUnits(v)
		if (v >= fundingLongsCrowded) != (u >= fundingLongU) || (v <= fundingShortsCrowded) != (u <= fundingShortU) {
			t.Fatalf("rate %v prints %s: crosses a threshold the raw rate does not (or vice versa)", v, fundingPct(u))
		}
	}
}

func TestFundingBlocksGolden(t *testing.T) {
	c := fundingCardFrom(quotesOf(liveFloat1640), nil, liqOf(), nil, fundAt)
	b := c.Blocks
	if b == nil {
		t.Fatal("no blocks on a live card")
	}
	if b.WhatHappened != "Within thresholds, largest ratio SOLUSDT -0.0062% vs -0.0100% · no liquidation events in the 1h window" ||
		b.WhyLevel != "No price level: +0.0300% and -0.0100% are the agent's own funding thresholds, not a market benchmark" ||
		b.Invalidates != nil ||
		b.Regime != "Local perp funding regime · 5/5 Binance majors · none past a threshold · SOLUSDT 0.62× its threshold" ||
		b.Limitations != fundingLimitations {
		t.Errorf("neutral blocks: %+v", *b)
	}
	wantFacts(t, b.Scenarios, []string{
		"If any major's funding rate reaches +0.0300% or above, the state turns positive funding above threshold",
		"If any major's funding rate falls to -0.0100% or below, the state turns negative funding below threshold",
	})

	ev := func(side string, usdV float64) Liquidation {
		return Liquidation{Symbol: "BTCUSDT", Side: side, Price: 1, USDValue: usdV, TS: fundAt.Add(-5 * time.Minute)}
	}
	c = fundingCardFrom(quotesOf(planBugFloat), nil, liqOf(ev("long_liq", 90000), ev("short_liq", 30000)), nil, fundAt)
	b = c.Blocks
	if b.WhatHappened != "XRPUSDT funding -0.0150% is past the short threshold -0.0100% · liquidations 1h: 2 events, $120.0K" ||
		b.Invalidates == nil ||
		*b.Invalidates != "XRPUSDT funding back above -0.0100% ends this reading (neutral once no major is past a threshold)" ||
		b.Regime != "Local perp funding regime · 5/5 Binance majors · 1 past a threshold · XRPUSDT 1.50× its threshold" {
		t.Errorf("crossed blocks: %+v", *b)
	}
	wantFacts(t, b.Scenarios, []string{
		"If every major returns inside -0.0100% to +0.0300%, the state turns within thresholds",
		"If a major is past +0.0300% at a larger ratio than XRPUSDT, the state turns positive funding above threshold",
	})
	for _, f := range append(append([]string{b.WhatHappened, b.WhyLevel, b.Regime}, b.Scenarios...), *b.Invalidates) {
		for _, bad := range []string{"squeeze", "crowd", "target", "buy", "sell", "$6", "price will"} {
			if strings.Contains(f, bad) {
				t.Errorf("block %q carries %q", f, bad)
			}
		}
	}
	// Degraded paths carry no blocks.
	for _, dc := range []Card{
		fundingCardFrom(quotesOf(map[string]float64{"BTCUSDT": 0.0001}), nil, liqOf(), nil, fundAt),
		fundingCardFrom(nil, errFakeFunding, liqOf(), nil, fundAt),
	} {
		if dc.Blocks != nil {
			t.Errorf("degraded card %q carries blocks", dc.Verdict)
		}
	}
}

var errFakeFunding = &fakeErr{"premiumIndex down"}

type fakeErr struct{ s string }

func (e *fakeErr) Error() string { return e.s }

func TestFundingMachineFields(t *testing.T) {
	q := quotesOf(liveFloat1640)
	delete(q, "BNBUSDT")
	env := fundingEnv(t, fundingCardFrom(q, nil, liqOf(), nil, fundAt))
	f, ok := env["funding"].(map[string]any)
	if !ok {
		t.Fatalf("no funding object: %v", env)
	}
	if f["state"] != "within_thresholds" || f["selected_symbol"] != "SOLUSDT" || f["selection_reason"] != "closest_to_own_threshold" ||
		f["rate_kind"] != "last_funding_rate" || f["crossed"] != 0.0 {
		t.Errorf("funding: %v", f)
	}
	if v, present := f["funding_interval"]; !present || v != nil {
		t.Errorf("funding_interval must be present and null (not verified): %v", v)
	}
	th := f["thresholds"].(map[string]any)
	if th["long"] != 0.0003 || th["short"] != -0.0001 {
		t.Errorf("thresholds %v", th)
	}
	cov := f["coverage"].(map[string]any)
	if cov["received"] != 4.0 || cov["total"] != 5.0 || cov["min_for_verdict"] != 3.0 ||
		len(cov["missing"].([]any)) != 1 || cov["missing"].([]any)[0] != "BNBUSDT" {
		t.Errorf("coverage %v", cov)
	}
	liq := f["liquidations"].(map[string]any)
	if liq["window_minutes"] != 60.0 || liq["count"] != 0.0 || liq["usd"] != 0.0 || liq["feed_limit"] != 200.0 ||
		liq["capped"] != false || liq["cluster"] != nil {
		t.Errorf("liquidations %v", liq)
	}
	res := env["results"].([]any)
	if len(res) != 5 {
		t.Fatalf("results: %d entries", len(res))
	}
	bnb, sol := res[3].(map[string]any), res[2].(map[string]any)
	if bnb["symbol"] != "BNBUSDT" || bnb["ok"] != false || bnb["reason"] != "source_offline" || bnb["rate"] != nil {
		t.Errorf("missing symbol entry %v", bnb)
	}
	if sol["symbol"] != "SOLUSDT" || sol["asset"] != "SOL" || sol["ok"] != true || sol["rate"] != -0.000062 ||
		sol["side_threshold"] != -0.0001 || sol["crossed"] != false || sol["selected"] != true {
		t.Errorf("selected entry %v", sol)
	}
	if r := sol["ratio_to_threshold"].(float64); r < 0.6199 || r > 0.6201 {
		t.Errorf("ratio_to_threshold %v", r)
	}
	if btc := res[0].(map[string]any); btc["selected"] != false || btc["side_threshold"] != 0.0003 {
		t.Errorf("btc entry %v", btc)
	}
	// Partial: no coin, every received entry unselected; offline rates: 5 missing entries.
	pf := fundingEnv(t, fundingCardFrom(quotesOf(map[string]float64{"BTCUSDT": 0.0001}), nil, liqOf(), nil, fundAt))
	if pf["funding"].(map[string]any)["state"] != "partial" || pf["funding"].(map[string]any)["selected_symbol"] != nil ||
		pf["asset"] != "" || pf["ok"] != false {
		t.Errorf("partial envelope: %v", pf["funding"])
	}
	of := fundingEnv(t, fundingCardFrom(nil, errFakeFunding, liqOf(), nil, fundAt))
	if of["funding"].(map[string]any)["state"] != "rates_offline" || len(of["results"].([]any)) != 5 {
		t.Errorf("rates-offline envelope: %v", of["funding"])
	}
	// Feed offline → liquidations null.
	nf := fundingEnv(t, fundingCardFrom(quotesOf(liveFloat1640), nil, nil, errFakeFunding, fundAt))
	if v, present := nf["funding"].(map[string]any)["liquidations"]; !present || v != nil {
		t.Errorf("feed offline: liquidations must be null, got %v", v)
	}
}

// A liquidation priced exactly on a shared edge belongs to the upper band (the
// backend bands by floor((price − anchor) ÷ width)): it must not become the
// lower band's last event.
func TestFundingClusterEdgeEventGoesUp(t *testing.T) {
	liq := &FundingResp{
		Feed: []Liquidation{
			{Symbol: "BTCUSDT", Side: "long_liq", Price: 63400, USDValue: 5000, TS: fundAt.Add(-2 * time.Minute)},
			{Symbol: "BTCUSDT", Side: "long_liq", Price: 63300, USDValue: 90000, TS: fundAt.Add(-12 * time.Minute)},
		},
		Zones: []LiqZone{
			{Symbol: "BTCUSDT", PriceBand: "63100-63400", TotalUSD: 412000, Count: 9, Side: "long_liq"},
			{Symbol: "BTCUSDT", PriceBand: "63400-63700", TotalUSD: 5000, Count: 1, Side: "long_liq"},
		},
	}
	c := fundingCardFrom(quotesOf(liveFloat1640), nil, liq, nil, fundAt)
	cl := c.Funding.Liquidations.Cluster
	if cl == nil || cl.PriceBand != "63100-63400" || cl.LastEventAt == nil || *cl.LastEventAt != "2026-09-15T16:28:00Z" {
		t.Errorf("edge event timed to the lower band: %+v (last %v)", cl, cl.LastEventAt)
	}
}

// The cluster says WHEN its last event happened, never how long ago: the same
// data read 30 s or 50 min later gives the same facts (the push hook hashes
// them), and the AI payload drops the time.
func TestFundingClusterEventTimeNotAge(t *testing.T) {
	liq := &FundingResp{
		Feed: []Liquidation{
			{Symbol: "BTCUSDT", Side: "long_liq", Price: 63300, USDValue: 90000, TS: fundAt.Add(-12 * time.Minute)},
			{Symbol: "BTCUSDT", Side: "long_liq", Price: 63150, USDValue: 60000, TS: fundAt.Add(-40 * time.Minute)},
		},
		Zones: []LiqZone{{Symbol: "BTCUSDT", PriceBand: "63100-63400", TotalUSD: 412000, Count: 9, Side: "long_liq"}},
	}
	q := quotesOf(liveFloat1640)
	q["BTCUSDT"] = fundingQuote{rate: 0.000097, mark: 63250}
	a := fundingCardFrom(q, nil, liq, nil, fundAt)
	b := fundingCardFrom(q, nil, liq, nil, fundAt.Add(30*time.Second))
	if strings.Join(a.Facts, "\n") != strings.Join(b.Facts, "\n") {
		t.Errorf("facts moved with the clock:\n%v\n%v", a.Facts, b.Facts)
	}
	last := a.Facts[len(a.Facts)-1]
	if last != "Cluster: long liqs lead by USD · last event 16:28 UTC · the BTCUSDT mark price is inside the band" {
		t.Errorf("cluster line %q", last)
	}
	cl := a.Funding.Liquidations.Cluster
	if cl == nil || cl.LastEventAt == nil || *cl.LastEventAt != "2026-09-15T16:28:00Z" || cl.BandVsMark == nil || *cl.BandVsMark != "inside" {
		t.Errorf("cluster readout %+v", cl)
	}
	for _, f := range fundingAIFacts(a.Facts) {
		if strings.Contains(f, "16:28") {
			t.Errorf("AI fact keeps the event time: %q", f)
		}
	}
	// Position words only (no distance): band below / above / mark inside.
	for mark, want := range map[float64]string{63900: "band below the BTCUSDT", 62000: "band above the BTCUSDT",
		63260: "mark price is inside the band", 63400: "mark price is inside the band", 63400.01: "band below the BTCUSDT",
		63099.99: "band above the BTCUSDT"} {
		q["BTCUSDT"] = fundingQuote{rate: 0.000097, mark: mark}
		c := fundingCardFrom(q, nil, liq, nil, fundAt)
		if !strings.Contains(c.Facts[len(c.Facts)-1], want) {
			t.Errorf("mark %v: %q, want %q", mark, c.Facts[len(c.Facts)-1], want)
		}
	}
}

// The digest score is computed on the coin the card shows, and that coin
// carries the largest score of the received majors.
func TestFundingDigestScoreOnShownCoin(t *testing.T) {
	sets := []map[string]float64{liveFloat1640, planBugFloat,
		{"BTCUSDT": 0.00041, "ETHUSDT": 0.00035, "SOLUSDT": -0.00012, "BNBUSDT": 0, "XRPUSDT": 0},
		{"BTCUSDT": 0.00029, "ETHUSDT": -0.00009, "SOLUSDT": 0, "BNBUSDT": 0, "XRPUSDT": 0}}
	for _, rates := range sets {
		c := fundingCardFrom(quotesOf(rates), nil, liqOf(), nil, fundAt)
		if c.Deviation != fundingDeviation(rates[c.Asset]) {
			t.Errorf("%s: Deviation %d, want fundingDeviation of the shown coin %d", c.Asset, c.Deviation, fundingDeviation(rates[c.Asset]))
		}
		for sym, r := range rates {
			if fundingDeviation(r) > c.Deviation {
				t.Errorf("%v: %s scores %d above the shown %s %d", rates, sym, fundingDeviation(r), c.Asset, c.Deviation)
			}
		}
	}
}

func TestFundingHowText(t *testing.T) {
	h := howTexts[keyFunding]
	if utf8.RuneCountInString(h) > 200 {
		t.Errorf("how text %d runes", utf8.RuneCountInString(h))
	}
	for _, bad := range []string{"squeeze", "crowd", "8h", "magnet"} {
		if strings.Contains(h, bad) {
			t.Errorf("how text carries %q: %q", bad, h)
		}
	}
}

// Every line fits 110 runes on every path: live, crossed both ways, partial,
// rates offline, feed offline, a full (capped) feed with large sums, long
// backend symbols and bands, the digest line and headline, every block.
func TestFundingLinesFitEveryPath(t *testing.T) {
	var big []Liquidation
	for i := 0; i < fundingFeedLimit; i++ {
		big = append(big, Liquidation{Symbol: "1000000MOGUSDT", Side: "long_liq", Price: 0.00012345,
			USDValue: 5e6, TS: fundAt.Add(-time.Duration(i) * time.Second)})
	}
	bigLiq := &FundingResp{Feed: big, Zones: []LiqZone{{Symbol: "1000000MOGUSDT", PriceBand: "0.00012345-0.00012407",
		TotalUSD: 999.99e6, Count: 99999, Side: "short_liq"}}}
	crazy := &FundingResp{Feed: big[:3], Zones: []LiqZone{{Symbol: strings.Repeat("VERYLONGSYMBOL", 6), PriceBand: strings.Repeat("9", 40) + "-" + strings.Repeat("9", 41),
		TotalUSD: 1e12, Count: 1 << 30, Side: "weird"}}}
	extreme := map[string]float64{"BTCUSDT": 0.9999, "ETHUSDT": -0.9999, "SOLUSDT": 0.5, "BNBUSDT": -0.5, "XRPUSDT": 0.3}
	marks := func(r map[string]float64) map[string]fundingQuote {
		q := quotesOf(r)
		for k, v := range q {
			v.mark = 0.00001
			q[k] = v
		}
		return q
	}
	cards := []Card{
		fundingCardFrom(quotesOf(liveFloat1640), nil, liqOf(), nil, fundAt),
		fundingCardFrom(quotesOf(planBugFloat), nil, bigLiq, nil, fundAt),
		fundingCardFrom(marks(extreme), nil, bigLiq, nil, fundAt),
		fundingCardFrom(quotesOf(map[string]float64{"BTCUSDT": -0.9999, "ETHUSDT": 0.9999}), nil, crazy, nil, fundAt),
		fundingCardFrom(quotesOf(map[string]float64{"BTCUSDT": 0.0005}), nil, nil, errFakeFunding, fundAt),
		fundingCardFrom(nil, errFakeFunding, bigLiq, nil, fundAt),
		fundingCardFrom(quotesOf(extreme), nil, crazy, nil, fundAt),
		offlineCard("Funding Agent", "Funding", "", keyFunding, howTexts[keyFunding]),
	}
	for i, c := range cards {
		lines := append([]string{c.Verdict, htmlToPlain(c.OneLiner()), digestHeadline(c)}, c.Facts...)
		if b := c.Blocks; b != nil {
			lines = append(lines, b.WhatHappened, b.WhyLevel, b.Regime, b.Limitations)
			lines = append(lines, b.Scenarios...)
			if b.Invalidates != nil {
				lines = append(lines, *b.Invalidates)
			}
		}
		for _, l := range lines {
			if n := utf8.RuneCountInString(l); n > fundingFactMaxRunes {
				t.Errorf("card %d: %d runes: %q", i, n, l)
			}
		}
	}
	// The capped feed says so.
	if f := strings.Join(cards[1].Facts, "|"); !strings.Contains(f, "Liquidations, newest 200 events of the 1h window") {
		t.Errorf("capped feed must say so: %v", cards[1].Facts)
	}
}

// Just past a threshold the ratio says ">1.00×", exactly on it "1.00×".
func TestFundingJustPastRatio(t *testing.T) {
	zero := map[string]float64{"ETHUSDT": 0, "SOLUSDT": 0, "BNBUSDT": 0, "XRPUSDT": 0}
	for rate, want := range map[float64]string{0.00030001: "(>1.00×,", 0.0003: "(1.00×,", -0.00010001: "(>1.00×,",
		-0.0001: "(1.00×,", 0.000303: "(1.01×,"} {
		q := quotesOf(zero)
		q["BTCUSDT"] = fundingQuote{rate: rate}
		c := fundingCardFrom(q, nil, liqOf(), nil, fundAt)
		if !strings.Contains(c.Facts[2], want) {
			t.Errorf("rate %v: %q, want %q", rate, c.Facts[2], want)
		}
	}
}

// capped = the page is full AND its oldest event is still inside the hour.
func TestFundingCappedOnlyWhenWindowMayHoldMore(t *testing.T) {
	page := func(inWindow int) *FundingResp {
		var feed []Liquidation
		for i := 0; i < fundingFeedLimit; i++ {
			ts := fundAt.Add(-time.Duration(i) * time.Second)
			if i >= inWindow {
				ts = fundAt.Add(-2 * time.Hour)
			}
			feed = append(feed, Liquidation{Symbol: "BTCUSDT", Side: "long_liq", Price: 1, USDValue: 1000, TS: ts})
		}
		return &FundingResp{Feed: feed}
	}
	full := fundingCardFrom(quotesOf(liveFloat1640), nil, page(fundingFeedLimit), nil, fundAt)
	if !full.Funding.Liquidations.Capped || !strings.HasPrefix(full.Facts[len(full.Facts)-1], "Liquidations, newest 200 events of the 1h window") {
		t.Errorf("full page inside the hour must be capped: %q", full.Facts[len(full.Facts)-1])
	}
	whole := fundingCardFrom(quotesOf(liveFloat1640), nil, page(50), nil, fundAt)
	if whole.Funding.Liquidations.Capped || whole.Facts[len(whole.Facts)-1] != "Liquidations, last 1h: 50 events · $50.0K · long liqs $50.0K (100%) · short liqs $0.00" {
		t.Errorf("a full page reaching past the hour counted the whole window: capped=%v %q",
			whole.Funding.Liquidations.Capped, whole.Facts[len(whole.Facts)-1])
	}
}

// The body does not move with the mark price while the mark stays on the
// same side of the band: identical envelopes for two ticks.
func TestFundingBodyStableUnderMarkTicks(t *testing.T) {
	liq := &FundingResp{
		Feed:  []Liquidation{{Symbol: "BTCUSDT", Side: "long_liq", Price: 63300, USDValue: 90000, TS: fundAt.Add(-5 * time.Minute)}},
		Zones: []LiqZone{{Symbol: "BTCUSDT", PriceBand: "63100-63400", TotalUSD: 90000, Count: 1, Side: "long_liq"}},
	}
	body := func(mark float64) string {
		q := quotesOf(liveFloat1640)
		q["BTCUSDT"] = fundingQuote{rate: 0.000097, mark: mark}
		b, _ := json.Marshal(cardEnvelope(fundingCardFrom(q, nil, liq, nil, fundAt)))
		return string(b)
	}
	for _, pair := range [][2]float64{{63200, 63399.5}, {63900, 64123.45}, {62000, 63099}} {
		if a, b := body(pair[0]), body(pair[1]); a != b {
			t.Errorf("marks %v: body moved\n%s\n%s", pair, a, b)
		}
	}
	for _, bad := range []string{"mark_price", "distance_pct", "63200"} {
		if strings.Contains(body(63200), bad) {
			t.Errorf("body carries %q", bad)
		}
	}
}

// The showcase conclusion of a funding card is a classification, never a
// direction (the generic one read the semaphore: "bullish … lean up").
func TestFundingConclusionNoDirection(t *testing.T) {
	for _, rates := range []map[string]float64{planBugFloat, liveFloat1640,
		{"BTCUSDT": 0.00041, "ETHUSDT": 0, "SOLUSDT": 0, "BNBUSDT": 0, "XRPUSDT": 0}} {
		c := fundingCardFrom(quotesOf(rates), nil, liqOf(), nil, fundAt)
		got := conclusionFor(c)
		if !strings.Contains(got, "funding classification on "+c.Asset+", not a forecast") {
			t.Errorf("conclusion %q", got)
		}
		for _, bad := range []string{"bullish", "bearish", "lean", "up,", "down,"} {
			if strings.Contains(got, bad) {
				t.Errorf("conclusion carries %q: %q", bad, got)
			}
		}
	}
	c := fundingCardFrom(quotesOf(planBugFloat), nil, liqOf(), nil, fundAt)
	want := "This is a funding classification on XRPUSDT, not a forecast: its last funding rate is past the agent's short threshold, " +
		"so shorts pay an elevated rate; it says nothing about where price goes. " +
		"XRPUSDT funding back above -0.0100% ends this reading (neutral once no major is past a threshold)."
	if got := conclusionFor(c); got != want {
		t.Errorf("conclusion:\n%s\nwant\n%s", got, want)
	}
}

// End to end: the real funding card (network stubs) → the digest ranking →
// the digest headline, one-liner and envelopes name the shown coin.
func TestFundingDigestEndToEnd(t *testing.T) {
	rates := map[string]string{"BTCUSDT": "0.00020000", "ETHUSDT": "0.00001000", "SOLUSDT": "0.00001000",
		"BNBUSDT": "0.00001000", "XRPUSDT": "-0.00015000"}
	fund := fundingStub(t, rates, nil, emptyLiq).FundingCard(context.Background())
	g := gathered{at: fund.DataTime, regime: "risk_on", cards: map[string]Card{
		keyFunding: fund, keyMomentum: momentumCard(0, false), keyTrend: trendCard(trendFlat, 17), keyMacro: macroOK("risk_on")}}
	p := g.selection()
	if p.Winner != keyFunding || p.Rule != ruleConfirmed {
		t.Fatalf("winner %q rule %q, want funding / %s", p.Winner, p.Rule, ruleConfirmed)
	}
	winner, top := topSelection(g)
	if winner != keyFunding {
		t.Fatalf("topSelection: %q", winner)
	}
	if h := digestHeadline(top); h != "Top signal: Funding Agent · XRPUSDT — Negative funding below threshold — shorts pay an elevated rate" {
		t.Errorf("digest headline %q", h)
	}
	env := digestEnvelope(g, "")
	if env.Verdict != digestHeadlineFor(p, top) || !strings.Contains(env.CardHTML, "🟢 <b>Funding Agent</b> · XRPUSDT") {
		t.Errorf("digest envelope verdict %q\n%s", env.Verdict, env.CardHTML)
	}
	if top := cardEnvelope(top); top.Asset != "XRP" {
		t.Errorf("top envelope asset %q, want the base coin XRP", top.Asset)
	}
	// Not the winner: the funding one-liner in the digest sections names it too.
	tr := trendCard(trendUp, 60)
	tr.DataTime = fund.DataTime // fresh at the sweep time, whatever the wall clock
	g.cards[keyTrend] = tr
	env = digestEnvelope(g, "")
	found := false
	for _, s := range env.Sections {
		if s == "🟢 Funding XRPUSDT: shorts pay an elevated rate · -0.0150%" {
			found = true
		}
	}
	if !found {
		t.Errorf("funding section line missing: %v", env.Sections)
	}
}
