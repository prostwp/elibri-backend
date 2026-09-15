package demobot

// funding_review_test.go — Funding stage 1, review fixes (2026-09-15): the
// cluster is picked without the mark price (the body must not flip between
// two bands while the mark wanders between them), "1 event", a Why line that
// does not compete with "Nearest threshold", crossed scenarios that change
// the classification, no cluster beside an empty 1h window, one machine
// asset on /showcase/example and /agents/funding, bot texts without
// "pressure".

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// Two BTC bands with a gap between them: the largest (by USD) sits above, the
// smaller below. A mark walking 118290 → 118310 → 118295 crosses the middle
// between their midpoints (118300) but no band edge.
func twoBTCZones() []LiqZone {
	return []LiqZone{
		{Symbol: "BTCUSDT", PriceBand: "117410-118000", TotalUSD: 90000, Count: 1, Side: "short_liq"},
		{Symbol: "BTCUSDT", PriceBand: "118600-119190", TotalUSD: 500000, Count: 5, Side: "long_liq"},
	}
}

func TestFundingClusterPickIgnoresMark(t *testing.T) {
	liq := &FundingResp{
		Feed:  []Liquidation{{Symbol: "BTCUSDT", Side: "long_liq", Price: 118700, USDValue: 90000, TS: fundAt.Add(-5 * time.Minute)}},
		Zones: twoBTCZones(),
	}
	body := func(mark float64) string {
		q := quotesOf(liveFloat1640)
		q["BTCUSDT"] = fundingQuote{rate: 0.000097, mark: mark}
		b, err := json.Marshal(cardEnvelope(fundingCardFrom(q, nil, liq, nil, fundAt)))
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	first := body(118290)
	for _, m := range []float64{118310, 118295, 0} {
		if m == 0 {
			continue // no mark: covered below (band_vs_mark drops, the pick stays)
		}
		if got := body(m); got != first {
			t.Errorf("mark %v: body moved with the mark between two bands\n%s\n%s", m, first, got)
		}
	}
	c := fundingCardFrom(quotesOf(liveFloat1640), nil, liq, nil, fundAt)
	if cl := c.Funding.Liquidations.Cluster; cl == nil || cl.PriceBand != "118600-119190" || cl.USD != 500000 {
		t.Fatalf("cluster %+v, want the largest BTC band 118600-119190", cl)
	}
	if !strings.Contains(first, `"band_vs_mark":"above"`) {
		t.Errorf("band_vs_mark must stay: %.600s", first)
	}
}

// The pick: BTCUSDT bands first (else every band), the largest USD, then more
// events, then the symbol, then the lower band, then the band text — never
// the served order.
func TestFundingClusterZoneRule(t *testing.T) {
	cases := []struct {
		name  string
		zones []LiqZone
		want  string
	}{
		{"largest BTC band, not the first served", twoBTCZones(), "BTCUSDT 118600-119190"},
		{"BTC beats a larger ETH band", []LiqZone{{Symbol: "ETHUSDT", PriceBand: "4000-4020", TotalUSD: 9e6, Count: 50},
			{Symbol: "BTCUSDT", PriceBand: "90000-90450", TotalUSD: 1000, Count: 1}}, "BTCUSDT 90000-90450"},
		{"no BTC band: the largest of all", []LiqZone{{Symbol: "SOLUSDT", PriceBand: "150-151", TotalUSD: 100, Count: 1},
			{Symbol: "ETHUSDT", PriceBand: "4000-4020", TotalUSD: 900, Count: 3}}, "ETHUSDT 4000-4020"},
		{"equal USD: more events", []LiqZone{{Symbol: "BTCUSDT", PriceBand: "90000-90450", TotalUSD: 500, Count: 2},
			{Symbol: "BTCUSDT", PriceBand: "95000-95475", TotalUSD: 500, Count: 7}}, "BTCUSDT 95000-95475"},
		{"equal USD and events: the lower band", []LiqZone{{Symbol: "BTCUSDT", PriceBand: "100000-100500", TotalUSD: 500, Count: 2},
			{Symbol: "BTCUSDT", PriceBand: "99000-99495", TotalUSD: 500, Count: 2}}, "BTCUSDT 99000-99495"},
		{"no BTC band, equal USD and events: the symbol first", []LiqZone{{Symbol: "SOLUSDT", PriceBand: "150-151", TotalUSD: 5, Count: 1},
			{Symbol: "ETHUSDT", PriceBand: "4000-4020", TotalUSD: 5, Count: 1}}, "ETHUSDT 4000-4020"},
		{"an unparsable band sorts after a parsable one", []LiqZone{{Symbol: "BTCUSDT", PriceBand: "garbage", TotalUSD: 5, Count: 1},
			{Symbol: "BTCUSDT", PriceBand: "90000-90450", TotalUSD: 5, Count: 1}}, "BTCUSDT 90000-90450"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, zones := range [][]LiqZone{tc.zones, {tc.zones[1], tc.zones[0]}} {
				z := fundingClusterZone(zones)
				if got := z.Symbol + " " + z.PriceBand; got != tc.want {
					t.Errorf("order %v: got %q, want %q", zones, got, tc.want)
				}
			}
		})
	}
}

// The same through the push hook: the mark moves between the two bands on
// every premiumIndex read, the data does not — one hash.
func TestFundingHookStableMarkBetweenBands(t *testing.T) {
	stubExternalBases(t)
	marks := []string{"118290.00", "118310.00", "118295.00"}
	var mu sync.Mutex
	n := 0
	prem := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sym := r.URL.Query().Get("symbol")
		mark := "100.00"
		if sym == "BTCUSDT" {
			mu.Lock()
			mark = marks[n%len(marks)]
			n++
			mu.Unlock()
		}
		_, _ = w.Write([]byte(`{"symbol":"` + sym + `","lastFundingRate":"` + liveRates1640[sym] + `","markPrice":"` + mark + `"}`))
	}))
	t.Cleanup(prem.Close)
	premiumIndexURL = prem.URL + "/?symbol="
	liqAt := time.Now().UTC().Add(-10 * time.Minute).Format(time.RFC3339)
	body := `{"captured_at":"x","feed":[
	  {"symbol":"BTCUSDT","side":"long_liq","qty":1,"price":118700,"usd_value":500000,"ts":"` + liqAt + `"},
	  {"symbol":"BTCUSDT","side":"short_liq","qty":1,"price":117500,"usd_value":90000,"ts":"` + liqAt + `"}],
	  "zones":[{"symbol":"BTCUSDT","price_band":"118600-119190","total_usd":500000,"count":5,"side":"long_liq"},
	           {"symbol":"BTCUSDT","price_band":"117410-118000","total_usd":90000,"count":1,"side":"short_liq"}]}`
	ag := newStubBackend(t, map[string]string{"/api/v1/funding/liquidations": body})

	var tg hookTarget
	for _, h := range hookTargets() {
		if h.Agent == keyFunding {
			tg = h
		}
	}
	if tg.Path == "" {
		t.Fatal("no funding hook target")
	}
	th := newTestHook(t, ag, "http://127.0.0.1:1", []hookTarget{tg})
	var hashes []string
	var bodies [][]byte
	for i := 0; i < len(marks); i++ {
		start := time.Now()
		st, b := th.fetch(context.Background(), tg)
		if st != http.StatusOK {
			t.Fatalf("read %d: status %d %s", i, st, b)
		}
		hashes = append(hashes, hashOf(t, tg.Agent, b, start, time.Now()))
		bodies = append(bodies, b)
	}
	for i := 1; i < len(hashes); i++ {
		if hashes[i] != hashes[0] {
			t.Errorf("read %d: hash moved with the mark only\n%s\n%s", i, bodies[0], bodies[i])
		}
	}
	if !strings.Contains(string(bodies[0]), "118600-119190") {
		t.Errorf("expected the largest band: %.800s", bodies[0])
	}
}

func TestFundingSingularEvent(t *testing.T) {
	liq := &FundingResp{
		Feed:  []Liquidation{{Symbol: "BTCUSDT", Side: "long_liq", Price: 63300, USDValue: 90000, TS: fundAt.Add(-5 * time.Minute)}},
		Zones: []LiqZone{{Symbol: "BTCUSDT", PriceBand: "63100-63400", TotalUSD: 90000, Count: 1, Side: "long_liq"}},
	}
	c := fundingCardFrom(quotesOf(planBugFloat), nil, liq, nil, fundAt)
	joined := strings.Join(c.Facts, "\n")
	for _, want := range []string{"Liquidations, last 1h: 1 event · $90.0K",
		"Observed liquidation cluster, last 1h: BTCUSDT 63100-63400 · $90.0K · 1 event"} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in\n%s", want, joined)
		}
	}
	if !strings.Contains(c.Blocks.WhatHappened, "liquidations 1h: 1 event, $90.0K") {
		t.Errorf("what_happened %q", c.Blocks.WhatHappened)
	}
	if strings.Contains(joined+c.Blocks.WhatHappened, "1 events") {
		t.Errorf("plural for one: %s", joined)
	}
}

// Within the thresholds the Why line names the measure (ratio), not a
// "closest" that could disagree with the Nearest-threshold line above it:
// BTC +0.0090% is 0.30× of its long threshold, while its nearest threshold
// in pp is the short one.
func TestFundingWhyLineDoesNotCompeteWithNearest(t *testing.T) {
	q := quotesOf(map[string]float64{"BTCUSDT": 0.00009, "ETHUSDT": 0, "SOLUSDT": 0, "BNBUSDT": 0, "XRPUSDT": 0})
	c := fundingCardFrom(q, nil, liqOf(), nil, fundAt)
	if c.Facts[1] != "Nearest threshold: short -0.0100%, 0.0190 pp away" {
		t.Fatalf("nearest line %q", c.Facts[1])
	}
	if c.Facts[2] != "Why BTCUSDT: largest ratio to its own side's threshold of 5 majors (0.30×, rate ÷ threshold)" {
		t.Errorf("why line %q", c.Facts[2])
	}
	if c.Blocks.WhatHappened != "Within thresholds, largest ratio BTCUSDT +0.0090% vs +0.0300% · no liquidation events in the 1h window" {
		t.Errorf("what_happened %q", c.Blocks.WhatHappened)
	}
	for _, l := range append([]string{c.Blocks.WhatHappened, c.HowItWorks}, c.Facts...) {
		if strings.Contains(l, "closest") {
			t.Errorf("%q says closest", l)
		}
	}
}

// Past a threshold, both scenarios change the classification: every major
// back inside, or another major past the other side's threshold at a larger
// ratio. No "stays past" tautology.
func TestFundingCrossedScenariosChangeState(t *testing.T) {
	c := fundingCardFrom(quotesOf(planBugFloat), nil, liqOf(), nil, fundAt)
	wantFacts(t, c.Blocks.Scenarios, []string{
		"If every major returns inside -0.0100% to +0.0300%, the state turns within thresholds",
		"If a major is past +0.0300% at a larger ratio than XRPUSDT, the state turns positive funding above threshold",
	})
	pos := fundingCardFrom(quotesOf(map[string]float64{"BTCUSDT": 0.00041, "ETHUSDT": 0, "SOLUSDT": 0, "BNBUSDT": 0,
		"XRPUSDT": 0}), nil, liqOf(), nil, fundAt)
	if pos.Blocks.Scenarios[1] != "If a major is past -0.0100% at a larger ratio than BTCUSDT, the state turns negative funding below threshold" {
		t.Errorf("positive scenario %q", pos.Blocks.Scenarios[1])
	}
	for _, s := range append(c.Blocks.Scenarios, pos.Blocks.Scenarios...) {
		if strings.Contains(s, "stays") {
			t.Errorf("tautology: %q", s)
		}
	}
}

// Zones and the feed window are counted by different clocks at the edge of
// the hour (the backend prunes on its clock, the card counts from the request
// time). With no event in the card's window the card shows no cluster.
func TestFundingNoClusterBesideEmptyWindow(t *testing.T) {
	liq := &FundingResp{
		Feed:  []Liquidation{{Symbol: "BTCUSDT", Side: "long_liq", Price: 63300, USDValue: 412000, TS: fundAt.Add(-61 * time.Minute)}},
		Zones: []LiqZone{{Symbol: "BTCUSDT", PriceBand: "63100-63400", TotalUSD: 412000, Count: 9, Side: "long_liq"}},
	}
	c := fundingCardFrom(quotesOf(liveFloat1640), nil, liq, nil, fundAt)
	if last := c.Facts[len(c.Facts)-1]; last != "Liquidations: no events in the current 1h window" {
		t.Errorf("last fact %q", last)
	}
	if strings.Contains(strings.Join(c.Facts, "\n"), "cluster") || strings.Contains(strings.Join(c.Facts, "\n"), "Cluster") {
		t.Errorf("a cluster beside an empty window: %v", c.Facts)
	}
	if c.Funding.Liquidations == nil || c.Funding.Liquidations.Cluster != nil {
		t.Errorf("machine cluster must be null on an empty window: %+v", c.Funding.Liquidations)
	}
}

// /showcase/example carries the same machine asset as /agents/funding (the
// base coin), not the display symbol.
func TestShowcaseExampleAssetMatchesAgent(t *testing.T) {
	rates := map[string]string{"BTCUSDT": "0.00020000", "ETHUSDT": "0.00001000", "SOLUSDT": "0.00001000",
		"BNBUSDT": "0.00001000", "XRPUSDT": "-0.00015000"}
	_, srv := newTestAPI(t, fundingStub(t, rates, nil, emptyLiq), true)
	st, _, body := httpGet(t, srv.URL+"/showcase/example")
	if st != 200 {
		t.Fatalf("example status %d: %s", st, body)
	}
	var ex testShowcaseExample
	if err := json.Unmarshal(body, &ex); err != nil {
		t.Fatal(err)
	}
	st, _, ab := httpGet(t, srv.URL+"/agents/funding")
	var env struct {
		Asset string `json:"asset"`
	}
	if err := json.Unmarshal(ab, &env); err != nil || st != 200 {
		t.Fatalf("/agents/funding %d %v: %s", st, err, ab)
	}
	if ex.Slug != keyFunding || ex.Asset != "XRP" || env.Asset != ex.Asset {
		t.Errorf("slug %q example asset %q, /agents/funding asset %q: want both XRP", ex.Slug, ex.Asset, env.Asset)
	}
	if !strings.Contains(ex.Detected, "XRPUSDT") {
		t.Errorf("detected keeps the display symbol: %q", ex.Detected)
	}
}

// The Telegram menu and /help describe funding as the card does: a last
// funding rate against thresholds, not "pressure".
func TestBotFundingTextsMatchCard(t *testing.T) {
	for _, bc := range botCommands {
		if bc.Command == keyFunding && strings.Contains(strings.ToLower(bc.Description), "pressure") {
			t.Errorf("menu: %q", bc.Description)
		}
	}
	if strings.Contains(helpText, "funding pressure") {
		t.Error("/help says funding pressure")
	}
	if !strings.Contains(helpText, "/funding — last perp funding rate vs thresholds &amp; 1h liquidations") {
		t.Errorf("/help funding line")
	}
}
