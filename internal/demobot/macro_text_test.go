package demobot

// macro_text_test.go — what the Macro cards say (macro_text.go), pinned on the
// live payload of 2026-09-15 04:44 UTC plus the properties behind it:
//   - a golden of all three cards and the global blocks;
//   - the contribution arithmetic IS the rule (internal/macro Composite, the
//     gold score) and the breakdown re-adds to the score;
//   - printed numbers never sit on the wrong side of their threshold;
//   - VIX never shows a session change beside its level status;
//   - freshness: the data stamp is the oldest lamp, mixed dates on one line,
//     the scheduled-weekend banner first, never "24h" / "market closed";
//   - Fear & Greed age: fresh / stale at the documented 36h / unknown;
//   - blocks and the machine readout over HTTP.

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/internal/macro"
)

// macroLiveFixture is /api/v1/macro as served on 2026-09-15 04:44 UTC (local
// backend), with the new F&G time fields added.
const macroLiveFixture = `{"regime":"risk_on","composite":83,"tradfin_market_open":true,"tradfin_ok":true,
 "tradfin_as_of":"2026-09-15T04:00:00Z","captured_at":"2026-09-15T04:44:32Z",
 "lamps":[
  {"key":"dxy","label":"Dollar (DXY)","value":99.61299896240234,"ok":true,"delta_pct":0.13369079310498477,"status":"neutral","as_of":"2026-09-15T04:00:00Z","source":"yahoo"},
  {"key":"rates","label":"US 10Y","value":4.960999965667725,"ok":true,"delta_pct":-0.36152089885574407,"status":"tailwind","as_of":"2026-09-14T12:20:00Z","source":"yahoo"},
  {"key":"vix","label":"VIX","value":17.100000381469727,"ok":true,"delta_pct":7.954546822879803,"status":"tailwind","as_of":"2026-09-14T07:00:00Z","source":"yahoo"},
  {"key":"spx","label":"S&P 500","value":7619.97998046875,"ok":true,"delta_pct":0.11220004530341451,"status":"tailwind","as_of":"2026-09-14T13:30:00Z","source":"yahoo"},
  {"key":"gold","label":"Gold","value":4343.7001953125,"ok":true,"delta_pct":0.07834460240114281,"status":"neutral","as_of":"2026-09-15T04:00:00Z","source":"yahoo"}],
 "fng":{"value":69,"label":"Greed","ok":true,"as_of":"2026-09-15T00:00:00Z","fetched_at":"2026-09-15T04:40:00Z"},
 "generated_idea":"Equities are bid and the macro tape leans risk-on, which has historically been a tailwind for crypto."}`

func macroRespOf(t *testing.T, s string) *MacroResp {
	t.Helper()
	var m MacroResp
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func assertLines(t *testing.T, label string, got, want []string) {
	t.Helper()
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Errorf("%s:\ngot:\n  %s\nwant:\n  %s", label, strings.Join(got, "\n  "), strings.Join(want, "\n  "))
	}
}

func TestMacroGlobalCardLiveGolden(t *testing.T) {
	c, regime := macroCardFrom(macroRespOf(t, macroLiveFixture))
	if regime != "risk_on" || c.State != "risk_on" || c.Short != "risk-on" || c.Emoji != emojiBull {
		t.Fatalf("regime %q state %q short %q emoji %q", regime, c.State, c.Short, c.Emoji)
	}
	if c.Verdict != "RISK-ON — rule score 83/100 (risk-on above 65, risk-off below 35)" {
		t.Errorf("verdict: %q", c.Verdict)
	}
	assertLines(t, "global facts", c.Facts, []string{
		"Positive: VIX 17.10 (<18) → +12.5 · S&P 500 +0.11% (rose) → +12.5 · 1 more",
		"Negative: none in this model",
		"Risk-on holds while the rule score stays above 65 with at least 3 voting lamps",
		"Data: 5 of 5 lamps live · Sep 14: US 10Y, VIX, S&P 500 · Sep 15: DXY, Gold",
		"Rule score 83 ≈ 50 + US 10Y +7.5 + VIX +12.5 + S&P 500 +12.5 · neutral: DXY, Gold",
		"BTC macro backdrop: risk-on (the regime itself); BTC direction is not inferred",
		"Gold macro backdrop: mixed, gold score 45/100 (experimental model, own weights)",
		"Crypto Fear & Greed 69 (Greed), Sep 15 · separate index, not in the rule score",
	})
	// The data stamp is the OLDEST live lamp (VIX, Sep 14 07:00), not the
	// build time 04:44 on Sep 15.
	if got := c.DataTime.Format(time.RFC3339); got != "2026-09-14T07:00:00Z" {
		t.Errorf("DataTime = %s, want the oldest lamp stamp", got)
	}
	if !strings.HasSuffix(c.RenderHTML(), "AlphaVizor · 2026-09-14 07:00 UTC · lamps: Yahoo</i>") {
		t.Errorf("footer:\n%s", c.RenderHTML())
	}
	if c.Deviation != 66 {
		t.Errorf("deviation %d, want 66 (priority rule unchanged)", c.Deviation)
	}

	b := c.Blocks
	if b == nil {
		t.Fatal("global card must carry blocks")
	}
	if b.WhyLevel != "" {
		t.Errorf("Macro has no price level: why_level must be empty, got %q", b.WhyLevel)
	}
	assertLines(t, "blocks", []string{b.WhatHappened, b.Regime, b.Scenarios[0], b.Scenarios[1], *b.Invalidates, b.Context}, []string{
		"Lamps (sessions Sep 14 to Sep 15): 3 positive, 0 negative, 2 neutral in this model; rule score 83/100",
		"Risk-on by this model's rule (score 83 above 65): a tradfin backdrop, not a BTC or gold forecast",
		"If the rule score stays above 65 with 3+ voting lamps, the model keeps reading risk-on",
		"If the rule score falls to 35-65, the reading turns mixed; below 35, risk-off",
		"Risk-on ends at a rule score of 65 or below, or with fewer than 3 voting lamps (now 83, 5 voting)",
		"Backdrops: BTC risk-on (direction not inferred) · gold mixed (experimental) · F&G 69, not scored",
	})
	if len(b.Scenarios) != 2 {
		t.Errorf("exactly two scenarios, got %d", len(b.Scenarios))
	}

	// The backend's idea line never renders (the live one said "historically").
	if strings.Contains(c.RenderHTML(), "historically") || strings.Contains(c.RenderHTML(), "Equities are bid") {
		t.Errorf("generated_idea leaked:\n%s", c.RenderHTML())
	}
}

func TestMacroAssetCardsLiveGolden(t *testing.T) {
	m := macroRespOf(t, macroLiveFixture)
	btc := macroAssetCardFrom(m, macroAssetBTC)
	if btc.Verdict != "BTC MACRO BACKDROP: RISK-ON — rule score 83/100; BTC direction is not inferred" {
		t.Errorf("btc verdict: %q", btc.Verdict)
	}
	assertLines(t, "btc facts", btc.Facts, []string{
		"DXY 99.61, session +0.13% (0 to +0.5%) → neutral, 0",
		"US 10Y 4.9610, session -0.36% (fell) → positive, +7.5",
		"VIX 17.10 (<18) → positive, +12.5",
		"S&P 500 7620, session +0.11% (rose) → positive, +12.5",
		"Gold (GC=F futures) 4344, session +0.08% (0 to +0.5%) → neutral, 0",
		"Rule score 83/100 ≈ 50 + the contributions above (risk-on above 65, risk-off below 35)",
		"Risk-on holds while the rule score stays above 65 with at least 3 voting lamps",
		"Data: 5 of 5 lamps live · Sep 14: US 10Y, VIX, S&P 500 · Sep 15: DXY, Gold",
		"Crypto Fear & Greed 69 (Greed), Sep 15 · separate index, not in the rule score",
	})
	if btc.Blocks != nil {
		t.Errorf("blocks ship on the global card only")
	}

	gold := macroAssetCardFrom(m, macroAssetGold)
	if gold.Verdict != "GOLD MACRO BACKDROP: MIXED — gold score 45/100 (experimental model)" || gold.State != goldNeutral {
		t.Errorf("gold verdict %q state %q", gold.Verdict, gold.State)
	}
	assertLines(t, "gold facts", gold.Facts, []string{
		"DXY 99.61, session +0.13% (0 to +0.5%) → neutral for gold, 0",
		"US 10Y 4.9610, session -0.36% (fell) → positive for gold, +12.5",
		"VIX 17.10 (<18) → negative for gold, -12.5",
		"S&P 500 7620, session +0.11% (rose) → negative for gold, -5.0",
		"Gold (GC=F futures) 4344, session +0.08% — the asset itself, not an input",
		"Gold score 45/100 = 50 + the contributions above (positive above 65, negative below 35)",
		"Mixed while the gold score is 35-65: above 65 reads positive, below 35 negative",
		"Rule: DXY, US 10Y count as in the risk model; VIX, S&P 500 count inverted; Gold itself does not vote",
		"Weights: DXY 40 · US 10Y 25 · VIX 25 · S&P 500 10 (thresholds are the risk model's)",
		"Data: 5 of 5 lamps live · Sep 14: US 10Y, VIX, S&P 500 · Sep 15: DXY, Gold",
		"Crypto Fear & Greed 69 (Greed), Sep 15 · separate index, not in the gold score",
	})
}

// ── the arithmetic is the rule ───────────────────────────────────────────────

func toMacroLamps(lamps []MacroLamp) []macro.Lamp {
	out := make([]macro.Lamp, 0, len(lamps))
	for _, l := range lamps {
		out = append(out, macro.Lamp{Key: l.Key, Value: l.Value, OK: l.OK, Status: l.Status})
	}
	return out
}

// Every status combination of the five lamps ("" = no data): the risk model
// scores exactly like macro.Composite, the gold model exactly like
// goldViewScore, and 50 + Σ vs_neutral is the unrounded score.
func TestMacroModelsAreTheRules(t *testing.T) {
	statuses := []string{"tailwind", "neutral", "headwind", ""}
	keys := []string{"dxy", "rates", "vix", "spx", "gold"}
	v := 1.0
	var walk func(i int, lamps []MacroLamp)
	walk = func(i int, lamps []MacroLamp) {
		if i == len(keys) {
			want := macro.Composite(toMacroLamps(lamps))
			r := riskModel.read(lamps)
			if (want == nil) != (r.score == nil) || (want != nil && *want != *r.score) {
				t.Fatalf("%v: risk model %v, macro.Composite %v", lamps, r.score, want)
			}
			for _, rd := range []modelRead{r, goldModel.read(lamps)} {
				if rd.score == nil {
					continue
				}
				sum := 50.0
				for _, lr := range rd.lamps {
					sum += lr.vsNeutral
				}
				if math.Abs(sum-rd.exact) > 1e-9 {
					t.Fatalf("%s %v: 50+Σ=%v, exact %v", rd.m.id, lamps, sum, rd.exact)
				}
			}
			return
		}
		for _, s := range statuses {
			l := MacroLamp{Key: keys[i], Status: s}
			if s != "" {
				l.Value, l.OK = &v, true
			}
			walk(i+1, append(append([]MacroLamp{}, lamps...), l))
		}
	}
	walk(0, nil)
}

// The card's thresholds are the backend rule constants, and the gold model is
// the only other rule.
func TestMacroBandsComeFromTheRule(t *testing.T) {
	if riskModel.high != macro.RuleRiskOnAbove || riskModel.low != macro.RuleRiskOffBelow || riskModel.minVoters != macro.RuleMinActiveLamps {
		t.Errorf("risk model bands drifted from internal/macro: %+v", riskModel)
	}
	if lampCondition("vix", "tailwind") != "<18" || lampCondition("dxy", "headwind") != "rose >0.5%" ||
		lampCondition("spx", "neutral") != "-0.5% to 0" {
		t.Errorf("conditions must print the rule constants")
	}
}

// ── printed numbers stay on their side ───────────────────────────────────────

// parsePrinted reads a printed percentage/level back ("+0.51%", "17.99").
func parsePrinted(t *testing.T, s string) float64 {
	t.Helper()
	v, err := strconv.ParseFloat(strings.TrimSuffix(s, "%"), 64)
	if err != nil {
		t.Fatalf("unparseable %q", s)
	}
	return v
}

func TestMacroPrintedNumbersStayOnTheirSide(t *testing.T) {
	side := func(v, lo, hi float64) int {
		switch {
		case v < lo:
			return -1
		case v > hi:
			return 1
		}
		return 0
	}
	edges := []float64{1e-12, 1e-9, 1e-6, 0.001, 0.004, 0.005, 0.006}
	for _, key := range []string{"dxy", "rates", "gold", "spx"} {
		lo, hi, _ := neutralBand(key)
		var probes []float64
		for x := -2.0; x <= 2.0; x += 0.0007 {
			probes = append(probes, x)
		}
		for _, th := range []float64{lo, hi} {
			for _, e := range edges {
				probes = append(probes, th-e, th+e, th)
			}
		}
		for _, d := range probes {
			got := parsePrinted(t, sessionPctShown(key, d))
			if side(got, lo, hi) != side(d, lo, hi) {
				t.Errorf("%s: %v prints %s — crosses the rule's threshold", key, d, sessionPctShown(key, d))
			}
			// And the printed side matches the backend status the rule gives.
			status := macro.LampStatus(key, 1, d)
			want := map[string]int{"neutral": 0}[status]
			if status != "neutral" {
				want = side(d, lo, hi)
			}
			if side(got, lo, hi) != want {
				t.Errorf("%s: %v (%s) prints %s", key, d, status, sessionPctShown(key, d))
			}
		}
	}
	calm, fear := macro.RuleVIXCalmBelow, macro.RuleVIXFearAbove
	for _, v := range []float64{calm - 1e-9, calm - 0.004, calm, calm + 1e-9, fear - 1e-9, fear, fear + 1e-9, fear + 0.004, 17.1, 30} {
		if got := parsePrinted(t, vixShown(v)); side(got, calm, fear) != side(v, calm, fear) {
			t.Errorf("VIX %v prints %s — crosses 18/25", v, vixShown(v))
		}
	}
	if s := sessionPctShown("spx", -0.0001); s != "+0.00%" && s != "-0.00%" {
		// -0.0001 is neutral for SPX: it must print inside [-0.5, 0].
		if parsePrinted(t, s) > 0 {
			t.Errorf("spx -0.0001 printed %s", s)
		}
	}
	if s := sessionPctShown("spx", -0.0001); s == "-0.00%" {
		t.Errorf("negative zero printed: %s", s)
	}
}

// VIX is read by its level: no card line ever prints a VIX session change.
var vixSessionText = regexp.MustCompile(`VIX [^·→]*session`)

func TestMacroVIXNeverShowsSessionChange(t *testing.T) {
	m := macroRespOf(t, macroLiveFixture) // VIX delta +7.95% in the payload
	g, _ := macroCardFrom(m)
	for _, c := range []Card{g, macroAssetCardFrom(m, macroAssetBTC), macroAssetCardFrom(m, macroAssetGold)} {
		for _, s := range macroTexts(c) {
			if vixSessionText.MatchString(s) || strings.Contains(s, "7.95") {
				t.Errorf("VIX session change printed: %q", s)
			}
		}
	}
}

// ── freshness ────────────────────────────────────────────────────────────────

func TestMacroWeekendBannerAndWording(t *testing.T) {
	m := macroRespOf(t, strings.Replace(macroLiveFixture, `"tradfin_market_open":true`, `"tradfin_market_open":false`, 1))
	g, _ := macroCardFrom(m)
	for _, c := range []Card{g, macroAssetCardFrom(m, macroAssetBTC), macroAssetCardFrom(m, macroAssetGold)} {
		if len(c.Facts) == 0 || c.Facts[0] != macroWeekendBanner {
			t.Errorf("%s: the weekend banner must be the first fact: %v", c.Asset, c.Facts)
		}
		if c.Macro == nil || !c.Macro.Freshness.ScheduledWeekend {
			t.Errorf("%s: freshness.scheduled_weekend must be true", c.Asset)
		}
	}
	if o, _ := macroCardFrom(macroRespOf(t, macroLiveFixture)); strings.Contains(strings.Join(o.Facts, "|"), "weekend") {
		t.Errorf("open week must not show the banner: %v", o.Facts)
	}
}

func TestMacroDataLineShapes(t *testing.T) {
	lamp := func(key, asOf string, live bool) MacroLamp {
		l := MacroLamp{Key: key, AsOf: asOf}
		if live {
			v := 1.0
			l.Value, l.OK = &v, true
		}
		return l
	}
	cases := []struct {
		lamps []MacroLamp
		want  string
	}{
		{[]MacroLamp{lamp("dxy", "2026-09-14T04:00:00Z", true), lamp("vix", "2026-09-14T07:00:00Z", true)},
			"Data: 2 of 2 lamps live · session Sep 14"},
		{[]MacroLamp{lamp("dxy", "2026-09-15T04:00:00Z", true), lamp("vix", "", true), lamp("gold", "2026-09-11T04:00:00Z", false)},
			"Data: 2 of 3 lamps live · Sep 15: DXY · date unknown: VIX · no data: Gold"},
	}
	for _, tc := range cases {
		if got := macroDataLine(tc.lamps); got != tc.want {
			t.Errorf("got  %q\nwant %q", got, tc.want)
		}
	}
}

func TestMacroMixedSourcesAndInstrument(t *testing.T) {
	s := strings.Replace(macroLiveFixture, `"status":"neutral","as_of":"2026-09-15T04:00:00Z","source":"yahoo"}]`,
		`"status":"neutral","as_of":"2026-09-15T04:00:00Z","source":"stooq"}]`, 1)
	s = strings.Replace(s, `"status":"tailwind","as_of":"2026-09-14T07:00:00Z","source":"yahoo"`,
		`"status":"tailwind","as_of":"2026-09-14T07:00:00Z","source":"stooq"`, 1)
	m := macroRespOf(t, s)
	c, _ := macroCardFrom(m)
	if c.SourceNote != "lamps: Stooq 2 · Yahoo 3" {
		t.Errorf("source note: %q", c.SourceNote)
	}
	gold := macroAssetCardFrom(m, macroAssetGold)
	if !strings.Contains(strings.Join(gold.Facts, "|"), "Gold (XAUUSD spot) 4344") {
		t.Errorf("stooq gold is XAUUSD spot: %v", gold.Facts)
	}
	if gold.Macro.Freshness.Sources["stooq"] != 2 || gold.Macro.Freshness.Sources["yahoo"] != 3 {
		t.Errorf("sources: %v", gold.Macro.Freshness.Sources)
	}
}

// ── Fear & Greed age ─────────────────────────────────────────────────────────

func TestMacroFearGreedAge(t *testing.T) {
	now := time.Date(2026, 9, 15, 4, 44, 32, 0, time.UTC)
	at := func(d time.Duration) string { return now.Add(-d).Format(time.RFC3339) }
	cases := []struct {
		name  string
		f     FearGreedCheck
		want  string
		stale bool
	}{
		{"fresh", FearGreedCheck{Value: 69, Label: "Greed", OK: true, AsOf: at(4 * time.Hour)},
			"Crypto Fear & Greed 69 (Greed), Sep 15 · separate index, not in the rule score", false},
		{"exactly 36h is not stale", FearGreedCheck{Value: 69, Label: "Greed", OK: true, AsOf: at(36 * time.Hour)},
			"Crypto Fear & Greed 69 (Greed), Sep 13 · separate index, not in the rule score", false},
		{"36h + 1s is stale", FearGreedCheck{Value: 69, Label: "Greed", OK: true, AsOf: at(36*time.Hour + time.Second)},
			"Crypto Fear & Greed 69 (Greed): stale, last update Sep 13 (36h ago) · not in the rule score", true},
		{"days old", FearGreedCheck{Value: 20, Label: "Extreme Fear", OK: true, AsOf: "2026-09-10T00:00:00Z"},
			"Crypto Fear & Greed 20 (Extreme Fear): stale, last update Sep 10 (5d ago) · not in the rule score", true},
		{"only our fetch time", FearGreedCheck{Value: 69, Label: "Greed", OK: true, FetchedAt: at(time.Hour)},
			"Crypto Fear & Greed 69 (Greed), Sep 15 · separate index, not in the rule score", false},
		{"older backend: no time at all", FearGreedCheck{Value: 69, Label: "Greed", OK: true},
			"Crypto Fear & Greed 69 (Greed), update time unknown · not in the rule score", false},
	}
	for _, tc := range cases {
		f := tc.f
		if got := fngLine(&f, now, riskModel.scoreName); got != tc.want {
			t.Errorf("%s:\ngot  %q\nwant %q", tc.name, got, tc.want)
		}
		m := macroRespOf(t, macroLiveFixture)
		m.CapturedAt = now.Format(time.RFC3339)
		m.FNG = &f
		c, _ := macroCardFrom(m)
		if fg := c.Macro.FearGreed; fg == nil || fg.Stale != tc.stale || fg.InScore || fg.StaleAfterHours != 36 {
			t.Errorf("%s: readout %+v, want stale=%v in_score=false", tc.name, fg, tc.stale)
		}
	}
	if fngLine(nil, now, "rule score") != "" || fngLine(&FearGreedCheck{OK: false}, now, "rule score") != "" {
		t.Error("no line without a live value")
	}
}

// ── machine readout over HTTP ────────────────────────────────────────────────

func TestHTTPMacroReadoutAndBlocks(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": macroLiveFixture})
	_, srv := newTestAPI(t, ag, true)
	status, _, body := httpGet(t, srv.URL+"/agents/macro")
	if status != 200 {
		t.Fatalf("status %d: %s", status, body)
	}
	var env struct {
		Confidence *int            `json:"confidence"`
		DataAsOf   string          `json:"data_as_of"`
		Blocks     json.RawMessage `json:"blocks"`
		Macro      *MacroReadout   `json:"macro"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env.Confidence != nil {
		t.Errorf("macro confidence stays null, got %d", *env.Confidence)
	}
	if env.DataAsOf != "2026-09-14T07:00:00Z" {
		t.Errorf("data_as_of %q, want the oldest lamp", env.DataAsOf)
	}
	if !strings.Contains(string(env.Blocks), `"why_level":""`) || !strings.Contains(string(env.Blocks), `"context":"Backdrops: BTC`) {
		t.Errorf("blocks: %s", env.Blocks)
	}
	r := env.Macro
	if r == nil {
		t.Fatalf("macro readout missing: %s", body)
	}
	if r.Model != "risk_appetite" || r.Reading != "risk_on" || r.RuleScore == nil || *r.RuleScore != 83 ||
		r.RuleScoreExact == nil || *r.RuleScoreExact != 82.5 || r.Bands != (MacroBands{Low: 35, High: 65}) ||
		r.MinVotingLamps != 3 || r.VotingLamps != 5 || r.LiveLamps != 5 || r.IsForecast || r.Experimental {
		t.Errorf("readout head: %+v", r)
	}
	vix := r.Lamps[2]
	if vix.Key != "vix" || vix.Contribution != "positive" || vix.Weight != 25 || *vix.Points != 25 || *vix.MaxPoints != 25 ||
		*vix.VsNeutral != 12.5 || vix.Rule != (MacroLampRule{Input: "level", PositiveWhen: "< 18", NegativeWhen: "> 25"}) ||
		vix.AsOf != "2026-09-14T07:00:00Z" || vix.Source != "yahoo" {
		t.Errorf("vix readout: %+v", vix)
	}
	if g := r.Lamps[4]; g.Instrument != "GC=F futures" || g.Rule.NegativeWhen != "> 0.5" {
		t.Errorf("gold readout: %+v", g)
	}
	fr := r.Freshness
	if fr.OldestAsOf != "2026-09-14T07:00:00Z" || !fr.Mixed || strings.Join(fr.SessionDates, ",") != "2026-09-14,2026-09-15" ||
		fr.ScheduledWeekend || fr.CapturedAt != "2026-09-15T04:44:32Z" {
		t.Errorf("freshness: %+v", fr)
	}
	if fg := r.FearGreed; fg == nil || fg.AgeHours == nil || *fg.AgeHours != 4.74 || fg.Stale {
		t.Errorf("fear_greed: %+v", fg)
	}

	// The gold view serves its own model, inverted rule on VIX.
	_, _, body = httpGet(t, srv.URL+"/agents/macro?asset=gold")
	var genv struct {
		Macro *MacroReadout `json:"macro"`
	}
	if err := json.Unmarshal(body, &genv); err != nil {
		t.Fatal(err)
	}
	if g := genv.Macro; g == nil || g.Model != "gold_backdrop" || !g.Experimental || g.Reading != "mixed" ||
		g.Lamps[2].Rule.PositiveWhen != "> 25" || g.Lamps[4].Voting || g.Lamps[4].Weight != 0 {
		t.Errorf("gold readout: %+v", g)
	}
}

// ── no causal, predictive or confidence wording anywhere ─────────────────────

// macroBanned: causes ("favor", "pulls", "haven"…), forecasts ("will"),
// strength/confidence names for the score, "24h" (the change is Close −
// session Open) and "market closed" (the week window knows no holidays).
var macroBanned = []string{
	"favor", "favour", "tends to", "pull", "unwind", "haven", "flight", "lifts", "weighs",
	"rotating", "money", "tailwind", "headwind", "support", "pressure", "will ",
	"24h", "24 h", "market closed", "confidence", "strength", "historically", "big money",
	"risk appetite score",
}

// macroTexts is every reader-facing string of a macro card.
func macroTexts(c Card) []string {
	out := append([]string{c.Verdict, c.Short, c.HowItWorks, c.SourceNote}, c.Facts...)
	if b := c.Blocks; b != nil {
		out = append(out, b.WhatHappened, b.WhyLevel, b.Regime, b.Context)
		out = append(out, b.Scenarios...)
		if b.Invalidates != nil {
			out = append(out, *b.Invalidates)
		}
	}
	return out
}

func checkMacroWording(t *testing.T, label string, c Card) {
	t.Helper()
	for _, s := range macroTexts(c) {
		low := strings.ToLower(s)
		for _, w := range macroBanned {
			if strings.Contains(low, w) {
				t.Errorf("%s: %q contains %q", label, s, w)
			}
		}
	}
}

func TestMacroCardsNoCausalWording(t *testing.T) {
	for _, fx := range []string{macroLiveFixture, goldBullFixture(), goldBearFixture(), goldMixedFixture(),
		macroUnknownFixture(false), macroUnknownFixture(true)} {
		m := macroRespOf(t, fx)
		g, _ := macroCardFrom(m)
		checkMacroWording(t, "global", g)
		checkMacroWording(t, "btc", macroAssetCardFrom(m, macroAssetBTC))
		checkMacroWording(t, "gold", macroAssetCardFrom(m, macroAssetGold))
	}
}

func ExampleMacroReadout() {
	fmt.Println(MacroBands{Low: macro.RuleRiskOffBelow, High: macro.RuleRiskOnAbove})
	// Output: {35 65}
}
