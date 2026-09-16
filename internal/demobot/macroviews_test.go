package demobot

// macroviews_test.go — B2 asset views of the macro lamps: the gold mapping
// table, the weighted gold score on positive/negative/mixed fixtures, the BTC
// backdrop framing, unknown-regime honesty per asset, the global card's
// context lines and the /agents/macro?asset= routing.

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// macroLampJSON builds one lamp. value/delta are JSON literals ("null" = N/D);
// status "" = no status.
func macroLampJSON(key, label string, value, delta, status string) string {
	ok := "true"
	if value == "null" {
		ok = "false"
	}
	return `{"key":"` + key + `","label":"` + label + `","value":` + value +
		`,"ok":` + ok + `,"delta_pct":` + delta + `,"status":"` + status + `"}`
}

func macroFixtureJSON(regime string, composite string, lamps ...string) string {
	return `{"regime":"` + regime + `","composite":` + composite + `,"tradfin_market_open":true,"tradfin_ok":true,
	  "captured_at":"2026-08-18T06:00:00Z",
	  "lamps":[` + strings.Join(lamps, ",") + `],
	  "fng":{"value":61,"label":"Greed","ok":true},
	  "generated_idea":""}`
}

// Every voting lamp is positive in the gold model (dollar and yields fell,
// VIX above 25, S&P 500 fell more than 0.5%).
func goldBullFixture() string {
	return macroFixtureJSON("risk_off", "20",
		macroLampJSON("dxy", "Dollar (DXY)", "97.90", "-0.8", "tailwind"),
		macroLampJSON("rates", "US 10Y", "4.10", "-0.6", "tailwind"),
		macroLampJSON("vix", "VIX", "28.4", "3.1", "headwind"),
		macroLampJSON("spx", "S&P 500", "7300.0", "-1.2", "headwind"),
		macroLampJSON("gold", "Gold", "3410.0", "0.9", "headwind"), // the asset itself — excluded from the vote
	)
}

// Every voting lamp is negative in the gold model.
func goldBearFixture() string {
	return macroFixtureJSON("risk_on", "80",
		macroLampJSON("dxy", "Dollar (DXY)", "99.90", "0.8", "headwind"),
		macroLampJSON("rates", "US 10Y", "4.60", "0.7", "headwind"),
		macroLampJSON("vix", "VIX", "15.2", "-1.0", "tailwind"),
		macroLampJSON("spx", "S&P 500", "7700.0", "0.9", "tailwind"),
		macroLampJSON("gold", "Gold", "3290.0", "-0.6", "tailwind"),
	)
}

// Split: dollar fell (positive for gold) but VIX below 18 (negative), rates
// and S&P 500 neutral → the weighted score lands mid-band.
func goldMixedFixture() string {
	return macroFixtureJSON("mixed", "50",
		macroLampJSON("dxy", "Dollar (DXY)", "98.60", "-0.3", "tailwind"),
		macroLampJSON("rates", "US 10Y", "4.30", "0.1", "neutral"),
		macroLampJSON("vix", "VIX", "16.9", "-0.4", "tailwind"),
		macroLampJSON("spx", "S&P 500", "7500.0", "-0.2", "neutral"),
		macroLampJSON("gold", "Gold", "3350.0", "0.0", "neutral"),
	)
}

// ── mapping table ────────────────────────────────────────────────────────────

func TestGoldLampViewMappingTable(t *testing.T) {
	cases := []struct {
		key, crypto, want string
	}{
		// Dollar / yields: same sign as the risk rule.
		{"dxy", "tailwind", goldSupport},
		{"dxy", "headwind", goldPressure},
		{"dxy", "neutral", goldNeutral},
		{"rates", "tailwind", goldSupport},
		{"rates", "headwind", goldPressure},
		{"rates", "neutral", goldNeutral},
		// VIX / SPX: inverted in this model.
		{"vix", "tailwind", goldPressure},
		{"vix", "headwind", goldSupport},
		{"vix", "neutral", goldNeutral},
		{"spx", "tailwind", goldPressure},
		{"spx", "headwind", goldSupport},
		{"spx", "neutral", goldNeutral},
		// The gold lamp itself never votes on its own backdrop.
		{"gold", "tailwind", ""},
		{"gold", "headwind", ""},
		// No crypto status → nothing to re-read.
		{"dxy", "", ""},
	}
	for _, tc := range cases {
		if got := goldLampView(tc.key, tc.crypto); got != tc.want {
			t.Errorf("goldLampView(%s, %s) = %q, want %q", tc.key, tc.crypto, got, tc.want)
		}
	}
}

// ── gold score on fixtures ───────────────────────────────────────────────────

func lampsOf(t *testing.T, fixture string) []MacroLamp {
	t.Helper()
	var m MacroResp
	if err := json.Unmarshal([]byte(fixture), &m); err != nil {
		t.Fatal(err)
	}
	return m.Lamps
}

func TestGoldViewScoreFixtures(t *testing.T) {
	if s := goldViewScore(lampsOf(t, goldBullFixture())); s == nil || *s != 100 {
		t.Errorf("bull fixture score = %v, want 100 (all four voters positive)", s)
	}
	if s := goldViewScore(lampsOf(t, goldBearFixture())); s == nil || *s != 0 {
		t.Errorf("bear fixture score = %v, want 0 (all four voters negative)", s)
	}
	// Mixed: dxy positive(+40) + vix negative(+0) + rates neutral(+12.5) +
	// spx neutral(+5) = 57.5/100 → 58 (mid-band).
	if s := goldViewScore(lampsOf(t, goldMixedFixture())); s == nil || *s != 58 {
		t.Errorf("mixed fixture score = %v, want 58", s)
	}
	// Fewer than 3 voting lamps → nil (no reading off half a map). Two live
	// lamps, one of which is gold itself → only 1 voter.
	thin := lampsOf(t, macroFixtureJSON("mixed", "null",
		macroLampJSON("dxy", "Dollar (DXY)", "98.60", "-0.3", "tailwind"),
		macroLampJSON("rates", "US 10Y", "null", "null", ""),
		macroLampJSON("vix", "VIX", "null", "null", ""),
		macroLampJSON("spx", "S&P 500", "null", "null", ""),
		macroLampJSON("gold", "Gold", "3350.0", "0.0", "neutral"),
	))
	if s := goldViewScore(thin); s != nil {
		t.Errorf("1-voter score = %v, want nil (min 3 voters)", *s)
	}
}

// ── per-asset cards ──────────────────────────────────────────────────────────

func TestMacroGoldCardPositive(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": goldBullFixture()})
	c := ag.MacroAssetCard(context.Background(), macroAssetGold)

	if c.Asset != "GOLD" {
		t.Errorf("asset = %q, want GOLD", c.Asset)
	}
	if c.Emoji != emojiBull || c.State != goldSupport {
		t.Errorf("emoji %q state %q, want bull / support (the Gold Agent branches on the state)", c.Emoji, c.State)
	}
	if c.Verdict != "GOLD MACRO BACKDROP: POSITIVE — gold score 100/100 (experimental model)" {
		t.Errorf("verdict = %q", c.Verdict)
	}
	if c.Status != statusOK || c.Offline {
		t.Errorf("status = %v offline = %v, want a real reading", c.Status, c.Offline)
	}
	joined := strings.Join(c.Facts, "|")
	// Each lamp: its rule condition and its contribution in this model — no
	// causes ("softer dollar lifts gold", "flight to safety" are gone).
	for _, want := range []string{
		"DXY 97.90, session -0.80% (fell) → positive for gold, +20.0",
		"US 10Y 4.1000, session -0.60% (fell) → positive for gold, +12.5",
		"VIX 28.40 (>25) → positive for gold, +12.5",
		"S&P 500 7300, session -1.20% (fell >0.5%) → positive for gold, +5.0",
		"Gold 3410, session +0.90% — the asset itself, not an input",
		"Gold score 100/100 = 50 + the contributions above (positive above 65, negative below 35)",
		"Positive holds while the gold score stays above 65 with at least 3 voting lamps",
		"Rule: DXY, US 10Y count as in the risk model; VIX, S&P 500 count inverted; Gold itself does not vote",
		"Weights: DXY 40 · US 10Y 25 · VIX 25 · S&P 500 10 (thresholds are the risk model's)",
		"not in the gold score",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("facts missing %q:\n%v", want, c.Facts)
		}
	}
}

func TestMacroGoldCardNegativeAndMixed(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": goldBearFixture()})
	c := ag.MacroAssetCard(context.Background(), macroAssetGold)
	if c.Emoji != emojiBear || c.State != goldPressure ||
		c.Verdict != "GOLD MACRO BACKDROP: NEGATIVE — gold score 0/100 (experimental model)" {
		t.Errorf("bear fixture: emoji %q state %q verdict %q", c.Emoji, c.State, c.Verdict)
	}
	joined := strings.Join(c.Facts, "|")
	for _, want := range []string{
		"DXY 99.90, session +0.80% (rose >0.5%) → negative for gold, -20.0",
		"US 10Y 4.6000, session +0.70% (rose >0.5%) → negative for gold, -12.5",
		"VIX 15.20 (<18) → negative for gold, -12.5",
		"S&P 500 7700, session +0.90% (rose) → negative for gold, -5.0",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("facts missing %q:\n%v", want, c.Facts)
		}
	}

	ag2 := newStubBackend(t, map[string]string{"/api/v1/macro": goldMixedFixture()})
	c2 := ag2.MacroAssetCard(context.Background(), macroAssetGold)
	if c2.Emoji != emojiNeutral || c2.State != goldNeutral || !strings.Contains(c2.Verdict, "MIXED — gold score 58/100") {
		t.Errorf("mixed fixture: emoji %q state %q verdict %q", c2.Emoji, c2.State, c2.Verdict)
	}
	// 57.5 rounds to 58: the sum is marked approximate, not "=".
	if !strings.Contains(strings.Join(c2.Facts, "|"), "Gold score 58/100 ≈ 50 + the contributions above") {
		t.Errorf("mixed score line: %v", c2.Facts)
	}
}

// Unknown-regime honesty carries into the asset views: zero real lamps →
// UNKNOWN verdict, machine status split on the tradfin clock, no lamp claims.
func TestMacroAssetCardsUnknownHonesty(t *testing.T) {
	for _, asset := range []string{macroAssetGold, macroAssetBTC} {
		ag := newStubBackend(t, map[string]string{"/api/v1/macro": macroUnknownFixture(false)})
		c := ag.MacroAssetCard(context.Background(), asset)
		if !strings.HasPrefix(c.Verdict, strings.ToUpper(c.Asset)+" MACRO BACKDROP: UNKNOWN") {
			t.Errorf("%s verdict = %q, want an UNKNOWN admission", asset, c.Verdict)
		}
		if c.Status != statusMarketClosed {
			t.Errorf("%s status = %v, want market_closed", asset, c.Status)
		}
		if c.Emoji != emojiNeutral {
			t.Errorf("%s emoji = %q, want neutral", asset, c.Emoji)
		}
		joined := strings.Join(c.Facts, "|")
		if strings.Contains(joined, "→ positive") || strings.Contains(joined, "→ negative") {
			t.Errorf("%s zero-input card claims lamp contributions: %v", asset, c.Facts)
		}

		// Open window twin: no_data, no closure claim.
		agOpen := newStubBackend(t, map[string]string{"/api/v1/macro": macroUnknownFixture(true)})
		cOpen := agOpen.MacroAssetCard(context.Background(), asset)
		if cOpen.Status != statusNoData {
			t.Errorf("%s open-window status = %v, want no_data", asset, cOpen.Status)
		}
		if strings.Contains(cOpen.Verdict, "weekend") || strings.Contains(cOpen.Verdict, "closed") {
			t.Errorf("%s open-window verdict claims closure: %q", asset, cOpen.Verdict)
		}
	}
}

// Too few voters for a gold read (but real lamps exist) → honest no_data, not
// a fabricated reading.
func TestMacroGoldCardTooFewVoters(t *testing.T) {
	fixture := macroFixtureJSON("mixed", "null",
		macroLampJSON("dxy", "Dollar (DXY)", "98.60", "-0.3", "tailwind"),
		macroLampJSON("rates", "US 10Y", "null", "null", ""),
		macroLampJSON("vix", "VIX", "null", "null", ""),
		macroLampJSON("spx", "S&P 500", "null", "null", ""),
		macroLampJSON("gold", "Gold", "3350.0", "0.0", "neutral"),
	)
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": fixture})
	c := ag.MacroAssetCard(context.Background(), macroAssetGold)
	if c.Status != statusNoData {
		t.Errorf("status = %v, want no_data (1 voter < 3)", c.Status)
	}
	if c.Verdict != "GOLD MACRO BACKDROP: no read — 1 of 4 lamps vote, the model needs 3" {
		t.Errorf("verdict = %q, want the voter-shortage admission", c.Verdict)
	}
	if c.Emoji != emojiNeutral || c.State != "" {
		t.Errorf("emoji = %q state = %q, want neutral / no state", c.Emoji, c.State)
	}
	if !strings.Contains(strings.Join(c.Facts, "|"), "US 10Y — no data") {
		t.Errorf("missing lamps must be stated: %v", c.Facts)
	}
}

func TestMacroBTCCardFraming(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": goldBearFixture()}) // risk_on
	c := ag.MacroAssetCard(context.Background(), macroAssetBTC)
	if c.Asset != "BTC" {
		t.Errorf("asset = %q, want BTC", c.Asset)
	}
	if c.Emoji != emojiBull {
		t.Errorf("emoji = %q, want bull (risk-on backdrop)", c.Emoji)
	}
	if c.Verdict != "BTC MACRO BACKDROP: RISK-ON — rule score 80/100; BTC direction is not inferred" {
		t.Errorf("verdict = %q", c.Verdict)
	}
	joined := strings.Join(c.Facts, "|")
	if !strings.Contains(joined, "VIX 15.20 (<18) → positive") || !strings.Contains(joined, "(rose >0.5%) → negative") {
		t.Errorf("per-lamp rule lines missing: %v", c.Facts)
	}
	// The composite is a rule score (labeled fact), not a confidence. This
	// fixture's statuses score 60, not the served 80 (a skew): the plain score
	// is shown, never a "50 + …" sum that would not add up.
	if c.Confidence != nil {
		t.Errorf("composite must not render as confidence, got %d", *c.Confidence)
	}
	if !strings.Contains(joined, "Rule score 80/100 (risk-on above 65, risk-off below 35)") {
		t.Errorf("score line missing: %v", c.Facts)
	}
}

// ── global card context lines ────────────────────────────────────────────────

func TestMacroCardAssetContextLines(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": goldBullFixture()})
	c, _ := ag.MacroCard(context.Background())
	joined := strings.Join(c.Facts, "|")
	if !strings.Contains(joined, "BTC macro backdrop: risk-off (the regime itself); BTC direction is not inferred") {
		t.Errorf("BTC context line missing: %v", c.Facts)
	}
	if !strings.Contains(joined, "Gold macro backdrop: positive, gold score 100/100 (separate experimental model, own weights)") {
		t.Errorf("gold context line missing: %v", c.Facts)
	}
}

// ── HTTP routing ─────────────────────────────────────────────────────────────

func TestHTTPMacroAssetViews(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": goldBullFixture()})
	_, srv := newTestAPI(t, ag, true)

	status, _, body := httpGet(t, srv.URL+"/agents/macro?asset=gold")
	if status != 200 {
		t.Fatalf("gold view status %d (%s)", status, body)
	}
	var env testEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env.Asset != "GOLD" || env.Semaphore != "bullish" || !env.OK {
		t.Errorf("gold envelope = asset %q semaphore %q ok %v", env.Asset, env.Semaphore, env.OK)
	}

	status, _, body = httpGet(t, srv.URL+"/agents/macro?asset=btc")
	if status != 200 {
		t.Fatalf("btc view status %d (%s)", status, body)
	}
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env.Asset != "BTC" || env.Semaphore != "bearish" {
		t.Errorf("btc envelope = asset %q semaphore %q, want BTC bearish (risk_off)", env.Asset, env.Semaphore)
	}

	// Unknown asset → 400 naming the allowed values.
	status, _, body = httpGet(t, srv.URL+"/agents/macro?asset=eurusd")
	if status != 400 {
		t.Fatalf("macro?asset=eurusd status %d, want 400 (%s)", status, body)
	}
	if !strings.Contains(string(body), "gold") || !strings.Contains(string(body), "btc") {
		t.Errorf("400 body must list the allowed macro assets: %s", body)
	}

	// No param stays the global regime card.
	status, _, body = httpGet(t, srv.URL+"/agents/macro")
	if status != 200 {
		t.Fatalf("global macro status %d (%s)", status, body)
	}
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env.Asset != "" || !strings.HasPrefix(env.Verdict, "RISK-OFF") {
		t.Errorf("global envelope = asset %q verdict %q, want the plain regime card", env.Asset, env.Verdict)
	}
}
