package demobot

// macroviews_test.go — B2 asset views of the macro lamps: the gold mapping
// table, the weighted gold score on bull/bear/mixed fixtures, the BTC view
// framing, unknown-regime honesty per asset, the global card's signal map and
// the /agents/macro?asset= routing.

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// macroFixtureJSON builds a /api/v1/macro payload from lamp triples.
// Each lamp: key, value, deltaPct (NaN-free strings), status ("" = no status).
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

// Flight-to-safety with a weak dollar: every voting lamp supports gold.
func goldBullFixture() string {
	return macroFixtureJSON("risk_off", "20",
		macroLampJSON("dxy", "Dollar (DXY)", "97.90", "-0.8", "tailwind"), // dollar down → gold support
		macroLampJSON("rates", "US 10Y", "4.10", "-0.6", "tailwind"),      // yields down → gold support
		macroLampJSON("vix", "VIX", "28.4", "3.1", "headwind"),            // fear → haven bid
		macroLampJSON("spx", "S&P 500", "7300.0", "-1.2", "headwind"),     // equities down → flight to safety
		macroLampJSON("gold", "Gold", "3410.0", "0.9", "headwind"),        // the asset itself — excluded from the vote
	)
}

// Risk-on with a firm dollar: every voting lamp leans against gold.
func goldBearFixture() string {
	return macroFixtureJSON("risk_on", "80",
		macroLampJSON("dxy", "Dollar (DXY)", "99.90", "0.8", "headwind"), // dollar up strong → pressure
		macroLampJSON("rates", "US 10Y", "4.60", "0.7", "headwind"),      // yields up → pressure
		macroLampJSON("vix", "VIX", "15.2", "-1.0", "tailwind"),          // calm → haven bid unwinds
		macroLampJSON("spx", "S&P 500", "7700.0", "0.9", "tailwind"),     // equities bid → pulls from havens
		macroLampJSON("gold", "Gold", "3290.0", "-0.6", "tailwind"),
	)
}

// Split tape: dollar softer (gold support) but calm VIX (gold pressure),
// rates/spx neutral → weighted score lands mid-band.
func goldMixedFixture() string {
	return macroFixtureJSON("mixed", "50",
		macroLampJSON("dxy", "Dollar (DXY)", "98.60", "-0.3", "tailwind"),
		macroLampJSON("rates", "US 10Y", "4.30", "0.1", "neutral"),
		macroLampJSON("vix", "VIX", "16.9", "-0.4", "tailwind"),
		macroLampJSON("spx", "S&P 500", "7500.0", "0.2", "neutral"),
		macroLampJSON("gold", "Gold", "3350.0", "0.0", "neutral"),
	)
}

// ── mapping table ────────────────────────────────────────────────────────────

func TestGoldLampViewMappingTable(t *testing.T) {
	cases := []struct {
		key, crypto, want string
	}{
		// Dollar / yields: same sign as crypto (both dislike a firm dollar and
		// rising yields).
		{"dxy", "tailwind", goldSupport},
		{"dxy", "headwind", goldPressure},
		{"dxy", "neutral", goldNeutral},
		{"rates", "tailwind", goldSupport},
		{"rates", "headwind", goldPressure},
		{"rates", "neutral", goldNeutral},
		// VIX / SPX: inverted (fear and falling equities feed the haven bid).
		{"vix", "tailwind", goldPressure},
		{"vix", "headwind", goldSupport},
		{"vix", "neutral", goldNeutral},
		{"spx", "tailwind", goldPressure},
		{"spx", "headwind", goldSupport},
		{"spx", "neutral", goldNeutral},
		// The gold lamp itself never votes on its own outlook.
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
		t.Errorf("bull fixture score = %v, want 100 (all four voters support)", s)
	}
	if s := goldViewScore(lampsOf(t, goldBearFixture())); s == nil || *s != 0 {
		t.Errorf("bear fixture score = %v, want 0 (all four voters pressure)", s)
	}
	// Mixed: dxy support(+40) + vix pressure(+0) + rates neutral(+12.5) +
	// spx neutral(+5) = 57.5/100 → 58 (mid-band).
	if s := goldViewScore(lampsOf(t, goldMixedFixture())); s == nil || *s != 58 {
		t.Errorf("mixed fixture score = %v, want 58", s)
	}
	// Fewer than 3 voting lamps → nil (no verdict off half a map). Two live
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

func TestMacroGoldCardBullish(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": goldBullFixture()})
	c := ag.MacroAssetCard(context.Background(), macroAssetGold)

	if c.Asset != "GOLD" {
		t.Errorf("asset = %q, want GOLD", c.Asset)
	}
	if c.Emoji != emojiBull {
		t.Errorf("emoji = %q, want bull (gold supported)", c.Emoji)
	}
	if !strings.Contains(c.Verdict, "SUPPORT") {
		t.Errorf("verdict = %q, want a SUPPORT headline", c.Verdict)
	}
	if c.Status != statusOK || c.Offline {
		t.Errorf("status = %v offline = %v, want a real reading", c.Status, c.Offline)
	}
	joined := strings.Join(c.Facts, "|")
	// Per-lamp gold framing with the documented reasons.
	for _, want := range []string{
		"→ support (softer dollar lifts gold)",
		"→ support (falling yields favor holding gold)",
		"→ support (fear = safe-haven bid)",
		"→ mild support (flight to safety)",
		"Gold itself: 3410",
		"Rule: dollar/yields up = pressure · fear (VIX>25) / equities down = support",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("facts missing %q:\n%v", want, c.Facts)
		}
	}
}

func TestMacroGoldCardBearishAndMixed(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": goldBearFixture()})
	c := ag.MacroAssetCard(context.Background(), macroAssetGold)
	if c.Emoji != emojiBear || !strings.Contains(c.Verdict, "PRESSURE") {
		t.Errorf("bear fixture: emoji %q verdict %q, want bear PRESSURE", c.Emoji, c.Verdict)
	}
	joined := strings.Join(c.Facts, "|")
	for _, want := range []string{
		"→ pressure (firmer dollar weighs on gold)",
		"→ pressure (rising yields raise the cost of holding gold)",
		"→ pressure (calm tape unwinds the haven bid)",
		"→ mild pressure (risk appetite pulls money from havens)",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("facts missing %q:\n%v", want, c.Facts)
		}
	}

	ag2 := newStubBackend(t, map[string]string{"/api/v1/macro": goldMixedFixture()})
	c2 := ag2.MacroAssetCard(context.Background(), macroAssetGold)
	if c2.Emoji != emojiNeutral || !strings.Contains(c2.Verdict, "MIXED") {
		t.Errorf("mixed fixture: emoji %q verdict %q, want neutral MIXED", c2.Emoji, c2.Verdict)
	}
}

// Unknown-regime honesty carries into the asset views: zero real lamps →
// UNKNOWN verdict, machine status split on the tradfin clock, no lamp claims.
func TestMacroAssetCardsUnknownHonesty(t *testing.T) {
	for _, asset := range []string{macroAssetGold, macroAssetBTC} {
		ag := newStubBackend(t, map[string]string{"/api/v1/macro": macroUnknownFixture(false)})
		c := ag.MacroAssetCard(context.Background(), asset)
		if !strings.HasPrefix(c.Verdict, strings.ToUpper(c.Asset)+" VIEW: UNKNOWN") {
			t.Errorf("%s verdict = %q, want an UNKNOWN admission", asset, c.Verdict)
		}
		if c.Status != statusMarketClosed {
			t.Errorf("%s status = %v, want market_closed", asset, c.Status)
		}
		if c.Emoji != emojiNeutral {
			t.Errorf("%s emoji = %q, want neutral", asset, c.Emoji)
		}
		joined := strings.Join(c.Facts, "|")
		if strings.Contains(joined, "→ support") || strings.Contains(joined, "→ pressure") ||
			strings.Contains(joined, "→ tailwind") || strings.Contains(joined, "→ headwind") {
			t.Errorf("%s zero-input card claims lamp directions: %v", asset, c.Facts)
		}

		// Open window twin: no_data, no "market closed" claim.
		agOpen := newStubBackend(t, map[string]string{"/api/v1/macro": macroUnknownFixture(true)})
		cOpen := agOpen.MacroAssetCard(context.Background(), asset)
		if cOpen.Status != statusNoData {
			t.Errorf("%s open-window status = %v, want no_data", asset, cOpen.Status)
		}
		if strings.Contains(cOpen.Verdict, "market closed") {
			t.Errorf("%s open-window verdict claims closure: %q", asset, cOpen.Verdict)
		}
	}
}

// Too few voters for a gold read (but real lamps exist) → honest no_data, not
// a fabricated verdict.
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
	if !strings.Contains(c.Verdict, "not enough live lamps") {
		t.Errorf("verdict = %q, want the voter-shortage admission", c.Verdict)
	}
	if c.Emoji != emojiNeutral {
		t.Errorf("emoji = %q, want neutral", c.Emoji)
	}
}

func TestMacroBTCCardFraming(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": goldBearFixture()}) // risk_on
	c := ag.MacroAssetCard(context.Background(), macroAssetBTC)
	if c.Asset != "BTC" {
		t.Errorf("asset = %q, want BTC", c.Asset)
	}
	if c.Emoji != emojiBull {
		t.Errorf("emoji = %q, want bull (risk-on = tailwind for BTC)", c.Emoji)
	}
	if !strings.Contains(c.Verdict, "BTC VIEW: TAILWIND") || !strings.Contains(c.Verdict, "risk-on") {
		t.Errorf("verdict = %q, want the risk-on tailwind framing", c.Verdict)
	}
	joined := strings.Join(c.Facts, "|")
	if !strings.Contains(joined, "→ tailwind") || !strings.Contains(joined, "→ headwind") {
		t.Errorf("per-lamp crypto lines missing: %v", c.Facts)
	}
	if c.Confidence == nil || *c.Confidence != 80 {
		t.Errorf("confidence = %v, want 80 (composite)", c.Confidence)
	}
}

// ── global card signal map ───────────────────────────────────────────────────

func TestMacroCardSignalMapBothViews(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": goldBullFixture()})
	c, _ := ag.MacroCard(context.Background())
	joined := strings.Join(c.Facts, "|")
	if !strings.Contains(joined, "BTC view: headwind — risk-off regime") {
		t.Errorf("BTC view line missing: %v", c.Facts)
	}
	if !strings.Contains(joined, "Gold view: support — 4 lamps for gold / 0 against / 0 neutral") {
		t.Errorf("Gold view line missing: %v", c.Facts)
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
