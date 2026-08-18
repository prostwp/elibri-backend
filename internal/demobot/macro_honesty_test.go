package demobot

// macro_honesty_test.go — the unknown-regime honesty contract (team-testing
// defect 2026-08): when the backend has ZERO real tradfin lamps, the bot must
// admit it — "UNKNOWN … no tradfin data" with a neutral semaphore — instead of
// a confident "MIXED — no single regime in control" built on nothing. Covers:
//   - the unknown card golden (closed + open window wording),
//   - the digest one-liner "⚪ Macro: unknown (no data)",
//   - version-skew reclassification (old backend: "mixed" + all-null lamps),
//   - the "Macro signals are split" line never rendering without a real lamp,
//   - regime "unknown" never winning /top (priority_test.go holds the pickTop
//     cases; here the HTTP envelope serves the unknown card as an honest 200).

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// macroUnknownFixture is the NEW backend shape for the all-null case: regime
// "unknown", per-lamp ok:false with propagated stale as_of, tradfin_ok false,
// idea blanked. F&G (crypto side) still live.
func macroUnknownFixture(marketOpen bool) string {
	openStr := "false"
	if marketOpen {
		openStr = "true"
	}
	return `{"regime":"unknown","composite":null,"tradfin_market_open":` + openStr + `,"tradfin_ok":false,
	  "captured_at":"2026-08-18T06:00:00Z",
	  "lamps":[
	    {"key":"dxy","label":"Dollar (DXY)","value":null,"ok":false,"delta_pct":null,"status":"","as_of":"2026-08-14T20:55:00Z"},
	    {"key":"rates","label":"US 10Y","value":null,"ok":false,"delta_pct":null,"status":"","as_of":""},
	    {"key":"vix","label":"VIX","value":null,"ok":false,"delta_pct":null,"status":"","as_of":"2026-08-14T20:55:00Z"},
	    {"key":"spx","label":"S&P 500","value":null,"ok":false,"delta_pct":null,"status":"","as_of":"2026-08-14T20:55:00Z"},
	    {"key":"gold","label":"Gold","value":null,"ok":false,"delta_pct":null,"status":"","as_of":""}],
	  "fng":{"value":61,"label":"Greed","ok":true},
	  "generated_idea":""}`
}

func TestMacroCardUnknownRegimeClosedGolden(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": macroUnknownFixture(false)})
	c, regime := ag.MacroCard(context.Background())

	if regime != "unknown" {
		t.Fatalf("regime: got %q, want unknown", regime)
	}
	if c.Offline {
		t.Fatal("unknown is an honest verdict about data absence, not an offline card")
	}
	if c.Confidence != nil {
		t.Errorf("confidence must be absent (composite null), got %v", c.Confidence)
	}
	if c.Deviation != 0 {
		t.Errorf("deviation: got %d, want 0", c.Deviation)
	}

	// Byte-exact card: neutral semaphore, an ADMISSION instead of a claim,
	// facts limited to what IS known (crypto F&G).
	want := "⚪ <b>Macro Agent</b>\n" +
		"<b>UNKNOWN — market closed, no tradfin data</b>\n" +
		"• Lamps: no tradfin data right now (market closed)\n" +
		"• Crypto Fear &amp; Greed: 61 — Greed\n" +
		"\n<i>Analytics, not financial advice · AlphaVizor · 2026-08-18 06:00 UTC</i>"
	if got := c.RenderHTML(); got != want {
		t.Errorf("unknown card golden mismatch:\ngot:\n%s\nwant:\n%s", got, want)
	}

	// Digest one-liner: "⚪ Macro: unknown (no data)".
	if got, want := htmlToPlain(c.OneLiner()), "⚪ Macro: unknown (no data)"; got != want {
		t.Errorf("one-liner: got %q, want %q", got, want)
	}
}

func TestMacroCardUnknownRegimeOpenWindowWording(t *testing.T) {
	// Inside the open window (data lag) the card must not claim the market is
	// closed — only that the data is missing right now.
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": macroUnknownFixture(true)})
	c, regime := ag.MacroCard(context.Background())

	if regime != "unknown" {
		t.Fatalf("regime: got %q, want unknown", regime)
	}
	if c.Verdict != "UNKNOWN — no tradfin data right now" {
		t.Errorf("verdict: got %q", c.Verdict)
	}
	if strings.Contains(strings.Join(c.Facts, "|"), "market closed") {
		t.Errorf("open window must not say 'market closed': %v", c.Facts)
	}
	if c.Emoji != emojiNeutral {
		t.Errorf("emoji: got %q, want neutral", c.Emoji)
	}
}

// TestMacroCardOldBackendMixedAllNull is the EXACT reported defect, replayed
// against an old backend payload (regime "mixed", every lamp null, composite
// null, split-sentence idea still attached): the bot must reclassify to
// unknown and refuse to render the "signals are split" knowledge claim.
func TestMacroCardOldBackendMixedAllNull(t *testing.T) {
	fixture := `{"regime":"mixed","composite":null,"tradfin_market_open":false,
	  "captured_at":"2026-08-18T06:00:00Z",
	  "lamps":[
	    {"key":"dxy","label":"Dollar (DXY)","value":null,"delta_pct":null,"status":""},
	    {"key":"rates","label":"US 10Y","value":null,"delta_pct":null,"status":""},
	    {"key":"vix","label":"VIX","value":null,"delta_pct":null,"status":""},
	    {"key":"spx","label":"S&P 500","value":null,"delta_pct":null,"status":""},
	    {"key":"gold","label":"Gold","value":null,"delta_pct":null,"status":""}],
	  "fng":{"value":61,"label":"Greed","ok":true},
	  "generated_idea":"Macro signals are split right now — no single regime is in control across the big-money markets."}`
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": fixture})
	c, regime := ag.MacroCard(context.Background())

	if regime != "unknown" {
		t.Fatalf("old-backend all-null mixed must reclassify to unknown, got %q", regime)
	}
	if !strings.HasPrefix(c.Verdict, "UNKNOWN") {
		t.Errorf("verdict must admit the unknown: %q", c.Verdict)
	}
	rendered := c.RenderHTML()
	for _, banned := range []string{"signals are split", "no single regime in control", "MIXED"} {
		if strings.Contains(rendered, banned) {
			t.Errorf("zero-input card rendered the knowledge claim %q:\n%s", banned, rendered)
		}
	}
}

// TestMacroCardMixedWithRealLampsKeepsClaim — regression guard: with REAL
// conflicting lamp values, "mixed" stays a legitimate verdict and the split
// sentence may render.
func TestMacroCardMixedWithRealLampsKeepsClaim(t *testing.T) {
	fixture := `{"regime":"mixed","composite":50,"tradfin_market_open":true,"tradfin_ok":true,
	  "captured_at":"2026-08-18T06:00:00Z",
	  "lamps":[
	    {"key":"dxy","label":"Dollar (DXY)","value":98.85,"ok":true,"delta_pct":-0.2,"status":"tailwind"},
	    {"key":"vix","label":"VIX","value":27.1,"ok":true,"delta_pct":null,"status":"headwind"},
	    {"key":"spx","label":"S&P 500","value":7580.1,"ok":true,"delta_pct":0.0,"status":"neutral"}],
	  "fng":{"value":61,"label":"Greed","ok":true},
	  "generated_idea":"Macro signals are split right now — no single regime is in control across the big-money markets."}`
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": fixture})
	c, regime := ag.MacroCard(context.Background())

	if regime != "mixed" {
		t.Fatalf("real lamps must keep mixed, got %q", regime)
	}
	if !strings.HasPrefix(c.Verdict, "MIXED") {
		t.Errorf("verdict: got %q, want the MIXED claim (real inputs)", c.Verdict)
	}
	joined := strings.Join(c.Facts, "|")
	if !strings.Contains(joined, "Lamps: 1 tailwind / 1 headwind / 1 neutral") {
		t.Errorf("lamp counts must render for real lamps: %v", c.Facts)
	}
	if !strings.Contains(joined, "signals are split") {
		t.Errorf("split sentence is legitimate WITH real lamps: %v", c.Facts)
	}
	if c.Confidence == nil || *c.Confidence != 50 {
		t.Errorf("confidence must flow from composite, got %v", c.Confidence)
	}
}

// TestHTTPMacroUnknownEnvelope: /agents/macro serves the unknown card as an
// honest 200 envelope (it is a verdict about data absence, not a 503 — the
// upstream responded fine).
func TestHTTPMacroUnknownEnvelope(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": macroUnknownFixture(false)})
	_, srv := newTestAPI(t, ag, true)

	status, _, body := httpGet(t, srv.URL+"/agents/macro")
	if status != 200 {
		t.Fatalf("status %d, want 200 (%s)", status, body)
	}
	var env testEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env.Agent != "Macro Agent" {
		t.Errorf("agent: %q", env.Agent)
	}
	if env.Verdict != "UNKNOWN — market closed, no tradfin data" {
		t.Errorf("verdict: %q", env.Verdict)
	}
	if env.Semaphore != "neutral" {
		t.Errorf("semaphore: %q, want neutral", env.Semaphore)
	}
	if env.Confidence != nil {
		t.Errorf("confidence: %v, want null", env.Confidence)
	}
	joined := strings.Join(env.Facts, "|")
	if !strings.Contains(joined, "no tradfin data right now (market closed)") {
		t.Errorf("facts must state the absence: %v", env.Facts)
	}
	if strings.Contains(joined, "signals are split") {
		t.Errorf("split sentence leaked into a zero-input envelope: %v", env.Facts)
	}
	if !strings.Contains(env.CardHTML, "UNKNOWN — market closed, no tradfin data") {
		t.Errorf("card_html must carry the unknown card:\n%s", env.CardHTML)
	}
}
