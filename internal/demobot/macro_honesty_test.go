package demobot

// macro_honesty_test.go — the unknown-regime honesty contract (team-testing
// defect 2026-08): when the backend has ZERO real tradfin lamps, the bot must
// admit it — "UNKNOWN … no current tradfin data" with a neutral semaphore —
// instead of a confident "MIXED" built on nothing. Covers:
//   - the unknown card golden (scheduled weekend + open window wording; never
//     "market closed": the week window knows no holidays),
//   - the digest one-liner "⚪ Macro: unknown (no data)",
//   - version-skew reclassification (old backend: "mixed" + all-null lamps),
//   - the backend's generated_idea never rendering (the card words the rule
//     itself; an old backend's causal sentence must not leak through),
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
// idea blanked. F&G (crypto side) still live, with its own date.
func macroUnknownFixture(marketOpen bool) string {
	openStr := "false"
	if marketOpen {
		openStr = "true"
	}
	return `{"regime":"unknown","composite":null,"tradfin_market_open":` + openStr + `,"tradfin_ok":false,
	  "tradfin_as_of":"2026-08-14T20:55:00Z","captured_at":"2026-08-18T06:00:00Z",
	  "lamps":[
	    {"key":"dxy","label":"Dollar (DXY)","value":null,"ok":false,"delta_pct":null,"status":"","as_of":"2026-08-14T20:55:00Z"},
	    {"key":"rates","label":"US 10Y","value":null,"ok":false,"delta_pct":null,"status":"","as_of":""},
	    {"key":"vix","label":"VIX","value":null,"ok":false,"delta_pct":null,"status":"","as_of":"2026-08-14T20:55:00Z"},
	    {"key":"spx","label":"S&P 500","value":null,"ok":false,"delta_pct":null,"status":"","as_of":"2026-08-14T20:55:00Z"},
	    {"key":"gold","label":"Gold","value":null,"ok":false,"delta_pct":null,"status":"","as_of":""}],
	  "fng":{"value":61,"label":"Greed","ok":true,"as_of":"2026-08-18T00:00:00Z","fetched_at":"2026-08-18T05:55:00Z"},
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
	if c.Blocks != nil || c.Macro != nil {
		t.Errorf("no blocks / machine readout without a reading: %+v %+v", c.Blocks, c.Macro)
	}

	// Byte-exact card: neutral semaphore, an ADMISSION instead of a claim,
	// the weekend named as scheduled (never "market closed"), the last data
	// date, and the crypto F&G with its own date — outside the score.
	want := "⚪ <b>Macro Agent</b>\n" +
		"<b>UNKNOWN — no current tradfin data (scheduled tradfin weekend)</b>\n" +
		"• Lamps: no current tradfin data (scheduled tradfin weekend)\n" +
		"• Last tradfin data in the feed: Aug 14\n" +
		"• Crypto Fear &amp; Greed 61 (Greed), Aug 18 · separate index, not in the rule score\n" +
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
	// Inside the open window (data lag) the card must not mention the weekend
	// — only that the data is missing.
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": macroUnknownFixture(true)})
	c, regime := ag.MacroCard(context.Background())

	if regime != "unknown" {
		t.Fatalf("regime: got %q, want unknown", regime)
	}
	if c.Verdict != "UNKNOWN — no current tradfin data" {
		t.Errorf("verdict: got %q", c.Verdict)
	}
	joined := strings.Join(c.Facts, "|")
	if strings.Contains(joined, "weekend") || strings.Contains(joined, "closed") {
		t.Errorf("open window must not claim a closure: %v", c.Facts)
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
	// An old backend's F&G carries no time: the card says so instead of
	// presenting the value as current.
	if !strings.Contains(rendered, "update time unknown") {
		t.Errorf("F&G without a time must say so:\n%s", rendered)
	}
}

// TestMacroCardMixedWithRealLamps — with REAL conflicting lamp values "mixed"
// is a legitimate verdict, worded by the rule, and the backend's idea line is
// never rendered (an old backend still sends a "big-money" sentence).
func TestMacroCardMixedWithRealLamps(t *testing.T) {
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
	if c.Verdict != "MIXED — rule score 50/100 (risk-on above 65, risk-off below 35)" {
		t.Errorf("verdict: got %q", c.Verdict)
	}
	joined := strings.Join(c.Facts, "|")
	for _, want := range []string{
		"Positive for rule score: DXY -0.20% (fell) → +16.7",
		"Negative for rule score: VIX 27.10 (>25) → -16.7",
		"Mixed while the rule score is 35-65: above 65 reads risk-on, below 35 risk-off",
		// Two voting weights of 25 plus a neutral 25: every share is 33.3.
		"Rule score 50 = 50 + DXY +16.7 + VIX -16.7 · neutral: S&P 500",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("facts missing %q:\n%v", want, c.Facts)
		}
	}
	if strings.Contains(joined, "big-money") || strings.Contains(joined, "signals are split") {
		t.Errorf("the backend idea line must not render: %v", c.Facts)
	}
	// The composite is a rule score in the verdict — never a "Confidence" bar.
	if c.Confidence != nil {
		t.Errorf("composite must not render as confidence, got %v", *c.Confidence)
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
	if env.Verdict != "UNKNOWN — no current tradfin data (scheduled tradfin weekend)" {
		t.Errorf("verdict: %q", env.Verdict)
	}
	if env.Semaphore != "neutral" {
		t.Errorf("semaphore: %q, want neutral", env.Semaphore)
	}
	// Machine-readable twin of the verdict: not a real reading, and the WHY
	// stays the enum market_closed outside the clock week (the open-window
	// twin asserts no_data in httpapi_status_test.go).
	if env.OK {
		t.Error("unknown regime must serve ok=false")
	}
	if env.Reason == nil || *env.Reason != "market_closed" {
		t.Errorf("reason: %v, want market_closed", env.Reason)
	}
	if env.Confidence != nil {
		t.Errorf("confidence: %v, want null", env.Confidence)
	}
	joined := strings.Join(env.Facts, "|")
	if !strings.Contains(joined, "no current tradfin data (scheduled tradfin weekend)") {
		t.Errorf("facts must state the absence: %v", env.Facts)
	}
	if strings.Contains(joined, "signals are split") {
		t.Errorf("split sentence leaked into a zero-input envelope: %v", env.Facts)
	}
	if !strings.Contains(env.CardHTML, "UNKNOWN — no current tradfin data (scheduled tradfin weekend)") {
		t.Errorf("card_html must carry the unknown card:\n%s", env.CardHTML)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatal(err)
	}
	if _, ok := raw["macro"]; ok {
		t.Errorf("no machine readout on a card without a reading: %s", raw["macro"])
	}
	if _, ok := raw["blocks"]; ok {
		t.Errorf("no blocks on a card without a reading: %s", raw["blocks"])
	}
}
