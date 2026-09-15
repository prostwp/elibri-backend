package demobot

// Content-honesty fixes from the 2026-09-15 live read of every card: each
// test pins a line that used to state something the data could not back.

import (
	"context"
	"strings"
	"testing"
	"time"
)

// Whale: the BTC feed is the public mempool — no exchange labels, so the
// backend's net flow is structurally $0 / "neutral". That is "not
// measurable", never "Flows balanced".
func TestWhaleCardUnlabeledFeedIsNotBalanced(t *testing.T) {
	ts := time.Now().UTC().Add(-30 * time.Minute).Format(time.RFC3339)
	fixture := `{"captured_at":"2026-09-14T22:58:00Z",
	  "flows":[{"asset":"BTC","net_flow_usd_24h":0,"direction":"neutral","tx_count_24h":48,"confidence":55,"partial":true}],
	  "transfers":[{"chain":"BTC","timestamp":"` + ts + `","asset":"BTC","amount_native":131.99,"amount_usd":10470000,"direction":"neutral","exchange":""}]}`
	ag := newStubBackend(t, map[string]string{"/api/v1/whale-flow": fixture})
	c := ag.WhaleCard(context.Background())

	if strings.Contains(c.Verdict, "balanced") {
		t.Errorf("unlabeled feed must not claim balance: %q", c.Verdict)
	}
	if c.Verdict != "48 large BTC transfers in 24h — exchange direction not measurable" {
		t.Errorf("verdict: %q", c.Verdict)
	}
	joined := strings.Join(c.Facts, "|")
	if strings.Contains(joined, "$0.00") || strings.Contains(joined, "neutral") || strings.Contains(joined, "unlabeled wallet") {
		t.Errorf("no fake $0 net flow, no meaningless direction words: %v", c.Facts)
	}
	if !strings.Contains(joined, "Net exchange flow: not measurable") {
		t.Errorf("the absence must be stated: %v", c.Facts)
	}
	if c.Confidence != nil {
		t.Errorf("no confidence behind an unmeasurable direction, got %d", *c.Confidence)
	}
	if env := cardEnvelope(c); !env.OK {
		t.Errorf("transfer activity is a real reading, ok must stay true: %+v", env)
	}
}

func TestWhaleCardUnlabeledFeedNoTransfers(t *testing.T) {
	fixture := `{"captured_at":"2026-09-14T22:58:00Z",
	  "flows":[{"asset":"BTC","net_flow_usd_24h":0,"direction":"neutral","tx_count_24h":0,"partial":true}],
	  "transfers":[]}`
	ag := newStubBackend(t, map[string]string{"/api/v1/whale-flow": fixture})
	if c := ag.WhaleCard(context.Background()); c.Verdict != "No large BTC transfers in 24h" {
		t.Errorf("verdict: %q", c.Verdict)
	}
}

// Trend: only a confirmed trend has a thesis that can break.
func TestTrendInvalidationFactPerState(t *testing.T) {
	up := trendInvalidationFact(trendUp, 74497, "below")
	if up != "Invalidation: below 74497 the structure is broken (1 ATR under the EMA cluster)" {
		t.Errorf("confirmed up: %q", up)
	}
	down := trendInvalidationFact(trendDown, 1.162, "above")
	if !strings.HasPrefix(down, "Invalidation: above 1.162") || !strings.Contains(down, "1 ATR over") {
		t.Errorf("confirmed down: %q", down)
	}
	for _, st := range []string{trendFlat, trendGrey, trendConflict} {
		got := trendInvalidationFact(st, 74497, "below")
		if strings.Contains(got, "structure is broken") || strings.HasPrefix(got, "Invalidation:") {
			t.Errorf("%s card must not claim a breakable structure: %q", st, got)
		}
		if !strings.Contains(got, "74497") {
			t.Errorf("%s: the reference level stays visible: %q", st, got)
		}
	}
}

// S/R: nearest level first on each side.
func TestSortSRByDistance(t *testing.T) {
	sup := []SRLevel{{Raw: 76407}, {Raw: 64033}, {Raw: 65228}}
	sortSRByDistance(sup, 78982)
	if sup[0].Raw != 76407 || sup[1].Raw != 65228 || sup[2].Raw != 64033 {
		t.Errorf("supports nearest-first: %v", sup)
	}
	res := []SRLevel{{Raw: 81376}, {Raw: 79346}, {Raw: 79950}}
	sortSRByDistance(res, 78982)
	if res[0].Raw != 79346 || res[1].Raw != 79950 || res[2].Raw != 81376 {
		t.Errorf("resistances nearest-first: %v", res)
	}
}

func TestMentionsWord(t *testing.T) {
	if mentionsWord(1) != "1 mention" || mentionsWord(2) != "2 mentions" || mentionsWord(0) != "0 mentions" {
		t.Errorf("pluralization: %q %q %q", mentionsWord(1), mentionsWord(2), mentionsWord(0))
	}
}

// Gold: a verdict embedded mid-sentence keeps its acronyms ("ADX", not "adx").
func TestLowerFirst(t *testing.T) {
	if got := lowerFirst("Flat — no trend to read (ADX 18.8 < 20)"); got != "flat — no trend to read (ADX 18.8 < 20)" {
		t.Errorf("lowerFirst: %q", got)
	}
	if got := lowerFirst("Confirmed UPTREND"); got != "confirmed UPTREND" {
		t.Errorf("lowerFirst keeps the rest: %q", got)
	}
	if lowerFirst("") != "" {
		t.Error("empty stays empty")
	}
}

// Gold and S/R name the same count the same way.
func TestTouchCountSameWordAsSR(t *testing.T) {
	if touchCount(1) != "1 swing pivot" || touchCount(4) != "4 swing pivots" {
		t.Errorf("touchCount: %q %q", touchCount(1), touchCount(4))
	}
}

// Momentum: a read carries its bar size so the overview can label mixed
// intervals (crypto 4h next to gold 1h).
func TestMomentumReadCarriesInterval(t *testing.T) {
	candles := srCardCycleCandles(250, func(int) float64 { return 100 })
	r, err := momentumReadFromCandles(btcSpec, candles)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if r.interval != btcSpec.Interval || r.interval == "" {
		t.Errorf("interval: got %q, want %q", r.interval, btcSpec.Interval)
	}
}

// Macro asset views: the composite is a score fact, the JSON confidence is
// null on both views (the global card is covered in cards_depth_test.go).
func TestMacroAssetViewsServeNullConfidence(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": goldBullFixture()})
	for _, asset := range []string{macroAssetBTC, macroAssetGold} {
		env := cardEnvelope(ag.MacroAssetCard(context.Background(), asset))
		if env.Confidence != nil {
			t.Errorf("%s view: confidence must be null, got %d", asset, *env.Confidence)
		}
	}
}

// How-it-works texts double as catalog descriptions on the web UI: no
// Telegram commands, no misstated ranking, no "works for any asset".
func TestHowTextsWebSafe(t *testing.T) {
	for key, txt := range howTexts {
		if len([]rune(txt)) > 200 {
			t.Errorf("%s: %d runes, Telegram alert cap is 200", key, len([]rune(txt)))
		}
		if strings.Contains(txt, "/momentum") || strings.Contains(txt, "/digest") {
			t.Errorf("%s: Telegram command in a web description: %q", key, txt)
		}
	}
	// The radar sorts by a five-part composite score, not by growth alone.
	if strings.Contains(howTexts[keyNews], "ranked by 48h mention growth") {
		t.Errorf("news how-text misstates the ranking: %q", howTexts[keyNews])
	}
	if strings.Contains(howTexts[keyVol], "breakout regime") {
		t.Errorf("vol how-text: ATR does not confirm a breakout: %q", howTexts[keyVol])
	}
	if strings.Contains(howTexts[keyRisk], "any asset") {
		t.Errorf("risk how-text overclaims: %q", howTexts[keyRisk])
	}
}
