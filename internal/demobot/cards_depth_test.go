package demobot

// cards_depth_test.go — tests for the card-depth upgrade: /news narrative
// radar, /macro AI read, /whale top-3 + baseline, /funding skew + nearest
// magnet zone, /momentum RS + volume, /sr nearest level, /fx day range.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// newStubBackend serves canned JSON per path; everything else 404s.
func newStubBackend(t *testing.T, routes map[string]string) *Agents {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if body, ok := routes[r.URL.Path]; ok {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(body))
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)
	return NewAgents(NewBackendClient(srv.URL))
}

// stubBinanceKlines serves `bars` synthetic closed 4h candles for any symbol,
// plus one trailing row — the bar still forming, as every real klines answer
// ends; the demobot never reads it. Prices trend gently up; volumes come from
// the volume func (the trailing row is index `bars`).
func stubBinanceKlines(t *testing.T, bars int, volume func(i int) float64) {
	t.Helper()
	start := time.Now().Unix() - int64(bars+2)*14400 // every bar closed
	rows := make([][]any, bars+1)
	price := 60000.0
	for i := range rows {
		price += 50
		rows[i] = []any{
			float64(start+int64(i)*14400) * 1000,
			fmt.Sprintf("%f", price-25), fmt.Sprintf("%f", price+100),
			fmt.Sprintf("%f", price-100), fmt.Sprintf("%f", price),
			fmt.Sprintf("%f", volume(i)),
		}
	}
	body, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	orig := binanceKlinesBase
	binanceKlinesBase = srv.URL
	t.Cleanup(func() { binanceKlinesBase = orig })
}

func deadYahoo(t *testing.T) {
	t.Helper()
	orig := yahooChartBase
	yahooChartBase = "http://127.0.0.1:1/v8/finance/chart/"
	t.Cleanup(func() { yahooChartBase = orig })
}

// ── /news: Narrative Radar ───────────────────────────────────────────────────

const narrativesFixture = `{
  "captured_at": "2026-08-18T06:00:00Z",
  "narratives": [
    {"narrative":"ai-agents","trend_score":84,"stage":"trending","sentiment_label":"bull","mention_count_24h":412,"confidence":71,
     "generated_idea":"Mentions of AI-agent tokens tripled in 24h with a bullish skew; this stage has historically preceded elevated volatility."},
    {"narrative":"rwa","trend_score":61,"stage":"early","sentiment_label":"neutral","mention_count_24h":120,"confidence":55},
    {"narrative":"btc-etf","trend_score":40,"stage":"mainstream","sentiment_label":"bear","mention_count_24h":300,"confidence":80},
    {"narrative":"fourth-item","trend_score":10,"stage":"declining","sentiment_label":"neutral","mention_count_24h":9,"confidence":10}
  ]
}`

func TestNewsCardTop3AndIdea(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/narratives": narrativesFixture})
	c := ag.NewsCard(context.Background())

	if c.Offline {
		t.Fatal("live narratives must not be offline")
	}
	if c.Emoji != emojiBull {
		t.Errorf("bull top narrative → 🟢, got %q", c.Emoji)
	}
	if !strings.Contains(c.Verdict, "ai-agents") || !strings.Contains(c.Verdict, "activity score 84/100") {
		t.Errorf("verdict must carry name and activity score, got %q", c.Verdict)
	}
	var ranked []string
	for _, f := range c.Facts {
		if strings.Contains(f, " matched items") && strings.Contains(f, "activity score") {
			ranked = append(ranked, f)
		}
	}
	if len(ranked) != 3 {
		t.Fatalf("want exactly 3 ranked narrative facts, got %d: %v", len(ranked), c.Facts)
	}
	if !strings.Contains(ranked[0], "1. ai-agents") || !strings.Contains(ranked[0], "412 matched items") {
		t.Errorf("first line wrong: %q", ranked[0])
	}
	if !strings.Contains(ranked[2], "3. Bitcoin ETFs and ETF issuers") {
		t.Errorf("third line wrong: %q", ranked[2])
	}
	if strings.Contains(strings.Join(c.Facts, "|"), "fourth-item") {
		t.Error("4th narrative must not render")
	}
	if !strings.HasPrefix(c.AIHTML, "<b>AI comment:</b> <i>") || !strings.Contains(c.AIHTML, "elevated volatility") {
		t.Errorf("AI comment block missing/wrong: %q", c.AIHTML)
	}
	if c.Confidence == nil || *c.Confidence != 71 {
		t.Errorf("top data quality must flow through, got %v", c.Confidence)
	}
	if !strings.Contains(c.SourceNote, "last 24h") || strings.Contains(c.SourceNote, "48h") {
		t.Errorf("footer must name the 24h count window, got %q", c.SourceNote)
	}
	rendered := c.RenderHTML()
	if !strings.Contains(rendered, "<b>AI comment:</b>") {
		t.Errorf("rendered card must include the AI comment block:\n%s", rendered)
	}
	if !strings.Contains(rendered, "2026-08-18 06:00 UTC") {
		t.Errorf("footer must use captured_at:\n%s", rendered)
	}
}

func TestNewsCardEscapesFeedText(t *testing.T) {
	fixture := `{"captured_at":"2026-08-18T06:00:00Z","narratives":[
	  {"narrative":"<b>evil</b>","trend_score":50,"stage":"early","sentiment_label":"neutral","mention_count_24h":5,"confidence":10,
	   "generated_idea":"<script>alert(1)</script>"}]}`
	ag := newStubBackend(t, map[string]string{"/api/v1/narratives": fixture})
	got := ag.NewsCard(context.Background()).RenderHTML()
	for _, bad := range []string{"<b>evil</b>", "<script>"} {
		if strings.Contains(got, bad) {
			t.Errorf("unescaped feed text %q leaked:\n%s", bad, got)
		}
	}
	if !strings.Contains(got, "&lt;script&gt;") {
		t.Errorf("idea must render escaped:\n%s", got)
	}
}

// Batch-2 silence threshold: below newsMinMentions 24h matched items on the
// leader, scores are noise and must not present as findings — the card says
// "below threshold" (never "warming up": the radar's start time is not
// served) and lists themes as name + matched items only, unnumbered.
func TestNewsCardBelowMentionThreshold(t *testing.T) {
	fixture := `{"captured_at":"2026-08-18T06:00:00Z","narratives":[
	  {"narrative":"zk","trend_score":72,"stage":"early","sentiment_label":"bull","mention_count_24h":3,"confidence":61,
	   "generated_idea":"zk chatter tripled off a tiny base."},
	  {"narrative":"rwa","trend_score":50,"stage":"early","sentiment_label":"neutral","mention_count_24h":2,"confidence":40}]}`
	ag := newStubBackend(t, map[string]string{"/api/v1/narratives": fixture})
	c := ag.NewsCard(context.Background())

	if c.Offline {
		t.Fatal("thin data is an honest 200 card, not an offline state")
	}
	if c.Emoji != emojiNeutral {
		t.Errorf("emoji: got %q, want neutral (no finding to color)", c.Emoji)
	}
	wantVerdict := "Below threshold: Zero-knowledge (ZK) networks leads by activity score with 3 matched items in 24h; 5 needed to score"
	if len([]rune(wantVerdict)) > narrativeFactMaxRunes {
		wantVerdict = "Below threshold: Zero-knowledge (ZK) networks leads with 3 matched items/24h; 5 needed"
	}
	if c.Verdict != wantVerdict {
		t.Errorf("verdict:\ngot:  %q\nwant: %q", c.Verdict, wantVerdict)
	}
	if c.Short != "below threshold" {
		t.Errorf("short: got %q", c.Short)
	}
	// Facts: name + matched items ONLY — no scores, no stages, no numbers.
	wantFacts := []string{narrativeLineOrderBelow, "Zero-knowledge (ZK) networks — 3 matched items",
		"Real-world assets (RWA) — 2 matched items", narrativeLineSources, narrativeLineMatch}
	if len(c.Facts) != len(wantFacts) {
		t.Fatalf("facts: got %v, want %v", c.Facts, wantFacts)
	}
	for i, want := range wantFacts {
		if c.Facts[i] != want {
			t.Errorf("fact[%d]: got %q, want %q", i, c.Facts[i], want)
		}
	}
	joined := strings.Join(c.Facts[1:3], "|")
	for _, banned := range []string{"score", "trending", "early", "1.", "2."} {
		if strings.Contains(joined, banned) {
			t.Errorf("score emphasis %q leaked into below-threshold facts: %v", banned, c.Facts)
		}
	}
	// No confidence bar, no AI idea: nothing below the threshold is a finding.
	if c.Confidence != nil {
		t.Errorf("confidence must be absent, got %v", c.Confidence)
	}
	if c.AIHTML != "" {
		t.Errorf("AI idea must not render on thin data, got %q", c.AIHTML)
	}
	// Singular wording at exactly one matched item.
	one := `{"captured_at":"2026-08-18T06:00:00Z","narratives":[
	  {"narrative":"zk","trend_score":9,"stage":"early","sentiment_label":"neutral","mention_count_24h":1,"confidence":5}]}`
	ag2 := newStubBackend(t, map[string]string{"/api/v1/narratives": one})
	if v := ag2.NewsCard(context.Background()).Verdict; !strings.Contains(v, "1 matched item") || strings.Contains(v, "1 matched items") {
		t.Errorf("singular wording: got %q", v)
	}
	// At the threshold (5) the scored card returns.
	at := `{"captured_at":"2026-08-18T06:00:00Z","narratives":[
	  {"narrative":"zk","trend_score":72,"stage":"early","sentiment_label":"bull","mention_count_24h":5,"confidence":61}]}`
	ag3 := newStubBackend(t, map[string]string{"/api/v1/narratives": at})
	if v := ag3.NewsCard(context.Background()).Verdict; !strings.Contains(v, "activity score 72/100") {
		t.Errorf("at-threshold card must score again: %q", v)
	}
}

// The digest's narrative extra obeys the same threshold: no scored 📖 line
// (and no AI-payload narrative) off a thin mention base.
func TestDigestNarrativeExtraBelowThreshold(t *testing.T) {
	stubExternalBases(t)
	thin := `{"captured_at":"2026-08-18T06:00:00Z","narratives":[
	  {"narrative":"zk","trend_score":72,"stage":"early","sentiment_label":"bull","mention_count_24h":3,"confidence":61}]}`
	ag := newStubBackend(t, map[string]string{"/api/v1/narratives": thin})
	g := ag.gather(context.Background())
	if g.topNarr != nil {
		t.Errorf("thin narrative must not enter the AI payload: %+v", g.topNarr)
	}
	for _, ex := range g.extras {
		if strings.Contains(ex, "📖") {
			t.Errorf("thin narrative rendered a scored digest extra: %q", ex)
		}
	}

	rich := `{"captured_at":"2026-08-18T06:00:00Z","narratives":[
	  {"narrative":"ai-agents","trend_score":84,"stage":"trending","sentiment_label":"bull","mention_count_24h":412,"confidence":71}]}`
	ag2 := newStubBackend(t, map[string]string{"/api/v1/narratives": rich})
	g2 := ag2.gather(context.Background())
	if g2.topNarr == nil {
		t.Fatal("rich narrative must flow into the payload")
	}
	found := false
	for _, ex := range g2.extras {
		if strings.Contains(ex, "📖") && strings.Contains(ex, "ai-agents") {
			found = true
		}
	}
	if !found {
		t.Errorf("rich narrative must render the digest extra: %v", g2.extras)
	}
}

func TestNewsCardEmptyAndOffline(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/narratives": `{"captured_at":"","narratives":[]}`})
	c := ag.NewsCard(context.Background())
	if !c.Offline || !strings.Contains(c.Verdict, "No radar snapshots") {
		t.Errorf("empty store must be an honest no-snapshots card, got %+v", c)
	}
	if c.AIHTML != "" {
		t.Error("no idea block without narratives")
	}

	dead := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	off := dead.NewsCard(context.Background())
	if !off.Offline || !strings.Contains(off.Verdict, "offline") {
		t.Errorf("dead backend must be an offline card, got %+v", off)
	}
}

func TestNewsRoutingAndKeyboard(t *testing.T) {
	stubExternalBases(t)
	bot := newBotWithClient(newTGClientWithBase("T", "http://127.0.0.1:1/botT"), NewAgents(NewBackendClient("http://127.0.0.1:1")))
	text, kb := bot.buildReply(context.Background(), keyNews, nil)
	if !strings.Contains(text, "Narrative Radar") {
		t.Errorf("/news must render the Narrative Radar card, got:\n%s", text)
	}
	if kb == nil {
		t.Fatal("/news needs the standard card keyboard")
	}
	var haveRefresh, haveHow bool
	for _, row := range kb.InlineKeyboard {
		for _, b := range row {
			if b.CallbackData == "r|news" {
				haveRefresh = true
			}
			if b.CallbackData == "h|news" {
				haveHow = true
			}
		}
	}
	if !haveRefresh || !haveHow {
		t.Errorf("keyboard must carry r|news and h|news, got %+v", kb.InlineKeyboard)
	}
	if howTexts[keyNews] == "" {
		t.Error("keyNews needs a How-it-works text")
	}
	if len(howTexts[keyNews]) > 200 {
		t.Errorf("how text exceeds the 200-char alert cap: %d", len(howTexts[keyNews]))
	}
}

// ── /macro: no AI mood read, composite as a score ────────────────────────────

const macroFixture = `{"regime":"risk_on","composite":62,"tradfin_market_open":true,
  "captured_at":"2026-08-18T06:00:00Z","lamps":[],"fng":{"value":63,"label":"Greed","ok":true},"generated_idea":""}`

// The mood read quotes a different Fear & Greed feed than the card's own F&G
// line and talks about news themes — it must not appear on the macro card
// even when the endpoint answers. The composite renders as a labeled score,
// never as "Confidence: N%".
const macroScoreFixture = `{"regime":"risk_on","composite":72,"tradfin_market_open":true,"tradfin_ok":true,
  "captured_at":"2026-08-18T06:00:00Z",
  "lamps":[
    {"key":"dxy","label":"Dollar (DXY)","value":98.85,"ok":true,"delta_pct":-0.2,"status":"tailwind"},
    {"key":"vix","label":"VIX","value":14.1,"ok":true,"delta_pct":-1.0,"status":"tailwind"},
    {"key":"spx","label":"S&P 500","value":7580.1,"ok":true,"delta_pct":0.4,"status":"tailwind"}],
  "fng":{"value":63,"label":"Greed","ok":true},"generated_idea":""}`

func TestMacroCardNoMoodReadAndScoreNotConfidence(t *testing.T) {
	ag := newStubBackend(t, map[string]string{
		"/api/v1/macro":            macroScoreFixture,
		"/api/v1/market/mood-read": `{"read":"The crypto market is in a greedy posture with a 70/100 reading.","source":"alphavizor-ai"}`,
	})
	c, regime := ag.MacroCard(context.Background())
	if regime != "risk_on" {
		t.Fatalf("regime: got %q", regime)
	}
	if c.AIHTML != "" || strings.Contains(c.RenderHTML(), "70/100") {
		t.Errorf("mood read must not render on the macro card: %q", c.AIHTML)
	}
	if c.Confidence != nil || strings.Contains(c.RenderHTML(), "Confidence:") {
		t.Errorf("composite is a score, not a confidence:\n%s", c.RenderHTML())
	}
	// The score rides in the verdict as a "rule score" with its bands.
	if c.Verdict != "RISK-ON — rule score 72/100 (risk-on above 65, risk-off below 35)" {
		t.Errorf("verdict: %q", c.Verdict)
	}
	// This fixture's three positive lamps add up to 100, not the served 72 —
	// a version skew. The card must then print no contribution numbers and no
	// breakdown that would not add up to its own headline.
	joined := strings.Join(c.Facts, "|")
	if strings.Contains(joined, "→ +") || strings.Contains(joined, "Rule score 72") {
		t.Errorf("contributions printed although they do not reproduce the score: %v", c.Facts)
	}
	if c.Deviation != 44 {
		t.Errorf("priority deviation must still flow from the composite: got %d, want 44", c.Deviation)
	}
}

// A score needs real lamps behind it: a payload with no lamps must not print a
// rule score under a regime the lamps never voted for — and the regime itself
// is reclassified to unknown (a regime is a knowledge claim).
func TestMacroCardNoScoreWithoutRealLamps(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/macro": macroFixture})
	c, regime := ag.MacroCard(context.Background())
	if regime != "unknown" || !strings.HasPrefix(c.Verdict, "UNKNOWN") {
		t.Errorf("zero real lamps must read unknown: regime %q verdict %q", regime, c.Verdict)
	}
	if strings.Contains(strings.ToLower(c.Verdict+strings.Join(c.Facts, "|")), "rule score 62") {
		t.Errorf("no score without real lamps: %q %v", c.Verdict, c.Facts)
	}
}

func TestMacroCardAIReadOmittedSilently(t *testing.T) {
	// Empty read → no block.
	ag := newStubBackend(t, map[string]string{
		"/api/v1/macro":            macroFixture,
		"/api/v1/market/mood-read": `{"read":"","source":"alphavizor-ai"}`,
	})
	if c, _ := ag.MacroCard(context.Background()); c.AIHTML != "" {
		t.Errorf("empty mood read must not render a block, got %q", c.AIHTML)
	}
	// Endpoint missing (404) → macro still renders, no block.
	ag2 := newStubBackend(t, map[string]string{"/api/v1/macro": macroScoreFixture})
	c, _ := ag2.MacroCard(context.Background())
	if c.AIHTML != "" {
		t.Errorf("mood-read failure must not render a block, got %q", c.AIHTML)
	}
	if !strings.Contains(c.Verdict, "RISK-ON") {
		t.Errorf("macro card must survive a mood-read failure, got %q", c.Verdict)
	}
}

// ── /whale: top-3 transfers + baseline sentence ──────────────────────────────

func whaleFixture(withBaseline bool) string {
	ts := func(minAgo int) string {
		return time.Now().UTC().Add(-time.Duration(minAgo) * time.Minute).Format(time.RFC3339)
	}
	baseline := ""
	if withBaseline {
		baseline = `"net_flow_prev_24h":-6200000,"flow_pct":196.8,`
	}
	// The snapshot is dated now and every transfer sits inside its 24h: the
	// top-3 window is (captured_at−24h, captured_at], so a transfer stamped
	// after captured_at would not show.
	return `{"captured_at":"` + ts(0) + `",
	  "flows":[{"asset":"BTC","net_flow_usd_24h":-18400000,` + baseline + `"direction":"outflow","tx_count_24h":37,"confidence":64,"partial":true}],
	  "transfers":[
	    {"chain":"BTC","timestamp":"` + ts(30) + `","asset":"BTC","amount_native":150.5,"amount_usd":17000000,"direction":"outflow","exchange":"Binance"},
	    {"chain":"BTC","timestamp":"` + ts(60) + `","asset":"BTC","amount_native":90.1,"amount_usd":10000000,"direction":"inflow","exchange":"Coinbase"},
	    {"chain":"BTC","timestamp":"` + ts(90) + `","asset":"BTC","amount_native":45.0,"amount_usd":5000000,"direction":"outflow","exchange":""},
	    {"chain":"BTC","timestamp":"` + ts(120) + `","asset":"BTC","amount_native":20.0,"amount_usd":2000000,"direction":"outflow","exchange":"Kraken"}
	  ]}`
}

func TestWhaleCardTopThreeAndBaseline(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/whale-flow": whaleFixture(true)})
	c := ag.WhaleCard(context.Background())

	var transfers []string
	for _, f := range c.Facts {
		if strings.Contains(f, "BTC ≈") {
			transfers = append(transfers, f)
		}
	}
	if len(transfers) != 3 {
		t.Fatalf("want top-3 transfers, got %d: %v", len(transfers), c.Facts)
	}
	if !strings.Contains(transfers[0], "$17.00M") || !strings.Contains(transfers[0], "Binance") {
		t.Errorf("biggest transfer first: %q", transfers[0])
	}
	if !strings.HasSuffix(transfers[2], " UTC, not the block time") || strings.Contains(transfers[2], "outflow") ||
		strings.Contains(transfers[2], " · from ") || strings.Contains(transfers[2], " · to ") {
		t.Errorf("unlabeled transfer: detection time only, no direction word without an exchange: %q", transfers[2])
	}
	if strings.Contains(strings.Join(c.Facts, "|"), "$2.00M") {
		t.Error("4th transfer must not render")
	}
	var baseline string
	for _, f := range c.Facts {
		if strings.Contains(f, "Prior 24h net flow") {
			baseline = f
		}
	}
	if baseline == "" {
		t.Fatalf("baseline sentence missing: %v", c.Facts)
	}
	if !strings.Contains(baseline, "-$6.20M") || !strings.Contains(baseline, "+197%") {
		t.Errorf("baseline wording: %q", baseline)
	}
}

func TestWhaleCardNoBaselineNoSentence(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/whale-flow": whaleFixture(false)})
	c := ag.WhaleCard(context.Background())
	for _, f := range c.Facts {
		if strings.Contains(f, "Prior 24h") {
			t.Errorf("no baseline in payload → no baseline sentence, got %q", f)
		}
	}
}

// ── /funding: liquidation skew + the largest observed band ───────────────────

func TestFundingCardSkewAndZoneRendering(t *testing.T) {
	stubExternalBases(t) // funding rates + klines dead → rates-offline path, no mark price
	now := time.Now().UTC().Format(time.RFC3339)
	fixture := `{"captured_at":"` + now + `","feed":[
	    {"symbol":"BTCUSDT","side":"long_liq","qty":1,"price":118000,"usd_value":90000,"ts":"` + now + `"},
	    {"symbol":"ETHUSDT","side":"short_liq","qty":10,"price":3400,"usd_value":30000,"ts":"` + now + `"}],
	  "zones":[
	    {"symbol":"BTCUSDT","price_band":"118200-118250","total_usd":412000,"count":9,"side":"long_liq"},
	    {"symbol":"BTCUSDT","price_band":"90000-90050","total_usd":100000,"count":3,"side":"long_liq"}]}`
	ag := newStubBackend(t, map[string]string{"/api/v1/funding/liquidations": fixture})
	c := ag.FundingCard(context.Background())

	joined := strings.Join(c.Facts, "|")
	if !strings.Contains(joined, "Liquidations, last 1h: 2 events · $120.0K · long liqs $90.0K (75%) · short liqs $30.0K") {
		t.Errorf("liquidation line missing/wrong: %v", c.Facts)
	}
	// The largest BTC band, with or without a mark price (fundingClusterZone).
	if !strings.Contains(joined, "Observed liquidation cluster, last 1h: BTCUSDT 118200-118250 · $412.0K · 9 events") {
		t.Errorf("cluster line missing/wrong: %v", c.Facts)
	}
	if !strings.Contains(joined, "Coverage: 0/5 Binance majors · funding-rate source offline right now") {
		t.Errorf("rates-offline coverage line missing: %v", c.Facts)
	}
}

// ── /momentum: backend RS + volume line ──────────────────────────────────────

func TestMomentumCardRSAndVolume(t *testing.T) {
	stubBinanceKlines(t, 250, func(i int) float64 {
		if i == 249 {
			return 150
		}
		return 100
	})
	deadYahoo(t)
	ag := newStubBackend(t, map[string]string{
		"/api/v1/market/momentum": `{"baseline":"BTC","items":{"ETH":{"rs_7d":-2.4,"rs_30d":5.1}}}`,
	})
	c := ag.MomentumCard(context.Background())
	joined := strings.Join(c.Facts, "|")
	// 2026-09-15: both lines are labelled context, outside the reading; the
	// RS line says it is a return gap, not ETH's own move.
	if !strings.Contains(joined, "Context, not part of the reading: ETH return minus BTC return incl. today, 7d -2.4 pp · 30d +5.1 pp") {
		t.Errorf("RS 7d/30d line missing: %v", c.Facts)
	}
	if !strings.Contains(joined, "Context, not part of the reading: BTC 4h volume 1.50× its 20-bar average") {
		t.Errorf("volume line missing: %v", c.Facts)
	}
	// Batch-2 language + thresholds: analytical verdict words only, and one
	// compact rule line documents what flips them.
	if strings.Contains(c.Verdict, "BUY") || strings.Contains(c.Verdict, "SELL") {
		t.Errorf("advice-words leaked into the momentum verdict: %q", c.Verdict)
	}
	// The header is a counter (gold is dead here), coloured by it.
	if c.Verdict != "2 bullish (BTC, ETH) · 0 bearish · 0 not confirmed · 1 unavailable · 4h" || c.Emoji != emojiBull {
		t.Errorf("uptrending stub must count two bullish reads: %q %q", c.Verdict, c.Emoji)
	}
	if !strings.Contains(joined, "BTC · 4h: bullish — RSI and MACD agree") {
		t.Errorf("per-asset lines must use analytical words: %v", c.Facts)
	}
	if !strings.Contains(joined, momentumRuleLine) {
		t.Errorf("driving-threshold rule line missing: %v", c.Facts)
	}
}

// ── /trend: invalidation level + ADX threshold beside the number ─────────────

func TestTrendCardInvalidationAndADXThreshold(t *testing.T) {
	stubBinanceKlines(t, 250, func(i int) float64 { return 100 })
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.TrendCard(context.Background(), btcSpec)

	if c.Verdict != "Confirmed UPTREND · 4h" {
		t.Fatalf("rising stub must confirm an uptrend on 4h, got %q", c.Verdict)
	}
	joined := strings.Join(c.Facts, "|")
	// ADX drives the verdict → its confirm threshold rides beside the number,
	// written the way the rule works: it confirms AT 25 (≥), not above it.
	if !strings.Contains(joined, "≥ 25 ✓") || strings.Contains(joined, "above 25") {
		t.Errorf("ADX threshold must read ≥ 25: %v", c.Facts)
	}
	// The invalidation line: exact wording (closed candle of the agent's
	// timeframe), a plausible integer level for a BTC-scale price.
	var invLine string
	for _, f := range c.Facts {
		if strings.HasPrefix(f, "Invalidated by a closed 4h candle below ") {
			invLine = f
		}
	}
	if invLine == "" {
		t.Fatalf("invalidation line missing: %v", c.Facts)
	}
	if !strings.HasSuffix(invLine, ", 1 ATR under the EMA cluster)") {
		t.Errorf("invalidation wording: %q", invLine)
	}
	var level float64
	if _, err := fmt.Sscanf(invLine, "Invalidated by a closed 4h candle below %f", &level); err != nil {
		t.Fatalf("cannot parse level from %q: %v", invLine, err)
	}
	// The stub trends 60050→72500 with ±100 wicks; the level must sit below
	// the last close but in the same price region (EMA cluster − 1 ATR).
	if level <= 60000 || level >= 72500 {
		t.Errorf("invalidation level %v out of the series' price region", level)
	}
}

// ── /sr: strength threshold beside the touch counts ──────────────────────────

func TestSRCardStrengthThreshold(t *testing.T) {
	// A swinging fixture — a monotone series now honestly degrades to
	// insufficient_history (zero swing structure), so the method lines need a
	// series with real levels.
	stubBinanceCandles(t, srCardCycleCandles(250, func(int) float64 { return 100 }))
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.SRCard(context.Background(), btcSpec)

	// 2026-09-15: the method moved off the card body into the how-it-works
	// text (≤200 chars, also the catalog description); the card keeps one
	// short window + closed-candle line, and each level names its class.
	found := false
	for _, f := range c.Facts {
		if strings.HasPrefix(f, "Window: ") && strings.Contains(f, "closed 4h candles") && strings.Contains(f, "0.25 ATR") {
			found = true
		}
	}
	if !found {
		t.Errorf("window/closed-candle line missing: %v", c.Facts)
	}
	if !strings.Contains(c.HowItWorks, "7+ pivots = established") {
		t.Errorf("how-it-works must state the established threshold: %q", c.HowItWorks)
	}
	if n := len([]rune(c.HowItWorks)); n > 200 {
		t.Errorf("how-it-works is %d chars, over the 200-char alert cap", n)
	}
}

func TestMomentumAssetCardVolume(t *testing.T) {
	stubBinanceKlines(t, 250, func(i int) float64 {
		if i == 249 {
			return 50
		}
		return 100
	})
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.MomentumAssetCard(context.Background(), btcSpec)
	if !strings.Contains(strings.Join(c.Facts, "|"), "Context, not part of the reading: volume 0.50× its 20-bar average (4h)") {
		t.Errorf("asset card volume line missing: %v", c.Facts)
	}
}

func TestVolRatio20(t *testing.T) {
	mk := func(vols ...float64) []types.OHLCVCandle {
		out := make([]types.OHLCVCandle, len(vols))
		for i, v := range vols {
			out[i] = types.OHLCVCandle{Volume: v}
		}
		return out
	}
	if _, ok := volRatio20(mk(1, 2, 3)); ok {
		t.Error("under 21 bars must not produce a ratio")
	}
	vols := make([]float64, 21)
	for i := range vols {
		vols[i] = 100
	}
	vols[20] = 250
	if r, ok := volRatio20(mk(vols...)); !ok || !almostEqual(r, 2.5, 1e-9) {
		t.Errorf("ratio: got %v/%v, want 2.5", r, ok)
	}
	zero := make([]float64, 21) // dead baseline → no ratio, never Inf
	if _, ok := volRatio20(mk(zero...)); ok {
		t.Error("zero baseline must not produce a ratio")
	}
}

// ── /sr: nearest-shown-level headline ────────────────────────────────────────

// The headline names the nearest SHOWN level (never "S1/R1": the card shows
// the three strongest per side, so it is the nearest among those only).
func TestSRHeadlineNearestShown(t *testing.T) {
	sup := []SRLevel{{Level: 61200, Raw: 61200, Touches: 4}, {Level: 60000, Raw: 60000, Touches: 2}}
	res := []SRLevel{{Level: 63000, Raw: 63000, Touches: 3}}
	c := srCardFrom(btcSpec, sup, res, 62000, 249, time.Time{}, nil)
	if want := "Price 62000 — 1.3% above the nearest shown support 61200 (candidate, 4 pivots)"; c.Verdict != want {
		t.Errorf("verdict: got %q, want %q", c.Verdict, want)
	}
	c = srCardFrom(btcSpec, sup, res, 62900, 249, time.Time{}, nil)
	if !strings.Contains(c.Verdict, "0.2% below the nearest shown resistance 63000") {
		t.Errorf("resistance-nearest: got %q", c.Verdict)
	}
	// FX keeps pip precision.
	fx := assetTable["eurusd"]
	c = srCardFrom(fx, []SRLevel{{Level: 1, Raw: 1.1583, Touches: 3}}, nil, 1.1601, 503, time.Time{}, nil)
	if !strings.Contains(c.Verdict, "support 1.1583") || !strings.Contains(c.Verdict, "Price 1.1601") {
		t.Errorf("fx nearest: got %q", c.Verdict)
	}
}

// ── /fx: day-range position ──────────────────────────────────────────────────

func TestDayRange(t *testing.T) {
	mk := func(n int, hi, lo, lastClose float64) []types.OHLCVCandle {
		base := time.Now().Unix() - int64(n)*3600
		out := make([]types.OHLCVCandle, n)
		for i := range out {
			out[i] = types.OHLCVCandle{Time: base + int64(i)*3600, High: hi, Low: lo, Close: (hi + lo) / 2}
		}
		out[n-1].Close = lastClose
		return out
	}
	if pos, ok := dayRange(mk(30, 1.20, 1.10, 1.19)); !ok || !almostEqual(pos, 0.9, 1e-9) {
		t.Errorf("near-high pos: got %v/%v, want 0.9", pos, ok)
	}
	if pos, ok := dayRange(mk(30, 1.20, 1.10, 1.11)); !ok || !almostEqual(pos, 0.1, 1e-9) {
		t.Errorf("near-low pos: got %v/%v, want 0.1", pos, ok)
	}
	if _, ok := dayRange(mk(5, 1.15, 1.15, 1.15)); ok {
		t.Error("degenerate range must not produce a position")
	}
	if _, ok := dayRange(nil); ok {
		t.Error("empty series must not produce a position")
	}
	// Only the trailing 24h counts: an old spike outside the window is ignored.
	candles := mk(30, 1.20, 1.10, 1.19)
	candles[0].High = 9.99
	candles[0].Time = candles[len(candles)-1].Time - 90000 // > 24h before the last bar
	if pos, ok := dayRange(candles); !ok || pos < 0.85 {
		t.Errorf("stale spike leaked into the range: pos %v ok %v", pos, ok)
	}
}

// FX stage 1: the range position is printed as a number (the same dayRange
// value that used to be bucketed into near day high / mid-range / near day
// low), and the line drops it when the range is degenerate.
func TestDayRangePositionOnFXLine(t *testing.T) {
	cases := map[float64]string{0.95: "95%", 0.8: "80%", 0.5: "50%", 0.2: "20%", 0.0: "0%"}
	for pos, want := range cases {
		r := fxOK("eurusd", 1.1, "up", 58.3, 0.24, pos, fxAt(15, 8))
		if got := fxMarketLine(r, fxNow); got != "EURUSD 1.1000 · 24h +0.24% · "+want+" of the 24h range" {
			t.Errorf("pos %v: got %q", pos, got)
		}
	}
	noRange := fxOK("eurusd", 1.1, "up", 58.3, 0.24, 0, fxAt(15, 8))
	noRange.HasRange = false
	if got := fxMarketLine(noRange, fxNow); got != "EURUSD 1.1000 · 24h +0.24%" {
		t.Errorf("without range: %q", got)
	}
}
