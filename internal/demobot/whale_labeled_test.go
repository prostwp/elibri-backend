package demobot

// whale_labeled_test.go — Whale stage 2 (2026-09-16): the card reads the
// LABELED source (Etherscan: ETH + stablecoins, attributed to exchange wallets
// through the backend's ExchangeRegistry) when the backend serves one, and
// keeps the unlabeled BTC mempool monitor otherwise.
//
// The rules are untouched: thresholds, windows, source selection,
// ClassifyDirection and its neutral band, and the semaphore all live in the
// backend. What is tested here is only what the card SAYS, plus one hard
// regression guarantee — the BTC path's text is byte-identical to what main
// shipped (testdata/whale_btc_path_golden.json, captured from main before this
// work), in EVERY field including how_it_works.

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// ── fixtures ────────────────────────────────────────────────────────────────
//
// whaleLabeledProd is the real prod payload (2026-09-15T23:24:43Z, with
// ETHERSCAN_API_KEY on), with ONE arithmetic repair: the live ETH snapshot was
// transcribed as inflow 45.00M / outflow 55.09M, whose |net|/gross is 0.10085 —
// at or above the backend's neutral band, so ClassifyDirection would have
// returned "outflow", not the "neutral" the payload carries. A fixture may not
// contradict the rule it illustrates, so the gross is 50.00M / 60.09M: the same
// net (-10,094,058.96) at ratio 0.0917, genuinely inside the band.
//
// Shape that matters: USDT carries a direction but is a STABLECOIN, ETH is a
// coin with a non-zero net the backend still calls neutral, BTC is the
// unlabeled monitor, USDC saw nothing. Each snapshot carries its own
// captured_at. The feed is 9 ETH-chain rows + 1 BTC row.
const whaleLabeledProd = `{"captured_at":"2026-09-15T23:24:43Z",
  "flows":[
    {"asset":"USDT","captured_at":"2026-09-15T23:24:43Z","net_flow_usd_24h":61104747.41,"direction":"inflow",
     "inflow_usd_24h":61104747.41,"outflow_usd_24h":0,"tx_count_24h":123,"confidence":100,"partial":true,
     "source":"etherscan","net_flow_prev_24h":24000000,"flow_pct":154.6,
     "exchange_breakdown":{"Binance":80,"Coinbase":43}},
    {"asset":"ETH","captured_at":"2026-09-15T23:24:43Z","net_flow_usd_24h":-10094058.96,"direction":"neutral",
     "inflow_usd_24h":50000000,"outflow_usd_24h":60094058.96,"tx_count_24h":77,"confidence":100,
     "partial":true,"source":"etherscan","exchange_breakdown":{"Binance":40,"Kraken":37}},
    {"asset":"BTC","captured_at":"2026-09-15T23:24:43Z","net_flow_usd_24h":0,"direction":"neutral",
     "tx_count_24h":72,"confidence":0,"partial":true,"source":"mempool"},
    {"asset":"USDC","captured_at":"2026-09-15T23:24:43Z","net_flow_usd_24h":0,"direction":"neutral",
     "tx_count_24h":0,"confidence":0,"partial":true,"source":"etherscan"}],
  "transfers":[
    {"chain":"ETH","tx_hash":"e1","timestamp":"2026-09-15T23:10:00Z","asset":"USDT","amount_native":25000000,"amount_usd":25000000,"direction":"inflow","exchange":"Binance"},
    {"chain":"ETH","tx_hash":"e2","timestamp":"2026-09-15T23:00:00Z","asset":"ETH","amount_native":4200,"amount_usd":18900000,"direction":"outflow","exchange":"Kraken"},
    {"chain":"ETH","tx_hash":"e3","timestamp":"2026-09-15T22:50:00Z","asset":"USDC","amount_native":9000000,"amount_usd":9000000,"direction":"inflow","exchange":"Coinbase"},
    {"chain":"ETH","tx_hash":"e4","timestamp":"2026-09-15T22:40:00Z","asset":"ETH","amount_native":300,"amount_usd":1350000,"direction":"inflow","exchange":"Binance"},
    {"chain":"ETH","tx_hash":"e5","timestamp":"2026-09-15T22:30:00Z","asset":"USDT","amount_native":1200000,"amount_usd":1200000,"direction":"outflow","exchange":"OKX"},
    {"chain":"ETH","tx_hash":"e6","timestamp":"2026-09-15T22:20:00Z","asset":"ETH","amount_native":120,"amount_usd":540000,"direction":"inflow","exchange":"Binance"},
    {"chain":"ETH","tx_hash":"e7","timestamp":"2026-09-15T22:10:00Z","asset":"ETH","amount_native":90,"amount_usd":405000,"direction":"outflow","exchange":"Coinbase"},
    {"chain":"ETH","tx_hash":"e8","timestamp":"2026-09-15T22:00:00Z","asset":"USDT","amount_native":350000,"amount_usd":350000,"direction":"inflow","exchange":"Binance"},
    {"chain":"ETH","tx_hash":"e9","timestamp":"2026-09-15T21:50:00Z","asset":"ETH","amount_native":40,"amount_usd":180000,"direction":"outflow","exchange":"Kraken"},
    {"chain":"BTC","tx_hash":"b1","timestamp":"2026-09-15T23:20:00Z","asset":"BTC","amount_native":30,"amount_usd":2280000,"direction":"neutral","exchange":""}]}`

// whaleCoinLeadFixture: a COIN carries the direction (ETH net out $20.09M on a
// $30.09M gross — ratio 0.667, well past the band). This is the only shape
// that may colour the card.
var whaleCoinLeadFixture = strings.Replace(whaleLabeledProd,
	`"net_flow_usd_24h":-10094058.96,"direction":"neutral",
     "inflow_usd_24h":50000000,"outflow_usd_24h":60094058.96`,
	`"net_flow_usd_24h":-20094058.96,"direction":"outflow",
     "inflow_usd_24h":5000000,"outflow_usd_24h":25094058.96`, 1)

// whaleLabeledOnly is the prod payload without the BTC monitor flow.
var whaleLabeledOnly = regexp.MustCompile(`(?s)\s*\{"asset":"BTC".*?"source":"mempool"\},`).
	ReplaceAllString(whaleLabeledProd, "")

// whaleLabeledNoNet: every labeled asset neutral, the coin's net inside the
// band (500K on a 50.00M gross = 0.01), no stablecoin flow at all.
const whaleLabeledNoNet = `{"captured_at":"2026-09-15T23:24:43Z",
  "flows":[
    {"asset":"ETH","captured_at":"2026-09-15T23:24:43Z","net_flow_usd_24h":500000,"direction":"neutral",
     "inflow_usd_24h":25250000,"outflow_usd_24h":24750000,"tx_count_24h":31,"confidence":90,
     "partial":true,"source":"etherscan"},
    {"asset":"USDT","captured_at":"2026-09-15T23:24:43Z","net_flow_usd_24h":0,"direction":"neutral",
     "inflow_usd_24h":0,"outflow_usd_24h":0,"tx_count_24h":0,"confidence":0,"partial":true,
     "source":"etherscan"}],
  "transfers":[]}`

// whaleLabeledStableOnly: only a stablecoin moved; no coin carries a flow.
const whaleLabeledStableOnly = `{"captured_at":"2026-09-15T23:24:43Z",
  "flows":[
    {"asset":"USDT","captured_at":"2026-09-15T23:24:43Z","net_flow_usd_24h":61104747.41,"direction":"inflow",
     "inflow_usd_24h":61104747.41,"outflow_usd_24h":0,"tx_count_24h":123,"confidence":100,
     "partial":true,"source":"etherscan","exchange_breakdown":{"Binance":80,"Coinbase":43}},
    {"asset":"ETH","captured_at":"2026-09-15T23:24:43Z","net_flow_usd_24h":0,"direction":"neutral",
     "inflow_usd_24h":0,"outflow_usd_24h":0,"tx_count_24h":0,"confidence":0,"partial":true,
     "source":"etherscan"}],
  "transfers":[]}`

// whaleLabeledSourceSilent is a full Etherscan failure: the worker still writes
// a snapshot per labeled asset, all of them empty, while BTC keeps reporting.
// This must NOT read as "no labeled transfer happened".
const whaleLabeledSourceSilent = `{"captured_at":"2026-09-15T23:24:43Z",
  "flows":[
    {"asset":"ETH","captured_at":"2026-09-15T23:24:43Z","net_flow_usd_24h":0,"direction":"neutral",
     "tx_count_24h":0,"confidence":0,"partial":true,"source":"etherscan"},
    {"asset":"USDT","captured_at":"2026-09-15T23:24:43Z","net_flow_usd_24h":0,"direction":"neutral",
     "tx_count_24h":0,"confidence":0,"partial":true,"source":"etherscan"},
    {"asset":"USDC","captured_at":"2026-09-15T23:24:43Z","net_flow_usd_24h":0,"direction":"neutral",
     "tx_count_24h":0,"confidence":0,"partial":true,"source":"etherscan"},
    {"asset":"BTC","captured_at":"2026-09-15T23:24:43Z","net_flow_usd_24h":0,"direction":"neutral",
     "tx_count_24h":72,"confidence":0,"partial":true,"source":"mempool"}],
  "transfers":[
    {"chain":"BTC","tx_hash":"b1","timestamp":"2026-09-15T23:20:00Z","asset":"BTC","amount_native":30,"amount_usd":2280000,"direction":"neutral","exchange":""}]}`

var whaleLabeledAt = time.Date(2026, 9, 15, 23, 30, 0, 0, time.UTC)

// whaleFooterClockRe matches the card footer's "2026-09-15 23:28 UTC" stamp.
var whaleFooterClockRe = regexp.MustCompile(`\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC`)

// whaleLabeledCards is every labeled-path card a reader can reach. Kept apart
// from whaleAllCards so the BTC golden set stays exactly what main shipped;
// whaleEveryCard merges the two for the sweeps that must cover both paths.
func whaleLabeledCards(t *testing.T) map[string]Card {
	return map[string]Card{
		"labeled_prod":        whaleCardFrom(whaleResp(t, whaleLabeledProd), whaleLabeledAt),
		"labeled_coin_lead":   whaleCardFrom(whaleResp(t, whaleCoinLeadFixture), whaleLabeledAt),
		"labeled_only":        whaleCardFrom(whaleResp(t, whaleLabeledOnly), whaleLabeledAt),
		"labeled_no_net":      whaleCardFrom(whaleResp(t, whaleLabeledNoNet), whaleLabeledAt),
		"labeled_stable_only": whaleCardFrom(whaleResp(t, whaleLabeledStableOnly), whaleLabeledAt),
		"labeled_silent":      whaleCardFrom(whaleResp(t, whaleLabeledSourceSilent), whaleLabeledAt),
	}
}

// whaleEveryCard is both paths together: the line-length and banned-word
// sweeps must hold on the labeled card exactly as they do on the monitor.
func whaleEveryCard(t *testing.T) map[string]Card {
	all := whaleAllCards(t)
	for k, v := range whaleLabeledCards(t) {
		all[k] = v
	}
	return all
}

// ── 1. the hard regression: the BTC path is main's text, byte for byte ──────

type whaleSurface struct {
	Asset      string   `json:"asset"`
	Emoji      string   `json:"emoji"`
	Verdict    string   `json:"verdict"`
	Short      string   `json:"short"`
	Facts      []string `json:"facts"`
	HowItWorks string   `json:"how_it_works"`
	SourceNote string   `json:"source_note"`
	Blocks     any      `json:"blocks"`
	HTML       string   `json:"html"`
	OneLiner   string   `json:"one_liner"`
	Conclusion string   `json:"conclusion"`
	Confidence *int     `json:"confidence"`
	Status     int      `json:"status"`
}

func whaleSurfaceOf(c Card) whaleSurface {
	return whaleSurface{
		Asset: c.Asset, Emoji: c.Emoji, Verdict: c.Verdict, Short: c.Short, Facts: c.Facts,
		HowItWorks: c.HowItWorks, SourceNote: c.SourceNote, Blocks: c.Blocks,
		HTML: c.RenderHTML(), OneLiner: c.OneLiner(), Conclusion: conclusionFor(c),
		Confidence: c.Confidence, Status: int(c.effectiveStatus()),
	}
}

// TestWhaleBTCPathMatchesMainGolden pins every unlabeled-path card against the
// text main shipped, in EVERY field. The golden was captured from main (77fcf1e)
// before the labeled source existed; a single changed character on the BTC path
// — the one running in production — fails here.
func TestWhaleBTCPathMatchesMainGolden(t *testing.T) {
	got := map[string]whaleSurface{}
	for name, c := range whaleAllCards(t) {
		got[name] = whaleSurfaceOf(c)
	}
	got["showcase_live"] = whaleSurfaceOf(whaleCardFrom(whaleResp(t, whaleLiveFixture), whaleAt))
	got["window"] = whaleSurfaceOf(whaleCardFrom(whaleResp(t, whaleWindowFixture), whaleAt))

	raw, err := os.ReadFile("testdata/whale_btc_path_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var want map[string]whaleSurface
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("card set changed: got %d, golden %d", len(got), len(want))
	}
	// Both sides are normalised through a generic decode so object keys sort
	// the same way: the golden was read back into `any`, the rebuilt card
	// marshals in struct order. Only the TEXT is being compared here.
	//
	// The footer clock is masked: a card with no captured_at (no_snapshot, the
	// offline card) dates itself by the wall clock, so the golden would else
	// expire a minute after it was captured. The data_as_of rule itself is
	// pinned where it belongs — TestWhaleGoldenLive asserts DataTime is the
	// snapshot's captured_at.
	norm := func(v whaleSurface) string {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		var generic any
		if err := json.Unmarshal(b, &generic); err != nil {
			t.Fatal(err)
		}
		out, err := json.MarshalIndent(generic, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		return whaleFooterClockRe.ReplaceAllString(string(out), "<data_as_of>")
	}
	for name, w := range want {
		g, ok := got[name]
		if !ok {
			t.Errorf("%s: missing from the rebuilt set", name)
			continue
		}
		if norm(g) != norm(w) {
			t.Errorf("%s: BTC-path text changed vs main:\ngot\n%s\nwant\n%s", name, norm(g), norm(w))
		}
	}
}

// The BTC monitor keeps its OWN description, unchanged since main. The catalog
// line (howTexts[keyWhale]) now covers both sources — correct for a list entry,
// wrong on a monitor card, which shows no labeled flow at all.
func TestWhaleBTCCardKeepsMainHowText(t *testing.T) {
	const mainHow = "Counts BTC transactions of $100K+ that backend polls of the mempool.space recent feed detected in 24h. " +
		"Sizes are total outputs, change included; no exchange direction."
	if whaleBTCMonitorHow != mainHow {
		t.Errorf("BTC how-text drifted from main:\n%q\nwant\n%q", whaleBTCMonitorHow, mainHow)
	}
	for name, c := range whaleAllCards(t) {
		if c.HowItWorks != mainHow {
			t.Errorf("%s: BTC-path card how-text:\n%q\nwant main's\n%q", name, c.HowItWorks, mainHow)
		}
	}
	for name, c := range whaleLabeledCards(t) {
		// The silent-source card IS a BTC monitor card, by design (point 3).
		want := whaleLabeledHow
		if c.Whale != nil && c.Whale.ReadSource == whaleReadBTCMonitor {
			want = mainHow
		}
		if c.HowItWorks != want {
			t.Errorf("%s: how-text:\n%q\nwant\n%q", name, c.HowItWorks, want)
		}
	}
	// The catalog entry is the dual-source line and is deliberately neither.
	if howTexts[keyWhale] == mainHow || howTexts[keyWhale] == whaleLabeledHow {
		t.Errorf("the catalog line must cover both sources: %q", howTexts[keyWhale])
	}
}

// A feed crowded out by ETH rows (the shape prod serves now that Etherscan is
// on) must not change one word of the BTC path.
func TestWhaleBTCPathFeedSkewedToETH(t *testing.T) {
	body := `{"captured_at":"2026-09-15T23:24:43Z",
	  "flows":[{"asset":"BTC","net_flow_usd_24h":0,"direction":"neutral","tx_count_24h":72,
	            "confidence":0,"partial":true,"source":"mempool"}],
	  "transfers":[
	    {"chain":"ETH","tx_hash":"e1","timestamp":"2026-09-15T23:10:00Z","asset":"USDT","amount_native":25000000,"amount_usd":25000000,"direction":"inflow","exchange":"Binance"},
	    {"chain":"ETH","tx_hash":"e2","timestamp":"2026-09-15T23:00:00Z","asset":"ETH","amount_native":4200,"amount_usd":18900000,"direction":"outflow","exchange":"Kraken"}]}`
	c := whaleCardFrom(whaleResp(t, body), whaleLabeledAt)
	if c.Asset != "BTC" || c.Emoji != emojiNeutral {
		t.Errorf("an unlabeled BTC snapshot stays the BTC monitor: asset %q emoji %s", c.Asset, c.Emoji)
	}
	if c.Verdict != "72 BTC transactions ≥ $100K seen by the monitor in 24h — exchange direction not measurable" {
		t.Errorf("verdict %q", c.Verdict)
	}
	want := []string{whaleLineThreshold, whaleLineDirection, whaleLineSample, whaleLineGaps,
		"None of the latest 2 monitor records received is a BTC transaction of this 24h window"}
	if strings.Join(c.Facts, "\n") != strings.Join(want, "\n") {
		t.Errorf("facts:\n%s\nwant\n%s", strings.Join(c.Facts, "\n"), strings.Join(want, "\n"))
	}
	// ("Binance" and "USDT" both occur legitimately inside the monitor's own
	// threshold line, "priced at the Binance BTCUSDT price" — so the check is
	// for the labeled ROWS, not for those substrings.)
	for _, leaked := range []string{"25000000 USDT", "Kraken", "to Binance ·", "from Kraken"} {
		if strings.Contains(c.RenderHTML(), leaked) {
			t.Errorf("labeled row %q leaked onto the BTC monitor card: %s", leaked, c.RenderHTML())
		}
	}
}

// Point 6: the badge is a NECESSARY condition, exactly as the docs promise.
// A non-BTC flow with counts and a net but no labeled badge is NOT a labeled
// reading — it is an unattributed snapshot, and the card stays the monitor.
func TestWhaleLabeledNeedsSourceBadge(t *testing.T) {
	body := `{"captured_at":"2026-09-15T23:24:43Z",
	  "flows":[
	    {"asset":"ETH","net_flow_usd_24h":-5000000,"direction":"outflow","inflow_usd_24h":1000000,
	     "outflow_usd_24h":6000000,"tx_count_24h":9,"confidence":50,"partial":true},
	    {"asset":"BTC","net_flow_usd_24h":0,"direction":"neutral","tx_count_24h":72,"partial":true,"source":"mempool"}],
	  "transfers":[]}`
	c := whaleCardFrom(whaleResp(t, body), whaleLabeledAt)
	if c.Whale == nil || c.Whale.ReadSource != whaleReadBTCMonitor {
		t.Fatalf("no badge → not a labeled reading: %+v", c.Whale)
	}
	if c.Asset != "BTC" {
		t.Errorf("asset %q", c.Asset)
	}
	for _, f := range c.Whale.Flows {
		if f.Asset == "ETH" && f.SourceKind != whaleSourceKindBTC {
			t.Errorf("an unbadged flow must not be marked labeled: %+v", f)
		}
	}
}

// Without any usable window end there is nothing to anchor a 24h reading to,
// so a labeled payload still falls back to the unlabeled card.
func TestWhaleLabeledNeedsWindowEnd(t *testing.T) {
	// Blank the response stamp, and drop every per-snapshot stamp (those are
	// the ones preceded by a comma; the top-level one follows "{").
	body := strings.Replace(whaleLabeledProd, `"captured_at":"2026-09-15T23:24:43Z"`, `"captured_at":""`, 1)
	body = strings.ReplaceAll(body, `,"captured_at":"2026-09-15T23:24:43Z",`, ",")
	c := whaleCardFrom(whaleResp(t, body), whaleLabeledAt)
	if c.Whale == nil || c.Whale.ReadSource != whaleReadBTCMonitor {
		t.Errorf("no window end → the unlabeled card: %+v", c.Whale)
	}
}

// ── 2. the lead asset is a coin; stablecoins never colour the card ──────────

// Point 2: a stablecoin deposit to an exchange is conventionally read as
// buying power, the opposite of what the backend's coin-oriented rule implies.
// It may not set the headline or the semaphore. The backend rule is untouched —
// only what the card promotes changes.
func TestWhaleLabeledStablecoinNeverLeads(t *testing.T) {
	// The real payload: USDT is directional, ETH (the coin) is neutral.
	c := whaleCardFrom(whaleResp(t, whaleLabeledProd), whaleLabeledAt)
	if c.Emoji != emojiNeutral {
		t.Errorf("a stablecoin flow must not colour the card, got %s", c.Emoji)
	}
	if c.Confidence != nil {
		t.Errorf("no confidence without a coin direction: %v", c.Confidence)
	}
	if !strings.Contains(c.Verdict, "Stablecoin flow to labeled exchange wallets over 24h: USDT $61.10M") {
		t.Errorf("the verdict must name the stablecoin flow, without a direction claim: %q", c.Verdict)
	}
	if c.Whale == nil || c.Whale.LeadAsset != nil {
		t.Errorf("lead_asset must be null when no coin carries a direction: %+v", c.Whale)
	}
	if d := c.Whale.Direction; d == nil || *d != whaleDirNoNet {
		t.Errorf("direction %v, want no_net_direction", d)
	}
	// The stablecoin's own numbers are still shown, as a line.
	if !strings.Contains(strings.Join(c.Facts, "\n"), "USDT: net to exchanges $61.10M over 24h · 123 transfers") {
		t.Errorf("the stablecoin numbers must still be reported: %v", c.Facts)
	}

	// A COIN direction is the only thing that colours the card.
	coin := whaleCardFrom(whaleResp(t, whaleCoinLeadFixture), whaleLabeledAt)
	if coin.Emoji != emojiBull {
		t.Errorf("a coin net FROM exchanges is 🟢, got %s", coin.Emoji)
	}
	if want := "Net from labeled exchange wallets over 24h: ETH $20.09M (77 transfers)"; coin.Verdict != want {
		t.Errorf("verdict:\n%q\nwant\n%q", coin.Verdict, want)
	}
	if coin.Whale.LeadAsset == nil || *coin.Whale.LeadAsset != "ETH" {
		t.Errorf("lead_asset %v, want ETH", coin.Whale.LeadAsset)
	}
	if coin.Confidence == nil || *coin.Confidence != 100 {
		t.Errorf("a coin direction keeps the backend's confidence: %v", coin.Confidence)
	}
}

// Only stablecoins moved: neutral semaphore, and the verdict names the
// stablecoin flow without a colour or a direction verdict.
func TestWhaleLabeledStablecoinOnly(t *testing.T) {
	c := whaleCardFrom(whaleResp(t, whaleLabeledStableOnly), whaleLabeledAt)
	if c.Emoji != emojiNeutral || c.Confidence != nil {
		t.Errorf("emoji %s confidence %v", c.Emoji, c.Confidence)
	}
	if !strings.HasPrefix(c.Verdict, "Stablecoin flow to labeled exchange wallets over 24h: USDT $61.10M") {
		t.Errorf("verdict %q", c.Verdict)
	}
	if semaphoreOf(c.Emoji) != "neutral" {
		t.Errorf("semaphore %q", semaphoreOf(c.Emoji))
	}
}

// The stablecoin set is explicit and covers what the backend values at $1.
func TestWhaleStablecoinSet(t *testing.T) {
	for _, s := range []string{"USDT", "USDC", "usdt"} {
		if !whaleIsStablecoin(s) {
			t.Errorf("%q must be a stablecoin", s)
		}
	}
	for _, c := range []string{"ETH", "BTC", "SOL"} {
		if whaleIsStablecoin(c) {
			t.Errorf("%q is a coin", c)
		}
	}
	// The backend values exactly these two at ~$1 today (whale/source_etherscan.go
	// stableTokens); the set must at least hold them.
	src, err := os.ReadFile("../whale/source_etherscan.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, sym := range []string{`Asset: "USDT"`, `Asset: "USDC"`} {
		if !strings.Contains(string(src), sym) {
			t.Fatalf("backend stablecoin set changed (%s missing): revisit whaleStablecoins", sym)
		}
	}
}

// ── 3. a silent labeled source is not "nothing moved" ───────────────────────

func TestWhaleLabeledSourceSilentFallsBackToBTC(t *testing.T) {
	c := whaleCardFrom(whaleResp(t, whaleLabeledSourceSilent), whaleLabeledAt)
	if c.Whale == nil || c.Whale.ReadSource != whaleReadBTCMonitor {
		t.Fatalf("all labeled snapshots empty → the BTC monitor, not a labeled zero: %+v", c.Whale)
	}
	// The BTC monitor's own reading is intact.
	if c.Verdict != "72 BTC transactions ≥ $100K seen by the monitor in 24h — exchange direction not measurable" {
		t.Errorf("verdict %q", c.Verdict)
	}
	if c.Asset != "BTC" {
		t.Errorf("asset %q", c.Asset)
	}
	// And the silence is disclosed, never dressed as a finding.
	f := strings.Join(c.Facts, "\n")
	if !strings.Contains(f, whaleLineLabeledSilent) {
		t.Errorf("the labeled source's silence must be stated: %v", c.Facts)
	}
	if n := utf8.RuneCountInString(whaleLineLabeledSilent); n > whaleFactMaxRunes {
		t.Errorf("silence line %d runes > %d", n, whaleFactMaxRunes)
	}
	for _, bad := range []string{"No labeled exchange transfer recorded", "no labeled transfer in 24h"} {
		if strings.Contains(f, bad) {
			t.Errorf("a dead source must not read as a measurement (%q): %v", bad, c.Facts)
		}
	}
}

// ── 4. the labeled window is the LEAD asset's own snapshot time ─────────────

func TestWhaleLabeledWindowFromLeadSnapshot(t *testing.T) {
	// Per-asset captured_at present: it dates the card and the window.
	c := whaleCardFrom(whaleResp(t, whaleCoinLeadFixture), whaleLabeledAt)
	want := time.Date(2026, 9, 15, 23, 24, 43, 0, time.UTC)
	if !c.DataTime.Equal(want) {
		t.Errorf("data time %s, want the lead snapshot's captured_at %s", c.DataTime, want)
	}
	if c.Whale.WindowEnd == nil || *c.Whale.WindowEnd != "2026-09-15T23:24:43Z" {
		t.Errorf("window_end %v", c.Whale.WindowEnd)
	}
	if !strings.Contains(c.Blocks.WhatHappened, "in the 24h to Sep 15 23:24 UTC") {
		t.Errorf("what_happened %q", c.Blocks.WhatHappened)
	}

	// A stale Etherscan snapshot beside a fresh BTC tick: the labeled numbers
	// must NOT be stamped with the BTC time.
	stale := strings.Replace(whaleCoinLeadFixture,
		`{"asset":"ETH","captured_at":"2026-09-15T23:24:43Z"`,
		`{"asset":"ETH","captured_at":"2026-09-15T19:00:00Z"`, 1)
	s := whaleCardFrom(whaleResp(t, stale), whaleLabeledAt)
	if got := time.Date(2026, 9, 15, 19, 0, 0, 0, time.UTC); !s.DataTime.Equal(got) {
		t.Errorf("stale lead: data time %s, want %s", s.DataTime, got)
	}
	if !strings.Contains(s.Blocks.WhatHappened, "in the 24h to Sep 15 19:00 UTC") {
		t.Errorf("stale lead what_happened %q", s.Blocks.WhatHappened)
	}

	// No per-asset captured_at at all: the window is not stamped with a time
	// that does not belong to it.
	// Only the per-snapshot stamps are dropped (preceded by a comma); the
	// response's own captured_at stays, and must not be borrowed as the window.
	noStamp := strings.ReplaceAll(whaleCoinLeadFixture, `,"captured_at":"2026-09-15T23:24:43Z",`, ",")
	n := whaleCardFrom(whaleResp(t, noStamp), whaleLabeledAt)
	if n.Whale == nil || n.Whale.ReadSource != whaleReadLabeled {
		t.Fatalf("still a labeled reading: %+v", n.Whale)
	}
	if n.Whale.WindowEnd != nil {
		t.Errorf("window_end must be null when the lead snapshot carries no time: %v", *n.Whale.WindowEnd)
	}
	if strings.Contains(n.Blocks.WhatHappened, "23:24") || !strings.Contains(n.Blocks.WhatHappened, "in the last 24h") {
		t.Errorf("unstamped window must not name a time: %q", n.Blocks.WhatHappened)
	}
}

// ── 5. the 10% line only when the share really is under the band ────────────

func TestWhaleLabeledBandLineOnlyWhenUnderBand(t *testing.T) {
	for _, tc := range []struct {
		name string
		flow WhaleFlow
		want string
	}{
		{
			// 0.0917 of gross — genuinely inside the band, so the reason holds.
			name: "under the band names it",
			flow: WhaleFlow{Asset: "ETH", Direction: "neutral", NetFlowUSD24h: -10094058.96,
				InflowUSD24h: 50000000, OutflowUSD24h: 60094058.96, TxCount24h: 77},
			want: "ETH: net $10.09M away from exchanges over 24h, under the 10% of gross flow needed to name it · 77 transfers",
		},
		{
			// Positive net leans toward exchanges (backend: netUSD > 0 is TO).
			name: "positive net leans toward exchanges",
			flow: WhaleFlow{Asset: "ETH", Direction: "neutral", NetFlowUSD24h: 5400000,
				InflowUSD24h: 32700000, OutflowUSD24h: 27300000, TxCount24h: 77},
			want: "ETH: net $5.40M toward exchanges over 24h, under the 10% of gross flow needed to name it · 77 transfers",
		},
		{
			// No gross on the wire: the share cannot be computed, so the band
			// must not be claimed as the reason.
			name: "no gross means no band claim",
			flow: WhaleFlow{Asset: "ETH", Direction: "neutral", NetFlowUSD24h: -10094058.96, TxCount24h: 77},
			want: "ETH: net $10.09M away from exchanges over 24h, the source named no direction · 77 transfers",
		},
		{
			// Share at/above the band: the backend would have named a direction,
			// so the band is not the reason here either.
			name: "at or above the band means no band claim",
			flow: WhaleFlow{Asset: "ETH", Direction: "neutral", NetFlowUSD24h: -10094058.96,
				InflowUSD24h: 45000000, OutflowUSD24h: 55094058.96, TxCount24h: 77},
			want: "ETH: net $10.09M away from exchanges over 24h, the source named no direction · 77 transfers",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := whaleLabeledAssetLine(tc.flow)
			if got != tc.want {
				t.Errorf("line:\n%q\nwant\n%q", got, tc.want)
			}
			if n := utf8.RuneCountInString(got); n > whaleFactMaxRunes {
				t.Errorf("%d runes > %d: %q", n, whaleFactMaxRunes, got)
			}
			for _, bad := range []string{"net to exchanges", "net from exchanges"} {
				if strings.Contains(got, bad) {
					t.Errorf("a neutral net must not read as a direction: %q", got)
				}
			}
		})
	}

	// The band the card prints is the backend's own constant.
	src, err := os.ReadFile("../whale/scorer.go")
	if err != nil {
		t.Fatal(err)
	}
	if !regexp.MustCompile(`neutralThreshold\s*=\s*0\.1\b`).Match(src) {
		t.Fatal("backend neutral band changed: update whaleNeutralBandPct and the line that prints it")
	}
	if whaleNeutralBandPct != 10 {
		t.Errorf("band %d, want 10", whaleNeutralBandPct)
	}
}

// The short form drops only "of gross flow" and keeps the side.
func TestWhaleLabeledBandLineShortForm(t *testing.T) {
	got := whaleLabeledAssetLine(WhaleFlow{Asset: "USDC", Direction: "neutral", NetFlowUSD24h: -100000000,
		InflowUSD24h: 600000000, OutflowUSD24h: 700000000, TxCount24h: 1234})
	want := "USDC: net $100.00M away from exchanges over 24h, under the 10% needed to name a direction · 1234 transfers"
	if got != want {
		t.Errorf("line:\n%q\nwant\n%q", got, want)
	}
	if n := utf8.RuneCountInString(got); n > whaleFactMaxRunes {
		t.Errorf("%d runes > %d", n, whaleFactMaxRunes)
	}
}

// ── 6/7. the envelope contract ─────────────────────────────────────────────

// Point 7: `asset` is composite on the labeled path — the assets that actually
// carry a flow — and the single headline coin lives in its own machine field.
func TestWhaleLabeledCompositeAsset(t *testing.T) {
	// No coin direction: response order, USDC excluded (no flow at all).
	prod := whaleCardFrom(whaleResp(t, whaleLabeledProd), whaleLabeledAt)
	if prod.Asset != "USDT/ETH" {
		t.Errorf("composite asset %q, want USDT/ETH", prod.Asset)
	}
	if env := cardEnvelope(prod); env.Asset != "USDT/ETH" {
		t.Errorf("envelope asset %q — it must not stay BTC when the card is about other assets", env.Asset)
	}
	// With a coin lead the composite leads with it, matching verdict and header.
	coin := whaleCardFrom(whaleResp(t, whaleCoinLeadFixture), whaleLabeledAt)
	if coin.Asset != "ETH/USDT" {
		t.Errorf("composite asset %q, want the lead coin first", coin.Asset)
	}
}

func TestWhaleLabeledJSONFields(t *testing.T) {
	c := whaleCardFrom(whaleResp(t, whaleCoinLeadFixture), whaleLabeledAt)
	raw, err := json.Marshal(c.Whale)
	if err != nil {
		t.Fatal(err)
	}
	var ro struct {
		ReadSource string  `json:"read_source"`
		Source     string  `json:"source"`
		Direction  string  `json:"direction"`
		LeadAsset  *string `json:"lead_asset"`
		TimeKind   string  `json:"time_kind"`
		AmountKind string  `json:"amount_kind"`
		Flows      []struct {
			Asset      string  `json:"asset"`
			NetFlowUSD float64 `json:"net_flow_usd_24h"`
			Direction  string  `json:"direction"`
			TxCount    int     `json:"tx_count"`
			SourceKind string  `json:"source_kind"`
			Partial    bool    `json:"partial"`
		} `json:"flows"`
	}
	if err := json.Unmarshal(raw, &ro); err != nil {
		t.Fatal(err)
	}
	if ro.ReadSource != whaleReadLabeled || ro.Direction != whaleDirFrom {
		t.Errorf("read_source %q direction %q", ro.ReadSource, ro.Direction)
	}
	if ro.LeadAsset == nil || *ro.LeadAsset != "ETH" {
		t.Errorf("lead_asset %v", ro.LeadAsset)
	}
	if ro.Source != "etherscan labeled exchange wallets" {
		t.Errorf("source %q", ro.Source)
	}
	if ro.TimeKind != "block_time" || ro.AmountKind != "transfer_amount" {
		t.Errorf("kinds %q / %q — the labeled source stamps the block, not a poll", ro.TimeKind, ro.AmountKind)
	}
	if len(ro.Flows) != 4 {
		t.Fatalf("flows %+v", ro.Flows)
	}
	if ro.Flows[0].Asset != "USDT" || ro.Flows[0].TxCount != 123 || ro.Flows[0].SourceKind != "labeled" {
		t.Errorf("USDT flow %+v", ro.Flows[0])
	}
	if ro.Flows[1].Asset != "ETH" || ro.Flows[1].Direction != whaleDirFrom {
		t.Errorf("ETH flow %+v", ro.Flows[1])
	}
	if ro.Flows[2].Asset != "BTC" || ro.Flows[2].SourceKind != "btc_monitor" {
		t.Errorf("BTC flow %+v", ro.Flows[2])
	}
	if !ro.Flows[0].Partial {
		t.Error("the backend marks every free-tier snapshot partial")
	}

	// The BTC path carries the same additive fields, and says so.
	btc := whaleCardFrom(whaleResp(t, whaleLive15), whaleAt)
	rawBTC, _ := json.Marshal(btc.Whale)
	if !strings.Contains(string(rawBTC), `"read_source":"`+whaleReadBTCMonitor+`"`) {
		t.Errorf("BTC readout: %s", rawBTC)
	}
	if !strings.Contains(string(rawBTC), `"lead_asset":null`) {
		t.Errorf("the BTC monitor has no lead asset: %s", rawBTC)
	}
}

// ── the labeled card's own words ───────────────────────────────────────────

func TestWhaleLabeledFacts(t *testing.T) {
	c := whaleCardFrom(whaleResp(t, whaleLabeledProd), whaleLabeledAt)
	f := strings.Join(c.Facts, "\n")
	for _, want := range []string{
		"USDT: net to exchanges $61.10M over 24h · 123 transfers",
		"ETH: net $10.09M away from exchanges over 24h, under the 10% of gross flow needed to name it · 77 transfers",
		"USDC: no labeled transfer in 24h",
		"An estimate over the wallets in our registry, not the whole market: unlisted wallets are invisible",
		"Address coverage is not reported: a labeled address that failed to load lowers these numbers unmarked",
		"BTC monitor, separate and unlabeled: 72 transactions ≥ $100K seen in 24h, no exchange direction",
	} {
		if !strings.Contains(f, want) {
			t.Errorf("missing fact:\n%q\nin\n%s", want, f)
		}
	}
	// Point 8: the exchange line must be found by its own prefix — "Binance"
	// also appears in the transfer lines below it, which would pass for free.
	var exchLine string
	for _, line := range c.Facts {
		if strings.HasPrefix(line, "Exchanges in the reported breakdown: ") {
			exchLine = line
		}
	}
	if exchLine == "" {
		t.Fatalf("no exchange-breakdown line: %v", c.Facts)
	}
	for _, want := range []string{"Binance", "Kraken", "Coinbase"} {
		if !strings.Contains(exchLine, want) {
			t.Errorf("exchange line %q lacks %q", exchLine, want)
		}
	}
}

// A labeled card names its own source and never the mempool monitor's words.
func TestWhaleLabeledSourceWords(t *testing.T) {
	c := whaleCardFrom(whaleResp(t, whaleLabeledProd), whaleLabeledAt)
	all := c.RenderHTML()
	if !strings.Contains(all, "Etherscan") {
		t.Errorf("the labeled source must be named: %s", all)
	}
	if c.SourceNote != "data: Etherscan labeled exchange wallets" {
		t.Errorf("source note %q", c.SourceNote)
	}
	for _, bad := range []string{"mempool", "Binance BTCUSDT", "newest mempool entries",
		"outputs include change", "total outputs", "poll coverage is not served"} {
		if strings.Contains(all, bad) {
			t.Errorf("mempool wording on the labeled path: %q in\n%s", bad, all)
		}
	}
	b, _ := json.Marshal(c.Blocks)
	if strings.Contains(string(b), "mempool") {
		t.Errorf("mempool wording in blocks: %s", b)
	}
}

func TestWhaleLabeledTransactionList(t *testing.T) {
	c := whaleCardFrom(whaleResp(t, whaleLabeledProd), whaleLabeledAt)
	f := strings.Join(c.Facts, "\n")
	for _, want := range []string{
		"25000000 USDT ≈ $25.00M · to Binance · block time Sep 15 23:10 UTC",
		"4200 ETH ≈ $18.90M · from Kraken · block time Sep 15 23:00 UTC",
		"9000000 USDC ≈ $9.00M · to Coinbase · block time Sep 15 22:50 UTC",
	} {
		if !strings.Contains(f, want) {
			t.Errorf("missing listed transfer:\n%q\nin\n%s", want, f)
		}
	}
	if c.Whale == nil || len(c.Whale.Top.Transactions) != whaleTopN {
		t.Fatalf("top: %+v", c.Whale)
	}
	if c.Whale.Top.Selection != "largest_labeled_among_latest_records" {
		t.Errorf("selection %q", c.Whale.Top.Selection)
	}
	if strings.Contains(f, "BTC ≈") {
		t.Errorf("the unlabeled BTC row must not be listed as a labeled transfer: %s", f)
	}
}

func TestWhaleLabeledAndBTCNotMixed(t *testing.T) {
	c := whaleCardFrom(whaleResp(t, whaleCoinLeadFixture), whaleLabeledAt)
	if strings.Contains(c.Verdict, "72") || strings.Contains(c.Verdict, "BTC") {
		t.Errorf("the BTC count must not enter the labeled verdict: %q", c.Verdict)
	}
	btcLines := 0
	for _, f := range c.Facts {
		if strings.Contains(f, "BTC monitor") {
			btcLines++
			if strings.Contains(f, "20.09") || strings.Contains(f, "ETH") {
				t.Errorf("labeled numbers inside the BTC line: %q", f)
			}
		}
	}
	if btcLines != 1 {
		t.Errorf("the BTC monitor gets exactly one separate line, got %d", btcLines)
	}
	only := whaleCardFrom(whaleResp(t, whaleLabeledOnly), whaleLabeledAt)
	if strings.Contains(only.RenderHTML(), "BTC monitor") {
		t.Errorf("no BTC snapshot → no BTC line: %s", only.RenderHTML())
	}
}

func TestWhaleLabeledNoNetDirection(t *testing.T) {
	c := whaleCardFrom(whaleResp(t, whaleLabeledNoNet), whaleLabeledAt)
	if c.Emoji != emojiNeutral || c.Confidence != nil {
		t.Errorf("emoji %s confidence %v", c.Emoji, c.Confidence)
	}
	if !strings.HasPrefix(c.Verdict, "No net direction at labeled exchange wallets over 24h") {
		t.Errorf("verdict %q", c.Verdict)
	}
	if all := strings.ToLower(c.RenderHTML()); strings.Contains(all, "balanced") {
		t.Errorf("a neutral net is not a balance: %s", all)
	}
}

// Source unavailable stays the standard offline card, unchanged.
func TestWhaleLabeledOffline(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "down", http.StatusBadGateway)
	}))
	t.Cleanup(srv.Close)
	off := NewAgents(NewBackendClient(srv.URL)).WhaleCard(context.Background())
	if off.effectiveStatus() != statusSourceOffline || off.Whale != nil || off.Blocks != nil {
		t.Errorf("offline: status %d readout %+v", off.effectiveStatus(), off.Whale)
	}
}

// End to end over HTTP: 200, no Last-Modified, the labeled readout served.
func TestWhaleLabeledHTTPEnvelope(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/whale-flow": whaleCoinLeadFixture})
	_, srv := newTestAPI(t, ag, true)
	status, hdr, body := httpGet(t, srv.URL+"/agents/whale")
	if status != 200 || hdr.Get("Last-Modified") != "" {
		t.Fatalf("status %d Last-Modified %q", status, hdr.Get("Last-Modified"))
	}
	for _, want := range []string{`"read_source":"labeled_exchange_wallets"`, `"lead_asset":"ETH"`,
		`"asset":"ETH/USDT"`,
		`"verdict":"Net from labeled exchange wallets over 24h: ETH $20.09M (77 transfers)"`} {
		if !strings.Contains(string(body), want) {
			t.Errorf("body lacks %s:\n%s", want, body)
		}
	}
}

func TestWhaleLabeledHookHashStable(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/whale-flow": whaleCoinLeadFixture})
	clk := &stepClock{t: whaleLabeledAt}
	ag.now = clk.now
	th := newTestHook(t, ag, "http://127.0.0.1:1", nil)
	tg := mustHookTarget(t, "/agents/whale")
	ctx := context.Background()

	read := func() (int, []byte, string) {
		start := time.Now()
		st, body := th.fetch(ctx, tg)
		return st, body, hashOf(t, tg.Agent, body, start, time.Now())
	}
	st1, b1, h1 := read()
	nextWallSecond()
	clk.set(whaleLabeledAt.Add(3 * time.Hour))
	st2, b2, h2 := read()
	if st1 != 200 || st2 != 200 {
		t.Fatalf("status %d / %d", st1, st2)
	}
	if !strings.Contains(string(b1), `"read_source":"labeled_exchange_wallets"`) {
		t.Fatalf("fixture must be the labeled shape: %.600s", b1)
	}
	if h1 != h2 {
		t.Errorf("hook hash moved with unchanged data:\n%s\n%s", b1, b2)
	}
}

func TestWhaleLabeledConclusionNoPriceDirection(t *testing.T) {
	for _, body := range []string{whaleCoinLeadFixture, whaleLabeledProd, whaleLabeledNoNet} {
		c := whaleCardFrom(whaleResp(t, body), whaleLabeledAt)
		got := conclusionFor(c)
		if !strings.Contains(got, "estimate") || !strings.Contains(got, "says nothing about where price goes") {
			t.Errorf("conclusion %q", got)
		}
		for _, bad := range []string{"bullish", "bearish", "lean", "buy", "sell", "expect"} {
			if strings.Contains(strings.ToLower(got), bad) {
				t.Errorf("conclusion carries %q: %q", bad, got)
			}
		}
	}
}

// ── the catalog line ───────────────────────────────────────────────────────

// The how-text map backs the catalog list and the Telegram [How it works]
// button, which are per-AGENT, not per-card: they must name both sources.
func TestWhaleHowTextCoversBothSources(t *testing.T) {
	h := howTexts[keyWhale]
	if n := utf8.RuneCountInString(h); n > 200 {
		t.Errorf("how-text %d runes > 200 (the Telegram alert cap): %q", n, h)
	}
	for _, want := range []string{"Etherscan", "labeled"} {
		if !strings.Contains(h, want) {
			t.Errorf("how-text does not name the labeled source (%q): %q", want, h)
		}
	}
	for _, want := range []string{"BTC", "mempool.space"} {
		if !strings.Contains(h, want) {
			t.Errorf("how-text does not name the BTC monitor (%q): %q", want, h)
		}
	}
	for _, bad := range []string{"inflow", "outflow", "bullish", "bearish", "signal", "expect"} {
		if strings.Contains(strings.ToLower(h), bad) {
			t.Errorf("how-text carries %q: %q", bad, h)
		}
	}
}
