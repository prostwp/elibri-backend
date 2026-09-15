package demobot

// whale_readable_test.go — Whale stage 1 (2026-09-15): the card's wording,
// states, blocks and machine fields. The rules are untouched (threshold
// $100K, mempool.space, 24h window, the newest 10 records); every case runs
// the pure builder (whaleCardFrom) at a fixed clock, so the texts are golden.

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"
)

var whaleAt = time.Date(2026, 9, 15, 15, 7, 0, 0, time.UTC)

// whaleLive15 is shaped like the prod payload of 2026-09-15 15:00 UTC: 66
// transactions counted, the 10 newest records of the table (one ETH among
// them), the BTC snapshot neutral and partial (mempool carries no labels).
const whaleLive15 = `{"captured_at":"2026-09-15T15:00:00Z",
  "flows":[{"asset":"BTC","net_flow_usd_24h":0,"direction":"neutral","inflow_usd_24h":0,"outflow_usd_24h":0,
            "tx_count_24h":66,"confidence":0,"partial":true,"source":"mempool","net_flow_prev_24h":0,"flow_pct":0}],
  "transfers":[
    {"chain":"BTC","tx_hash":"a1","timestamp":"2026-09-15T15:00:00Z","asset":"BTC","amount_native":2465,"amount_usd":187020000,"direction":"neutral","exchange":""},
    {"chain":"BTC","tx_hash":"a2","timestamp":"2026-09-15T15:00:00Z","asset":"BTC","amount_native":12.5,"amount_usd":950000,"direction":"neutral","exchange":""},
    {"chain":"ETH","tx_hash":"e1","timestamp":"2026-09-15T14:55:00Z","asset":"ETH","amount_native":5000,"amount_usd":900000000,"direction":"neutral","exchange":""},
    {"chain":"BTC","tx_hash":"a3","timestamp":"2026-09-15T14:50:00Z","asset":"BTC","amount_native":55.6,"amount_usd":4380000,"direction":"neutral","exchange":""},
    {"chain":"BTC","tx_hash":"a4","timestamp":"2026-09-15T14:50:00Z","asset":"BTC","amount_native":3.2,"amount_usd":243000,"direction":"neutral","exchange":""},
    {"chain":"BTC","tx_hash":"a5","timestamp":"2026-09-15T14:40:00Z","asset":"BTC","amount_native":50.36,"amount_usd":3930000,"direction":"neutral","exchange":""},
    {"chain":"BTC","tx_hash":"a6","timestamp":"2026-09-15T14:40:00Z","asset":"BTC","amount_native":2.1,"amount_usd":160000,"direction":"neutral","exchange":""},
    {"chain":"BTC","tx_hash":"a7","timestamp":"2026-09-15T14:30:00Z","asset":"BTC","amount_native":1.5,"amount_usd":114000,"direction":"neutral","exchange":""},
    {"chain":"BTC","tx_hash":"a8","timestamp":"2026-09-15T14:30:00Z","asset":"BTC","amount_native":4,"amount_usd":304000,"direction":"neutral","exchange":""},
    {"chain":"BTC","tx_hash":"a9","timestamp":"2026-09-15T14:20:00Z","asset":"BTC","amount_native":1.4,"amount_usd":106000,"direction":"neutral","exchange":""}]}`

const whaleZero15 = `{"captured_at":"2026-09-15T15:00:00Z",
  "flows":[{"asset":"BTC","net_flow_usd_24h":0,"direction":"neutral","tx_count_24h":0,"confidence":0,"partial":true,"source":"mempool"}],
  "transfers":[]}`

// Count > 0, but none of the newest records is a BTC transaction of the
// window (ETH crowded them out, or they are older than 24h).
const whaleNoneShown15 = `{"captured_at":"2026-09-15T15:00:00Z",
  "flows":[{"asset":"BTC","net_flow_usd_24h":0,"direction":"neutral","tx_count_24h":5,"partial":true,"source":"mempool"}],
  "transfers":[
    {"chain":"ETH","tx_hash":"e1","timestamp":"2026-09-15T14:55:00Z","asset":"ETH","amount_native":90,"amount_usd":400000},
    {"chain":"BTC","tx_hash":"old","timestamp":"2026-09-14T14:00:00Z","asset":"BTC","amount_native":9,"amount_usd":700000}]}`

const whaleNoSnapshot15 = `{"captured_at":"","flows":[],"transfers":[]}`

func whaleResp(t *testing.T, body string) *WhaleResp {
	t.Helper()
	var w WhaleResp
	if err := json.Unmarshal([]byte(body), &w); err != nil {
		t.Fatal(err)
	}
	return &w
}

// The sample the card names is the backend's: the mempool.space recent
// endpoint, polled every 10 minutes by the whale worker. The 10 entries per
// call are the external API's behaviour (checked live 2026-09-15) and cannot
// be pinned from here.
func TestWhaleSampleMatchesBackend(t *testing.T) {
	main, err := os.ReadFile("../../cmd/server/main.go")
	if err != nil {
		t.Fatal(err)
	}
	if !regexp.MustCompile(`whaleWorker := &whale\.Worker\{[^}]*RefreshInterval:\s*10 \* time\.Minute,`).Match(main) {
		t.Fatal("whale poll interval changed in cmd/server: update whaleMonitorPoll and the sample lines")
	}
	src, err := os.ReadFile("../whale/source_mempool.go")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(src), `mempoolRecentURL = "https://mempool.space/api/mempool/recent"`) {
		t.Fatal("whale BTC source changed: the sample lines name mempool.space /api/mempool/recent")
	}
	if whaleMonitorPoll != 10*time.Minute || whaleMempoolRecentSize != 10 {
		t.Errorf("poll %s, entries %d", whaleMonitorPoll, whaleMempoolRecentSize)
	}
	for got, want := range map[string]string{
		whaleLineSample: "Each 10-minute poll sees only the 10 newest mempool.space entries; most transactions ≥ $100K are never seen",
		whaleLineGaps:   "Missed polls or a missing BTC price lower this count unmarked: poll coverage is not served",
	} {
		if got != want {
			t.Errorf("%q, want %q", got, want)
		}
	}
}

func TestWhaleGoldenLive(t *testing.T) {
	c := whaleCardFrom(whaleResp(t, whaleLive15), whaleAt)
	if want := "66 BTC transactions ≥ $100K seen by the monitor in 24h — exchange direction not measurable"; c.Verdict != want {
		t.Errorf("verdict:\n%q\nwant\n%q", c.Verdict, want)
	}
	if want := "66 tx ≥ $100K seen, direction n/a"; c.Short != want {
		t.Errorf("short: %q", c.Short)
	}
	want := []string{
		whaleLineThreshold,
		whaleLineDirection,
		whaleLineSample,
		whaleLineGaps,
		"Largest BTC among the latest 10 monitor records received, not a 24h top · outputs include change",
		"Outputs total 2465 BTC ≈ $187.02M · detected by the monitor at Sep 15 15:00 UTC, not the block time",
		"Outputs total 55.60 BTC ≈ $4.38M · detected by the monitor at Sep 15 14:50 UTC, not the block time",
		"Outputs total 50.36 BTC ≈ $3.93M · detected by the monitor at Sep 15 14:40 UTC, not the block time",
	}
	if strings.Join(c.Facts, "\n") != strings.Join(want, "\n") {
		t.Errorf("facts:\n%s\nwant\n%s", strings.Join(c.Facts, "\n"), strings.Join(want, "\n"))
	}
	if c.Emoji != emojiNeutral || c.Confidence != nil || c.effectiveStatus() != statusOK {
		t.Errorf("emoji %s, confidence %v, status %d", c.Emoji, c.Confidence, c.effectiveStatus())
	}
	if c.SourceNote != "data: mempool.space" {
		t.Errorf("source note %q", c.SourceNote)
	}
	if !c.noValidator {
		t.Error("whale must never carry a validator")
	}
	if !c.DataTime.Equal(time.Date(2026, 9, 15, 15, 0, 0, 0, time.UTC)) {
		t.Errorf("data time %s, want captured_at", c.DataTime)
	}
}

func TestWhaleGoldenZero(t *testing.T) {
	c := whaleCardFrom(whaleResp(t, whaleZero15), whaleAt)
	if want := "The monitor registered no BTC transaction ≥ $100K in 24h"; c.Verdict != want {
		t.Errorf("verdict %q", c.Verdict)
	}
	if c.Short != "none ≥ $100K registered" {
		t.Errorf("short %q", c.Short)
	}
	want := []string{whaleLineThreshold, whaleLineDirection, whaleLineSample, whaleLineGaps}
	if strings.Join(c.Facts, "\n") != strings.Join(want, "\n") {
		t.Errorf("facts:\n%s", strings.Join(c.Facts, "\n"))
	}
	if c.effectiveStatus() != statusOK || c.Whale == nil || c.Whale.State != whaleStateNone {
		t.Errorf("zero is a real reading of the monitor: status %d, readout %+v", c.effectiveStatus(), c.Whale)
	}
	if c.Blocks == nil || c.Blocks.WhatHappened != "The monitor registered no BTC transaction ≥ $100K in the 24h to Sep 15 15:00 UTC" {
		t.Errorf("blocks %+v", c.Blocks)
	}
}

func TestWhaleOneTransactionSingular(t *testing.T) {
	body := strings.Replace(whaleZero15, `"tx_count_24h":0`, `"tx_count_24h":1`, 1)
	c := whaleCardFrom(whaleResp(t, body), whaleAt)
	if c.Verdict != "1 BTC transaction ≥ $100K seen by the monitor in 24h — exchange direction not measurable" {
		t.Errorf("verdict %q", c.Verdict)
	}
	if c.Blocks.WhatHappened != "The monitor detected 1 BTC transaction ≥ $100K in the 24h to Sep 15 15:00 UTC" {
		t.Errorf("what_happened %q", c.Blocks.WhatHappened)
	}
}

// The counter says 5, the list the card received holds no BTC transaction of
// the window: the card says so instead of "no transfers" under a count of 5.
func TestWhaleCountButNoneAmongLatest(t *testing.T) {
	c := whaleCardFrom(whaleResp(t, whaleNoneShown15), whaleAt)
	want := []string{whaleLineThreshold, whaleLineDirection, whaleLineSample, whaleLineGaps,
		"None of the latest 2 monitor records received is a BTC transaction of this 24h window"}
	if strings.Join(c.Facts, "\n") != strings.Join(want, "\n") {
		t.Errorf("facts:\n%s", strings.Join(c.Facts, "\n"))
	}
	if c.Whale.Top.RecordsReceived != 2 || len(c.Whale.Top.Transactions) != 0 {
		t.Errorf("top %+v", c.Whale.Top)
	}
}

func TestWhaleNoSnapshot(t *testing.T) {
	c := whaleCardFrom(whaleResp(t, whaleNoSnapshot15), whaleAt)
	if c.Verdict != "No BTC count from the monitor yet" || c.Short != "no data" {
		t.Errorf("verdict %q short %q", c.Verdict, c.Short)
	}
	if c.effectiveStatus() != statusNoData || c.Blocks != nil {
		t.Errorf("status %d, blocks %+v", c.effectiveStatus(), c.Blocks)
	}
	if c.Whale == nil || c.Whale.State != whaleStateNoSnapshot || c.Whale.Count != nil || c.Whale.Direction != nil {
		t.Errorf("readout %+v", c.Whale)
	}
}

// Source unavailable: the standard honest offline card (503 source_offline),
// no readout, no blocks.
func TestWhaleSourceOffline(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "down", http.StatusBadGateway)
	}))
	t.Cleanup(srv.Close)
	c := NewAgents(NewBackendClient(srv.URL)).WhaleCard(context.Background())
	if c.effectiveStatus() != statusSourceOffline || c.Whale != nil || c.Blocks != nil {
		t.Errorf("status %d readout %+v blocks %+v", c.effectiveStatus(), c.Whale, c.Blocks)
	}
}

// The data selection is unchanged: the card still asks the backend for the
// newest 10 records.
func TestWhaleRequestUnchanged(t *testing.T) {
	var mu sync.Mutex
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		got = r.URL.Path + "?" + r.URL.RawQuery
		mu.Unlock()
		_, _ = w.Write([]byte(whaleLive15))
	}))
	t.Cleanup(srv.Close)
	NewAgents(NewBackendClient(srv.URL)).WhaleCard(context.Background())
	mu.Lock()
	defer mu.Unlock()
	if got != "/api/v1/whale-flow?limit=10" {
		t.Errorf("request %q", got)
	}
}

// The threshold the card prints is the backend's ingest floor.
func TestWhaleThresholdMatchesBackend(t *testing.T) {
	src, err := os.ReadFile("../whale/worker.go")
	if err != nil {
		t.Fatal(err)
	}
	if !regexp.MustCompile(`transferFetchMinUSD\s*=\s*100_000\.0`).Match(src) {
		t.Fatal("backend ingest floor changed: update whaleThresholdUSD and every text that prints $100K")
	}
	if whaleThresholdUSD != 100_000 || whaleThresholdShown != "$100K" {
		t.Errorf("threshold %v / %q", whaleThresholdUSD, whaleThresholdShown)
	}
}

func TestWhaleBlocksGolden(t *testing.T) {
	c := whaleCardFrom(whaleResp(t, whaleLive15), whaleAt)
	b, err := json.Marshal(c.Blocks)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"what_happened":"The monitor detected 66 BTC transactions ≥ $100K in the 24h to Sep 15 15:00 UTC",` +
		`"why_level":"No price level: $100K is the monitor's size threshold per transaction, not a market level",` +
		`"scenarios":null,"invalidates":null,"regime":"",` +
		`"limitations":"A sample: the 10 newest mempool entries per 10-min poll; outputs include change; no exchange labels",` +
		`"source":"mempool.space recent-transactions feed, polled by the AlphaVizor backend; USD at Binance BTCUSDT"}`
	if string(b) != want {
		t.Errorf("blocks:\n%s\nwant\n%s", b, want)
	}
	// Without a parseable captured_at: no clock time in the sentence.
	body := strings.Replace(whaleZero15, `"captured_at":"2026-09-15T15:00:00Z"`, `"captured_at":""`, 1)
	if got := whaleCardFrom(whaleResp(t, body), whaleAt).Blocks.WhatHappened; got != "The monitor registered no BTC transaction ≥ $100K in the last 24h" {
		t.Errorf("no captured_at: %q", got)
	}
}

func TestWhaleMachineFields(t *testing.T) {
	c := whaleCardFrom(whaleResp(t, whaleLive15), whaleAt)
	raw, err := json.Marshal(cardEnvelope(c))
	if err != nil {
		t.Fatal(err)
	}
	var env struct {
		Whale json.RawMessage `json:"whale"`
		OK    bool            `json:"ok"`
	}
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatal(err)
	}
	want := `{"state":"activity_observed","count":66,"threshold_usd":100000,"window":"24h",` +
		`"window_end":"2026-09-15T15:00:00Z","source":"mempool.space","direction":"not_measurable",` +
		`"time_kind":"first_detected_by_monitor","amount_kind":"total_outputs_incl_change",` +
		`"coverage":null,"last_successful_poll":null,` +
		`"top":{"selection":"largest_btc_among_latest_records","records_requested":10,"records_received":10,"transactions":[` +
		`{"tx_hash":"a1","amount_btc":2465,"amount_usd":187020000,"detected_at":"2026-09-15T15:00:00Z"},` +
		`{"tx_hash":"a3","amount_btc":55.6,"amount_usd":4380000,"detected_at":"2026-09-15T14:50:00Z"},` +
		`{"tx_hash":"a5","amount_btc":50.36,"amount_usd":3930000,"detected_at":"2026-09-15T14:40:00Z"}]}}`
	if string(env.Whale) != want {
		t.Errorf("whale:\n%s\nwant\n%s", env.Whale, want)
	}
	if !env.OK {
		t.Error("activity is a real reading: ok must be true")
	}
	// Zero: an empty list is [], never null.
	z := whaleCardFrom(whaleResp(t, whaleZero15), whaleAt)
	raw, _ = json.Marshal(z.Whale)
	if !strings.Contains(string(raw), `"state":"no_observations","count":0,`) || !strings.Contains(string(raw), `"transactions":[]`) {
		t.Errorf("zero readout %s", raw)
	}
	// Every other agent: no whale object.
	raw, _ = json.Marshal(cardEnvelope(Card{Agent: "Trend Agent"}))
	if strings.Contains(string(raw), `"whale"`) {
		t.Errorf("whale object leaked onto another agent: %s", raw)
	}
}

// Labeled direction (not produced by the mempool feed today, kept for a
// labeled source): the semaphore rule is unchanged, the words carry no
// forecast.
func whaleDirectional(t *testing.T, dir string) Card {
	body := `{"captured_at":"2026-09-15T15:00:00Z",
	  "flows":[{"asset":"BTC","net_flow_usd_24h":-18400000,"direction":"` + dir + `","tx_count_24h":37,"confidence":64,"partial":true,
	            "net_flow_prev_24h":-6200000,"flow_pct":196.8}],
	  "transfers":[
	    {"chain":"BTC","tx_hash":"x1","timestamp":"2026-09-15T14:30:00Z","asset":"BTC","amount_native":150.5,"amount_usd":17000000,"direction":"outflow","exchange":"Binance"},
	    {"chain":"BTC","tx_hash":"x2","timestamp":"2026-09-15T14:00:00Z","asset":"BTC","amount_native":90.1,"amount_usd":10000000,"direction":"inflow","exchange":"Coinbase"}]}`
	return whaleCardFrom(whaleResp(t, body), whaleAt)
}

func TestWhaleDirectionalWording(t *testing.T) {
	in, out := whaleDirectional(t, "inflow"), whaleDirectional(t, "outflow")
	if in.Emoji != emojiBear || out.Emoji != emojiBull {
		t.Errorf("semaphore rule changed: in %s out %s", in.Emoji, out.Emoji)
	}
	if in.Verdict != "Net to labeled exchange wallets over 24h · 37 BTC transactions ≥ $100K seen by the monitor" ||
		out.Verdict != "Net from labeled exchange wallets over 24h · 37 BTC transactions ≥ $100K seen by the monitor" {
		t.Errorf("verdicts %q / %q", in.Verdict, out.Verdict)
	}
	if in.Short != "net to exchanges" || out.Short != "net from exchanges" {
		t.Errorf("shorts %q / %q", in.Short, out.Short)
	}
	if in.Confidence == nil || *in.Confidence != 64 {
		t.Errorf("labeled direction keeps its confidence: %v", in.Confidence)
	}
	f := strings.Join(out.Facts, "\n")
	for _, want := range []string{"Net flow 24h: -$18.40M (37 tx ≥ $100K) — partial data, labeled wallets only",
		"Prior 24h net flow: -$6.20M → +197% change",
		// 112 runes with "not the block time": the labeled line keeps the
		// label and the detection time, in the shorter form.
		"Outputs total 150.50 BTC ≈ $17.00M · from Binance · detected by monitor at Sep 15 14:30 UTC, not block time"} {
		if !strings.Contains(f, want) {
			t.Errorf("missing %q in\n%s", want, f)
		}
	}
	if *out.Whale.Direction != "net_from_exchanges" || *in.Whale.Direction != "net_to_exchanges" {
		t.Errorf("direction %s / %s", *out.Whale.Direction, *in.Whale.Direction)
	}
}

func whaleAllCards(t *testing.T) map[string]Card {
	huge := `{"captured_at":"2026-09-15T15:00:00Z",
	  "flows":[{"asset":"BTC","direction":"neutral","tx_count_24h":2147483647,"partial":true}],
	  "transfers":[{"chain":"BTC","tx_hash":"h","timestamp":"2026-09-15T14:59:00Z","asset":"BTC","amount_native":20999999.99,"amount_usd":9.99e15,"direction":"neutral","exchange":""},
	    {"chain":"BTC","tx_hash":"l","timestamp":"2026-09-15T14:58:00Z","asset":"BTC","amount_native":20999999.99,"amount_usd":9.99e15,"direction":"outflow","exchange":"` + strings.Repeat("VeryLongExchangeName", 5) + `"}]}`
	balanced := strings.Replace(strings.Replace(whaleLive15, `"partial":true`, `"partial":false`, 1), `"source":"mempool"`, `"source":"x"`, 1)
	return map[string]Card{
		"live":       whaleCardFrom(whaleResp(t, whaleLive15), whaleAt),
		"zero":       whaleCardFrom(whaleResp(t, whaleZero15), whaleAt),
		"none_shown": whaleCardFrom(whaleResp(t, whaleNoneShown15), whaleAt),
		"no_snap":    whaleCardFrom(whaleResp(t, whaleNoSnapshot15), whaleAt),
		"huge":       whaleCardFrom(whaleResp(t, huge), whaleAt),
		"inflow":     whaleDirectional(t, "inflow"),
		"outflow":    whaleDirectional(t, "outflow"),
		"no_net":     whaleCardFrom(whaleResp(t, balanced), whaleAt),
		"offline":    offlineCard("Whale Flow Agent", "Whale", "BTC", keyWhale, howTexts[keyWhale]),
	}
}

// whaleLines is every line a reader sees. No digest headline: whale is never
// the digest winner (topSelection picks the trio or macro).
func whaleLines(c Card) []string {
	lines := append([]string{c.Verdict, htmlToPlain(c.OneLiner())}, c.Facts...)
	if b := c.Blocks; b != nil {
		lines = append(lines, b.WhatHappened, b.WhyLevel, b.Regime, b.Limitations, b.Source)
		lines = append(lines, b.Scenarios...)
		if b.Invalidates != nil {
			lines = append(lines, *b.Invalidates)
		}
	}
	return lines
}

func TestWhaleLinesFitEveryPath(t *testing.T) {
	for key, c := range whaleAllCards(t) {
		for _, l := range whaleLines(c) {
			if n := utf8.RuneCountInString(l); n > whaleFactMaxRunes {
				t.Errorf("%s: %d runes > %d: %q", key, n, whaleFactMaxRunes, l)
			}
		}
	}
}

// No flow, trading or forecast words on any path — the monitor sees neither
// exchange direction (on the mempool feed) nor price. Checked on the text a
// reader sees: the Telegram card, the digest line and every block.
func TestWhaleNoFlowOrForecastWords(t *testing.T) {
	banned := []string{"inflow", "outflow", "accumulat", "dump", "whales", "buying", "selling", "sell pressure",
		"buy pressure", "bullish", "bearish", "transferred", "large btc transfers", "· seen ", "smart money",
		" will ", "expect", "likely", "forecast says", "signal"}
	for key, c := range whaleAllCards(t) {
		b, err := json.Marshal(c.Blocks)
		if err != nil {
			t.Fatal(err)
		}
		all := strings.ToLower(c.RenderHTML() + "\n" + htmlToPlain(c.OneLiner()) + "\n" + string(b))
		for _, w := range banned {
			if strings.Contains(all, w) {
				t.Errorf("%s: %q must not appear:\n%s", key, w, all)
			}
		}
		// Never "balanced": neutral from the backend also means "no labeled
		// flows at all" (ClassifyDirection with total 0).
		if strings.Contains(all, "balanced") {
			t.Errorf("%s: a neutral direction is not a balance:\n%s", key, all)
		}
	}
	for _, w := range banned {
		if strings.Contains(strings.ToLower(howTexts[keyWhale]), w) {
			t.Errorf("how-text carries %q: %q", w, howTexts[keyWhale])
		}
	}
}

func TestWhaleHowText(t *testing.T) {
	h := howTexts[keyWhale]
	if n := utf8.RuneCountInString(h); n > 200 {
		t.Errorf("how-text %d > 200 runes: %q", n, h)
	}
	for _, want := range []string{"$100K", "mempool.space", "change", "no exchange direction"} {
		if !strings.Contains(h, want) {
			t.Errorf("how-text lacks %q: %q", want, h)
		}
	}
}

// The body is a function of the payload: the request clock changes nothing
// when captured_at is present (no age, no "N min ago").
func TestWhaleBodyStableAcrossClock(t *testing.T) {
	w := whaleResp(t, whaleLive15)
	a, _ := json.Marshal(cardEnvelope(whaleCardFrom(w, whaleAt)))
	b, _ := json.Marshal(cardEnvelope(whaleCardFrom(w, whaleAt.Add(5*time.Hour))))
	if string(a) != string(b) {
		t.Errorf("body moved with the clock:\n%s\n%s", a, b)
	}
	if regexp.MustCompile(`(?i)\bago\b|\bmin old\b`).Match(a) {
		t.Errorf("no running age on the card: %s", a)
	}
}

// /showcase/example concludes on a whale card with an activity count, never
// a direction ("neutral reading … nothing leans … level structure").
func TestWhaleConclusionNoDirection(t *testing.T) {
	live := whaleCardFrom(whaleResp(t, whaleLive15), whaleAt)
	if got := conclusionFor(live); got != "This is an activity count, not a forecast: the monitor saw 66 BTC transactions ≥ $100K in 24h "+
		"in a small sample (the 10 newest mempool entries per poll); it sees neither exchange direction nor price, so it says nothing about where price goes." {
		t.Errorf("live conclusion %q", got)
	}
	zero := whaleCardFrom(whaleResp(t, whaleZero15), whaleAt)
	if got := conclusionFor(zero); got != "This is an activity count, not a forecast: the monitor registered no BTC transaction ≥ $100K in 24h; "+
		"each poll sees only the 10 newest mempool entries, so that does not mean none happened." {
		t.Errorf("zero conclusion %q", got)
	}
	for key, c := range whaleAllCards(t) {
		got := strings.ToLower(conclusionFor(c))
		for _, bad := range []string{"bullish", "bearish", "lean", "level structure", "inflow", "outflow"} {
			if strings.Contains(got, bad) {
				t.Errorf("%s: conclusion carries %q: %q", key, bad, got)
			}
		}
	}
}

// The AI payload carries no time stamps: the detection times are dropped from
// the whale facts (their hash is the AI cache key).
func TestWhaleAIFactsNoTimes(t *testing.T) {
	g := fakeGathered()
	g.cards[keyWhale] = whaleCardFrom(whaleResp(t, whaleLive15), whaleAt)
	p := aiPayload(g)
	if strings.Contains(p, "UTC") || strings.Contains(p, "Sep 15") {
		t.Errorf("whale detection time leaked into the AI payload: %s", p)
	}
	if !strings.Contains(p, "Outputs total 2465 BTC ≈ $187.02M") {
		t.Errorf("the transaction itself must stay: %s", p)
	}
}

// End to end over HTTP: 200, no Last-Modified, the whale object served.
func TestWhaleHTTPEnvelope(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/whale-flow": whaleLive15})
	_, srv := newTestAPI(t, ag, true)
	status, hdr, body := httpGet(t, srv.URL+"/agents/whale")
	if status != 200 || hdr.Get("Last-Modified") != "" {
		t.Fatalf("status %d Last-Modified %q", status, hdr.Get("Last-Modified"))
	}
	for _, want := range []string{`"whale":{"state":"activity_observed","count":66,`, `"scenarios":null`, `"invalidates":null`,
		`"verdict":"66 BTC transactions ≥ $100K seen by the monitor in 24h — exchange direction not measurable"`} {
		if !strings.Contains(string(body), want) {
			t.Errorf("body lacks %s:\n%s", want, body)
		}
	}
}

// No snapshot: no list, even when the table has BTC records — there is no
// count to sit beside, and the window would hang off the request clock.
func TestWhaleNoSnapshotShowsNoList(t *testing.T) {
	body := `{"captured_at":"","flows":[{"asset":"ETH","direction":"neutral","tx_count_24h":3,"partial":true}],
	  "transfers":[{"chain":"BTC","tx_hash":"b1","timestamp":"2026-09-15T15:00:00Z","asset":"BTC","amount_native":20,"amount_usd":1500000}]}`
	c := whaleCardFrom(whaleResp(t, body), whaleAt)
	want := []string{whaleLineThreshold, whaleLineSample, whaleLineGaps}
	if strings.Join(c.Facts, "\n") != strings.Join(want, "\n") {
		t.Errorf("facts:\n%s", strings.Join(c.Facts, "\n"))
	}
	if len(c.Whale.Top.Transactions) != 0 || c.Whale.Top.RecordsReceived != 1 || c.Whale.State != whaleStateNoSnapshot {
		t.Errorf("readout %+v", c.Whale)
	}
	later := whaleCardFrom(whaleResp(t, body), whaleAt.Add(-30*time.Minute))
	if c.RenderHTML() == "" || strings.Contains(c.RenderHTML(), "Outputs total") ||
		strings.Join(c.Facts, "|") != strings.Join(later.Facts, "|") {
		t.Errorf("no-snapshot facts must not depend on the clock: %v / %v", c.Facts, later.Facts)
	}
}

// A labeled read the backend calls neutral: either the net sits inside its
// neutral band or there are no labeled flows (ClassifyDirection, total 0).
// The words fit both and claim no balance; the semaphore stays neutral.
func TestWhaleNoNetLabeledDirection(t *testing.T) {
	for _, net := range []string{
		`"net_flow_usd_24h":0,"inflow_usd_24h":0,"outflow_usd_24h":0`,
		`"net_flow_usd_24h":1000,"inflow_usd_24h":500500,"outflow_usd_24h":499500`,
	} {
		body := `{"captured_at":"2026-09-15T15:00:00Z","flows":[{"asset":"BTC",` + net +
			`,"direction":"neutral","tx_count_24h":12,"partial":false}],"transfers":[]}`
		c := whaleCardFrom(whaleResp(t, body), whaleAt)
		if c.Verdict != "No net labeled exchange direction over 24h · 12 BTC transactions ≥ $100K seen by the monitor" ||
			c.Short != "no net labeled direction" || c.Emoji != emojiNeutral || c.Confidence != nil {
			t.Errorf("%s: verdict %q short %q emoji %s", net, c.Verdict, c.Short, c.Emoji)
		}
		if *c.Whale.Direction != "no_net_direction" {
			t.Errorf("direction %s", *c.Whale.Direction)
		}
		if all := strings.ToLower(c.RenderHTML()); strings.Contains(all, "balanced") {
			t.Errorf("no balance claim: %s", all)
		}
	}
}

// Hook stability on a mempool-shaped payload (BTC, partial, a non-empty
// transfer list): the hook's own hash does not move with the request clock.
func TestWhaleHookHashStableMempoolShape(t *testing.T) {
	ag := newStubBackend(t, map[string]string{"/api/v1/whale-flow": whaleLive15})
	clk := &stepClock{t: whaleAt}
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
	clk.set(whaleAt.Add(3 * time.Hour))
	st2, b2, h2 := read()
	if st1 != 200 || st2 != 200 {
		t.Fatalf("status %d / %d", st1, st2)
	}
	if !strings.Contains(string(b1), "Outputs total 2465 BTC") || !strings.Contains(string(b1), `"direction":"not_measurable"`) {
		t.Fatalf("fixture must be the mempool shape with a list: %.600s", b1)
	}
	if h1 != h2 {
		t.Errorf("hook hash moved with unchanged data:\n%s\n%s", b1, b2)
	}
}
