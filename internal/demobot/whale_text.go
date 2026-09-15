package demobot

// whale_text.go — everything the Whale card SAYS (stage 1, 2026-09-15).
//
// The rules are unchanged: the backend ingests BTC transactions from the
// mempool.space recent feed whose total outputs are worth at least $100K at
// the Binance BTCUSDT price when detected (whale/worker.go
// transferFetchMinUSD), counts them over the 24h to its snapshot
// (tx_count_24h), and the card asks for the newest 10 records of the table
// (every chain) and shows the 3 largest BTC ones inside that 24h. What
// changed is only what the card claims:
//
//   - the count is "transactions ≥ $100K seen by the monitor", not "large BTC
//     transfers": it is what periodic polls of a bounded recent list caught,
//     not every such transaction on the network;
//   - a time is when the monitor first detected the transaction (the poll
//     time the backend stamps), not the block time;
//   - a size is the transaction's total outputs, change to the sender
//     included — not an amount that changed hands;
//   - the list is the largest BTC among the latest 10 records received, not a
//     24h top: the backend has no 24h-by-size query;
//   - zero reads "the monitor registered none", never "no transfers";
//   - the scale of the sample is named: each 10-minute poll sees only the 10
//     newest mempool.space entries, so most transactions ≥ $100K are never
//     seen; poll coverage and the last successful poll are not served by the
//     backend, so a missed poll lowers the count unmarked (the readout keeps
//     both null);
//   - no flow, trading or forecast words; every line fits whaleFactMaxRunes.

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	whaleFactMaxRunes = 110
	// whaleThresholdUSD is the backend's ingest floor (whale/worker.go
	// transferFetchMinUSD, pinned by TestWhaleThresholdMatchesBackend).
	whaleThresholdUSD   = 100_000.0
	whaleThresholdShown = "$100K"
	// whaleFeedLimit is the number of newest records the card asks for — the
	// data selection, unchanged.
	whaleFeedLimit = 10
	whaleWindow    = 24 * time.Hour
	whaleTopN      = 3
	// whaleMonitorPoll is the backend's whale poll interval (cmd/server
	// main.go, the whale Worker's RefreshInterval; pinned by
	// TestWhaleSampleMatchesBackend).
	whaleMonitorPoll = 10 * time.Minute
	// whaleMempoolRecentSize is how many entries one call of mempool.space
	// /api/mempool/recent returns: the external API's behaviour (a live call
	// on 2026-09-15 returned 10), not a constant of our code. Each poll sees
	// only these, so the count is a small sample of the network.
	whaleMempoolRecentSize = 10
)

// Whale machine states (whale.state). "Source unavailable" is the standard
// offline card (503, source_offline) and carries no readout.
const (
	whaleStateActivity   = "activity_observed"
	whaleStateNone       = "no_observations"
	whaleStateNoSnapshot = "no_snapshot"
)

// Exchange direction (whale.direction). The mempool feed carries no exchange
// labels, so a BTC snapshot from it is always not_measurable; the other
// values exist for a labeled source.
const (
	whaleDirNotMeasurable = "not_measurable"
	whaleDirTo            = "net_to_exchanges"
	whaleDirFrom          = "net_from_exchanges"
	// whaleDirNoNet: a labeled read with no net direction — the net is inside
	// the backend's neutral band, or there are no labeled flows at all
	// (ClassifyDirection returns neutral for both). Never "balanced".
	whaleDirNoNet = "no_net_direction"
)

// WhaleReadout is the envelope's "whale" object (whale cards with a backend
// answer, additive 2026-09-15; docs/demobot-http.md "Whale card").
type WhaleReadout struct {
	State        string  `json:"state"`         // activity_observed | no_observations | no_snapshot
	Count        *int    `json:"count"`         // tx_count_24h; null on no_snapshot
	ThresholdUSD float64 `json:"threshold_usd"` // per transaction, total outputs at the spot when detected
	Window       string  `json:"window"`        // always "24h"
	WindowEnd    *string `json:"window_end"`    // the snapshot's captured_at; null when missing/unparsable
	Source       string  `json:"source"`        // always "mempool.space"
	Direction    *string `json:"direction"`     // not_measurable | net_to_exchanges | net_from_exchanges | no_net_direction; null on no_snapshot
	TimeKind     string  `json:"time_kind"`     // always "first_detected_by_monitor"
	AmountKind   string  `json:"amount_kind"`   // always "total_outputs_incl_change"
	// Coverage and LastSuccessfulPoll are always null: the backend serves no
	// poll statistics (its captured_at is the worker tick, written even when
	// the mempool poll failed).
	Coverage           any      `json:"coverage"`
	LastSuccessfulPoll *string  `json:"last_successful_poll"`
	Top                WhaleTop `json:"top"`
}

// WhaleTop is how the listed transactions were picked.
type WhaleTop struct {
	Selection        string    `json:"selection"`         // always "largest_btc_among_latest_records"
	RecordsRequested int       `json:"records_requested"` // 10
	RecordsReceived  int       `json:"records_received"`  // records the backend returned (every chain)
	Transactions     []WhaleTx `json:"transactions"`      // the listed ones, largest first; [] when none
}

// WhaleTx is one listed transaction at raw precision.
type WhaleTx struct {
	TxHash     string  `json:"tx_hash"`
	AmountBTC  float64 `json:"amount_btc"` // total outputs, change included
	AmountUSD  float64 `json:"amount_usd"`
	DetectedAt string  `json:"detected_at"` // first detected by the monitor, not the block time
}

const (
	whaleLineThreshold = "Threshold: " + whaleThresholdShown + " per transaction, priced at the Binance BTCUSDT price when detected"
	whaleLineDirection = "Exchange direction: not measurable (these BTC wallets carry no exchange labels)"
	// whaleLineGaps: the backend serves no poll statistics, so a missed poll —
	// or a poll with no Binance price, whose rows fail the $100K floor at $0
	// (whale/source_mempool.go) — lowers the count with nothing to say so.
	whaleLineGaps = "Missed polls or a missing BTC price lower this count unmarked: poll coverage is not served"

	whaleWhyLevel    = "No price level: " + whaleThresholdShown + " is the monitor's size threshold per transaction, not a market level"
	whaleBlockSource = "mempool.space recent-transactions feed, polled by the AlphaVizor backend; USD at Binance BTCUSDT"
)

var (
	whalePollMinutes = int(whaleMonitorPoll / time.Minute)
	// whaleLineSample names the scale of the sample: the count is what these
	// polls caught, not the network's activity.
	whaleLineSample = fmt.Sprintf("Each %d-minute poll sees only the %d newest mempool.space entries; most transactions ≥ %s are never seen",
		whalePollMinutes, whaleMempoolRecentSize, whaleThresholdShown)
	whaleLimitations = fmt.Sprintf("A sample: the %d newest mempool entries per %d-min poll; outputs include change; no exchange labels",
		whaleMempoolRecentSize, whalePollMinutes)
)

// whaleFit joins parts with " · ", dropping trailing parts until the line
// fits; a single part still too long is cut with "…".
func whaleFit(parts ...string) string {
	for len(parts) > 1 && utf8.RuneCountInString(strings.Join(parts, " · ")) > whaleFactMaxRunes {
		parts = parts[:len(parts)-1]
	}
	s := strings.Join(parts, " · ")
	if r := []rune(s); len(r) > whaleFactMaxRunes {
		s = string(r[:whaleFactMaxRunes-1]) + "…"
	}
	return s
}

// whaleTxCount: "1 BTC transaction", "66 BTC transactions".
func whaleTxCount(n int) string {
	if n == 1 {
		return "1 BTC transaction"
	}
	return fmt.Sprintf("%d BTC transactions", n)
}

func whaleRecords(n int) string {
	if n == 1 {
		return "1 monitor record"
	}
	return fmt.Sprintf("%d monitor records", n)
}

func whaleClock(t time.Time) string { return t.UTC().Format("Jan 2 15:04") + " UTC" }

func whaleLabeled(t WhaleTransfer) bool {
	return t.Exchange != "" && (t.Direction == "inflow" || t.Direction == "outflow")
}

// whaleTxLine is one listed transaction. The detection-time part is never
// dropped for the exchange label: a labeled line that does not fit loses the
// label.
func whaleTxLine(t WhaleTransfer) string {
	amt := fmt.Sprintf("Outputs total %s BTC ≈ %s", trimFloat(t.AmountNative), usd(t.AmountUSD))
	when := "detected by the monitor at " + whaleClock(t.Timestamp) + ", not the block time"
	if whaleLabeled(t) {
		side := "to "
		if t.Direction == "outflow" {
			side = "from "
		}
		short := "detected by monitor at " + whaleClock(t.Timestamp) + ", not block time"
		for _, w := range []string{when, short} {
			if l := strings.Join([]string{amt, side + t.Exchange, w}, " · "); utf8.RuneCountInString(l) <= whaleFactMaxRunes {
				return l
			}
		}
	}
	return whaleFit(amt, when)
}

// whaleDetectedRe matches the detection-time part of a listed transaction.
var whaleDetectedRe = regexp.MustCompile(` · detected by (the )?monitor at [A-Z][a-z]{2} \d{1,2} \d{2}:\d{2} UTC, not (the )?block time`)

// whaleAIFacts drops the detection times from the AI payload: the payload
// carries no time stamps (see momentumAIFacts).
func whaleAIFacts(facts []string) []string {
	out := make([]string, len(facts))
	for i, f := range facts {
		out[i] = whaleDetectedRe.ReplaceAllString(f, "")
	}
	return out
}

// whaleCardFrom builds the card from one backend answer. Pure: `now` is used
// only when the snapshot carries no parseable captured_at (the 24h window then
// ends at the request clock, as before).
func whaleCardFrom(w *WhaleResp, now time.Time) Card {
	c := Card{
		Agent:      "Whale Flow Agent",
		ShortName:  "Whale",
		Asset:      "BTC",
		Command:    keyWhale,
		HowItWorks: howTexts[keyWhale],
		DataTime:   parseWhen(w.CapturedAt),
		SourceNote: "data: mempool.space",
		// No validator, ever: the transfers come from the backend's live table
		// (the newest `limit` rows), not from the snapshot captured_at names —
		// a new transfer pushes an old one out of the list under the same
		// captured_at. The list window is the snapshot's 24h, like the
		// backend's own counter, and a transfer stamped after captured_at is
		// not shown: the body must not carry data newer than its data_as_of.
		noValidator: true,
	}
	ro := &WhaleReadout{ThresholdUSD: whaleThresholdUSD, Window: "24h", Source: "mempool.space",
		TimeKind: "first_detected_by_monitor", AmountKind: "total_outputs_incl_change",
		Top: WhaleTop{Selection: "largest_btc_among_latest_records", RecordsRequested: whaleFeedLimit,
			RecordsReceived: len(w.Transfers), Transactions: []WhaleTx{}}}
	c.Whale = ro

	windowEnd, capErr := time.Parse(time.RFC3339, w.CapturedAt)
	if capErr != nil {
		windowEnd = now
	} else {
		end := windowEnd.UTC().Format(time.RFC3339)
		ro.WindowEnd = &end
	}

	var btc *WhaleFlow
	for i := range w.Flows {
		if w.Flows[i].Asset == "BTC" {
			btc = &w.Flows[i]
			break
		}
	}
	if btc == nil {
		// No list: without a snapshot there is no count to sit beside, and
		// without its captured_at the window would hang off the request clock.
		ro.State = whaleStateNoSnapshot
		c.Emoji, c.Verdict, c.Short = emojiNeutral, "No BTC count from the monitor yet", "no data"
		c.Status = statusNoData // upstream alive, nothing to read yet
		c.Facts = append(c.Facts, whaleLineThreshold, whaleLineSample, whaleLineGaps)
		return c
	}

	shown := whaleShown(w.Transfers, windowEnd)
	for _, t := range shown {
		ro.Top.Transactions = append(ro.Top.Transactions, WhaleTx{TxHash: t.TxHash, AmountBTC: t.AmountNative,
			AmountUSD: t.AmountUSD, DetectedAt: t.Timestamp.UTC().Format(time.RFC3339)})
	}

	n := btc.TxCount24h
	count := n
	ro.Count = &count
	ro.State = whaleStateNone
	if n > 0 {
		ro.State = whaleStateActivity
	}
	dir := whaleDirNoNet
	seen := fmt.Sprintf("%s ≥ %s seen by the monitor", whaleTxCount(n), whaleThresholdShown)
	switch {
	case btc.Direction == "inflow":
		dir = whaleDirTo
		c.Emoji, c.Verdict, c.Short = emojiBear, "Net to labeled exchange wallets over 24h · "+seen, "net to exchanges"
	case btc.Direction == "outflow":
		dir = whaleDirFrom
		c.Emoji, c.Verdict, c.Short = emojiBull, "Net from labeled exchange wallets over 24h · "+seen, "net from exchanges"
	case btc.Partial && n == 0:
		dir = whaleDirNotMeasurable
		c.Emoji = emojiNeutral
		c.Verdict = "The monitor registered no BTC transaction ≥ " + whaleThresholdShown + " in 24h"
		c.Short = "none ≥ " + whaleThresholdShown + " registered"
	case btc.Partial:
		// The BTC feed (public mempool) never labels exchange wallets, so the
		// backend's net flow is structurally $0 and its direction "neutral".
		// That is "not measurable", not "balanced".
		dir = whaleDirNotMeasurable
		c.Emoji = emojiNeutral
		c.Verdict = seen + " in 24h — exchange direction not measurable"
		c.Short = fmt.Sprintf("%d tx ≥ %s seen, direction n/a", n, whaleThresholdShown)
	default:
		// Neutral from the backend means either a net inside its neutral band
		// or no labeled flows at all (ClassifyDirection, total == 0): the
		// words fit both, and claim no balance.
		c.Emoji, c.Verdict, c.Short = emojiNeutral, "No net labeled exchange direction over 24h · "+seen, "no net labeled direction"
	}
	ro.Direction = &dir
	directional := dir == whaleDirTo || dir == whaleDirFrom
	if btc.Confidence > 0 && directional {
		conf := btc.Confidence
		c.Confidence = &conf
	}

	c.Facts = append(c.Facts, whaleLineThreshold)
	if dir == whaleDirNotMeasurable {
		c.Facts = append(c.Facts, whaleLineDirection)
	} else {
		flowLine := fmt.Sprintf("Net flow 24h: %s (%d tx ≥ %s)", usd(btc.NetFlowUSD24h), n, whaleThresholdShown)
		if btc.Partial {
			flowLine += " — partial data, labeled wallets only"
		}
		c.Facts = append(c.Facts, whaleFit(flowLine))
	}
	// Baseline comparison — only when the payload actually carries a prior-24h
	// figure (a zero baseline means "no snapshot to compare").
	if btc.NetFlowPrev24h != nil && *btc.NetFlowPrev24h != 0 {
		base := fmt.Sprintf("Prior 24h net flow: %s", usd(*btc.NetFlowPrev24h))
		if btc.FlowPct != nil {
			base += fmt.Sprintf(" → %+.0f%% change", *btc.FlowPct)
		}
		c.Facts = append(c.Facts, whaleFit(base))
	}
	c.Facts = append(c.Facts, whaleLineSample, whaleLineGaps)
	if len(shown) > 0 {
		c.Facts = append(c.Facts, whaleTopLines(shown, len(w.Transfers))...)
	} else if n > 0 {
		c.Facts = append(c.Facts, fmt.Sprintf("None of the latest %s received is a BTC transaction of this 24h window",
			whaleRecords(len(w.Transfers))))
	}

	c.Blocks = whaleBlocks(n, dir, ro.WindowEnd != nil, windowEnd)
	return c
}

// whaleShown is the list: BTC records inside (windowEnd−24h, windowEnd], the
// largest by USD first, at most whaleTopN. Stable on ties (served order).
func whaleShown(transfers []WhaleTransfer, windowEnd time.Time) []WhaleTransfer {
	cutoff := windowEnd.Add(-whaleWindow)
	var recent []WhaleTransfer
	for _, t := range transfers {
		if t.Chain == "BTC" && t.Timestamp.After(cutoff) && !t.Timestamp.After(windowEnd) {
			recent = append(recent, t)
		}
	}
	sort.SliceStable(recent, func(i, j int) bool { return recent[i].AmountUSD > recent[j].AmountUSD })
	if len(recent) > whaleTopN {
		recent = recent[:whaleTopN]
	}
	return recent
}

func whaleTopLines(shown []WhaleTransfer, received int) []string {
	if len(shown) == 0 {
		return nil
	}
	out := []string{fmt.Sprintf("Largest BTC among the latest %s received, not a 24h top · outputs include change",
		whaleRecords(received))}
	for _, t := range shown {
		out = append(out, whaleTxLine(t))
	}
	return out
}

// whaleBlocks is the content-ready form: what the monitor detected, what the
// threshold is, what the data cannot say and where it comes from. No
// scenarios, nothing to invalidate and no regime: the agent sees neither
// price nor (on the mempool feed) exchange direction.
func whaleBlocks(n int, dir string, hasEnd bool, windowEnd time.Time) *ContentBlocks {
	window := "in the last 24h"
	if hasEnd {
		window = "in the 24h to " + whaleClock(windowEnd)
	}
	what := fmt.Sprintf("The monitor detected %s ≥ %s %s", whaleTxCount(n), whaleThresholdShown, window)
	if n == 0 {
		what = fmt.Sprintf("The monitor registered no BTC transaction ≥ %s %s", whaleThresholdShown, window)
	}
	switch dir {
	case whaleDirTo:
		what = whaleFit(what, "net to labeled exchange wallets")
	case whaleDirFrom:
		what = whaleFit(what, "net from labeled exchange wallets")
	}
	return &ContentBlocks{WhatHappened: what, WhyLevel: whaleWhyLevel, Limitations: whaleLimitations, Source: whaleBlockSource}
}

// whaleConclusion is the /showcase/example conclusion of a whale card: an
// activity count, never a direction. The generic sentence called it "a
// neutral reading … nothing leans either way", which an unmeasured direction
// cannot say.
func whaleConclusion(c Card) string {
	ro := c.Whale
	if ro == nil {
		return "The monitor's source is offline right now, so there is nothing to read."
	}
	if ro.State == whaleStateNoSnapshot || ro.Count == nil || ro.Direction == nil {
		return "The monitor has no BTC count yet, so there is nothing to read."
	}
	n := *ro.Count
	switch {
	case *ro.Direction != whaleDirNotMeasurable:
		return fmt.Sprintf("This is a labeled-wallet estimate, not a forecast: the monitor saw %s ≥ %s in 24h; "+
			"it says nothing about where price goes.", whaleTxCount(n), whaleThresholdShown)
	case n == 0:
		return fmt.Sprintf("This is an activity count, not a forecast: the monitor registered no BTC transaction ≥ %s in 24h; "+
			"each poll sees only the %d newest mempool entries, so that does not mean none happened.",
			whaleThresholdShown, whaleMempoolRecentSize)
	}
	return fmt.Sprintf("This is an activity count, not a forecast: the monitor saw %s ≥ %s in 24h in a small sample "+
		"(the %d newest mempool entries per poll); it sees neither exchange direction nor price, so it says nothing about where price goes.",
		whaleTxCount(n), whaleThresholdShown, whaleMempoolRecentSize)
}
