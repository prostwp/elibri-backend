package demobot

// whale_labeled.go — what the Whale card SAYS when the backend read a LABELED
// source (stage 2, 2026-09-16).
//
// The rules are unchanged and all live in the backend: the $100K ingest floor
// (whale/worker.go transferFetchMinUSD), the 24h window, the choice of source
// (cmd/server/main.go, by ETHERSCAN_API_KEY), ClassifyDirection's neutral band
// and the semaphore. This file only chooses words for a reading the card could
// not describe before.
//
// What makes a reading "labeled": the backend attributes a transfer to an
// exchange by looking its counterparty up in the ExchangeRegistry
// (whale/registry.go), which holds ETH-chain hot wallets only, and stamps the
// resulting snapshot with its source badge. So a flow is labeled when it is not
// BTC AND carries that badge — the badge is necessary, never merely suggestive.
// BTC is deliberately absent from that registry: its snapshot comes from the
// public mempool feed and keeps the card it already had. The assets are taken
// from the response (ETH, USDT, USDC today, whatever the backend watches
// tomorrow), never hardcoded.
//
// What the card claims, and what it refuses to:
//
//   - the net flow is an ESTIMATE over the wallets in our registry, never the
//     market's exchange flow: the backend marks every free-tier snapshot
//     partial (whale/worker.go, partial := true) and the card says so in
//     words, not as a badge;
//   - a direction is a FLOW of coins, never a price call: no forecast, no
//     advice, and the semaphore is the backend's existing rule;
//   - only a COIN may lead the card or colour it. A stablecoin deposit is
//     conventionally read as buying power — the opposite of what the backend's
//     coin-oriented rule implies — so stablecoin flows are reported with their
//     numbers but never set the headline, the lead asset or the semaphore;
//   - a non-zero net the backend still calls neutral is NOT a direction — and
//     the card names the band only when the share really is under it;
//   - stablecoins are named (USDT, USDC), never folded into "crypto";
//   - address coverage is not reported by the backend, so a labeled address
//     that failed to load lowers these numbers unmarked, and the card says so;
//   - when the labeled source returns nothing at all, that is a dead source,
//     not a measurement: the card falls back to the BTC monitor and discloses
//     the silence;
//   - mempool wording (poll sampling, total outputs with change, the BTC
//     price) belongs to the BTC path only and never appears here;
//   - every line fits whaleFactMaxRunes.

import (
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode/utf8"
)

// Which source produced the reading the card is showing (whale.read_source).
const (
	whaleReadBTCMonitor = "btc_mempool_monitor"
	whaleReadLabeled    = "labeled_exchange_wallets"
)

// Per-flow provenance in the readout (whale.flows[].source_kind).
const (
	whaleSourceKindBTC     = "btc_monitor"
	whaleSourceKindLabeled = "labeled"
)

const (
	// whaleLabeledSourceBadge is the backend's own source badge for a snapshot
	// read through the ExchangeRegistry (whale/worker.go sourceNameForAsset →
	// EtherscanSource.Name()). Without ETHERSCAN_API_KEY no snapshot carries
	// it, so the card stays on the BTC monitor. It is a NECESSARY condition:
	// an unbadged snapshot is unattributed, whatever counts it carries.
	whaleLabeledSourceBadge = "etherscan"

	// whaleNeutralBandPct is the backend's neutral band as a percentage: the
	// net must reach this share of the gross flow before ClassifyDirection
	// names a direction (whale/scorer.go neutralThreshold = 0.1, pinned by
	// TestWhaleLabeledBandLineOnlyWhenUnderBand). The card prints it only when
	// the share really is under it; it does not re-implement the rule.
	whaleNeutralBandPct = 10

	// whaleLabeledTopExchanges caps how many exchange labels one line names.
	whaleLabeledTopExchanges = 4
)

const (
	whaleLabeledSourceNote = "data: Etherscan labeled exchange wallets"
	whaleLabeledReadoutSrc = "etherscan labeled exchange wallets"

	whaleLabeledLineEstimate = "An estimate over the wallets in our registry, not the whole market: unlisted wallets are invisible"
	// whaleLabeledLineCoverage: the backend logs per-address fetch failures
	// but serves none of them, so a dropped address is invisible in these
	// numbers — the same honesty the BTC card keeps about missed polls.
	whaleLabeledLineCoverage = "Address coverage is not reported: a labeled address that failed to load lowers these numbers unmarked"
	whaleLabeledLineWindow   = "Threshold: " + whaleThresholdShown + " per transfer at the price when recorded · window: the 24h to the snapshot"

	// whaleLineLabeledSilent is shown on the BTC monitor card when the labeled
	// source returned empty snapshots for every asset. A total Etherscan
	// failure looks exactly like "nothing moved" in the data (tx 0, net 0,
	// badge present), and the two must never be confused.
	whaleLineLabeledSilent = "The labeled source returned nothing this tick; address coverage is not reported"

	whaleLabeledWhyLevel    = "No price level: " + whaleThresholdShown + " is the size threshold per transfer, not a market level"
	whaleLabeledLimitations = "An estimate over listed wallets only; address coverage is not reported; stablecoins valued at $1"
	whaleLabeledBlockSource = "Etherscan labeled exchange-wallet transfers, polled by the AlphaVizor backend; ETH at Binance ETHUSDT"

	// whaleBTCMonitorHow is the BTC monitor's own description, character for
	// character as main shipped it. The catalog line (howTexts[keyWhale]) now
	// covers both sources, which is right for a list entry and wrong on a
	// monitor card: that card shows no labeled flow at all.
	whaleBTCMonitorHow = "Counts BTC transactions of $100K+ that backend polls of the mempool.space recent feed detected in 24h. " +
		"Sizes are total outputs, change included; no exchange direction."

	// whaleLabeledHow is the labeled card's own description.
	whaleLabeledHow = "Sums transfers of $100K+ between whale wallets and exchange wallets we have labeled (ETH, USDT, USDC via Etherscan) over 24h. An estimate over listed wallets, not the whole market."
)

// whaleStablecoins is the explicit set of assets treated as dollar tokens.
//
// Why it exists: the backend's semaphore rule was written for COINS — coins
// moving to an exchange read as supply arriving, coins leaving as supply
// locked away. A stablecoin deposit is the opposite kind of event: dollars
// arriving on an exchange are conventionally read as buying power. Colouring a
// card red because $61M of USDT landed on exchanges would tell the reader the
// opposite of what most of the market would take from it, so stablecoins are
// reported with their numbers and never set the headline or the colour.
//
// USDT and USDC are the two the backend actually ingests and values at ≈$1
// (stableTokens in whale/source_etherscan.go, pinned by TestWhaleStablecoinSet).
// The rest are listed so that adding one to the backend's registry cannot
// silently promote a dollar token to the headline before anyone revisits this.
var whaleStablecoins = map[string]bool{
	"USDT": true, "USDC": true, "DAI": true, "BUSD": true,
	"TUSD": true, "FDUSD": true, "PYUSD": true, "USDE": true, "USDS": true,
}

func whaleIsStablecoin(asset string) bool { return whaleStablecoins[strings.ToUpper(asset)] }

// WhaleFlowRead is one asset's reading in the readout (whale.flows[]). It is
// the backend's own snapshot, re-labeled with which source produced it —
// additive, and served on both paths.
type WhaleFlowRead struct {
	Asset         string  `json:"asset"`
	NetFlowUSD24h float64 `json:"net_flow_usd_24h"`
	// Direction is the card's vocabulary, not the backend's raw enum:
	// net_to_exchanges | net_from_exchanges | no_net_direction, and
	// not_measurable for the unlabeled BTC monitor.
	Direction  string `json:"direction"`
	TxCount    int    `json:"tx_count"`
	SourceKind string `json:"source_kind"` // labeled | btc_monitor
	Partial    bool   `json:"partial"`
}

// whaleIsLabeledFlow reports whether one flow is a labeled-wallet reading: not
// BTC, and carrying the backend's labeled source badge. Nothing else qualifies
// — a snapshot without the badge was not attributed to exchange wallets, so its
// counts and net are not a labeled reading however large they are.
func whaleIsLabeledFlow(f WhaleFlow) bool {
	if strings.EqualFold(f.Asset, "BTC") {
		return false
	}
	return f.Source == whaleLabeledSourceBadge
}

func whaleDirectionalRaw(dir string) bool { return dir == "inflow" || dir == "outflow" }

// whaleLabeledFlowsOf returns the labeled flows in the response's own order.
func whaleLabeledFlowsOf(flows []WhaleFlow) []WhaleFlow {
	var out []WhaleFlow
	for _, f := range flows {
		if whaleIsLabeledFlow(f) {
			out = append(out, f)
		}
	}
	return out
}

// whaleLabeledAllEmpty reports whether every labeled snapshot is empty — no
// transfers and no net. That is what a total source failure looks like, and it
// is indistinguishable in the data from a genuinely quiet day, so the card
// refuses to read it as either and falls back to the BTC monitor.
func whaleLabeledAllEmpty(flows []WhaleFlow) bool {
	for _, f := range flows {
		if f.TxCount24h != 0 || f.NetFlowUSD24h != 0 {
			return false
		}
	}
	return true
}

// whaleLabeledAnyStamp reports whether any labeled snapshot carries its own
// captured_at, so a labeled window can be dated without borrowing the BTC tick.
func whaleLabeledAnyStamp(flows []WhaleFlow) bool {
	for _, f := range flows {
		if !f.CapturedAt.IsZero() {
			return true
		}
	}
	return false
}

// whaleFlowHasActivity: the asset moved something this window.
func whaleFlowHasActivity(f WhaleFlow) bool {
	return f.TxCount24h > 0 || f.NetFlowUSD24h != 0
}

// whaleReadFlows maps every served flow into the readout, naming the source
// that produced it. Served on both paths so a consumer can see the whole
// answer, not just the headline reading.
func whaleReadFlows(flows []WhaleFlow) []WhaleFlowRead {
	out := make([]WhaleFlowRead, 0, len(flows))
	for _, f := range flows {
		kind, dir := whaleSourceKindBTC, whaleDirNotMeasurable
		if whaleIsLabeledFlow(f) {
			kind, dir = whaleSourceKindLabeled, whaleDirNoNet
		}
		switch f.Direction {
		case "inflow":
			dir = whaleDirTo
		case "outflow":
			dir = whaleDirFrom
		}
		out = append(out, WhaleFlowRead{Asset: f.Asset, NetFlowUSD24h: f.NetFlowUSD24h,
			Direction: dir, TxCount: f.TxCount24h, SourceKind: kind, Partial: f.Partial})
	}
	return out
}

// whaleLabeledCoinLead picks the headline: the COIN whose net carries a
// direction, largest absolute net first. Stablecoins are skipped — see
// whaleStablecoins. Returns nil when no coin carries a direction.
func whaleLabeledCoinLead(flows []WhaleFlow) *WhaleFlow {
	var lead *WhaleFlow
	for i := range flows {
		if whaleIsStablecoin(flows[i].Asset) || !whaleDirectionalRaw(flows[i].Direction) {
			continue
		}
		if lead == nil || whaleAbs(flows[i].NetFlowUSD24h) > whaleAbs(lead.NetFlowUSD24h) {
			lead = &flows[i]
		}
	}
	return lead
}

// whaleLabeledStableLead is the largest directional STABLECOIN flow. It can
// carry the verdict sentence when no coin has a direction, but never the
// semaphore, the confidence or the lead asset.
func whaleLabeledStableLead(flows []WhaleFlow) *WhaleFlow {
	var lead *WhaleFlow
	for i := range flows {
		if !whaleIsStablecoin(flows[i].Asset) || !whaleDirectionalRaw(flows[i].Direction) {
			continue
		}
		if lead == nil || whaleAbs(flows[i].NetFlowUSD24h) > whaleAbs(lead.NetFlowUSD24h) {
			lead = &flows[i]
		}
	}
	return lead
}

func whaleAbs(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

// whaleSide is the flow's direction as a preposition, never a price word.
func whaleSide(dir string) string {
	if dir == "outflow" {
		return "from"
	}
	return "to"
}

func whaleTransfersWord(n int) string {
	if n == 1 {
		return "1 transfer"
	}
	return fmt.Sprintf("%d transfers", n)
}

// whaleLabeledCardFrom builds the labeled card. Pure: respEnd is the response's
// own captured_at (the newest tick across all assets), used only as a fallback
// anchor — the window is dated by the LEAD labeled snapshot's own captured_at
// whenever it has one, so a stale Etherscan read is never stamped with a fresh
// BTC time.
func whaleLabeledCardFrom(w *WhaleResp, ro *WhaleReadout, respEnd time.Time) Card {
	labeled := whaleLabeledFlowsOf(w.Flows)
	lead := whaleLabeledCoinLead(labeled)
	stable := whaleLabeledStableLead(labeled)

	totalTx := 0
	assets := make([]string, 0, len(labeled))
	active := make([]string, 0, len(labeled))
	for _, f := range labeled {
		totalTx += f.TxCount24h
		assets = append(assets, f.Asset)
		if whaleFlowHasActivity(f) {
			active = append(active, f.Asset)
		}
	}

	// The snapshot that dates the reading: the lead coin, else the stablecoin
	// carrying the sentence, else the first labeled asset with any activity.
	headline := lead
	if headline == nil {
		headline = stable
	}
	if headline == nil {
		for i := range labeled {
			if whaleFlowHasActivity(labeled[i]) {
				headline = &labeled[i]
				break
			}
		}
	}

	anchor, stamped := respEnd, false
	if headline != nil && !headline.CapturedAt.IsZero() {
		anchor, stamped = headline.CapturedAt.UTC(), true
	} else if respEnd.IsZero() {
		// No response stamp either: take any labeled stamp we have rather than
		// anchoring the list on nothing.
		for i := range labeled {
			if !labeled[i].CapturedAt.IsZero() {
				anchor, stamped = labeled[i].CapturedAt.UTC(), true
				break
			}
		}
	}

	c := Card{
		Agent:      "Whale Flow Agent",
		ShortName:  "Whale",
		Asset:      whaleLabeledAssetLabel(active, lead),
		Command:    keyWhale,
		HowItWorks: whaleLabeledHow,
		DataTime:   anchor,
		SourceNote: whaleLabeledSourceNote,
		// No validator, for the same reason the BTC card carries none: the
		// listed transfers come from the backend's live table, which can change
		// under an unchanged captured_at.
		noValidator: true,
	}
	c.AssetKey = c.Asset

	ro.ReadSource = whaleReadLabeled
	ro.Source = whaleLabeledReadoutSrc
	ro.TimeKind = "block_time"
	ro.AmountKind = "transfer_amount"
	ro.Top.Selection = "largest_labeled_among_latest_records"
	count := totalTx
	ro.Count = &count
	// The window is dated only when the lead snapshot dated itself.
	ro.WindowEnd = nil
	if stamped {
		end := anchor.Format(time.RFC3339)
		ro.WindowEnd = &end
	}
	if lead != nil {
		leadAsset := lead.Asset
		ro.LeadAsset = &leadAsset
	}

	// ── verdict, semaphore, confidence ──────────────────────────────────────
	// Only a coin direction colours the card; the backend's rule for WHICH
	// colour is untouched (net to exchanges 🔴, net from exchanges 🟢).
	dir := whaleDirNoNet
	switch {
	case lead != nil && lead.Direction == "inflow":
		dir = whaleDirTo
		c.Emoji = emojiBear
		c.Verdict = fmt.Sprintf("Net to labeled exchange wallets over 24h: %s %s (%s)",
			lead.Asset, usd(whaleAbs(lead.NetFlowUSD24h)), whaleTransfersWord(lead.TxCount24h))
		c.Short = fmt.Sprintf("%s net to exchanges %s", lead.Asset, usd(whaleAbs(lead.NetFlowUSD24h)))
	case lead != nil:
		dir = whaleDirFrom
		c.Emoji = emojiBull
		c.Verdict = fmt.Sprintf("Net from labeled exchange wallets over 24h: %s %s (%s)",
			lead.Asset, usd(whaleAbs(lead.NetFlowUSD24h)), whaleTransfersWord(lead.TxCount24h))
		c.Short = fmt.Sprintf("%s net from exchanges %s", lead.Asset, usd(whaleAbs(lead.NetFlowUSD24h)))
	case stable != nil:
		// A dollar token moved, no coin did. Reported with its numbers, with no
		// colour and no direction verdict: what it means for price is exactly
		// the reading this card refuses to make.
		c.Emoji = emojiNeutral
		c.Verdict = whaleFit(fmt.Sprintf("Stablecoin flow %s labeled exchange wallets over 24h: %s %s (%s)",
			whaleSide(stable.Direction), stable.Asset, usd(whaleAbs(stable.NetFlowUSD24h)),
			whaleTransfersWord(stable.TxCount24h)), "no coin direction")
		c.Short = fmt.Sprintf("stablecoins %s exchanges %s",
			whaleSide(stable.Direction), usd(whaleAbs(stable.NetFlowUSD24h)))
	case totalTx == 0:
		c.Emoji = emojiNeutral
		c.Verdict = whaleFit(fmt.Sprintf("No labeled exchange transfer recorded in 24h (%s)",
			strings.Join(assets, ", ")))
		c.Short = "no labeled transfer"
	default:
		// The backend called every labeled asset neutral: the net is inside its
		// band, or there were no attributable flows. Never "balanced".
		c.Emoji = emojiNeutral
		c.Verdict = whaleFit(fmt.Sprintf("No net direction at labeled exchange wallets over 24h · %s",
			whaleLabeledTransfersPhrase(totalTx)))
		c.Short = "no net labeled direction"
	}
	ro.Direction = &dir
	// Confidence is the backend's data-quality score for the reading we lead
	// with — so only a coin direction carries one.
	if lead != nil && lead.Confidence > 0 {
		conf := lead.Confidence
		c.Confidence = &conf
	}

	ro.State = whaleStateNone
	if totalTx > 0 {
		ro.State = whaleStateActivity
	}

	// ── facts ───────────────────────────────────────────────────────────────
	c.Facts = append(c.Facts, whaleFit(fmt.Sprintf(
		"Read from labeled exchange wallets via Etherscan · assets: %s", strings.Join(assets, ", "))))
	for _, f := range labeled {
		c.Facts = append(c.Facts, whaleLabeledAssetLine(f))
	}
	c.Facts = append(c.Facts, whaleLabeledLineEstimate, whaleLabeledLineCoverage)
	if ex := whaleLabeledExchanges(labeled); ex != "" {
		c.Facts = append(c.Facts, whaleFit("Exchanges in the reported breakdown: "+ex))
	}
	c.Facts = append(c.Facts, whaleLabeledLineWindow)
	if lead != nil && lead.NetFlowPrev24h != nil && *lead.NetFlowPrev24h != 0 {
		base := fmt.Sprintf("%s prior 24h net flow: %s", lead.Asset, usd(*lead.NetFlowPrev24h))
		if lead.FlowPct != nil {
			base += fmt.Sprintf(" → %+.0f%% change", *lead.FlowPct)
		}
		c.Facts = append(c.Facts, whaleFit(base))
	}
	// The BTC monitor keeps its own line and its own numbers.
	if btc := whaleBTCFlowOf(w.Flows); btc != nil {
		c.Facts = append(c.Facts, whaleLabeledBTCLine(btc.TxCount24h))
	}

	shown := whaleLabeledShown(w.Transfers, anchor)
	for _, t := range shown {
		ro.Top.Transactions = append(ro.Top.Transactions, WhaleTx{TxHash: t.TxHash,
			Asset: t.Asset, AmountNative: t.AmountNative, AmountUSD: t.AmountUSD,
			Exchange: t.Exchange, Side: whaleSide(t.Direction),
			DetectedAt: t.Timestamp.UTC().Format(time.RFC3339)})
	}
	if len(shown) > 0 {
		c.Facts = append(c.Facts, whaleFit(fmt.Sprintf(
			"Largest labeled transfers among the latest %s received, not a 24h top", whaleLabeledRecords(len(w.Transfers)))))
		for _, t := range shown {
			c.Facts = append(c.Facts, whaleLabeledTxLine(t))
		}
	} else if totalTx > 0 {
		// The count stands either way — it is the backend's own 24h counter,
		// not a total of the rows we were handed.
		if len(w.Transfers) == 0 {
			c.Facts = append(c.Facts, "No transfer record came with this snapshot, so no individual transfer is listed")
		} else {
			c.Facts = append(c.Facts, whaleFit(fmt.Sprintf(
				"None of the latest %s received is a labeled transfer of this 24h window", whaleLabeledRecords(len(w.Transfers)))))
		}
	}

	c.Blocks = whaleLabeledBlocks(lead, stable, totalTx, assets, anchor, stamped)
	return c
}

// whaleLabeledAssetLabel is the card's asset label: every labeled asset that
// actually carries a flow, joined — the card is about all of them, so naming
// one would misdescribe it and naming BTC would be simply wrong. The lead coin
// comes first so the label agrees with the header and the verdict.
func whaleLabeledAssetLabel(active []string, lead *WhaleFlow) string {
	if len(active) == 0 {
		return ""
	}
	ordered := make([]string, 0, len(active))
	if lead != nil {
		ordered = append(ordered, lead.Asset)
	}
	for _, a := range active {
		if lead != nil && a == lead.Asset {
			continue
		}
		ordered = append(ordered, a)
	}
	return strings.Join(ordered, "/")
}

// whaleLabeledRecords: "1 record", "10 records" — the served rows. The BTC
// card calls them "monitor records"; that is the mempool feed's word and does
// not belong on this path.
func whaleLabeledRecords(n int) string {
	if n == 1 {
		return "1 record"
	}
	return fmt.Sprintf("%d records", n)
}

// whaleLabeledTransfersPhrase: "31 labeled transfers", "1 labeled transfer".
func whaleLabeledTransfersPhrase(n int) string {
	if n == 1 {
		return "1 labeled transfer"
	}
	return fmt.Sprintf("%d labeled transfers", n)
}

// whaleLabeledUnderBand reports whether the flow's net really is inside the
// backend's neutral band, computed from the gross the backend served. When the
// gross is absent the share cannot be computed, so nothing is claimed.
func whaleLabeledUnderBand(f WhaleFlow) bool {
	gross := f.InflowUSD24h + f.OutflowUSD24h
	if gross <= 0 {
		return false
	}
	return whaleAbs(f.NetFlowUSD24h)/gross < float64(whaleNeutralBandPct)/100
}

// whaleLabeledAssetLine is one asset's own line. A neutral verdict over a
// non-zero net names the side it leans and, only when the arithmetic agrees,
// the band that suppressed the direction.
func whaleLabeledAssetLine(f WhaleFlow) string {
	switch {
	case whaleDirectionalRaw(f.Direction):
		return whaleFit(fmt.Sprintf("%s: net %s exchanges %s over 24h · %s",
			f.Asset, whaleSide(f.Direction), usd(whaleAbs(f.NetFlowUSD24h)), whaleTransfersWord(f.TxCount24h)))
	case f.TxCount24h == 0:
		return whaleFit(fmt.Sprintf("%s: no labeled transfer in 24h", f.Asset))
	case f.NetFlowUSD24h != 0:
		// The tilt still has a side: usd() shows a sign only when negative, so
		// "net $5.40M" alone leaves the reader guessing which way it leans.
		side := "toward"
		if f.NetFlowUSD24h < 0 {
			side = "away from"
		}
		amount := usd(whaleAbs(f.NetFlowUSD24h))
		if !whaleLabeledUnderBand(f) {
			// The band is not demonstrably the reason (no gross served, or the
			// share is at/above it) — so it is not offered as one.
			return whaleFit(fmt.Sprintf("%s: net %s %s exchanges over 24h, the source named no direction · %s",
				f.Asset, amount, side, whaleTransfersWord(f.TxCount24h)))
		}
		long := fmt.Sprintf("%s: net %s %s exchanges over 24h, under the %d%% of gross flow needed to name it · %s",
			f.Asset, amount, side, whaleNeutralBandPct, whaleTransfersWord(f.TxCount24h))
		if utf8.RuneCountInString(long) <= whaleFactMaxRunes {
			return long
		}
		return whaleFit(fmt.Sprintf("%s: net %s %s exchanges over 24h, under the %d%% needed to name a direction · %s",
			f.Asset, amount, side, whaleNeutralBandPct, whaleTransfersWord(f.TxCount24h)))
	default:
		return whaleFit(fmt.Sprintf("%s: no net direction over 24h · %s",
			f.Asset, whaleTransfersWord(f.TxCount24h)))
	}
}

// whaleLabeledBTCLine keeps the unlabeled monitor beside the labeled reading
// without mixing a single number into it.
func whaleLabeledBTCLine(n int) string {
	if n == 0 {
		return whaleFit(fmt.Sprintf("BTC monitor, separate and unlabeled: no transaction ≥ %s seen in 24h, no exchange direction",
			whaleThresholdShown))
	}
	return whaleFit(fmt.Sprintf("BTC monitor, separate and unlabeled: %d transactions ≥ %s seen in 24h, no exchange direction",
		n, whaleThresholdShown))
}

func whaleBTCFlowOf(flows []WhaleFlow) *WhaleFlow {
	for i := range flows {
		if flows[i].Asset == "BTC" {
			return &flows[i]
		}
	}
	return nil
}

// whaleLabeledExchanges names the exchanges the backend actually reported,
// ranked by transfer count DESC then name ASC (deterministic).
func whaleLabeledExchanges(flows []WhaleFlow) string {
	total := map[string]int{}
	for _, f := range flows {
		for name, n := range f.ExchangeBreakdown {
			total[name] += n
		}
	}
	if len(total) == 0 {
		return ""
	}
	names := make([]string, 0, len(total))
	for name := range total {
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool {
		if total[names[i]] != total[names[j]] {
			return total[names[i]] > total[names[j]]
		}
		return names[i] < names[j]
	})
	if len(names) > whaleLabeledTopExchanges {
		names = names[:whaleLabeledTopExchanges]
	}
	return strings.Join(names, ", ")
}

// whaleLabeledShown is the list: labeled transfers inside
// (windowEnd−24h, windowEnd], largest by USD first, at most whaleTopN. Stable
// on ties (served order).
func whaleLabeledShown(transfers []WhaleTransfer, windowEnd time.Time) []WhaleTransfer {
	cutoff := windowEnd.Add(-whaleWindow)
	var recent []WhaleTransfer
	for _, t := range transfers {
		if whaleLabeled(t) && t.Timestamp.After(cutoff) && !t.Timestamp.After(windowEnd) {
			recent = append(recent, t)
		}
	}
	sort.SliceStable(recent, func(i, j int) bool { return recent[i].AmountUSD > recent[j].AmountUSD })
	if len(recent) > whaleTopN {
		recent = recent[:whaleTopN]
	}
	return recent
}

// whaleLabeledTxLine is one listed transfer. The time is the BLOCK time —
// Etherscan stamps the block, unlike the BTC monitor's poll time.
func whaleLabeledTxLine(t WhaleTransfer) string {
	amt := fmt.Sprintf("%s %s ≈ %s", trimFloat(t.AmountNative), t.Asset, usd(t.AmountUSD))
	side := whaleSide(t.Direction) + " " + t.Exchange
	when := "block time " + whaleClock(t.Timestamp)
	return whaleFit(amt, side, when)
}

// whaleLabeledBlocks is the content-ready form. No scenarios, nothing to
// invalidate and no regime: the agent sees a flow between wallets, not price.
// The window is named with a time only when the lead snapshot dated itself.
func whaleLabeledBlocks(lead, stable *WhaleFlow, totalTx int, assets []string, windowEnd time.Time, stamped bool) *ContentBlocks {
	window := "in the last 24h"
	if stamped {
		window = "in the 24h to " + whaleClock(windowEnd)
	}
	var what string
	switch {
	case lead != nil:
		what = whaleFit(fmt.Sprintf("Labeled exchange wallets show a net %s %s exchanges in %s %s",
			usd(whaleAbs(lead.NetFlowUSD24h)), whaleSide(lead.Direction), lead.Asset, window))
	case stable != nil:
		what = whaleFit(fmt.Sprintf("Labeled exchange wallets show a stablecoin net %s %s exchanges in %s %s",
			usd(whaleAbs(stable.NetFlowUSD24h)), whaleSide(stable.Direction), stable.Asset, window))
	case totalTx == 0:
		what = whaleFit(fmt.Sprintf("No labeled exchange transfer was recorded %s (%s)",
			window, strings.Join(assets, ", ")))
	default:
		what = whaleFit(fmt.Sprintf("Labeled exchange wallets show no net direction %s · %s",
			window, whaleLabeledTransfersPhrase(totalTx)))
	}
	return &ContentBlocks{WhatHappened: what, WhyLevel: whaleLabeledWhyLevel,
		Limitations: whaleLabeledLimitations, Source: whaleLabeledBlockSource}
}

// whaleLabeledConclusion is the /showcase/example conclusion of a labeled
// card: a flow between wallets, never a price direction.
func whaleLabeledConclusion(ro *WhaleReadout) string {
	const tail = "; it says nothing about where price goes."
	const head = "This is an estimate over labeled wallets, not a forecast: "

	if ro.LeadAsset != nil {
		for _, f := range ro.Flows {
			if f.Asset != *ro.LeadAsset {
				continue
			}
			side := "to"
			if f.Direction == whaleDirFrom {
				side = "from"
			}
			return fmt.Sprintf("%s%s %s moved %s exchange wallets in 24h, measured over the wallets in our registry only%s",
				head, usd(whaleAbs(f.NetFlowUSD24h)), f.Asset, side, tail)
		}
	}
	// No coin direction. A stablecoin flow is reported as what it is, with no
	// claim about what it implies.
	for _, f := range ro.Flows {
		if f.SourceKind != whaleSourceKindLabeled || !whaleIsStablecoin(f.Asset) {
			continue
		}
		if f.Direction != whaleDirTo && f.Direction != whaleDirFrom {
			continue
		}
		side := "to"
		if f.Direction == whaleDirFrom {
			side = "from"
		}
		return fmt.Sprintf("%s%s of %s, a stablecoin, moved %s exchange wallets in 24h while no coin carried a net direction%s",
			head, usd(whaleAbs(f.NetFlowUSD24h)), f.Asset, side, tail)
	}
	return head + "no net direction at the exchange wallets we track over 24h" + tail
}
