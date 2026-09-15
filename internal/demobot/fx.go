package demobot

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// ── Asset registry ───────────────────────────────────────────────────────────

const (
	srcBinance = "binance"
	srcYahoo   = "yahoo"
)

// assetSpec describes one tradeable asset the analysis commands accept.
type assetSpec struct {
	Display string // card label shown to a reader: "BTC", "GOLD · COMEX GC=F"
	// Key is the machine asset value served by the HTTP envelope. Empty means
	// "same as Display" — they only differ where the honest human label is not
	// a plain ticker (gold).
	Key      string
	Source   string // srcBinance | srcYahoo
	Symbol   string // exchange-native symbol: "BTCUSDT", "EURUSD=X"
	Fallback string // secondary Yahoo symbol tried when the primary fails
	Interval string // candle interval the indicator math runs on
}

var btcSpec = assetSpec{Display: "BTC", Source: srcBinance, Symbol: "BTCUSDT", Interval: "4h"}

// xauSpec: spot XAUUSD does not exist on this feed — XAUUSD=X and XAU=X both
// answer 404 (verified live 2026-08-18 and again 2026-08-26). GC=F (COMEX gold
// futures) is not a fallback in practice, it IS the source.
//
// Display therefore names the contract: a card printing levels off futures
// prices under the label "XAUUSD" is off by the basis on every number it
// shows. Key keeps "XAUUSD" so the documented HTTP contract is untouched —
// see Card.AssetKey for why the two are separate fields.
var xauSpec = assetSpec{
	Display:  "GOLD · COMEX GC=F",
	Key:      "XAUUSD",
	Source:   srcYahoo,
	Symbol:   "XAUUSD=X",
	Fallback: "GC=F",
	Interval: "1h",
}

var assetTable = map[string]assetSpec{
	"btc":    btcSpec,
	"eth":    {Display: "ETH", Source: srcBinance, Symbol: "ETHUSDT", Interval: "4h"},
	"eurusd": {Display: "EURUSD", Source: srcYahoo, Symbol: "EURUSD=X", Interval: "1h"},
	"gbpusd": {Display: "GBPUSD", Source: srcYahoo, Symbol: "GBPUSD=X", Interval: "1h"},
	"usdjpy": {Display: "USDJPY", Source: srcYahoo, Symbol: "USDJPY=X", Interval: "1h"},
	"xauusd": xauSpec,
}

var assetAliases = map[string]string{
	"xau":     "xauusd",
	"gold":    "xauusd",
	"bitcoin": "btc",
}

// fxPairs is the /fx overview set, in render order.
var fxPairs = []string{"eurusd", "gbpusd", "usdjpy", "xauusd"}

// defaultMomentumKeys is the default /momentum trio in scan form — used when
// only a timeframe is requested (B1: "/momentum 1d", "?tf=1d").
var defaultMomentumKeys = []string{"btc", "eth", "xauusd"}

// resolveAsset maps a user-typed argument to an assetSpec. Empty argument =
// BTC, matching the original command behavior.
func resolveAsset(arg string) (assetSpec, error) {
	if strings.TrimSpace(arg) == "" {
		return btcSpec, nil
	}
	key, err := resolveAssetKey(arg)
	if err != nil {
		return assetSpec{}, err
	}
	return assetTable[key], nil
}

// resolveAssetKey maps a user-typed argument to its canonical registry key.
func resolveAssetKey(arg string) (string, error) {
	key := strings.ToLower(strings.TrimSpace(arg))
	if canonical, ok := assetAliases[key]; ok {
		key = canonical
	}
	if _, ok := assetTable[key]; ok {
		return key, nil
	}
	return "", fmt.Errorf("unknown asset %q — try: btc, eth, eurusd, gbpusd, usdjpy, xau/gold", arg)
}

// scanMaxAssets caps one momentum scan (B1: "up to 6" — the registry size).
const scanMaxAssets = 6

// parseAssetList validates a comma list against the registry (B1): every
// entry must resolve (aliases allowed), duplicates collapse to the first
// appearance, at most scanMaxAssets RAW entries. Errors carry the offending
// entry and the allowed list so a 400 is self-explanatory.
func parseAssetList(raw string) ([]string, error) {
	entries := strings.Split(raw, ",")
	if len(entries) > scanMaxAssets {
		return nil, fmt.Errorf("too many assets (%d) — up to %d per scan", len(entries), scanMaxAssets)
	}
	seen := map[string]bool{}
	keys := make([]string, 0, len(entries))
	for _, e := range entries {
		if strings.TrimSpace(e) == "" {
			return nil, fmt.Errorf("empty asset entry in %q — use a plain comma list like btc,eurusd", raw)
		}
		key, err := resolveAssetKey(e)
		if err != nil {
			return nil, err
		}
		if seen[key] {
			continue
		}
		seen[key] = true
		keys = append(keys, key)
	}
	return keys, nil
}

// momentumTFs are the timeframes the momentum scan accepts (B1). Binance
// serves all three natively; Yahoo serves 1h and 1d natively while 4h is
// honestly aggregated from 1h (see fetchYahooCandlesTF).
var momentumTFs = map[string]bool{"1h": true, "4h": true, "1d": true}

const momentumTFList = "1h, 4h, 1d"

// specWithTF re-bases a spec on the requested timeframe; "" keeps the asset's
// native interval.
func specWithTF(spec assetSpec, tf string) (assetSpec, error) {
	if tf == "" {
		return spec, nil
	}
	if !momentumTFs[tf] {
		return assetSpec{}, fmt.Errorf("unknown timeframe %q — allowed: %s", tf, momentumTFList)
	}
	spec.Interval = tf
	return spec, nil
}

// offlineVerdict is the degraded wording for this asset's source. FX pairs get
// the FX phrasing; gold does not — telling a reader that the "FX data source"
// failed on a card headed "GOLD · COMEX GC=F" names the wrong market.
func (s assetSpec) offlineVerdict() string {
	if s.isGold() {
		return "Gold futures data source unavailable right now"
	}
	return fxOfflineVerdict
}

// sourceNote is the footer line. The gold note carries the instrument
// disclosure, which the spec requires on EVERY card — degraded ones included,
// where it used to be dropped in favour of the bare Yahoo credit.
func (s assetSpec) sourceNote() string {
	if s.isGold() {
		return goldSourceNote
	}
	return "data: Yahoo Finance"
}

// isGold marks the COMEX contract, the one asset whose human label is not its
// ticker.
func (s assetSpec) isGold() bool { return s.Key == "XAUUSD" }

// ── Market hours ─────────────────────────────────────────────────────────────

// isForexOpen approximates the FX trading week: closed from Friday 21:00 UTC
// to Sunday 21:00 UTC. Real venue boundaries drift ±1h with US/AU daylight
// saving; the fixed-UTC window is a documented simplification for the demo
// bot, not an execution-grade calendar.
func isForexOpen(now time.Time) bool {
	t := now.UTC()
	switch t.Weekday() {
	case time.Saturday:
		return false
	case time.Friday:
		return t.Hour() < 21
	case time.Sunday:
		return t.Hour() >= 21
	default:
		return true
	}
}

const fxClosedBanner = "⏸ Forex market closed (weekend) — data as of Friday close"

const fxOfflineVerdict = "FX data source unavailable right now"

// ── /fx overview ─────────────────────────────────────────────────────────────
//
// FX stage 1 (2026-09-15): presentation and honesty only. The rule is
// unchanged — per instrument, on closed 1h Yahoo bars: EMA50 vs EMA200,
// RSI(14), the last close against the bar ~24h back, and the last close's
// place inside the trailing-24h high-low range (fxReadFromCandles, dayRange).
//
// Honesty rules for every line below:
//   - no combined verdict: the pairs' raw directions cannot be added up (a
//     rising USDJPY is a stronger USD, a rising EURUSD a weaker one), so the
//     header states coverage and the semaphore stays neutral;
//   - no row colour: EMA50 vs EMA200 is one of four facts on a row, and a
//     coloured dot made it read as the row's verdict;
//   - the EMA fact says what it is ("EMA50 below EMA200"), never "trend";
//   - "24h" only when the reference close is 24h to 24h+fxDayTolerance back;
//     otherwise the line names the close it is measured from;
//   - the card's data time is the OLDEST bar shown, every row names its own;
//   - pairs are judged by the fixed Forex week (momentumFreshness); gold is
//     COMEX GC=F futures with hours the service does not know, so it is
//     judged by bar age alone and never told the market is closed;
//   - every line fits fxLineMaxRunes.

// fxLineMaxRunes is the readability budget for one line of FX text.
const fxLineMaxRunes = 110

// fxDayTolerance: the reference bar for "24h" may sit up to this much more
// than 24h back — one or two missing hourly bars on Yahoo, or the daily COMEX
// break (one missing GC=F bar; its UTC hour moves with daylight saving).
// Further back the reference is the close before a session gap (the
// weekend) and the line says so.
const fxDayTolerance = 2 * time.Hour

// results[].change_window values.
const (
	fxWindow24h        = "24h"
	fxWindowSinceClose = "since_previous_close"
)

// fxNoRecentBar is gold's results[].freshness when its last closed bar is
// older than fxGoldMaxAge: the service has no COMEX calendar, so it states
// the bar age, never "market closed" and never "delayed".
const fxNoRecentBar = "no_recent_bar"

// fxGoldMaxAge is gold's bar-age bound: the pairs' two bars (barMaxAge) plus
// one for the daily COMEX break — one GC=F bar is missing every day.
// Hypothesis, not measured: Yahoo may also publish GC=F late, so a 2h bound
// could flag gold after the break for nothing (and move the push hook's
// hash). Pairs keep the Momentum rule.
const fxGoldMaxAge = 3 * time.Hour

// fxGoldHeader opens the gold section.
const fxGoldHeader = "Gold: COMEX GC=F futures, not spot XAUUSD; Forex hours and the weekend banner do not apply"

// fxRead is one instrument's computed snapshot for the overview.
type fxRead struct {
	Pair         string
	spec         assetSpec // zero in hand-built test sweeps: read as a pair
	OK           bool
	Insufficient bool    // fetched fine but too little history for EMA200
	Dir          string  // up | down | flat (EMA50 vs EMA200)
	RSI          float64 // RSI(14) on 1h
	Price        float64 // close of the last closed bar
	DayChangePct float64
	HasDay       bool      // false when no reference bar ≥24h back exists
	RefAt        time.Time // close time of the reference bar
	SinceClose   bool      // reference more than 24h+fxDayTolerance back
	DayPos       float64   // 0..1 position of the last close inside the trailing-24h range
	HasRange     bool      // false when the trailing 24h range is degenerate
	CloseAt      time.Time // close time of the last CLOSED bar used
}

// interval is the bar size of a read (1h for every /fx instrument).
func (r fxRead) interval() string {
	if r.spec.Interval != "" {
		return r.spec.Interval
	}
	return "1h"
}

// fxLabel names the instrument on a row. Gold rows say GOLD: the section
// header right above names the contract.
func fxLabel(r fxRead) string {
	if r.spec.isGold() {
		return "GOLD"
	}
	return r.Pair
}

// fxPrice prints the last close at the S/R card's precision for the asset.
func fxPrice(r fxRead) string {
	return fmt.Sprintf("%.*f", srDecimals(r.spec, r.Price), r.Price)
}

// fxFreshness judges one read's last closed bar at the card's clock:
// on_time | market_closed | data_delayed for pairs (the Momentum rule, so a
// PAIR never reads differently on the two cards), on_time | no_recent_bar for
// gold (bar age only). This holds for the pairs only: gold does read
// differently on the two cards for now — Momentum still judges it by the
// Forex window (market_closed + the weekend banner), this card by bar age.
func fxFreshness(r fxRead, now time.Time) string {
	if r.spec.isGold() {
		if !r.CloseAt.IsZero() && now.Sub(r.CloseAt) > fxGoldMaxAge {
			return fxNoRecentBar
		}
		return momentumOnTime
	}
	return momentumFreshness(srcYahoo, r.interval(), r.CloseAt, now)
}

// fxMarketLine is a row's first line — what moved and where price sits:
// "EURUSD 1.1543 · 24h -0.05% · 43% of the 24h range[ · data delayed]".
// The digest FX block shows exactly this line.
func fxMarketLine(r fxRead, now time.Time) string {
	label := fxLabel(r)
	if r.Insufficient {
		return label + ": insufficient history for EMA50/EMA200/RSI(14) on " + r.interval() + " bars"
	}
	if !r.OK {
		return label + ": data unavailable right now"
	}
	s := label + " " + fxPrice(r)
	if r.HasDay {
		if r.SinceClose {
			s += fmt.Sprintf(" · since %s close %+.2f%%", momentumBarTime(r.RefAt), r.DayChangePct)
		} else {
			s += fmt.Sprintf(" · 24h %+.2f%%", r.DayChangePct)
		}
	}
	if r.HasRange {
		// After a gap every bar since the reference close is inside the
		// trailing-24h window (the reference is the last bar before it), so
		// the range is exactly "since then".
		span := "the 24h range"
		if r.HasDay && r.SinceClose {
			span = "the range since then"
		}
		s += fmt.Sprintf(" · %.0f%% of %s", r.DayPos*100, span)
	}
	switch fxFreshness(r, now) {
	case momentumDataDelayed:
		s += " · data delayed"
	case fxNoRecentBar:
		s += " · no bar in the last 3h" // fxGoldMaxAge
	}
	return s
}

// fxIndicators is the indicator half of a row: "EMA50 below EMA200 · RSI(1h) 40.8".
func fxIndicators(r fxRead) string {
	return fmt.Sprintf("EMA50 %s EMA200 · RSI(%s) %.1f", fxEMAWords(r.Dir), r.interval(), r.RSI)
}

// fxContextLine is a row's second line — the indicators and the row's own
// bar time: "EURUSD: EMA50 below EMA200 · RSI(1h) 40.8 · last bar Sep 15 08:00 UTC".
func fxContextLine(r fxRead) string {
	return fxLabel(r) + ": " + fxIndicators(r) + " · last bar " + momentumBarTime(r.CloseAt)
}

func fxEMAWords(dir string) string {
	switch dir {
	case "up":
		return "above"
	case "down":
		return "below"
	}
	return "equal to"
}

// fxEMARelation is results[].ema_relation: above | below | equal.
func fxEMARelation(dir string) string {
	if w := fxEMAWords(dir); w != "equal to" {
		return w
	}
	return "equal"
}

// fxResult is one row's machine outcome (results[]).
func fxResult(r fxRead, now time.Time) AssetResult {
	switch {
	case r.OK:
	case r.Insufficient:
		return assetResult(r.Pair, statusInsufficientHistory)
	default:
		return assetResult(r.Pair, statusSourceOffline)
	}
	res := assetResult(r.Pair, statusOK)
	price, rsi := r.Price, r.RSI
	res.Timeframe = r.interval()
	if !r.CloseAt.IsZero() {
		res.DataAsOf = r.CloseAt.UTC().Format(time.RFC3339)
	}
	res.Freshness = fxFreshness(r, now)
	res.Price, res.RSI = &price, &rsi
	res.EMARelation = fxEMARelation(r.Dir)
	if r.HasDay {
		chg := r.DayChangePct
		res.ChangePct = &chg
		res.ChangeWindow = fxWindow24h
		if r.SinceClose {
			res.ChangeWindow = fxWindowSinceClose
		}
		if !r.RefAt.IsZero() {
			res.ChangeFrom = r.RefAt.UTC().Format(time.RFC3339)
		}
	}
	if r.HasRange {
		pos := r.DayPos * 100
		res.RangePositionPct = &pos
	}
	return res
}

// fxStatus is the outcome of a whole sweep: ok with at least one reading;
// without one, insufficient_history when at least one instrument answered
// with too little history (the source is alive — the others may be dead),
// source_offline only when none answered (including an empty sweep).
func fxStatus(reads []fxRead) cardStatus {
	short := false
	for _, r := range reads {
		if r.OK {
			return statusOK
		}
		short = short || r.Insufficient
	}
	if short {
		return statusInsufficientHistory
	}
	return statusSourceOffline
}

// fxAllShort: every instrument answered, all with too little history — the
// one sweep the plain "insufficient history on 1h bars" wording is exact for.
// A sweep that also has dead instruments states its coverage instead.
func fxAllShort(reads []fxRead) bool {
	for _, r := range reads {
		if !r.Insufficient {
			return false
		}
	}
	return len(reads) > 0
}

// fxPairsRead counts the pairs with a reading — the weekend wording (the
// card's banner, the digest's "forex closed") only speaks when one is shown.
func fxPairsRead(reads []fxRead) int {
	n := 0
	for _, r := range reads {
		if r.OK && !r.spec.isGold() {
			n++
		}
	}
	return n
}

// fxCoverage is the header's coverage part: "3 pairs + gold read[ · N short
// history][ · N unavailable][ · N data delayed][ · gold: no recent bar]".
// A gold read without a recent bar leaves the "+ gold" coverage and is named
// at the end instead, so the header never implies it is current.
func fxCoverage(reads []fxRead, now time.Time) string {
	var pairs, pairsOK, short, dead, delayed int
	goldOK, goldStale := false, false
	for _, r := range reads {
		gold := r.spec.isGold()
		if !gold {
			pairs++
		}
		switch {
		case r.OK:
			if gold {
				goldOK = true
			} else {
				pairsOK++
			}
			switch fxFreshness(r, now) {
			case momentumDataDelayed:
				delayed++
			case fxNoRecentBar:
				goldStale = true
			}
		case r.Insufficient:
			short++
		default:
			dead++
		}
	}
	var who []string
	if pairs > 0 {
		who = append(who, fxPairsWord(pairsOK, pairs))
	}
	if goldOK && (!goldStale || pairs == 0) {
		who = append(who, "gold")
	}
	parts := []string{strings.Join(who, " + ") + " read"}
	if short > 0 {
		parts = append(parts, fmt.Sprintf("%d short history", short))
	}
	if dead > 0 {
		parts = append(parts, fmt.Sprintf("%d unavailable", dead))
	}
	if delayed > 0 {
		parts = append(parts, fmt.Sprintf("%d data delayed", delayed))
	}
	if goldStale {
		parts = append(parts, "gold: no recent bar")
	}
	return strings.Join(parts, " · ")
}

// fxVerdictPrefix opens every FX header built from coverage.
const fxVerdictPrefix = "FX overview · 1h · "

// fxPairsWord: "1 pair", "3 pairs", "2 of 3 pairs".
func fxPairsWord(ok, total int) string {
	word := "pairs"
	if total == 1 {
		word = "pair"
	}
	if ok == total {
		return fmt.Sprintf("%d %s", total, word)
	}
	return fmt.Sprintf("%d of %d %s", ok, total, word)
}

// fxOverviewCard assembles the /fx card from reads with at least one reading
// (pure — golden-tested without network). Header = coverage, never a
// direction: "FX overview · 1h · 3 pairs + gold read[ · N short
// history][ · N unavailable][ · N data delayed][ · gold: no recent bar]"
// ("1h", not "1h bars": the worst case of every part at once must still fit
// fxLineMaxRunes). Facts: the weekend banner (pairs only), two lines per read
// pair (one per failed one), then the gold section.
func fxOverviewCard(reads []fxRead, now time.Time) Card {
	var oldest time.Time
	var pairFacts, goldFacts []string
	for _, r := range reads {
		lines := []string{fxMarketLine(r, now)}
		if r.OK {
			if !r.CloseAt.IsZero() && (oldest.IsZero() || r.CloseAt.Before(oldest)) {
				oldest = r.CloseAt
			}
			lines = append(lines, fxContextLine(r))
		}
		if r.spec.isGold() {
			goldFacts = append(goldFacts, lines...)
		} else {
			pairFacts = append(pairFacts, lines...)
		}
	}
	coverage := fxCoverage(reads, now)
	if oldest.IsZero() {
		oldest = now
	}
	c := Card{
		Emoji:      emojiNeutral,
		Agent:      "FX Agent",
		ShortName:  "FX",
		Command:    keyFX,
		HowItWorks: howTexts[keyFX],
		DataTime:   oldest.UTC(),
		SourceNote: "data: Yahoo Finance",
		Verdict:    fxVerdictPrefix + coverage,
		Short:      coverage,
	}
	// "Data as of Friday close" speaks about pairs the card shows: with every
	// pair down there is nothing for it to date.
	if fxPairsRead(reads) > 0 && !isForexOpen(now) {
		c.Facts = append(c.Facts, fxClosedBanner)
	}
	c.Facts = append(c.Facts, pairFacts...)
	if len(goldFacts) > 0 {
		c.Facts = append(c.Facts, fxGoldHeader)
		c.Facts = append(c.Facts, goldFacts...)
	}
	for _, r := range reads {
		c.Results = append(c.Results, fxResult(r, now))
	}
	return c
}

// fxDigestTitle heads the digest FX block: the oldest bar shown ("as of",
// the block's data_as_of; zero = none), the Forex closure — only when a pair
// was read (pairsRead), since it dates the pairs' data — and gold's contract
// because its row says just GOLD.
func fxDigestTitle(now, asOf time.Time, pairsRead bool) string {
	var notes []string
	if pairsRead && !isForexOpen(now) {
		notes = append(notes, "forex closed: pairs show Friday data")
	}
	if !asOf.IsZero() {
		notes = append(notes, "as of "+momentumBarTime(asOf))
	}
	notes = append(notes, "gold = COMEX GC=F futures")
	return "<b>FX</b> <i>(" + strings.Join(notes, " · ") + ")</i>"
}

// fxDigestNewerLine names the rows whose last bar is newer than the block's
// "as of" (the market lines carry no time), grouped by bar time:
// "Newer bars: GBPUSD, GOLD Sep 15 08:00 UTC". "" when every read shares it.
func fxDigestNewerLine(reads []fxRead, asOf time.Time) string {
	var times []time.Time
	names := map[time.Time][]string{}
	for _, r := range reads {
		if !r.OK || r.CloseAt.IsZero() || !r.CloseAt.After(asOf) {
			continue
		}
		t := r.CloseAt.UTC()
		if _, seen := names[t]; !seen {
			times = append(times, t)
		}
		names[t] = append(names[t], fxLabel(r))
	}
	if len(times) == 0 {
		return ""
	}
	sort.Slice(times, func(i, j int) bool { return times[i].Before(times[j]) })
	var groups []string
	for _, t := range times {
		groups = append(groups, strings.Join(names[t], ", ")+" "+t.Format("Jan 2 15:04"))
	}
	return "Newer bars: " + strings.Join(groups, " · ") + " UTC"
}

// fxAILine is a row for the AI payload: the card's market line and
// indicators, with gold named as the futures contract it is (the model sees
// no section header and would otherwise call it spot).
func fxAILine(r fxRead, now time.Time) string {
	line := fxMarketLine(r, now) + " · " + fxIndicators(r)
	if r.spec.isGold() {
		line = "GOLD (COMEX GC=F futures)" + strings.TrimPrefix(line, fxLabel(r))
	}
	return line
}

// fxUnreadLine marks a row of an instrument that produced no reading
// (unavailable or short history) — the landing example never quotes one as
// the card's fact.
func fxUnreadLine(f string) bool {
	return strings.HasSuffix(f, ": data unavailable right now") || strings.Contains(f, ": insufficient history for ")
}
