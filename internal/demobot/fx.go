package demobot

import (
	"fmt"
	"math"
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
//
// FX stage 2 (2026-09-16) changes the PRESENTATION again, and only that: the
// card is the one place the instruments are COMPARED. Every number above keeps
// its rule; what changed is the shape.
//
//   - one row per instrument under one column header (fxTableHeader), so the
//     same fact sits in the same place on every row and the percentages — the
//     24h change and the place in the range — read against each other as they
//     are. No normalisation to USD and no dollar-strength claim: that is a NEW
//     rule (stage 3) and needs its own decision;
//   - the row order is deterministic and signed (fxOrderNote): by the SIZE of
//     the 24h change, ties by the registry order. It is not a ranking — there
//     is no leader, no best pair and no recommendation on this card;
//   - a row whose bar is not fresh states the bar's age (fxRowFlag) and sinks
//     below the fresh ones, so a dead instrument is visible without ever
//     topping the table;
//   - gold keeps its own section under the COMEX disclosure.

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

// ── the comparison table (stage 2) ───────────────────────────────────────────

// fxTableHeader names the columns once so a row can carry values only — that
// is what makes four instruments comparable at a glance instead of eight
// prose lines. "UTC" belongs to the bar column: every time on this card is UTC.
const fxTableHeader = "Pair · price · 24h change · place in 24h range · EMA50 vs EMA200 · RSI(1h) · last bar UTC"

// fxOrderNote signs the row order AND says what it is not. The order is a
// reading aid — the biggest move is the easiest row to find — never a ranking:
// the card has no leader and gives no recommendation.
const fxOrderNote = "Ordered by 24h change size, not by importance · rows without a fresh bar last"

// fxGapNote explains the rows whose change is measured from a named close
// (after the weekend or a session gap) — printed once, only when such a row is
// shown, so the change and range columns are never read as plain 24h numbers.
const fxGapNote = "After a session gap the change and the range are measured from the named close"

// fxCaptionLine reports whether a card fact is a caption rather than an
// instrument row: the weekend banner, the column header, the order and gap
// notes, the gold disclosure. It answers "is this a reading?", which is what
// the leading-fact pick needs (strongestFact); the landing's data block keeps
// the captions that qualify the values it quotes — see fxExampleLines.
func fxCaptionLine(f string) bool {
	switch f {
	case fxClosedBanner, fxTableHeader, fxOrderNote, fxGapNote, fxGoldHeader:
		return true
	}
	return false
}

// fxExampleLines is the landing's data block: the head of the card's table,
// capped, with the dead rows taken out of the middle. It shows fewer rows than
// the card does — and a cut-down list carries every way of being misread that
// the full table carries. So the lines that qualify the values stay, and only
// the gap note goes.
//
// What must survive the filter, because it qualifies the numbers beside it:
//   - the weekend banner: without it Friday's prices read as today's;
//   - gold's COMEX disclosure: without it a futures price reads as spot
//     XAUUSD, which the card is not allowed to imply anywhere;
//   - the column header: the landing shows values only, so without it
//     "91% · below · 62.9" reaches the reader unlabelled;
//   - the order note: the rows are sorted by the size of the 24h move, and
//     the block keeps only the first of them. An unsigned list cut at the top
//     of a sort reads as a ranking — the one reading this card refuses. It
//     costs a row, and a row is the cheaper loss.
//
// Dropped: the dead rows (not readings) and the gap note — the change cell of
// such a row names the close it is measured from ("+0.31% since Sep 11
// 22:00"), so the value qualifies itself where a 24h number would not.
//
// The disclosure takes a slot only while the gold row it discloses still fits
// under it (nothing but gold rows follows it), and never trails alone.
func fxExampleLines(facts []string, max int) []string {
	out := make([]string, 0, max)
	for _, f := range facts {
		if len(out) >= max {
			break
		}
		if f == fxGapNote || fxUnreadLine(f) || strings.TrimSpace(f) == "" {
			continue
		}
		if f == fxGoldHeader && len(out)+2 > max {
			break
		}
		out = append(out, f)
	}
	if n := len(out); n > 0 && out[n-1] == fxGoldHeader {
		out = out[:n-1]
	}
	return out
}

// fxConclusion words the FX card for a trader. The generic wording did not
// fit it twice over: it read the neutral semaphore as "nothing leans either
// way on the broader market" — the combined verdict over the pairs this card
// refuses to give, and it gave it while a row showed +0.32% at 91% of its
// day's range — and it pointed at "level structure", which this card has no
// levels for. So the FX card says what it is instead.
func fxConclusion() string {
	return "For a trader this is the pairs side by side, not one verdict over them: " +
		"each row holds for as long as its own numbers do, the order is by the size " +
		"of the 24h move rather than importance, and nothing here is added up into " +
		"one read across them."
}

// fxGoldContract is what a gold row has to carry when it is quoted away from
// the section header that discloses it (fxNamedRow, fxAILine). ASCII, so its
// byte length is its rune length — fxQuotedMaxRunes counts on that, and a
// test pins it.
const fxGoldContract = " (COMEX GC=F futures)"

// fxQuotedMaxRunes bounds a card row quoted ON ITS OWN — the landing's
// `explained`, the one line that leaves the table. fxLineMaxRunes keeps the
// card's rows aligned inside a table and this line is not in one, so it is
// allowed past that budget by exactly two things and no others: the contract
// a gold row must carry out of its section, and the full stop endSentence
// puts on a quoted fragment. A wrapped line is the smaller cost against a
// futures price read as spot.
const fxQuotedMaxRunes = fxLineMaxRunes + len(fxGoldContract) + 1

// fxNamedRow names the contract inside a gold row, for the one place a row is
// quoted away from its section header (the landing's `explained`). The card
// and the digest keep the header above the row and are not touched. The same
// constant fxAILine uses, for the same reason: on its own, "GOLD" reads as
// spot XAUUSD — and the two disclosures of one contract must not drift apart.
func fxNamedRow(f string) string {
	const gold = "GOLD · "
	if strings.HasPrefix(f, gold) {
		return "GOLD" + fxGoldContract + " · " + strings.TrimPrefix(f, gold)
	}
	return f
}

// fxSectionName is a row's section in results[]: the comparable pairs, or the
// COMEX contract that is shown apart from them.
func fxSectionName(r fxRead) string {
	if r.spec.isGold() {
		return "gold"
	}
	return "pairs"
}

// fxAgeWords is a bar's age in the coarsest unit that is still honest: whole
// hours below two days, whole days above. Capped at ">99d" so a long outage
// cannot push a row past the line budget.
func fxAgeWords(d time.Duration) string {
	h := int(d / time.Hour)
	switch {
	case h < 1:
		return "<1h"
	case h < 48:
		return fmt.Sprintf("%dh", h)
	}
	if days := h / 24; days <= 99 {
		return fmt.Sprintf("%dd", days)
	}
	return ">99d"
}

// fxRowFlag is a row's last column: empty while the bar is fresh (and on a
// weekend, where the banner dates the pairs' data), otherwise the state AND
// the bar's age — a row the reader must discount says how far behind it is
// instead of only that it is behind. The bounds are the stage-1 ones
// (fxFreshness): two bars for the pairs, fxGoldMaxAge for gold.
func fxRowFlag(r fxRead, now time.Time) string {
	age := fxAgeWords(now.Sub(r.CloseAt))
	switch fxFreshness(r, now) {
	case momentumDataDelayed:
		return "delayed, bar " + age + " old"
	case fxNoRecentBar:
		return "no recent bar, " + age + " old"
	}
	return ""
}

// fxChangeCell is the change column: the percentage, and after a session gap
// the close it is measured from (fxGapNote carries the explanation, so the
// cell stays short enough for the budget). Without a reference bar the cell
// says so rather than printing a 0.00%.
func fxChangeCell(r fxRead) string {
	if !r.HasDay {
		return "no 24h reference"
	}
	s := fmt.Sprintf("%+.2f%%", r.DayChangePct)
	if r.SinceClose {
		s += " since " + r.RefAt.UTC().Format("Jan 2 15:04")
	}
	return s
}

// fxRangeCell is the place-in-range column: 0% at the low, 100% at the high,
// "no range" on a degenerate high-low window (nothing to place price inside).
func fxRangeCell(r fxRead) string {
	if !r.HasRange {
		return "no range"
	}
	return fmt.Sprintf("%.0f%%", r.DayPos*100)
}

// fxTableRow is one instrument's row under fxTableHeader. An instrument
// without a reading keeps its row and states which of the two absences it is
// — it must be visible in the comparison, not dropped from it.
func fxTableRow(r fxRead, now time.Time) string {
	label := fxLabel(r)
	switch {
	case r.Insufficient:
		return label + " · insufficient history for EMA50/EMA200/RSI(14) on " + r.interval() + " bars"
	case !r.OK:
		return label + " · data unavailable right now"
	}
	cells := []string{
		label, fxPrice(r), fxChangeCell(r), fxRangeCell(r),
		fxEMARelation(r.Dir), fmt.Sprintf("%.1f", r.RSI),
		r.CloseAt.UTC().Format("Jan 2 15:04"),
	}
	if flag := fxRowFlag(r, now); flag != "" {
		cells = append(cells, flag)
	}
	return strings.Join(cells, " · ")
}

// fxRowRank buckets a row for the order: a reading with a fresh bar, then a
// reading whose bar is behind, then too little history, then an instrument
// that answered with nothing. A pair inside the weekend window is NOT behind —
// the banner dates it — so the weekend does not reshuffle the table.
func fxRowRank(r fxRead, now time.Time) int {
	switch {
	case r.OK:
		switch fxFreshness(r, now) {
		case momentumDataDelayed, fxNoRecentBar:
			return 1
		}
		return 0
	case r.Insufficient:
		return 2
	default:
		return 3
	}
}

// fxSortChange is the ordering key inside a bucket: the SIZE of the change,
// direction ignored (a -0.40% and a +0.40% move are equally worth a look, and
// the raw signs of different pairs cannot be compared anyway). A reading
// without a reference close has no key and follows the rows that have one.
func fxSortChange(r fxRead) float64 {
	if !r.OK || !r.HasDay {
		return -1
	}
	return math.Abs(r.DayChangePct)
}

// fxRegistryOrder is each instrument's place in the registry sweep (fxPairs).
// It is the tie-break, so the order depends on the DATA only and never on the
// order the concurrent sweep happens to return reads in. An instrument outside
// the registry (hand-built reads) sorts last among ties.
var fxRegistryOrder = func() map[string]int {
	m := make(map[string]int, len(fxPairs))
	for i, key := range fxPairs {
		m[assetTable[key].Display] = i
	}
	return m
}()

func fxRegistryIndex(r fxRead) int {
	if i, ok := fxRegistryOrder[r.spec.Display]; ok {
		return i
	}
	return len(fxRegistryOrder)
}

// fxOrdered is the table's row order, deterministic by construction: bucket,
// then change size, then the registry order for ties — two reads of the same
// data can never swap rows, whatever order the concurrent sweep returns.
func fxOrdered(reads []fxRead, now time.Time) []fxRead {
	idx := make([]int, len(reads))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		ra, rb := reads[idx[a]], reads[idx[b]]
		if ka, kb := fxRowRank(ra, now), fxRowRank(rb, now); ka != kb {
			return ka < kb
		}
		if ka, kb := fxSortChange(ra), fxSortChange(rb); ka != kb {
			return ka > kb
		}
		return fxRegistryIndex(ra) < fxRegistryIndex(rb)
	})
	out := make([]fxRead, 0, len(reads))
	for _, i := range idx {
		out = append(out, reads[i])
	}
	return out
}

// fxTable is the table the card and the digest both show: the captions, the
// pair rows, the gold rows, the reads in shown order (pairs, then gold — so a
// caller can pair a row with its results[] entry) and the oldest bar behind
// them.
type fxTable struct {
	caption  []string
	pairRows []string
	goldRows []string
	shown    []fxRead
	oldest   time.Time
}

// lines is the whole table as text, captions first — the digest's FX block.
func (t fxTable) lines() []string {
	out := append([]string{}, t.caption...)
	out = append(out, t.pairRows...)
	return append(out, t.goldRows...)
}

// fxTableOf builds the table from reads with at least one reading. The order
// note is omitted with a single row (one row is not an order) and the gap note
// only when a shown row is measured from a named close.
func fxTableOf(reads []fxRead, now time.Time) fxTable {
	var t fxTable
	var pairs, gold []fxRead
	for _, r := range reads {
		if r.spec.isGold() {
			gold = append(gold, r)
		} else {
			pairs = append(pairs, r)
		}
	}
	gap := false
	for _, section := range [][]fxRead{fxOrdered(pairs, now), fxOrdered(gold, now)} {
		for _, r := range section {
			row := fxTableRow(r, now)
			if r.spec.isGold() {
				t.goldRows = append(t.goldRows, row)
			} else {
				t.pairRows = append(t.pairRows, row)
			}
			t.shown = append(t.shown, r)
			if !r.OK {
				continue
			}
			gap = gap || r.SinceClose
			if !r.CloseAt.IsZero() && (t.oldest.IsZero() || r.CloseAt.Before(t.oldest)) {
				t.oldest = r.CloseAt
			}
		}
	}
	t.caption = []string{fxTableHeader}
	if len(t.shown) > 1 {
		t.caption = append(t.caption, fxOrderNote)
	}
	if gap {
		t.caption = append(t.caption, fxGapNote)
	}
	return t
}

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

// fxResult is one row's machine outcome (results[]). row is the row's 1-based
// place in the shown table: stage 2 serves it beside the row's section and the
// label the row prints, so a site redraws the exact table without parsing
// facts. Present on every row of a card that RENDERS the table, a row without
// a reading included — it has a place in the comparison too.
//
// row <= 0 means the caller renders NO table (no instrument produced a
// reading): there is then no place to describe, and the three fields stay
// absent rather than describing rows the reader never sees.
func fxResult(r fxRead, now time.Time, row int) AssetResult {
	res := assetResult(r.Pair, statusOK)
	switch {
	case r.OK:
	case r.Insufficient:
		res = assetResult(r.Pair, statusInsufficientHistory)
	default:
		res = assetResult(r.Pair, statusSourceOffline)
	}
	if row > 0 {
		res.Row, res.Section, res.Label = row, fxSectionName(r), fxLabel(r)
	}
	if !r.OK {
		return res
	}
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

// fxCoverageCount is one sweep's coverage, counted ONCE. The header
// (fxCoverage) and the content blocks (fxBlocksOf) both word it, so the two can
// never disagree on what was read — in particular on gold, which a stale bar
// takes out of the "read" list and names separately.
type fxCoverageCount struct {
	pairs, pairsOK, short, dead, delayed int
	goldOK, goldStale                    bool
}

func fxCoverageOf(reads []fxRead, now time.Time) fxCoverageCount {
	var n fxCoverageCount
	for _, r := range reads {
		gold := r.spec.isGold()
		if !gold {
			n.pairs++
		}
		switch {
		case r.OK:
			if gold {
				n.goldOK = true
			} else {
				n.pairsOK++
			}
			switch fxFreshness(r, now) {
			case momentumDataDelayed:
				n.delayed++
			case fxNoRecentBar:
				n.goldStale = true
			}
		case r.Insufficient:
			n.short++
		default:
			n.dead++
		}
	}
	return n
}

// read lists what counts as read: the pairs (as "3 pairs" / "2 of 3 pairs")
// and gold — but gold only with a recent bar, unless it is all the card has.
func (n fxCoverageCount) read() []string {
	var who []string
	if n.pairs > 0 {
		who = append(who, fxPairsWord(n.pairsOK, n.pairs))
	}
	if n.goldOK && (!n.goldStale || n.pairs == 0) {
		who = append(who, "gold")
	}
	return who
}

// goldNamedStale: gold has a reading but no recent bar, and is therefore left
// out of read() and named on its own.
func (n fxCoverageCount) goldNamedStale() bool { return n.goldStale }

// fxCoverage is the header's coverage part: "3 pairs + gold read[ · N short
// history][ · N unavailable][ · N data delayed][ · gold: no recent bar]".
// A gold read without a recent bar leaves the "+ gold" coverage and is named
// at the end instead, so the header never implies it is current.
func fxCoverage(reads []fxRead, now time.Time) string {
	n := fxCoverageOf(reads, now)
	parts := []string{strings.Join(n.read(), " + ") + " read"}
	if n.short > 0 {
		parts = append(parts, fmt.Sprintf("%d short history", n.short))
	}
	if n.dead > 0 {
		parts = append(parts, fmt.Sprintf("%d unavailable", n.dead))
	}
	if n.delayed > 0 {
		parts = append(parts, fmt.Sprintf("%d data delayed", n.delayed))
	}
	if n.goldNamedStale() {
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

// ── content blocks (stage 2, 2026-09-16) ─────────────────────────────────────
//
// The FX card had no blocks at all, so the site had no CURRENT READING and no
// SCOPE AND LIMITATIONS section for it: the page was the coverage header and a
// numbered list in which the column header read as "clue No. 01" and the order
// note as "clue No. 02". The blocks below give those two sections something to
// hold WITHOUT adding a rule.
//
// what_happened states the COVERAGE and that the card is a comparison. It does
// not name an instrument, does not add the pairs up and says nothing about any
// currency: that is stage 3 and needs its own decision (fxOrderNote,
// fxConclusion, and the fxStage3Claim test that guards it).
//
// scenarios and invalidates stay nil, and so do why_level and regime: an
// overview of four instruments holds no single idea to invalidate, no level
// and no one regime. Per-instrument numbers live in results[].

// fxLimitations is the card's scope caveat, in the words the card already
// uses: no common base (the honesty rule behind the missing combined verdict),
// gold's contract (fxGoldHeader) and what the row order is not (fxOrderNote).
const fxLimitations = "The instruments are not normalised to a common base and are never added into one reading; " +
	"gold is COMEX GC=F futures, not spot XAUUSD; the row order is a reading aid, not a ranking."

// fxOrderClause is the order note (fxOrderNote) worded for prose, true on
// every path: the size of the move each row SHOWS (a row after a session gap
// shows its change since the named close, not over 24h), rows without a fresh
// bar last, and not a ranking.
const fxOrderClause = "rows are ordered by the size of the move each row shows, rows without a fresh bar last, not by importance"

// fxBlocksOf words the coverage from the SAME count the header is built from
// (fxCoverageOf — the header's "read" list and its "gold: no recent bar"), and
// the order clause only when the table actually shows an order (fxOrderNote,
// more than one row). nil when the header's read list is empty.
func fxBlocksOf(reads []fxRead, now time.Time, rows int) *ContentBlocks {
	n := fxCoverageOf(reads, now)
	who := n.read()
	if len(who) == 0 || (n.pairsOK == 0 && !n.goldOK) {
		return nil
	}
	what := strings.Join(who, " and ") + " read side by side on closed 1h bars"
	if n.goldNamedStale() {
		what += "; gold has no recent bar"
	}
	if rows > 1 {
		what += "; " + fxOrderClause
	}
	return &ContentBlocks{WhatHappened: what + ".", Limitations: fxLimitations}
}

// fxOverviewCard assembles the /fx card from reads with at least one reading
// (pure — golden-tested without network). Header = coverage, never a
// direction: "FX overview · 1h · 3 pairs + gold read[ · N short
// history][ · N unavailable][ · N data delayed][ · gold: no recent bar]"
// ("1h", not "1h bars": the worst case of every part at once must still fit
// fxLineMaxRunes). Facts (stage 2): the weekend banner (pairs only), the table
// captions, one row per pair, then the gold section under its disclosure.
func fxOverviewCard(reads []fxRead, now time.Time) Card {
	table := fxTableOf(reads, now)
	coverage := fxCoverage(reads, now)
	oldest := table.oldest
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
	c.Facts = append(c.Facts, table.caption...)
	c.Facts = append(c.Facts, table.pairRows...)
	if len(table.goldRows) > 0 {
		c.Facts = append(c.Facts, fxGoldHeader)
		c.Facts = append(c.Facts, table.goldRows...)
	}
	// results[] follows the shown rows one-to-one, so row N of the JSON is
	// row N of the card.
	for i, r := range table.shown {
		c.Results = append(c.Results, fxResult(r, now, i+1))
	}
	c.Blocks = fxBlocksOf(reads, now, len(table.shown))
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

// Stage 2 dropped fxDigestNewerLine: the block shows the same table as the
// card and every row carries its own bar time, so there is nothing left for a
// separate "Newer bars:" line to disclose.

// fxAILine is a row for the AI payload: the card's market line and
// indicators, with gold named as the futures contract it is (the model sees
// no section header and would otherwise call it spot).
func fxAILine(r fxRead, now time.Time) string {
	line := fxMarketLine(r, now) + " · " + fxIndicators(r)
	if r.spec.isGold() {
		line = "GOLD" + fxGoldContract + strings.TrimPrefix(line, fxLabel(r))
	}
	return line
}

// fxUnreadLine marks a row of an instrument that produced no reading
// (unavailable or short history) — the landing example never quotes one as
// the card's fact.
// It matches both shapes the two paths print: the stage-1 "<label>: …" of the
// degraded card and the stage-2 table row "<label> · …".
func fxUnreadLine(f string) bool {
	return strings.HasSuffix(f, "data unavailable right now") || strings.Contains(f, "insufficient history for ")
}
