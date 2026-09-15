package demobot

// funding_text.go — how the Funding card picks its coin and everything it
// SAYS (stage 1, 2026-09-15).
//
// The rule is unchanged: a last funding rate at or above +0.03% is past the
// long threshold, at or below -0.01% past the short threshold
// (fundingLongsCrowded / fundingShortsCrowded, agents.go), anything between is
// within the thresholds. What changed:
//
//   - the coin: coins past the threshold of their own side come first; among
//     them, and when none is past, the one with the largest ratio to its OWN
//     side's threshold (rate ÷ 0.03% above zero, |rate| ÷ 0.01% below). Ties
//     resolve in fundingSymbols order. The old pick (largest |rate|, then the
//     asymmetric thresholds) could show "balanced" on BTC +0.020% while XRP
//     -0.015% sat past its threshold. The digest score (fundingDeviation) is
//     computed on the same coin;
//   - honesty: no forecast ("squeeze risk building", "squeeze fuel above", "no
//     crowd to punish", "magnet zone" are gone), no "/8h" (premiumIndex serves
//     no funding interval, so the card says "last funding rate"), coverage of
//     the five majors, an empty liquidation window says only that, a
//     liquidation cluster carries the time of its last event, never an age
//     (an age changes with the request clock and would push the hook every
//     sweep);
//   - print: a rate is printed with four decimals of a percent, truncated
//     toward zero (fundingUnits), so a printed rate never sits on the other
//     side of a threshold from the real one; every distance and ratio on the
//     card is computed from the printed numbers;
//   - the liquidation cluster is picked without any price (fundingClusterZone:
//     the largest BTCUSDT band by USD), so a mark price wandering between two
//     bands does not swap the cluster; the mark only sets band_vs_mark. No
//     cluster is shown beside an empty 1h window;
//   - every line — verdict, facts, digest line, each content block — fits
//     fundingFactMaxRunes.

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	// fundingFactMaxRunes is the readability budget for one line of text.
	fundingFactMaxRunes = 110
	// fundingMinCoverage: fewer majors than this answering → a partial card
	// with no verdict. With 3 of 5 a majority of the list is still compared;
	// with 2 or fewer the "furthest/closest of the majors" claim is about a
	// minority of them.
	fundingMinCoverage = 3
	// fundingFeedLimit is the liquidation feed page the card asks the backend
	// for (its maximum; the default is 40). A full page means the 1h window
	// may hold more events than the card counted — the card then says so.
	fundingFeedLimit = 200
	// fundingWindow is the liquidation window, counted back from the request.
	fundingWindow = time.Hour
)

// Printed units: 1 unit = 0.0001% = 1e-6 of rate. The thresholds in units.
const (
	fundingLongU  = 300  // +0.0300%
	fundingShortU = -100 // -0.0100%
)

// Threshold magnitudes in 1e-8 of rate — the exact integer scale of Binance's
// 8-decimal rates, used to compare ratios without float ties.
const (
	fundingLongK8  = 30000
	fundingShortK8 = 10000
)

// fundingQuote is one symbol's premiumIndex read: the last funding rate and
// the mark price (0 when the answer carried none).
type fundingQuote struct {
	rate float64
	mark float64
}

// fundingRow is one received rate, with its printed form.
type fundingRow struct {
	sym          string
	rate         float64
	k8           int64 // rate in 1e-8, rounded
	u            int64 // printed units (see fundingUnits)
	crossedLong  bool  // the rule, on the raw rate
	crossedShort bool
}

func newFundingRow(sym string, rate float64) fundingRow {
	r := fundingRow{sym: sym, rate: rate,
		crossedLong:  rate >= fundingLongsCrowded,
		crossedShort: rate <= fundingShortsCrowded}
	r.k8 = int64(math.Round(rate * 1e8))
	r.u = fundingUnits(rate)
	return r
}

// fundingUnits prints a rate in units of 0.0001%, truncated toward zero, with
// guards so the printed value always agrees with the rule on the raw rate:
// a rate under +0.03% never prints +0.0300%, a rate past -0.01% always prints
// -0.0100% or below.
func fundingUnits(rate float64) int64 {
	u := int64(math.Round(rate*1e8)) / 100 // Go division truncates toward zero
	long, short := rate >= fundingLongsCrowded, rate <= fundingShortsCrowded
	switch {
	case long && u < fundingLongU:
		u = fundingLongU
	case !long && u >= fundingLongU:
		u = fundingLongU - 1
	}
	switch {
	case short && u > fundingShortU:
		u = fundingShortU
	case !short && u <= fundingShortU:
		u = fundingShortU + 1
	}
	return u
}

func (r fundingRow) crossed() bool  { return r.crossedLong || r.crossedShort }
func (r fundingRow) negative() bool { return r.rate < 0 }

// sideThreshold is the rule's threshold on the rate's own side (raw).
func (r fundingRow) sideThreshold() float64 {
	if r.negative() {
		return fundingShortsCrowded
	}
	return fundingLongsCrowded
}

// ratio is |rate| over its own side's threshold magnitude, raw precision.
func (r fundingRow) ratio() float64 { return math.Abs(r.rate) / math.Abs(r.sideThreshold()) }

func (r fundingRow) thK8() int64 {
	if r.negative() {
		return fundingShortK8
	}
	return fundingLongK8
}

func absI(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}

// beats: a crossed row beats one that is not; otherwise the larger ratio to
// its own side's threshold, compared exactly on the 1e-8 integers. Equal is
// not a win — the earlier symbol in fundingSymbols keeps the pick.
func (r fundingRow) beats(o fundingRow) bool {
	if r.crossed() != o.crossed() {
		return r.crossed()
	}
	return absI(r.k8)*o.thK8() > absI(o.k8)*r.thK8()
}

// ratioShown is the printed ratio, from the printed rate: "0.62×", "1.50×".
func (r fundingRow) ratioShown() string {
	th := int64(fundingLongU)
	if r.negative() {
		th = -fundingShortU
	}
	h := absI(r.u) * 100 / th
	if h == 100 && r.crossed() && r.rate != r.sideThreshold() {
		return ">1.00×" // past the threshold, but by less than the print shows
	}
	return fmt.Sprintf("%d.%02d×", h/100, h%100)
}

func (r fundingRow) pct() string { return fundingPct(r.u) }

// fundingPct prints units as a signed percent with four decimals.
func fundingPct(u int64) string {
	if u == 0 {
		return "0.0000%"
	}
	sign := "+"
	if u < 0 {
		sign = "-"
	}
	a := absI(u)
	return fmt.Sprintf("%s%d.%04d%%", sign, a/10000, a%10000)
}

// fundingPP prints a non-negative distance in units as percentage points.
func fundingPP(u int64) string { return fmt.Sprintf("%d.%04d pp", u/10000, u%10000) }

// fundingPastPP is the distance past a threshold the rate is really past (the
// exact-threshold case is worded separately): under one printed unit it reads
// "<0.0001 pp", never "0.0000 pp".
func fundingPastPP(u int64) string {
	if u <= 0 {
		return "<0.0001 pp"
	}
	return fundingPP(u)
}

// fundingBase is the base coin of a USDT-margined symbol ("SOLUSDT" → "SOL"):
// the machine asset, as on every other agent.
func fundingBase(sym string) string { return strings.TrimSuffix(sym, "USDT") }

var (
	fundingLongShown  = fundingPct(fundingLongU)
	fundingShortShown = fundingPct(fundingShortU)
)

// fundingRows lists the received rates in fundingSymbols order, and the
// symbols that did not answer.
func fundingRows(quotes map[string]fundingQuote) (rows []fundingRow, missing []string) {
	missing = []string{}
	for _, sym := range fundingSymbols {
		q, ok := quotes[sym]
		if !ok {
			missing = append(missing, sym)
			continue
		}
		rows = append(rows, newFundingRow(sym, q.rate))
	}
	return rows, missing
}

// fundingPick is the index of the shown coin (see beats); -1 on no rows.
func fundingPick(rows []fundingRow) int {
	best := -1
	for i, r := range rows {
		if best < 0 || r.beats(rows[best]) {
			best = i
		}
	}
	return best
}

// fundingFit joins parts with " · ", dropping trailing parts until the line
// fits; a single part that is still too long is cut with "…" (backend symbols
// and bands are not length-bounded).
func fundingFit(parts ...string) string {
	for len(parts) > 1 && utf8.RuneCountInString(strings.Join(parts, " · ")) > fundingFactMaxRunes {
		parts = parts[:len(parts)-1]
	}
	s := strings.Join(parts, " · ")
	if r := []rune(s); len(r) > fundingFactMaxRunes {
		s = string(r[:fundingFactMaxRunes-1]) + "…"
	}
	return s
}

func fundingCoverageLine(n int, missing []string) string {
	if len(missing) == 0 {
		return fmt.Sprintf("Coverage: %d/%d Binance majors", n, len(fundingSymbols))
	}
	return fmt.Sprintf("Coverage: %d/%d Binance majors · missing %s", n, len(fundingSymbols), strings.Join(missing, ", "))
}

// Funding machine states (funding.state).
const (
	fundingStatePositive     = "positive_above_threshold"
	fundingStateNegative     = "negative_below_threshold"
	fundingStateWithin       = "within_thresholds"
	fundingStatePartial      = "partial"
	fundingStateRatesOffline = "rates_offline"
)

const (
	fundingVerdictPositive = "Positive funding above threshold — longs pay an elevated rate"
	fundingVerdictNegative = "Negative funding below threshold — shorts pay an elevated rate"
	fundingVerdictWithin   = "Funding within the agent's thresholds"
)

var fundingWhyLevel = fmt.Sprintf("No price level: %s and %s are the agent's own funding thresholds, not a market benchmark",
	fundingLongShown, fundingShortShown)

const fundingLimitations = "One funding rate per symbol: it does not measure open interest, leverage or positions on other venues"

// ── liquidations ─────────────────────────────────────────────────────────────

// fundingLiqRead is the 1h window as the card counted it.
type fundingLiqRead struct {
	count                  int
	usd, longUSD, shortUSD float64
	capped                 bool
}

func fundingLiqWindow(liq *FundingResp, now time.Time) fundingLiqRead {
	var r fundingLiqRead
	cutoff := now.Add(-fundingWindow)
	for _, l := range liq.Feed {
		if l.TS.Before(cutoff) {
			continue
		}
		r.count++
		r.usd += l.USDValue
		switch l.Side {
		case "long_liq":
			r.longUSD += l.USDValue
		case "short_liq":
			r.shortUSD += l.USDValue
		}
	}
	// Capped only when the page is full AND its oldest event is still inside
	// the window: then older events of the hour may be missing. A full page
	// that reaches past the hour counted the whole window.
	if len(liq.Feed) >= fundingFeedLimit {
		oldest := liq.Feed[0].TS
		for _, l := range liq.Feed {
			if l.TS.Before(oldest) {
				oldest = l.TS
			}
		}
		r.capped = !oldest.Before(cutoff)
	}
	return r
}

func fundingLiqLine(r fundingLiqRead) string {
	if r.count == 0 {
		return "Liquidations: no events in the current 1h window"
	}
	head := "Liquidations, last 1h: " + fundingEvents(r.count)
	if r.capped {
		head = fmt.Sprintf("Liquidations, newest %s of the 1h window", fundingEvents(r.count))
	}
	long := "long liqs " + usd(r.longUSD)
	if r.usd > 0 {
		long += fmt.Sprintf(" (%d%%)", int(math.Round(r.longUSD/r.usd*100)))
	}
	return fundingFit(head, usd(r.usd), long, "short liqs "+usd(r.shortUSD))
}

// fundingLiqPhrase is the liquidation half of blocks.what_happened.
func fundingLiqPhrase(r *fundingLiqRead) string {
	switch {
	case r == nil:
		return "liquidation feed offline"
	case r.count == 0:
		return "no liquidation events in the 1h window"
	case r.capped:
		return fmt.Sprintf("liquidations 1h: newest %s, %s", fundingEvents(r.count), usd(r.usd))
	}
	return fmt.Sprintf("liquidations 1h: %s, %s", fundingEvents(r.count), usd(r.usd))
}

// fundingEvents counts events in English: "1 event", "9 events".
func fundingEvents(n int) string {
	if n == 1 {
		return "1 event"
	}
	return fmt.Sprintf("%d events", n)
}

// bandBounds parses a "63100-63400" band.
func bandBounds(band string) (lo, hi float64, ok bool) {
	parts := strings.Split(band, "-")
	if len(parts) != 2 {
		return 0, 0, false
	}
	l, err1 := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
	h, err2 := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	if err1 != nil || err2 != nil || math.IsNaN(l) || math.IsNaN(h) || math.IsInf(l, 0) || math.IsInf(h, 0) || l > h {
		return 0, 0, false
	}
	return l, h, true
}

// Cluster band against the mark price (funding.liquidations.cluster.band_vs_mark).
const (
	fundingBandAbove  = "above"
	fundingBandBelow  = "below"
	fundingBandInside = "inside"
)

var fundingEventTimeRe = regexp.MustCompile(` · last event \d{2}:\d{2} UTC`)

// fundingAIFacts drops the cluster's event time from the AI payload: the
// payload carries no time stamps (see momentumAIFacts).
func fundingAIFacts(facts []string) []string {
	out := make([]string, len(facts))
	for i, f := range facts {
		out[i] = fundingEventTimeRe.ReplaceAllString(f, "")
	}
	return out
}

// fundingClusterZone picks the observed cluster without any price: among the
// BTCUSDT bands (every band when none is BTCUSDT) the largest USD, then more
// events, then the symbol, then the lower band (an unparsable band after a
// parsable one), then the band text — never the served order. The old pick,
// the band nearest to the mark price, swapped bands while the mark wandered
// between two of them: a new body and a hook push with no new data.
func fundingClusterZone(zones []LiqZone) LiqZone {
	pool := make([]LiqZone, 0, len(zones))
	for _, z := range zones {
		if z.Symbol == "BTCUSDT" {
			pool = append(pool, z)
		}
	}
	if len(pool) == 0 {
		pool = zones
	}
	best := pool[0]
	for _, z := range pool[1:] {
		if fundingZoneBeats(z, best) {
			best = z
		}
	}
	return best
}

func fundingZoneBeats(a, b LiqZone) bool {
	if a.TotalUSD != b.TotalUSD {
		return a.TotalUSD > b.TotalUSD
	}
	if a.Count != b.Count {
		return a.Count > b.Count
	}
	if a.Symbol != b.Symbol {
		return a.Symbol < b.Symbol
	}
	alo, _, aok := bandBounds(a.PriceBand)
	blo, _, bok := bandBounds(b.PriceBand)
	if aok != bok {
		return aok
	}
	if aok && alo != blo {
		return alo < blo
	}
	return a.PriceBand < b.PriceBand
}

// fundingCluster reads the observed cluster (fundingClusterZone) and where
// its band sits against the CURRENT mark price of its symbol. It returns the
// two card lines and the machine readout.
func fundingCluster(liq *FundingResp, marks map[string]float64, now time.Time) ([]string, *FundingClusterOut) {
	if len(liq.Zones) == 0 {
		return nil, nil
	}
	z := fundingClusterZone(liq.Zones)
	out := &FundingClusterOut{Symbol: z.Symbol, Side: z.Side, PriceBand: z.PriceBand, USD: z.TotalUSD, Count: z.Count}
	lineA := fundingFit("Observed liquidation cluster, last 1h: "+z.Symbol+" "+z.PriceBand, usd(z.TotalUSD),
		fundingEvents(z.Count))

	side := "side not reported"
	switch z.Side {
	case "long_liq":
		side = "long liqs lead by USD"
	case "short_liq":
		side = "short liqs lead by USD"
	}
	parts := []string{"Cluster: " + side}
	lo, hi, bandOK := bandBounds(z.PriceBand)
	if bandOK {
		var last time.Time
		cutoff := now.Add(-fundingWindow)
		// [lo, hi): the backend bands by floor((price − anchor) ÷ width), so a
		// shared edge belongs to the upper band. The printed edges are rounded
		// (whole dollars on BTC): an event within that rounding of an edge may
		// still be timed to the neighbour band — the time only, never USD/count.
		for _, l := range liq.Feed {
			if l.Symbol == z.Symbol && l.Price >= lo && l.Price < hi && !l.TS.Before(cutoff) && l.TS.After(last) {
				last = l.TS
			}
		}
		if !last.IsZero() {
			parts = append(parts, "last event "+last.UTC().Format("15:04")+" UTC")
			out.LastEventAt = rfcPtr(last)
		}
	}
	// Only WHERE the band sits against the mark price, never a distance: the
	// mark moves every second, and a distance in the body would push the hook
	// (and digest/top when funding wins) on every sweep with no new data. The
	// position changes only when the mark crosses a band edge.
	if mark := marks[z.Symbol]; mark > 0 && bandOK {
		pos := fundingBandInside
		switch {
		case lo > mark:
			pos = fundingBandAbove
		case hi < mark:
			pos = fundingBandBelow
		}
		out.BandVsMark = &pos
		if pos == fundingBandInside {
			parts = append(parts, "the "+z.Symbol+" mark price is inside the band")
		} else {
			parts = append(parts, fmt.Sprintf("band %s the %s mark price", pos, z.Symbol))
		}
	}
	return []string{lineA, fundingFit(parts...)}, out
}

// ── the card ─────────────────────────────────────────────────────────────────

// fundingCardFrom builds the card from one read. Pure: the clock is `now`
// (the request time, also the card's DataTime — funding is a point-in-time
// read). The caller serves the all-offline card itself.
func fundingCardFrom(quotes map[string]fundingQuote, ratesErr error, liq *FundingResp, liqErr error, now time.Time) Card {
	c := Card{
		Agent:      "Funding Agent",
		ShortName:  "Funding",
		Command:    keyFunding,
		HowItWorks: howTexts[keyFunding],
		// Funding IS a point-in-time read — now is honest here, and the
		// footer labels it so (item 7 exempts funding but requires the label).
		DataTime:   now,
		SourceNote: "as of request time",
		// Rates, mark prices, the liquidation feed and the 1h window from now
		// are separate reads; the request time is not a version of that body.
		noValidator: true,
	}
	if ratesErr != nil {
		quotes = nil
	}
	rows, missing := fundingRows(quotes)
	marks := map[string]float64{}
	for sym, q := range quotes {
		if q.mark > 0 && !math.IsInf(q.mark, 0) {
			marks[sym] = q.mark
		}
	}
	ro := &FundingReadout{
		RateKind:   "last_funding_rate",
		Thresholds: FundingThresholds{Long: fundingLongsCrowded, Short: fundingShortsCrowded},
		Coverage: FundingCoverage{Received: len(rows), Total: len(fundingSymbols), Missing: missing,
			MinForVerdict: fundingMinCoverage},
	}
	for _, r := range rows {
		if r.crossed() {
			ro.Crossed++
		}
	}

	var liqRead *fundingLiqRead
	if liqErr == nil && liq != nil {
		lr := fundingLiqWindow(liq, now)
		liqRead = &lr
	}

	sel := -1
	switch {
	case len(rows) == 0:
		ro.State = fundingStateRatesOffline
		c.Emoji, c.Verdict, c.Short = emojiNeutral, "Funding rates unavailable — liquidations only", "rates offline"
		// The headline reading (the rates) was not produced — the envelope
		// says ok=false/source_offline while the liquidation facts render.
		c.Status = statusSourceOffline
		c.Facts = append(c.Facts, fmt.Sprintf("Coverage: 0/%d Binance majors · funding-rate source offline right now", len(fundingSymbols)))
	case len(rows) < fundingMinCoverage:
		ro.State = fundingStatePartial
		c.Emoji = emojiNeutral
		c.Verdict = fmt.Sprintf("Partial funding data — %d/%d Binance majors answered, no verdict", len(rows), len(fundingSymbols))
		c.Short = fmt.Sprintf("partial data (%d/%d)", len(rows), len(fundingSymbols))
		c.Status = statusSourceOffline // no comparison across majors: not a reading to rank
		var got []string
		for _, r := range rows {
			s := r.sym + " " + r.pct()
			switch {
			case r.crossedLong:
				s += " (past the long threshold)"
			case r.crossedShort:
				s += " (past the short threshold)"
			}
			got = append(got, s)
		}
		c.Facts = append(c.Facts,
			fundingFit("Rates received: "+strings.Join(got, " · ")),
			fmt.Sprintf("Thresholds: long %s, short %s · fewer than %d of %d majors, no comparison",
				fundingLongShown, fundingShortShown, fundingMinCoverage, len(fundingSymbols)),
			fundingCoverageLine(len(rows), missing))
	default:
		sel = fundingPick(rows)
		s := rows[sel]
		// Human label = the full symbol (header, digest line and headline);
		// machine asset = the base coin, like every other agent.
		c.Asset, c.AssetKey = s.sym, fundingBase(s.sym)
		c.Deviation = fundingDeviation(s.rate)
		sym := s.sym
		ro.SelectedSymbol = &sym
		reason := "closest_to_own_threshold"
		switch {
		case s.crossedLong:
			ro.State = fundingStatePositive
			c.Emoji, c.Verdict = emojiBear, fundingVerdictPositive
			c.Short = "longs pay an elevated rate · " + s.pct()
			c.confirmed = true
		case s.crossedShort:
			ro.State = fundingStateNegative
			c.Emoji, c.Verdict = emojiBull, fundingVerdictNegative
			c.Short = "shorts pay an elevated rate · " + s.pct()
			c.confirmed = true
		default:
			ro.State = fundingStateWithin
			c.Emoji, c.Verdict = emojiNeutral, fundingVerdictWithin
			c.Short = "within thresholds · " + s.pct()
		}
		if s.crossed() {
			reason = "furthest_past_own_threshold"
		}
		ro.SelectionReason = &reason
		c.Facts = append(c.Facts, fundingCheckLine(s), fundingDistanceLine(s), fundingWhyLine(s, len(rows), ro.Crossed))
		var others []string
		for i, r := range rows {
			if i != sel {
				others = append(others, r.sym+" "+r.pct())
			}
		}
		if len(others) > 0 {
			c.Facts = append(c.Facts, fundingFit("Other majors: "+strings.Join(others, " · ")))
		}
		c.Facts = append(c.Facts, fundingCoverageLine(len(rows), missing))
		c.Blocks = fundingBlocks(s, len(rows), ro.Crossed, liqRead)
	}

	switch {
	case liqRead == nil:
		c.Facts = append(c.Facts, "Liquidation feed offline right now")
	default:
		c.Facts = append(c.Facts, fundingLiqLine(*liqRead))
		lo := &FundingLiquidations{WindowMinutes: int(fundingWindow / time.Minute), Count: liqRead.count,
			USD: liqRead.usd, LongUSD: liqRead.longUSD, ShortUSD: liqRead.shortUSD,
			FeedLimit: fundingFeedLimit, Capped: liqRead.capped}
		// The backend builds its bands over ITS in-memory window, pruned on its
		// own clock; the card counts the feed from the request time. At the
		// edge of the hour the bands can hold events the card's window does
		// not — so with no event in the window the card shows no cluster
		// (cluster null) rather than a band beside "no events".
		if liqRead.count > 0 {
			lines, cl := fundingCluster(liq, marks, now)
			c.Facts = append(c.Facts, lines...)
			lo.Cluster = cl
		}
		ro.Liquidations = lo
	}

	c.Results = fundingResults(rows, sel)
	c.Funding = ro
	return c
}

func fundingCheckLine(s fundingRow) string {
	switch {
	case s.crossedLong:
		return fmt.Sprintf("Last funding rate: %s ≥ %s (long threshold) → longs pay an elevated rate", s.pct(), fundingLongShown)
	case s.crossedShort:
		return fmt.Sprintf("Last funding rate: %s ≤ %s (short threshold) → shorts pay an elevated rate", s.pct(), fundingShortShown)
	}
	return fmt.Sprintf("Last funding rate: %s < %s < %s → within thresholds", fundingShortShown, s.pct(), fundingLongShown)
}

func fundingDistanceLine(s fundingRow) string {
	switch {
	case s.crossedLong && s.rate == fundingLongsCrowded:
		return fmt.Sprintf("Exactly at the long threshold %s · short threshold %s", fundingLongShown, fundingShortShown)
	case s.crossedLong:
		return fmt.Sprintf("Past the long threshold by %s · short threshold %s", fundingPastPP(s.u-fundingLongU), fundingShortShown)
	case s.crossedShort && s.rate == fundingShortsCrowded:
		return fmt.Sprintf("Exactly at the short threshold %s · long threshold %s", fundingShortShown, fundingLongShown)
	case s.crossedShort:
		return fmt.Sprintf("Past the short threshold by %s · long threshold %s", fundingPastPP(fundingShortU-s.u), fundingLongShown)
	}
	dLong, dShort := fundingLongU-s.u, s.u-fundingShortU
	switch {
	case dShort < dLong:
		return fmt.Sprintf("Nearest threshold: short %s, %s away", fundingShortShown, fundingPP(dShort))
	case dLong < dShort:
		return fmt.Sprintf("Nearest threshold: long %s, %s away", fundingLongShown, fundingPP(dLong))
	}
	return fmt.Sprintf("Nearest threshold: equidistant, %s from both", fundingPP(dLong))
}

func fundingWhyLine(s fundingRow, n, crossed int) string {
	if s.crossed() {
		return fmt.Sprintf("Why %s: furthest past its own side's threshold (%s, rate ÷ threshold) · %d of %d past a threshold",
			s.sym, s.ratioShown(), crossed, n)
	}
	// "largest ratio", never "closest": the line above names the nearest
	// threshold in pp, which can be the other side's.
	return fmt.Sprintf("Why %s: largest ratio to its own side's threshold of %d majors (%s, rate ÷ threshold)",
		s.sym, n, s.ratioShown())
}

// fundingBlocks is the content-ready form: what happened (the rate against
// its threshold, the 1h liquidations), two conditional classification
// changes, what ends a reading past a threshold, the LOCAL funding regime.
// No price, no target, no direction.
func fundingBlocks(s fundingRow, n, crossed int, liq *fundingLiqRead) *ContentBlocks {
	b := &ContentBlocks{WhyLevel: fundingWhyLevel, Limitations: fundingLimitations}
	band := fmt.Sprintf("inside %s to %s", fundingShortShown, fundingLongShown)
	regime := fmt.Sprintf("Local perp funding regime · %d/%d Binance majors · ", n, len(fundingSymbols))
	switch {
	case s.crossed():
		side, th := "long", fundingLongShown
		other, otherState := fundingShortShown, "negative funding below threshold"
		ends := fmt.Sprintf("%s funding back below %s ends this reading (neutral once no major is past a threshold)", s.sym, th)
		if s.crossedShort {
			side, th = "short", fundingShortShown
			other, otherState = fundingLongShown, "positive funding above threshold"
			ends = fmt.Sprintf("%s funding back above %s ends this reading (neutral once no major is past a threshold)", s.sym, th)
		}
		b.WhatHappened = fundingFit(fmt.Sprintf("%s funding %s is past the %s threshold %s", s.sym, s.pct(), side, th), fundingLiqPhrase(liq))
		// Two changes of classification: every major back inside, or a major
		// past the other side's threshold at a larger ratio (it becomes the
		// shown coin — crossed rows compare by ratio only). No number for the
		// ratio: its print is truncated, "larger" is exact.
		b.Scenarios = []string{
			fmt.Sprintf("If every major returns %s, the state turns within thresholds", band),
			fmt.Sprintf("If a major is past %s at a larger ratio than %s, the state turns %s", other, s.sym, otherState),
		}
		b.Invalidates = &ends
		b.Regime = regime + fmt.Sprintf("%d past a threshold · %s %s its threshold", crossed, s.sym, s.ratioShown())
	default:
		th := fundingLongShown
		if s.negative() {
			th = fundingShortShown
		}
		b.WhatHappened = fundingFit(fmt.Sprintf("Within thresholds, largest ratio %s %s vs %s", s.sym, s.pct(), th),
			fundingLiqPhrase(liq))
		b.Scenarios = []string{
			fmt.Sprintf("If any major's funding rate reaches %s or above, the state turns positive funding above threshold", fundingLongShown),
			fmt.Sprintf("If any major's funding rate falls to %s or below, the state turns negative funding below threshold", fundingShortShown),
		}
		b.Regime = regime + fmt.Sprintf("none past a threshold · %s %s its threshold", s.sym, s.ratioShown())
	}
	return b
}

// fundingResults is one entry per major in fundingSymbols order: the read at
// raw precision for every received rate, ok=false/source_offline for a symbol
// that did not answer. selected marks the shown coin (false everywhere on a
// partial card).
func fundingResults(rows []fundingRow, sel int) []AssetResult {
	bySym := map[string]int{}
	for i, r := range rows {
		bySym[r.sym] = i
	}
	out := make([]AssetResult, 0, len(fundingSymbols))
	for _, sym := range fundingSymbols {
		i, ok := bySym[sym]
		if !ok {
			reason := statusSourceOffline.reason()
			out = append(out, AssetResult{Asset: fundingBase(sym), Symbol: sym, OK: false, Reason: &reason})
			continue
		}
		r := rows[i]
		rate, th, ratio := r.rate, r.sideThreshold(), r.ratio()
		crossed, selected := r.crossed(), i == sel
		out = append(out, AssetResult{Asset: fundingBase(sym), Symbol: sym, OK: true, Rate: &rate, SideThreshold: &th,
			Crossed: &crossed, RatioToThreshold: &ratio, Selected: &selected})
	}
	return out
}

// fundingConclusion is the /showcase/example conclusion of a funding card
// with a verdict: a classification of the shown coin's funding, never a
// direction. The generic conclusion read the semaphore ("bullish reading on
// XRPUSDT: the numbers above lean up"), which a funding rate does not say.
func fundingConclusion(c Card) string {
	sym := *c.Funding.SelectedSymbol
	var s string
	switch c.Funding.State {
	case fundingStatePositive:
		s = fmt.Sprintf("This is a funding classification on %s, not a forecast: its last funding rate is past the agent's long threshold, so longs pay an elevated rate; it says nothing about where price goes.", sym)
	case fundingStateNegative:
		s = fmt.Sprintf("This is a funding classification on %s, not a forecast: its last funding rate is past the agent's short threshold, so shorts pay an elevated rate; it says nothing about where price goes.", sym)
	default:
		s = fmt.Sprintf("This is a funding classification on %s, not a forecast: its last funding rate is within the agent's thresholds; it says nothing about where price goes.", sym)
	}
	if c.Blocks != nil && c.Blocks.Invalidates != nil {
		s += " " + endSentence(*c.Blocks.Invalidates)
	}
	return s
}
