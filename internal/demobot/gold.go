package demobot

// gold.go — the gold agent's own logic: the "levels of the day" layer.
//
// Everything here is pure: bars in, levels out, no network and no clock. The
// regime half of the agent is NOT reimplemented here — it reads the platform
// trend state machine, so the gold agent inherits any fix to it instead of
// drifting into a second definition of "trend".
//
// Spec: Спека_золотой_агент.md, section 4.2.

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// dayUnwindCap is how many nested inside-days the level search walks back
// before giving up and reporting no levels at all.
//
// Measured, not chosen: over 1256 closed daily gold bars (five years) an
// inside day occurs on 14.8% of days, nesting depth 1 on 13.6% and depth 2 on
// 1.2% — depth 3 never occurred. Any cap ≥ 2 therefore behaves identically on
// the observed record; a cap of 1 would leave 1.2% of days without levels.
// Three is two plus one step of headroom for a market that changes character.
// Reproduce with:
//
//	MEASURE_GOLD_DAYLEVELS=1 go test ./internal/demobot/ \
//	    -run TestMeasureGoldDayLevels -v
const dayUnwindCap = 3

// Where the live price sits relative to the day's range.
const (
	dayAbove  = "above"
	dayInside = "inside"
	dayBelow  = "below"
)

// goldDayLevels is the bounding range the agent calls "the day's levels": the
// high/low of the last closed daily bar, or — when that bar is an inside day
// and therefore bounds nothing new — the range of the nearest enclosing day.
//
// Undefined is a real, expected answer, not an error: a range nested deeper
// than dayUnwindCap tells the reader the market is compressing, which is more
// honest than handing out levels that price is already inside of.
type goldDayLevels struct {
	High    float64
	Low     float64
	BarTime int64 // open stamp of the bar the range came from
	Unwound int   // inside-days walked back; 0 = the last closed day itself
	Defined bool
}

// insideDay reports whether bar a sits entirely within bar b's range. Equal
// edges count as inside: a range that merely repeats yesterday's bounds
// nothing new either.
func insideDay(a, b types.OHLCVCandle) bool {
	return a.High <= b.High && a.Low >= b.Low
}

// goldDayLevelsOf picks the bounding range from a series of CLOSED daily bars
// (oldest first). It walks back while each bar is inside its predecessor, up
// to dayUnwindCap steps; a chain still nested after that yields Defined=false.
//
// Running out of history is NOT the deep-nesting case: if the walk reaches
// the first bar of the series there is nothing left to be inside of, so that
// bar's own range is the answer.
func goldDayLevelsOf(bars []types.OHLCVCandle) goldDayLevels {
	if len(bars) == 0 {
		return goldDayLevels{}
	}
	i := len(bars) - 1
	unwound := 0
	for i > 0 && insideDay(bars[i], bars[i-1]) {
		if unwound == dayUnwindCap {
			// Still nested after the cap — report nothing rather than a
			// range price has already broken out of.
			return goldDayLevels{}
		}
		i--
		unwound++
	}
	return goldDayLevels{
		High:    bars[i].High,
		Low:     bars[i].Low,
		BarTime: bars[i].Time,
		Unwound: unwound,
		Defined: true,
	}
}

// positionOf words where a live price sits inside the range. Touching an edge
// is still "inside": the spec's trigger needs a CLOSE beyond the level, so
// price sitting exactly on it has not yet done anything.
//
// An undefined range places nothing — "" — because there is no range to be
// inside of.
func (d goldDayLevels) positionOf(price float64) string {
	if !d.Defined {
		return ""
	}
	switch {
	case price > d.High:
		return dayAbove
	case price < d.Low:
		return dayBelow
	default:
		return dayInside
	}
}

// triggerLine is the conditional sentence the card shows. It states what
// WOULD turn the day, and never that it will — the whole agent's honesty
// rests on that distinction (spec section 4.3).
//
// It takes the current price because a position-blind trigger contradicts the
// line above it: with price already at 4652, "day turns up on a close above
// 4615" reads as though nothing has happened. The level still needs a CLOSE to
// count, so the fix is to say both things at once — the side is taken
// intraday, the confirmation is not in yet. hasPrice=false falls back to the
// plain conditional.
func (d goldDayLevels) triggerLine(price float64, hasPrice bool) string {
	if !d.Defined {
		return "Day range undefined — the market is compressing, no day levels to give"
	}
	hi, lo := goldPx(d.High), goldPx(d.Low)
	plain := fmt.Sprintf("Day turns up on a close above %s, down on a close below %s", hi, lo)
	if !hasPrice {
		return plain
	}
	switch d.positionOf(price) {
	case dayAbove:
		return fmt.Sprintf(
			"Price is already above %s intraday — the up day counts only on a CLOSE above it; a close below %s turns it down",
			hi, lo)
	case dayBelow:
		return fmt.Sprintf(
			"Price is already below %s intraday — the down day counts only on a CLOSE below it; a close above %s turns it up",
			lo, hi)
	default:
		return plain
	}
}

// goldPx formats a price that the card COMPARES against another price on the
// same card: the day's high and low, the live price, and the trigger levels.
//
// trimFloat rounds anything ≥1000 to a whole number, and gold trades near
// 4600 in ticks of 0.10. That made the card assert inequalities its own text
// contradicted: with a raw high of 4614.6 and a live close of 4614.8 the
// comparison correctly said "above", while the words read "Price now 4615 —
// above the range" against a level also printed as 4615.
//
// Two decimals is the tick resolution of the contract, so a printed pair can
// always be told apart when the code says they differ. Numbers the card does
// NOT compare — clustered S/R means, the invalidation level — keep trimFloat:
// two decimals on a cluster mean would be false precision.
func goldPx(v float64) string { return fmt.Sprintf("%.2f", v) }

// insideDayNote is the fact explaining an unwound range, or "" when the last
// closed day bounds the day on its own.
func (d goldDayLevels) insideDayNote() string {
	switch {
	case !d.Defined || d.Unwound == 0:
		return ""
	case d.Unwound == 1:
		return "Yesterday was an inside day — levels taken from the day before it"
	default:
		return fmt.Sprintf("%d nested inside days — levels taken from the enclosing day", d.Unwound)
	}
}

// ── the agent ────────────────────────────────────────────────────────────────

// goldDailySpec is the gold agent's own view of the metal: the SAME source as
// the shipped gold config, re-based on daily candles.
//
// Display says what the instrument actually is. Spot XAUUSD does not exist on
// this feed — XAUUSD=X and XAU=X both answer 404 (verified 2026-08-26), so the
// GC=F fallback is not a fallback in practice, it is the source. A card that
// says "XAUUSD" over COMEX futures prices is off by the basis on every level
// it prints, so this agent names the contract.
//
// The registry key stays "xauusd" and the API's asset field keeps its
// documented "XAUUSD" via Card.AssetKey — external consumers are not broken
// by our honesty.
var goldDailySpec = assetSpec{
	Display:  "GOLD · COMEX GC=F",
	Key:      "XAUUSD",
	Source:   srcYahoo,
	Symbol:   "GC=F",
	Interval: "1d",
}

// goldIntradaySpec is the same contract at 1h, used ONLY to place the current
// price against the day's levels. The daily series carries closed days only,
// so it cannot answer "where is price now".
var goldIntradaySpec = assetSpec{
	Display:  "GOLD · COMEX GC=F",
	Key:      "XAUUSD",
	Source:   srcYahoo,
	Symbol:   "GC=F",
	Interval: "1h",
}

// goldHow must stay under 200 chars — Telegram caps callback alerts there.
//
// It says "describes the period" on purpose. The Этап 5 history run found the
// regime read carries no measurable edge on the NEXT DAY (+0.1 pp over
// baseline on the out-of-sample half), and a one-bar peek moves the verdict on
// only 3.9% of days (measured pairwise by date — an earlier aggregate-count
// comparison reported this as 1 day in 2290 and was wrong). The agent
// describes a regime; it does not forecast a day, and the words a reader sees
// must not imply otherwise.
// goldSourceNote is the instrument disclosure. Spec section 10 requires it on
// EVERY card, so it lives here rather than inline: the degraded builders need
// the same string, and they used to fall back to a bare "data: Yahoo Finance"
// that quietly dropped "not spot XAUUSD".
// goldClosedBanner is the weekend notice for the COMEX contract. Weekend only
// — see the limitation noted at its use site.
const goldClosedBanner = "⏸ Weekend — COMEX gold futures are closed, data is as of the last session"

const goldSourceNote = "data: Yahoo Finance · COMEX gold futures GC=F, not spot XAUUSD"

const goldHow = "Daily regime on COMEX gold futures (ADX/EMA state machine), the previous day's range as the levels of the day, macro gold view and volatility. Describes the period — it is not a forecast of the day."

// GoldCard is the gold agent: daily regime, the day's levels, macro backdrop,
// volatility — and honest silence about direction whenever the regime has not
// confirmed one.
//
// It COMPOSES the platform blocks rather than reimplementing them: the regime
// is TrendCard's own state machine read off Card.State, the macro backdrop is
// the gold lamp view, volatility is the ATR agent. Any fix to those lands here
// for free, and this agent can never drift into a second definition of trend.
//
// Conflict priority is fixed (spec section 6), first rule wins:
//  1. no price          → silent, whatever else is alive
//  2. regime unconfirmed → context and levels, no direction, however loud macro is
//  3. regime confirmed, macro against → regime plus an explicit conflict line
//  4. regime confirmed, macro agreeing or absent → regime plain
func (a *Agents) GoldCard(ctx context.Context) Card {
	daily, err := a.candlesFor(ctx, goldDailySpec)
	if err != nil {
		return assetOffline(goldDailySpec, "Gold Agent", "Gold", keyGold, goldHow)
	}

	trend := a.TrendCard(ctx, goldDailySpec)
	if trend.Status == statusInsufficientHistory {
		c := insufficientCard(goldDailySpec, "Gold Agent", "Gold", keyGold, goldHow, "the daily regime (EMA200/ADX)")
		c.DataTime = closeTimeOf(daily, goldDailySpec.Interval)
		return c
	}

	levels := goldDayLevelsOf(daily)
	macro := a.MacroAssetCard(ctx, macroAssetGold)

	c := Card{
		Agent:      "Gold Agent",
		ShortName:  "Gold",
		Asset:      goldDailySpec.Display,
		AssetKey:   goldDailySpec.Key,
		Command:    keyGold,
		HowItWorks: goldHow,
		DataTime:   closeTimeOf(daily, goldDailySpec.Interval),
		State:      trend.State,
		SourceNote: goldSourceNote,
	}

	// The current price is fetched BEFORE the verdict, because the verdict
	// depends on having it (conflict priority rule 1, spec section 6: "нет
	// цены — молчим целиком"). It used to be fetched after, and the rule was
	// simply not implemented: a dead 1h feed left a green "confirmed UPTREND"
	// card serving ok=true with nothing anywhere saying a source had failed.
	//
	// Why no price means no direction, when the regime itself is computed off
	// daily bars that ARE present: everything a reader does with the verdict
	// runs through "where is price relative to the day's levels". Without it
	// the card cannot say whether the day has already turned, so a confirmed
	// direction on top would be a claim the card cannot place in time.
	px, pxAt, hasPx := a.goldLatestPrice(ctx)
	pxAge := time.Since(pxAt)

	confirmed := hasPx && (trend.State == trendUp || trend.State == trendDown)
	macroState := macro.State // "" when the lamps gave no read
	against := (trend.State == trendUp && macroState == goldPressure) ||
		(trend.State == trendDown && macroState == goldSupport)

	// The verdict DESCRIBES the regime; it does not call the day.
	//
	// It used to read "Day bias UP — daily regime confirms an uptrend". The
	// Этап 5 run measured that promise and did not find it: 33.8% vs a 32.1%
	// baseline on the day closing beyond the levels, +0.1 pp on the
	// out-of-sample half — every gap inside the noise. "Bias" is a forecast
	// word, so it is gone. What survives is what was actually measured: the
	// state machine's reading of the period, and the conditional trigger,
	// which was always honest.
	//
	// The semaphore still follows the regime, exactly as the Trend Agent's
	// does: "the market is in an uptrend" is a description, and colouring it
	// green states that, not a call to buy.
	switch {
	case !hasPx:
		// A partial failure must READ as one and must not serve ok=true.
		c.Emoji = emojiNeutral
		c.Verdict = "Live price unavailable — no direction claimed"
		c.Short = "no live price"
		c.Offline = true
		c.Status = statusSourceOffline
	case !confirmed:
		c.Emoji = emojiNeutral
		c.Verdict = "Daily regime: not confirmed — no direction claimed"
		c.Short = "regime unconfirmed"
	case trend.State == trendUp:
		c.Emoji, c.Short = emojiBull, "regime uptrend"
		c.Verdict = "Daily regime: confirmed UPTREND"
	default:
		c.Emoji, c.Short = emojiBear, "regime downtrend"
		c.Verdict = "Daily regime: confirmed DOWNTREND"
	}

	// The regime fact carries the REASON, so it earns its line only when the
	// verdict above did not already say everything: an unconfirmed state has a
	// why ("flat — no trend to read (ADX 18 < 20)"), a confirmed one would
	// merely repeat the headline. Always in the trend agent's own words —
	// never a re-derived one.
	// One fact per state, and never two that read as an argument. Without a
	// price the regime IS known, so hiding it would be its own dishonesty —
	// but printing a bare "Regime: confirmed uptrend" under a header saying
	// "no direction claimed" reads as the card contradicting itself. So the
	// two are said in one breath: what was read, and why it is not being
	// turned into a direction.
	switch {
	case !hasPx:
		c.Facts = append(c.Facts, fmt.Sprintf(
			"Intraday price feed is down — the daily regime reads %s, but without a live price "+
				"the card cannot place it against the day's levels and does not state a direction",
			strings.ToLower(trend.Verdict)))
	case !confirmed:
		c.Facts = append(c.Facts, "Regime: "+strings.ToLower(trend.Verdict))
	}

	// The day's levels and where price sits against them.
	if note := levels.insideDayNote(); note != "" {
		c.Facts = append(c.Facts, note)
	}
	if levels.Defined {
		c.Facts = append(c.Facts, fmt.Sprintf("Day range %s: %s – %s",
			time.Unix(levels.BarTime, 0).UTC().Format("2006-01-02"),
			goldPx(levels.Low), goldPx(levels.High)))
		if hasPx {
			// The price carries its own timestamp: the card's footer stamp
			// comes from the DAILY series and says nothing about how fresh
			// this number is.
			line := fmt.Sprintf("Price now %s — %s the range (%s UTC)",
				goldPx(px), levels.positionOf(px), pxAt.Format("15:04"))
			if pxAge > goldPriceStale {
				line += fmt.Sprintf(" ⚠ %.0fh old — the feed has not updated", pxAge.Hours())
			}
			c.Facts = append(c.Facts, line)
		}
	}
	c.Facts = append(c.Facts, levels.triggerLine(px, hasPx))

	// Macro backdrop. An absent read is stated, never silently dropped.
	switch {
	case macroState == "":
		c.Facts = append(c.Facts, "Macro: no gold read right now")
	case against:
		c.Facts = append(c.Facts, fmt.Sprintf(
			"Macro backdrop: %s for gold — CONFLICT with the regime, both stand as read", macroState))
	default:
		c.Facts = append(c.Facts, "Macro backdrop: "+macroState+" for gold")
	}

	// Key levels: the nearest clustered support and resistance from the S/R
	// agent, on the same daily series. One line, not the whole S/R card — the
	// reader who wants the full ladder has /sr for it.
	if line := a.goldKeyLevels(ctx, px, hasPx); line != "" {
		c.Facts = append(c.Facts, line)
	}

	// Volatility: the ATR agent's own read, stated in its own words.
	if vol := a.VolCard(ctx, goldDailySpec); !vol.Offline && vol.State != "" {
		if lv, ok := vol.Levels.(VolLevels); ok {
			c.Facts = append(c.Facts, fmt.Sprintf("Volatility: %s (ATR now %.2f× its 30-bar average)",
				vol.State, lv.ExpansionRatio))
		}
	}

	// Invalidation rides along from the trend card's own levels.
	// Invalidation belongs to a stated direction. Under a header that claims
	// none, a bare "Invalidation: below X" is a directional level with nothing
	// to invalidate — so confirmed states only.
	if lv, ok := trend.Levels.(TrendLevels); ok && lv.Invalidation > 0 && confirmed {
		c.Facts = append(c.Facts, fmt.Sprintf("Invalidation: %s %s — below this the regime above is broken",
			lv.InvalidationSide, trimFloat(lv.Invalidation)))
		c.Levels = lv
	}

	// The FX weekend banner does not belong on a COMEX card: it names the
	// wrong market and quotes a "Friday close" that is not this contract's.
	//
	// KNOWN LIMITATION, stated rather than papered over: this is the weekend
	// only. COMEX exchange holidays and the daily session break are NOT
	// modelled — we have no holiday calendar source — so on a holiday the
	// agent treats the market as open and simply serves the last closed daily
	// bar. Because the card always prints the DATE of the day it describes,
	// that staleness is visible to a reader rather than hidden.
	if !isForexOpen(time.Now()) {
		c.Facts = append([]string{goldClosedBanner}, c.Facts...)
	}
	return c
}

// goldLatestPrice is the last CLOSED hourly close — the honest stand-in for
// "price now" on a daily agent. ok=false when the intraday series is
// unavailable, and the card then simply omits the placement rather than
// pretending the day's range is untouched.
func (a *Agents) goldLatestPrice(ctx context.Context) (float64, time.Time, bool) {
	bars, err := a.candlesFor(ctx, goldIntradaySpec)
	if err != nil || len(bars) == 0 {
		return 0, time.Time{}, false
	}
	last := bars[len(bars)-1]
	return last.Close, time.Unix(last.Time, 0).UTC().Add(time.Hour), true
}

// goldPriceStale is how old the "live" price may be before the card must say
// so. Six hours spans an ordinary overnight gap in the COMEX session without
// crying wolf, and is far short of the multi-day staleness that used to pass
// silently: the old code checked only that the intraday series was non-empty,
// so a feed frozen last week still printed "Price now" under a footer stamped
// from the daily series.
const goldPriceStale = 6 * time.Hour

// touchCount words a touch tally: "1 touch", "4 touches". A card that prints
// "1 touches" reads as unfinished work, and this text is the product.
func touchCount(n int) string {
	if n == 1 {
		return "1 touch"
	}
	return fmt.Sprintf("%d touches", n)
}

// goldKeyLevels renders the nearest clustered support/resistance around the
// current price as one line. Empty when the S/R read is unavailable or nothing
// clustered on the relevant side — an absent level is never padded with the
// day range, which is a different thing measured a different way.
func (a *Agents) goldKeyLevels(ctx context.Context, price float64, hasPrice bool) string {
	if !hasPrice {
		return ""
	}
	// Every clustered level, NOT the S/R card's top-3-by-strength: "nearest"
	// and "strongest" are different questions, and searching the strongest
	// three for the nearest one returns the wrong level whenever a weak level
	// sits closer than a strong one (Этап 6, Д5).
	candles, err := a.candlesFor(ctx, goldDailySpec)
	if err != nil || len(candles) == 0 {
		return ""
	}
	supports, resistances := supportResistanceAll(candles, srWing, srTolPct)

	// Nearest support BELOW price and nearest resistance ABOVE it: a "support"
	// the market already sank through is not support any more.
	var sup, res *SRLevel
	for i := range supports {
		p := &supports[i]
		if p.Raw < price && (sup == nil || p.Raw > sup.Raw) {
			sup = p
		}
	}
	for i := range resistances {
		p := &resistances[i]
		if p.Raw > price && (res == nil || p.Raw < res.Raw) {
			res = p
		}
	}
	switch {
	case sup == nil && res == nil:
		return ""
	case res == nil:
		return fmt.Sprintf("Nearest levels: support %s (%s), nothing clustered above",
			goldPx(sup.Raw), touchCount(sup.Touches))
	case sup == nil:
		return fmt.Sprintf("Nearest levels: resistance %s (%s), nothing clustered below",
			goldPx(res.Raw), touchCount(res.Touches))
	default:
		return fmt.Sprintf("Nearest levels: support %s (%s) · resistance %s (%s)",
			goldPx(sup.Raw), touchCount(sup.Touches), goldPx(res.Raw), touchCount(res.Touches))
	}
}
