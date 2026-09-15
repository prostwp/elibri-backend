package demobot

// gold.go — the gold agent's own logic: the "levels of the day" layer.
//
// Everything here is pure: bars in, levels out, no network and no clock. The
// regime half of the agent is NOT reimplemented here — it reads the platform
// trend state machine, so the gold agent inherits any fix to it instead of
// drifting into a second definition of "trend". What the card SAYS lives in
// gold_text.go.
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

// positionOf words where a price sits inside the range. Touching an edge is
// still "inside": the spec's trigger needs a CLOSE beyond the level, so price
// sitting exactly on it has not yet done anything.
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

// goldPx formats a price that the card COMPARES against another price: the
// day's high and low, the last 1h close, the S/R levels, and the invalidation
// level (its line compares a daily close with it).
//
// trimFloat rounds anything ≥1000 to a whole number, and gold trades near
// 4600 in ticks of 0.10. That made the card assert inequalities its own text
// contradicted: with a raw high of 4614.6 and a live close of 4614.8 the
// comparison correctly said "above", while the words read "4615 — above the
// range" against a level also printed as 4615.
//
// Two decimals is the tick resolution of the contract, so a printed pair can
// always be told apart when the code says they differ. Every price the gold
// card prints goes through goldPx.
func goldPx(v float64) string { return fmt.Sprintf("%.2f", v) }

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

// goldIntradaySpec is the same contract at 1h, used ONLY to place the last
// closed hourly price against the day's levels. The daily series carries
// closed days only, so it cannot place the price inside the current day.
var goldIntradaySpec = assetSpec{
	Display:  "GOLD · COMEX GC=F",
	Key:      "XAUUSD",
	Source:   srcYahoo,
	Symbol:   "GC=F",
	Interval: "1h",
}

// goldClosedBanner is the weekend notice for the COMEX contract. Weekend only
// — see the limitation noted at its use site (gold_text.go).
const goldClosedBanner = "⏸ Weekend — COMEX gold futures are closed, data is as of the last session"

// goldSourceNote is the instrument disclosure. Spec section 10 requires it on
// EVERY card, so it lives here rather than inline: the degraded builders need
// the same string, and they used to fall back to a bare "data: Yahoo Finance"
// that quietly dropped "not spot XAUUSD".
const goldSourceNote = "data: Yahoo Finance · COMEX gold futures GC=F, not spot XAUUSD"

// goldHow must stay under 200 chars — Telegram caps callback alerts there.
//
// It says "not a forecast" on purpose. The Этап 5 history run found the
// regime read carries no measurable edge on the NEXT DAY (+0.1 pp over
// baseline on the out-of-sample half), and a one-bar peek moves the verdict on
// only 3.9% of days (measured pairwise by date). The agent describes a
// regime; it does not forecast a day, and the words a reader sees must not
// imply otherwise.
const goldHow = "COMEX gold futures GC=F: daily ADX/EMA regime; day levels from the last closed day, or the enclosing day after inside days; last closed 1h price; macro; volatility. Describes, not a forecast."

// GoldCard is the gold agent: daily regime, the day's levels, macro backdrop,
// volatility — and honest silence about direction whenever the regime has not
// confirmed one.
//
// It COMPOSES the platform blocks rather than reimplementing them: the regime
// is TrendCard's own state machine read off Card.State, the macro backdrop is
// the gold lamp view, volatility is the ATR agent. Any fix to those lands here
// for free, and this agent can never drift into a second definition of trend.
// It gathers the parts; goldCardFrom (gold_text.go) words them.
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

	// The price is fetched BEFORE the verdict, because the verdict depends on
	// having it (conflict priority rule 1, spec section 6): a dead 1h feed
	// once left a green "confirmed UPTREND" card serving ok=true with nothing
	// saying a source had failed.
	px, pxAt, hasPx := a.goldLatestPrice(ctx)
	in := goldInputs{
		trend:     trend,
		levels:    goldDayLevelsOf(daily),
		dailyAsOf: closeTimeOf(daily, goldDailySpec.Interval),
		px:        px,
		pxAt:      pxAt,
		hasPx:     hasPx,
		macro:     a.MacroAssetCard(ctx, macroAssetGold),
		now:       a.clock(),
	}
	if hasPx {
		in.sup, in.res = goldNearestLevels(daily, px)
	}
	// Volatility: the ATR agent's own read, in the Volatility card's own words
	// (volShortLine) — never the machine state.
	if vol := a.VolCard(ctx, goldDailySpec); !vol.Offline && vol.State != "" {
		if lv, ok := vol.Levels.(VolLevels); ok {
			in.vol = volShortLine(lv.Ratio, lv.Timeframe)
		}
	}
	return goldCardFrom(in)
}

// goldLatestPrice is the last CLOSED hourly close and its close time (Yahoo
// bars are cut at fetch time, candlesWindow). ok=false when the intraday
// series is unavailable, and the card then says so rather than pretending the
// day's range is untouched.
func (a *Agents) goldLatestPrice(ctx context.Context) (float64, time.Time, bool) {
	bars, err := a.candlesFor(ctx, goldIntradaySpec)
	if err != nil || len(bars) == 0 {
		return 0, time.Time{}, false
	}
	last := bars[len(bars)-1]
	return last.Close, time.Unix(last.Time, 0).UTC().Add(time.Hour), true
}

// goldPriceStale is how old the last closed 1h price may be before the card
// flags it as stale. Six hours spans an ordinary overnight gap in the COMEX
// session without crying wolf, and is far short of the multi-day staleness
// that used to pass silently. The flag flips once, at the threshold; the card
// never prints a running age (the body must not change between requests while
// the data is the same).
const goldPriceStale = 6 * time.Hour

// lowerFirst lowercases only the first rune, for embedding a verdict
// mid-sentence. strings.ToLower on the whole verdict turned "ADX 18.8" into
// "adx 18.8" and "UPTREND" into "uptrend" on a live gold card.
func lowerFirst(s string) string {
	for i, r := range s {
		return strings.ToLower(string(r)) + s[i+len(string(r)):]
	}
	return s
}

// goldNearestLevels picks the nearest clustered support BELOW price and the
// nearest resistance ABOVE it — a "support" the market already sank through
// is not support any more. nil on a side where nothing clustered.
//
// Every clustered level, NOT the S/R card's top-3-by-strength: "nearest" and
// "strongest" are different questions, and searching the strongest three for
// the nearest one returns the wrong level whenever a weak level sits closer
// than a strong one (Этап 6, Д5). An absent level is never padded with the
// day range, which is a different thing measured a different way.
func goldNearestLevels(candles []types.OHLCVCandle, price float64) (sup, res *SRLevel) {
	if len(candles) == 0 {
		return nil, nil
	}
	supports, resistances := supportResistanceAll(candles, srWing, srTolPct)
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
	return sup, res
}
