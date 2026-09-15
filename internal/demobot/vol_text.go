package demobot

// vol_text.go — everything the Volatility card SAYS.
//
// Presentation only. The rule is volState (agents.go): ATR(14) of the last
// closed candle over the mean of the 30 ATR(14) values before it; a ratio
// ≤ 0.80 is compressed, ≥ 1.25 expanding, anything between normal. Nothing
// here changes a state, the semaphore (always neutral), the candles or the
// timeframe.
//
// Honesty rules for every line below:
//   - analytics only: no forecast ("expansion often follows" is gone — the
//     agent never measured what follows a compression), no advice, no price
//     level, no direction, no breakout;
//   - the formula sees a LEVEL, not a dynamic: the machine state keeps its
//     name "expanding", the card says "elevated" (ATR above its baseline),
//     never "widening" or "expanding";
//   - 0.80 and 1.25 are the agent's own thresholds, never a market benchmark;
//   - the ratio is printed through volRatioShown only (three decimals, rounded
//     toward 1), so a printed ratio never sits on the other side of a
//     threshold from the real one, and the verdict's percentage comes from
//     the printed ratio;
//   - the card prints one absolute number (ATR, one decimal finer than the
//     card's price precision) and no absolute baseline, so no pair of printed
//     numbers can give another ratio than the printed one (the old EURUSD
//     card printed 0.0008 / 0.0009 beside 0.92×); the raw values are in
//     levels;
//   - at most four content lines on the card (the weekend banner is
//     furniture), every line — verdict, facts, digest line, each content
//     block — fits volFactMaxRunes.

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// volFactMaxRunes is the readability budget for one line of volatility text.
const volFactMaxRunes = 110

// The rule's thresholds — the single definition, read by volState.
const (
	volCompressedAt = 0.80 // ratio at or below: compressed
	volExpandingAt  = 1.25 // ratio at or above: expanding (worded "elevated")
)

// The ratio in thousandths at the two thresholds (integer arithmetic keeps the
// printed values exact).
const (
	volCompressedK = 800
	volExpandingK  = 1250
	volRatioCapK   = 99999 // widest printed ratio: 99.999
)

var (
	volLoShown = fmt.Sprintf("%.2f", volCompressedAt)
	volHiShown = fmt.Sprintf("%.2f", volExpandingAt)
)

// volThresholdLine is the one statement of the thresholds on the card.
var volThresholdLine = fmt.Sprintf("Agent's own thresholds, not a market benchmark: ≤ %s compressed, ≥ %s elevated, between them normal",
	volLoShown, volHiShown)

var volWhyLevel = fmt.Sprintf("No price level: %s and %s are the agent's own ATR/baseline thresholds, not a market benchmark",
	volLoShown, volHiShown)

const volLimitations = "Measures how far price moves per candle, not its direction; it does not confirm a breakout"

// volRead is one computed reading: the last closed candle's ATR(14), the mean
// of the 30 ATR(14) values before it, the last close and the timeframe.
type volRead struct {
	atr, baseline, price float64
	interval             string
}

func (r volRead) ratio() float64 { return r.atr / r.baseline }

// valid: a ratio exists and every printed number is finite. A zero baseline
// (0/0) or overflowing prices degrade to insufficient history instead.
func (r volRead) valid() bool {
	fin := func(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
	return fin(r.atr) && fin(r.baseline) && fin(r.price) && r.atr >= 0 && r.baseline > 0 && r.price > 0 &&
		fin(r.ratio()) && fin(r.atr/r.price*100)
}

// Reasons an invalid read degrades with (after "insufficient history for ").
const (
	volReasonNonFinite = "ATR(14) 30-bar baseline (non-finite input)"
	volReasonFlatZero  = "ATR(14) 30-bar baseline (the baseline is flat zero — nothing to compare against)"
	volReasonNoPrice   = "ATR(14) as a share of price (the last close is not positive)"
)

// invalidReason names why a read is not valid, in the order a reader needs:
// numbers that do not exist, then no baseline, then no price to relate ATR to.
func (r volRead) invalidReason() string {
	fin := func(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
	switch {
	case !fin(r.atr) || !fin(r.baseline) || !fin(r.price) || r.atr < 0:
		return volReasonNonFinite
	case r.baseline <= 0:
		return volReasonFlatZero
	case r.price <= 0:
		return volReasonNoPrice
	}
	return volReasonNonFinite // the ratio or the share of price overflows
}

// volRatioK is the ratio as printed, in thousandths, rounded toward 1: below
// 1 it is ceiled, from 1 up floored, so a ratio above 0.80 never prints at or
// below 0.800 and a ratio under 1.25 never prints at or above 1.250 (at %.2f,
// 1.246 printed "1.25" under a normal state). The guards catch float noise at
// the edges. Never more than 0.001 from the real ratio below the cap.
func volRatioK(r float64) int {
	x := math.Min(r*1000, 1e8)
	var k int
	if r < 1 {
		k = int(math.Ceil(x))
	} else {
		k = int(math.Floor(x))
	}
	switch {
	case r <= volCompressedAt && k > volCompressedK:
		k = volCompressedK
	case r > volCompressedAt && k <= volCompressedK:
		k = volCompressedK + 1
	}
	switch {
	case r >= volExpandingAt && k < volExpandingK:
		k = volExpandingK
	case r < volExpandingAt && k >= volExpandingK:
		k = volExpandingK - 1
	}
	return k
}

func volKShown(k int) string {
	if k > volRatioCapK {
		return ">99.999"
	}
	return fmt.Sprintf("%d.%03d", k/1000, k%1000)
}

// volRatioShown is the only way volatility text prints the ratio.
func volRatioShown(r float64) string { return volKShown(volRatioK(r)) }

// volDistance words the printed ratio's distance from 1 as a whole percent,
// truncated toward 0: inside the range it never reaches "20% below" or
// "25% above".
func volDistance(r float64) string {
	d := volRatioK(r) - 1000
	side := "above"
	if d < 0 {
		side, d = "below", -d
	}
	p := d / 10
	switch {
	case p == 0:
		return "within 1% of its 30-bar baseline"
	case p > 999:
		return ">999% " + side + " its 30-bar baseline"
	}
	return fmt.Sprintf("%d%% %s its 30-bar baseline", p, side)
}

// volWord is the state as the card text says it. The machine state stays
// expanding | normal | compressed; only expanding is reworded ("elevated":
// the formula sees a level, not a widening). Not "within range" for normal:
// to a trader "range" means a sideways market, and the FX card already uses
// "place in range".
func volWord(state string) string {
	switch state {
	case volExpanding:
		return "elevated"
	case volCompressed:
		return "compressed"
	}
	return "normal"
}

// volShortLine is the compact read — the digest one-liner, and the Gold
// card's volatility line: "normal · 1d · ATR 1.000× its 30-bar baseline".
func volShortLine(ratio float64, tf string) string {
	return fmt.Sprintf("%s · %s · ATR %s× its 30-bar baseline", volWord(volState(ratio)), candleWord(tf), volRatioShown(ratio))
}

// volCheckLine is the one checklist: the printed ratio against the threshold
// that decides the state.
func volCheckLine(r float64) string {
	shown := volRatioShown(r)
	switch volState(r) {
	case volExpanding:
		return fmt.Sprintf("ATR/baseline check: %s ≥ %s → elevated", shown, volHiShown)
	case volCompressed:
		return fmt.Sprintf("ATR/baseline check: %s ≤ %s → compressed", shown, volLoShown)
	}
	return fmt.Sprintf("ATR/baseline check: %s < %s < %s → normal", volLoShown, shown, volHiShown)
}

// volPctShown is ATR as a share of the last close, bounded to six runes.
func volPctShown(v float64) string {
	switch {
	case v >= 999.995:
		return ">999%"
	case v > 0 && v < 0.005:
		return "<0.01%"
	}
	return fmt.Sprintf("%.2f%%", v)
}

// volATRLine prints ATR one decimal finer than the card's price precision
// (FX 5, USDJPY 3, gold and ETH 2, BTC 1) with its share of price.
func volATRLine(spec assetSpec, r volRead) string {
	return fmt.Sprintf("ATR(14): %.*f, %s of price · baseline = mean of the previous 30 ATR(14) values",
		srDecimals(spec, r.price)+1, r.atr, volPctShown(r.atr/r.price*100))
}

func volScenarios(tfw string, r float64) []string {
	switch volState(r) {
	case volExpanding:
		return []string{
			fmt.Sprintf("If closed %s candles keep the ratio at %s or above, the state stays elevated", tfw, volHiShown),
			fmt.Sprintf("If a closed %s candle puts the ratio below %s, the state turns normal (compressed at %s or below)",
				tfw, volHiShown, volLoShown),
		}
	case volCompressed:
		return []string{
			fmt.Sprintf("If closed %s candles keep the ratio at %s or below, the state stays compressed", tfw, volLoShown),
			fmt.Sprintf("If a closed %s candle puts the ratio above %s, the state turns normal (elevated at %s or above)",
				tfw, volLoShown, volHiShown),
		}
	}
	return []string{
		fmt.Sprintf("If a closed %s candle puts the ratio at %s or below, the state turns compressed", tfw, volLoShown),
		fmt.Sprintf("If a closed %s candle puts the ratio at %s or above, the state turns elevated", tfw, volHiShown),
	}
}

func volStateChanges(tfw string, r float64) string {
	switch volState(r) {
	case volExpanding:
		return fmt.Sprintf("A closed %s candle with the ratio below %s ends the elevated state", tfw, volHiShown)
	case volCompressed:
		return fmt.Sprintf("A closed %s candle with the ratio above %s ends the compressed state", tfw, volLoShown)
	}
	return fmt.Sprintf("A closed %s candle with the ratio at %s or below (compressed) or %s or above (elevated) changes the state",
		tfw, volLoShown, volHiShown)
}

// volBlocks is the content-ready form (same ContentBlocks contract as trend /
// S/R / momentum): why_level explains the thresholds (there is no price
// level), invalidates is null (no directional idea), state_changes_when and
// limitations are volatility's own additive keys, regime is the LOCAL
// amplitude regime — not a market regime.
func volBlocks(label string, r volRead) *ContentBlocks {
	ratio, tfw := r.ratio(), candleWord(r.interval)
	word, shown := volWord(volState(ratio)), volRatioShown(ratio)
	return &ContentBlocks{
		WhatHappened: fmt.Sprintf("%s %s ATR(14) is %s (ratio %s): %s.", label, tfw, volDistance(ratio), shown, word),
		WhyLevel:     volWhyLevel,
		Scenarios:    volScenarios(tfw, ratio),
		Regime:       fmt.Sprintf("Local amplitude regime · %s %s · %s: ATR %s× its 30-bar baseline", label, tfw, word, shown),
		StateChanges: volStateChanges(tfw, ratio),
		Limitations:  volLimitations,
	}
}

// volLevelsOf is the machine readout at raw precision.
func volLevelsOf(r volRead) VolLevels {
	ratio := r.ratio()
	return VolLevels{
		ExpansionRatio: ratio,
		State:          volState(ratio),
		Timeframe:      r.interval,
		ATR:            r.atr,
		ATRPct:         r.atr / r.price * 100,
		Baseline:       r.baseline,
		Ratio:          ratio,
		Thresholds:     VolThresholds{Compressed: volCompressedAt, Expanding: volExpandingAt},
	}
}

// volCardFrom builds the card from one valid read. Pure: no clock — the
// single-asset Binance card is stamped, so its body must be a function of the
// bars alone. The Yahoo furniture (decorateFXAt) and the validator flags are
// the caller's.
func volCardFrom(spec assetSpec, r volRead, dataTime time.Time) Card {
	ratio := r.ratio()
	state := volState(ratio)
	word, tfw := volWord(state), candleWord(r.interval)
	return Card{
		Emoji:      emojiNeutral,
		Agent:      "Volatility Agent",
		ShortName:  "Volatility",
		Asset:      spec.Display,
		AssetKey:   spec.Key,
		Command:    keyVol,
		HowItWorks: howTexts[keyVol],
		DataTime:   dataTime,
		Levels:     volLevelsOf(r),
		State:      state,
		Verdict:    fmt.Sprintf("%s · %s — ATR %s", strings.ToUpper(word), tfw, volDistance(ratio)),
		Short:      volShortLine(ratio, r.interval),
		Facts: []string{
			volCheckLine(ratio),
			volATRLine(spec, r),
			volThresholdLine,
			fmt.Sprintf("Read on closed %s candles; measures how far price moves per candle, not direction or breakout", tfw),
		},
		Blocks: volBlocks(spec.Display, r),
	}
}
