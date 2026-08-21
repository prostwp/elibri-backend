// Package demobot is the thin Telegram presentation layer over the AlphaVizor
// backend REST API. Indicator math in this file is intentionally small and
// self-contained: the ml package keeps its implementations unexported, so the
// bot carries its own copies (same Wilder conventions as internal/ml
// features_v2.go) with first-order tests in indicators_test.go.
package demobot

import (
	"math"
	"sort"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// isFinite rejects NaN and ±Inf — the candle parsers use it as the choke
// point so no non-finite float can reach indicator math, cards or levels
// (review fix 4).
func isFinite(f float64) bool {
	return !math.IsNaN(f) && !math.IsInf(f, 0)
}

// rsiWilder returns the latest RSI(period) using Wilder smoothing:
// seed averages are simple means of the first `period` changes, then
// avg = (avg*(p-1) + current) / p. Short series → ok=false, NEVER a
// neutral-looking sentinel (adversarial review item 6).
func rsiWilder(closes []float64, period int) (float64, bool) {
	if len(closes) < period+1 {
		return 0, false
	}
	var gain, loss float64
	for i := 1; i <= period; i++ {
		d := closes[i] - closes[i-1]
		if d > 0 {
			gain += d
		} else {
			loss -= d
		}
	}
	avgGain := gain / float64(period)
	avgLoss := loss / float64(period)
	for i := period + 1; i < len(closes); i++ {
		d := closes[i] - closes[i-1]
		var g, l float64
		if d > 0 {
			g = d
		} else {
			l = -d
		}
		avgGain = (avgGain*float64(period-1) + g) / float64(period)
		avgLoss = (avgLoss*float64(period-1) + l) / float64(period)
	}
	if avgLoss == 0 {
		return 100, true
	}
	rs := avgGain / avgLoss
	return 100 - 100/(1+rs), true
}

// emaSeries mirrors internal/ml/features_v2.go: seed = first value,
// k = 2/(period+1).
func emaSeries(values []float64, period int) []float64 {
	n := len(values)
	out := make([]float64, n)
	if n == 0 {
		return out
	}
	k := 2.0 / (float64(period) + 1.0)
	out[0] = values[0]
	for i := 1; i < n; i++ {
		out[i] = values[i]*k + out[i-1]*(1-k)
	}
	return out
}

// emaLast returns the last EMA value; ok=false when the series is shorter
// than the period (a seed-heavy EMA over 3 bars is noise, not an EMA200).
func emaLast(values []float64, period int) (float64, bool) {
	if len(values) < period {
		return 0, false
	}
	s := emaSeries(values, period)
	return s[len(s)-1], true
}

// trueRange for bar i (i >= 1).
func trueRange(h, l, prevClose float64) float64 {
	tr := h - l
	if v := math.Abs(h - prevClose); v > tr {
		tr = v
	}
	if v := math.Abs(l - prevClose); v > tr {
		tr = v
	}
	return tr
}

// atrSeriesWilder returns the Wilder ATR series. out[i] is valid for
// i >= period (seed at i == period is the simple mean of the first `period`
// true ranges); earlier entries are 0.
func atrSeriesWilder(high, low, close []float64, period int) []float64 {
	n := len(close)
	out := make([]float64, n)
	if n < period+1 {
		return out
	}
	var sum float64
	for i := 1; i <= period; i++ {
		sum += trueRange(high[i], low[i], close[i-1])
	}
	out[period] = sum / float64(period)
	for i := period + 1; i < n; i++ {
		tr := trueRange(high[i], low[i], close[i-1])
		out[i] = (out[i-1]*float64(period-1) + tr) / float64(period)
	}
	return out
}

// adxWilder computes the classic Wilder ADX(period): Wilder-smoothed
// +DM/-DM/TR → DI± → DX, then Wilder-smoothed DX. ok=false when the series
// is too short to produce a single ADX value (needs 2*period bars) — no
// sentinel 0 that would render as a confident "flat".
func adxWilder(high, low, close []float64, period int) (float64, bool) {
	n := len(close)
	if n < 2*period+1 {
		return 0, false
	}
	// Seed sums over the first `period` bars of DM/TR (bars 1..period).
	var smTR, smPlus, smMinus float64
	for i := 1; i <= period; i++ {
		up := high[i] - high[i-1]
		dn := low[i-1] - low[i]
		if up > dn && up > 0 {
			smPlus += up
		}
		if dn > up && dn > 0 {
			smMinus += dn
		}
		smTR += trueRange(high[i], low[i], close[i-1])
	}
	dxAt := func() float64 {
		if smTR == 0 {
			return 0
		}
		diPlus := 100 * smPlus / smTR
		diMinus := 100 * smMinus / smTR
		if diPlus+diMinus == 0 {
			return 0
		}
		return 100 * math.Abs(diPlus-diMinus) / (diPlus + diMinus)
	}
	var adx float64
	var dxCount int
	var dxSum float64
	// First DX available after the seed window.
	dxSum += dxAt()
	dxCount++
	for i := period + 1; i < n; i++ {
		up := high[i] - high[i-1]
		dn := low[i-1] - low[i]
		var plusDM, minusDM float64
		if up > dn && up > 0 {
			plusDM = up
		}
		if dn > up && dn > 0 {
			minusDM = dn
		}
		// Wilder smoothing of the running sums.
		smPlus = smPlus - smPlus/float64(period) + plusDM
		smMinus = smMinus - smMinus/float64(period) + minusDM
		smTR = smTR - smTR/float64(period) + trueRange(high[i], low[i], close[i-1])

		dx := dxAt()
		if dxCount < period {
			dxSum += dx
			dxCount++
			if dxCount == period {
				adx = dxSum / float64(period) // ADX seed = simple mean of first period DXs
			}
			continue
		}
		adx = (adx*float64(period-1) + dx) / float64(period)
	}
	if dxCount < period {
		return dxSum / float64(dxCount), true // best effort on barely-long-enough series
	}
	return adx, true
}

// swingPoint is one swing extreme with the bar index it occurred on — the
// index is what lets B4 attach volume, dates and order to a level's touches.
type swingPoint struct {
	idx   int
	price float64
}

// swingPointsIdx finds swing highs/lows with their bar indices: bar i is a
// swing high when high[i] is strictly greater than the highs of `wing` bars
// on each side (mirror for lows). Bars too close to either end never qualify.
func swingPointsIdx(high, low []float64, wing int) (swingHighs, swingLows []swingPoint) {
	n := len(high)
	for i := wing; i < n-wing; i++ {
		isHigh, isLow := true, true
		for j := 1; j <= wing; j++ {
			if high[i] <= high[i-j] || high[i] <= high[i+j] {
				isHigh = false
			}
			if low[i] >= low[i-j] || low[i] >= low[i+j] {
				isLow = false
			}
			if !isHigh && !isLow {
				break
			}
		}
		if isHigh {
			swingHighs = append(swingHighs, swingPoint{idx: i, price: high[i]})
		}
		if isLow {
			swingLows = append(swingLows, swingPoint{idx: i, price: low[i]})
		}
	}
	return swingHighs, swingLows
}

// swingLevels is the values-only view of swingPointsIdx (chronological order
// preserved) — the trend agent's HH/HL read keys on it.
func swingLevels(high, low []float64, wing int) (swingHighs, swingLows []float64) {
	hp, lp := swingPointsIdx(high, low, wing)
	swingHighs = make([]float64, len(hp))
	for i, p := range hp {
		swingHighs[i] = p.price
	}
	swingLows = make([]float64, len(lp))
	for i, p := range lp {
		swingLows[i] = p.price
	}
	return swingHighs, swingLows
}

// hhhlStructure classifies market structure from the swing pivots (review
// fix 7: alternation-aware). The pivots are merged CHRONOLOGICALLY and the
// last six must form a proper alternating sequence (H-L-H-L-H-L or the L-first
// mirror) — a double top/bottom inside the window, or a same-bar high+low
// pair, is not a market structure and reads as "mixed". Within an alternating
// window (exactly 3 highs + 3 lows):
//
//	both strictly rising  → "hh_hl" (higher highs + higher lows)
//	both strictly falling → "lh_ll" (lower highs + lower lows)
//	anything else         → "mixed"
//	fewer than 6 pivots total → "" (not computable — no claim)
func hhhlStructure(swingHighs, swingLows []swingPoint) string {
	type pivot struct {
		idx    int
		price  float64
		isHigh bool
	}
	pivots := make([]pivot, 0, len(swingHighs)+len(swingLows))
	for _, p := range swingHighs {
		pivots = append(pivots, pivot{p.idx, p.price, true})
	}
	for _, p := range swingLows {
		pivots = append(pivots, pivot{p.idx, p.price, false})
	}
	const need = 6
	if len(pivots) < need {
		return ""
	}
	sort.Slice(pivots, func(i, j int) bool {
		if pivots[i].idx != pivots[j].idx {
			return pivots[i].idx < pivots[j].idx
		}
		return !pivots[i].isHigh // deterministic order for same-bar pairs; alternation check flags them below
	})
	tail := pivots[len(pivots)-need:]
	for k := 1; k < len(tail); k++ {
		if tail[k].isHigh == tail[k-1].isHigh || tail[k].idx == tail[k-1].idx {
			return "mixed" // not an alternating pivot sequence
		}
	}
	var highs, lows []float64 // 3 each, chronological, by alternation
	for _, p := range tail {
		if p.isHigh {
			highs = append(highs, p.price)
		} else {
			lows = append(lows, p.price)
		}
	}
	dir := func(vals []float64) string {
		rising, falling := true, true
		for i := 1; i < len(vals); i++ {
			if vals[i] <= vals[i-1] {
				rising = false
			}
			if vals[i] >= vals[i-1] {
				falling = false
			}
		}
		switch {
		case rising:
			return "up"
		case falling:
			return "down"
		default:
			return "mixed"
		}
	}
	h, l := dir(highs), dir(lows)
	switch {
	case h == "up" && l == "up":
		return "hh_hl"
	case h == "down" && l == "down":
		return "lh_ll"
	default:
		return "mixed"
	}
}

// SRLevel is one clustered support/resistance level. Level is rounded to an
// integer per the original BTC-scale card contract; Raw keeps the cluster
// mean at full precision for FX-scale prices (EURUSD at 1.1583 must not
// collapse to "1"). Touches is the number of swing points that fell into the
// cluster.
//
// B4 fields — every formula documented at its computation site:
//   - Strength = touches + 0.5 per touch whose bar volume exceeds the window's
//     median volume (srStrengthOf). On volume-less series (FX via Yahoo)
//     Strength == Touches.
//   - Weakening = ≥7 touches AND the mean volume of the last 3 touches is
//     below the mean of the first 3 (srWeakening) — the market keeps testing
//     the level but with fading participation.
//   - Breaks/Holds = frequency counts over the window (breakHoldStats):
//     "held 6 of 8 tests" wording, never a probability claim.
//   - LastTouch = the bar time (open, UTC) of the newest touch.
type SRLevel struct {
	Level     int
	Raw       float64
	Touches   int
	Strength  float64
	Weakening bool
	Breaks    int
	Holds     int
	LastTouch time.Time
}

// srCluster is a price cluster of swing points with the member bar indices in
// chronological order.
type srCluster struct {
	mean float64
	idxs []int
}

// clusterSwingPoints groups nearby swing points: prices within tolPct percent
// of the running cluster mean merge into one cluster (identical merge rule to
// the original clusterLevels). Member indices come out chronological.
func clusterSwingPoints(points []swingPoint, tolPct float64) []srCluster {
	if len(points) == 0 {
		return nil
	}
	sorted := append([]swingPoint(nil), points...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].price < sorted[j].price })

	var out []srCluster
	var sum float64
	var idxs []int
	flush := func() {
		if len(idxs) > 0 {
			mean := sum / float64(len(idxs))
			chron := append([]int(nil), idxs...)
			sort.Ints(chron)
			out = append(out, srCluster{mean: mean, idxs: chron})
		}
		sum, idxs = 0, nil
	}
	for _, p := range sorted {
		if len(idxs) > 0 {
			mean := sum / float64(len(idxs))
			if math.Abs(p.price-mean) > mean*tolPct/100 {
				flush()
			}
		}
		sum += p.price
		idxs = append(idxs, p.idx)
	}
	flush()
	// Same-bar guard (review fix 12): an outside bar can be a swing high AND a
	// swing low; when both points land in one cluster the bar must count as
	// ONE touch — dedupe member indices (the mean keeps both prices, which is
	// the honest cluster center for an outside bar).
	for i := range out {
		seen := make(map[int]bool, len(out[i].idxs))
		uniq := out[i].idxs[:0]
		for _, idx := range out[i].idxs {
			if seen[idx] {
				continue
			}
			seen[idx] = true
			uniq = append(uniq, idx)
		}
		out[i].idxs = uniq
	}
	return out
}

// sortSRLevels orders levels by Strength desc, then Touches desc, then Level
// desc, then Raw desc — deterministic across runs.
func sortSRLevels(out []SRLevel) {
	sort.Slice(out, func(i, j int) bool {
		if out[i].Strength != out[j].Strength {
			return out[i].Strength > out[j].Strength
		}
		if out[i].Touches != out[j].Touches {
			return out[i].Touches > out[j].Touches
		}
		if out[i].Level != out[j].Level {
			return out[i].Level > out[j].Level
		}
		return out[i].Raw > out[j].Raw // FX-scale ties (equal int levels)
	})
}

// clusterLevels is the values-only view of clusterSwingPoints kept for the
// first-order clustering tests: same merge rule, Strength defaults to the
// touch count (no volume data at this layer).
func clusterLevels(levels []float64, tolPct float64) []SRLevel {
	points := make([]swingPoint, len(levels))
	for i, v := range levels {
		points[i] = swingPoint{idx: i, price: v}
	}
	clusters := clusterSwingPoints(points, tolPct)
	out := make([]SRLevel, 0, len(clusters))
	for _, c := range clusters {
		out = append(out, SRLevel{
			Level:    int(math.Round(c.mean)),
			Raw:      c.mean,
			Touches:  len(c.idxs),
			Strength: float64(len(c.idxs)),
		})
	}
	sortSRLevels(out)
	return out
}

// medianVolume is the median of the NON-ZERO bar volumes in the window
// (review fix 11: zero-volume bars — Yahoo FX serves null → 0 — carry no
// participation information, and letting them drag the median toward zero on
// a mixed series would hand every real-volume touch a fake bonus). 0 when no
// bar carries volume — the volume features switch off cleanly.
func medianVolume(candles []types.OHLCVCandle) float64 {
	vols := make([]float64, 0, len(candles))
	for _, c := range candles {
		if c.Volume > 0 {
			vols = append(vols, c.Volume)
		}
	}
	if len(vols) == 0 {
		return 0
	}
	sort.Float64s(vols)
	n := len(vols)
	if n%2 == 1 {
		return vols[n/2]
	}
	return (vols[n/2-1] + vols[n/2]) / 2
}

// srVolumeFeaturesEnabled gates the volume-derived metrics for one level:
// they run only when the window has a real (non-zero) median AND at least
// half of THIS level's touches carry volume — otherwise the level's volume
// picture is mostly holes and any bonus/weakening claim would be built on
// absence (review fix 11).
func srVolumeFeaturesEnabled(idxs []int, candles []types.OHLCVCandle, medVol float64) bool {
	if medVol <= 0 || len(idxs) == 0 {
		return false
	}
	withVol := 0
	for _, i := range idxs {
		if candles[i].Volume > 0 {
			withVol++
		}
	}
	return withVol*2 >= len(idxs)
}

// srStrengthOf implements the B4 strength formula:
//
//	strength = touches + Σ over touches (volume_at_touch > median_volume ? 0.5 : 0)
//
// where median_volume is the window's NON-ZERO-volume median. A touch that
// arrived on above-median volume carried real participation, so it counts
// half a touch extra. With the volume features disabled (no volume data, or
// mostly volume-less touches) strength equals the plain touch count.
func srStrengthOf(idxs []int, candles []types.OHLCVCandle, medVol float64) float64 {
	strength := float64(len(idxs))
	if !srVolumeFeaturesEnabled(idxs, candles, medVol) {
		return strength
	}
	for _, i := range idxs {
		if candles[i].Volume > medVol {
			strength += 0.5
		}
	}
	return strength
}

// srWeakening implements the B4 weakening detector: an ESTABLISHED level
// (≥ srStrongTouches touches) whose last 3 touches carried a lower mean
// volume than its first 3 — the market keeps coming back but with fading
// participation. Runs only when the level's volume features are enabled;
// idxs chronological.
func srWeakening(idxs []int, candles []types.OHLCVCandle, medVol float64) bool {
	if len(idxs) < srStrongTouches || !srVolumeFeaturesEnabled(idxs, candles, medVol) {
		return false
	}
	mean3 := func(ii []int) float64 {
		var sum float64
		for _, i := range ii {
			sum += candles[i].Volume
		}
		return sum / float64(len(ii))
	}
	first := mean3(idxs[:3])
	last := mean3(idxs[len(idxs)-3:])
	return last < first
}

// breakHoldResult is one level's test history over the window. tests ==
// breaks + holds (only RESOLVED tests count); lastTestIdx is the newest bar
// whose close entered the band with a known approach — including an
// unresolved trailing entry, because the touch itself is a fact even when its
// outcome is unknown. -1 when the level was never tested.
type breakHoldResult struct {
	tests, breaks, holds int
	lastTestIdx          int
}

// breakHoldStats counts how a level behaved over the window (B4). Frequencies
// over THIS window — never probabilities.
//
// Definitions ("the band" = 0.25×ATR(14), FROZEN at the test's entry bar —
// review fix 9: a volatility spike after the touch must not reclassify the
// same move):
//   - a TEST begins when a close lands inside the band after the previous
//     known close was outside it; the approach side is that outside side.
//   - within the next 3 bars: a close beyond the level on the OPPOSITE side
//     by more than the frozen band → BREAK; a close back beyond the frozen
//     band on the APPROACH side → HOLD (a hold IS the rejection — price must
//     LEAVE the band away from the level).
//   - neither within 3 bars (price stalls inside the band) → UNRESOLVED:
//     dropped from the counts entirely — an unfinished consolidation is not a
//     rejection (review fix 1). Scanning then waits for a fresh exit before
//     the next test can begin.
//   - a test still open when the window ends is dropped the same way.
//   - warmup bars without a valid ATR are skipped; a close that gaps across
//     the whole band in one bar never lands inside it and is not a test.
func breakHoldStats(level float64, closes, atr []float64) breakHoldResult {
	n := len(closes)
	if len(atr) < n {
		n = len(atr)
	}
	r := breakHoldResult{lastTestIdx: -1}
	outsideSide := 0 // -1 below the band, +1 above, 0 unknown yet
	i := 0
	for i < n {
		if atr[i] <= 0 {
			i++
			continue
		}
		band := 0.25 * atr[i]
		d := closes[i] - level
		if math.Abs(d) > band {
			if d > 0 {
				outsideSide = 1
			} else {
				outsideSide = -1
			}
			i++
			continue
		}
		if outsideSide == 0 { // inside, but no approach known yet
			i++
			continue
		}
		// Test entry: the touch is a dated fact regardless of the outcome.
		approach := outsideSide
		r.lastTestIdx = i
		frozenBand := band // fix 9: entry bar's ATR governs the whole test
		resolved := false
		resolveAt := i
		for j := i + 1; j <= i+3 && j < n; j++ {
			dj := closes[j] - level
			if math.Abs(dj) <= frozenBand {
				continue // still inside the frozen band
			}
			r.tests++
			if (dj > 0) == (approach > 0) {
				r.holds++ // rejection: left the band away from the level
			} else {
				r.breaks++ // through the level beyond the far band edge
			}
			resolved = true
			resolveAt = j
			break
		}
		if !resolved {
			// Unresolved — consolidation inside the band, or the window ended
			// mid-test. Either way: not a hold, not a break, not a test.
			if i+3 >= n {
				return r
			}
			outsideSide = 0 // next test needs a fresh exit → entry
			i = i + 4
			continue
		}
		i = resolveAt // the resolving bar re-seeds outsideSide on its own pass
	}
	return r
}

// supportResistance clusters all swing points, enriches each level with the
// B4 metrics (volume-weighted strength, weakening, break/hold frequency,
// last-touch date), then splits them around the last close: supports below,
// resistances above, top-3 each by strength.
func supportResistance(candles []types.OHLCVCandle, wing int, tolPct float64) (supports, resistances []SRLevel) {
	if len(candles) == 0 {
		return nil, nil
	}
	highs := make([]float64, len(candles))
	lows := make([]float64, len(candles))
	closes := make([]float64, len(candles))
	for i, c := range candles {
		highs[i] = c.High
		lows[i] = c.Low
		closes[i] = c.Close
	}
	sh, sl := swingPointsIdx(highs, lows, wing)
	clusters := clusterSwingPoints(append(sh, sl...), tolPct)

	medVol := medianVolume(candles)
	atr := atrSeriesWilder(highs, lows, closes, 14)

	levels := make([]SRLevel, 0, len(clusters))
	for _, cl := range clusters {
		lvl := SRLevel{
			Level:     int(math.Round(cl.mean)),
			Raw:       cl.mean,
			Touches:   len(cl.idxs),
			Strength:  srStrengthOf(cl.idxs, candles, medVol),
			Weakening: srWeakening(cl.idxs, candles, medVol),
			LastTouch: time.Unix(candles[cl.idxs[len(cl.idxs)-1]].Time, 0).UTC(),
		}
		bh := breakHoldStats(cl.mean, closes, atr)
		lvl.Breaks, lvl.Holds = bh.breaks, bh.holds
		// last_touch = the newest of (last swing in the cluster, last close-
		// test of the level) — a recent test of an old level refreshes it
		// (review fix 6).
		if bh.lastTestIdx >= 0 {
			if testAt := time.Unix(candles[bh.lastTestIdx].Time, 0).UTC(); testAt.After(lvl.LastTouch) {
				lvl.LastTouch = testAt
			}
		}
		levels = append(levels, lvl)
	}
	sortSRLevels(levels)

	last := candles[len(candles)-1].Close
	for _, c := range levels {
		// Split on the full-precision mean: integer rounding would put every
		// sub-1.0 FX level on the "support" side of any price.
		if c.Raw < last && len(supports) < 3 {
			supports = append(supports, c)
		}
		if c.Raw > last && len(resistances) < 3 {
			resistances = append(resistances, c)
		}
	}
	return supports, resistances
}

// dayRange returns the last close's position inside the trailing-24h
// high-low range: 0 = at the low, 1 = at the high. The window is anchored
// on the last CLOSED bar's open time, so weekend session gaps mean "the
// last trading day". ok=false on an empty series or a degenerate range.
func dayRange(candles []types.OHLCVCandle) (float64, bool) {
	if len(candles) == 0 {
		return 0, false
	}
	last := candles[len(candles)-1]
	cutoff := last.Time - 86400
	hi, lo := math.Inf(-1), math.Inf(1)
	for i := len(candles) - 1; i >= 0; i-- {
		if candles[i].Time <= cutoff {
			break
		}
		if candles[i].High > hi {
			hi = candles[i].High
		}
		if candles[i].Low < lo {
			lo = candles[i].Low
		}
	}
	if !(hi > lo) {
		return 0, false
	}
	pos := (last.Close - lo) / (hi - lo)
	return math.Max(0, math.Min(1, pos)), true
}

// macdLast returns MACD(12,26,9) line, signal and histogram for the last
// bar; ok=false under 35 bars (26 for the slow EMA + 9 for the signal).
func macdLast(closes []float64) (line, signal, hist float64, ok bool) {
	if len(closes) < 35 {
		return 0, 0, 0, false
	}
	e12 := emaSeries(closes, 12)
	e26 := emaSeries(closes, 26)
	macdVals := make([]float64, len(closes))
	for i := range closes {
		macdVals[i] = e12[i] - e26[i]
	}
	sig := emaSeries(macdVals, 9)
	last := len(closes) - 1
	return macdVals[last], sig[last], macdVals[last] - sig[last], true
}
