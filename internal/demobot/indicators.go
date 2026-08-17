// Package demobot is the thin Telegram presentation layer over the AlphaVizor
// backend REST API. Indicator math in this file is intentionally small and
// self-contained: the ml package keeps its implementations unexported, so the
// bot carries its own copies (same Wilder conventions as internal/ml
// features_v2.go) with first-order tests in indicators_test.go.
package demobot

import (
	"math"
	"sort"

	"github.com/prostwp/elibri-backend/pkg/types"
)

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

// swingLevels finds swing highs/lows: bar i is a swing high when high[i] is
// strictly greater than the highs of `wing` bars on each side (mirror for
// lows). Bars too close to either end never qualify.
func swingLevels(high, low []float64, wing int) (swingHighs, swingLows []float64) {
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
			swingHighs = append(swingHighs, high[i])
		}
		if isLow {
			swingLows = append(swingLows, low[i])
		}
	}
	return swingHighs, swingLows
}

// SRLevel is one clustered support/resistance level. Level is rounded to an
// integer per the original BTC-scale card contract; Raw keeps the cluster
// mean at full precision for FX-scale prices (EURUSD at 1.1583 must not
// collapse to "1"). Touches is the number of swing points that fell into
// the cluster (its "strength").
type SRLevel struct {
	Level   int
	Raw     float64
	Touches int
}

// clusterLevels groups nearby swing levels: values within tolPct percent of
// the running cluster mean merge into one level. Result is sorted by Touches
// descending, then by Level descending for determinism.
func clusterLevels(levels []float64, tolPct float64) []SRLevel {
	if len(levels) == 0 {
		return nil
	}
	sorted := append([]float64(nil), levels...)
	sort.Float64s(sorted)

	var out []SRLevel
	var sum float64
	var count int
	flush := func() {
		if count > 0 {
			mean := sum / float64(count)
			out = append(out, SRLevel{Level: int(math.Round(mean)), Raw: mean, Touches: count})
		}
		sum, count = 0, 0
	}
	for _, v := range sorted {
		if count > 0 {
			mean := sum / float64(count)
			if math.Abs(v-mean) > mean*tolPct/100 {
				flush()
			}
		}
		sum += v
		count++
	}
	flush()

	sort.Slice(out, func(i, j int) bool {
		if out[i].Touches != out[j].Touches {
			return out[i].Touches > out[j].Touches
		}
		if out[i].Level != out[j].Level {
			return out[i].Level > out[j].Level
		}
		return out[i].Raw > out[j].Raw // FX-scale ties (equal int levels)
	})
	return out
}

// supportResistance clusters all swing points, then splits them around the
// last close: supports below, resistances above, top-3 each by strength.
func supportResistance(candles []types.OHLCVCandle, wing int, tolPct float64) (supports, resistances []SRLevel) {
	if len(candles) == 0 {
		return nil, nil
	}
	highs := make([]float64, len(candles))
	lows := make([]float64, len(candles))
	for i, c := range candles {
		highs[i] = c.High
		lows[i] = c.Low
	}
	sh, sl := swingLevels(highs, lows, wing)
	clusters := clusterLevels(append(sh, sl...), tolPct)
	last := candles[len(candles)-1].Close
	for _, c := range clusters {
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
