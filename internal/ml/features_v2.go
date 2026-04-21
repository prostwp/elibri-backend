package ml

import (
	"math"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// ─── Canonical feature order ─────────────────────────────────
// CRITICAL: must match FEATURE_NAMES in ml-training/feature_engine.py exactly.
// Any change requires version bump in both sides + retrain.
var FeatureNamesV2 = []string{
	"rsi_7", "rsi_14", "rsi_21",
	"macd_hist", "macd_signal", "bb_position", "stoch_k_14",
	"ema_cross_20_50", "ema_cross_50_200", "adx_14",
	"price_vs_ema_50", "price_vs_ema_200",
	"atr_norm_14", "bb_width", "vol_regime",
	"vol_ratio_5", "vol_ratio_20", "taker_buy_ratio",
	"return_1", "return_5", "return_20",
	"higher_highs_10", "lower_lows_10",
	"doji_last", "engulfing_last", "hammer_last",
	"btc_corr_30", "btc_beta_30",
	"rsi_14_lag_4", "return_5_lag_4", "vol_ratio_20_lag_4",
}

// ExtractFeaturesV2 computes 31 features matching Python feature_engine.
// TakerBuyVolume is optional — pass 0 if unavailable (Binance public API gives it).
// BtcCloses: aligned BTC close series for cross-asset features (pass nil for BTC itself).
func ExtractFeaturesV2(candles []types.OHLCVCandle, takerBuyVolumes []float64, btcCloses []float64) []float64 {
	n := len(candles)
	if n < 30 {
		return make([]float64, len(FeatureNamesV2))
	}

	closes := make([]float64, n)
	highs := make([]float64, n)
	lows := make([]float64, n)
	opens := make([]float64, n)
	vols := make([]float64, n)
	for i, c := range candles {
		closes[i] = c.Close
		highs[i] = c.High
		lows[i] = c.Low
		opens[i] = c.Open
		vols[i] = c.Volume
	}

	last := n - 1
	close := closes[last]

	rsi7 := wilderRSI(closes, 7)[last]
	rsi14Series := wilderRSI(closes, 14)
	rsi14 := rsi14Series[last]
	rsi21 := wilderRSI(closes, 21)[last]

	macdLine, macdSig, _ := macd(closes)
	macdHist := macdLine - macdSig
	macdHistNorm := math.Tanh(macdHist / (close + 1e-12) * 100)
	macdSigNorm := math.Tanh(macdSig / (close + 1e-12) * 100)

	_, bbLower, bbMid, bbWidth, bbPos := bollinger(closes, 20, 2.0)

	stochK := stoch(highs, lows, closes, 14) / 100.0

	ema20 := emaSeries(closes, 20)[last]
	ema50Series := emaSeries(closes, 50)
	ema50 := ema50Series[last]
	ema200Series := emaSeries(closes, 200)
	ema200 := ema200Series[last]
	emaCross2050 := signFloat(ema20 - ema50)
	emaCross50200 := signFloat(ema50 - ema200)

	adx14 := adx(highs, lows, closes, 14) / 100.0

	priceVsEma50 := (close - ema50) / (ema50 + 1e-12)
	priceVsEma200 := (close - ema200) / (ema200 + 1e-12)

	atr14Series := wilderATR(highs, lows, closes, 14)
	atr14 := atr14Series[last]
	atrNorm := atr14 / (close + 1e-12)

	volRegime := rollingPercentile(atr14Series, closes, 100, last)

	vr5 := vols[last] / (meanOfLast(vols, 5) + 1e-12)
	vr20 := vols[last] / (meanOfLast(vols, 20) + 1e-12)
	vr5 = clampV2(vr5, 0, 20)
	vr20 = clampV2(vr20, 0, 20)

	takerBuy := 0.0
	if len(takerBuyVolumes) == n {
		takerBuy = takerBuyVolumes[last] / (vols[last] + 1e-12)
	}

	ret1 := (closes[last] - closes[last-1]) / (closes[last-1] + 1e-12)
	ret5 := (closes[last] - closes[last-5]) / (closes[last-5] + 1e-12)
	ret20 := 0.0
	if last >= 20 {
		ret20 = (closes[last] - closes[last-20]) / (closes[last-20] + 1e-12)
	}

	// Higher highs / lower lows in last 10 bars.
	hh := 0.0
	ll := 0.0
	if last >= 9 {
		maxH := highs[last-9]
		minL := lows[last-9]
		for i := last - 8; i <= last; i++ {
			if highs[i] > maxH {
				maxH = highs[i]
			}
			if lows[i] < minL {
				minL = lows[i]
			}
		}
		if highs[last] >= maxH {
			hh = 1
		}
		if lows[last] <= minL {
			ll = 1
		}
	}

	// Candlestick patterns on last bar.
	doji := 0.0
	body := math.Abs(closes[last] - opens[last])
	full := highs[last] - lows[last] + 1e-12
	if body/full < 0.1 {
		doji = 1
	}

	engulfing := 0.0
	prevBody := closes[last-1] - opens[last-1]
	curBody := closes[last] - opens[last]
	if prevBody < 0 && curBody > 0 && math.Abs(curBody) > math.Abs(prevBody) {
		engulfing = 1
	} else if prevBody > 0 && curBody < 0 && math.Abs(curBody) > math.Abs(prevBody) {
		engulfing = -1
	}

	hammer := 0.0
	lowerWick := math.Min(opens[last], closes[last]) - lows[last]
	upperWick := highs[last] - math.Max(opens[last], closes[last])
	if lowerWick > 2*body && upperWick < body {
		hammer = 1
	}

	// Cross-asset: rolling 30-bar beta + correlation vs BTC.
	// Default 0.0 (uncorrelated) when BTC context unavailable — signals to
	// the model that cross-asset feature is missing. Previous default 1.0
	// implied "perfect correlation", which could bias BTC-era patterns onto
	// non-BTC predictions.
	btcCorr, btcBeta := 0.0, 0.0
	if len(btcCloses) >= n && last >= 30 {
		corr, beta := rollingCorrBeta(closes, btcCloses, last, 30)
		btcCorr = corr
		btcBeta = clampV2(beta, -3, 3)
	}

	// Lagged features (4 bars back).
	lagIdx := last - 4
	if lagIdx < 0 {
		lagIdx = 0
	}
	rsi14Lag4 := rsi14Series[lagIdx]

	ret5Lag4 := 0.0
	if lagIdx-5 >= 0 {
		ret5Lag4 = (closes[lagIdx] - closes[lagIdx-5]) / (closes[lagIdx-5] + 1e-12)
	}

	// Patch 2N+2 parity fix: vol_ratio_20_lag_4 window was [lagIdx-20, lagIdx)
	// (20 items, EXCLUDING lagIdx) but Python's rolling(20).mean() at lagIdx
	// spans [lagIdx-19, lagIdx] (20 items INCLUDING lagIdx). Off-by-one gave
	// a ~1% drift on 60-bar fixtures. Now both sides compute the mean on the
	// same 20 bars inclusive of the current lag position.
	vr20Lag4 := 1.0
	if lagIdx >= 19 {
		vr20Lag4 = vols[lagIdx] / (meanOfRange(vols, lagIdx-19, lagIdx+1) + 1e-12)
		vr20Lag4 = clampV2(vr20Lag4, 0, 20)
	}

	// bbMid/bbLower unused in final vec (reserved for future features)
	_ = bbMid
	_ = bbLower

	return []float64{
		rsi7, rsi14, rsi21,
		macdHistNorm, macdSigNorm, bbPos, stochK,
		emaCross2050, emaCross50200, adx14,
		priceVsEma50, priceVsEma200,
		atrNorm, bbWidth, volRegime,
		vr5, vr20, takerBuy,
		ret1, ret5, ret20,
		hh, ll,
		doji, engulfing, hammer,
		btcCorr, btcBeta,
		rsi14Lag4, ret5Lag4, vr20Lag4,
	}
}

// ─── Primitives (identical formulas to feature_engine.py) ────

func wilderRSI(close []float64, period int) []float64 {
	n := len(close)
	out := make([]float64, n)
	for i := range out {
		out[i] = 50
	}
	if n < period+1 {
		return out
	}
	var gains, losses float64
	for i := 1; i <= period; i++ {
		d := close[i] - close[i-1]
		if d > 0 {
			gains += d
		} else {
			losses -= d
		}
	}
	avgGain := gains / float64(period)
	avgLoss := losses / float64(period)
	rs := avgGain / (avgLoss + 1e-12)
	out[period] = 100 - 100/(1+rs)
	for i := period + 1; i < n; i++ {
		d := close[i] - close[i-1]
		var g, l float64
		if d > 0 {
			g = d
		} else {
			l = -d
		}
		avgGain = (avgGain*(float64(period)-1) + g) / float64(period)
		avgLoss = (avgLoss*(float64(period)-1) + l) / float64(period)
		rs = avgGain / (avgLoss + 1e-12)
		out[i] = 100 - 100/(1+rs)
	}
	return out
}

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

func macd(close []float64) (macdLine, signal, hist float64) {
	e12 := emaSeries(close, 12)
	e26 := emaSeries(close, 26)
	macdVals := make([]float64, len(close))
	for i := range close {
		macdVals[i] = e12[i] - e26[i]
	}
	sigSeries := emaSeries(macdVals, 9)
	last := len(close) - 1
	return macdVals[last], sigSeries[last], macdVals[last] - sigSeries[last]
}

func bollinger(close []float64, period int, numStd float64) (upper, lower, mid, width, pos float64) {
	n := len(close)
	if n < 1 {
		return 0, 0, 0, 0, 0.5
	}
	start := n - period
	if start < 0 {
		start = 0
	}
	var sum, sumSq float64
	count := 0
	for i := start; i < n; i++ {
		sum += close[i]
		sumSq += close[i] * close[i]
		count++
	}
	if count == 0 {
		return 0, 0, 0, 0, 0.5
	}
	mid = sum / float64(count)
	variance := sumSq/float64(count) - mid*mid
	if variance < 0 {
		variance = 0
	}
	sd := math.Sqrt(variance)
	upper = mid + numStd*sd
	lower = mid - numStd*sd
	width = (upper - lower) / (mid + 1e-12)
	last := close[n-1]
	pos = (last - lower) / (upper - lower + 1e-12)
	if pos < 0 {
		pos = 0
	}
	if pos > 1 {
		pos = 1
	}
	return
}

func stoch(high, low, close []float64, period int) float64 {
	n := len(close)
	start := n - period
	if start < 0 {
		start = 0
	}
	h := high[start]
	l := low[start]
	for i := start + 1; i < n; i++ {
		if high[i] > h {
			h = high[i]
		}
		if low[i] < l {
			l = low[i]
		}
	}
	if h == l {
		return 50
	}
	return (close[n-1] - l) / (h - l) * 100
}

func adx(high, low, close []float64, period int) float64 {
	n := len(close)
	if n < period+1 {
		return 0
	}
	plusDM := make([]float64, n)
	minusDM := make([]float64, n)
	tr := make([]float64, n)
	for i := 1; i < n; i++ {
		up := high[i] - high[i-1]
		dn := low[i-1] - low[i]
		if up > dn && up > 0 {
			plusDM[i] = up
		}
		if dn > up && dn > 0 {
			minusDM[i] = dn
		}
		trv := high[i] - low[i]
		if v := math.Abs(high[i] - close[i-1]); v > trv {
			trv = v
		}
		if v := math.Abs(low[i] - close[i-1]); v > trv {
			trv = v
		}
		tr[i] = trv
	}
	// rolling sum approximations
	sumN := func(a []float64, to int) float64 {
		var s float64
		start := to - period + 1
		if start < 0 {
			start = 0
		}
		for i := start; i <= to; i++ {
			s += a[i]
		}
		return s
	}
	plusDISeries := make([]float64, n)
	minusDISeries := make([]float64, n)
	for i := 0; i < n; i++ {
		trSum := sumN(tr, i) + 1e-12
		plusDISeries[i] = 100 * sumN(plusDM, i) / trSum
		minusDISeries[i] = 100 * sumN(minusDM, i) / trSum
	}
	dx := make([]float64, n)
	for i := 0; i < n; i++ {
		d := math.Abs(plusDISeries[i]-minusDISeries[i]) / (plusDISeries[i] + minusDISeries[i] + 1e-12)
		dx[i] = 100 * d
	}
	// ADX = rolling mean of DX over period.
	start := n - period
	if start < 0 {
		start = 0
	}
	var sum float64
	c := 0
	for i := start; i < n; i++ {
		sum += dx[i]
		c++
	}
	if c == 0 {
		return 0
	}
	return sum / float64(c)
}

func wilderATR(high, low, close []float64, period int) []float64 {
	n := len(close)
	out := make([]float64, n)
	if n < period+1 {
		return out
	}
	var sum float64
	for i := 1; i <= period; i++ {
		trv := high[i] - low[i]
		if v := math.Abs(high[i] - close[i-1]); v > trv {
			trv = v
		}
		if v := math.Abs(low[i] - close[i-1]); v > trv {
			trv = v
		}
		sum += trv
	}
	out[period] = sum / float64(period)
	for i := period + 1; i < n; i++ {
		trv := high[i] - low[i]
		if v := math.Abs(high[i] - close[i-1]); v > trv {
			trv = v
		}
		if v := math.Abs(low[i] - close[i-1]); v > trv {
			trv = v
		}
		out[i] = (out[i-1]*(float64(period)-1) + trv) / float64(period)
	}
	return out
}

func rollingPercentile(atrSeries, closes []float64, window, idx int) float64 {
	if idx < 20 {
		return 0.5
	}
	atrNorm := make([]float64, idx+1)
	for i := 0; i <= idx; i++ {
		atrNorm[i] = atrSeries[i] / (closes[i] + 1e-12)
	}
	start := idx - window + 1
	if start < 0 {
		start = 0
	}
	cur := atrNorm[idx]
	cnt, less := 0, 0
	for i := start; i <= idx; i++ {
		cnt++
		if atrNorm[i] < cur {
			less++
		}
	}
	if cnt == 0 {
		return 0.5
	}
	return float64(less) / float64(cnt)
}

func meanOfLast(values []float64, n int) float64 {
	if len(values) == 0 {
		return 0
	}
	start := len(values) - n
	if start < 0 {
		start = 0
	}
	var s float64
	c := 0
	for i := start; i < len(values); i++ {
		s += values[i]
		c++
	}
	if c == 0 {
		return 0
	}
	return s / float64(c)
}

func meanOfRange(values []float64, start, end int) float64 {
	if start < 0 {
		start = 0
	}
	if end > len(values) {
		end = len(values)
	}
	var s float64
	c := 0
	for i := start; i < end; i++ {
		s += values[i]
		c++
	}
	if c == 0 {
		return 0
	}
	return s / float64(c)
}

func rollingCorrBeta(a, b []float64, idx, window int) (corr, beta float64) {
	start := idx - window
	if start < 1 {
		start = 1
	}
	var sumA, sumB, sumAB, sumA2, sumB2 float64
	n := 0
	for i := start; i <= idx; i++ {
		rA := (a[i] - a[i-1]) / (a[i-1] + 1e-12)
		rB := (b[i] - b[i-1]) / (b[i-1] + 1e-12)
		sumA += rA
		sumB += rB
		sumAB += rA * rB
		sumA2 += rA * rA
		sumB2 += rB * rB
		n++
	}
	if n < 2 {
		return 0, 0
	}
	fn := float64(n)
	meanA := sumA / fn
	meanB := sumB / fn
	covAB := sumAB/fn - meanA*meanB
	varB := sumB2/fn - meanB*meanB
	varA := sumA2/fn - meanA*meanA
	if varB < 1e-12 || varA < 1e-12 {
		return 0, 0
	}
	corr = covAB / math.Sqrt(varA*varB)
	beta = covAB / varB
	return
}

func signFloat(v float64) float64 {
	if v > 0 {
		return 1
	}
	if v < 0 {
		return -1
	}
	return 0
}

func clampV2(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
