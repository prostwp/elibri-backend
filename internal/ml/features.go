package ml

import (
	"math"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// Features — 6 normalized features matching frontend mlPredictor.ts
type Features struct {
	RSINorm      float64 `json:"rsi_norm"`
	MACDNorm     float64 `json:"macd_norm"`
	VolRatio     float64 `json:"vol_ratio"`
	ATRNorm      float64 `json:"atr_norm"`
	MomentumNorm float64 `json:"momentum_norm"`
	BBPosition   float64 `json:"bb_position"`
}

// ExtractFeatures computes 6 ML features from candle data
func ExtractFeatures(candles []types.OHLCVCandle) Features {
	if len(candles) < 30 {
		return Features{0.5, 0.5, 0.5, 0.5, 0.5, 0.5}
	}

	closes := make([]float64, len(candles))
	volumes := make([]float64, len(candles))
	for i, c := range candles {
		closes[i] = c.Close
		volumes[i] = c.Volume
	}

	rsi := calcRSI(closes, 14)
	macdHist := calcMACDHistogram(closes)
	volRatio := calcVolumeRatio(volumes, 20)
	atr := calcATR(candles, 14)
	momentum := calcMomentum(closes, 5)
	bbPos := calcBBPosition(closes, 20)

	lastPrice := closes[len(closes)-1]

	return Features{
		RSINorm:      clamp(rsi/100, 0, 1),
		MACDNorm:     clamp((math.Tanh(macdHist*100)+1)/2, 0, 1),
		VolRatio:     clamp(volRatio/5, 0, 1),
		ATRNorm:      clamp(atr/lastPrice*20, 0, 1),
		MomentumNorm: clamp((math.Tanh(momentum*10)+1)/2, 0, 1),
		BBPosition:   clamp(bbPos, 0, 1),
	}
}

func calcRSI(closes []float64, period int) float64 {
	if len(closes) < period+1 {
		return 50
	}
	recent := closes[len(closes)-period-1:]
	var gains, losses float64
	for i := 1; i < len(recent); i++ {
		diff := recent[i] - recent[i-1]
		if diff > 0 {
			gains += diff
		} else {
			losses -= diff
		}
	}
	avgGain := gains / float64(period)
	avgLoss := losses / float64(period)
	if avgLoss == 0 {
		return 100
	}
	rs := avgGain / avgLoss
	return 100 - 100/(1+rs)
}

func calcEMA(values []float64, period int) float64 {
	if len(values) == 0 {
		return 0
	}
	k := 2.0 / float64(period+1)
	ema := values[0]
	for i := 1; i < len(values); i++ {
		ema = values[i]*k + ema*(1-k)
	}
	return ema
}

func calcMACDHistogram(closes []float64) float64 {
	ema12 := calcEMA(closes, 12)
	ema26 := calcEMA(closes, 26)
	macdLine := ema12 - ema26

	// Signal line from MACD values
	if len(closes) < 35 {
		return macdLine
	}
	macdVals := make([]float64, 0)
	for i := 26; i < len(closes); i++ {
		e12 := calcEMA(closes[:i+1], 12)
		e26 := calcEMA(closes[:i+1], 26)
		macdVals = append(macdVals, e12-e26)
	}
	signalLine := calcEMA(macdVals, 9)
	return macdLine - signalLine
}

func calcVolumeRatio(volumes []float64, period int) float64 {
	if len(volumes) < period+1 {
		return 1
	}
	lastVol := volumes[len(volumes)-1]
	avgVol := 0.0
	for _, v := range volumes[len(volumes)-period-1 : len(volumes)-1] {
		avgVol += v
	}
	avgVol /= float64(period)
	if avgVol == 0 {
		return 1
	}
	return lastVol / avgVol
}

func calcATR(candles []types.OHLCVCandle, period int) float64 {
	if len(candles) < period+1 {
		return 0
	}
	var sum float64
	for i := len(candles) - period; i < len(candles); i++ {
		h := candles[i].High
		l := candles[i].Low
		pc := candles[i-1].Close
		tr := math.Max(h-l, math.Max(math.Abs(h-pc), math.Abs(l-pc)))
		sum += tr
	}
	return sum / float64(period)
}

func calcMomentum(closes []float64, period int) float64 {
	if len(closes) < period+1 {
		return 0
	}
	last := closes[len(closes)-1]
	prev := closes[len(closes)-1-period]
	if prev == 0 {
		return 0
	}
	return (last - prev) / prev
}

func calcBBPosition(closes []float64, period int) float64 {
	if len(closes) < period {
		return 0.5
	}
	recent := closes[len(closes)-period:]
	var sum float64
	for _, v := range recent {
		sum += v
	}
	sma := sum / float64(period)

	var variance float64
	for _, v := range recent {
		variance += (v - sma) * (v - sma)
	}
	stddev := math.Sqrt(variance / float64(period))

	upper := sma + 2*stddev
	lower := sma - 2*stddev
	last := closes[len(closes)-1]

	if upper == lower {
		return 0.5
	}
	return (last - lower) / (upper - lower)
}

func clamp(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
