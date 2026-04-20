package ml

import (
	"fmt"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// TradingStyleHorizon maps trading-style preset to bars ahead.
// Frontend TradingStyleNode values: "scalp", "day", "swing", "position".
// Aligned with Python HORIZON_MAP per interval. Frontend picks interval
// separately; these horizon overrides only apply when different from the
// interval's default training horizon.
var TradingStyleHorizon = map[string]int{
	"scalp":    12,  // 5m × 12 = 1 hour ahead
	"day":      16,  // 15m × 16 = 4 hours ahead
	"swing":    18,  // 4h × 18 = 3 days ahead
	"position": 10,  // 1d × 10 = 10 days ahead
}

// PredictionV2 is the richer response for /api/v1/ml/predict.
type PredictionV2 struct {
	Symbol            string              `json:"symbol"`
	Interval          string              `json:"interval"`
	Direction         string              `json:"direction"`           // buy/sell/neutral
	Confidence        float64             `json:"confidence"`          // 0-100
	Probability       float64             `json:"probability"`         // 0-1 (raw P(up))
	PriceTarget       float64             `json:"price_target"`
	Timeframe         string              `json:"timeframe"`
	HorizonBars       int                 `json:"horizon_bars"`
	ModelVersion      string              `json:"model_version"`
	PredictedAt       int64               `json:"predicted_at"`
	FeatureImportance []FeatureImp        `json:"feature_importance"` // top-10
	SimilarSituations []SimilarSituation  `json:"similar_situations"` // top-5
	Metrics           SimplePredictMetrics `json:"metrics"`
	Features          map[string]float64  `json:"features,omitempty"` // for debug
	FallbackReason    string              `json:"fallback_reason,omitempty"`
	// VolGate is set by risk-tier post-filter (Patch 2C). "" = passed;
	// "blocked_low_vol" = atr_norm_14 below tier floor. When set,
	// Direction is downgraded to "neutral".
	VolGate string `json:"vol_gate,omitempty"`
	// RiskTier echoes the tier applied to this prediction (for UI badges).
	RiskTier string `json:"risk_tier,omitempty"`
}

type SimplePredictMetrics struct {
	Accuracy       float64 `json:"accuracy"`
	Sharpe         float64 `json:"sharpe"`
	F1             float64 `json:"f1"`
	NFolds         int     `json:"n_folds"`
	HCPrecision    float64 `json:"hc_precision"`    // precision when model is >80% confident
	HCSignalRate   float64 `json:"hc_signal_rate"`  // fraction of bars that pass filter
	HCSignalsTotal int     `json:"hc_signals_total"`
	NTestTotal     int     `json:"n_test_total"`
	AvgOutcome5    float64 `json:"avg_outcome_5"`
	AvgOutcome10   float64 `json:"avg_outcome_10"`
	AvgOutcome20   float64 `json:"avg_outcome_20"`
	// HighConfidence flag: true if THIS prediction passes the 0.80/0.20 filter.
	HighConfidence bool `json:"high_confidence"`
}

// PredictV2 runs the ensemble + pattern matcher for a symbol/interval
// with the default Balanced risk tier. Kept as a thin wrapper so callers
// that don't care about tiers stay unchanged.
func PredictV2(
	symbol, interval, tradingStyle string,
	candles []types.OHLCVCandle,
	takerBuyVolumes, btcCloses []float64,
) PredictionV2 {
	return PredictV2WithTier(symbol, interval, tradingStyle, string(TierBalanced), candles, takerBuyVolumes, btcCloses)
}

// PredictV2WithTier runs the ensemble + pattern matcher and applies a
// risk-tier vol gate on top. tierName: "conservative"|"balanced"|"aggressive"
// (falls back to balanced on unknown). The gate never overrides probability
// or confidence values — it only downgrades Direction to "neutral" and
// marks VolGate, so the UI can still display raw ML output for transparency.
func PredictV2WithTier(
	symbol, interval, tradingStyle, tierName string,
	candles []types.OHLCVCandle,
	takerBuyVolumes, btcCloses []float64,
) PredictionV2 {
	// Horizon resolution priority:
	//  1. Loaded model's own trained horizon (authoritative — HC precision
	//     was measured against THIS horizon, so using a different one makes
	//     the reported metrics meaningless).
	//  2. Map from trading_style preset.
	//  3. Swing default.
	horizon := 0
	if m, ok := GetModelV2(symbol, interval); ok && m.Horizon > 0 {
		horizon = m.Horizon
	}
	if horizon == 0 {
		horizon = TradingStyleHorizon[tradingStyle]
	}
	if horizon == 0 {
		horizon = TradingStyleHorizon["swing"]
	}

	out := PredictionV2{
		Symbol:      symbol,
		Interval:    interval,
		PredictedAt: time.Now().Unix(),
		HorizonBars: horizon,
		Timeframe:   fmt.Sprintf("%d bars (%s)", horizon, tradingStyle),
	}
	if len(candles) < 30 {
		out.Direction = "neutral"
		out.Confidence = 0
		out.FallbackReason = "not enough candles (need 30+)"
		return out
	}

	lastPrice := candles[len(candles)-1].Close

	// Try V2 ensemble first.
	model, hasModel := GetModelV2(symbol, interval)
	feats := ExtractFeaturesV2(candles, takerBuyVolumes, btcCloses)

	var prob float64
	if hasModel {
		p, _ := model.Predict(feats)
		prob = p
		out.ModelVersion = fmt.Sprintf("ensemble_v2_%s_%s", symbol, interval)
	} else {
		// Fallback: use legacy V1 model on 6-feature subset.
		legacyFeat := Features{
			RSINorm:      feats[1] / 100.0,   // rsi_14
			MACDNorm:     (feats[3] + 1) / 2, // tanh-normalized already
			VolRatio:     feats[16] / 5,      // vol_ratio_20 clipped
			ATRNorm:      feats[12] * 20,     // atr_norm
			MomentumNorm: (feats[19] + 1) / 2, // return_5
			BBPosition:   feats[5],            // bb_position
		}
		// clamp 0-1
		legacyFeat.RSINorm = clampV2(legacyFeat.RSINorm, 0, 1)
		legacyFeat.MACDNorm = clampV2(legacyFeat.MACDNorm, 0, 1)
		legacyFeat.VolRatio = clampV2(legacyFeat.VolRatio, 0, 1)
		legacyFeat.ATRNorm = clampV2(legacyFeat.ATRNorm, 0, 1)
		legacyFeat.MomentumNorm = clampV2(legacyFeat.MomentumNorm, 0, 1)

		p := defaultModel().Predict(legacyFeat)
		prob = p
		out.ModelVersion = "fallback_v1"
		out.FallbackReason = "no v2 model for " + symbol + "/" + interval
	}

	out.Probability = prob

	// Per-model adaptive threshold (from analyze_thresholds.py).
	thr := GetThreshold(symbol, interval)

	// Direction gated by the SAME adaptive threshold as HC precision.
	// Previously direction used a fixed 0.55/0.45 band which was looser
	// than thr.ThresholdHigh — users saw direction=buy + high_confidence=false
	// simultaneously, confusing. Now direction is 'buy' only when prob
	// crosses the level where HC precision was measured.
	// Fallback band (0.55/0.45) applied when no per-model threshold loaded.
	buyThr := thr.ThresholdHigh
	sellThr := thr.ThresholdLow
	if buyThr <= 0.5 || sellThr >= 0.5 {
		buyThr, sellThr = 0.55, 0.45 // sane default
	}

	switch {
	case prob > buyThr:
		out.Direction = "buy"
		out.Confidence = prob * 100
	case prob < sellThr:
		out.Direction = "sell"
		out.Confidence = (1 - prob) * 100
	default:
		out.Direction = "neutral"
		out.Confidence = 50
	}

	// Risk-tier vol gate (Patch 2C). Runs AFTER direction so the caller can
	// still see the raw ML verdict in logs/debug via `probability`. When
	// realized vol (atr_norm_14 = feats[12]) is below the tier's per-TF
	// floor, we flip direction to neutral and expose the reason.
	tier := GetTier(tierName)
	out.RiskTier = tierName
	if out.RiskTier == "" {
		out.RiskTier = string(TierBalanced)
	}
	if len(feats) > 12 && out.Direction != "neutral" {
		if floor, ok := tier.MinVolPctByTF[interval]; ok && feats[12] < floor {
			out.Direction = "neutral"
			out.VolGate = "blocked_low_vol"
		}
	}

	// Price target: ATR-based.
	atr := calcATR(candles, 14)
	switch out.Direction {
	case "buy":
		out.PriceTarget = lastPrice + atr*2.5
	case "sell":
		out.PriceTarget = lastPrice - atr*2.5
	default:
		out.PriceTarget = lastPrice
	}

	// Feature importance + metrics.
	if hasModel {
		out.FeatureImportance = model.TopFeatures(10)

		// Prefer threshold-derived HC metrics (from analyze_thresholds.py) over
		// the training-time values (which used static 0.80 filter and may be 0).
		hcPrecision := model.Metrics.HCPrecision
		hcSignalRate := model.Metrics.HCSignalRate
		hcSignalsTotal := model.Metrics.HCSignalsTotal
		if thr.Precision > 0 {
			hcPrecision = thr.Precision
			hcSignalRate = thr.Fraction
			hcSignalsTotal = thr.NSignals
		}

		out.Metrics = SimplePredictMetrics{
			Accuracy:       model.Metrics.AvgAccuracy,
			Sharpe:         model.Metrics.AvgSharpe,
			F1:             model.Metrics.AvgF1,
			NFolds:         model.Metrics.NFolds,
			HCPrecision:    hcPrecision,
			HCSignalRate:   hcSignalRate,
			HCSignalsTotal: hcSignalsTotal,
			NTestTotal:     model.Metrics.NTestTotal,
			HighConfidence: prob > thr.ThresholdHigh || prob < thr.ThresholdLow,
		}
	}

	// Similar situations.
	if pat, hasPat := GetPatternsV2(symbol, interval); hasPat {
		sims := pat.Query(feats, 5)
		out.SimilarSituations = sims
		a5, a10, a20 := AggregateOutcome(sims)
		out.Metrics.AvgOutcome5 = a5
		out.Metrics.AvgOutcome10 = a10
		out.Metrics.AvgOutcome20 = a20
	}

	// Feature debug map (small, for UI transparency).
	out.Features = make(map[string]float64, len(FeatureNamesV2))
	for i, name := range FeatureNamesV2 {
		if i < len(feats) {
			out.Features[name] = feats[i]
		}
	}

	return out
}
