package scenario

import (
	"context"
	"fmt"

	"github.com/prostwp/elibri-backend/internal/market"
	"github.com/prostwp/elibri-backend/internal/ml"
	"github.com/prostwp/elibri-backend/pkg/types"
)

// Evaluation is the outcome of a single scenario tick.
// Ready=true means the runner should emit an alert; everything else is
// either "no signal" (neutral/hold) or "blocked by a gate" (keep the
// scenario alive but don't alert).
type Evaluation struct {
	Ready            bool
	Direction        string  // "buy" | "sell" (only when Ready)
	Confidence       float64 // 0-100
	EntryPrice       float64
	StopLoss         float64
	TakeProfit       float64
	PositionSizeUSD  float64
	Label            string // trend_aligned | mean_reversion | random
	LabelReason      string
	BarTime          int64  // last candle open_time (unix seconds)
	HCPass           bool   // model high-confidence flag
	BlockReason      string // "" | "low_vol" | "label_not_allowed" | "same_bar" | "ml_down"
	ModelVersion     string
}

// Evaluate runs one tick: pulls candles, runs PredictV2WithTier, applies
// tier policy (HC gate is already inside PredictV2; we additionally check
// label permission + dedup). Pure function — no DB writes, no side effects.
//
// equityUSD is the notional base for position sizing. If 0, sizing is
// skipped (PositionSizeUSD=0) — caller decides whether to still alert.
func Evaluate(
	ctx context.Context,
	s *ActiveScenario,
	equityUSD float64,
) Evaluation {
	ev := Evaluation{}

	// Kill-switch: if no models loaded, don't tick (saves Binance calls).
	if ml.V2Health().NModels == 0 {
		ev.BlockReason = "ml_down"
		return ev
	}

	// Fetch fresh candles. Need ~300 for feature window + HC.
	// Uses 30s-TTL cache so N scenarios on same (symbol, interval) share
	// a single Binance REST call per 30s window (PHASE 1 fix for rate limits).
	candles, err := market.FetchCryptoCandlesCached(s.Symbol, s.Interval, 300)
	if err != nil || len(candles) < 30 {
		ev.BlockReason = fmt.Sprintf("candles: %v", err)
		return ev
	}
	last := candles[len(candles)-1]
	ev.BarTime = last.Time

	// Dedup: same bar + same direction already alerted → skip.
	// (Direction comes from prediction below; we return early only after it.)

	// BTC cross-asset context for non-BTC symbols.
	var btcCloses []float64
	if s.Symbol != "BTCUSDT" {
		btcCandles, err := market.FetchCryptoCandlesCached("BTCUSDT", s.Interval, 300)
		if err == nil {
			btcCloses = closesOf(btcCandles)
		}
	}

	// Taker-buy volumes (quoteVolume proxy). Binance REST already fills this
	// as .Volume; feature extractor handles nil slice gracefully.
	var takerBuyVols []float64

	pred := ml.PredictV2WithTier(
		s.Symbol, s.Interval, "swing", s.RiskTier,
		candles, takerBuyVols, btcCloses,
	)
	ev.Confidence = pred.Confidence
	ev.HCPass = pred.Metrics.HighConfidence
	ev.ModelVersion = pred.ModelVersion

	if pred.VolGate == "blocked_low_vol" {
		ev.BlockReason = "low_vol"
		return ev
	}
	if pred.Direction == "neutral" {
		return ev // not an error, just no signal this bar
	}

	// Classify via 1d trend anchor. Reuse api.ClassifySignal to stay in sync
	// with the multi-predict handler (same label rules, same 5m mean_rev
	// disable).
	//
	// PHASE 1 fix: route through ml.GetDailyDirectionCached so the runner
	// and the HTTP handler share the same 15-min TTL cache. Previously the
	// runner called resolveDailyDir directly, triggering a 1d PredictV2
	// every tick per scenario — multiplied Binance 1d fetches by N.
	dailyDir := ml.GetDailyDirectionCached(s.Symbol, func() string {
		return resolveDailyDir(s.Symbol)
	})
	label, reason := ml.ClassifySignal(pred.Direction, dailyDir, s.Interval, pred.Features)
	ev.Label = label
	ev.LabelReason = reason

	// Tier label gate.
	tier := ml.GetTier(s.RiskTier)
	if !tier.LabelAllowed(label) {
		ev.BlockReason = "label_not_allowed"
		return ev
	}

	// Dedup: same bar already fired (last_signal_bar_time in DB) and same dir.
	if s.LastSignalBarTime == last.Time && s.LastSignalDirection == pred.Direction {
		ev.BlockReason = "same_bar"
		return ev
	}

	// Only HC signals become trades. Non-HC passes through as "for UI only"
	// — scenario runner doesn't alert on them.
	if !pred.Metrics.HighConfidence {
		return ev
	}

	// Turtle position sizing: risk$ / (ATR × SL_mult).
	atr := pred.Features["atr_norm_14"] * last.Close // denormalize
	slDist := atr * tier.SLAtrMult
	tpDist := atr * tier.TPAtrMult
	if slDist <= 0 {
		ev.BlockReason = "atr_zero"
		return ev
	}

	riskUSD := equityUSD * tier.RiskPerTradePct
	posSize := 0.0
	if riskUSD > 0 && slDist > 0 {
		posSize = riskUSD / slDist
	}

	entry := last.Close
	var sl, tp float64
	if pred.Direction == "buy" {
		sl = entry - slDist
		tp = entry + tpDist
	} else {
		sl = entry + slDist
		tp = entry - tpDist
	}

	ev.Ready = true
	ev.Direction = pred.Direction
	ev.EntryPrice = entry
	ev.StopLoss = sl
	ev.TakeProfit = tp
	ev.PositionSizeUSD = posSize * entry // notional USD value
	return ev
}

func closesOf(cs []types.OHLCVCandle) []float64 {
	out := make([]float64, len(cs))
	for i, c := range cs {
		out[i] = c.Close
	}
	return out
}

// resolveDailyDir fetches a cheap 1d PredictV2 to anchor MTF classification.
//
// This is the compute function — callers MUST wrap it with
// ml.GetDailyDirectionCached(symbol, func() string { return resolveDailyDir(symbol) })
// to avoid hammering Binance + re-predicting 1d on every tick.
//
// Returns "neutral" on fetch failure (treated as "no trend lock" by
// ClassifySignal — signal may still fire as random/mean_reversion per tier).
func resolveDailyDir(symbol string) string {
	candles, err := market.FetchCryptoCandlesCached(symbol, "1d", 300)
	if err != nil || len(candles) < 30 {
		return "neutral"
	}
	pred := ml.PredictV2(symbol, "1d", "swing", candles, nil, nil)
	return pred.Direction
}
