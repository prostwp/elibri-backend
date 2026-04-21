package ml

import "fmt"

// ClassifySignal labels a signal as one of:
//   - "trend_aligned"   — 1d trend agrees with signal direction AND adx > 20
//   - "mean_reversion"  — 1d neutral AND RSI extreme AND price at BB edge
//   - "random"          — neither condition holds
//
// Pure rule-based — no model required. `features` is keyed by FEATURE_NAMES
// entries (rsi_14, adx_14, bb_position, …). `interval` is the signal TF.
// Returns (label, reason).
//
// Patch 2F: mean-reversion is forcibly downgraded to "random" on 5m because
// backtest_v2 showed −69% return and 70% drawdown on 5m mean-rev — pure
// noise. Keep the override here so it applies everywhere ClassifySignal is
// called (HTTP predict, scenario runner, backtester).
func ClassifySignal(signalDir, dailyDir, interval string, features map[string]float64) (string, string) {
	// adx_14 is stored as ADX/100 (see feature_engine.py:`"adx_14": adx14/100.0`).
	// Previous `adx > 20` check would only fire when absolute ADX ≥ 2000 —
	// effectively never — meaning "trend_aligned" was unreachable and every
	// signal got labelled "random". Denormalize to absolute ADX (0-100 scale).
	adxNorm := features["adx_14"]
	adxAbs := adxNorm * 100.0
	// rsi_14 is ALSO normalized (rsi/100) in feature_engine — use absolute
	// thresholds on the denormalized value.
	rsiNorm := features["rsi_14"]
	rsiAbs := rsiNorm * 100.0
	bb := features["bb_position"]

	if signalDir != "neutral" && dailyDir == signalDir && adxAbs > 20 {
		return "trend_aligned", fmt.Sprintf("1d %s aligned, adx=%.1f", signalDir, adxAbs)
	}
	if dailyDir == "neutral" && (rsiAbs < 30 || rsiAbs > 70) && (bb < 0.1 || bb > 0.9) {
		label := "mean_reversion"
		reason := fmt.Sprintf("1d flat, rsi=%.1f, bb_pos=%.2f", rsiAbs, bb)
		if interval == "5m" && label == "mean_reversion" {
			label = "random"
			reason = "mean_rev disabled on 5m (backtest confirmed pure noise)"
		}
		return label, reason
	}
	return "random", fmt.Sprintf("no regime match (1d=%s, adx=%.1f, rsi=%.1f)", dailyDir, adxAbs, rsiAbs)
}
