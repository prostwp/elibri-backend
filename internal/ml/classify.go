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
	adx := features["adx_14"]
	rsi := features["rsi_14"]
	bb := features["bb_position"]

	if signalDir != "neutral" && dailyDir == signalDir && adx > 20 {
		return "trend_aligned", fmt.Sprintf("1d %s aligned, adx_14=%.1f", signalDir, adx)
	}
	if dailyDir == "neutral" && (rsi < 30 || rsi > 70) && (bb < 0.1 || bb > 0.9) {
		label := "mean_reversion"
		reason := fmt.Sprintf("1d flat, rsi_14=%.1f, bb_pos=%.2f", rsi, bb)
		if interval == "5m" && label == "mean_reversion" {
			label = "random"
			reason = "mean_rev disabled on 5m (backtest confirmed pure noise)"
		}
		return label, reason
	}
	return "random", fmt.Sprintf("no regime match (1d=%s, adx=%.1f, rsi=%.1f)", dailyDir, adx, rsi)
}
