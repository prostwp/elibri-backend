package ml

import "testing"

// TestClassifySignal covers every branch of the tier classifier. Because
// ClassifySignal feeds into tier.LabelAllowed (which Blocks `random` for
// Conservative/Balanced), a bug here silently makes every signal look like
// noise — verified in Patch 2G where adx_14 was compared against the wrong
// scale and trend_aligned was unreachable.
//
// feature keys and their scale (matches feature_engine.py + features_v2.go):
//   adx_14  : stored as adx/100  — classify denormalizes to absolute 0-100
//   rsi_14  : stored absolute    — classify compares directly
//   bb_position : stored 0-1     — classify compares directly
func TestClassifySignal(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		signalDir  string
		dailyDir   string
		interval   string
		features   map[string]float64
		wantLabel  string
		wantReason string // substring match; empty means skip
	}{
		{
			name:      "trend_aligned — signal buy, daily buy, strong adx",
			signalDir: "buy", dailyDir: "buy", interval: "1h",
			features:  map[string]float64{"adx_14": 0.41, "rsi_14": 60, "bb_position": 0.5},
			wantLabel: "trend_aligned",
		},
		{
			name:      "trend_aligned — signal sell, daily sell",
			signalDir: "sell", dailyDir: "sell", interval: "4h",
			features:  map[string]float64{"adx_14": 0.25, "rsi_14": 40},
			wantLabel: "trend_aligned",
		},
		{
			name:      "random — adx too low (20 absolute ≈ 0.20 normalized)",
			signalDir: "buy", dailyDir: "buy", interval: "1h",
			features:  map[string]float64{"adx_14": 0.19, "rsi_14": 60},
			wantLabel: "random",
		},
		{
			name:      "random — adx boundary exactly 20 is NOT trend_aligned (strict >)",
			signalDir: "buy", dailyDir: "buy", interval: "1h",
			features:  map[string]float64{"adx_14": 0.20, "rsi_14": 60},
			wantLabel: "random",
		},
		{
			name:      "mean_reversion — 1d flat, RSI overbought, BB upper",
			signalDir: "sell", dailyDir: "neutral", interval: "1h",
			features:  map[string]float64{"rsi_14": 72, "bb_position": 0.95, "adx_14": 0.10},
			wantLabel: "mean_reversion",
		},
		{
			name:      "mean_reversion — 1d flat, RSI oversold, BB lower",
			signalDir: "buy", dailyDir: "neutral", interval: "15m",
			features:  map[string]float64{"rsi_14": 28, "bb_position": 0.05, "adx_14": 0.08},
			wantLabel: "mean_reversion",
		},
		{
			name:      "mean_reversion disabled on 5m (Patch 2F) — downgraded to random",
			signalDir: "buy", dailyDir: "neutral", interval: "5m",
			features:  map[string]float64{"rsi_14": 28, "bb_position": 0.05},
			wantLabel: "random",
			wantReason: "mean_rev disabled on 5m",
		},
		{
			name:      "random — daily flat but RSI not extreme",
			signalDir: "buy", dailyDir: "neutral", interval: "1h",
			features:  map[string]float64{"rsi_14": 55, "bb_position": 0.5, "adx_14": 0.10},
			wantLabel: "random",
		},
		{
			name:      "random — daily buy but signal sell (mismatch)",
			signalDir: "sell", dailyDir: "buy", interval: "4h",
			features:  map[string]float64{"adx_14": 0.50, "rsi_14": 50},
			wantLabel: "random",
		},
		{
			name:      "random — signal neutral (degenerate) never trend_aligned",
			signalDir: "neutral", dailyDir: "buy", interval: "1h",
			features:  map[string]float64{"adx_14": 0.50, "rsi_14": 50},
			wantLabel: "random",
		},
		{
			name:      "random — missing features map produces random (no panic)",
			signalDir: "buy", dailyDir: "buy", interval: "1h",
			features:  map[string]float64{},
			wantLabel: "random",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotLabel, gotReason := ClassifySignal(tc.signalDir, tc.dailyDir, tc.interval, tc.features)
			if gotLabel != tc.wantLabel {
				t.Fatalf("label: got %q want %q (reason=%q)", gotLabel, tc.wantLabel, gotReason)
			}
			if tc.wantReason != "" && !contains(gotReason, tc.wantReason) {
				t.Fatalf("reason should contain %q, got %q", tc.wantReason, gotReason)
			}
		})
	}
}

// contains is a tiny substring-match helper; keeps the test file
// dependency-free.
func contains(s, sub string) bool {
	if sub == "" {
		return true
	}
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
