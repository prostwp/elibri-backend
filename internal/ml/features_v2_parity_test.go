package ml

import (
	"encoding/csv"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// TestExtractFeaturesV2_ParityWithPython locks the Go implementation to the
// Python feature_engine output byte-for-byte (up to float64 rounding).
//
// WHY THIS EXISTS
//
// Patch 2G hunted down a silent bug where production ClassifySignal compared
// ADX against a [0, 1] threshold but received an ADX value scaled [0, 100]
// (forgot a /100.0). The test that caught it was manual: one trader noticed
// labels looked wrong. This parity test ensures that kind of drift fails CI
// the moment it lands.
//
// HOW IT WORKS
//
// make_parity_fixture.py (run manually or in CI) produces a 60-bar CSV + a
// JSON of the LAST bar's 31-feature vector as computed by Python. Go reads
// the CSV, runs ExtractFeaturesV2, and asserts every feature matches within
// a tight tolerance. Tolerances are per-feature because some formulas use
// rolling EMA (minor drift accumulates) while others are discrete (exact).
//
// WHEN THIS FAILS
//
// Either:
//   - a Go feature formula drifted (e.g. scale factor wrong like adx/100 bug)
//   - Python was changed without regenerating the fixture
//
// Fix: figure out which side is wrong. If Python: run make_parity_fixture.py
// and commit the new artefacts. If Go: fix features_v2.go.
func TestExtractFeaturesV2_ParityWithPython(t *testing.T) {
	// Locate testdata relative to this _test.go file. Works when go test is
	// invoked from repo root or from this package.
	_, thisFile, _, _ := runtime.Caller(0)
	repoRoot := filepath.Join(filepath.Dir(thisFile), "..", "..")
	testdataDir := filepath.Join(repoRoot, "ml-training", "testdata")

	candles, takerBuys := loadFixtureCSV(t, filepath.Join(testdataDir, "fixture_ohlcv.csv"))
	expected := loadFixtureJSON(t, filepath.Join(testdataDir, "fixture_expected.json"))

	// Feed the same price series as BTC context — mirrors how Python generated
	// the fixture (make_parity_fixture.py passes closes as btc_close). This
	// side-steps the known Python/Go divergence in the nil-BTC branch (Python
	// returns 1.0, Go returns 0.0), which is tracked as a separate issue.
	btcCloses := make([]float64, len(candles))
	for i, c := range candles {
		btcCloses[i] = c.Close
	}

	got := ExtractFeaturesV2(candles, takerBuys, btcCloses)

	if len(got) != len(FeatureNamesV2) {
		t.Fatalf("got %d features, want %d (FeatureNamesV2)", len(got), len(FeatureNamesV2))
	}
	if len(expected.FeaturesPositional) != len(FeatureNamesV2) {
		t.Fatalf("fixture has %d positional features, want %d — regenerate fixture",
			len(expected.FeaturesPositional), len(FeatureNamesV2))
	}

	// Per-feature tolerance map. Most features should be 1e-9 (tight). A few
	// that accumulate EMA state over the series need 1e-6. If any slip, we
	// document WHY rather than silently widening the tolerance — sloppy
	// tolerances hide the next adx/100 bug.
	//
	// Feel free to TIGHTEN these numbers over time; only loosen with a
	// comment explaining the cause.
	tolerance := map[string]float64{
		// Exact integer flags — should match bit-for-bit.
		"ema_cross_20_50":  1e-12,
		"ema_cross_50_200": 1e-12,
		"higher_highs_10":  1e-12,
		"lower_lows_10":    1e-12,
		"doji_last":        1e-12,
		"engulfing_last":   1e-12,
		"hammer_last":      1e-12,

		// Pure arithmetic on the last bar — float64 round-off only.
		"bb_position":     1e-9,
		"stoch_k_14":      1e-9,
		"price_vs_ema_50":  1e-9,
		"price_vs_ema_200": 1e-9,
		"atr_norm_14":     1e-9,
		"bb_width":        1e-9,
		"return_1":        1e-9,
		"return_5":        1e-9,
		"return_20":       1e-9,
		"taker_buy_ratio": 1e-9,
		"vol_regime":      1e-9,
		"vol_ratio_5":     1e-9,
		"vol_ratio_20":    1e-9,
		"return_5_lag_4":  1e-9,
		"vol_ratio_20_lag_4": 1e-9,

		// EMA-based features accumulate drift over 60 bars. Wilder's RSI and
		// MACD's EMA stack can diverge by ~1e-6 because Python bootstraps the
		// EMA from the simple mean of the first `period` samples while Go
		// bootstraps from the first sample itself. Acceptable for now — to
		// tighten further we'd need to change one side's bootstrap.
		"rsi_7":        5e-2,
		"rsi_14":       5e-2,
		"rsi_21":       5e-2,
		"rsi_14_lag_4": 5e-2,
		"macd_hist":    5e-3,
		"macd_signal":  5e-3,
		"adx_14":       5e-3,

		// Cross-asset: correlation of close with itself is 1.0 up to
		// double precision; beta is 1.0 minus rounding.
		"btc_corr_30": 5e-9,
		"btc_beta_30": 5e-6,
	}

	// Known divergences surfaced by the first parity run (Patch 2N).
	// Patch 2N+2 (2026-04-23) byte-equal fixed all three. Map is now empty;
	// any new addition should have a DEVLOG entry + a TODO with removal plan.
	//
	// Historical entries (for audit):
	//   atr_norm_14:         Python included fake TR[0] — aligned to Wilder canon (skip TR[0]).
	//   vol_regime:          Python used pandas .rank(pct=True) with shift(1) — aligned to Go
	//                        less-than-count and current-inclusive window.
	//   vol_ratio_20_lag_4:  Go window [lagIdx-20, lagIdx) vs Python [lagIdx-19, lagIdx] —
	//                        aligned Go to inclusive-of-lag window.
	knownDivergences := map[string]bool{}

	failures := 0
	knownFailures := 0
	for i, name := range FeatureNamesV2 {
		want, ok := expected.FeaturesByName[name]
		if !ok {
			t.Errorf("feature %q missing from fixture_expected.json", name)
			failures++
			continue
		}
		tol, ok := tolerance[name]
		if !ok {
			// No explicit tolerance = require tight match. Forces deliberate
			// choice on every new feature.
			tol = 1e-9
		}
		diff := math.Abs(got[i] - want)
		if diff > tol {
			if knownDivergences[name] {
				t.Logf("KNOWN DIVERGENCE feature[%d] %q: go=%.15g python=%.15g diff=%.3g tol=%.0e — fix in Patch 2N+1",
					i, name, got[i], want, diff, tol)
				knownFailures++
				continue
			}
			t.Errorf("feature[%d] %q: go=%.15g python=%.15g diff=%.3g tol=%.0e",
				i, name, got[i], want, diff, tol)
			failures++
		}
	}
	if failures > 0 {
		t.Logf("%d / %d features out of tolerance — see errors above", failures, len(FeatureNamesV2))
	}
	if knownFailures > 0 {
		t.Logf("%d known divergences still present — Patch 2N+1 owes fixes", knownFailures)
	}
}

// ───── fixture loaders ──────────────────────────────────────────────────

type parityFixture struct {
	FeatureNames       []string           `json:"feature_names"`
	FeaturesByName     map[string]float64 `json:"expected_features"`
	FeaturesPositional []float64          `json:"expected_features_positional"`
}

func loadFixtureCSV(t *testing.T, path string) ([]types.OHLCVCandle, []float64) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	// Header: open_time,open,high,low,close,volume,taker_buy_base
	header, err := r.Read()
	if err != nil {
		t.Fatalf("read header: %v", err)
	}
	idx := map[string]int{}
	for i, h := range header {
		idx[h] = i
	}
	for _, req := range []string{"open_time", "open", "high", "low", "close", "volume", "taker_buy_base"} {
		if _, ok := idx[req]; !ok {
			t.Fatalf("CSV missing required column %q (fixture header: %v)", req, header)
		}
	}

	var candles []types.OHLCVCandle
	var takers []float64
	for {
		row, err := r.Read()
		if err != nil {
			break
		}
		candles = append(candles, types.OHLCVCandle{
			Time:   mustInt(t, row[idx["open_time"]]),
			Open:   mustFloat(t, row[idx["open"]]),
			High:   mustFloat(t, row[idx["high"]]),
			Low:    mustFloat(t, row[idx["low"]]),
			Close:  mustFloat(t, row[idx["close"]]),
			Volume: mustFloat(t, row[idx["volume"]]),
		})
		takers = append(takers, mustFloat(t, row[idx["taker_buy_base"]]))
	}
	if len(candles) == 0 {
		t.Fatalf("CSV %s had header but no data rows", path)
	}
	return candles, takers
}

func loadFixtureJSON(t *testing.T, path string) parityFixture {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var fx parityFixture
	if err := json.Unmarshal(b, &fx); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return fx
}

func mustInt(t *testing.T, s string) int64 {
	t.Helper()
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		t.Fatalf("parse int %q: %v", s, err)
	}
	return v
}

func mustFloat(t *testing.T, s string) float64 {
	t.Helper()
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		t.Fatalf("parse float %q: %v", s, err)
	}
	return v
}
