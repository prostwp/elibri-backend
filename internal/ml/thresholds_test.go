package ml

import (
	"math"
	"os"
	"sync"
	"testing"
)

// TestDeriveAbsoluteThresholds exercises the parser that Patch 2G rewrote.
// Original bug: switch matched only 5 hardcoded keys; thr_0.675 / thr_0.775
// fell through to default 0.80/0.20, silently over-filtering signals for
// months. This test locks in the new behaviour.
func TestDeriveAbsoluteThresholds(t *testing.T) {
	t.Parallel()

	cases := []struct {
		key      string
		mean     float64
		std      float64
		wantHigh float64
		wantLow  float64
	}{
		{"thr_0.55", 0, 0, 0.55, 0.45},
		{"thr_0.60", 0, 0, 0.60, 0.40},
		{"thr_0.625", 0, 0, 0.625, 0.375}, // the one that USED to fall through
		{"thr_0.675", 0, 0, 0.675, 0.325}, // and this one
		{"thr_0.70", 0, 0, 0.70, 0.30},
		{"thr_0.775", 0, 0, 0.775, 0.225}, // and this one
		{"thr_0.80", 0, 0, 0.80, 0.20},
		{"thr_0.85", 0, 0, 0.85, 0.15},
		// Boundary: thr_0.5 rejected by `v > 0.5` — falls to default.
		{"thr_0.5", 0, 0, 0.80, 0.20},
		// Invalid / unknown — safe default.
		{"", 0, 0, 0.80, 0.20},
		{"garbage", 0, 0, 0.80, 0.20},
		{"thr_", 0, 0, 0.80, 0.20},
		{"thr_abc", 0, 0, 0.80, 0.20},
		{"thr_1.5", 0, 0, 0.80, 0.20}, // above 1.0 range
		// top_10pct: mean+1.28*std, clamped. mean=0.6, std=0.1 → high=0.728, low=0.472.
		{"top_10pct", 0.6, 0.1, 0.6 + 1.28*0.1, 0.6 - 1.28*0.1},
		// top_10pct with tiny std hits the clamp.
		{"top_10pct", 0.5, 0.001, 0.51, 0.49},
	}

	for _, tc := range cases {
		t.Run(tc.key, func(t *testing.T) {
			high, low := deriveAbsoluteThresholds(tc.key, tc.mean, tc.std)
			if math.Abs(high-tc.wantHigh) > 1e-9 {
				t.Errorf("high: got %v want %v", high, tc.wantHigh)
			}
			if math.Abs(low-tc.wantLow) > 1e-9 {
				t.Errorf("low: got %v want %v", low, tc.wantLow)
			}
		})
	}
}

// TestApplyEnvOverrides covers the ML_THRESHOLD_OVERRIDES parser added in
// Patch 2G. Format: "SYMBOL_INTERVAL:value,SYMBOL_INTERVAL:value"
func TestApplyEnvOverrides(t *testing.T) {
	// Serial — mutates package-level `thresholds` map.
	reset := func() {
		thresholdsMu.Lock()
		thresholds = make(map[string]HCThreshold)
		thresholdsMu.Unlock()
		_ = os.Unsetenv("ML_THRESHOLD_OVERRIDES")
	}

	t.Run("single override", func(t *testing.T) {
		reset()
		os.Setenv("ML_THRESHOLD_OVERRIDES", "BTCUSDT_4h:0.60")
		defer reset()

		n := applyEnvOverrides()
		if n != 1 {
			t.Fatalf("expected 1 override, got %d", n)
		}
		got := GetThreshold("BTCUSDT", "4h")
		if got.ThresholdHigh != 0.60 || got.ThresholdLow != 0.40 {
			t.Errorf("BTCUSDT_4h: high=%v low=%v", got.ThresholdHigh, got.ThresholdLow)
		}
		if got.Key != "env_override" {
			t.Errorf("Key: got %q want env_override", got.Key)
		}
	})

	t.Run("multiple overrides with whitespace", func(t *testing.T) {
		reset()
		os.Setenv("ML_THRESHOLD_OVERRIDES", "BTCUSDT_4h:0.60 , BTCUSDT_1h:0.65 , ETHUSDT_1d:0.70")
		defer reset()

		if n := applyEnvOverrides(); n != 3 {
			t.Fatalf("expected 3, got %d", n)
		}
		for _, want := range []struct {
			sym, iv string
			v       float64
		}{
			{"BTCUSDT", "4h", 0.60},
			{"BTCUSDT", "1h", 0.65},
			{"ETHUSDT", "1d", 0.70},
		} {
			got := GetThreshold(want.sym, want.iv)
			if math.Abs(got.ThresholdHigh-want.v) > 1e-9 {
				t.Errorf("%s_%s: high=%v want %v", want.sym, want.iv, got.ThresholdHigh, want.v)
			}
		}
	})

	t.Run("invalid entries skipped", func(t *testing.T) {
		reset()
		// Malformed: no colon, negative value, out-of-range, empty.
		os.Setenv("ML_THRESHOLD_OVERRIDES", "BADKEY,,BTCUSDT_4h:abc,BTCUSDT_1h:1.5,BTCUSDT_5m:-0.5,BTCUSDT_15m:0.60")
		defer reset()

		// Only the last (valid) entry applies.
		if n := applyEnvOverrides(); n != 1 {
			t.Fatalf("expected 1 valid override, got %d", n)
		}
		got := GetThreshold("BTCUSDT", "15m")
		if got.ThresholdHigh != 0.60 {
			t.Errorf("BTCUSDT_15m high: got %v want 0.60", got.ThresholdHigh)
		}
	})

	t.Run("empty env is no-op", func(t *testing.T) {
		reset()
		os.Unsetenv("ML_THRESHOLD_OVERRIDES")
		if n := applyEnvOverrides(); n != 0 {
			t.Errorf("expected 0 overrides on empty env, got %d", n)
		}
	})
}

// TestGetThreshold safe defaults when nothing loaded.
func TestGetThresholdDefaults(t *testing.T) {
	thresholdsMu.Lock()
	thresholds = make(map[string]HCThreshold)
	thresholdsMu.Unlock()

	got := GetThreshold("UNKNOWN", "5m")
	if got.ThresholdHigh != 0.80 || got.ThresholdLow != 0.20 {
		t.Errorf("default: high=%v low=%v", got.ThresholdHigh, got.ThresholdLow)
	}
	if got.Key != "default_0.80" {
		t.Errorf("Key: got %q want default_0.80", got.Key)
	}
}

// TestThresholdsConcurrentAccess — sanity check that the sync.RWMutex
// actually serializes read/write correctly under load. Detects the
// "forgot to take the lock" bug class.
func TestThresholdsConcurrentAccess(t *testing.T) {
	thresholdsMu.Lock()
	thresholds = make(map[string]HCThreshold)
	thresholds["BTCUSDT_1h"] = HCThreshold{ThresholdHigh: 0.7, ThresholdLow: 0.3}
	thresholdsMu.Unlock()

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(2)
		go func() { defer wg.Done(); _ = GetThreshold("BTCUSDT", "1h") }()
		go func() {
			defer wg.Done()
			thresholdsMu.Lock()
			thresholds["ETHUSDT_1h"] = HCThreshold{ThresholdHigh: 0.7, ThresholdLow: 0.3}
			thresholdsMu.Unlock()
		}()
	}
	wg.Wait()
	// If `go test -race ./internal/ml` passes, we're good.
}
