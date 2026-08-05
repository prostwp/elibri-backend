package macro

// compute_test.go — table-driven tests for the pure compute layer (mirrors the
// rigor of whale/scorer_test.go + funding/store_test.go). Covers:
//   - Pearson: +1 / -1 / 0 / <2pts / zero-variance / clamp.
//   - LampStatus: every lamp's rules at the boundary values.
//   - Composite: all-tailwind / all-headwind / mix / N/D renormalisation / nil.
//   - ClassifyRegime: 34/35/65/66/nil thresholds.
//   - BuildDiagnosis: each template + the BANNED-WORD assertion (incl. \bsupport\b).
//   - CorrelationLabel: nil / strong / inverse.

import (
	"math"
	"testing"
)

func floatPtr(f float64) *float64 { return &f }
func intPtr(i int) *int           { return &i }

// lamp is a tiny helper to build a Lamp with just key+status (the only fields
// Composite/BuildDiagnosis read).
func lamp(key, status string) Lamp { return Lamp{Key: key, Status: status} }

func TestPearson(t *testing.T) {
	t.Run("perfect positive", func(t *testing.T) {
		xs := []float64{1, 2, 3, 4, 5}
		ys := []float64{2, 4, 6, 8, 10} // y = 2x → +1
		got := Pearson(xs, ys)
		if got == nil {
			t.Fatalf("Pearson = nil, want ≈+1")
		}
		if math.Abs(*got-1) > 1e-9 {
			t.Errorf("Pearson = %v, want ≈+1", *got)
		}
	})

	t.Run("perfect negative", func(t *testing.T) {
		xs := []float64{1, 2, 3, 4, 5}
		ys := []float64{10, 8, 6, 4, 2} // y = -2x+12 → -1
		got := Pearson(xs, ys)
		if got == nil {
			t.Fatalf("Pearson = nil, want ≈-1")
		}
		if math.Abs(*got+1) > 1e-9 {
			t.Errorf("Pearson = %v, want ≈-1", *got)
		}
	})

	t.Run("zero correlation", func(t *testing.T) {
		// Symmetric ys around the mean, uncorrelated with the monotone xs.
		xs := []float64{1, 2, 3, 4, 5}
		ys := []float64{2, -2, 2, -2, 2}
		got := Pearson(xs, ys)
		if got == nil {
			t.Fatalf("Pearson = nil, want a finite small coefficient")
		}
		if math.Abs(*got) > 0.5 {
			t.Errorf("Pearson = %v, want near 0", *got)
		}
	})

	t.Run("too few points", func(t *testing.T) {
		if got := Pearson([]float64{1}, []float64{2}); got != nil {
			t.Errorf("Pearson(1pt) = %v, want nil", *got)
		}
		if got := Pearson(nil, nil); got != nil {
			t.Errorf("Pearson(empty) = %v, want nil", *got)
		}
	})

	t.Run("length mismatch", func(t *testing.T) {
		if got := Pearson([]float64{1, 2, 3}, []float64{1, 2}); got != nil {
			t.Errorf("Pearson(mismatched len) = %v, want nil", *got)
		}
	})

	t.Run("zero variance is nil (weekend flat window)", func(t *testing.T) {
		// Every tradfin value identical (Friday close repeated) → undefined.
		xs := []float64{100, 100, 100, 100}
		ys := []float64{50, 51, 52, 53}
		if got := Pearson(xs, ys); got != nil {
			t.Errorf("Pearson(flat xs) = %v, want nil (zero variance)", *got)
		}
		// Both flat.
		if got := Pearson([]float64{5, 5, 5}, []float64{9, 9, 9}); got != nil {
			t.Errorf("Pearson(both flat) = %v, want nil", *got)
		}
	})

	t.Run("result is clamped to [-1,1]", func(t *testing.T) {
		// A genuinely perfect relationship can drift to 1.0000000002 in float;
		// assert the output never escapes the range across a few perfect series.
		series := [][2][]float64{
			{{1, 2, 3, 4, 5, 6, 7}, {3, 6, 9, 12, 15, 18, 21}},
			{{0.1, 0.2, 0.3}, {0.3, 0.6, 0.9}},
		}
		for _, s := range series {
			got := Pearson(s[0], s[1])
			if got == nil {
				t.Fatalf("Pearson = nil, want a coefficient")
			}
			if *got > 1 || *got < -1 {
				t.Errorf("Pearson = %v, escaped [-1,1]", *got)
			}
		}
	})
}

func TestLampStatus(t *testing.T) {
	cases := []struct {
		name  string
		key   string
		value float64
		delta float64
		want  string
	}{
		// DXY (inverse): delta<0 tailwind, >+0.5 headwind, else neutral.
		{"dxy down → tailwind", KeyDXY, 98.0, -0.2, StatusTailwind},
		{"dxy flat → neutral", KeyDXY, 98.0, 0.0, StatusNeutral},
		{"dxy small up → neutral (≤0.5)", KeyDXY, 98.0, 0.5, StatusNeutral},
		{"dxy strong up → headwind (>0.5)", KeyDXY, 98.0, 0.6, StatusHeadwind},
		// Rates (inverse, like DXY).
		{"rates down → tailwind", KeyRates, 4.4, -0.05, StatusTailwind},
		{"rates strong up → headwind", KeyRates, 4.4, 0.7, StatusHeadwind},
		{"rates flat → neutral", KeyRates, 4.4, 0.0, StatusNeutral},
		// Gold (inverse, like DXY).
		{"gold down → tailwind", KeyGold, 4500, -0.3, StatusTailwind},
		{"gold strong up → headwind", KeyGold, 4500, 1.0, StatusHeadwind},
		{"gold small up → neutral", KeyGold, 4500, 0.4, StatusNeutral},
		// SPX (direct): delta>0 tailwind, <-0.5 headwind, else neutral.
		{"spx up → tailwind", KeySPX, 7580, 0.62, StatusTailwind},
		{"spx flat → neutral", KeySPX, 7580, 0.0, StatusNeutral},
		{"spx small down → neutral (≥-0.5)", KeySPX, 7580, -0.5, StatusNeutral},
		{"spx strong down → headwind (<-0.5)", KeySPX, 7580, -0.6, StatusHeadwind},
		// VIX (level): <18 tailwind, >25 headwind, else neutral (delta ignored).
		{"vix calm → tailwind", KeyVIX, 17.59, 5.0, StatusTailwind},
		{"vix at 18 → neutral (not <18)", KeyVIX, 18.0, -5.0, StatusNeutral},
		{"vix mid → neutral", KeyVIX, 21.0, 0.0, StatusNeutral},
		{"vix at 25 → neutral (not >25)", KeyVIX, 25.0, 0.0, StatusNeutral},
		{"vix fear → headwind", KeyVIX, 30.0, -1.0, StatusHeadwind},
		// Unknown key → neutral.
		{"unknown → neutral", "btc", 1.0, 9.0, StatusNeutral},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if got := LampStatus(tc.key, tc.value, tc.delta); got != tc.want {
				t.Errorf("LampStatus(%s, %v, %v) = %q, want %q", tc.key, tc.value, tc.delta, got, tc.want)
			}
		})
	}
}

func TestComposite(t *testing.T) {
	t.Run("all tailwind → 100", func(t *testing.T) {
		lamps := []Lamp{
			lamp(KeyDXY, StatusTailwind), lamp(KeyVIX, StatusTailwind),
			lamp(KeySPX, StatusTailwind), lamp(KeyRates, StatusTailwind),
			lamp(KeyGold, StatusTailwind),
		}
		got := Composite(lamps)
		if got == nil || *got != 100 {
			t.Fatalf("Composite(all tailwind) = %v, want 100", got)
		}
	})

	t.Run("all headwind → 0", func(t *testing.T) {
		lamps := []Lamp{
			lamp(KeyDXY, StatusHeadwind), lamp(KeyVIX, StatusHeadwind),
			lamp(KeySPX, StatusHeadwind), lamp(KeyRates, StatusHeadwind),
			lamp(KeyGold, StatusHeadwind),
		}
		got := Composite(lamps)
		if got == nil || *got != 0 {
			t.Fatalf("Composite(all headwind) = %v, want 0", got)
		}
	})

	t.Run("all neutral → 50", func(t *testing.T) {
		lamps := []Lamp{
			lamp(KeyDXY, StatusNeutral), lamp(KeyVIX, StatusNeutral),
			lamp(KeySPX, StatusNeutral), lamp(KeyRates, StatusNeutral),
			lamp(KeyGold, StatusNeutral),
		}
		got := Composite(lamps)
		if got == nil || *got != 50 {
			t.Fatalf("Composite(all neutral) = %v, want 50", got)
		}
	})

	t.Run("N/D renormalisation: 1 lamp out", func(t *testing.T) {
		// Gold N/D (status ""). Remaining active weights = 25+25+25+15 = 90, all
		// tailwind → score 90 → 90/90*100 = 100.
		lamps := []Lamp{
			lamp(KeyDXY, StatusTailwind), lamp(KeyVIX, StatusTailwind),
			lamp(KeySPX, StatusTailwind), lamp(KeyRates, StatusTailwind),
			lamp(KeyGold, ""), // N/D
		}
		got := Composite(lamps)
		if got == nil || *got != 100 {
			t.Fatalf("Composite(gold N/D, rest tailwind) = %v, want 100 (renormalised)", got)
		}
	})

	t.Run("N/D renormalisation: 2 lamps out, mixed", func(t *testing.T) {
		// Active: DXY tailwind(25), VIX headwind(0), SPX neutral(12.5). N/D: rates,
		// gold. activeWeights = 25+25+25 = 75; score = 25+0+12.5 = 37.5 →
		// 37.5/75*100 = 50.
		lamps := []Lamp{
			lamp(KeyDXY, StatusTailwind), lamp(KeyVIX, StatusHeadwind),
			lamp(KeySPX, StatusNeutral), lamp(KeyRates, ""), lamp(KeyGold, ""),
		}
		got := Composite(lamps)
		if got == nil || *got != 50 {
			t.Fatalf("Composite(2 N/D mix) = %v, want 50", got)
		}
	})

	t.Run("3+ N/D → nil (not a fake 50)", func(t *testing.T) {
		lamps := []Lamp{
			lamp(KeyDXY, StatusTailwind), lamp(KeyVIX, StatusTailwind),
			lamp(KeySPX, ""), lamp(KeyRates, ""), lamp(KeyGold, ""),
		}
		if got := Composite(lamps); got != nil {
			t.Errorf("Composite(3 N/D) = %v, want nil", *got)
		}
	})

	t.Run("all N/D → nil", func(t *testing.T) {
		lamps := []Lamp{
			lamp(KeyDXY, ""), lamp(KeyVIX, ""), lamp(KeySPX, ""),
			lamp(KeyRates, ""), lamp(KeyGold, ""),
		}
		if got := Composite(lamps); got != nil {
			t.Errorf("Composite(all N/D) = %v, want nil", *got)
		}
	})

	t.Run("empty → nil", func(t *testing.T) {
		if got := Composite(nil); got != nil {
			t.Errorf("Composite(nil) = %v, want nil", *got)
		}
	})
}

func TestClassifyRegime(t *testing.T) {
	cases := []struct {
		name      string
		composite *int
		want      string
	}{
		{"34 → risk_off", intPtr(34), RegimeRiskOff},
		{"0 → risk_off", intPtr(0), RegimeRiskOff},
		{"35 → mixed (boundary)", intPtr(35), RegimeMixed},
		{"50 → mixed", intPtr(50), RegimeMixed},
		{"65 → mixed (boundary)", intPtr(65), RegimeMixed},
		{"66 → risk_on", intPtr(66), RegimeRiskOn},
		{"100 → risk_on", intPtr(100), RegimeRiskOn},
		{"nil → mixed", nil, RegimeMixed},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if got := ClassifyRegime(tc.composite); got != tc.want {
				t.Errorf("ClassifyRegime(%v) = %q, want %q", tc.composite, got, tc.want)
			}
		})
	}
}

func TestBuildDiagnosis(t *testing.T) {
	// Each (regime, lamps) combination should produce a non-empty, SAFE string,
	// except composite-nil which the handler signals via regime+the empty idea.
	cases := []struct {
		name      string
		regime    string
		lamps     []Lamp
		wantEmpty bool
	}{
		{
			name:   "risk_on dxy+vix tailwind (template #1)",
			regime: RegimeRiskOn,
			lamps:  []Lamp{lamp(KeyDXY, StatusTailwind), lamp(KeyVIX, StatusTailwind)},
		},
		{
			name:   "risk_on spx dominant (template #4)",
			regime: RegimeRiskOn,
			lamps:  []Lamp{lamp(KeyDXY, StatusNeutral), lamp(KeyVIX, StatusNeutral), lamp(KeySPX, StatusTailwind)},
		},
		{
			name:   "risk_on generic",
			regime: RegimeRiskOn,
			lamps:  []Lamp{lamp(KeyDXY, StatusNeutral), lamp(KeyVIX, StatusNeutral), lamp(KeySPX, StatusNeutral)},
		},
		{
			name:   "risk_off dxy+vix headwind (template #2)",
			regime: RegimeRiskOff,
			lamps:  []Lamp{lamp(KeyDXY, StatusHeadwind), lamp(KeyVIX, StatusHeadwind)},
		},
		{
			name:   "risk_off gold flight (template #5)",
			regime: RegimeRiskOff,
			lamps:  []Lamp{lamp(KeyDXY, StatusNeutral), lamp(KeyVIX, StatusNeutral), lamp(KeyGold, StatusHeadwind)},
		},
		{
			name:   "risk_off generic",
			regime: RegimeRiskOff,
			lamps:  []Lamp{lamp(KeyDXY, StatusNeutral), lamp(KeyVIX, StatusNeutral)},
		},
		{
			name:   "mixed (template #3)",
			regime: RegimeMixed,
			lamps:  []Lamp{lamp(KeyDXY, StatusTailwind), lamp(KeyVIX, StatusHeadwind)},
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := BuildDiagnosis(tc.regime, tc.lamps)
			if tc.wantEmpty {
				if got != "" {
					t.Errorf("BuildDiagnosis = %q, want empty", got)
				}
				return
			}
			if got == "" {
				t.Fatalf("BuildDiagnosis = empty, want a sentence")
			}
			// THE critical assertion: zero banned words on the diagnosis output.
			if !IsDiagnosisSafe(got) {
				t.Errorf("BuildDiagnosis output contains a BANNED word: %q", got)
			}
			// Explicit "support" guard (the dnevnik headline gotcha).
			if bannedDiagnosisPattern.MatchString(got) {
				t.Errorf("diagnosis matched banned pattern: %q", got)
			}
		})
	}
}

// TestBuildDiagnosis_NoBannedWordEver enumerates every regime × the full lamp
// set so no reachable template can ever emit a banned word (belt-and-suspenders
// over the table above). "support" must never appear.
func TestBuildDiagnosis_NoBannedWordEver(t *testing.T) {
	statuses := []string{StatusTailwind, StatusNeutral, StatusHeadwind, ""}
	regimes := []string{RegimeRiskOn, RegimeMixed, RegimeRiskOff}
	keys := []string{KeyDXY, KeyVIX, KeySPX, KeyRates, KeyGold}

	for _, regime := range regimes {
		for _, sDXY := range statuses {
			for _, sVIX := range statuses {
				for _, sSPX := range statuses {
					for _, sGold := range statuses {
						lamps := []Lamp{
							lamp(keys[0], sDXY), lamp(keys[1], sVIX),
							lamp(keys[2], sSPX), lamp(keys[4], sGold),
						}
						got := BuildDiagnosis(regime, lamps)
						if got != "" && !IsDiagnosisSafe(got) {
							t.Fatalf("regime=%s dxy=%q vix=%q spx=%q gold=%q → banned output: %q",
								regime, sDXY, sVIX, sSPX, sGold, got)
						}
					}
				}
			}
		}
	}
}

func TestCorrelationLabel(t *testing.T) {
	cases := []struct {
		name string
		pair string
		coef *float64
		want string
	}{
		{"nil coef → empty", PairBTCSPX, nil, ""},
		{"spx strong positive → moving like stocks", PairBTCSPX, floatPtr(0.78), "moving like stocks"},
		{"spx strong negative → moving against stocks", PairBTCSPX, floatPtr(-0.7), "moving against stocks"},
		{"spx near zero → barely linked", PairBTCSPX, floatPtr(0.1), "barely linked"},
		{"gold strong positive → moving with gold", PairBTCGold, floatPtr(0.65), "moving with gold"},
		{"gold near zero → barely linked", PairBTCGold, floatPtr(0.12), "barely linked"},
		{"dxy strong negative → inverse to the dollar", PairBTCDXY, floatPtr(-0.64), "inverse to the dollar"},
		{"dxy strong positive → tracking the dollar", PairBTCDXY, floatPtr(0.7), "tracking the dollar"},
		{"dxy near zero → barely linked", PairBTCDXY, floatPtr(-0.05), "barely linked"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if got := CorrelationLabel(tc.pair, tc.coef); got != tc.want {
				t.Errorf("CorrelationLabel(%s, %v) = %q, want %q", tc.pair, tc.coef, got, tc.want)
			}
		})
	}
}
