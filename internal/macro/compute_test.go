package macro

// compute_test.go — table-driven tests for the pure compute layer (mirrors the
// rigor of whale/scorer_test.go + funding/store_test.go). Covers:
//   - Pearson: +1 / -1 / 0 / <2pts / zero-variance / clamp.
//   - LampStatus: every lamp's rules at the boundary values.
//   - Composite: all-tailwind / all-headwind / mix / N/D renormalisation / nil.
//   - ClassifyRegime: 34/35/65/66 thresholds + nil-composite split
//     (real lamps → mixed, zero real lamps → unknown).
//   - TradfinOK: value / status / empty lamp sets.
//   - TradfinWindowOpen: Sun 22:00 / Fri 21:00 UTC boundaries + mid-week + Sat.
//   - BuildDiagnosis: each template + the BANNED-WORD assertion (incl. \bsupport\b)
//     + the zero-data guard (unknown / empty lamps → "" — never "signals are split").
//   - CorrelationLabel: nil / strong / inverse.

import (
	"math"
	"testing"
	"time"
)

func floatPtr(f float64) *float64 { return &f }
func intPtr(i int) *int           { return &i }

// lamp is a tiny helper to build a Lamp with just key+status (the only fields
// Composite/BuildDiagnosis read). In production a status implies a value, so a
// status-only lamp counts as "real" for the honesty guards.
func lamp(key, status string) Lamp { return Lamp{Key: key, Status: status} }

// valuedLamp builds a value-carrying lamp with no status (e.g. a directional
// lamp whose session delta is unknown).
func valuedLamp(key string, v float64) Lamp { return Lamp{Key: key, Value: &v, OK: true} }

// emptyLamp builds an all-N/D lamp (no value, no status) — the weekend shape.
func emptyLamp(key string) Lamp { return Lamp{Key: key} }

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
	// A composite value only exists when ≥3 lamps carry statuses, so the scored
	// cases run over a realistic valued lamp set.
	scored := []Lamp{
		{Key: KeyDXY, Value: floatPtr(99), Status: StatusTailwind},
		{Key: KeyVIX, Value: floatPtr(17), Status: StatusTailwind},
		{Key: KeySPX, Value: floatPtr(7500), Status: StatusHeadwind},
	}
	allNull := []Lamp{
		emptyLamp(KeyDXY), emptyLamp(KeyRates), emptyLamp(KeyVIX),
		emptyLamp(KeySPX), emptyLamp(KeyGold),
	}

	cases := []struct {
		name      string
		composite *int
		lamps     []Lamp
		want      string
	}{
		{"34 → risk_off", intPtr(34), scored, RegimeRiskOff},
		{"0 → risk_off", intPtr(0), scored, RegimeRiskOff},
		{"35 → mixed (boundary)", intPtr(35), scored, RegimeMixed},
		{"50 → mixed", intPtr(50), scored, RegimeMixed},
		{"65 → mixed (boundary)", intPtr(65), scored, RegimeMixed},
		{"66 → risk_on", intPtr(66), scored, RegimeRiskOn},
		{"100 → risk_on", intPtr(100), scored, RegimeRiskOn},
		// nil composite splits on data presence — the honesty rule:
		// real-but-insufficient lamps stay "mixed", ZERO real lamps are
		// "unknown" (never a knowledge claim off no inputs).
		{"nil + 1 valued lamp → mixed", nil, []Lamp{valuedLamp(KeyDXY, 99), emptyLamp(KeyVIX)}, RegimeMixed},
		{"nil + 2 valued lamps → mixed", nil, []Lamp{valuedLamp(KeyDXY, 99), valuedLamp(KeyVIX, 17)}, RegimeMixed},
		{"nil + all-null lamps → unknown", nil, allNull, RegimeUnknown},
		{"nil + no lamps at all → unknown", nil, nil, RegimeUnknown},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if got := ClassifyRegime(tc.composite, tc.lamps); got != tc.want {
				t.Errorf("ClassifyRegime(%v, lamps) = %q, want %q", tc.composite, got, tc.want)
			}
		})
	}
}

func TestTradfinOK(t *testing.T) {
	if TradfinOK(nil) {
		t.Error("TradfinOK(nil) = true, want false")
	}
	if TradfinOK([]Lamp{emptyLamp(KeyDXY), emptyLamp(KeyVIX)}) {
		t.Error("all-null lamps → true, want false")
	}
	if !TradfinOK([]Lamp{emptyLamp(KeyDXY), valuedLamp(KeyVIX, 17.5)}) {
		t.Error("one valued lamp → false, want true")
	}
	// Robustness over artificial inputs: a status implies data too.
	if !TradfinOK([]Lamp{lamp(KeyDXY, StatusTailwind)}) {
		t.Error("status-only lamp → false, want true")
	}
}

// TestTradfinWindowOpen pins the futures-week approximation to its documented
// boundaries: open from Sunday 22:00 UTC to Friday 21:00 UTC. The fixture
// weekdays are asserted first so a bad date can't silently test nothing.
func TestTradfinWindowOpen(t *testing.T) {
	day := func(y int, m time.Month, d int, wd time.Weekday) time.Time {
		t.Helper()
		dt := time.Date(y, m, d, 0, 0, 0, 0, time.UTC)
		if dt.Weekday() != wd {
			t.Fatalf("fixture %v is a %v, want %v", dt, dt.Weekday(), wd)
		}
		return dt
	}
	sun := day(2026, time.August, 16, time.Sunday)
	fri := day(2026, time.August, 14, time.Friday)
	sat := day(2026, time.August, 15, time.Saturday)
	wed := day(2026, time.August, 12, time.Wednesday)

	at := func(base time.Time, h, m int) time.Time {
		return base.Add(time.Duration(h)*time.Hour + time.Duration(m)*time.Minute)
	}

	cases := []struct {
		name string
		t    time.Time
		want bool
	}{
		{"Sun 21:59 → closed", at(sun, 21, 59), false},
		{"Sun 22:00 → open (boundary)", at(sun, 22, 0), true},
		{"Sun 22:01 → open", at(sun, 22, 1), true},
		{"Fri 20:59 → open", at(fri, 20, 59), true},
		{"Fri 21:00 → closed (boundary)", at(fri, 21, 0), false},
		{"Fri 21:01 → closed", at(fri, 21, 1), false},
		{"Fri 23:59 → closed", at(fri, 23, 59), false},
		{"Sat noon → closed", at(sat, 12, 0), false},
		{"Sun 00:00 → closed", at(sun, 0, 0), false},
		{"Wed noon → open (mid-week)", at(wed, 12, 0), true},
		{"Mon 00:30 → open", at(day(2026, time.August, 17, time.Monday), 0, 30), true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if got := TradfinWindowOpen(tc.t); got != tc.want {
				t.Errorf("TradfinWindowOpen(%v %v) = %v, want %v", tc.t.Weekday(), tc.t.Format("15:04"), got, tc.want)
			}
		})
	}

	// Non-UTC input is normalised: Sun 22:30 UTC expressed in a +02:00 zone
	// (Mon 00:30 local) is still inside the open window.
	loc := time.FixedZone("UTC+2", 2*3600)
	if !TradfinWindowOpen(at(sun, 22, 30).In(loc)) {
		t.Error("zone-shifted Sun 22:30 UTC must be open")
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
		{
			// THE honesty case (team-testing defect): no data → no sentence.
			// "Macro signals are split" must never render off zero real lamps.
			name:      "unknown → empty (no knowledge claim)",
			regime:    RegimeUnknown,
			lamps:     []Lamp{emptyLamp(KeyDXY), emptyLamp(KeyVIX), emptyLamp(KeySPX)},
			wantEmpty: true,
		},
		{
			// Belt-and-braces: even a caller that mislabels the all-null case
			// as "mixed" gets no split-sentence.
			name:      "mixed with zero real lamps → empty",
			regime:    RegimeMixed,
			lamps:     []Lamp{emptyLamp(KeyDXY), emptyLamp(KeyVIX), emptyLamp(KeySPX), emptyLamp(KeyRates), emptyLamp(KeyGold)},
			wantEmpty: true,
		},
		{
			name:      "unknown with real lamps still empty (regime gate wins)",
			regime:    RegimeUnknown,
			lamps:     []Lamp{lamp(KeyDXY, StatusTailwind)},
			wantEmpty: true,
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
	regimes := []string{RegimeRiskOn, RegimeMixed, RegimeRiskOff, RegimeUnknown}
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
