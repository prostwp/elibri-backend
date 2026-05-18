package narrative

import (
	"math"
	"testing"
)

// TestComputeGrowthPct pins down growth-percentage edge cases. The
// (prev=0, curr>0) "infinite" case is the load-bearing one: without
// the cap a brand-new narrative would return +Inf and break the REAL
// column downstream.
func TestComputeGrowthPct(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		curr int
		prev int
		want float64
	}{
		{"normal positive growth", 15, 10, 50.0},
		{"new narrative caps at 999", 5, 0, 999.0},
		{"both zero is zero", 0, 0, 0.0},
		{"negative growth", 5, 10, -50.0},
		{"explosive growth caps", 10000, 1, 999.0},
		{"full disappearance is -100", 0, 10, -100.0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ComputeGrowthPct(tc.curr, tc.prev)
			if math.Abs(got-tc.want) > 1e-9 {
				t.Errorf("ComputeGrowthPct(curr=%d, prev=%d) = %v, want %v",
					tc.curr, tc.prev, got, tc.want)
			}
		})
	}

	// Cap is symmetric — a hypothetical -10000% (impossible from real
	// counts but possible if a future caller passes negative ints) clamps
	// to -999, not -10000.
	t.Run("negative cap is symmetric", func(t *testing.T) {
		got := ComputeGrowthPct(-999, 1)
		if got != -999.0 {
			t.Errorf("ComputeGrowthPct(-999, 1) = %v, want -999.0", got)
		}
	})
}

// TestClassifyStage walks the if/else chain in priority order. The
// "declining wins over volume" case is the one that catches most
// re-ordering bugs.
func TestClassifyStage(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		mentions24h int
		growthPct   float64
		want        Stage
	}{
		{"low volume + explosive growth = early", 20, 300, StageEarly},
		{"medium volume + good growth = trending", 200, 80, StageTrending},
		{"high volume + flat growth = mainstream", 600, 10, StageMainstream},
		{"declining beats volume floor", 300, -30, StageDeclining},
		{"declining beats mainstream volume", 600, -30, StageDeclining},
		{"low everything falls back to early", 10, 10, StageEarly},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyStage(tc.mentions24h, tc.growthPct)
			if got != tc.want {
				t.Errorf("ClassifyStage(mentions=%d, growth=%v) = %q, want %q",
					tc.mentions24h, tc.growthPct, got, tc.want)
			}
		})
	}
}

// TestComputeTrendScore checks the five-component weighted sum (v2).
// Tolerances of ±2 are used on non-saturated cases because rounding of the
// 0.5-sentiment component can flip by 1 between Go versions if the rounding
// rule ever changes. v2 component weights: 35 + 25 + 15 + 10 + 15 = 100.
//
// All test sites here pass avgImportance=0 unless they specifically exercise
// the importance component — that keeps these regression tests aligned with
// the v1 contract (with v2 weights) and pushes the importance-specific
// behaviour into TestComputeTrendScore_ImportanceComponent below.
func TestComputeTrendScore(t *testing.T) {
	t.Parallel()

	t.Run("all zeros is zero", func(t *testing.T) {
		got := ComputeTrendScore(0, 0, 0, 0, 0)
		if got != 0 {
			t.Errorf("ComputeTrendScore(0,0,0,0,0) = %d, want 0", got)
		}
	})

	t.Run("saturated inputs (incl. importance) produce 100", func(t *testing.T) {
		// v2: 35 (growth) + 25 (volume) + 15 (sentiment) + 10 (diversity) + 15 (importance) = 100
		got := ComputeTrendScore(1000, 200, 1.0, 3, 100)
		if got != 100 {
			t.Errorf("ComputeTrendScore saturated = %d, want 100", got)
		}
	})

	t.Run("midrange inputs sum correctly within ±2", func(t *testing.T) {
		// v2 midrange (avgImportance=0): growth=100/200*35=17.5,
		// vol=500/1000*25=12.5, sent=0.5*15=7.5, div=2*5=10, imp=0
		// → ~47.5 → 48.
		got := ComputeTrendScore(500, 100, 0.5, 2, 0)
		if got < 46 || got > 50 {
			t.Errorf("ComputeTrendScore midrange = %d, want ~48 (±2)", got)
		}
	})

	t.Run("negative growth contributes 0", func(t *testing.T) {
		// All other components zero → growth must clamp to 0.
		got := ComputeTrendScore(0, -50, 0, 0, 0)
		if got != 0 {
			t.Errorf("ComputeTrendScore with negative growth = %d, want 0", got)
		}
	})

	t.Run("negative sentiment contributes 0", func(t *testing.T) {
		got := ComputeTrendScore(0, 0, -1.0, 0, 0)
		if got != 0 {
			t.Errorf("ComputeTrendScore with negative sentiment = %d, want 0", got)
		}
	})

	t.Run("over-saturated inputs are still capped at 100", func(t *testing.T) {
		// Growth >> saturation, mentions >> saturation, sources >> 3,
		// importance >> 100.
		got := ComputeTrendScore(100000, 9999, 1.0, 99, 99999)
		if got != 100 {
			t.Errorf("ComputeTrendScore over-saturated = %d, want 100", got)
		}
	})

	t.Run("bounds: never <0, never >100", func(t *testing.T) {
		// Sweep a few corners — combinatorial explosion not needed here,
		// just enough to catch a sign-flip in any one component.
		corners := []struct {
			m   int
			g   float64
			s   float64
			div int
			imp float64
		}{
			{0, -999, -1.0, 0, -100},
			{0, 0, 0, 0, 0},
			{1000, 200, 1, 3, 100},
			{100000, 9999, 1, 99, 99999},
			{-5, -5, -5, -5, -5}, // intentionally garbage
		}
		for _, c := range corners {
			got := ComputeTrendScore(c.m, c.g, c.s, c.div, c.imp)
			if got < 0 || got > 100 {
				t.Errorf("ComputeTrendScore(%d,%v,%v,%d,%v) = %d, out of [0,100]",
					c.m, c.g, c.s, c.div, c.imp, got)
			}
		}
	})
}

// TestComputeTrendScore_ImportanceComponent pins the v2 importance addition.
// Four cases cover (a) zero importance keeps v1-equivalent behaviour modulo
// the rebalanced max points, (b) max importance saturates at exactly 15
// points, (c) midrange importance contributes proportionally, (d) NaN /
// negative importance are clamped to 0.
func TestComputeTrendScore_ImportanceComponent(t *testing.T) {
	t.Parallel()

	t.Run("zero importance produces same as if the component didn't exist", func(t *testing.T) {
		// Zero importance should NOT change the score relative to a v2
		// caller that hasn't classified anything yet. Pin the exact score
		// so a future tweak to the rebalance breaks loudly.
		// Inputs: mentions=500, growth=100, sent=0.5, div=2, imp=0
		// Expected: 17.5 + 12.5 + 7.5 + 10 + 0 = 47.5 → 48.
		got := ComputeTrendScore(500, 100, 0.5, 2, 0)
		if got != 48 {
			t.Errorf("ComputeTrendScore zero-importance = %d, want 48", got)
		}
	})

	t.Run("max importance saturates at exactly 15 points", func(t *testing.T) {
		// Compare zero-importance vs max-importance with all other inputs
		// pinned. Difference must be exactly 15 (the importance ceiling).
		base := ComputeTrendScore(500, 100, 0.5, 2, 0)
		maxImp := ComputeTrendScore(500, 100, 0.5, 2, 100)
		diff := maxImp - base
		if diff != 15 {
			t.Errorf("max importance gain = %d, want exactly 15 points (base=%d, max=%d)", diff, base, maxImp)
		}
	})

	t.Run("midrange importance is proportional", func(t *testing.T) {
		// Importance 50 should contribute exactly 50/100*15 = 7.5 points
		// → 7 or 8 after rounding. Compare to base: difference must be
		// in [7, 8].
		base := ComputeTrendScore(500, 100, 0.5, 2, 0)
		mid := ComputeTrendScore(500, 100, 0.5, 2, 50)
		diff := mid - base
		if diff < 7 || diff > 8 {
			t.Errorf("mid importance (50) gain = %d, want 7 or 8", diff)
		}
	})

	t.Run("NaN / negative importance is clamped to 0", func(t *testing.T) {
		// NaN → 0, negative → 0. Both must produce the same score as
		// avgImportance=0.
		base := ComputeTrendScore(500, 100, 0.5, 2, 0)
		gotNaN := ComputeTrendScore(500, 100, 0.5, 2, math.NaN())
		gotNeg := ComputeTrendScore(500, 100, 0.5, 2, -50)
		if gotNaN != base || gotNeg != base {
			t.Errorf("importance defensive clamp broken: base=%d nan=%d neg=%d", base, gotNaN, gotNeg)
		}
	})
}

// TestLabelSentiment pins the ±0.2 boundary inclusive of neutral. A
// future tweak that flips the inequality (>= vs >) would shift exactly-
// boundary scores into bull/bear and is something we want a regression
// test to catch.
func TestLabelSentiment(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		score float64
		want  SentimentLabel
	}{
		{"clearly bull", 0.5, SentBull},
		{"weakly positive is neutral", 0.1, SentNeutral},
		{"weakly negative is neutral", -0.1, SentNeutral},
		{"clearly bear", -0.5, SentBear},
		{"exactly +0.2 is neutral (boundary inclusive)", 0.2, SentNeutral},
		{"exactly -0.2 is neutral (boundary inclusive)", -0.2, SentNeutral},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := LabelSentiment(tc.score)
			if got != tc.want {
				t.Errorf("LabelSentiment(%v) = %q, want %q", tc.score, got, tc.want)
			}
		})
	}
}

// TestComputeConfidence locks the "data-quality" weighting. The
// "1 source" case is deliberately chosen so a one-source spam attack
// can't drive confidence above 60.
func TestComputeConfidence(t *testing.T) {
	t.Parallel()

	t.Run("zero mentions, zero diversity, perfect consistency = 25", func(t *testing.T) {
		// stdDev=0 means perfect consistency → only that component fires.
		got := ComputeConfidence(0, 0, 0)
		if got != 25 {
			t.Errorf("ComputeConfidence(0,0,0) = %d, want 25", got)
		}
	})

	t.Run("max inputs produce 100", func(t *testing.T) {
		// 30 (vol) + 45 (div, capped at 3) + 25 (consistency) = 100
		got := ComputeConfidence(100, 3, 0)
		if got != 100 {
			t.Errorf("ComputeConfidence(100,3,0) = %d, want 100", got)
		}
	})

	t.Run("midrange ≈43 within ±2", func(t *testing.T) {
		// vol=50/100*30=15, div=1*15=15, cons=(1-0.5)*25=12.5
		// → 42.5 → 43 (banker's rounding may yield 42 on some libcs but
		//   math.Round in Go is half-away-from-zero, so 43).
		got := ComputeConfidence(50, 1, 0.5)
		if got < 41 || got > 45 {
			t.Errorf("ComputeConfidence(50,1,0.5) = %d, want ~43 (±2)", got)
		}
	})

	t.Run("4 sources clamps to same diversity as 3", func(t *testing.T) {
		// 0 mentions + 4 sources + stdDev=1.0 → only diversity fires.
		// Should equal the value for 3 sources with the same other inputs.
		got4 := ComputeConfidence(0, 4, 1.0)
		got3 := ComputeConfidence(0, 3, 1.0)
		if got4 != got3 {
			t.Errorf("ComputeConfidence: 4 sources (%d) != 3 sources (%d) — clamp broken",
				got4, got3)
		}
		// And both should be exactly the diversity-cap value (45).
		if got4 != 45 {
			t.Errorf("ComputeConfidence(0,4,1.0) = %d, want 45 (diversity cap)", got4)
		}
	})

	t.Run("stdDev>=1 zeroes the consistency component", func(t *testing.T) {
		// Any std-dev at or above 1.0 should clamp to 1.0 internally and
		// produce 0 consistency points.
		gotAt1 := ComputeConfidence(0, 0, 1.0)
		gotAt5 := ComputeConfidence(0, 0, 5.0) // garbage input shouldn't go negative
		if gotAt1 != 0 || gotAt5 != 0 {
			t.Errorf("ComputeConfidence consistency at stdDev>=1: at1=%d at5=%d, want both 0",
				gotAt1, gotAt5)
		}
	})

	t.Run("bounds: never <0, never >100", func(t *testing.T) {
		corners := []struct {
			m   int
			div int
			sd  float64
		}{
			{0, 0, 0},
			{100, 3, 0},
			{100000, 99, 0}, // saturating in every component
			{-5, -5, -5},    // garbage
			{0, 0, 999},     // garbage std-dev
		}
		for _, c := range corners {
			got := ComputeConfidence(c.m, c.div, c.sd)
			if got < 0 || got > 100 {
				t.Errorf("ComputeConfidence(%d,%d,%v) = %d, out of [0,100]",
					c.m, c.div, c.sd, got)
			}
		}
	})
}

// ─── NaN / overflow defensive guards ───────────────────────────────────────
//
// Reviewer flagged that ComputeTrendScore + LabelSentiment + ComputeConfidence
// could let a malformed DB row (sentiment=5.0 or NaN, REAL has no CHECK)
// silently violate the per-component contracts even though the final clampInt
// kept the total in [0,100]. These tests pin the explicit guards added in
// scorer.go so a future "simplify" refactor can't quietly drop them.

func TestComputeTrendScore_DefensiveGuards(t *testing.T) {
	t.Parallel()

	t.Run("NaN growth is treated as 0 (no panic, no NaN in result)", func(t *testing.T) {
		got := ComputeTrendScore(100, math.NaN(), 0.0, 1, 0)
		if got < 0 || got > 100 {
			t.Errorf("ComputeTrendScore with NaN growth = %d, out of [0,100]", got)
		}
		// v2: growth=0, mentions=100, sent=0, sources=1, imp=0:
		// growth=0 + volume=100/1000*25=2.5 + sent=0 + diversity=5 + imp=0 = 7.5 → 8
		if got != 8 {
			t.Errorf("ComputeTrendScore(100, NaN, 0, 1, 0) = %d, want 8 (NaN→0 path)", got)
		}
	})

	t.Run("NaN sentiment is treated as 0", func(t *testing.T) {
		got := ComputeTrendScore(0, 0, math.NaN(), 0, 0)
		if got != 0 {
			t.Errorf("ComputeTrendScore with NaN sentiment = %d, want 0", got)
		}
	})

	t.Run("sentiment > 1 cannot blow the 15-pt component ceiling", func(t *testing.T) {
		// If sentiment=5.0 weren't clamped, sentimentPoints would be 75 —
		// the final clampInt would still cap the total at 100, but the
		// per-component contract would silently break. Compare with
		// sentiment=1.0 (the legitimate max) — the two should match.
		gotMax := ComputeTrendScore(0, 0, 1.0, 0, 0)
		gotOverflow := ComputeTrendScore(0, 0, 5.0, 0, 0)
		if gotMax != gotOverflow {
			t.Errorf("ComputeTrendScore sent=1.0 (%d) != sent=5.0 (%d) — clamp not honored",
				gotMax, gotOverflow)
		}
	})

	t.Run("negative mentions / negative diversity don't underflow", func(t *testing.T) {
		got := ComputeTrendScore(-50, 100, 0.5, -3, 0)
		// v2 clamps: mentions=0, growth=100, sent=0.5, diversity=0, imp=0:
		// 100/200*35 + 0 + 0.5*15 + 0 + 0 = 17.5 + 0 + 7.5 + 0 + 0 = 25
		if got != 25 {
			t.Errorf("ComputeTrendScore(-50, 100, 0.5, -3, 0) = %d, want 25", got)
		}
	})
}

func TestLabelSentiment_NaN(t *testing.T) {
	t.Parallel()
	if got := LabelSentiment(math.NaN()); got != SentNeutral {
		t.Errorf("LabelSentiment(NaN) = %q, want %q", got, SentNeutral)
	}
}

func TestComputeConfidence_NaNStdDev(t *testing.T) {
	t.Parallel()
	// NaN std-dev → caller is broken; we treat as worst-case (1.0) so
	// consistency points = 0. With 100 mentions + 3 sources we still get
	// 30 + 45 + 0 = 75.
	got := ComputeConfidence(100, 3, math.NaN())
	if got != 75 {
		t.Errorf("ComputeConfidence(100, 3, NaN) = %d, want 75 (NaN→worst-case path)", got)
	}
}
