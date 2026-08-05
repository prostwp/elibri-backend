package macro

// compute.go — pure, deterministic functions (heavy-tested in compute_test.go,
// like whale/scorer.go). No I/O. Every function here is total and side-effect-
// free so the unit tests can hit 100% of the branches offline.

import (
	"math"
	"regexp"
)

const (
	// strongPct is the |session delta| (in %) above which a directional lamp move is
	// "strong" enough to flip out of neutral (DXY/SPX/Gold/Rates). Tuned in
	// discovery §7; a single const so the rule is one-line editable.
	strongPct = 0.5

	// vixTailwindBelow / vixHeadwindAbove are absolute VIX levels (not deltas):
	// calm tape (<18) is a tailwind for crypto, fearful tape (>25) a headwind.
	vixTailwindBelow = 18.0
	vixHeadwindAbove = 25.0

	// Composite lamp weights (discovery §7). Sum = 100. Renormalised over the
	// ACTIVE (non-N/D) lamps so a missing market doesn't silently zero a slice.
	weightDXY   = 25.0
	weightVIX   = 25.0
	weightSPX   = 25.0
	weightRates = 15.0
	weightGold  = 10.0

	// minActiveLamps — below this many live lamps the composite is unstable and
	// we return nil (regime → mixed; handler shows "Not enough live markets").
	minActiveLamps = 3

	// regimeRiskOffBelow / regimeRiskOnAbove bracket the mixed band on composite.
	regimeRiskOffBelow = 35
	regimeRiskOnAbove  = 65
)

// bannedDiagnosisPattern mirrors the frontend isIdeaSafe() / DESIGN_SYSTEM §8
// banned table. BuildDiagnosis output is asserted against it in compute_test.go
// — "support" is the headline gotcha (dnevnik 2026-05-30 15:05). Word-boundary
// matched, case-insensitive. "favors"/"tailwind"/"headwind"/"softening"/"bid"/
// "weighs"/"pressures" are all allowed (none appear here).
var bannedDiagnosisPattern = regexp.MustCompile(
	`(?i)\b(buy|sell|long|short|entry|exit|target|stop[ -]?loss|take[ -]?profit|trading signal|signal|support|resistance|breakout|breakdown|position|momentum|traders? should|sharpe)\b`,
)

// IsDiagnosisSafe reports whether s contains zero banned trade-lexicon words.
// The handler/test use it as the final tripwire before emitting generated_idea.
func IsDiagnosisSafe(s string) bool {
	return !bannedDiagnosisPattern.MatchString(s)
}

// Pearson returns the correlation coefficient of xs,ys (same length). Returns
// nil when there are <2 points OR either series has zero variance (guards /0 —
// on weekends every tradfin point is the same Friday close, so the dollar/SPX
// series go flat and the correlation is genuinely undefined). The result is
// clamped to [-1,+1] to absorb float drift past the boundary.
func Pearson(xs, ys []float64) *float64 {
	if len(xs) != len(ys) || len(xs) < 2 {
		return nil
	}
	n := float64(len(xs))

	var sumX, sumY float64
	for i := range xs {
		sumX += xs[i]
		sumY += ys[i]
	}
	meanX := sumX / n
	meanY := sumY / n

	var covXY, varX, varY float64
	for i := range xs {
		dx := xs[i] - meanX
		dy := ys[i] - meanY
		covXY += dx * dy
		varX += dx * dx
		varY += dy * dy
	}
	if varX == 0 || varY == 0 {
		// Zero variance → undefined correlation (degenerate on weekends).
		return nil
	}

	c := covXY / math.Sqrt(varX*varY)
	if math.IsNaN(c) || math.IsInf(c, 0) {
		return nil
	}
	// Clamp float drift back into the valid range.
	if c > 1 {
		c = 1
	} else if c < -1 {
		c = -1
	}
	return &c
}

// LampStatus maps a lamp's value + 24h delta to tailwind|neutral|headwind.
// Rules (discovery §7 — one-line editable consts above). delta24h is in % (e.g.
// +0.5 means +0.5%). For VIX the absolute level matters more than the delta.
//
//	dxy:   delta<0 → tailwind | delta>+0.5 → headwind | else neutral
//	rates: delta<0 → tailwind | delta>+0.5 → headwind | else neutral  (yield ↓ = easier for crypto)
//	spx:   delta>0 → tailwind | delta<-0.5 → headwind | else neutral
//	gold:  delta<0 → tailwind | delta>+0.5 → headwind | else neutral  (money leaving gold = risk-on)
//	vix:   value<18 → tailwind | value>25 → headwind | else neutral   (level beats delta)
func LampStatus(key string, value float64, delta24h float64) string {
	switch key {
	case KeyDXY, KeyRates, KeyGold:
		// A firmer dollar / higher yields / a bid for gold all weigh on crypto.
		if delta24h < 0 {
			return StatusTailwind
		}
		if delta24h > strongPct {
			return StatusHeadwind
		}
		return StatusNeutral
	case KeySPX:
		// Equities up = risk appetite = tailwind for crypto.
		if delta24h > 0 {
			return StatusTailwind
		}
		if delta24h < -strongPct {
			return StatusHeadwind
		}
		return StatusNeutral
	case KeyVIX:
		// Calm tape favours crypto; fear weighs on it. Level, not delta.
		if value < vixTailwindBelow {
			return StatusTailwind
		}
		if value > vixHeadwindAbove {
			return StatusHeadwind
		}
		return StatusNeutral
	default:
		return StatusNeutral
	}
}

// lampWeight returns the composite weight for a lamp key (0 for unknown keys).
func lampWeight(key string) float64 {
	switch key {
	case KeyDXY:
		return weightDXY
	case KeyVIX:
		return weightVIX
	case KeySPX:
		return weightSPX
	case KeyRates:
		return weightRates
	case KeyGold:
		return weightGold
	default:
		return 0
	}
}

// Composite folds the 5 lamp statuses into a 0..100 score. Each ACTIVE lamp
// (Status != "" → it has live data) contributes:
//
//	tailwind → +weight | neutral → +weight/2 | headwind → +0
//
// The weights of the active lamps are renormalised, so dropping one N/D market
// rescales the rest rather than capping the max below 100:
//
//	composite = round( sum(score) / sum(activeWeights) * 100 )
//
// If fewer than minActiveLamps (3) lamps are live, the composite is unstable →
// returns nil (handler shows "Not enough live markets", regime → mixed). Never
// fabricates a default 50.
func Composite(lamps []Lamp) *int {
	var scoreSum, weightSum float64
	active := 0
	for _, l := range lamps {
		if l.Status == "" {
			continue // N/D lamp — excluded from the score + renormalisation.
		}
		w := lampWeight(l.Key)
		if w == 0 {
			continue
		}
		active++
		weightSum += w
		switch l.Status {
		case StatusTailwind:
			scoreSum += w
		case StatusNeutral:
			scoreSum += w / 2
		case StatusHeadwind:
			// +0
		}
	}
	if active < minActiveLamps || weightSum == 0 {
		return nil
	}
	v := int(math.Round(scoreSum / weightSum * 100))
	if v < 0 {
		v = 0
	} else if v > 100 {
		v = 100
	}
	return &v
}

// ClassifyRegime buckets a composite into a regime. A nil composite → mixed
// (the handler emits the "not enough live markets" copy separately).
//
//	<35 → risk_off | 35..65 → mixed | >65 → risk_on
func ClassifyRegime(composite *int) string {
	if composite == nil {
		return RegimeMixed
	}
	c := *composite
	if c < regimeRiskOffBelow {
		return RegimeRiskOff
	}
	if c > regimeRiskOnAbove {
		return RegimeRiskOn
	}
	return RegimeMixed
}

// lampByKey returns the status of a lamp by key ("" if absent or N/D).
func lampStatusByKey(lamps []Lamp, key string) string {
	for _, l := range lamps {
		if l.Key == key {
			return l.Status
		}
	}
	return ""
}

// BuildDiagnosis assembles a rule-based, safe, templated sentence from the
// regime + lamp signs. NO LLM. The 6 templates below are written with ZERO
// banned words (no buy/sell/long/short/entry/target/support/resistance/
// breakout/momentum/position/signal) — "favors"/"softening"/"bid"/"weighs"/
// "tailwind"/"headwind" are fine. Returns "" if the assembled string somehow
// trips the banned filter (graceful — the frontend shows its own empty state),
// and "" when the composite is nil (≥3 lamps N/D → no honest read to give).
//
// Selector: regime × the dominant lamp's sign. Any uncovered mixed case falls
// back to template #3.
func BuildDiagnosis(regime string, lamps []Lamp) string {
	dxy := lampStatusByKey(lamps, KeyDXY)
	vix := lampStatusByKey(lamps, KeyVIX)
	spx := lampStatusByKey(lamps, KeySPX)
	gold := lampStatusByKey(lamps, KeyGold)

	var out string
	switch regime {
	case RegimeRiskOn:
		switch {
		case dxy == StatusTailwind && vix == StatusTailwind:
			// Template #1.
			out = "The dollar is softening and volatility is low — a risk-on backdrop that tends to favor crypto."
		case spx == StatusTailwind:
			// Template #4.
			out = "Equities are bid and the macro tape leans risk-on, which has historically been a tailwind for crypto."
		default:
			// Generic risk-on (still safe, no banned words).
			out = "The big-money markets are leaning risk-on, a backdrop that tends to favor crypto."
		}
	case RegimeRiskOff:
		switch {
		case dxy == StatusHeadwind && vix == StatusHeadwind:
			// Template #2.
			out = "A firmer dollar and rising fear point to a risk-off backdrop — a headwind for crypto."
		case gold == StatusHeadwind:
			// Template #5 (flight to gold + dollar).
			out = "Money is rotating into gold and the dollar — a flight-to-safety tone that weighs on crypto."
		default:
			// Generic risk-off (still safe).
			out = "The big-money markets are leaning risk-off, a backdrop that weighs on crypto."
		}
	default:
		// Template #3 — mixed / anything uncovered.
		out = "Macro signals are split right now — no single regime is in control across the big-money markets."
	}

	// Final tripwire: if a template ever drifts into banned territory, drop it
	// rather than emit something the frontend would silently blank.
	if !IsDiagnosisSafe(out) {
		return ""
	}
	return out
}

// CorrelationLabel maps a coefficient to a human label for a given pair. A nil
// coef → "" (UI shows "Building correlation window"). Thresholds: |c|<0.2 barely
// linked; strong positive/negative tracked per pair.
func CorrelationLabel(pair string, coef *float64) string {
	if coef == nil {
		return ""
	}
	c := *coef
	abs := math.Abs(c)

	if abs < 0.2 {
		return "barely linked"
	}

	switch pair {
	case PairBTCSPX:
		if c >= 0.6 {
			return "moving like stocks"
		}
		if c <= -0.6 {
			return "moving against stocks"
		}
		if c > 0 {
			return "loosely tracking stocks"
		}
		return "loosely diverging from stocks"
	case PairBTCGold:
		if c >= 0.6 {
			return "moving with gold"
		}
		if c <= -0.6 {
			return "moving against gold"
		}
		if c > 0 {
			return "loosely tracking gold"
		}
		return "loosely diverging from gold"
	case PairBTCDXY:
		if c <= -0.6 {
			return "inverse to the dollar"
		}
		if c >= 0.6 {
			return "tracking the dollar"
		}
		if c < 0 {
			return "loosely inverse to the dollar"
		}
		return "loosely tracking the dollar"
	default:
		return "loosely linked"
	}
}
