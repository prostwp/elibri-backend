package ml

// Patch 2C: risk layer — pure config/policy, zero ML retraining.
// Tiers gate prediction output post-inference: low-vol rejection,
// confidence thresholds, label allow-lists, and sizing hints.

type RiskTier string

const (
	TierConservative RiskTier = "conservative"
	TierBalanced     RiskTier = "balanced"
	TierAggressive   RiskTier = "aggressive"
)

// TierPolicy is the decision envelope applied on top of raw model output.
// All fields optional: zero values mean "no gate" / "use defaults".
//
// Patch 2E: removed MinConfidence to eliminate the double-gate with HC
// threshold (best_thresholds.json per-TF). HC threshold is now the single
// source of truth for confidence — tier only governs vol, labels, rate
// limit, and sizing.
type TierPolicy struct {
	// MinVolPctByTF is the minimum atr_norm_14 (feature[12]) required per
	// interval. Signal is blocked when realized vol falls below floor.
	MinVolPctByTF map[string]float64
	// MaxTradesPerDay is a soft cap. Enforcement lives in paper-trade layer;
	// exposed here so frontend can surface the budget.
	MaxTradesPerDay int
	// AllowedLabels restricts which signal-classification labels are
	// actionable for this tier. Labels outside the set flip direction
	// to "neutral" with Blocked=true.
	AllowedLabels []string
	// RiskPerTradePct is the fraction of equity risked per trade — used
	// by the frontend position-size calculator (not by the backend).
	RiskPerTradePct float64
	// SLAtrMult / TPAtrMult are ATR multipliers suggested for SL/TP
	// placement. Frontend composes final price targets.
	SLAtrMult float64
	TPAtrMult float64
}

// RiskTiers holds the three production tiers. Values below were tuned to
// match the "safe start / balanced / yolo" UX segments in frontend.
var RiskTiers = map[RiskTier]TierPolicy{
	TierConservative: {
		MinVolPctByTF: map[string]float64{
			"5m":  0.008,
			"15m": 0.010,
			"1h":  0.015,
			"4h":  0.020,
			"1d":  0.025,
		},
		MaxTradesPerDay: 3,
		AllowedLabels:   []string{"trend_aligned"},
		RiskPerTradePct: 0.0025, // 0.25%
		SLAtrMult:       1.5,
		TPAtrMult:       2.5,
	},
	TierBalanced: {
		MinVolPctByTF: map[string]float64{
			"5m":  0.005,
			"15m": 0.007,
			"1h":  0.010,
			"4h":  0.015,
			"1d":  0.020,
		},
		MaxTradesPerDay: 7,
		AllowedLabels:   []string{"trend_aligned", "mean_reversion"},
		RiskPerTradePct: 0.005, // 0.5%
		SLAtrMult:       1.5,
		TPAtrMult:       2.5,
	},
	TierAggressive: {
		MinVolPctByTF: map[string]float64{
			"5m":  0.0025,
			"15m": 0.004,
			"1h":  0.006,
			"4h":  0.010,
			"1d":  0.015,
		},
		MaxTradesPerDay: 20,
		AllowedLabels:   []string{"trend_aligned", "mean_reversion"},
		RiskPerTradePct: 0.01, // 1.0%
		SLAtrMult:       1.2,
		TPAtrMult:       2.0,
	},
}

// GetTier returns the policy for a tier string, falling back to Balanced
// when unknown/empty. Never returns a nil policy.
func GetTier(t string) TierPolicy {
	tier := RiskTier(t)
	if policy, ok := RiskTiers[tier]; ok {
		return policy
	}
	return RiskTiers[TierBalanced]
}

// IsValidTier reports whether t names a known tier. Used for input
// validation on /auth/me (PATCH).
func IsValidTier(t string) bool {
	_, ok := RiskTiers[RiskTier(t)]
	return ok
}

// LabelAllowed reports whether a classification label is actionable
// under the given policy. Empty allow-list means "no label gate".
func (p TierPolicy) LabelAllowed(label string) bool {
	if len(p.AllowedLabels) == 0 {
		return true
	}
	for _, l := range p.AllowedLabels {
		if l == label {
			return true
		}
	}
	return false
}
