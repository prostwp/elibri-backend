package ml

import "testing"

func TestGetTier(t *testing.T) {
	t.Parallel()

	cases := []struct {
		in   string
		want RiskTier
	}{
		{"conservative", TierConservative},
		{"balanced", TierBalanced},
		{"aggressive", TierAggressive},
		// Case-sensitivity: exact match only.
		{"Conservative", TierBalanced}, // unknown → balanced fallback
		{"CONSERVATIVE", TierBalanced},
		{"", TierBalanced}, // empty → balanced
		{"yolo", TierBalanced}, // unknown named tier → balanced
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			got := GetTier(tc.in)
			// RiskTiers is indexed by RiskTier. Verify we got the
			// corresponding policy struct — compare a known field.
			want := RiskTiers[tc.want]
			if got.MaxTradesPerDay != want.MaxTradesPerDay {
				t.Errorf("tier %q: got MaxTradesPerDay=%d want %d",
					tc.in, got.MaxTradesPerDay, want.MaxTradesPerDay)
			}
		})
	}
}

func TestLabelAllowed(t *testing.T) {
	t.Parallel()

	conservative := RiskTiers[TierConservative]
	balanced := RiskTiers[TierBalanced]
	aggressive := RiskTiers[TierAggressive]

	if !conservative.LabelAllowed("trend_aligned") {
		t.Error("conservative must allow trend_aligned")
	}
	if conservative.LabelAllowed("mean_reversion") {
		t.Error("conservative must NOT allow mean_reversion")
	}
	if conservative.LabelAllowed("random") {
		t.Error("conservative must NOT allow random")
	}

	if !balanced.LabelAllowed("trend_aligned") {
		t.Error("balanced must allow trend_aligned")
	}
	if !balanced.LabelAllowed("mean_reversion") {
		t.Error("balanced must allow mean_reversion")
	}
	if balanced.LabelAllowed("random") {
		t.Error("balanced must NOT allow random")
	}

	if !aggressive.LabelAllowed("trend_aligned") {
		t.Error("aggressive must allow trend_aligned")
	}
	if !aggressive.LabelAllowed("mean_reversion") {
		t.Error("aggressive must allow mean_reversion")
	}
}

// TestRiskTiersMonotonicallyRelaxVolFloor — conservative > balanced > aggressive
// for every TF. Sanity locks in the tier ordering so a future edit that
// accidentally flips Aggressive's floor above Conservative's is caught.
func TestRiskTiersMonotonicallyRelaxVolFloor(t *testing.T) {
	t.Parallel()
	c := RiskTiers[TierConservative].MinVolPctByTF
	b := RiskTiers[TierBalanced].MinVolPctByTF
	a := RiskTiers[TierAggressive].MinVolPctByTF

	for _, tf := range []string{"5m", "15m", "1h", "4h", "1d"} {
		if !(c[tf] >= b[tf] && b[tf] >= a[tf]) {
			t.Errorf("%s floor: c=%v b=%v a=%v (want c >= b >= a)",
				tf, c[tf], b[tf], a[tf])
		}
	}
}

// TestRiskTiersRiskPctOrdering — same monotonic sanity on risk-per-trade.
func TestRiskTiersRiskPctOrdering(t *testing.T) {
	t.Parallel()
	c := RiskTiers[TierConservative].RiskPerTradePct
	b := RiskTiers[TierBalanced].RiskPerTradePct
	a := RiskTiers[TierAggressive].RiskPerTradePct
	if !(c < b && b < a) {
		t.Errorf("risk per trade: c=%v b=%v a=%v (want c < b < a)", c, b, a)
	}
}
