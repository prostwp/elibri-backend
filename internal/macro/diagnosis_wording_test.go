package macro

import (
	"strings"
	"testing"
)

// The diagnosis line ships to the site's macro page. It must not claim things
// the pipeline never observes: nobody's positions ("big money"), no measured
// history ("historically"), and — since 2026-09-15 — no cause or effect at all.
// The lamps are five prices scored by a fixed rule; nothing here measures
// money flows, a haven bid, or what crypto does afterwards. Every template
// branch is exercised.

// causalPhrases are the wordings the 2026-09-15 plan removed ("tends to favor
// crypto", "unwinds the haven bid", "flight to safety", …) plus their near
// variants. Lower-case substring match.
var causalPhrases = []string{
	"big money", "big-money", "historically",
	"favor", "favour", "tends to", "pulls", "pulling", "unwind", "haven",
	"flight", "lifts", "weighs", "rotating", "money", "for crypto",
	"tailwind", "headwind", "backdrop that", "will ", "24h",
}

func diagnosisCases() []struct {
	regime string
	lamps  []Lamp
} {
	lamp := func(key, status string) Lamp {
		v := 1.0
		return Lamp{Key: key, Value: &v, OK: true, Status: status}
	}
	return []struct {
		regime string
		lamps  []Lamp
	}{
		{RegimeRiskOn, []Lamp{lamp(KeyDXY, StatusTailwind), lamp(KeyVIX, StatusTailwind)}},
		{RegimeRiskOn, []Lamp{lamp(KeySPX, StatusTailwind)}},
		{RegimeRiskOn, []Lamp{lamp(KeyRates, StatusTailwind)}},
		{RegimeRiskOff, []Lamp{lamp(KeyDXY, StatusHeadwind), lamp(KeyVIX, StatusHeadwind)}},
		{RegimeRiskOff, []Lamp{lamp(KeyGold, StatusHeadwind)}},
		{RegimeRiskOff, []Lamp{lamp(KeyRates, StatusHeadwind)}},
		{RegimeMixed, []Lamp{lamp(KeySPX, StatusNeutral)}},
	}
}

func TestBuildDiagnosisMakesNoUnbackedClaims(t *testing.T) {
	for _, tc := range diagnosisCases() {
		out := strings.ToLower(BuildDiagnosis(tc.regime, tc.lamps))
		if out == "" {
			t.Errorf("%s %v: empty diagnosis — the branch was not exercised", tc.regime, tc.lamps)
		}
		for _, banned := range causalPhrases {
			if strings.Contains(out, banned) {
				t.Errorf("%s %v: %q contains %q", tc.regime, tc.lamps, out, banned)
			}
		}
	}
}

// Exhaustive: every regime × every status combination of the five lamps —
// no reachable template may carry a causal phrase or a banned trade word.
func TestBuildDiagnosisNoCausalPhraseEver(t *testing.T) {
	statuses := []string{StatusTailwind, StatusNeutral, StatusHeadwind, ""}
	regimes := []string{RegimeRiskOn, RegimeMixed, RegimeRiskOff, RegimeUnknown}
	v := 1.0
	mk := func(key, s string) Lamp { return Lamp{Key: key, Value: &v, OK: true, Status: s} }
	seen := map[string]bool{}
	for _, regime := range regimes {
		for _, a := range statuses {
			for _, b := range statuses {
				for _, c := range statuses {
					for _, d := range statuses {
						for _, e := range statuses {
							lamps := []Lamp{mk(KeyDXY, a), mk(KeyRates, b), mk(KeyVIX, c), mk(KeySPX, d), mk(KeyGold, e)}
							out := BuildDiagnosis(regime, lamps)
							if out == "" || seen[out] {
								continue
							}
							seen[out] = true
							low := strings.ToLower(out)
							for _, banned := range causalPhrases {
								if strings.Contains(low, banned) {
									t.Errorf("regime %s: %q contains %q", regime, out, banned)
								}
							}
							if !IsDiagnosisSafe(out) {
								t.Errorf("regime %s: %q trips the banned trade-word filter", regime, out)
							}
						}
					}
				}
			}
		}
	}
	if len(seen) != 7 {
		t.Errorf("reached %d distinct templates, want all 7: %v", len(seen), seen)
	}
}

// The thresholds a template names are the rule's own constants.
func TestBuildDiagnosisPrintsRuleConstants(t *testing.T) {
	v := 1.0
	mk := func(key, s string) Lamp { return Lamp{Key: key, Value: &v, OK: true, Status: s} }
	on := BuildDiagnosis(RegimeRiskOn, []Lamp{mk(KeyDXY, StatusTailwind), mk(KeyVIX, StatusTailwind)})
	if !strings.Contains(on, "VIX is below 18") {
		t.Errorf("risk-on template must print the VIX calm threshold: %q", on)
	}
	off := BuildDiagnosis(RegimeRiskOff, []Lamp{mk(KeyDXY, StatusHeadwind), mk(KeyVIX, StatusHeadwind)})
	if !strings.Contains(off, "more than 0.5%") || !strings.Contains(off, "VIX is above 25") {
		t.Errorf("risk-off template must print the strong-move and VIX fear thresholds: %q", off)
	}
	gen := BuildDiagnosis(RegimeRiskOn, []Lamp{mk(KeyRates, StatusTailwind)})
	if !strings.Contains(gen, "score above 65") {
		t.Errorf("generic risk-on must print the band: %q", gen)
	}
}
