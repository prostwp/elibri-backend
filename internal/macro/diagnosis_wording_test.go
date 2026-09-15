package macro

import (
	"strings"
	"testing"
)

// The diagnosis line ships to the demobot macro card. It must not claim
// things the pipeline never observes: nobody's positions ("big money") and no
// measured history ("historically"). Every template branch is exercised.
func TestBuildDiagnosisMakesNoUnbackedClaims(t *testing.T) {
	lamp := func(key, status string) Lamp {
		v := 1.0
		return Lamp{Key: key, Value: &v, OK: true, Status: status}
	}
	cases := []struct {
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
	for _, tc := range cases {
		out := strings.ToLower(BuildDiagnosis(tc.regime, tc.lamps))
		if out == "" {
			t.Errorf("%s %v: empty diagnosis — the branch was not exercised", tc.regime, tc.lamps)
		}
		for _, banned := range []string{"big money", "big-money", "historically"} {
			if strings.Contains(out, banned) {
				t.Errorf("%s %v: %q contains %q", tc.regime, tc.lamps, out, banned)
			}
		}
	}
}
