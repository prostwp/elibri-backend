package demobot

import "testing"

// The hook rule, as documented on pickTop:
//  1. Macro regime "risk_off" → macro is always top.
//  2. Otherwise the available agent with the highest deviation-from-neutral
//     among funding, momentum, trend wins.
//  3. Ties are broken by fixed order: funding > momentum > trend.
//  4. Nothing available → macro.

func TestPickTopScenarioRiskOff(t *testing.T) {
	// Even with a screaming funding signal, RISK-OFF macro takes the top slot.
	got := pickTop("risk_off", map[string]int{
		keyFunding:  95,
		keyMomentum: 80,
		keyTrend:    70,
	})
	if got != keyMacro {
		t.Fatalf("risk_off scenario: got %q, want %q", got, keyMacro)
	}
}

func TestPickTopScenarioStrongestDeviation(t *testing.T) {
	// risk_on regime → the strongest of funding/momentum/trend wins.
	got := pickTop("risk_on", map[string]int{
		keyFunding:  80,
		keyMomentum: 60,
		keyTrend:    40,
	})
	if got != keyFunding {
		t.Fatalf("strongest-deviation scenario: got %q, want %q", got, keyFunding)
	}
	// Momentum ahead → momentum wins.
	got = pickTop("mixed", map[string]int{
		keyFunding:  10,
		keyMomentum: 75,
		keyTrend:    74,
	})
	if got != keyMomentum {
		t.Fatalf("momentum-ahead scenario: got %q, want %q", got, keyMomentum)
	}
}

func TestPickTopScenarioTieBreak(t *testing.T) {
	// Exact tie momentum vs trend → fixed order puts momentum first.
	got := pickTop("mixed", map[string]int{
		keyFunding:  10,
		keyMomentum: 55,
		keyTrend:    55,
	})
	if got != keyMomentum {
		t.Fatalf("tie-break scenario: got %q, want %q", got, keyMomentum)
	}
	// Three-way tie → funding is first in the fixed order.
	got = pickTop("mixed", map[string]int{
		keyFunding:  55,
		keyMomentum: 55,
		keyTrend:    55,
	})
	if got != keyFunding {
		t.Fatalf("three-way tie: got %q, want %q", got, keyFunding)
	}
}

func TestPickTopNothingAvailable(t *testing.T) {
	// All signal agents offline → macro (even if macro itself is degraded,
	// the caller renders an honest offline card).
	if got := pickTop("mixed", map[string]int{}); got != keyMacro {
		t.Fatalf("empty availability: got %q, want %q", got, keyMacro)
	}
	// Unavailable agents simply do not appear in the map.
	got := pickTop("risk_on", map[string]int{keyTrend: 5})
	if got != keyTrend {
		t.Fatalf("only trend available: got %q, want %q", got, keyTrend)
	}
}
