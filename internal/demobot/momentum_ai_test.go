package demobot

import (
	"strings"
	"testing"
	"time"
)

// The composite card prints each asset's bar time; the AI payload must not
// (no time stamps — its hash is the AI cache key). The stripper removes
// exactly what momentumCheckLine writes, for every month and day width, and
// keeps the freshness words.
func TestMomentumAIFactsDropBarTimes(t *testing.T) {
	for m := time.January; m <= time.December; m++ {
		for _, d := range []int{1, 9, 10, 28} {
			at := time.Date(2026, m, d, 7, 5, 0, 0, time.UTC)
			for _, fresh := range []string{momentumOnTime, momentumMarketClosed, momentumDataDelayed} {
				r := momRead(xauSpec, 50, 1, at)
				line := momentumCheckLine("GOLD · COMEX GC=F", r, fresh)
				got := momentumAIFacts([]string{line})[0]
				want := "GOLD · COMEX GC=F: " + momentumChecklist(50, 1)
				if w := momentumFreshWords(fresh); w != "" {
					want += " · " + w
				}
				if got != want {
					t.Fatalf("%s:\n got %q\nwant %q", line, got, want)
				}
			}
		}
	}

	bar := momNow.Add(-90 * time.Minute)
	mom := Card{Agent: "Momentum Agent", ShortName: "Momentum", Asset: "BTC/ETH/XAUUSD", Command: keyMomentum, DataTime: momNow}
	composeMomentum(&mom, defaultTrio(
		momRead(btcSpec, 60, 1, bar), momRead(assetTable["eth"], 50, 1, bar), momRead(xauSpec, 50, 1, bar)), "", momNow)
	g := fakeGathered()
	g.cards[keyMomentum] = mom
	p := aiPayload(g)
	if strings.Contains(p, "UTC") || strings.Contains(p, "last bar") {
		t.Errorf("AI payload carries bar times: %s", p)
	}
	if !strings.Contains(p, "RSI 60.0 ≥ 55 ✓") {
		t.Errorf("AI payload lost the checklist: %s", p)
	}
	if !strings.Contains(strings.Join(mom.Facts, "|"), "last bar") {
		t.Error("the card itself keeps the bar times")
	}
}
