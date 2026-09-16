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

// The RS context line must not reach the AI payload: it is the one number on
// the card that is not as of the card's stamp (live prices, momentumRSContext),
// and at 0.1 pp it would move the payload hash — the AI cache key — on most
// sweeps, so a digest and a top of the SAME sweep could narrate one market in
// two texts. The card itself keeps the line.
func TestMomentumAIFactsDropRSContext(t *testing.T) {
	rs7, rs30 := -2.44, 7.9
	rs := momentumRSContext(MomentumItem{RS7D: &rs7, RS30D: &rs30})
	if rs == "" {
		t.Fatal("the fixture produced no RS line")
	}
	vol := momentumVolumeContext("BTC 4h volume", 0.69)
	got := momentumAIFacts([]string{"BTC: " + momentumChecklist(60, 1), vol, rs})
	want := []string{"BTC: " + momentumChecklist(60, 1), vol}
	if len(got) != len(want) {
		t.Fatalf("got %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("fact %d:\n got %q\nwant %q", i, got[i], want[i])
		}
	}

	// The same market with a moved RS digit must produce the SAME payload —
	// that payload is the cache key.
	bar := momNow.Add(-90 * time.Minute)
	payload := func(v float64) string {
		mom := Card{Agent: "Momentum Agent", ShortName: "Momentum", Asset: "BTC/ETH/XAUUSD", Command: keyMomentum, DataTime: momNow}
		composeMomentum(&mom, defaultTrio(
			momRead(btcSpec, 60, 1, bar), momRead(assetTable["eth"], 50, 1, bar), momRead(xauSpec, 50, 1, bar)), "", momNow)
		mom.Facts = append(mom.Facts, momentumRSContext(MomentumItem{RS7D: &v, RS30D: &rs30}))
		g := fakeGathered()
		g.cards[keyMomentum] = mom
		return aiPayload(g)
	}
	lo, hi := -2.44, -2.46
	if payload(lo) != payload(hi) {
		t.Errorf("the payload moved with the RS digit alone:\n%s\n%s", payload(lo), payload(hi))
	}
	if strings.Contains(payload(lo), "ETH return minus BTC return") {
		t.Errorf("AI payload carries the RS line: %s", payload(lo))
	}
}
