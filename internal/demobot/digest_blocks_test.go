package demobot

import (
	"testing"
	"time"
)

// A trend winner's blocks are that one card's content sentences. The digest
// must not ship them as its own: live on 2026-09-15 the digest read
// "Macro: risk-on" in its sections while blocks.regime said "flat — no trend".
// /agents/top keeps them — there the envelope IS the winner card.
func TestDigestEnvelopeDropsWinnerBlocks(t *testing.T) {
	at := time.Date(2026, 9, 15, 0, 0, 0, 0, time.UTC)
	trend := Card{
		Agent: "Trend Agent", ShortName: "Trend", Asset: "BTC", AssetKey: "btc",
		Command: keyTrend, Verdict: "Confirmed UPTREND · 4h", Short: "confirmed uptrend",
		Emoji: emojiBull, State: trendUp, Deviation: 90, DataTime: at,
		Blocks: &ContentBlocks{},
	}
	macro := Card{
		Agent: "Macro Agent", ShortName: "Macro", Command: keyMacro,
		Verdict: "RISK-ON — tradfin lamps lean into risk", Short: "risk-on",
		Emoji: emojiBull, Deviation: 10, DataTime: at,
	}
	g := gathered{cards: map[string]Card{keyTrend: trend, keyMacro: macro}, regime: "risk_on"}

	winner, top := topSelection(g)
	if winner != keyTrend {
		t.Fatalf("fixture must make trend the winner, got %q", winner)
	}
	if cardEnvelope(top).Blocks == nil {
		t.Fatal("the winner card itself (the /agents/top envelope) must keep its blocks")
	}
	env := digestEnvelope(g, "")
	if env.Blocks != nil {
		t.Errorf("digest envelope must not inherit the winner's blocks: %+v", env.Blocks)
	}
	if env.Agent != digestAgentName || env.Asset != "" {
		t.Errorf("digest identity changed: agent %q asset %q", env.Agent, env.Asset)
	}
}
