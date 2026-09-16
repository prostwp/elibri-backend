package demobot

import (
	"strings"
	"testing"
	"time"
)

// A trend winner's blocks are that one card's content sentences. The digest
// must not ship them as its own: live on 2026-09-15 the digest read
// "Macro: risk-on" in its sections while blocks.regime said "flat — no trend".
// /agents/top keeps them — there the envelope IS the winner card.
//
// Since 2026-09-16 the digest has its OWN blocks instead of none: they
// describe the sweep and the selection, and carry nothing of the winner's
// reading (no regime, no scenarios, no invalidation).
func TestDigestEnvelopeDropsWinnerBlocks(t *testing.T) {
	at := time.Date(2026, 9, 15, 0, 0, 0, 0, time.UTC)
	trend := Card{
		Agent: "Trend Agent", ShortName: "Trend", Asset: "BTC", AssetKey: "btc",
		Command: keyTrend, Verdict: "Confirmed UPTREND · 4h", Short: "confirmed uptrend",
		Emoji: emojiBull, State: trendUp, Deviation: 90, DataTime: at, confirmed: true,
		Blocks: &ContentBlocks{},
	}
	macro := Card{
		Agent: "Macro Agent", ShortName: "Macro", Command: keyMacro,
		Verdict: "RISK-ON — tradfin lamps lean into risk", Short: "risk-on",
		Emoji: emojiBull, Deviation: 10, DataTime: at,
	}
	g := gathered{cards: map[string]Card{keyTrend: trend, keyMacro: macro}, regime: "risk_on", at: at.Add(time.Hour)}

	winner, top := topSelection(g)
	if winner != keyTrend {
		t.Fatalf("fixture must make trend the winner, got %q", winner)
	}
	if cardEnvelope(top).Blocks == nil {
		t.Fatal("the winner card itself (the /agents/top envelope) must keep its blocks")
	}
	env := digestEnvelope(g, "")
	if env.Blocks == nil {
		t.Fatal("the digest envelope must carry its own blocks")
	}
	if env.Blocks == trend.Blocks {
		t.Error("digest envelope must not inherit the winner's blocks pointer")
	}
	for _, f := range []string{env.Blocks.WhyLevel, env.Blocks.Regime, env.Blocks.Context, env.Blocks.StateChanges, env.Blocks.Source} {
		if f != "" {
			t.Errorf("the digest holds no single reading, got %q", f)
		}
	}
	if env.Blocks.Scenarios != nil || env.Blocks.Invalidates != nil {
		t.Errorf("the digest has no idea to run forward or invalidate: %+v", env.Blocks)
	}
	if env.Blocks.Limitations != selectionCaveat(g.selection()) {
		t.Errorf("limitations: %q", env.Blocks.Limitations)
	}
	if !strings.Contains(env.Blocks.WhatHappened, selectionLine(g.selection())) {
		t.Errorf("what_happened must say how the card was chosen: %q", env.Blocks.WhatHappened)
	}

	// /agents/top keeps the winner's own sentences and only ADDS the selection
	// caveat — and never through the shared pointer.
	tb := topBlocks(g.selection(), top)
	if tb == top.Blocks {
		t.Error("topBlocks must copy, not alias the winner card's blocks")
	}
	if top.Blocks.Limitations != "" {
		t.Errorf("the winner card's own blocks were mutated: %q", top.Blocks.Limitations)
	}
	if tb.Limitations != selectionCaveat(g.selection()) {
		t.Errorf("top limitations: %q", tb.Limitations)
	}
	if env.Agent != digestAgentName || env.Asset != "" {
		t.Errorf("digest identity changed: agent %q asset %q", env.Agent, env.Asset)
	}
}
