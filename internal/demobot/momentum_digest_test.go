package demobot

// momentum_digest_test.go — review of stage 1 (2026-09-15): the composite
// momentum card inside the digest. The colour follows the reads the digest
// ranks (BTC/ETH), the selection line names each agent's ranked scope, and
// the headline names the card's assets.

import (
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// Reviewer scenario: BTC 50/+1 and ETH 52/−1 are not confirmed, gold 62/+3 is
// bullish. The digest says "No confirmed reading" — the momentum card must
// not show a bullish colour beside it, neither as the highlighted card nor as
// a one-liner under "Everything else".
func TestMomentumDigestNoColourBesideNoConfirmed(t *testing.T) {
	now := momNow
	bar4h, bar1h := now.Add(-90*time.Minute), now.Add(-30*time.Minute)
	mom := Card{Agent: "Momentum Agent", ShortName: "Momentum", Asset: "BTC/ETH/XAUUSD", Command: keyMomentum, DataTime: now}
	composeMomentum(&mom, defaultTrio(
		momRead(btcSpec, 50, 1, bar4h),
		momRead(assetTable["eth"], 52, -1, bar4h),
		momRead(xauSpec, 62, 3, bar1h)), "", now)
	if mom.Verdict != "1 bullish (GOLD) · 0 bearish · 2 not confirmed" {
		t.Errorf("the counter still covers gold: %q", mom.Verdict)
	}
	if mom.Emoji != emojiNeutral || mom.confirmed || mom.Deviation != 0 {
		t.Errorf("colour/digest inputs: emoji %q confirmed %v dev %d", mom.Emoji, mom.confirmed, mom.Deviation)
	}
	if !strings.Contains(strings.Join(mom.Facts, "|"), momentumColourLine) {
		t.Errorf("a card mixing BTC/ETH and gold must say what the colour follows: %q", mom.Facts)
	}

	trend := Card{Agent: "Trend Agent", ShortName: "Trend", Asset: "BTC", DataTime: bar4h, Verdict: "Grey zone", Short: "grey zone", Emoji: emojiNeutral}
	macro := Card{Agent: "Macro Agent", ShortName: "Macro", DataTime: now, Verdict: "MIXED", Short: "mixed", Emoji: emojiNeutral}
	for _, fundScore := range []int{0, 30} { // 0: momentum wins the fallback; 30: funding does, momentum is a section
		fund := Card{Agent: "Funding Agent", ShortName: "Funding", DataTime: now, Verdict: "Funding balanced", Short: "balanced", Emoji: emojiNeutral, Deviation: fundScore}
		g := gathered{at: now, regime: "mixed", cards: map[string]Card{keyMomentum: mom, keyTrend: trend, keyFunding: fund, keyMacro: macro}}
		p := g.selection()
		if p.Rule != ruleUnconfirmed {
			t.Fatalf("funding %d: rule %q, want the unconfirmed fallback", fundScore, p.Rule)
		}
		line := selectionLine(p)
		if !strings.HasPrefix(line, "No confirmed reading among Funding, Momentum (BTC/ETH), Trend (BTC)") {
			t.Errorf("selection line must name the ranked scope: %q", line)
		}
		html := renderDigestHTML(g, "")
		if strings.Contains(html, emojiBull) || strings.Contains(html, emojiBear) {
			t.Errorf("funding %d: a colour beside \"no confirmed\":\n%s", fundScore, html)
		}
		if env := digestEnvelope(g, ""); env.Semaphore != "neutral" {
			t.Errorf("funding %d: digest semaphore %q", fundScore, env.Semaphore)
		}
	}

	// A confirmed BTC read still colours the card and is confirmed — the
	// colour and the digest agree in both directions.
	c := Card{DataTime: now}
	composeMomentum(&c, defaultTrio(
		momRead(btcSpec, 60, 1, bar4h), momRead(assetTable["eth"], 52, -1, bar4h), momRead(xauSpec, 40, -3, bar1h)), "", now)
	if c.Emoji != emojiBull || !c.confirmed || c.Verdict != "1 bullish (BTC) · 1 bearish (GOLD) · 1 not confirmed" {
		t.Errorf("BTC bullish beside a bearish gold: emoji %q confirmed %v verdict %q", c.Emoji, c.confirmed, c.Verdict)
	}

	// Without any BTC/ETH read (an FX/gold scan, never in the digest) the
	// colour follows all reads, and no colour line is needed.
	c = Card{DataTime: now}
	eur, gbp := assetTable["eurusd"], assetTable["gbpusd"]
	composeMomentum(&c, []momentumAsset{
		{spec: eur, read: momRead(eur, 60, 1, bar1h), status: statusOK},
		{spec: gbp, read: momRead(gbp, 50, 1, bar1h), status: statusOK},
	}, "", now)
	if c.Emoji != emojiBull || c.Verdict != "1 bullish (EURUSD) · 0 bearish · 1 not confirmed · 1h" {
		t.Errorf("FX-only scan: emoji %q verdict %q", c.Emoji, c.Verdict)
	}
	if strings.Contains(strings.Join(c.Facts, "|"), momentumColourLine) {
		t.Errorf("no colour line without BTC/ETH: %q", c.Facts)
	}
}

// Every variant of the selection line names the ranked scope and fits.
func TestSelectionLineNamesScope(t *testing.T) {
	all := []topCandidate{{Key: keyFunding, Eligible: true}, {Key: keyMomentum, Eligible: true}, {Key: keyTrend, Eligible: true}}
	one := []topCandidate{{Key: keyMomentum, Eligible: true}}
	for _, p := range []topPick{
		{Rule: ruleMacroRiskOff, Winner: keyMacro},
		{Rule: ruleConfirmed, Winner: keyFunding, Candidates: all},
		{Rule: ruleConfirmed, Winner: keyMomentum, Candidates: one},
		{Rule: ruleUnconfirmed, Winner: keyMomentum},
		{Rule: ruleFallbackMacro, Winner: keyMacro},
	} {
		l := selectionLine(p)
		if !strings.Contains(l, "Momentum (BTC/ETH)") || !strings.Contains(l, "Trend (BTC)") {
			t.Errorf("%s: scope missing: %q", p.Rule, l)
		}
		if n := utf8.RuneCountInString(l); n > 110 {
			t.Errorf("%s: %d chars: %q", p.Rule, n, l)
		}
	}
	digestShowNoHighlight = true
	t.Cleanup(func() { digestShowNoHighlight = false })
	h := digestHeadlineFor(topPick{Rule: ruleUnconfirmed, NoHighlight: true}, Card{})
	if h != "No highlighted reading — nothing confirmed among Funding, Momentum (BTC/ETH), Trend (BTC)" {
		t.Errorf("no-highlight headline: %q", h)
	}
}

// The digest headline names the highlighted card's assets whenever it has
// any — the composite momentum card included; market-wide cards stay bare.
func TestDigestHeadlineNamesMomentumAssets(t *testing.T) {
	now := momNow
	bar := now.Add(-90 * time.Minute)
	mom := Card{Agent: "Momentum Agent", ShortName: "Momentum", Asset: "BTC/ETH/XAUUSD", Command: keyMomentum, DataTime: now}
	composeMomentum(&mom, defaultTrio(
		momRead(btcSpec, 60, 1, bar), momRead(assetTable["eth"], 50, 1, bar), momRead(xauSpec, 50, 1, bar)), "", now)
	g := gathered{at: now, regime: "mixed", cards: map[string]Card{
		keyMomentum: mom,
		keyFunding:  {Agent: "Funding Agent", ShortName: "Funding", DataTime: now, Verdict: "Funding balanced", Short: "balanced"},
		keyMacro:    {Agent: "Macro Agent", ShortName: "Macro", DataTime: now, Verdict: "MIXED", Short: "mixed"},
	}}
	// Mixed timeframes (crypto 4h, gold 1h): no shared tf in the counter.
	want := "Top signal: Momentum Agent · BTC/ETH/XAUUSD — 1 bullish (BTC) · 0 bearish · 2 not confirmed"
	if got := digestEnvelope(g, "").Verdict; got != want {
		t.Errorf("digest headline:\n got %q\nwant %q", got, want)
	}
	// The /showcase digest row words its headline with the same function.
	if got := digestHeadlineFor(g.selection(), g.cards[keyMomentum]); got != want {
		t.Errorf("showcase digest headline: %q", got)
	}
	// Telegram prints the winner's own card header, which names the assets.
	if html := renderDigestHTML(g, ""); !strings.Contains(html, "<b>Momentum Agent</b> · BTC/ETH/XAUUSD") {
		t.Errorf("telegram digest must name the assets:\n%s", html)
	}
	for _, tc := range []struct {
		card Card
		want string
	}{
		{Card{Agent: "Trend Agent", Asset: "BTC", Verdict: "Confirmed UPTREND · 4h"}, "Top signal: Trend Agent · BTC — Confirmed UPTREND · 4h"},
		{Card{Agent: "Funding Agent", Verdict: "Longs crowded — squeeze risk building"}, "Top signal: Funding Agent — Longs crowded — squeeze risk building"},
		{Card{Agent: "Macro Agent", Verdict: "RISK-OFF"}, "Top signal: Macro Agent — RISK-OFF"},
	} {
		if got := digestHeadline(tc.card); got != tc.want {
			t.Errorf("unchanged headline for %s: %q, want %q", tc.card.Agent, got, tc.want)
		}
	}
}
