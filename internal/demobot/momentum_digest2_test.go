package demobot

// momentum_digest2_test.go — second review of stage 1 (2026-09-15): the
// default trio with BTC/ETH dead and gold alive. The colour fallback is
// decided by the REQUESTED assets, and a momentum card without a BTC/ETH
// reading never competes in the digest.

import (
	"strings"
	"testing"
	"time"
)

func TestMomentumCryptoDeadGoldAliveIsNeutralAndNotRanked(t *testing.T) {
	now := momNow
	bar1h, bar4h := now.Add(-30*time.Minute), now.Add(-90*time.Minute)
	trend := Card{Agent: "Trend Agent", ShortName: "Trend", Asset: "BTC", DataTime: bar4h, Verdict: "Grey zone", Short: "grey zone", Emoji: emojiNeutral}
	macro := Card{Agent: "Macro Agent", ShortName: "Macro", DataTime: now, Verdict: "MIXED", Short: "mixed", Emoji: emojiNeutral}

	for _, st := range []cardStatus{statusSourceOffline, statusInsufficientHistory} {
		mom := Card{Agent: "Momentum Agent", ShortName: "Momentum", Asset: "BTC/ETH/XAUUSD", Command: keyMomentum, DataTime: now}
		composeMomentum(&mom, []momentumAsset{
			{spec: btcSpec, status: st},
			{spec: assetTable["eth"], status: st},
			{spec: xauSpec, read: momRead(xauSpec, 62, 3, bar1h), status: statusOK},
		}, "", now)
		if mom.Verdict != "1 bullish (GOLD) · 0 bearish · 0 not confirmed · 2 unavailable · 1h" {
			t.Errorf("%v: the counter still covers gold: %q", st, mom.Verdict)
		}
		if mom.Emoji != emojiNeutral || mom.confirmed || !mom.noRankedRead {
			t.Errorf("%v: emoji %q confirmed %v noRankedRead %v", st, mom.Emoji, mom.confirmed, mom.noRankedRead)
		}
		facts := strings.Join(mom.Facts, "|")
		if !strings.Contains(facts, momentumColourNoneLine) || strings.Contains(facts, momentumColourLine) {
			t.Errorf("%v: colour line must say no BTC/ETH read is available: %q", st, mom.Facts)
		}

		for _, fs := range []int{0, 30} {
			fund := Card{Agent: "Funding Agent", ShortName: "Funding", DataTime: now, Verdict: "Funding balanced", Short: "balanced", Emoji: emojiNeutral, Deviation: fs}
			g := gathered{at: now, regime: "mixed", cards: map[string]Card{keyMomentum: mom, keyTrend: trend, keyFunding: fund, keyMacro: macro}}
			p := g.selection()
			for _, c := range p.Candidates {
				if c.Key == keyMomentum && (c.Eligible || c.Excluded != excludedNoRankedRead) {
					t.Errorf("%v/%d: momentum without a BTC/ETH read must be excluded as no_ranked_read: %+v", st, fs, c)
				}
			}
			if p.Winner == keyMomentum {
				t.Errorf("%v/%d: momentum must not be the highlighted card", st, fs)
			}
			env := digestEnvelope(g, "")
			if env.Semaphore != "neutral" || strings.Contains(env.Verdict, "bullish") {
				t.Errorf("%v/%d: digest headline %q semaphore %q", st, fs, env.Verdict, env.Semaphore)
			}
			found := false
			for _, c := range env.Digest.Selection.Candidates {
				if c.Agent == keyMomentum {
					found = c.Excluded != nil && *c.Excluded == "no_ranked_read"
				}
			}
			if !found {
				t.Errorf("%v/%d: digest.selection.candidates must serve excluded=no_ranked_read", st, fs)
			}
			for _, html := range []string{renderDigestHTML(g, ""), env.CardHTML} {
				if strings.Contains(html, emojiBull) || strings.Contains(html, emojiBear) {
					t.Errorf("%v/%d: a colour in the digest:\n%s", st, fs, html)
				}
			}
			for _, s := range env.Sections {
				if strings.HasPrefix(s, emojiBull) {
					t.Errorf("%v/%d: green one-liner: %q", st, fs, s)
				}
			}
			if _, top := topSelection(g); top.Agent == "Momentum Agent" {
				t.Errorf("%v/%d: /agents/top must not pick momentum", st, fs)
			}
		}
	}

	// Live BTC/ETH: nothing changes — eligible, ranked as before.
	live := Card{Agent: "Momentum Agent", ShortName: "Momentum", Asset: "BTC/ETH/XAUUSD", Command: keyMomentum, DataTime: now}
	composeMomentum(&live, defaultTrio(
		momRead(btcSpec, 60, 1, bar4h), momRead(assetTable["eth"], 50, 1, bar4h), momRead(xauSpec, 62, 3, bar1h)), "", now)
	if live.noRankedRead {
		t.Error("a card with BTC/ETH reads is ranked")
	}
	if cand := rankCandidate(keyMomentum, live, now); !cand.Eligible || !cand.Confirmed || cand.Score != 20 {
		t.Errorf("live BTC/ETH candidate: %+v", cand)
	}

	// A requested but dead BTC beside a bullish EURUSD: neutral, and said so.
	eur, gbp := assetTable["eurusd"], assetTable["gbpusd"]
	c := Card{DataTime: now}
	composeMomentum(&c, []momentumAsset{
		{spec: btcSpec, status: statusSourceOffline},
		{spec: eur, read: momRead(eur, 60, 1, bar1h), status: statusOK},
	}, "", now)
	if c.Emoji != emojiNeutral || !strings.Contains(strings.Join(c.Facts, "|"), momentumColourNoneLine) {
		t.Errorf("btc dead + eurusd bullish: emoji %q facts %q", c.Emoji, c.Facts)
	}

	// No BTC/ETH in the request (?assets=eurusd,gbpusd): colours by its reads.
	c = Card{DataTime: now}
	composeMomentum(&c, []momentumAsset{
		{spec: eur, read: momRead(eur, 60, 1, bar1h), status: statusOK},
		{spec: gbp, read: momRead(gbp, 40, -1, bar1h), status: statusOK},
	}, "", now)
	if c.Emoji != emojiNeutral { // one bullish, one bearish → both ways
		t.Errorf("fx scan both ways: %q", c.Emoji)
	}
	c = Card{DataTime: now}
	composeMomentum(&c, []momentumAsset{
		{spec: eur, read: momRead(eur, 60, 1, bar1h), status: statusOK},
		{spec: gbp, status: statusSourceOffline},
	}, "", now)
	if c.Emoji != emojiBull {
		t.Errorf("fx scan with one bullish read must stay green: %q", c.Emoji)
	}
	facts := strings.Join(c.Facts, "|")
	if strings.Contains(facts, momentumColourLine) || strings.Contains(facts, momentumColourNoneLine) {
		t.Errorf("fx scan needs no colour line: %q", c.Facts)
	}
}
