package demobot

// whale_colour_test.go — blocks.what_happened on the labeled card says WHY the
// card has its colour and names the stablecoin flow next to the coin's
// (2026-09-16). Rules, colour, facts, verdict, card_html and the showcase are
// pinned byte for byte against 4172a81.

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// whaleColourLive is the live shape the site showed on 2026-09-16: ETH net
// $121.92M to exchanges on 360 transfers (ratio 0.63, past the band — the card
// is red) while USDT net $219.74M also went to exchanges.
var whaleColourLive = strings.Replace(strings.Replace(whaleLabeledProd,
	`"net_flow_usd_24h":-10094058.96,"direction":"neutral",
     "inflow_usd_24h":50000000,"outflow_usd_24h":60094058.96,"tx_count_24h":77`,
	`"net_flow_usd_24h":121920000,"direction":"inflow",
     "inflow_usd_24h":150000000,"outflow_usd_24h":28080000,"tx_count_24h":360`, 1),
	`"net_flow_usd_24h":61104747.41,"direction":"inflow",
     "inflow_usd_24h":61104747.41`,
	`"net_flow_usd_24h":219740000,"direction":"inflow",
     "inflow_usd_24h":219740000`, 1)

// whaleColourCoinNoStable: ETH carries the direction (from exchanges), USDT saw
// no net at all.
var whaleColourCoinNoStable = strings.Replace(whaleCoinLeadFixture,
	`"net_flow_usd_24h":61104747.41,"direction":"inflow",
     "inflow_usd_24h":61104747.41`,
	`"net_flow_usd_24h":0,"direction":"neutral",
     "inflow_usd_24h":0`, 1)

// whaleColourBanned is the whale sweep plus the showcase's own list.
var whaleColourBanned = []string{"inflow", "outflow", "accumulat", "dump", "whales", "buying", "selling",
	"sell pressure", "buy pressure", "bullish", "bearish", "transferred", "smart money", " will ", "expect",
	"likely", "forecast says", "signal", "lean", "buy", "sell"}

func whaleColourWhat(t *testing.T, body string) (Card, string) {
	t.Helper()
	c := whaleCardFrom(whaleResp(t, body), whaleLabeledAt)
	if c.Blocks == nil {
		t.Fatal("no blocks")
	}
	w := c.Blocks.WhatHappened
	if n := utf8.RuneCountInString(w); n > whaleBlockProseMaxRunes {
		t.Errorf("%d runes > %d: %q", n, whaleBlockProseMaxRunes, w)
	}
	low := strings.ToLower(w)
	for _, b := range whaleColourBanned {
		if strings.Contains(low, b) {
			t.Errorf("%q must not appear: %q", b, w)
		}
	}
	return c, w
}

// whaleColourStableLater: the stablecoin's snapshot is from a later tick than
// the coin's (a failed coin read leaves each asset on its own latest snapshot).
var whaleColourStableLater = strings.Replace(whaleColourLive,
	`"asset":"USDT","captured_at":"2026-09-15T23:24:43Z"`, `"asset":"USDT","captured_at":"2026-09-16T03:14:00Z"`, 1)

// whaleColourStableUnstamped: the stablecoin's snapshot carries no time.
var whaleColourStableUnstamped = strings.Replace(whaleColourLive,
	`"asset":"USDT","captured_at":"2026-09-15T23:24:43Z",`, `"asset":"USDT",`, 1)

// whaleColourCoinNoGross: ETH moved with a non-zero net the backend calls
// neutral, but no gross was served, so the band cannot be shown to be why.
var whaleColourCoinNoGross = strings.Replace(whaleLabeledProd,
	`"inflow_usd_24h":50000000,"outflow_usd_24h":60094058.96`, `"inflow_usd_24h":0,"outflow_usd_24h":0`, 1)

const (
	whaleColourReasonTo = "The colour follows ETH only: ETH moving to exchanges is counted as supply arriving there. " +
		"Stablecoin moves do not change the colour."
	whaleColourReasonFrom = "The colour follows ETH only: ETH moving off exchanges is counted as supply locked away. " +
		"Stablecoin moves do not change the colour."
)

func TestWhaleColourWhatHappened(t *testing.T) {
	for _, tc := range []struct {
		name, body, emoji, want string
	}{
		{"live: coin to + stablecoin to, same window", whaleColourLive, emojiBear,
			"Labeled exchange wallets show a net $121.92M to exchanges in ETH in the 24h to Sep 15 23:24 UTC. " +
				"USDT shows a net $219.74M to exchanges in the same window. " + whaleColourReasonTo},
		{"coin from + stablecoin to", whaleCoinLeadFixture, emojiBull,
			"Labeled exchange wallets show a net $20.09M from exchanges in ETH in the 24h to Sep 15 23:24 UTC. " +
				"USDT shows a net $61.10M to exchanges in the same window. " + whaleColourReasonFrom},
		{"coin without stablecoin", whaleColourCoinNoStable, emojiBull,
			"Labeled exchange wallets show a net $20.09M from exchanges in ETH in the 24h to Sep 15 23:24 UTC. " +
				whaleColourReasonFrom},
		{"stablecoin snapshot on its own later time", whaleColourStableLater, emojiBear,
			"Labeled exchange wallets show a net $121.92M to exchanges in ETH in the 24h to Sep 15 23:24 UTC. " +
				"USDT shows a net $219.74M to exchanges in the 24h to Sep 16 03:14 UTC. " + whaleColourReasonTo},
		{"stablecoin snapshot without a time", whaleColourStableUnstamped, emojiBear,
			"Labeled exchange wallets show a net $121.92M to exchanges in ETH in the 24h to Sep 15 23:24 UTC. " +
				"USDT shows a net $219.74M to exchanges over its latest 24h snapshot. " + whaleColourReasonTo},
		{"stablecoin only, no coin moved", whaleLabeledStableOnly, emojiNeutral,
			"Labeled exchange wallets show a stablecoin net $61.10M to exchanges in USDT in the 24h to Sep 15 23:24 UTC. " +
				"Stablecoin moves do not change the colour, so the card stays neutral."},
		{"stablecoin + coin inside the band (prod)", whaleLabeledProd, emojiNeutral,
			"Labeled exchange wallets show a stablecoin net $61.10M to exchanges in USDT in the 24h to Sep 15 23:24 UTC. " +
				"No coin's net reached the 10% of gross flow needed for a direction, and stablecoin moves do not change the colour, so the card stays neutral."},
		{"stablecoin + coin moved, band not demonstrable", whaleColourCoinNoGross, emojiNeutral,
			"Labeled exchange wallets show a stablecoin net $61.10M to exchanges in USDT in the 24h to Sep 15 23:24 UTC. " +
				"No coin carried a net direction, and stablecoin moves do not change the colour, so the card stays neutral."},
		{"no net direction", whaleLabeledNoNet, emojiNeutral,
			"Labeled exchange wallets show no net direction in the 24h to Sep 15 23:24 UTC · 31 labeled transfers."},
	} {
		c, w := whaleColourWhat(t, tc.body)
		if c.Emoji != tc.emoji {
			t.Errorf("%s: fixture colour %q, want %q", tc.name, c.Emoji, tc.emoji)
		}
		if w != tc.want {
			t.Errorf("%s:\n got %q\nwant %q", tc.name, w, tc.want)
		}
		if c.Emoji == emojiBear && strings.Contains(w, "locked away") || c.Emoji == emojiBull && strings.Contains(w, "arriving") {
			t.Errorf("%s: reason contradicts the colour: %q", tc.name, w)
		}
	}
}

func TestWhaleColourNoTransferBranchClosed(t *testing.T) {
	w := whaleLabeledBlocks(nil, nil, nil, 0, []string{"ETH", "USDT"}, whaleLabeledAt, true).WhatHappened
	if w != "No labeled exchange transfer was recorded in the 24h to Sep 15 23:30 UTC (ETH, USDT)." {
		t.Errorf("what_happened %q", w)
	}
	long := whaleLabeledBlocks(nil, nil, nil, 0, []string{strings.Repeat("A", 200)}, whaleLabeledAt, true).WhatHappened
	if n := utf8.RuneCountInString(long); n > whaleFactMaxRunes {
		t.Errorf("%d runes > %d", n, whaleFactMaxRunes)
	}
}

// The colour reason is never dropped, whatever the length of the names: the
// stablecoin sentence goes first, then the first sentence is shortened.
func TestWhaleColourReasonSurvivesBudget(t *testing.T) {
	at := whaleLabeledAt
	for _, n := range []int{45, 80, 200} {
		for _, dir := range []string{"inflow", "outflow"} {
			lead := &WhaleFlow{Asset: strings.Repeat("X", n), Direction: dir, NetFlowUSD24h: 999.99e6, CapturedAt: at}
			stable := &WhaleFlow{Asset: strings.Repeat("S", n), Direction: "inflow", NetFlowUSD24h: 999.99e6,
				CapturedAt: at.Add(-5 * time.Hour)}
			w := whaleLabeledBlocks(lead, stable, nil, 1, nil, at, true).WhatHappened
			if c := utf8.RuneCountInString(w); c > whaleBlockProseMaxRunes {
				t.Errorf("%d/%s: %d runes > %d", n, dir, c, whaleBlockProseMaxRunes)
			}
			reason := whaleLabeledColourReason(lead)
			if !strings.HasSuffix(w, reason) || !strings.HasPrefix(reason, "The colour follows ") {
				t.Errorf("%d/%s: the colour reason was dropped: %q", n, dir, w)
			}
			if !strings.HasPrefix(w, "Labeled exchange wallets show a net") {
				t.Errorf("%d/%s: the coin's flow was dropped: %q", n, dir, w)
			}
		}
	}
}

// Everything a reader sees apart from blocks.what_happened — verdict, short,
// facts, card_html, one-liner, the showcase's detected/explained/data/
// conclusion, the other blocks, colour, confidence — is byte for byte what
// 4172a81 served, on every whale path.
func TestWhaleColourSurfaceMatches4172a81(t *testing.T) {
	type surf struct {
		Emoji, Verdict, Short, HTML, OneLiner, Detected, Explained, Conclusion, How, SourceNote, Asset string
		Facts, Data                                                                                    []string
		Conf                                                                                           *int
		Status                                                                                         int
		BlocksRest                                                                                     string
	}
	cards := whaleEveryCard(t)
	cards["dump_live"] = whaleCardFrom(whaleResp(t, whaleColourLive), whaleLabeledAt)
	cards["dump_coin_no_stable"] = whaleCardFrom(whaleResp(t, whaleColourCoinNoStable), whaleLabeledAt)
	got := map[string]surf{}
	for k, c := range cards {
		var rest string
		if c.Blocks != nil {
			b := *c.Blocks
			b.WhatHappened = ""
			j, err := json.Marshal(b)
			if err != nil {
				t.Fatal(err)
			}
			rest = string(j)
		}
		got[k] = surf{c.Emoji, c.Verdict, c.Short, whaleFooterClockRe.ReplaceAllString(c.RenderHTML(), "<t>"), c.OneLiner(),
			detectedSentence(c), strongestFact(c), conclusionFor(c), c.HowItWorks, c.SourceNote, c.Asset,
			c.Facts, exampleFacts(c), c.Confidence, int(c.effectiveStatus()), rest}
	}
	raw, err := os.ReadFile("testdata/whale_surface_4172a81_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var want map[string]surf
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("card set changed: got %d, golden %d", len(got), len(want))
	}
	for k, w := range want {
		g, _ := json.Marshal(got[k])
		wb, _ := json.Marshal(w)
		if string(g) != string(wb) {
			t.Errorf("%s changed vs 4172a81:\ngot  %s\nwant %s", k, g, wb)
		}
	}
}

// A first sentence too long for the bound is shortened, never the reason.
func TestWhaleColourProseFitCutsFirstNotReason(t *testing.T) {
	reason := whaleLabeledColourReason(&WhaleFlow{Asset: "ETH", Direction: "inflow"})
	w := whaleProseFit(strings.Repeat("a", 400)+".", []string{"Middle."}, reason)
	if n := utf8.RuneCountInString(w); n > whaleBlockProseMaxRunes {
		t.Errorf("%d runes > %d", n, whaleBlockProseMaxRunes)
	}
	if !strings.HasSuffix(w, "… "+reason) || strings.Contains(w, "Middle") {
		t.Errorf("got %q", w)
	}
}
