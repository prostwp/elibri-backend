package demobot

// gold_roll_test.go — the contract-roll check behind the gold card
// (gold_roll.go) and what the card says in and out of a roll window.
// Offline: every Yahoo answer is scripted (goldYahooStub) or read from
// testdata.

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// ── contracts ────────────────────────────────────────────────────────────────

func TestGoldCandidates(t *testing.T) {
	cases := []struct {
		day              string
		prev, near, next string
	}{
		{"2026-07-29", "GCM26", "GCQ26", "GCZ26"}, // the 2026-07-31 roll: Q26 → Z26, October skipped
		{"2026-07-31", "GCM26", "GCQ26", "GCZ26"},
		{"2026-08-03", "GCQ26", "GCZ26", "GCG27"}, // a daily bar still on Q26 here is "previous"
		{"2026-09-22", "GCQ26", "GCZ26", "GCG27"},
		{"2026-10-15", "GCQ26", "GCZ26", "GCG27"},
		{"2025-08-01", "GCQ25", "GCZ25", "GCG26"}, // the established daily roll on the 1st
		{"2025-11-26", "GCQ25", "GCZ25", "GCG26"}, // the 2025-11-28 roll: Z25 → G26
		{"2025-12-01", "GCZ25", "GCG26", "GCJ26"},
		{"2026-02-02", "GCG26", "GCJ26", "GCM26"},
		{"2026-03-31", "GCG26", "GCJ26", "GCM26"}, // hourly switch 2026-03-31 00:00 (candidate roll 2026-04-01)
		{"2026-04-01", "GCJ26", "GCM26", "GCQ26"},
		{"2026-12-31", "GCZ26", "GCG27", "GCJ27"},
	}
	for _, tc := range cases {
		d, _ := time.Parse("2006-01-02", tc.day)
		got := goldCandidates(d)
		if got != [3]string{tc.prev, tc.near, tc.next} {
			t.Errorf("%s: %v, want %s/%s/%s", tc.day, got, tc.prev, tc.near, tc.next)
		}
	}
	for code, want := range map[string]string{"GCQ26": "GCZ26", "GCZ26": "GCG27", "GCG27": "GCJ27", "GCJ27": "GCM27", "GCM27": "GCQ27", "GCV26": "", "x": ""} {
		if got := goldNextContract(code); got != want {
			t.Errorf("goldNextContract(%q) = %q, want %q", code, got, want)
		}
	}
}

// A 404 is "Yahoo does not serve this contract"; anything else is a failed
// request. The error text is unchanged for every other caller.
func TestYahooHTTPErrorKeepsItsText(t *testing.T) {
	err := error(&yahooHTTPError{Symbol: "GC=F", Status: 429})
	if err.Error() != "yahoo chart GC=F: HTTP 429" {
		t.Errorf("text changed: %q", err)
	}
	if goldNotServed(err) || !goldNotServed(&yahooHTTPError{Status: http.StatusNotFound}) || goldNotServed(errors.New("HTTP 404")) {
		t.Error("only a typed 404 means not served")
	}
}

// Daily bars match on all four prices, hourly bars on the close (the hybrid
// hours after a switch carry the old contract's open/high/low).
func TestGoldBarOnComparesWhatTheCardUses(t *testing.T) {
	bar := types.OHLCVCandle{Time: 100, Open: 4040, High: 4105, Low: 4038, Close: 4102.6}
	hybrid := []types.OHLCVCandle{{Time: 100, Open: 4098, High: 4105.4, Low: 4095.4, Close: 4102.6}}
	if !goldBarOn(bar, hybrid, "1h") {
		t.Error("hourly: same close is the same contract")
	}
	if goldBarOn(bar, hybrid, "1d") {
		t.Error("daily: all four prices must match")
	}
	if !goldBarOn(bar, []types.OHLCVCandle{{Time: 100, Open: 4040.04, High: 4105, Low: 4038, Close: 4102.64}}, "1d") {
		t.Error("within half a tick is the same print")
	}
	if goldBarOn(bar, []types.OHLCVCandle{{Time: 160, Open: 4040, High: 4105, Low: 4038, Close: 4102.6}}, "1h") {
		t.Error("another stamp is not the same bar")
	}
}

// ── the base golden: out of a window, byte-identical but for one line ────────

// stripDisclosure removes the one line the change adds to every card, and
// gold.roll, from an envelope.
func stripDisclosure(t *testing.T, raw json.RawMessage) map[string]any {
	t.Helper()
	var env map[string]any
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatal(err)
	}
	facts, _ := env["facts"].([]any)
	kept := []any{}
	n := 0
	for _, f := range facts {
		if f == goldSplicedLine {
			n++
			continue
		}
		kept = append(kept, f)
	}
	if n != 1 {
		t.Errorf("disclosure appears %d times in facts, want 1", n)
	}
	env["facts"] = kept
	html, _ := env["card_html"].(string)
	line := "• " + esc(goldSplicedLine) + "\n"
	if strings.Count(html, line) != 1 {
		t.Errorf("disclosure appears %d times in card_html, want 1", strings.Count(html, line))
	}
	env["card_html"] = strings.Replace(html, line, "", 1)
	if g, ok := env["gold"].(map[string]any); ok {
		delete(g, "roll")
	}
	return env
}

func TestGoldRollOutsideWindowMatchesBase(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", goldRollBasePath))
	if err != nil {
		t.Fatal(err)
	}
	var base map[string]json.RawMessage
	if err := json.Unmarshal(raw, &base); err != nil {
		t.Fatal(err)
	}
	got := goldRollDump(t)
	if len(got) != len(base) {
		t.Fatalf("%d cards, base has %d", len(got), len(base))
	}
	for name, b := range base {
		g, ok := got[name]
		if !ok {
			t.Errorf("%s: missing", name)
			continue
		}
		var want map[string]any
		if err := json.Unmarshal(b, &want); err != nil {
			t.Fatal(err)
		}
		if have := stripDisclosure(t, g); !reflect.DeepEqual(have, want) {
			hb, _ := json.MarshalIndent(have, "", " ")
			wb, _ := json.MarshalIndent(want, "", " ")
			t.Errorf("%s differs from 437c6a4 beyond the disclosure line:\n--- now\n%s\n--- 437c6a4\n%s", name, hb, wb)
		}
		// Every card of the dump is established out of a window.
		var env struct {
			Gold struct {
				Roll *GoldRoll `json:"roll"`
			} `json:"gold"`
		}
		if err := json.Unmarshal(g, &env); err != nil {
			t.Fatal(err)
		}
		if r := env.Gold.Roll; r == nil || r.State != goldRollNone {
			t.Errorf("%s: roll %+v, want none", name, r)
		} else if strings.HasPrefix(name, "pipeline/") &&
			(*r.CurrentContract != "GCZ26" || *r.NextContract != "GCG27" || *r.DailyContract != "GCZ26" || *r.HourlyContract != "GCZ26") {
			t.Errorf("%s: contracts %s/%s daily %s hourly %s", name, *r.CurrentContract, *r.NextContract, *r.DailyContract, *r.HourlyContract)
		}
	}
}

// ── a synthetic window ───────────────────────────────────────────────────────

// dailyEnding is n daily bars shaped by shape(i), the last one opening at last.
func dailyEnding(last time.Time, n int, shape func(i int) (o, h, l, c float64)) []types.OHLCVCandle {
	out := make([]types.OHLCVCandle, n)
	for i := range out {
		o, h, l, c := shape(i)
		out[i] = types.OHLCVCandle{Time: last.Unix() - int64(n-1-i)*86400, Open: o, High: h, Low: l, Close: c}
	}
	return out
}

// The window moment: Wednesday 2026-07-29, the last closed daily bar is the
// 1d candle of 07-28 — both inside July, so the candidates are GCQ26/GCZ26.
var (
	goldWindowNow    = time.Date(2026, 7, 29, 10, 30, 0, 0, time.UTC)
	goldWindowDay    = time.Date(2026, 7, 28, 4, 0, 0, 0, time.UTC)
	goldWindowHour   = time.Date(2026, 7, 29, 9, 0, 0, 0, time.UTC)
	goldWindowSpread = 50.0
)

// goldWindowCase scripts a roll window: GC=F's daily bars are GCQ26's, its
// hourly bars are GCZ26's. The uptrend's last day range is 5103.00 – 5113.00;
// the 1h close 5155.00 is ABOVE it in raw numbers and 5105.00 in GCQ26 prices
// — inside it. mut edits the contracts' answers for the unknown cases.
func goldWindowCase(t *testing.T, mut func(s *goldYahooStub)) (*Agents, *goldYahooStub) {
	t.Helper()
	daily := dailyEnding(goldWindowDay, 260, risingDay)
	h1 := fixedHourly(goldWindowHour, 5155)
	s := newGoldYahooStub(t)
	s.set("GC=F", yahooSeries{d1: daily, h1: h1})
	s.set("GCQ26.CMX", yahooSeries{d1: daily, h1: shiftBars(h1, -goldWindowSpread)})
	s.set("GCZ26.CMX", yahooSeries{d1: shiftBars(daily, goldWindowSpread), h1: h1})
	if mut != nil {
		mut(s)
	}
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	now := goldWindowNow
	ag.now = func() time.Time { return now }
	return ag, s
}

func TestGoldRollWindowCard(t *testing.T) {
	ag, _ := goldWindowCase(t, nil)
	c := ag.GoldCard(context.Background())
	if c.Offline || c.Verdict != "Daily regime: confirmed UPTREND" {
		t.Fatalf("verdict %q offline %v — the daily regime stands in a window", c.Verdict, c.Offline)
	}
	want := []string{
		"Last closed 1h price 5155.00 at 2026-07-29 10:00 UTC — on contract GCZ26",
		"Contract roll GCQ26 → GCZ26: 1h price on GCZ26, day range on GCQ26; price not placed against it",
		"Day range 5103.00 – 5113.00: high/low of the closed 1d candle of 2026-07-28",
		"A daily close above 5113.00 classifies the day as an upside break",
		"A daily close below 5103.00 classifies the day as a downside break",
		"A closed 1d candle below 3986.57 invalidates the daily uptrend reading (1 ATR under the EMA cluster)",
		"Macro backdrop: no gold read available",
		"Volatility: normal · 1d · ATR 1.000× its 30-bar baseline",
		goldSplicedLine,
		goldNoIdeaRoll,
	}
	if strings.Join(c.Facts, "\n") != strings.Join(want, "\n") {
		t.Errorf("facts:\n%s\nwant:\n%s", strings.Join(c.Facts, "\n"), strings.Join(want, "\n"))
	}
	joined := strings.Join(c.Facts, "\n") + c.Blocks.WhatHappened + strings.Join(c.Blocks.Scenarios, "\n")
	for _, bad := range []string{"above the day range", "below the day range", "inside the day range", "already above", "already below"} {
		if strings.Contains(joined, bad) {
			t.Errorf("a window must not place the 1h price against the day range: %q in\n%s", bad, joined)
		}
	}
	g := c.Gold
	if g.PricePosition != nil || g.Idea != nil {
		t.Errorf("price_position %v idea %+v, want both null", g.PricePosition, g.Idea)
	}
	if g.Price == nil || *g.Price != 5155 {
		t.Errorf("the price itself still ships: %v", g.Price)
	}
	r := g.Roll
	if r == nil || r.State != goldRollWindow || r.Reason == nil || *r.Reason != goldRollHourlyAhead ||
		*r.CurrentContract != "GCQ26" || *r.NextContract != "GCZ26" || *r.DailyContract != "GCQ26" || *r.HourlyContract != "GCZ26" {
		t.Fatalf("roll %+v", r)
	}
	if c.Blocks.WhatHappened != "Snapshot, not an event: 1d regime confirmed uptrend; last 1h close 5155.00 on GCZ26 (contract roll)" {
		t.Errorf("what_happened %q", c.Blocks.WhatHappened)
	}
	if c.Blocks.Invalidates == nil || strings.Contains(*c.Blocks.Invalidates, "1h") {
		t.Errorf("invalidates %v", c.Blocks.Invalidates)
	}
	b, _ := json.Marshal(cardEnvelope(c))
	if !bytes.Contains(b, []byte(`"roll":{"state":"window","reason":"hourly_ahead_of_daily","current_contract":"GCQ26","next_contract":"GCZ26","daily_contract":"GCQ26","hourly_contract":"GCZ26"}`)) {
		t.Errorf("envelope roll: %s", b)
	}
}

// In a window the nearest levels are taken from the last daily close (same
// contract as the levels), and the line says so.
func TestGoldRollWindowLevelsFromDailyClose(t *testing.T) {
	daily := dailyEnding(goldWindowDay, 260, zigzagDay)
	lastClose := daily[len(daily)-1].Close
	h1 := fixedHourly(goldWindowHour, lastClose+200) // far above every level in raw numbers
	s := newGoldYahooStub(t)
	s.set("GC=F", yahooSeries{d1: daily, h1: h1})
	s.set("GCQ26.CMX", yahooSeries{d1: daily, h1: shiftBars(h1, -goldWindowSpread)})
	s.set("GCZ26.CMX", yahooSeries{d1: shiftBars(daily, goldWindowSpread), h1: h1})
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	ag.now = func() time.Time { return goldWindowNow }
	c := ag.GoldCard(context.Background())

	sup, res := goldNearestLevels(daily, lastClose)
	want := goldKeyLevelsLineFrom(goldLevelsFromDailyClose, sup, res)
	if sup == nil || res == nil || !strings.HasPrefix(want, "Nearest to 1d close: support ") {
		t.Fatalf("fixture must cluster both sides of the daily close: %q", want)
	}
	if !contains(c.Facts, want) {
		t.Errorf("levels line %q missing from\n%s", want, strings.Join(c.Facts, "\n"))
	}
	if rawSup, rawRes := goldNearestLevels(daily, lastClose+200); rawSup != nil && rawRes == nil && contains(c.Facts, goldKeyLevelsLine(rawSup, rawRes)) {
		t.Error("the 1h price of the other contract picked the levels")
	}
}

func contains(xs []string, x string) bool {
	for _, s := range xs {
		if s == x {
			return true
		}
	}
	return false
}

// Every way the check can fail to establish the contracts: gold.roll says
// unknown with its reason, and the card reads as it did before the check —
// here that is the raw "above the day range" (see goldCardFrom for why).
func TestGoldRollUnknownReadsAsBefore(t *testing.T) {
	cases := map[string]struct {
		mut    func(s *goldYahooStub)
		reason string
	}{
		"near 404": {func(s *goldYahooStub) { delete(s.series, "GCQ26.CMX") }, goldRollNearMissing},
		"next 404": {func(s *goldYahooStub) { delete(s.series, "GCZ26.CMX") }, goldRollNextMissing},
		"request failed": {func(s *goldYahooStub) {
			s.series["GCQ26.CMX"] = yahooSeries{status: http.StatusInternalServerError}
		}, goldRollFetchFailed},
		"matches both": {func(s *goldYahooStub) {
			gc := s.series["GC=F"]
			s.series["GCQ26.CMX"], s.series["GCZ26.CMX"] = gc, gc
		}, goldRollMatchesBoth},
		"matches neither": {func(s *goldYahooStub) {
			gc := s.series["GC=F"]
			s.series["GCQ26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, 100), h1: shiftBars(gc.h1, 100)}
			s.series["GCZ26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, 200), h1: shiftBars(gc.h1, 200)}
		}, goldRollMatchesNone},
		"hourly bar matches neither": {func(s *goldYahooStub) {
			q := s.series["GCQ26.CMX"]
			q.h1 = shiftBars(q.h1, -10)
			s.series["GCQ26.CMX"] = q
			z := s.series["GCZ26.CMX"]
			z.h1 = shiftBars(z.h1, 10)
			s.series["GCZ26.CMX"] = z
		}, goldRollMatchesNone},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			ag, _ := goldWindowCase(t, tc.mut)
			c := ag.GoldCard(context.Background())
			r := c.Gold.Roll
			if r == nil || r.State != goldRollUnknown || r.Reason == nil || *r.Reason != tc.reason {
				t.Fatalf("roll %+v, want unknown/%s", r, tc.reason)
			}
			if r.CurrentContract != nil || r.NextContract != nil || r.DailyContract != nil || r.HourlyContract != nil {
				t.Errorf("unknown names no contract: %+v", r)
			}
			// As before the check: the position is placed, no roll line.
			if c.Facts[0] != "Last closed 1h price 5155.00 at 2026-07-29 10:00 UTC — above the day range" {
				t.Errorf("price line %q", c.Facts[0])
			}
			for _, f := range c.Facts {
				if strings.HasPrefix(f, "Contract roll") || f == goldNoIdeaRoll || strings.HasPrefix(f, goldLevelsFromDailyClose) {
					t.Errorf("unknown must not read as a window: %q", f)
				}
			}
			if c.Gold.PricePosition == nil || *c.Gold.PricePosition != dayAbove || c.Gold.Idea == nil {
				t.Errorf("unknown must keep the stage-1/2 readout: %+v", c.Gold)
			}
		})
	}
}

// No intraday price: nothing to place, nothing is fetched.
func TestGoldRollNoPriceFetchesNothing(t *testing.T) {
	ag, s := goldWindowCase(t, func(s *goldYahooStub) {
		gc := s.series["GC=F"]
		gc.h1 = nil // an empty 1h answer: the stub serves no bars → the feed is down
		s.series["GC=F"] = gc
	})
	c := ag.GoldCard(context.Background())
	if r := c.Gold.Roll; r == nil || r.State != goldRollUnknown || *r.Reason != goldRollNoPrice {
		t.Fatalf("roll %+v", c.Gold.Roll)
	}
	if n := s.total("GCM26") + s.total("GCQ26") + s.total("GCZ26"); n != 0 {
		t.Errorf("%d contract requests without a price", n)
	}
}

// ── cache and requests ───────────────────────────────────────────────────────

func TestGoldRollRequestsAreCachedPerBar(t *testing.T) {
	ag, s := goldWindowCase(t, nil)
	clock := goldWindowNow
	ag.now = func() time.Time { return clock }
	for i := 0; i < 5; i++ {
		ag.klines = newKlineCache() // as if the 60s candle cache had expired
		if st := ag.GoldCard(context.Background()).Gold.Roll.State; st != goldRollWindow {
			t.Fatalf("read %d: %s", i, st)
		}
	}
	for _, k := range []string{"GCM26.CMX|1d", "GCM26.CMX|1h", "GCQ26.CMX|1d", "GCQ26.CMX|1h", "GCZ26.CMX|1d", "GCZ26.CMX|1h"} {
		if n := s.count(k); n != 1 {
			t.Errorf("%s: %d requests over five reads of one bar, want 1", k, n)
		}
	}

	// A new hourly bar closes: the hourly contracts are asked again, the
	// daily ones are not (their bar has not changed) — and neither is the
	// expired GCM26, whose 404 is remembered (goldRollGoneFor).
	next := goldWindowHour.Add(time.Hour)
	gc := s.series["GC=F"]
	s.set("GC=F", yahooSeries{d1: gc.d1, h1: fixedHourly(next, 5160)})
	s.set("GCZ26.CMX", yahooSeries{d1: s.series["GCZ26.CMX"].d1, h1: fixedHourly(next, 5160)})
	clock = clock.Add(time.Hour)
	ag.klines = newKlineCache()
	if st := ag.GoldCard(context.Background()).Gold.Roll.State; st != goldRollWindow {
		t.Fatalf("next hour: %s", st)
	}
	for k, want := range map[string]int{"GCM26.CMX|1d": 1, "GCQ26.CMX|1d": 1, "GCZ26.CMX|1d": 1, "GCM26.CMX|1h": 1, "GCQ26.CMX|1h": 2, "GCZ26.CMX|1h": 2} {
		if n := s.count(k); n != want {
			t.Errorf("after a new hourly bar %s: %d requests, want %d", k, n, want)
		}
	}
}

// An unknown answer is asked again only after goldRollRetry, under the same
// bars; the retry picks up a contract that came back.
func TestGoldRollUnknownIsRetried(t *testing.T) {
	ag, s := goldWindowCase(t, func(s *goldYahooStub) {
		s.series["GCQ26.CMX"] = yahooSeries{status: http.StatusBadGateway}
	})
	clock := goldWindowNow
	ag.now = func() time.Time { return clock }
	read := func() GoldRoll {
		ag.klines = newKlineCache()
		return *ag.GoldCard(context.Background()).Gold.Roll
	}
	if r := read(); r.State != goldRollUnknown {
		t.Fatalf("%+v", r)
	}
	clock = clock.Add(goldRollRetry - time.Minute)
	read()
	if n := s.count("GCQ26.CMX|1d"); n != 1 {
		t.Errorf("retried before goldRollRetry: %d requests", n)
	}
	s.set("GCQ26.CMX", yahooSeries{d1: s.series["GC=F"].d1, h1: shiftBars(s.series["GC=F"].h1, -goldWindowSpread)})
	clock = clock.Add(2 * time.Minute)
	if r := read(); r.State != goldRollWindow {
		t.Errorf("after the retry: %+v", r)
	}
}

// ── the push hook ────────────────────────────────────────────────────────────

func goldHookHash(t *testing.T, c Card) string {
	t.Helper()
	b, err := json.Marshal(cardEnvelope(c))
	if err != nil {
		t.Fatal(err)
	}
	n, err := hookNormalize(keyGold, b, goldWindowNow, goldWindowNow)
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(n)
	return string(h[:])
}

// Same data, same body; a change of roll state is a new body (an event).
func TestGoldRollHookBody(t *testing.T) {
	ag, _ := goldWindowCase(t, nil)
	first := goldHookHash(t, ag.GoldCard(context.Background()))
	ag.klines = newKlineCache()
	if goldHookHash(t, ag.GoldCard(context.Background())) != first {
		t.Error("the body moved under unchanged data")
	}
	// The same GC=F bars, but both contracts answer GC=F's own bars: the
	// check establishes nothing — unknown.
	ag2, _ := goldWindowCase(t, func(s *goldYahooStub) {
		gc := s.series["GC=F"]
		s.series["GCQ26.CMX"] = yahooSeries{d1: gc.d1, h1: gc.h1}
		s.series["GCZ26.CMX"] = yahooSeries{d1: gc.d1, h1: gc.h1}
	})
	if goldHookHash(t, ag2.GoldCard(context.Background())) == first {
		t.Error("window → unknown must change the body")
	}
	ag3, _ := goldWindowCase(t, func(s *goldYahooStub) {
		gc := s.series["GC=F"]
		s.series["GCQ26.CMX"] = yahooSeries{d1: gc.d1, h1: gc.h1} // both bars on Q26: none
		s.series["GCZ26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, goldWindowSpread), h1: shiftBars(gc.h1, goldWindowSpread)}
	})
	c3 := ag3.GoldCard(context.Background())
	if c3.Gold.Roll.State != goldRollNone || goldHookHash(t, c3) == first {
		t.Errorf("window → none must change the body (state %s)", c3.Gold.Roll.State)
	}
}

// ── the budget ───────────────────────────────────────────────────────────────

// Every line of every path fits goldFactMaxRunes, the window paths included:
// five-digit prices, stale, established levels with two-digit pivot counts,
// one-sided levels, every regime, no range, no macro.
func TestGoldRollTextFitsOneLine(t *testing.T) {
	cur, nxt := "GCQ26", "GCZ26"
	win := GoldRoll{State: goldRollWindow, CurrentContract: &cur, NextContract: &nxt, DailyContract: &cur, HourlyContract: &nxt}
	cards := goldAllCards()
	for key, in := range goldAllInputs() {
		in.roll = win
		cards["window/"+key] = goldCardFrom(in)
		r := goldRollNoPrice
		in.roll = GoldRoll{State: goldRollUnknown, Reason: &r}
		cards["unknown/"+key] = goldCardFrom(in)
	}
	for key, c := range cards {
		for _, l := range goldLines(c) {
			if n := utf8.RuneCountInString(l); n > goldFactMaxRunes {
				t.Errorf("%s: %d runes > %d: %q", key, n, goldFactMaxRunes, l)
			}
		}
	}
	for _, l := range []string{goldSplicedLine, goldNoIdeaRoll} {
		if n := utf8.RuneCountInString(l); n > goldFactMaxRunes {
			t.Errorf("%d runes: %q", n, l)
		}
	}
}

// goldAllInputs is goldAllCards' input set plus one-sided worst-case levels.
func goldAllInputs() map[string]goldInputs {
	out := map[string]goldInputs{}
	for _, st := range []string{trendUp, trendDown, trendGrey, trendFlat, trendConflict} {
		for _, stale := range []bool{false, true} {
			for _, px := range []float64{4354.9, 99999.9} {
				in := goldFixture(st)
				in.px = px
				if stale {
					in.pxAt = goldNow.Add(-30 * time.Hour)
				}
				sup, res := SRLevel{Raw: 99888.8, Touches: 12}, SRLevel{Raw: 99999.95, Touches: 11}
				key := st + "/" + goldPx(px)
				if stale {
					key += "/stale"
				}
				in.sup, in.res = &sup, &res
				out[key+"/both"] = in
				in.res = nil
				out[key+"/sup"] = in
				in.sup, in.res = nil, &res
				out[key+"/res"] = in
				in2 := in
				in2.levels, in2.macro = goldDayLevels{}, Card{}
				out[key+"/bare"] = in2
				in3 := in
				in3.levels = goldDayLevels{High: 12345.6, Low: 10234.5, Defined: true, Unwound: 3, BarTime: in.levels.BarTime}
				out[key+"/nested"] = in3
			}
		}
	}
	return out
}

// No forecast or advice words on the new paths either.
func TestGoldRollBannedWords(t *testing.T) {
	ag, _ := goldWindowCase(t, nil)
	c := ag.GoldCard(context.Background())
	text := strings.ToLower(strings.Join(goldLines(c), "\n"))
	for _, b := range []string{"will ", "expect", "bias", "should", "buy", "sell", "long", "short", "entry", "target", "likely", "probab"} {
		if strings.Contains(text, b) {
			t.Errorf("%q on a window card:\n%s", b, text)
		}
	}
}
