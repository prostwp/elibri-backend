package demobot

// gold_roll_cands_test.go — the roll check with three candidate contracts
// (previous, near, next), either order of a window, the log line on a change
// of state, the daily and hourly rounds running side by side, and which GC=F
// and contract rows can take part in a match. Offline: goldYahooStub.

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// goldNoPlacement fails when any line of the card places the 1h price
// against daily data, or the blocks call a defined range undefined.
func goldNoPlacement(t *testing.T, c Card) {
	t.Helper()
	text := strings.Join(goldLines(c), "\n")
	for _, bad := range []string{"above the day range", "below the day range", "inside the day range", "already above", "already below", "no day range defined"} {
		if strings.Contains(text, bad) {
			t.Errorf("%q on a card whose 1h price is on another contract:\n%s", bad, text)
		}
	}
	if c.Gold.PricePosition != nil || c.Gold.Idea != nil {
		t.Errorf("price_position %v idea %+v, want both null", c.Gold.PricePosition, c.Gold.Idea)
	}
	if c.Blocks == nil || !strings.HasPrefix(c.Blocks.WhyLevel, "Day range = high/low") {
		t.Errorf("why_level must still describe the defined range: %+v", c.Blocks)
	}
}

// The daily bar of the 1st of the delivery month still on the contract whose
// month has just begun (Yahoo switching the daily series a day later than on
// 2025-08-01), the hourly bar already on the next one. With the candidates
// taken only after the bar's month this matched neither — unknown, and the
// raw "above the day range" came back on the last day of the window.
var (
	goldFirstDay     = time.Date(2026, 4, 1, 4, 0, 0, 0, time.UTC) // Wednesday, candidates J26/M26/Q26
	goldFirstDayHour = time.Date(2026, 4, 2, 9, 0, 0, 0, time.UTC)
	goldFirstDayNow  = time.Date(2026, 4, 2, 10, 30, 0, 0, time.UTC)
)

func goldFirstDayCase(t *testing.T, hourlyOnPrevious bool) Card {
	t.Helper()
	daily := dailyEnding(goldFirstDay, 260, risingDay) // GCJ26's daily bars
	h1 := fixedHourly(goldFirstDayHour, 5155)
	s := newGoldYahooStub(t)
	s.set("GC=F", yahooSeries{d1: daily, h1: h1})
	if hourlyOnPrevious {
		s.set("GCJ26.CMX", yahooSeries{d1: daily, h1: h1})
		s.set("GCM26.CMX", yahooSeries{d1: shiftBars(daily, goldWindowSpread), h1: shiftBars(h1, goldWindowSpread)})
	} else {
		s.set("GCJ26.CMX", yahooSeries{d1: daily, h1: shiftBars(h1, -goldWindowSpread)})
		s.set("GCM26.CMX", yahooSeries{d1: shiftBars(daily, goldWindowSpread), h1: h1})
	}
	s.set("GCQ26.CMX", yahooSeries{d1: shiftBars(daily, 3*goldWindowSpread), h1: shiftBars(h1, 3*goldWindowSpread)})
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	ag.now = func() time.Time { return goldFirstDayNow }
	return ag.GoldCard(context.Background())
}

func TestGoldRollDailyBarOnPreviousContract(t *testing.T) {
	c := goldFirstDayCase(t, false)
	r := c.Gold.Roll
	if r == nil || r.State != goldRollWindow || *r.Reason != goldRollHourlyAhead ||
		*r.CurrentContract != "GCJ26" || *r.NextContract != "GCM26" || *r.DailyContract != "GCJ26" || *r.HourlyContract != "GCM26" {
		t.Fatalf("roll %+v %v, want window GCJ26 → GCM26", r, goldRollFields(r))
	}
	goldNoPlacement(t, c)
	for _, want := range []string{
		"Last closed 1h price 5155.00 at 2026-04-02 10:00 UTC — on contract GCM26",
		"Contract roll GCJ26 → GCM26: 1h price on GCM26, day range on GCJ26; price not placed against it",
		goldNoIdeaRoll,
	} {
		if !contains(c.Facts, want) {
			t.Errorf("missing %q in\n%s", want, strings.Join(c.Facts, "\n"))
		}
	}

	// Both bars on the previous contract: established "none", not unknown.
	c = goldFirstDayCase(t, true)
	r = c.Gold.Roll
	if r == nil || r.State != goldRollNone || r.Reason != nil || *r.CurrentContract != "GCJ26" || *r.NextContract != "GCM26" {
		t.Fatalf("roll %+v %v, want none on GCJ26", r, goldRollFields(r))
	}
}

func goldRollFields(r *GoldRoll) string {
	if r == nil {
		return "nil"
	}
	return fmt.Sprintf("reason=%s cur=%s next=%s daily=%s hourly=%s",
		goldDeref(r.Reason), goldDeref(r.CurrentContract), goldDeref(r.NextContract), goldDeref(r.DailyContract), goldDeref(r.HourlyContract))
}

// The hourly bar on an EARLIER contract than the daily one: the two are still
// prices of two contracts, so the card places nothing; the reason says which
// way round.
func TestGoldRollHourlyBehindDaily(t *testing.T) {
	ag, _ := goldWindowCase(t, func(s *goldYahooStub) {
		gc := s.series["GC=F"]
		s.series["GCQ26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, -goldWindowSpread), h1: gc.h1}
		s.series["GCZ26.CMX"] = yahooSeries{d1: gc.d1, h1: shiftBars(gc.h1, goldWindowSpread)}
	})
	c := ag.GoldCard(context.Background())
	r := c.Gold.Roll
	if r == nil || r.State != goldRollWindow || *r.Reason != goldRollHourlyBehind ||
		*r.CurrentContract != "GCQ26" || *r.NextContract != "GCZ26" || *r.DailyContract != "GCZ26" || *r.HourlyContract != "GCQ26" {
		t.Fatalf("roll %+v %v", r, goldRollFields(r))
	}
	goldNoPlacement(t, c)
	if !contains(c.Facts, "Contract roll GCQ26 → GCZ26: 1h price on GCQ26, day range on GCZ26; price not placed against it") {
		t.Errorf("roll line missing:\n%s", strings.Join(c.Facts, "\n"))
	}
	if c.Blocks.WhatHappened != "Snapshot, not an event: 1d regime confirmed uptrend; last 1h close 5155.00 on GCQ26 (contract roll)" {
		t.Errorf("what_happened %q", c.Blocks.WhatHappened)
	}
}

// Three candidates: two carrying the bar is unknown; the previous contract's
// 404 (expired) does not count, a failed request for it does when nothing
// matched, and does not when the bar matched a served candidate.
func TestGoldRollPreviousCandidate(t *testing.T) {
	cases := map[string]struct {
		mut    func(s *goldYahooStub)
		state  string
		reason string
	}{
		"previous and near both carry the daily bar": {func(s *goldYahooStub) {
			s.series["GCM26.CMX"] = s.series["GCQ26.CMX"]
		}, goldRollUnknown, goldRollMatchesBoth},
		"previous request failed, bar matched near": {func(s *goldYahooStub) {
			s.series["GCM26.CMX"] = yahooSeries{status: http.StatusBadGateway}
		}, goldRollWindow, goldRollHourlyAhead},
		"previous request failed, nothing matched": {func(s *goldYahooStub) {
			gc := s.series["GC=F"]
			s.series["GCM26.CMX"] = yahooSeries{status: http.StatusBadGateway}
			s.series["GCQ26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, 100), h1: shiftBars(gc.h1, 100)}
			s.series["GCZ26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, 200), h1: shiftBars(gc.h1, 200)}
		}, goldRollUnknown, goldRollFetchFailed},
		"previous not served (expired), nothing matched": {func(s *goldYahooStub) {
			gc := s.series["GC=F"]
			s.series["GCQ26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, 100), h1: shiftBars(gc.h1, 100)}
			s.series["GCZ26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, 200), h1: shiftBars(gc.h1, 200)}
		}, goldRollUnknown, goldRollMatchesNone},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			ag, _ := goldWindowCase(t, tc.mut)
			r := ag.GoldCard(context.Background()).Gold.Roll
			if r == nil || r.State != tc.state || goldDeref(r.Reason) != tc.reason {
				t.Fatalf("roll %+v %v, want %s/%s", r, goldRollFields(r), tc.state, tc.reason)
			}
		})
	}
}

// ── rows that take part in a match ───────────────────────────────────────────

// risingHourly is n hourly bars with distinct closes, the last opening at last.
func risingHourly(last time.Time, n int, base float64) []types.OHLCVCandle {
	out := make([]types.OHLCVCandle, n)
	for i := range out {
		c := base + float64(i)
		out[i] = types.OHLCVCandle{Time: last.Unix() - int64(n-1-i)*3600, Open: c - 0.5, High: c + 1, Low: c - 1, Close: c}
	}
	return out
}

// Only GC=F's last CLOSED bar is matched, and only against the contract bar
// of the same stamp. Around the real clock: GC=F's 1h answer ends with the
// forming bar and Yahoo's current-quote row, and so do the contracts'. The
// near contract's forming and quote rows equal GC=F's — if either took part,
// the hourly bar would read as near; it reads as next, whose closed bar of
// the same stamp is GC=F's.
func TestGoldRollMatchesClosedBarsOnly(t *testing.T) {
	now := time.Now().UTC()
	hour := now.Truncate(time.Hour)
	if now.Sub(hour) < 2*time.Second {
		time.Sleep(2 * time.Second) // keep the quote row strictly inside the forming hour
		now = time.Now().UTC()
	}
	quoteAt := now.Add(-time.Second).Truncate(time.Second)
	if quoteAt.Unix()%1800 == 0 {
		quoteAt = quoteAt.Add(time.Second)
	}
	closed := risingHourly(hour.Add(-time.Hour), 8, 5150)
	forming := types.OHLCVCandle{Time: hour.Unix(), Open: 5170, High: 5172, Low: 5169, Close: 5171}
	quote := types.OHLCVCandle{Time: quoteAt.Unix(), Open: 5171, High: 5171, Low: 5171, Close: 5171}
	gcH := append(append([]types.OHLCVCandle{}, closed...), forming, quote)

	dayOpen := time.Date(hour.Year(), hour.Month(), hour.Day(), 4, 0, 0, 0, time.UTC).AddDate(0, 0, -2)
	daily := dailyEnding(dayOpen, 260, risingDay)
	cands := goldCandidates(time.Unix(daily[len(daily)-1].Time, 0).UTC())
	near, next := goldYahooSymbol(cands[goldCandNear]), goldYahooSymbol(cands[goldCandNext])

	s := newGoldYahooStub(t)
	s.set("GC=F", yahooSeries{d1: daily, h1: gcH})
	s.set(near, yahooSeries{d1: daily, h1: append(shiftBars(closed, -goldWindowSpread), forming, quote)})
	s.set(next, yahooSeries{d1: shiftBars(daily, goldWindowSpread), h1: append(append([]types.OHLCVCandle{}, closed...),
		shiftBars([]types.OHLCVCandle{forming, quote}, goldWindowSpread)...)})
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	ag.now = func() time.Time { return now }
	c := ag.GoldCard(context.Background())

	if c.Gold.Price == nil || *c.Gold.Price != closed[len(closed)-1].Close {
		t.Fatalf("price %v, want the last closed bar's close %v", c.Gold.Price, closed[len(closed)-1].Close)
	}
	r := c.Gold.Roll
	if r == nil || r.State != goldRollWindow || *r.HourlyContract != cands[goldCandNext] || *r.DailyContract != cands[goldCandNear] {
		t.Fatalf("roll %+v %v: the forming bar or the quote row took part", r, goldRollFields(r))
	}
}

// Pairing is by stamp only. The next contract carries GC=F's closes one hour
// LATER (its bar at GC=F's last stamp is GC=F's previous close, the bar at the
// same position of the answer and the nearest later stamp carry GC=F's last
// close): nothing matches, and the check says so.
func TestGoldRollMatchesByStampOnly(t *testing.T) {
	ag, _ := goldWindowCase(t, func(s *goldYahooStub) {
		gcH := risingHourly(goldWindowHour, 8, 5150)
		gc := s.series["GC=F"]
		s.series["GC=F"] = yahooSeries{d1: gc.d1, h1: gcH}
		later := make([]types.OHLCVCandle, len(gcH))
		for i, b := range gcH {
			b.Time += 3600
			later[i] = b
		}
		s.series["GCZ26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, goldWindowSpread), h1: later}
		s.series["GCQ26.CMX"] = yahooSeries{d1: gc.d1, h1: shiftBars(gcH, -goldWindowSpread)}
	})
	r := ag.GoldCard(context.Background()).Gold.Roll
	if r == nil || r.State != goldRollUnknown || goldDeref(r.Reason) != goldRollMatchesNone {
		t.Fatalf("roll %+v %v, want unknown/%s", r, goldRollFields(r), goldRollMatchesNone)
	}
}

// A quote row that outlived the closed-bar cut (a fetch long after the last
// trade) is the card's price as before, but it is not a bar: it is not
// matched, no contract is asked for 1h bars, and the state is unknown.
func TestGoldRollQuoteRowIsNotMatched(t *testing.T) {
	quote := goldWindowHour.Add(17*time.Minute + 33*time.Second)
	ag, s := goldWindowCase(t, func(s *goldYahooStub) {
		gc := s.series["GC=F"]
		h1 := append(append([]types.OHLCVCandle{}, gc.h1...), types.OHLCVCandle{Time: quote.Unix(), Open: 5155, High: 5155, Low: 5155, Close: 5155})
		s.series["GC=F"] = yahooSeries{d1: gc.d1, h1: h1}
		z := s.series["GCZ26.CMX"]
		s.series["GCZ26.CMX"] = yahooSeries{d1: z.d1, h1: h1}
	})
	ag.now = func() time.Time { return quote.Add(2 * time.Hour) }
	c := ag.GoldCard(context.Background())
	if r := c.Gold.Roll; r == nil || r.State != goldRollUnknown || goldDeref(r.Reason) != goldRollQuoteRow {
		t.Fatalf("roll %+v %v", r, goldRollFields(r))
	}
	if n := s.count("GCQ26.CMX|1h") + s.count("GCZ26.CMX|1h") + s.count("GCM26.CMX|1h"); n != 0 {
		t.Errorf("%d contract 1h requests for a quote row", n)
	}
}

// ── rounds side by side ──────────────────────────────────────────────────────

// The first daily and the first hourly contract request are held until both
// have arrived: run one after the other, the first would wait out the
// deadline and the card would be unknown.
func TestGoldRollDailyAndHourlyRunTogether(t *testing.T) {
	ag, s := goldWindowCase(t, nil)
	var once1d, once1h sync.Once
	saw1d, saw1h := make(chan struct{}), make(chan struct{})
	var mu sync.Mutex
	timedOut := false
	s.before = func(sym, iv string) {
		if sym == "GC=F" {
			return
		}
		if iv == "1d" {
			once1d.Do(func() { close(saw1d) })
		} else {
			once1h.Do(func() { close(saw1h) })
		}
		for _, ch := range []chan struct{}{saw1d, saw1h} {
			select {
			case <-ch:
			case <-time.After(3 * time.Second):
				mu.Lock()
				timedOut = true
				mu.Unlock()
				return
			}
		}
	}
	c := ag.GoldCard(context.Background())
	mu.Lock()
	defer mu.Unlock()
	if timedOut {
		t.Error("the daily and hourly contract requests did not overlap")
	}
	if c.Gold.Roll.State != goldRollWindow {
		t.Errorf("roll %+v", c.Gold.Roll)
	}
}

// ── the log ──────────────────────────────────────────────────────────────────

// One line per change: the first state, none/window again only when the
// state or a contract changes (not per bar), unknown once per reason and
// pair of bars — a retry that fails the same way under the same bars adds
// nothing.
func TestGoldRollLogsOncePerChange(t *testing.T) {
	ag, s := goldWindowCase(t, nil)
	var mu sync.Mutex
	var lines []string
	ag.goldRoll.logf = func(format string, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		lines = append(lines, fmt.Sprintf(format, args...))
	}
	clock := goldWindowNow
	ag.now = func() time.Time { return clock }
	read := func() GoldRoll {
		ag.klines = newKlineCache()
		return *ag.GoldCard(context.Background()).Gold.Roll
	}
	gcD := s.series["GC=F"].d1
	zD := s.series["GCZ26.CMX"].d1
	qD := s.series["GCQ26.CMX"].d1

	read()
	read()
	// A new hourly bar, still a window: no line.
	h2 := goldWindowHour.Add(time.Hour)
	s.set("GC=F", yahooSeries{d1: gcD, h1: fixedHourly(h2, 5160)})
	s.set("GCQ26.CMX", yahooSeries{d1: qD, h1: shiftBars(fixedHourly(h2, 5160), -goldWindowSpread)})
	s.set("GCZ26.CMX", yahooSeries{d1: zD, h1: fixedHourly(h2, 5160)})
	clock = clock.Add(time.Hour)
	read()
	// The next bar: the request for next fails — unknown, one line; the retry
	// after goldRollRetry fails the same way: no line.
	h3 := h2.Add(time.Hour)
	s.set("GC=F", yahooSeries{d1: gcD, h1: fixedHourly(h3, 5165)})
	s.set("GCQ26.CMX", yahooSeries{d1: qD, h1: shiftBars(fixedHourly(h3, 5165), -goldWindowSpread)})
	s.set("GCZ26.CMX", yahooSeries{status: http.StatusInternalServerError})
	clock = clock.Add(time.Hour)
	if r := read(); r.State != goldRollUnknown {
		t.Fatalf("%+v", r)
	}
	clock = clock.Add(goldRollRetry + time.Minute)
	read()
	if n := s.count("GCZ26.CMX|1h"); n != 4 {
		t.Errorf("GCZ26 1h asked %d times, want 4 (three bars and one retry)", n)
	}
	// It comes back: window again, one line.
	s.set("GCZ26.CMX", yahooSeries{d1: zD, h1: fixedHourly(h3, 5165)})
	clock = clock.Add(goldRollRetry + time.Minute)
	if r := read(); r.State != goldRollWindow {
		t.Fatalf("%+v", r)
	}

	mu.Lock()
	defer mu.Unlock()
	want := []string{
		"[demobot] gold roll: window GCQ26 → GCZ26 (hourly_ahead_of_daily), 1d bar on GCQ26, 1h bar on GCZ26 (1d bar 2026-07-28 UTC, 1h bar 2026-07-29 09:00 UTC)",
		"[demobot] gold roll: unknown (request_failed: yahoo chart GCZ26.CMX: HTTP 500) (1d bar 2026-07-28 UTC, 1h bar 2026-07-29 11:00 UTC)",
		"[demobot] gold roll: window GCQ26 → GCZ26 (hourly_ahead_of_daily), 1d bar on GCQ26, 1h bar on GCZ26 (1d bar 2026-07-28 UTC, 1h bar 2026-07-29 11:00 UTC)",
	}
	if strings.Join(lines, "\n") != strings.Join(want, "\n") {
		t.Errorf("log:\n%s\nwant:\n%s", strings.Join(lines, "\n"), strings.Join(want, "\n"))
	}
}
