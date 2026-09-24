package demobot

// gold_roll_load_test.go — how many contract requests the roll check sends
// to Yahoo (which answers 429 when asked too often). Offline: the counting
// stub of gold_roll_golden_test.go; the reads are driven through goldRollOf
// with a stepped clock, one read a minute, as the push hook reads the card.

import (
	"context"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// A gold weekend: the last bars close Friday 21:00 UTC and the next one
// closes Sunday 23:00 UTC — 50 hours, 3000 reads, under the same two bars.
// Candidates of a July day: GCM26 (previous, expired), GCQ26, GCZ26.
var (
	goldWeekendFrom  = time.Date(2026, 7, 24, 21, 0, 0, 0, time.UTC)
	goldWeekendReads = 50 * 60
	goldWeekendDay   = time.Date(2026, 7, 23, 4, 0, 0, 0, time.UTC)
	goldWeekendHour  = time.Date(2026, 7, 24, 20, 0, 0, 0, time.UTC)
)

// goldContractRequests is every request for a dated contract (GC=F excluded).
func goldContractRequests(s *goldYahooStub) int {
	return s.total("GCM26") + s.total("GCQ26") + s.total("GCZ26")
}

// goldWeekendRequests reads the roll once a minute over the weekend with the
// contracts answering as mut scripts them, and returns the contract requests.
func goldWeekendRequests(t *testing.T, mut func(s *goldYahooStub, d1, h1 []types.OHLCVCandle)) (int, *goldYahooStub, GoldRoll) {
	t.Helper()
	d1 := dailyEnding(goldWeekendDay, 30, risingDay)
	h1 := fixedHourly(goldWeekendHour, 2300)
	s := newGoldYahooStub(t)
	mut(s, d1, h1)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	ag.goldRoll.logf = func(string, ...any) {}
	clock := goldWeekendFrom
	ag.now = func() time.Time { return clock }
	var last GoldRoll
	for i := 0; i < goldWeekendReads; i++ {
		last = ag.goldRollOf(context.Background(), d1, h1[len(h1)-1], true)
		clock = clock.Add(time.Minute)
	}
	return goldContractRequests(s), s, last
}

// Before this change every unknown was asked again every 15 minutes, three
// requests a side: 200 rounds × 3 × 2 sides = 1200 in every case below. Now
// a failed request (5xx here) still is; every other unknown doubles its
// pause up to goldRollRetryMax — 28 rounds a side — and an expired previous
// contract is asked once a day instead of every round.
func TestGoldRollWeekendRequests(t *testing.T) {
	neither := func(s *goldYahooStub, d1, h1 []types.OHLCVCandle) {
		s.set("GCQ26.CMX", yahooSeries{d1: shiftBars(d1, 100), h1: shiftBars(h1, 100)})
		s.set("GCZ26.CMX", yahooSeries{d1: shiftBars(d1, 200), h1: shiftBars(h1, 200)})
	}
	status := func(code int) func(s *goldYahooStub, d1, h1 []types.OHLCVCandle) {
		return func(s *goldYahooStub, d1, h1 []types.OHLCVCandle) {
			for _, c := range []string{"GCM26.CMX", "GCQ26.CMX", "GCZ26.CMX"} {
				s.set(c, yahooSeries{status: code})
			}
		}
	}
	cases := []struct {
		name   string
		mut    func(s *goldYahooStub, d1, h1 []types.OHLCVCandle)
		reason string
		want   int
	}{
		{"matches neither, previous expired (404)", neither, goldRollMatchesNone, 118},
		{"matches neither, all three served", func(s *goldYahooStub, d1, h1 []types.OHLCVCandle) {
			neither(s, d1, h1)
			s.set("GCM26.CMX", yahooSeries{d1: shiftBars(d1, -100), h1: shiftBars(h1, -100)})
		}, goldRollMatchesNone, 168},
		{"near not served (404)", func(s *goldYahooStub, d1, h1 []types.OHLCVCandle) {
			s.set("GCZ26.CMX", yahooSeries{d1: shiftBars(d1, 200), h1: shiftBars(h1, 200)})
		}, goldRollNearMissing, 118},
		{"429 from every contract", status(http.StatusTooManyRequests), goldRollFetchFailed, 168},
		{"500 from every contract", status(http.StatusInternalServerError), goldRollFetchFailed, 1200},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n, _, r := goldWeekendRequests(t, tc.mut)
			if r.State != goldRollUnknown || goldDeref(r.Reason) != tc.reason {
				t.Fatalf("roll %+v %s, want unknown/%s", r, goldDeref(r.Reason), tc.reason)
			}
			t.Logf("%s: %d contract requests over the weekend", tc.name, n)
			if n != tc.want {
				t.Errorf("%d contract requests over the weekend, want %d", n, tc.want)
			}
		})
	}
}

// A trading day, outside any roll: from Tuesday 22:00 UTC (the session open)
// to Wednesday 22:00, one read a minute. The last closed 1h bar changes on
// every hour but the 21:00 break (24 bars in view), the last closed 1d bar
// once, at the 21:00 close (2 bars in view). Both bars on GCQ26, GCZ26 50
// higher, GCM26 expired (404).
func goldTradingDayRequests(t *testing.T) (int, *goldYahooStub) {
	t.Helper()
	from := time.Date(2026, 7, 21, 22, 0, 0, 0, time.UTC)
	var h1 []types.OHLCVCandle
	for ts := from.Add(-26 * time.Hour); ts.Before(from.Add(24 * time.Hour)); ts = ts.Add(time.Hour) {
		if ts.Hour() == 21 {
			continue
		}
		px := 2300 + float64(len(h1))
		h1 = append(h1, types.OHLCVCandle{Time: ts.Unix(), Open: px, High: px + 1, Low: px - 1, Close: px})
	}
	d1 := dailyEnding(time.Date(2026, 7, 22, 4, 0, 0, 0, time.UTC), 30, risingDay)
	s := newGoldYahooStub(t)
	s.set("GCQ26.CMX", yahooSeries{d1: d1, h1: h1})
	s.set("GCZ26.CMX", yahooSeries{d1: shiftBars(d1, 50), h1: shiftBars(h1, 50)})
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	ag.goldRoll.logf = func(string, ...any) {}
	clock := from
	ag.now = func() time.Time { return clock }
	for i := 0; i < 24*60; i++ {
		var hour types.OHLCVCandle
		for _, b := range h1 {
			if b.Time+3600 <= clock.Unix() {
				hour = b
			}
		}
		days := d1[:len(d1)-1] // Tuesday's bar is the last closed one …
		if clock.Hour() >= 21 && clock.Day() == 22 {
			days = d1 // … until Wednesday's closes at 21:00
		}
		if r := ag.goldRollOf(context.Background(), days, hour, true); r.State != goldRollNone {
			t.Fatalf("%s: roll %+v %s", clock, r, goldDeref(r.Reason))
		}
		clock = clock.Add(time.Minute)
	}
	return goldContractRequests(s), s
}

func TestGoldRollTradingDayRequests(t *testing.T) {
	// Before: 24 hourly and 2 daily bars × 3 contracts = 78, GCM26 26 of
	// them. Now GCM26 is asked by the first two rounds (side by side) and
	// then remembered for a day: 2 + 26 × 2 = 54.
	n, s := goldTradingDayRequests(t)
	if n != 54 || s.total("GCM26") != 2 {
		t.Errorf("trading day: %d contract requests, GCM26 %d; want 54 and 2", n, s.total("GCM26"))
	}
}

// ── the pause under the same bars ────────────────────────────────────────────

// goldRollRounds reads the window case's two bars once a minute for n
// minutes and returns the minutes at which GCQ26's daily series was asked —
// one per round of the daily side.
func goldRollRounds(t *testing.T, mut func(s *goldYahooStub), n int) []int {
	t.Helper()
	ag, s := goldWindowCase(t, mut)
	ag.goldRoll.logf = func(string, ...any) {}
	gc := s.series["GC=F"]
	clock := goldWindowNow
	ag.now = func() time.Time { return clock }
	var at []int
	for i := 0; i < n; i++ {
		before := s.count("GCQ26.CMX|1d")
		ag.goldRollOf(context.Background(), gc.d1, gc.h1[len(gc.h1)-1], true)
		if s.count("GCQ26.CMX|1d") != before {
			at = append(at, i)
		}
		clock = clock.Add(time.Minute)
	}
	return at
}

// Yahoo answered and the bar matched nothing, or it refused the request
// (429, 403, 401): the pause doubles, 15 → 30 → 60 → 120 minutes, and stays
// at 120. A failed request (5xx) is asked again every 15 minutes, as before.
func TestGoldRollUnknownPauseGrows(t *testing.T) {
	neither := func(s *goldYahooStub) {
		gc := s.series["GC=F"]
		s.series["GCQ26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, 100), h1: shiftBars(gc.h1, 100)}
		s.series["GCZ26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, 200), h1: shiftBars(gc.h1, 200)}
	}
	status := func(code int) func(s *goldYahooStub) {
		return func(s *goldYahooStub) {
			s.series["GCQ26.CMX"] = yahooSeries{status: code}
			s.series["GCZ26.CMX"] = yahooSeries{status: code}
		}
	}
	growing := []int{0, 15, 45, 105, 225, 345, 465}
	var every15 []int
	for m := 0; m <= 465; m += 15 {
		every15 = append(every15, m)
	}
	cases := map[string]struct {
		mut  func(s *goldYahooStub)
		want []int
	}{
		"matches neither": {neither, growing},
		"near not served": {func(s *goldYahooStub) { delete(s.series, "GCQ26.CMX") }, growing},
		"429":             {status(http.StatusTooManyRequests), growing},
		"403":             {status(http.StatusForbidden), growing},
		"401":             {status(http.StatusUnauthorized), growing},
		"500":             {status(http.StatusInternalServerError), every15},
		"502":             {status(http.StatusBadGateway), every15},
		"matches both":    {func(s *goldYahooStub) { gc := s.series["GC=F"]; s.series["GCQ26.CMX"], s.series["GCZ26.CMX"] = gc, gc }, growing},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if got := goldRollRounds(t, tc.mut, 466); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("rounds at minutes %v, want %v", got, tc.want)
			}
		})
	}
}

// A new bar is asked at once, whatever the pause under the old one, and its
// own unknowns start again from 15 minutes; the other side keeps its pause.
func TestGoldRollNewBarResetsThePause(t *testing.T) {
	ag, s := goldWindowCase(t, func(s *goldYahooStub) {
		gc := s.series["GC=F"]
		s.series["GCQ26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, 100), h1: shiftBars(gc.h1, 100)}
		s.series["GCZ26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, 200), h1: shiftBars(gc.h1, 200)}
	})
	ag.goldRoll.logf = func(string, ...any) {}
	d1 := s.series["GC=F"].d1
	hour := s.series["GC=F"].h1[len(s.series["GC=F"].h1)-1]
	clock := goldWindowNow
	ag.now = func() time.Time { return clock }
	var at1h, at1d []int
	for i := 0; i < 300; i++ {
		if i == 230 { // a new hourly bar, still matching nothing; the pause was 120
			next := goldWindowHour.Add(time.Hour)
			h1 := fixedHourly(next, 5160)
			s.set("GCQ26.CMX", yahooSeries{d1: shiftBars(d1, 100), h1: shiftBars(h1, 100)})
			s.set("GCZ26.CMX", yahooSeries{d1: shiftBars(d1, 200), h1: shiftBars(h1, 200)})
			hour = h1[len(h1)-1]
		}
		n1h, n1d := s.count("GCQ26.CMX|1h"), s.count("GCQ26.CMX|1d")
		ag.goldRollOf(context.Background(), d1, hour, true)
		if s.count("GCQ26.CMX|1h") != n1h {
			at1h = append(at1h, i)
		}
		if s.count("GCQ26.CMX|1d") != n1d {
			at1d = append(at1d, i)
		}
		clock = clock.Add(time.Minute)
	}
	if want := []int{0, 15, 45, 105, 225, 230, 245, 275}; !reflect.DeepEqual(at1h, want) {
		t.Errorf("1h rounds at %v, want %v", at1h, want)
	}
	if want := []int{0, 15, 45, 105, 225}; !reflect.DeepEqual(at1d, want) {
		t.Errorf("1d rounds at %v, want %v", at1d, want)
	}
}

// ── the previous contract's 404 ──────────────────────────────────────────────

// goldRollHourAt moves the window case to a new hourly bar opening at h:
// GC=F's close 5155 on GCZ26, GCQ26 50 lower — still a window.
func goldRollHourAt(s *goldYahooStub, h time.Time) types.OHLCVCandle {
	h1 := fixedHourly(h, 5155)
	s.set("GCQ26.CMX", yahooSeries{d1: s.series["GCQ26.CMX"].d1, h1: shiftBars(h1, -goldWindowSpread)})
	s.set("GCZ26.CMX", yahooSeries{d1: s.series["GCZ26.CMX"].d1, h1: h1})
	return h1[len(h1)-1]
}

// An expired previous contract (404) is asked on the first bar, not on the
// bars after it, and again once goldRollGoneFor has run out. The card is the
// same either way. Near's 404 is not remembered; neither is a failed request
// for previous.
func TestGoldRollExpiredContractRemembered(t *testing.T) {
	t.Run("previous 404", func(t *testing.T) {
		ag, s := goldWindowCase(t, nil)
		ag.goldRoll.logf = func(string, ...any) {}
		d1 := s.series["GC=F"].d1
		clock := goldWindowNow
		ag.now = func() time.Time { return clock }
		read := func(h time.Time) GoldRoll {
			return ag.goldRollOf(context.Background(), d1, goldRollHourAt(s, h), true)
		}
		first := read(goldWindowHour)
		if first.State != goldRollWindow || s.count("GCM26.CMX|1h") != 1 || s.count("GCM26.CMX|1d") != 1 {
			t.Fatalf("first bar: %+v, GCM26 1h %d 1d %d", first, s.count("GCM26.CMX|1h"), s.count("GCM26.CMX|1d"))
		}
		for i := 1; i < 24; i++ {
			clock = goldWindowNow.Add(time.Duration(i) * time.Hour)
			if r := read(goldWindowHour.Add(time.Duration(i) * time.Hour)); !reflect.DeepEqual(goldRollFields(&r), goldRollFields(&first)) {
				t.Fatalf("bar %d: %v, want %v", i, goldRollFields(&r), goldRollFields(&first))
			}
		}
		if n := s.count("GCM26.CMX|1h"); n != 1 {
			t.Errorf("GCM26 asked on %d hourly bars within a day, want 1", n)
		}
		if n := s.count("GCQ26.CMX|1h"); n != 24 {
			t.Errorf("GCQ26 asked on %d hourly bars, want 24", n)
		}
		clock = goldWindowNow.Add(goldRollGoneFor)
		read(goldWindowHour.Add(24 * time.Hour))
		if n := s.count("GCM26.CMX|1h"); n != 2 {
			t.Errorf("a day later GCM26 asked %d times in all, want 2", n)
		}
	})
	t.Run("near 404 is asked on the next bar", func(t *testing.T) {
		ag, s := goldWindowCase(t, nil)
		ag.goldRoll.logf = func(string, ...any) {}
		d1 := s.series["GC=F"].d1
		clock := goldWindowNow
		ag.now = func() time.Time { return clock }
		gc := s.series["GC=F"].h1
		delete(s.series, "GCQ26.CMX")
		if r := ag.goldRollOf(context.Background(), d1, gc[len(gc)-1], true); goldDeref(r.Reason) != goldRollNearMissing {
			t.Fatalf("%+v", r)
		}
		clock = clock.Add(time.Hour)
		h := fixedHourly(goldWindowHour.Add(time.Hour), 5160)
		ag.goldRollOf(context.Background(), d1, h[len(h)-1], true)
		if n := s.count("GCQ26.CMX|1h"); n != 2 {
			t.Errorf("near asked %d times over two bars, want 2", n)
		}
	})
	t.Run("previous request failed is asked on the next bar", func(t *testing.T) {
		ag, s := goldWindowCase(t, func(s *goldYahooStub) { s.series["GCM26.CMX"] = yahooSeries{status: http.StatusBadGateway} })
		ag.goldRoll.logf = func(string, ...any) {}
		d1 := s.series["GC=F"].d1
		clock := goldWindowNow
		ag.now = func() time.Time { return clock }
		goldRollReadWindow(t, ag, d1, goldRollHourAt(s, goldWindowHour))
		clock = clock.Add(time.Hour)
		goldRollReadWindow(t, ag, d1, goldRollHourAt(s, goldWindowHour.Add(time.Hour)))
		if n := s.count("GCM26.CMX|1h"); n != 2 {
			t.Errorf("previous asked %d times over two bars, want 2", n)
		}
	})
}

func goldRollReadWindow(t *testing.T, ag *Agents, d1 []types.OHLCVCandle, hour types.OHLCVCandle) {
	t.Helper()
	if r := ag.goldRollOf(context.Background(), d1, hour, true); r.State != goldRollWindow {
		t.Fatalf("%+v", r)
	}
}

// ── the log ──────────────────────────────────────────────────────────────────

// An unmatched bar while previous did not answer says so; one while all
// three answered does not. A remembered 404 reads the same as a fresh one.
func TestGoldRollLogNamesSilentCandidates(t *testing.T) {
	neither := func(s *goldYahooStub) {
		gc := s.series["GC=F"]
		s.series["GCQ26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, 100), h1: shiftBars(gc.h1, 100)}
		s.series["GCZ26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, 200), h1: shiftBars(gc.h1, 200)}
	}
	capture := func(ag *Agents) *[]string {
		var mu sync.Mutex
		lines := &[]string{}
		ag.goldRoll.logf = func(format string, args ...any) {
			mu.Lock()
			defer mu.Unlock()
			*lines = append(*lines, fmt.Sprintf(format, args...))
		}
		return lines
	}

	ag, s := goldWindowCase(t, neither)
	lines := capture(ag)
	clock := goldWindowNow
	ag.now = func() time.Time { return clock }
	d1 := s.series["GC=F"].d1
	h := s.series["GC=F"].h1
	ag.goldRollOf(context.Background(), d1, h[len(h)-1], true)
	// The next bar, GCM26's 404 now remembered.
	clock = clock.Add(time.Hour)
	h2 := fixedHourly(goldWindowHour.Add(time.Hour), 5160)
	s.set("GCQ26.CMX", yahooSeries{d1: s.series["GCQ26.CMX"].d1, h1: shiftBars(h2, 100)})
	s.set("GCZ26.CMX", yahooSeries{d1: s.series["GCZ26.CMX"].d1, h1: shiftBars(h2, 200)})
	ag.goldRollOf(context.Background(), d1, h2[len(h2)-1], true)
	want := []string{
		"[demobot] gold roll: unknown (matches_neither_contract: did not answer: yahoo chart GCM26.CMX: HTTP 404) (1d bar 2026-07-28 UTC, 1h bar 2026-07-29 09:00 UTC)",
		"[demobot] gold roll: unknown (matches_neither_contract: did not answer: yahoo chart GCM26.CMX: HTTP 404) (1d bar 2026-07-28 UTC, 1h bar 2026-07-29 10:00 UTC)",
	}
	if strings.Join(*lines, "\n") != strings.Join(want, "\n") {
		t.Errorf("log:\n%s\nwant:\n%s", strings.Join(*lines, "\n"), strings.Join(want, "\n"))
	}
	if n := s.count("GCM26.CMX|1h"); n != 1 {
		t.Errorf("GCM26 1h asked %d times, want 1", n)
	}

	ag, s = goldWindowCase(t, func(s *goldYahooStub) {
		neither(s)
		gc := s.series["GC=F"]
		s.series["GCM26.CMX"] = yahooSeries{d1: shiftBars(gc.d1, -100), h1: shiftBars(gc.h1, -100)}
	})
	lines = capture(ag)
	h = s.series["GC=F"].h1
	ag.goldRollOf(context.Background(), s.series["GC=F"].d1, h[len(h)-1], true)
	if want := "[demobot] gold roll: unknown (matches_neither_contract) (1d bar 2026-07-28 UTC, 1h bar 2026-07-29 09:00 UTC)"; len(*lines) != 1 || (*lines)[0] != want {
		t.Errorf("all three answered: log %q, want %q", *lines, want)
	}
}

// Previous is remembered as expired only once its delivery month is over.
func TestGoldPrevExpired(t *testing.T) {
	for day, want := range map[string]bool{
		"2026-07-28": true,  // GCM26, June
		"2026-09-01": true,  // GCQ26, August
		"2027-01-05": true,  // GCZ26, December 2026
		"2026-08-03": false, // GCQ26 in its own delivery month
		"2026-12-01": false, // GCZ26 likewise
		"2026-02-02": false, // GCG26 likewise
	} {
		d, err := time.Parse("2006-01-02", day)
		if err != nil {
			t.Fatal(err)
		}
		if got := goldPrevExpired(d); got != want {
			t.Errorf("%s (previous %s): expired %v, want %v", day, goldCandidates(d)[goldCandPrev], got, want)
		}
	}
}

// A remembered 404 is per contract AND interval, and runs out after a day.
func TestGoldRollGoneIsPerInterval(t *testing.T) {
	m := newGoldRollMemo()
	at := goldWindowNow
	m.markGone("GCM26", "1h", &yahooHTTPError{Symbol: "GCM26.CMX", Status: http.StatusNotFound}, at)
	if m.goneErr("GCM26", "1d", at) != nil {
		t.Error("an hourly 404 silenced the daily series")
	}
	if err := m.goneErr("GCM26", "1h", at.Add(goldRollGoneFor-time.Minute)); !goldNotServed(err) {
		t.Errorf("within the day: %v", err)
	}
	if err := m.goneErr("GCM26", "1h", at.Add(goldRollGoneFor)); err != nil {
		t.Errorf("after the day: %v", err)
	}
}

// 2026-08-03: the candidates are GCQ26 (previous), GCZ26 and GCG27, and
// GCQ26 still trades. Had Yahoo switched the daily series a day late, the
// daily bar is GCQ26's. A momentary 404 for it must not be remembered: the
// card is unknown until the retry 15 minutes later, then the window.
func TestGoldRollPreviousInItsDeliveryMonthNotRemembered(t *testing.T) {
	day := time.Date(2026, 8, 3, 4, 0, 0, 0, time.UTC)
	hourAt := time.Date(2026, 8, 4, 9, 0, 0, 0, time.UTC)
	d1 := dailyEnding(day, 30, risingDay)
	h1 := fixedHourly(hourAt, 5155)
	if c := goldCandidates(day); c != [3]string{"GCQ26", "GCZ26", "GCG27"} {
		t.Fatalf("candidates %v", c)
	}
	s := newGoldYahooStub(t)
	gcq := yahooSeries{d1: d1, h1: shiftBars(h1, -goldWindowSpread)}
	s.set("GCZ26.CMX", yahooSeries{d1: shiftBars(d1, goldWindowSpread), h1: h1})
	s.set("GCG27.CMX", yahooSeries{d1: shiftBars(d1, 2*goldWindowSpread), h1: shiftBars(h1, 2*goldWindowSpread)})
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	ag.goldRoll.logf = func(string, ...any) {}
	clock := time.Date(2026, 8, 4, 10, 30, 0, 0, time.UTC)
	ag.now = func() time.Time { return clock }
	read := func() GoldRoll { return ag.goldRollOf(context.Background(), d1, h1[len(h1)-1], true) }

	if r := read(); r.State != goldRollUnknown || goldDeref(r.Reason) != goldRollMatchesNone {
		t.Fatalf("GCQ26 404: %+v %s", r, goldDeref(r.Reason))
	}
	s.set("GCQ26.CMX", gcq) // it answers again
	clock = clock.Add(goldRollRetry - time.Minute)
	if r := read(); r.State != goldRollUnknown {
		t.Fatalf("before the retry: %+v", r)
	}
	clock = clock.Add(time.Minute)
	r := read()
	if r.State != goldRollWindow || goldDeref(r.DailyContract) != "GCQ26" || goldDeref(r.HourlyContract) != "GCZ26" {
		t.Fatalf("after the retry: %+v %v, want window GCQ26 → GCZ26", r, goldRollFields(&r))
	}
	if n := s.count("GCQ26.CMX|1d"); n != 2 {
		t.Errorf("GCQ26 1d asked %d times, want 2", n)
	}
}
