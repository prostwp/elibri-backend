package demobot

// gold_roll_real_test.go — the 2026-07-31 roll replayed on Yahoo's own
// answers, saved for the roll research on 2026-09-23 (gold_roll/data, report
// gold_roll/out/отчёт.md). testdata/gold_roll_2026-07 holds slices of them:
//
//	GC=F_1d.json       GC=F 2y daily, 2025-07-28 … 2026-08-03
//	GC=F_1h.json       GC=F 730d hourly, 2026-06-26 … 2026-08-03 11:00
//	GCZ26.CMX_1d.json  GCZ26 2y daily, 2026-06-26 … 2026-08-03
//	GCZ26.CMX_1h.json  GCZ26 730d hourly, same span as GC=F_1h
//
// GCQ26 — the contract GC=F left on 2026-07-31 — expired in August and Yahoo
// answers 404 for it now (it did in the saved data). Hypothesis: during the
// window, while it was still the front contract, Yahoo served it; nothing
// saved shows that. It is RECONSTRUCTED here from GC=F's own bars: GC=F's
// daily bars up to 2026-07-30 (they match no served contract and the next
// bar, 07-31, is GCZ26's) and GC=F's hourly bars before the hourly switch at
// 2026-07-29 07:00. After the switch the reconstruction serves no GCQ26 hourly
// bar, which the check reads exactly like a GCQ26 bar that differs — "not
// this contract". Every GC=F and GCZ26 number is Yahoo's own.
//
// Because GCQ26 here is GC=F under another name, its daily side matches by
// construction: this replay checks the GCZ26 side and the card's wording, not
// that a real GCQ26 answer would match. The independent check is the
// 2025-11-28 roll against Yahoo's own GCZ25 answers, with nothing
// reconstructed (TestGoldRollDataRealZ25 in gold_roll_data_test.go, run on
// the research data): the card says "none" on exactly the hours whose 1h
// close is GCZ25's and never names a wrong contract; GCG26 is not served, so
// that roll's window hours read unknown there.

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

func loadRollFixture(t *testing.T, name string) []types.OHLCVCandle {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "gold_roll_2026-07", name))
	if err != nil {
		t.Fatal(err)
	}
	bars, err := parseYahooChart(b)
	if err != nil {
		t.Fatal(err)
	}
	return bars
}

// barsBetween keeps bars opening at or after from and CLOSED by at.
func barsBetween(bars []types.OHLCVCandle, from, at time.Time, sec int64) []types.OHLCVCandle {
	var out []types.OHLCVCandle
	for _, b := range bars {
		if b.Time >= from.Unix() && b.Time+sec <= at.Unix() {
			out = append(out, b)
		}
	}
	return out
}

func barsBefore(bars []types.OHLCVCandle, t time.Time) []types.OHLCVCandle {
	var out []types.OHLCVCandle
	for _, b := range bars {
		if b.Time < t.Unix() {
			out = append(out, b)
		}
	}
	return out
}

var (
	rollDailySwitch  = time.Date(2026, 7, 31, 4, 0, 0, 0, time.UTC) // first GCZ26 daily bar of GC=F
	rollHourlySwitch = time.Date(2026, 7, 29, 7, 0, 0, 0, time.UTC) // first GCZ26 hourly bar of GC=F
)

type rollReplay struct {
	gcD, gcH, zD, zH []types.OHLCVCandle
}

func loadRollReplay(t *testing.T) rollReplay {
	return rollReplay{
		gcD: loadRollFixture(t, "GC=F_1d.json"), gcH: loadRollFixture(t, "GC=F_1h.json"),
		zD: loadRollFixture(t, "GCZ26.CMX_1d.json"), zH: loadRollFixture(t, "GCZ26.CMX_1h.json"),
	}
}

// cardAt builds the gold card as it read at `at`: every series cut to what was
// closed then (GC=F daily to Yahoo's 1y window, hourly to its 1mo window).
// withQ26=false is Yahoo today: GCQ26 answers 404.
func (r rollReplay) cardAt(t *testing.T, at time.Time, withQ26 bool) Card {
	t.Helper()
	s := newGoldYahooStub(t)
	gcD := barsBetween(r.gcD, at.AddDate(-1, 0, 0), at, 86400)
	gcH := barsBetween(r.gcH, at.AddDate(0, -1, 0), at, 3600)
	s.set("GC=F", yahooSeries{d1: gcD, h1: gcH})
	s.set("GCZ26.CMX", yahooSeries{d1: barsBetween(r.zD, at.AddDate(0, -1, 0), at, 86400), h1: barsBetween(r.zH, at.AddDate(0, -1, 0), at, 3600)})
	if withQ26 {
		s.set("GCQ26.CMX", yahooSeries{d1: barsBefore(gcD, rollDailySwitch), h1: barsBefore(gcH, rollHourlySwitch)})
	}
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	ag.now = func() time.Time { return at }
	return ag.GoldCard(context.Background())
}

// rawPosition is what the card placed before the check: the last closed 1h
// close against the day range of the last closed daily bars, no contract
// asked.
func (r rollReplay) rawPosition(at time.Time) (float64, string) {
	d := barsBetween(r.gcD, at.AddDate(-1, 0, 0), at, 86400)
	h := barsBetween(r.gcH, at.AddDate(0, -1, 0), at, 3600)
	px := h[len(h)-1].Close
	return px, goldDayLevelsOf(d).positionOf(px)
}

// The research's daily reading moment for the 1d candle of 2026-07-30 (read
// at 08:25:53 UTC the next morning, 2026-07-31 — the table row "2026-07-30" of
// отчёт.md, "В ежедневный момент оценки"). The raw card said the price 4125.10
// was above the day range 4028.50 – 4118.50; in GCZ26's own prices that range
// is 4089.00 – 4179.00 and the price is inside it.
func TestGoldRollReplayFalseAboveIsGone(t *testing.T) {
	r := loadRollReplay(t)
	at := time.Date(2026, 7, 31, 8, 25, 53, 0, time.UTC)

	px, raw := r.rawPosition(at)
	if goldPx(px) != "4125.10" || raw != dayAbove {
		t.Fatalf("fixture: raw %s %s, want 4125.10 above", goldPx(px), raw)
	}

	// Yahoo today (GCQ26 404): unknown — the card reads as before, and the
	// false "above" is still there. This is the documented cost of "unknown
	// reads as before"; live, the expiring contract is served.
	before := r.cardAt(t, at, false)
	if st := before.Gold.Roll; st.State != goldRollUnknown || *st.Reason != goldRollNearMissing {
		t.Fatalf("GCQ26 404: roll %+v", st)
	}
	if !strings.Contains(before.Facts[0], "4125.10 at 2026-07-31 08:00 UTC — above the day range") {
		t.Fatalf("GCQ26 404 must read as before: %q", before.Facts[0])
	}

	c := r.cardAt(t, at, true)
	roll := c.Gold.Roll
	if roll.State != goldRollWindow || *roll.DailyContract != "GCQ26" || *roll.HourlyContract != "GCZ26" {
		t.Fatalf("roll %+v", roll)
	}
	text := strings.Join(goldLines(c), "\n")
	for _, bad := range []string{"above the day range", "below the day range", "inside the day range", "already above", "already below"} {
		if strings.Contains(text, bad) {
			t.Errorf("window card still places the price: %q\n%s", bad, text)
		}
	}
	if c.Gold.PricePosition != nil || c.Gold.Idea != nil {
		t.Errorf("price_position %v idea %+v", c.Gold.PricePosition, c.Gold.Idea)
	}
	for _, want := range []string{
		"Last closed 1h price 4125.10 at 2026-07-31 08:00 UTC — on contract GCZ26",
		"Contract roll GCQ26 → GCZ26: 1h price on GCZ26, day range on GCQ26; price not placed against it",
		"Day range 4028.50 – 4118.50: high/low of the closed 1d candle of 2026-07-30",
	} {
		if !contains(c.Facts, want) {
			t.Errorf("missing %q in\n%s", want, strings.Join(c.Facts, "\n"))
		}
	}
}

// Every closed hourly bar from the hourly switch (2026-07-29 07:00) to the
// last one before the daily bar of 07-31 closes (07-31 20:00, a Friday) is a
// window hour: 60 hours, as in the research table. Each is read one minute
// after it closes. In every one the card places nothing; the raw card placed
// the price beyond the range in some of them.
func TestGoldRollReplayWholeWindow(t *testing.T) {
	r := loadRollReplay(t)
	var hours, window, rawBeyond int
	for _, b := range r.gcH {
		open := time.Unix(b.Time, 0).UTC()
		if open.Before(rollHourlySwitch) || open.After(time.Date(2026, 7, 31, 20, 0, 0, 0, time.UTC)) {
			continue
		}
		hours++
		at := open.Add(time.Hour + time.Minute)
		if _, raw := r.rawPosition(at); raw != dayInside {
			rawBeyond++
		}
		c := r.cardAt(t, at, true)
		if c.Gold.Roll.State == goldRollWindow {
			window++
		} else {
			t.Errorf("%s: roll %s %v", open.Format("2006-01-02 15:04"), c.Gold.Roll.State, c.Gold.Roll.Reason)
		}
		if c.Gold.PricePosition != nil {
			t.Errorf("%s: price placed %s", open.Format("2006-01-02 15:04"), *c.Gold.PricePosition)
		}
	}
	t.Logf("window hours %d, established as window %d, raw card beyond the range in %d", hours, window, rawBeyond)
	if hours != 60 || window != 60 {
		t.Errorf("hours %d window %d, want 60/60", hours, window)
	}
	// отчёт.md: 38 of the 60 hours had the raw price beyond the range.
	if rawBeyond != 38 {
		t.Errorf("raw beyond-range hours %d, the research counted 38", rawBeyond)
	}
}

// Around the window the check says none: before the hourly switch both bars
// are GCQ26's, after the daily roll both are GCZ26's — and the card places the
// price exactly as it always did.
func TestGoldRollReplayOutsideWindow(t *testing.T) {
	r := loadRollReplay(t)
	cases := []struct {
		at       time.Time
		contract string
	}{
		{time.Date(2026, 7, 29, 7, 1, 0, 0, time.UTC), "GCQ26"},    // last closed hour 06:00, before the switch
		{time.Date(2026, 8, 3, 8, 25, 53, 0, time.UTC), "GCZ26"},   // Monday after the daily roll
		{time.Date(2026, 7, 28, 14, 25, 53, 0, time.UTC), "GCQ26"}, // a day earlier
	}
	for _, tc := range cases {
		c := r.cardAt(t, tc.at, true)
		roll := c.Gold.Roll
		if roll.State != goldRollNone || *roll.DailyContract != tc.contract || *roll.HourlyContract != tc.contract {
			t.Errorf("%s: roll %+v", tc.at, roll)
			continue
		}
		_, raw := r.rawPosition(tc.at)
		if c.Gold.PricePosition == nil || *c.Gold.PricePosition != raw {
			t.Errorf("%s: position %v, raw %s", tc.at, c.Gold.PricePosition, raw)
		}
	}
}
