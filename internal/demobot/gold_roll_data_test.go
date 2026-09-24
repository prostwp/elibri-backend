package demobot

// gold_roll_data_test.go — the numbers gold_roll.go and docs/demobot-http.md
// quote about the roll, recomputed from the saved Yahoo answers of the roll
// research (gold_roll/data, 2026-09-23). Those files are not in the
// repository; the test runs only when GOLD_ROLL_DATA points at them and is
// skipped otherwise:
//
//	GOLD_ROLL_DATA=/path/to/gold_roll/data go test -run TestGoldRollData -v ./internal/demobot/
//
// It prints the numbers and pins them, so a changed figure fails here before
// it reaches a comment or the docs.

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

func goldRollDataDir(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("GOLD_ROLL_DATA")
	if dir == "" {
		t.Skip("GOLD_ROLL_DATA not set: the research data is not in the repository")
	}
	return dir
}

func goldRollDataBars(t *testing.T, dir, name string) []types.OHLCVCandle {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		t.Fatal(err)
	}
	bars, err := parseYahooChart(b)
	if err != nil {
		t.Fatal(err)
	}
	return bars
}

// The hours after GC=F's hourly series entered GCZ25 and before its daily
// series did: bars opening 2025-07-30 07:00 … 2025-08-01 04:00 UTC, both ends
// included. How many carry GCZ25's close, and how many all four of its prices
// — the reason gold_roll.go matches hourly bars on the close.
func TestGoldRollDataHourlyCloseOnly(t *testing.T) {
	dir := goldRollDataDir(t)
	gc := goldRollDataBars(t, dir, "GC=F_1h_730d.json")
	z := goldRollDataBars(t, dir, "GCZ25.CMX_1h_730d.json")
	from := time.Date(2025, 7, 30, 7, 0, 0, 0, time.UTC).Unix()
	to := time.Date(2025, 8, 1, 4, 0, 0, 0, time.UTC).Unix()
	byTime := map[int64]types.OHLCVCandle{}
	for _, b := range z {
		byTime[b.Time] = b
	}
	var hours, closeEq, allEq, missing int
	for _, b := range gc {
		if b.Time < from || b.Time > to {
			continue
		}
		hours++
		zb, ok := byTime[b.Time]
		if !ok {
			missing++
			continue
		}
		if goldSamePrint(b.Close, zb.Close) {
			closeEq++
			if goldSamePrint(b.Open, zb.Open) && goldSamePrint(b.High, zb.High) && goldSamePrint(b.Low, zb.Low) {
				allEq++
			}
		}
	}
	t.Logf("GC=F 1h bars opening 2025-07-30 07:00 … 2025-08-01 04:00 UTC (inclusive): %d; GCZ25 close equal %d, all four equal %d, no GCZ25 bar %d",
		hours, closeEq, allEq, missing)
	if hours != 44 || closeEq != 44 || allEq != 2 || missing != 0 {
		t.Errorf("hours %d close %d all four %d missing %d — update gold_roll.go and the docs", hours, closeEq, allEq, missing)
	}
}

// goldRollDataRolls are GC=F's daily roll dates (first daily bar of the new
// contract): established by price matching, and the research's candidates
// (Hypothesis there: the contracts are not served, the dates come from a
// jump in GC=F / GCZ26).
var (
	goldRollDataEstablished = []string{"2025-08-01", "2025-11-28", "2026-07-31"}
	goldRollDataCandidates  = []string{"2024-12-02", "2025-02-03", "2025-04-01", "2025-06-02", "2026-01-30", "2026-04-01", "2026-05-29"}
)

// The day range can come from an earlier bar than the last closed one
// (goldDayLevelsOf unwinds up to three inside days). If a daily roll fell
// between that bar and the last closed one, the range would be a range of the
// old contract while the last daily bar — and, with roll state none, the 1h
// price — are on the new one. Counted on every daily point of the saved
// 2-year series.
func TestGoldRollDataRangeAcrossRoll(t *testing.T) {
	dir := goldRollDataDir(t)
	daily := goldRollDataBars(t, dir, "GC=F_2y.json")
	day := func(ts int64) string { return time.Unix(ts, 0).UTC().Format("2006-01-02") }
	isRoll := func(list []string) map[string]bool {
		m := map[string]bool{}
		for _, d := range list {
			m[d] = true
		}
		return m
	}
	est, cand := isRoll(goldRollDataEstablished), isRoll(goldRollDataCandidates)
	// The saved series ends with the forming bar of 2026-09-23: drop it.
	closed := daily[:len(daily)-1]
	var points, unwound, acrossEst, acrossCand int
	for i := 1; i < len(closed); i++ {
		lv := goldDayLevelsOf(closed[:i+1])
		points++
		if !lv.Defined || lv.Unwound == 0 {
			continue
		}
		unwound++
		// A roll between the range bar and the last bar: some bar after the
		// range bar, up to and including the last, is a first new-contract bar.
		for j := i - lv.Unwound + 1; j <= i; j++ {
			d := day(closed[j].Time)
			if est[d] {
				acrossEst++
				t.Logf("established roll %s between range bar %s and last bar %s", d, day(lv.BarTime), day(closed[i].Time))
			}
			if cand[d] {
				acrossCand++
				t.Logf("candidate roll %s between range bar %s and last bar %s", d, day(lv.BarTime), day(closed[i].Time))
			}
		}
	}
	t.Logf("daily points %s … %s: %d; range from an earlier bar: %d; a roll between range bar and last bar: established %d, candidates %d",
		day(closed[1].Time), day(closed[len(closed)-1].Time), points, unwound, acrossEst, acrossCand)
	if points != 502 || unwound != 62 || acrossEst != 0 || acrossCand != 0 {
		t.Errorf("points %d unwound %d across established %d candidates %d — update gold.go / gold_text.go comments and the docs",
			points, unwound, acrossEst, acrossCand)
	}
}

// The 2025-11-28 roll on Yahoo's own GCZ25 answers — the contract GC=F left,
// which Yahoo still serves — with no reconstructed series: the independent
// check beside gold_roll_real_test.go, whose GCQ26 is built from GC=F's own
// bars. GCG26, the contract entered, is not served (404), so a window cannot
// be established here; what can be checked is that the card says "none"
// exactly on the hours whose 1h close is GCZ25's — before the hourly switch
// at 2025-11-25 09:00 and in the hybrid hours after it — and unknown on every
// other hour, never a wrong contract. Every closed hourly bar opening
// 2025-11-24 00:00 … 2025-11-28 18:00 UTC, read one minute after it closes.
func TestGoldRollDataRealZ25(t *testing.T) {
	dir := goldRollDataDir(t)
	gcD := goldRollDataBars(t, dir, "GC=F_2y.json")
	gcH := goldRollDataBars(t, dir, "GC=F_1h_730d.json")
	zD := goldRollDataBars(t, dir, "GCZ25.CMX.json")
	zH := goldRollDataBars(t, dir, "GCZ25.CMX_1h_730d.json")
	zClose := map[int64]float64{}
	for _, b := range zH {
		zClose[b.Time] = b.Close
	}
	from := time.Date(2025, 11, 24, 0, 0, 0, 0, time.UTC)
	to := time.Date(2025, 11, 28, 18, 0, 0, 0, time.UTC)
	switchAt := time.Date(2025, 11, 25, 9, 0, 0, 0, time.UTC)
	var hours, none, unknown, wrong, noneAfter, unknownBefore int
	reasons := map[string]int{}
	for _, b := range gcH {
		open := time.Unix(b.Time, 0).UTC()
		if open.Before(from) || open.After(to) {
			continue
		}
		hours++
		at := open.Add(time.Hour + time.Minute)
		s := newGoldYahooStub(t)
		s.set("GC=F", yahooSeries{d1: barsBetween(gcD, at.AddDate(-1, 0, 0), at, 86400), h1: barsBetween(gcH, at.AddDate(0, -1, 0), at, 3600)})
		s.set("GCZ25.CMX", yahooSeries{d1: barsBetween(zD, at.AddDate(0, -1, 0), at, 86400), h1: barsBetween(zH, at.AddDate(0, -1, 0), at, 3600)})
		ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
		ag.goldRoll.logf = func(string, ...any) {}
		ag.now = func() time.Time { return at }
		r := ag.GoldCard(context.Background()).Gold.Roll
		onZ := false
		if zc, ok := zClose[b.Time]; ok && goldSamePrint(zc, b.Close) {
			onZ = true
		}
		switch {
		case r.State == goldRollNone && onZ && *r.HourlyContract == "GCZ25":
			none++
			if !open.Before(switchAt) {
				noneAfter++
				t.Logf("%s: 1h close is GCZ25's after the switch — none", open.Format("2006-01-02 15:04"))
			}
		case r.State == goldRollUnknown && !onZ:
			unknown++
			reasons[*r.Reason]++
			if open.Before(switchAt) {
				unknownBefore++
			}
		default:
			wrong++
			t.Errorf("%s: roll %s %v, 1h close on GCZ25 %v", open.Format("2006-01-02 15:04"), r.State, goldRollFields(r), onZ)
		}
	}
	t.Logf("GC=F 1h bars opening 2025-11-24 00:00 … 2025-11-28 18:00 UTC: %d; none on GCZ25 %d (of them from 2025-11-25 09:00: %d), unknown %d (of them before it: %d) %v, other %d",
		hours, none, noneAfter, unknown, unknownBefore, reasons, wrong)
	// The 43 unknown hours are the research's 43 window hours of this roll
	// (отчёт.md, 2025-11-25 09:00 … 2025-11-28 18:00 without the four hybrid
	// hours, whose close is still GCZ25's and which read "none" here).
	if hours != 79 || none != 36 || noneAfter != 4 || unknown != 43 || unknownBefore != 0 || wrong != 0 {
		t.Errorf("counts moved — update gold_roll_real_test.go and the docs")
	}
}
