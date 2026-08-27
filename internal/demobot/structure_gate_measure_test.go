package demobot

// structure_gate_measure_test.go — a MEASUREMENT harness, not an assertion
// suite. It prints how the B3 swing-structure gate behaves on real series so
// every structure-gate number that leaves this repo can be reproduced with
// one command instead of being quoted from memory:
//
//	MEASURE_STRUCTURE_GATE=1 go test ./internal/demobot/ \
//	    -run TestMeasureStructureGate -v
//
// It is env-guarded because it hits Yahoo and Binance — the normal suite must
// stay offline and deterministic.
//
// Method (identical in every measurement below):
//   - walk the series bar by bar; the decision at bar i uses ONLY bars[0..i],
//     so nothing here can see the future;
//   - warm up 220 bars, the floor for EMA200 + ADX(14);
//   - classify with the PRODUCTION functions (classifyTrend, swingPointsIdx,
//     hhhlStructure). Only the four-case demotion switch is restated here,
//     kept character-for-character in step with TrendCard.

import (
	"context"
	"fmt"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

const measureWarmup = 220

// applyStructureGate mirrors the demotion switch in TrendCard verbatim. If
// that switch ever changes, this must change with it.
func applyStructureGate(state, structure string) string {
	switch {
	case state == trendUp && (structure == "lh_ll" || structure == "mixed"):
		return trendGrey
	case state == trendDown && (structure == "hh_hl" || structure == "mixed"):
		return trendGrey
	}
	return state
}

// hhhlStructurePreFix reproduces the SHIPPED-BEFORE behaviour of
// hhhlStructure: a non-alternating tail returned "mixed" (a structural claim
// that demotes) instead of "" (no claim). Kept here, and only here, so the
// cost of that defect stays reproducible after the production function was
// fixed — without it, column A below silently measures the fix and reports
// the defect as free.
//
// It must stay a byte-for-byte copy of hhhlStructure except for the one
// marked branch.
func hhhlStructurePreFix(swingHighs, swingLows []swingPoint) string {
	type pivot struct {
		idx    int
		price  float64
		isHigh bool
	}
	pivots := make([]pivot, 0, len(swingHighs)+len(swingLows))
	for _, p := range swingHighs {
		pivots = append(pivots, pivot{p.idx, p.price, true})
	}
	for _, p := range swingLows {
		pivots = append(pivots, pivot{p.idx, p.price, false})
	}
	const need = 6
	if len(pivots) < need {
		return ""
	}
	sort.Slice(pivots, func(i, j int) bool {
		if pivots[i].idx != pivots[j].idx {
			return pivots[i].idx < pivots[j].idx
		}
		return !pivots[i].isHigh
	})
	tail := pivots[len(pivots)-need:]
	for k := 1; k < len(tail); k++ {
		if tail[k].isHigh == tail[k-1].isHigh || tail[k].idx == tail[k-1].idx {
			return "mixed" // ← THE DEFECT: unreadable window claimed as disagreement
		}
	}
	var highs, lows []float64
	for _, p := range tail {
		if p.isHigh {
			highs = append(highs, p.price)
		} else {
			lows = append(lows, p.price)
		}
	}
	dir := func(vals []float64) string {
		rising, falling := true, true
		for i := 1; i < len(vals); i++ {
			if vals[i] <= vals[i-1] {
				rising = false
			}
			if vals[i] >= vals[i-1] {
				falling = false
			}
		}
		switch {
		case rising:
			return "up"
		case falling:
			return "down"
		default:
			return "mixed"
		}
	}
	h, l := dir(highs), dir(lows)
	switch {
	case h == "up" && l == "up":
		return "hh_hl"
	case h == "down" && l == "down":
		return "lh_ll"
	default:
		return "mixed"
	}
}

// gateReadingAt is one decision-time reading over bars[0..i].
type gateReadingAt struct {
	ok        bool
	raw       string // classifyTrend before the gate
	gated     string // after the CURRENT (fixed) gate
	gatedPre  string // after the SHIPPED-BEFORE gate — the defect's cost
	structure string // hhhlStructure output ("" → "(unread)")
	cause     string // why the structure was not classified
}

func readGateAt(bars []types.OHLCVCandle, i int) gateReadingAt {
	w := bars[:i+1]
	closes := closesOf(w)
	highs, lows := highsLowsOf(w)
	ema50, ok50 := emaLast(closes, 50)
	ema200, ok200 := emaLast(closes, 200)
	adx, okADX := adxWilder(highs, lows, closes, 14)
	if !ok50 || !ok200 || !okADX {
		return gateReadingAt{}
	}
	raw := classifyTrend(adx, ema50, ema200, closes[len(closes)-1])
	sh, sl := swingPointsIdx(highs, lows, 3)
	structure := hhhlStructure(sh, sl)
	cause := structureCause(sh, sl)
	pre := hhhlStructurePreFix(sh, sl)
	if structure == "" {
		structure = "(unread)"
	}
	return gateReadingAt{
		ok:        true,
		raw:       raw,
		gated:     applyStructureGate(raw, structure),
		gatedPre:  applyStructureGate(raw, pre),
		structure: structure,
		cause:     cause,
	}
}

// structureCause names WHY hhhlStructure could not read an alternating
// sequence. It re-walks the same pivot ordering hhhlStructure uses.
func structureCause(swingHighs, swingLows []swingPoint) string {
	type pivot struct {
		idx    int
		isHigh bool
	}
	pivots := make([]pivot, 0, len(swingHighs)+len(swingLows))
	for _, p := range swingHighs {
		pivots = append(pivots, pivot{p.idx, true})
	}
	for _, p := range swingLows {
		pivots = append(pivots, pivot{p.idx, false})
	}
	const need = 6
	if len(pivots) < need {
		return "too-few-pivots"
	}
	sort.Slice(pivots, func(i, j int) bool {
		if pivots[i].idx != pivots[j].idx {
			return pivots[i].idx < pivots[j].idx
		}
		return !pivots[i].isHigh
	})
	tail := pivots[len(pivots)-need:]
	sameIdx, sameType := 0, 0
	for k := 1; k < len(tail); k++ {
		switch {
		case tail[k].idx == tail[k-1].idx:
			sameIdx++
		case tail[k].isHigh == tail[k-1].isHigh:
			sameType++
		}
	}
	switch {
	case sameIdx > 0 && sameType > 0:
		return "both"
	case sameIdx > 0:
		return "same-bar-pivots"
	case sameType > 0:
		return "consecutive-same-type"
	}
	return "alternating-ok"
}

// pivotTail renders the last six pivots as a readable sequence.
func pivotTail(swingHighs, swingLows []swingPoint) string {
	type pivot struct {
		idx    int
		isHigh bool
	}
	pivots := []pivot{}
	for _, p := range swingHighs {
		pivots = append(pivots, pivot{p.idx, true})
	}
	for _, p := range swingLows {
		pivots = append(pivots, pivot{p.idx, false})
	}
	if len(pivots) < 6 {
		return "(fewer than six pivots)"
	}
	sort.Slice(pivots, func(i, j int) bool {
		if pivots[i].idx != pivots[j].idx {
			return pivots[i].idx < pivots[j].idx
		}
		return !pivots[i].isHigh
	})
	s := ""
	for _, p := range pivots[len(pivots)-6:] {
		mark := "L"
		if p.isHigh {
			mark = "H"
		}
		s += fmt.Sprintf("%s@%d ", mark, p.idx)
	}
	return s
}

type gateSummary struct {
	label           string
	n               int
	confirmedBefore float64
	confirmedPreFix float64
	confirmedAfter  float64
	mixedShare      float64
	causes          map[string]int
	rawStates       map[string]int
	preStates       map[string]int
	gatedStates     map[string]int
}

func measureSeries(label string, bars []types.OHLCVCandle) gateSummary {
	s := gateSummary{
		label:       label,
		causes:      map[string]int{},
		rawStates:   map[string]int{},
		preStates:   map[string]int{},
		gatedStates: map[string]int{},
	}
	mixed := 0
	for i := measureWarmup; i < len(bars); i++ {
		r := readGateAt(bars, i)
		if !r.ok {
			continue
		}
		s.n++
		s.rawStates[r.raw]++
		s.preStates[r.gatedPre]++
		s.gatedStates[r.gated]++
		s.causes[r.cause]++
		if r.structure == "mixed" {
			mixed++
		}
	}
	if s.n == 0 {
		return s
	}
	pct := func(v int) float64 { return 100 * float64(v) / float64(s.n) }
	s.confirmedBefore = pct(s.rawStates[trendUp] + s.rawStates[trendDown])
	s.confirmedPreFix = pct(s.preStates[trendUp] + s.preStates[trendDown])
	s.confirmedAfter = pct(s.gatedStates[trendUp] + s.gatedStates[trendDown])
	s.mixedShare = pct(mixed)
	return s
}

func TestMeasureStructureGate(t *testing.T) {
	if os.Getenv("MEASURE_STRUCTURE_GATE") == "" {
		t.Skip("set MEASURE_STRUCTURE_GATE=1 (hits Yahoo + Binance)")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()

	type series struct {
		label string
		bars  []types.OHLCVCandle
	}
	var all []series

	kc := newKlineCache()
	for _, tc := range []struct{ sym, iv, label string }{
		{"BTCUSDT", "4h", "BTC 4h  (Trend Agent default)"},
		{"ETHUSDT", "4h", "ETH 4h"},
		{"BTCUSDT", "1d", "BTC 1d"},
	} {
		b, err := kc.fetch(ctx, tc.sym, tc.iv, 1000)
		if err != nil {
			t.Logf("%s: fetch failed: %v", tc.label, err)
			continue
		}
		all = append(all, series{tc.label, dropUnclosedBars(b, tc.iv, time.Now())})
		time.Sleep(400 * time.Millisecond)
	}
	for _, tc := range []struct{ iv, rng, label string }{
		{"1h", "1mo", "GOLD 1h (shipped config)"},
		{"1d", "5y", "GOLD 1d (gold-agent candidate)"},
	} {
		b, err := fetchYahooChart(ctx, "GC=F", tc.iv, tc.rng)
		if err != nil {
			t.Logf("%s: fetch failed: %v", tc.label, err)
			continue
		}
		all = append(all, series{tc.label, dropUnclosedBars(b, tc.iv, time.Now())})
		time.Sleep(1200 * time.Millisecond)
	}
	if len(all) == 0 {
		t.Skip("no series could be fetched")
	}

	fmt.Printf("\nmeasured %s UTC · warmup %d bars · decision at bar i uses bars[0..i]\n",
		time.Now().UTC().Format("2006-01-02 15:04"), measureWarmup)

	fmt.Println("\n=== 1. what the structure gate costs ===")
	fmt.Println("  raw      = classifyTrend, no gate")
	fmt.Println("  SHIPPED  = the gate as it shipped (non-alternating window → \"mixed\" → demote)")
	fmt.Println("  FIXED    = the gate after the fix (non-alternating window → no claim)")
	fmt.Printf("%-32s %6s %8s %9s %8s %9s\n", "series", "n", "raw", "SHIPPED", "FIXED", "mixed")
	sums := make([]gateSummary, 0, len(all))
	for _, s := range all {
		g := measureSeries(s.label, s.bars)
		sums = append(sums, g)
		if g.n == 0 {
			fmt.Printf("%-32s   (no usable window)\n", g.label)
			continue
		}
		fmt.Printf("%-32s %6d %7.1f%% %8.1f%% %7.1f%% %8.1f%%\n",
			g.label, g.n, g.confirmedBefore, g.confirmedPreFix, g.confirmedAfter, g.mixedShare)
	}

	fmt.Println("\n=== 2. why the structure could not be read ===")
	fmt.Printf("%-32s %14s %10s %14s %10s\n",
		"series", "alternating", "same-bar", "same-type", "too-few")
	for _, g := range sums {
		if g.n == 0 {
			continue
		}
		p := func(k string) float64 { return 100 * float64(g.causes[k]) / float64(g.n) }
		fmt.Printf("%-32s %13.1f%% %9.1f%% %13.1f%% %9.1f%%\n",
			g.label, p("alternating-ok"), p("same-bar-pivots"),
			p("consecutive-same-type")+p("both"), p("too-few-pivots"))
	}

	fmt.Println("\n=== 3. state distribution, before → after the gate ===")
	for _, g := range sums {
		if g.n == 0 {
			continue
		}
		fmt.Printf("\n  %s (n=%d)\n", g.label, g.n)
		fmt.Printf("    %-9s %8s   %8s   %8s\n", "", "raw", "SHIPPED", "FIXED")
		for _, st := range []string{trendUp, trendDown, trendGrey, trendFlat, trendConflict} {
			fmt.Printf("    %-9s %7.1f%%   %7.1f%%   %7.1f%%\n", st,
				100*float64(g.rawStates[st])/float64(g.n),
				100*float64(g.preStates[st])/float64(g.n),
				100*float64(g.gatedStates[st])/float64(g.n))
		}
	}

	fmt.Println("\n=== 3b. candidate fixes, confirmation rate after the gate ===")
	fmt.Println("  A = as shipped: a non-alternating window reads \"mixed\" and demotes")
	fmt.Println("  B = split: a non-alternating window reads as UNREADABLE (no demotion);")
	fmt.Println("      a genuine mixed (alternating pivots, prices not aligned) still demotes")
	fmt.Println("  C = minimal: only an explicit opposite structure demotes")
	fmt.Printf("\n%-32s %8s %8s %8s %8s\n", "series", "before", "A", "B", "C")
	for _, s := range all {
		a, b, c, before, n := 0, 0, 0, 0, 0
		for i := measureWarmup; i < len(s.bars); i++ {
			w := s.bars[:i+1]
			closes := closesOf(w)
			highs, lows := highsLowsOf(w)
			ema50, ok50 := emaLast(closes, 50)
			ema200, ok200 := emaLast(closes, 200)
			adx, okADX := adxWilder(highs, lows, closes, 14)
			if !ok50 || !ok200 || !okADX {
				continue
			}
			n++
			raw := classifyTrend(adx, ema50, ema200, closes[len(closes)-1])
			if raw != trendUp && raw != trendDown {
				continue
			}
			before++
			sh, sl := swingPointsIdx(highs, lows, 3)
			st := hhhlStructure(sh, sl)
			cause := structureCause(sh, sl)
			// A — the SHIPPED-BEFORE behaviour. This must call the pre-fix
			// replica: calling the production function here made A silently
			// track the fix and report the defect as costing nothing.
			if applyStructureGate(raw, hhhlStructurePreFix(sh, sl)) == raw {
				a++
			}
			// B — a window that never alternated carries no structural claim.
			stB := st
			if cause != "alternating-ok" {
				stB = ""
			}
			if applyStructureGate(raw, stB) == raw {
				b++
			}
			// C — demote only on the explicitly opposite structure.
			if !((raw == trendUp && st == "lh_ll") || (raw == trendDown && st == "hh_hl")) {
				c++
			}
		}
		if n == 0 {
			continue
		}
		p := func(v int) float64 { return 100 * float64(v) / float64(n) }
		fmt.Printf("%-32s %7.1f%% %7.1f%% %7.1f%% %7.1f%%\n", s.label, p(before), p(a), p(b), p(c))
	}

	fmt.Println("\n=== 4. sample pivot sequences (GOLD 1d) ===")
	for _, s := range all {
		if s.label != "GOLD 1d (gold-agent candidate)" {
			continue
		}
		highs, lows := highsLowsOf(s.bars)
		shown := 0
		for i := 400; i < len(s.bars) && shown < 4; i += 137 {
			sh, sl := swingPointsIdx(highs[:i+1], lows[:i+1], 3)
			fmt.Printf("  %s  %-44s → %s\n",
				time.Unix(s.bars[i].Time, 0).UTC().Format("2006-01-02"),
				pivotTail(sh, sl), structureCause(sh, sl))
			shown++
		}
	}
	fmt.Println()
}
