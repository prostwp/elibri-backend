package demobot

// gold_backtest_test.go — Этап 5, прогон золотого агента на истории.
//
//	MEASURE_GOLD_BACKTEST=1 go test ./internal/demobot/ \
//	    -run TestGoldBacktest -v
//
// Env-guarded: it hits Yahoo, while the normal suite stays offline.
//
// WHAT IS MEASURED. The run scores the daily trend state machine — the one
// place a direction can come from in this agent (macro never moves it, it only
// adds a conflict line) — plus the day-levels layer.
//
// It was built against the verdict wording "Day bias UP/DOWN". That wording is
// GONE: this run is why. It found no edge the sample could detect (it resolves
// about 12 pp and larger — the report's section 7), so the card now
// describes the regime ("Daily regime: confirmed UPTREND") instead of calling
// the day. The run is kept scoring the same underlying state so the claim
// "describes the period, does not forecast the day" stays checkable, and so a
// future change that quietly re-introduces a forecast can be re-measured
// against the same ruler.
//
// NO LOOK-AHEAD, by construction:
//   - the decision at bar i is computed from bars[0..i] and nothing else;
//   - it calls trendReadOf and goldDayLevelsOf — the SAME functions GoldCard
//     calls, not a restatement of them;
//   - the outcome is read from bar i+1, which the decision never saw.
//
// TWO OUTCOME DEFINITIONS, because a single one can be an artifact of how it
// was worded:
//
//	RANGE  the spec's own wording: day i+1 CLOSES beyond the day levels the
//	       card printed (up = close > High, down = close < Low, else inside)
//	CLOSE  the crude one: day i+1 closes above / below day i's close
//
// EVERY rate is printed beside its BASELINE on the same day set — the share of
// days that went that way regardless of what the agent said. A hit rate
// without its baseline says nothing: in a rising market "up" is right most of
// the time for free.

import (
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// goldBacktestWarmup is the first bar a decision may be taken on: EMA200 plus
// ADX(14) headroom, the same floor the live card enforces via trendReadOf.
// Shared with the live agent (trendMinBars) so the run can never take
// decisions the production card would refuse, nor refuse ones it would take.
const goldBacktestWarmup = trendMinBars

// btDecision is one day's decision paired with what the next day did.
type btDecision struct {
	Date        time.Time
	Claim       string // "up" | "down" | "" (silent)
	State       string // the full state, for the silence breakdown
	RangeOut    string // "up" | "down" | "inside"
	CloseOut    string // "up" | "down" | "flat" (equal settlements)
	LevelsKnown bool   // false → deep inside-day nesting, RANGE undefined
}

// runGoldBacktest walks the series once, taking every decision from the
// production functions.
func runGoldBacktest(bars []types.OHLCVCandle) []btDecision {
	var out []btDecision
	for i := goldBacktestWarmup; i < len(bars)-1; i++ {
		window := bars[:i+1]
		r := trendReadOf(window)
		if !r.OK {
			continue
		}
		levels := goldDayLevelsOf(window)
		next := bars[i+1]

		d := btDecision{
			Date:        time.Unix(bars[i].Time, 0).UTC(),
			State:       r.State,
			LevelsKnown: levels.Defined,
		}
		// The card's own rule: a direction exists only in a confirmed regime.
		if r.Confirmed() {
			if r.State == trendUp {
				d.Claim = "up"
			} else {
				d.Claim = "down"
			}
		}
		switch {
		case !levels.Defined:
			d.RangeOut = ""
		case next.Close > levels.High:
			d.RangeOut = "up"
		case next.Close < levels.Low:
			d.RangeOut = "down"
		default:
			d.RangeOut = "inside"
		}
		// A tie is neither direction. It used to fall into "down", which
		// inflated the down baseline and could score a DOWN claim correct on
		// a day price did not fall.
		switch {
		case next.Close > bars[i].Close:
			d.CloseOut = "up"
		case next.Close < bars[i].Close:
			d.CloseOut = "down"
		default:
			d.CloseOut = "flat"
		}
		out = append(out, d)
	}
	return out
}

// btRate is a share with the sample it came from, so noise stays visible.
type btRate struct {
	Hits, N int
}

func (r btRate) pct() float64 {
	if r.N == 0 {
		return math.NaN()
	}
	return 100 * float64(r.Hits) / float64(r.N)
}

// stderr is the binomial standard error in percentage points. Printed beside
// every rate on purpose: with ~170 directional days per half, a 3-point gap is
// inside the noise and must not be read as a finding.
func (r btRate) stderr() float64 {
	if r.N == 0 {
		return math.NaN()
	}
	p := float64(r.Hits) / float64(r.N)
	return 100 * math.Sqrt(p*(1-p)/float64(r.N))
}

func (r btRate) String() string {
	if r.N == 0 {
		return "     n/a"
	}
	return fmt.Sprintf("%5.1f%% ±%.1f (n=%d)", r.pct(), r.stderr(), r.N)
}

// btReport is one window's full picture.
type btReport struct {
	Label                        string
	N                            int
	Silent                       int
	SaidUp                       int
	SaidDown                     int
	StateMix                     map[string]int
	NoLevels                     int
	RangeUpOnUp, RangeDownOnDown btRate // conditional
	RangeUpBase, RangeDownBase   btRate // baseline, same day set
	CloseUpOnUp, CloseDownOnDown btRate
	CloseUpBase, CloseDownBase   btRate
	// Complements: the same outcome on the days the agent did NOT claim that
	// side. Disjoint from the conditional set — this is the honest contrast.
	RangeUpOff, RangeDownOff btRate
	CloseUpOff, CloseDownOff btRate
	RangeInsideShare         btRate
	ClusterLen               float64
}

func summarize(label string, ds []btDecision) btReport {
	rep := btReport{Label: label, N: len(ds), StateMix: map[string]int{}}
	for _, d := range ds {
		rep.StateMix[d.State]++
		switch d.Claim {
		case "up":
			rep.SaidUp++
		case "down":
			rep.SaidDown++
		default:
			rep.Silent++
		}
		if !d.LevelsKnown {
			rep.NoLevels++
		}

		// CLOSE outcome — defined on every day.
		rep.CloseUpBase.N++
		if d.CloseOut == "up" {
			rep.CloseUpBase.Hits++
		}
		rep.CloseDownBase.N++
		if d.CloseOut == "down" {
			rep.CloseDownBase.Hits++
		}
		if d.Claim == "up" {
			rep.CloseUpOnUp.N++
			if d.CloseOut == "up" {
				rep.CloseUpOnUp.Hits++
			}
		} else {
			rep.CloseUpOff.N++
			if d.CloseOut == "up" {
				rep.CloseUpOff.Hits++
			}
		}
		if d.Claim == "down" {
			rep.CloseDownOnDown.N++
			if d.CloseOut == "down" {
				rep.CloseDownOnDown.Hits++
			}
		} else {
			rep.CloseDownOff.N++
			if d.CloseOut == "down" {
				rep.CloseDownOff.Hits++
			}
		}

		// RANGE outcome — only on days where the levels existed. The baseline
		// is restricted to the SAME days, or the comparison is rigged.
		if !d.LevelsKnown {
			continue
		}
		rep.RangeUpBase.N++
		if d.RangeOut == "up" {
			rep.RangeUpBase.Hits++
		}
		rep.RangeDownBase.N++
		if d.RangeOut == "down" {
			rep.RangeDownBase.Hits++
		}
		rep.RangeInsideShare.N++
		if d.RangeOut == "inside" {
			rep.RangeInsideShare.Hits++
		}
		if d.Claim == "up" {
			rep.RangeUpOnUp.N++
			if d.RangeOut == "up" {
				rep.RangeUpOnUp.Hits++
			}
		} else {
			rep.RangeUpOff.N++
			if d.RangeOut == "up" {
				rep.RangeUpOff.Hits++
			}
		}
		if d.Claim == "down" {
			rep.RangeDownOnDown.N++
			if d.RangeOut == "down" {
				rep.RangeDownOnDown.Hits++
			}
		} else {
			rep.RangeDownOff.N++
			if d.RangeOut == "down" {
				rep.RangeDownOff.Hits++
			}
		}
	}
	rep.ClusterLen = meanClusterLen(ds)
	return rep
}

// lift prints the conditional rate against the days the agent did NOT claim,
// with an error bar wide enough to be honest about how these days are related.
//
// Two corrections over the first version, both from the Этап 6 review:
//
//  1. THE COMPARISON. It used to be conditional-vs-baseline, where the
//     baseline was ALL days — a set that CONTAINS the conditional days. A
//     subset compared against its own superset is not two independent
//     samples, and the difference is mechanically damped: the claimed days
//     drag the baseline toward themselves. The honest contrast is against
//     the COMPLEMENT — the days the agent stayed silent or claimed the other
//     side. Those two sets are disjoint, so the difference means what it
//     looks like and the independent-variance formula applies to it.
//
//  2. THE EFFECTIVE SAMPLE. Days are not independent draws. EMA200 and
//     ADX(14) run over 220+ bars, so a regime persists for weeks and the
//     agent repeats the same claim across a whole cluster. Treating 523
//     claim-days as 523 experiments overstates what was actually observed.
//     The error bar is therefore widened by sqrt(mean cluster length),
//     the standard first-order correction for serial dependence.
//
// Both corrections make the interval WIDER, i.e. both make it harder to call
// something a finding. That is the direction an honest fix should go when the
// headline result is "no effect".
func lift(name string, cond, comp btRate, clusterLen float64) string {
	if cond.N == 0 || comp.N == 0 {
		return fmt.Sprintf("  %-28s %s", name, "n/a")
	}
	diff := cond.pct() - comp.pct()
	se := math.Sqrt(cond.stderr()*cond.stderr() + comp.stderr()*comp.stderr())
	if clusterLen > 1 {
		se *= math.Sqrt(clusterLen)
	}
	verdict := "inside the noise"
	if math.Abs(diff) > 2*se {
		verdict = "OUTSIDE the noise"
	}
	return fmt.Sprintf("  %-28s claimed %s · not-claimed %s · diff %+.1f pp (±%.1f, %s)",
		name, cond, comp, diff, se, verdict)
}

// meanClusterLen is the average length of a run of identical claims — the
// measure of how much less information the sample carries than its day count
// suggests. A regime that holds for three weeks contributes one observation
// of "this regime", not fifteen.
func meanClusterLen(ds []btDecision) float64 {
	if len(ds) == 0 {
		return 1
	}
	runs, prev := 1, ds[0].Claim
	for _, d := range ds[1:] {
		if d.Claim != prev {
			runs++
			prev = d.Claim
		}
	}
	return float64(len(ds)) / float64(runs)
}

func printReport(rep btReport) {
	fmt.Printf("\n── %s ──  %d decision days\n", rep.Label, rep.N)
	if rep.N == 0 {
		return
	}
	p := func(v int) float64 { return 100 * float64(v) / float64(rep.N) }
	fmt.Printf("  silent %d (%.1f%%) · said UP %d (%.1f%%) · said DOWN %d (%.1f%%)\n",
		rep.Silent, p(rep.Silent), rep.SaidUp, p(rep.SaidUp), rep.SaidDown, p(rep.SaidDown))
	fmt.Printf("  states:")
	for _, st := range []string{trendUp, trendDown, trendGrey, trendFlat, trendConflict} {
		fmt.Printf(" %s %.1f%%", st, p(rep.StateMix[st]))
	}
	fmt.Println()
	if rep.NoLevels > 0 {
		fmt.Printf("  day levels undefined on %d days (%.1f%%) — excluded from RANGE\n",
			rep.NoLevels, p(rep.NoLevels))
	}
	fmt.Printf("  RANGE: day closed inside the levels on %s\n", rep.RangeInsideShare)
	fmt.Printf("  unconditional: RANGE up %s · RANGE down %s · CLOSE up %s\n",
		rep.RangeUpBase, rep.RangeDownBase, rep.CloseUpBase)
	fmt.Printf("  mean run of identical claims: %.1f days → error bars widened x%.2f\n",
		rep.ClusterLen, math.Sqrt(rep.ClusterLen))
	fmt.Println(lift("RANGE up | said UP", rep.RangeUpOnUp, rep.RangeUpOff, rep.ClusterLen))
	fmt.Println(lift("RANGE down | said DOWN", rep.RangeDownOnDown, rep.RangeDownOff, rep.ClusterLen))
	fmt.Println(lift("CLOSE up | said UP", rep.CloseUpOnUp, rep.CloseUpOff, rep.ClusterLen))
	fmt.Println(lift("CLOSE down | said DOWN", rep.CloseDownOnDown, rep.CloseDownOff, rep.ClusterLen))
}

// runGoldBacktestPeeking lets the STATE MACHINE see one bar into the future.
//
// It turned out to probe the agent rather than the harness: EMA200 and ADX(14)
// are computed over 220+ bars, so one extra bar moves the state on 1 day out
// of 2290 (523 → 524 "UP"). That near-zero response is itself a finding — the
// regime read is slow and describes a period, not a day — but it means this
// control CANNOT validate the scoring path. runGoldBacktestOracle does that.
func runGoldBacktestPeeking(bars []types.OHLCVCandle) []btDecision {
	var out []btDecision
	for i := goldBacktestWarmup; i < len(bars)-1; i++ {
		window := bars[:i+2] // ← THE POISON: one bar into the future
		r := trendReadOf(window)
		if !r.OK {
			continue
		}
		levels := goldDayLevelsOf(bars[:i+1]) // levels stay honest
		next := bars[i+1]
		d := btDecision{
			Date:        time.Unix(bars[i].Time, 0).UTC(),
			State:       r.State,
			LevelsKnown: levels.Defined,
		}
		if r.Confirmed() {
			if r.State == trendUp {
				d.Claim = "up"
			} else {
				d.Claim = "down"
			}
		}
		switch {
		case !levels.Defined:
			d.RangeOut = ""
		case next.Close > levels.High:
			d.RangeOut = "up"
		case next.Close < levels.Low:
			d.RangeOut = "down"
		default:
			d.RangeOut = "inside"
		}
		// A tie is neither direction. It used to fall into "down", which
		// inflated the down baseline and could score a DOWN claim correct on
		// a day price did not fall.
		switch {
		case next.Close > bars[i].Close:
			d.CloseOut = "up"
		case next.Close < bars[i].Close:
			d.CloseOut = "down"
		default:
			d.CloseOut = "flat"
		}
		out = append(out, d)
	}
	return out
}

// runGoldBacktestOracle is the control that validates the SCORING path: the
// claim is copied straight from the outcome it will be scored against, on the
// same days the real agent chose to speak.
//
// A sound harness must score this at (or very near) 100%. Anything less means
// the conditional and the outcome are not lined up — the failure mode that
// makes a null result meaningless, because a mis-wired scorer reports "no
// edge" for a perfect forecaster too.
//
// It is a plumbing test. It says nothing about gold and nothing about the
// agent; it only proves the ruler is not bent.
func runGoldBacktestOracle(honest []btDecision, from string) []btDecision {
	out := make([]btDecision, 0, len(honest))
	for _, d := range honest {
		if d.Claim != "" { // speak on exactly the days the agent spoke
			switch from {
			case "close":
				d.Claim = d.CloseOut
			default:
				d.Claim = d.RangeOut
			}
			// "inside"/"flat" are not directions the agent can claim; drop to
			// silence so the oracle only ever asserts a real side.
			if d.Claim == "inside" || d.Claim == "flat" {
				d.Claim = ""
			}
		}
		out = append(out, d)
	}
	return out
}

// pctOrFail returns the rate and whether it is a real number. NaN must never
// reach a comparison: `math.NaN() < 99.9` is FALSE, so an empty branch used to
// slip through the oracle guard as a success — the exact silent failure the
// guard exists to catch.
func pctOrFail(r btRate) (float64, bool) {
	if r.N == 0 {
		return 0, false
	}
	v := r.pct()
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, false
	}
	return v, true
}

// claimWord renders an empty claim as the word the report uses for it.
func claimWord(c string) string {
	if c == "" {
		return "молчание"
	}
	return c
}

func TestGoldBacktest(t *testing.T) {
	if os.Getenv("MEASURE_GOLD_BACKTEST") == "" {
		t.Skip("set MEASURE_GOLD_BACKTEST=1 (hits Yahoo)")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	raw, err := fetchYahooChart(ctx, "GC=F", "1d", "10y")
	if err != nil {
		t.Skipf("fetch failed: %v", err)
	}
	bars := dropUnclosedBars(raw, "1d", time.Now())
	if len(bars) < goldBacktestWarmup+50 {
		t.Skipf("too few bars: %d", len(bars))
	}

	ds := runGoldBacktest(bars)
	if len(ds) == 0 {
		t.Fatal("no decisions produced")
	}

	fmt.Printf("\nGOLD AGENT · ЭТАП 5 · прогон на истории\n")
	fmt.Printf("measured %s UTC\n", time.Now().UTC().Format("2006-01-02 15:04"))
	fmt.Printf("source GC=F daily · %d closed bars · %s .. %s\n",
		len(bars),
		time.Unix(bars[0].Time, 0).UTC().Format("2006-01-02"),
		time.Unix(bars[len(bars)-1].Time, 0).UTC().Format("2006-01-02"))
	fmt.Printf("warmup %d bars · %d decision days · decision at bar i uses bars[0..i] only\n",
		goldBacktestWarmup, len(ds))
	// Honest accounting of choices, not the "0 attempts" line this used to
	// print. That claim was false: two decisions WERE informed by this same
	// history, and a run that hides them invites exactly the self-deception
	// Этап 5 exists to prevent.
	fmt.Println("\nПОДБОР НАСТРОЕК — честный учёт:")
	fmt.Println("  порогов, подобранных под золото: 0.")
	fmt.Println("    ADX 20/25, EMA 20/50/200, ATR 14 — платформенные, общие для всех агентов,")
	fmt.Println("    выбраны до этого агента и под него не двигались.")
	fmt.Println("  решений, принятых ПОСЛЕ просмотра этой же истории: 2.")
	fmt.Println("    1) dayUnwindCap=3 — выбран после замера вложенности на 5 годах")
	fmt.Println("       (глубже 2 не встретилось; 3 = запас). Любой cap >= 2 даёт то же самое.")
	fmt.Println("    2) вариант гейта структуры — выбран после сравнения A/B/C на истории.")
	fmt.Println("       ВАЖНО: сравнение шло по доле СОХРАНЁННЫХ подтверждений, а не по их")
	fmt.Println("       правильности. По такой метрике вариант, который почти не понижает")
	fmt.Println("       режим, выигрывает автоматически — даже если плодит ложные")
	fmt.Println("       подтверждения. Это известное ограничение, не измеренное здесь.")
	fmt.Println("  выборка этого прогона ПЕРЕСЕКАЕТСЯ с той, на которой приняты оба решения.")

	printReport(summarize("ВСЯ ВЫБОРКА", ds))

	// Split-half. The agent has nothing to tune, so this is a STABILITY check,
	// not a train/test split: the two halves must not disagree.
	mid := len(ds) / 2
	first, second := ds[:mid], ds[mid:]
	fmt.Printf("\n\n=== ДЕЛЕНИЕ НАДВОЕ (Этап 5) ===\n")
	fmt.Printf("первая половина  %s .. %s\n",
		first[0].Date.Format("2006-01-02"), first[len(first)-1].Date.Format("2006-01-02"))
	fmt.Printf("вторая половина  %s .. %s\n",
		second[0].Date.Format("2006-01-02"), second[len(second)-1].Date.Format("2006-01-02"))
	printReport(summarize("ПЕРВАЯ ПОЛОВИНА", first))
	printReport(summarize("ВТОРАЯ ПОЛОВИНА", second))

	// ── контроли ───────────────────────────────────────────────────────────
	// A null result is worth exactly as much as the ruler that produced it, so
	// the ruler is checked before the result is believed.

	fmt.Printf("\n\n=== КОНТРОЛЬ 1: ОРАКУЛ (проверка тракта подсчёта) ===\n")
	fmt.Println("вердикт берётся прямо из будущего исхода, в те же дни, когда говорил агент.")
	fmt.Println("исправный харнесс обязан дать здесь ~100%. Меньше — счёт разъехался с исходом.")
	// BOTH scorers are checked. The oracle used to copy only CloseOut and
	// assert only CLOSE, so the entire RANGE scoring path could have been
	// broken while the run still printed "тракт подсчёта исправен".
	oracleClose := summarize("ОРАКУЛ · CLOSE", runGoldBacktestOracle(ds, "close"))
	printReport(oracleClose)
	oracleRange := summarize("ОРАКУЛ · RANGE", runGoldBacktestOracle(ds, "range"))
	printReport(oracleRange)

	checks := []struct {
		name string
		rate btRate
	}{
		{"CLOSE up | said UP", oracleClose.CloseUpOnUp},
		{"CLOSE down | said DOWN", oracleClose.CloseDownOnDown},
		{"RANGE up | said UP", oracleRange.RangeUpOnUp},
		{"RANGE down | said DOWN", oracleRange.RangeDownOnDown},
	}
	allOK := true
	for _, c := range checks {
		v, ok := pctOrFail(c.rate)
		switch {
		case !ok:
			allOK = false
			t.Errorf("ОРАКУЛ %s: пустая или нечисловая доля (n=%d) — проверка не состоялась, "+
				"а не прошла", c.name, c.rate.N)
		case v < 99.9:
			allOK = false
			t.Errorf("ОРАКУЛ %s = %.1f%%, должно быть 100%% — тракт подсчёта сломан, "+
				"нулевой результат ничего не значит", c.name, v)
		}
	}
	if allOK {
		fmt.Println("\n  ✓ оба тракта подсчёта исправны: оракул набирает 100% и на CLOSE, и на RANGE")
	}

	fmt.Printf("\n\n=== КОНТРОЛЬ 2: ОДИН БАР ВПЕРЁД (свойство агента, не харнесса) ===\n")
	fmt.Println("машине состояний дают заглянуть на бар вперёд. Реакция почти нулевая:")
	fmt.Println("EMA200 и ADX(14) считаются по 220+ барам, один бар их не двигает.")
	peekDs := runGoldBacktestPeeking(bars)

	// Compared PAIRWISE, by date. Comparing aggregate UP counts (523 vs 524)
	// does not support "one verdict changed": [up, silent] → [silent, up]
	// leaves the count identical while two decisions moved, and DOWN / grey /
	// conflict were not compared at all.
	byDate := make(map[int64]btDecision, len(ds))
	for _, d := range ds {
		byDate[d.Date.Unix()] = d
	}
	compared, changedClaim, changedState := 0, 0, 0
	var examples []string
	for _, p := range peekDs {
		h, ok := byDate[p.Date.Unix()]
		if !ok {
			continue
		}
		compared++
		if h.State != p.State {
			changedState++
		}
		if h.Claim != p.Claim {
			changedClaim++
			if len(examples) < 3 {
				examples = append(examples, fmt.Sprintf("%s: %q→%q",
					p.Date.Format("2006-01-02"), claimWord(h.Claim), claimWord(p.Claim)))
			}
		}
	}
	fmt.Printf("  сопоставлено дней попарно: %d\n", compared)
	fmt.Printf("  изменился ВЕРДИКТ (up/down/молчание): %d дн. (%.2f%%)\n",
		changedClaim, 100*float64(changedClaim)/float64(compared))
	fmt.Printf("  изменилось СОСТОЯНИЕ (включая grey/flat/conflict): %d дн. (%.2f%%)\n",
		changedState, 100*float64(changedState)/float64(compared))
	if len(examples) > 0 {
		fmt.Printf("  примеры: %s\n", strings.Join(examples, " · "))
	}
	fmt.Println("  вывод: режимный вердикт описывает период, а не следующий день.")
	fmt.Println()
}
