package demobot

// gold_roll.go — which COMEX contract the gold card's two GC=F series are on.
//
// GC=F is Yahoo's continuous front-month future, and its hourly series moves
// to the next contract about two daily bars BEFORE the daily series does
// (measured on the saved Yahoo answers, gold_roll/out/отчёт.md: first hourly
// bar of the new contract 2025-07-30 07:00, 2025-11-25 09:00 and 2026-07-29
// 07:00 UTC against daily rolls on 2025-08-01, 2025-11-28 and 2026-07-31).
// In those hours the last closed 1h price is a price of the new contract
// while the day range, the S/R levels and the EMAs are prices of the old one,
// about 1–1.7% apart. Placing one against the other said "above the day
// range" in 52 of the 103 hours of the two verified windows where, in one
// contract's prices, the price sat inside it.
//
// Nothing here changes a rule. It only establishes, from Yahoo's own dated
// contracts, whether the last closed 1h bar and the last closed 1d bar of
// GC=F come from the same contract; gold_text.go decides what the card then
// may and may not say.
//
// How it is established:
//   - the candidates are three consecutive contracts of GC=F's cycle G/J/M/Q/Z
//     (Feb, Apr, Jun, Aug, Dec — October is skipped: GC=F went Q25 → Z25 and
//     Q26 → Z26 although V25 and V26 existed) around the date of the last
//     closed daily bar: the first contract delivering after that date's month
//     (near), the one after it (next), and the one before it (previous). The
//     daily bar is on near until its roll at the end of the month before
//     delivery, then on next. The established daily roll 2025-08-01 and the
//     research's candidate rolls 2025-04-01, 2025-06-02 and 2026-04-01 fell on
//     the first trading day of the delivery month: had Yahoo switched the
//     daily series one day later, that day's bar would still be the contract
//     whose month has just begun — previous, by then;
//   - the last closed 1d bar of GC=F is compared with the bar of the same
//     stamp in each candidate's daily series on all four prices (the saved
//     answers: every day inside a verified contract period matched on O, H, L
//     and C; not one day matched on the close alone). Bars are paired by
//     their stamp only — never by position, never by the nearest stamp;
//   - the last closed 1h bar is compared on its CLOSE — the one hourly number
//     the card uses. All four prices do NOT work hourly: in the first days
//     after a switch GC=F's hourly open/high/low still carry the old contract
//     (GC=F against GCZ25, bars opening 2025-07-30 07:00 … 2025-08-01 04:00
//     UTC, both ends included: the close matches in 44 of 44 hours, all four
//     prices in 2 — gold_roll_data_test.go), so a four-price rule would miss
//     the very window it is for;
//   - equal means within goldRollTol, half the 0.10 tick: the two contracts
//     sit tens of dollars apart, the same bar of one contract differs by 0;
//   - a GC=F row whose stamp is off the half-hour grid is Yahoo's current
//     quote, not a bar (see goldOnBarGrid); it is never matched.
//
// Each bar on exactly one candidate: the same one → none; different ones →
// window, whichever of the two series is on the later contract (the hourly
// one runs ahead in every measured roll; the reverse would put two contracts
// against each other just the same, and gold.roll.reason tells them apart).
// Any other outcome is unknown — never "no roll" and never "roll".
//
// One bar matching one served candidate is enough although another candidate
// did not answer: a bar that is one contract's print cannot also be
// another's — adjacent contracts print tens of dollars apart (on the
// research's unmatched days the nearest contract was at least 18.50 away),
// against a tolerance of 0.05. A bar that matches nothing while near or next
// did not answer is unknown with that candidate's reason. The previous
// contract is the exception: once it has expired Yahoo answers 404 for it
// (gold_roll/data/_fetch_log.tsv: every expired contract but GCZ25 answered
// 404, every unexpired one 200), and an expired contract is not what GC=F is
// on — so its 404 leaves an unmatched bar "matches_neither_contract", while
// a failed request for it is "request_failed" like any other.
//
// Load on Yahoo, which answers 429 when asked too often: the previous
// contract's 404 is remembered for a day once its delivery month is over
// (goldRollGoneFor), and an unknown
// answer is asked again under the same bars after a pause that grows unless
// the request itself failed (goldRollRetry).

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// Machine values of gold.roll.state.
const (
	goldRollNone    = "none"    // both bars are on one contract
	goldRollWindow  = "window"  // the 1h bar is on one contract, the 1d bar on another
	goldRollUnknown = "unknown" // not established — see gold.roll.reason
)

// Machine values of gold.roll.reason. A window says which way round it is;
// unknown says why nothing was established; none carries no reason.
const (
	goldRollHourlyAhead  = "hourly_ahead_of_daily" // window: the 1h bar on the later contract (every measured roll)
	goldRollHourlyBehind = "hourly_behind_daily"   // window: the 1h bar on the earlier contract

	goldRollNoPrice     = "no_intraday_price"        // no 1h bar to place
	goldRollNearMissing = "near_contract_not_served" // Yahoo answered 404 for near
	goldRollNextMissing = "next_contract_not_served" // … for next
	goldRollFetchFailed = "request_failed"           // transport error, timeout, a non-404 status, a bad body
	goldRollMatchesBoth = "matches_both_contracts"   // two or more candidates carry the bar
	goldRollMatchesNone = "matches_neither_contract" // incl. no bar of that stamp in any series
	goldRollQuoteRow    = "row_is_a_quote"           // GC=F's row is Yahoo's current quote, not a bar
	goldRollNoDailyBars = "no_daily_bars"
	goldRollNotChecked  = "not_checked" // defensive: never set on a card GoldCard builds
)

// goldRollContractDisp is the contract code prefix as printed: GCZ26.
const goldRollContractDisp = "GC"

// goldRollTol is how far two prices may differ and still be the same print:
// half a tick (COMEX gold ticks in 0.10). Yahoo stores them as float32-ish
// numbers (4125.10009765625), so exact equality would be the wrong test.
const goldRollTol = 0.05

// goldRollRetry is how long an unknown answer stands before the contracts are
// asked again under the same bars. A definite answer stands until a new bar
// closes (it names the contract of THAT bar; nothing later can change it).
//
// The pause is goldRollRetry after a failed request (transport error,
// timeout, 5xx, a bad body): the next try may well get through. Every other
// unknown — Yahoo answered and the bar still matched nothing or two
// contracts, near or next answered 404, or Yahoo refused the request with
// any other 4xx (429 too many, 403/401 blocked) — doubles the pause on each
// repeat under the same bars, 15 → 30 → 60 → 120 minutes, and stays at
// goldRollRetryMax until a new bar closes. Asking again at once would get
// the same answers (or, on a refusal, feed the limit or the block); asking never
// again under the same bar would freeze an unknown for a whole weekend, or a
// whole day on the daily side, although a contract's series that caught up
// later, or a late print, can still settle it.
const goldRollRetry = 15 * time.Minute

// goldRollRetryMax caps the growing pause. It bounds how late an unknown
// settles once Yahoo's answers line up — it matters on the daily side, whose
// bar stays the same for a day — and keeps a weekend (50 hours under the
// same bars) to about 28 rounds instead of 200.
const goldRollRetryMax = 2 * time.Hour

// goldRollGoneFor is how long a 404 for the PREVIOUS contract is remembered
// — only once its delivery month is over (goldPrevExpired), per contract and
// interval. A contract that has expired does not come back, so it need not
// be asked on every new bar (about 24 requests a trading day, for months); a
// day bounds the damage of a 404 that was only momentary.
//
// Not remembered: near and next, and previous in its own delivery month (on
// the first days of a delivery month the daily bar may still be on it —
// file comment). For those a 404 that was only momentary would hold the card
// unknown, or unknown instead of window, for the day; asked again, they
// settle it at the next retry.
//
// Per interval: a 404 of the hourly series says nothing certain about the
// daily one. It costs nothing in practice — both series are asked side by
// side on the first round anyway, and each once a day after that.
const goldRollGoneFor = 24 * time.Hour

// goldRollFetchTimeout bounds one round of contract requests. The round runs
// detached from the caller's context: it is shared by every concurrent reader
// and cached, and one reader hanging up must not leave all of them unknown.
// The daily and the hourly rounds run side by side, so a slow Yahoo costs
// the first read after a bar closes at most this once, not twice.
const goldRollFetchTimeout = 20 * time.Second

// goldRollRange is the window asked for per contract series. The bar looked up
// is the last CLOSED GC=F bar, so a month is ample; the answer is small.
const goldRollRange = "1mo"

// goldCycle is the delivery months GC=F rolls through, with their codes.
var goldCycle = []struct {
	month time.Month
	code  byte
}{{time.February, 'G'}, {time.April, 'J'}, {time.June, 'M'}, {time.August, 'Q'}, {time.December, 'Z'}}

// goldContract names the contract delivering in (year, month): "GCZ26".
func goldContract(year int, month time.Month) string {
	for _, c := range goldCycle {
		if c.month == month {
			return fmt.Sprintf("%s%c%02d", goldRollContractDisp, c.code, year%100)
		}
	}
	return ""
}

// goldContractAfter is the first cycle contract delivering in a month
// strictly after (year, month).
func goldContractAfter(year int, month time.Month) (int, time.Month) {
	for _, c := range goldCycle {
		if c.month > month {
			return year, c.month
		}
	}
	return year + 1, goldCycle[0].month
}

// goldContractBefore is the last cycle contract delivering in a month
// strictly before (year, month).
func goldContractBefore(year int, month time.Month) (int, time.Month) {
	for i := len(goldCycle) - 1; i >= 0; i-- {
		if goldCycle[i].month < month {
			return year, goldCycle[i].month
		}
	}
	return year - 1, goldCycle[len(goldCycle)-1].month
}

// Positions in goldCandidates' answer, oldest contract first.
const (
	goldCandPrev = iota
	goldCandNear
	goldCandNext
)

// goldCandidates are the three contracts the bars of this date can be on,
// oldest first: previous, near (the first cycle contract after the date's
// month — GC=F's daily contract until its roll at the end of the month
// before delivery) and next.
func goldCandidates(day time.Time) [3]string {
	y, m := goldContractAfter(day.Year(), day.Month())
	py, pm := goldContractBefore(y, m)
	ny, nm := goldContractAfter(y, m)
	return [3]string{goldContract(py, pm), goldContract(y, m), goldContract(ny, nm)}
}

// goldNextContract is the cycle successor of a contract code, "" when the code
// is not one of ours.
func goldNextContract(code string) string {
	if len(code) != 5 {
		return ""
	}
	var yy int
	if _, err := fmt.Sscanf(code[3:], "%02d", &yy); err != nil {
		return ""
	}
	for _, c := range goldCycle {
		if c.code == code[2] {
			y, m := goldContractAfter(2000+yy, c.month)
			return goldContract(y, m)
		}
	}
	return ""
}

// goldYahooSymbol is Yahoo's symbol of a dated COMEX contract.
func goldYahooSymbol(code string) string { return code + ".CMX" }

// goldOnBarGrid reports whether a GC=F row is a bar: its stamp sits on the
// half-hour grid. Hourly bars open on the hour, and on half-day sessions on
// the half hour (2025-11-28 14:30, 2025-12-24 14:30 …); Yahoo's current-quote
// row carries the time of the last trade (2026-09-23 08:30:52). In the
// research's GC=F 1h answer 25 of 17388 rows are off the hour: 24 half-day
// bars on :30 and the one quote row. Daily bars open at 04:00/05:00 UTC, or
// 13:30/14:30 on a half day — on the grid as well. The closed-bar cut
// already drops a quote row minutes old; this keeps one that outlived the cut
// (a fetch long after the last trade) out of the match.
func goldOnBarGrid(ts int64) bool { return ts%1800 == 0 }

// goldRollSide is what one GC=F bar was matched to.
type goldRollSide struct {
	contract string // the one candidate it matched; "" when not established
	idx      int    // its position in the candidates, oldest first
	reason   string // why not, when contract == ""
	detail   string // the request error behind reason, for the log only
	quick    bool   // a failed request, not a refusal (4xx): retried after goldRollRetry, not later
}

// goldRollFetch is one candidate's answer.
type goldRollFetch struct {
	bars []types.OHLCVCandle
	err  error
}

// goldSamePrint reports whether two prices are the same print (goldRollTol).
func goldSamePrint(a, b float64) bool { return math.Abs(a-b) <= goldRollTol }

// goldBarOn reports whether GC=F's bar is the bar of the SAME STAMP in a
// contract's series: all four prices on daily bars, the close on hourly ones
// (see the file comment for why). No other bar of the series is looked at —
// not the one at the same position, not the nearest one.
func goldBarOn(bar types.OHLCVCandle, series []types.OHLCVCandle, interval string) bool {
	for _, s := range series {
		if s.Time != bar.Time {
			continue
		}
		if interval == "1h" {
			return goldSamePrint(bar.Close, s.Close)
		}
		return goldSamePrint(bar.Open, s.Open) && goldSamePrint(bar.High, s.High) &&
			goldSamePrint(bar.Low, s.Low) && goldSamePrint(bar.Close, s.Close)
	}
	return false
}

// goldNotServed reports whether a contract request failed because Yahoo does
// not serve that contract (404), as opposed to the request itself failing.
func goldNotServed(err error) bool {
	var he *yahooHTTPError
	return errors.As(err, &he) && he.Status == http.StatusNotFound
}

// goldRefused reports whether Yahoo refused a request (a 4xx: 429 too many,
// 403/401 blocked …) rather than the request failing (transport, timeout,
// 5xx, a bad body). A refusal is not lifted by asking again in 15 minutes.
// 404 never gets here — it is "not served" (goldNotServed).
func goldRefused(err error) bool {
	var he *yahooHTTPError
	return errors.As(err, &he) && he.Status >= 400 && he.Status < 500
}

// goldRollSideOf matches one bar against the three candidates' series.
func goldRollSideOf(bar types.OHLCVCandle, interval string, cands [3]string, got [3]goldRollFetch) goldRollSide {
	matched, n := -1, 0
	for i := range cands {
		if got[i].err == nil && goldBarOn(bar, got[i].bars, interval) {
			matched, n = i, n+1
		}
	}
	switch {
	case n > 1:
		return goldRollSide{reason: goldRollMatchesBoth}
	case n == 1:
		return goldRollSide{contract: cands[matched], idx: matched}
	}
	// Nothing matched: a candidate that did not answer may hold the bar —
	// near first (the daily bar's usual contract), then next, then previous,
	// whose 404 means expired (file comment) and does not count.
	for _, i := range []int{goldCandNear, goldCandNext, goldCandPrev} {
		err := got[i].err
		switch {
		case err == nil:
			continue
		case goldNotServed(err) && i == goldCandNear:
			return goldRollSide{reason: goldRollNearMissing, detail: err.Error()}
		case goldNotServed(err) && i == goldCandNext:
			return goldRollSide{reason: goldRollNextMissing, detail: err.Error()}
		case goldNotServed(err): // previous: expired
			continue
		}
		return goldRollSide{reason: goldRollFetchFailed, detail: err.Error(), quick: !goldRefused(err)}
	}
	// Here only previous can be without an answer (its 404); the log says so,
	// or the line would read as if all three had answered.
	var silent []string
	for i := range cands {
		if got[i].err != nil {
			silent = append(silent, got[i].err.Error())
		}
	}
	side := goldRollSide{reason: goldRollMatchesNone}
	if len(silent) > 0 {
		side.detail = "did not answer: " + strings.Join(silent, "; ")
	}
	return side
}

// goldRollFrom combines the two sides into the card's roll state.
func goldRollFrom(day, hour goldRollSide) GoldRoll {
	switch {
	case day.contract == "":
		return goldRollUnknownOf(day.reason)
	case hour.contract == "":
		return goldRollUnknownOf(hour.reason)
	}
	d, h := day.contract, hour.contract
	if day.idx == hour.idx {
		cur, nxt := d, goldNextContract(d)
		return GoldRoll{State: goldRollNone, CurrentContract: &cur, NextContract: &nxt, DailyContract: &d, HourlyContract: &h}
	}
	// Two contracts: current/next are the earlier and the later of them,
	// whichever series is on which.
	cur, nxt, why := d, h, goldRollHourlyAhead
	if hour.idx < day.idx {
		cur, nxt, why = h, d, goldRollHourlyBehind
	}
	return GoldRoll{State: goldRollWindow, Reason: &why, CurrentContract: &cur, NextContract: &nxt, DailyContract: &d, HourlyContract: &h}
}

func goldRollUnknownOf(reason string) GoldRoll {
	r := reason
	return GoldRoll{State: goldRollUnknown, Reason: &r}
}

// ── cache and log ────────────────────────────────────────────────────────────

// goldRollMemo keeps the last answer per interval, keyed by the GC=F bar it
// was established for. The card is read every minute by the push hook and on
// every GET, but the answer can only change when a new bar closes: a
// definite answer is kept until then, an unknown one is asked again after
// its pause (goldRollRetry). Concurrent readers of the same key join one
// round of requests. Three requests per new hourly bar, three per new daily
// bar — two while the previous contract's 404 is remembered (goldRollGoneFor).
type goldRollMemo struct {
	mu   sync.Mutex
	last map[string]*goldRollEntry // by interval

	goneMu sync.Mutex
	gone   map[string]goldRollGone // expired previous contracts answered 404, by "code|interval"

	logf    func(format string, args ...any) // log.Printf; tests capture it
	logMu   sync.Mutex
	lastLog string // the key of the last state logged (goldRollLogKey)
}

type goldRollEntry struct {
	key  string
	done chan struct{}
	side goldRollSide
	at   time.Time
	// For an unknown side: how long it stands, and how many unknowns in a
	// row under this key have grown the pause (goldRollPause).
	retry  time.Duration
	streak int
}

// goldRollGone is a remembered 404: the answer, and until when it stands.
type goldRollGone struct {
	err   error
	until time.Time
}

func newGoldRollMemo() *goldRollMemo {
	return &goldRollMemo{last: map[string]*goldRollEntry{}, gone: map[string]goldRollGone{}, logf: log.Printf}
}

// goldRollPause is how long an unknown side stands before its bar is asked
// again, given the growing unknowns before it under the same bar; it returns
// the streak to carry. See goldRollRetry.
func goldRollPause(s goldRollSide, streak int) (time.Duration, int) {
	if s.quick {
		return goldRollRetry, 0
	}
	p := goldRollRetry
	for i := 0; i < streak && p < goldRollRetryMax; i++ {
		p *= 2
	}
	return min(p, goldRollRetryMax), streak + 1
}

func (m *goldRollMemo) side(interval, key string, now time.Time, load func() goldRollSide) goldRollSide {
	m.mu.Lock()
	streak := 0
	if e := m.last[interval]; e != nil && e.key == key {
		select {
		case <-e.done:
			if e.side.contract != "" || now.Sub(e.at) < e.retry {
				m.mu.Unlock()
				return e.side
			}
			streak = e.streak
		default: // in flight — join it
			m.mu.Unlock()
			<-e.done
			return e.side
		}
	}
	e := &goldRollEntry{key: key, done: make(chan struct{})}
	m.last[interval] = e
	m.mu.Unlock()

	e.side, e.at = load(), now
	if e.side.contract == "" {
		e.retry, e.streak = goldRollPause(e.side, streak)
	}
	close(e.done)
	return e.side
}

// goneErr is the remembered 404 of a contract's series, nil when there is
// none (or it has run out).
func (m *goldRollMemo) goneErr(code, interval string, now time.Time) error {
	k := code + "|" + interval
	m.goneMu.Lock()
	defer m.goneMu.Unlock()
	if g, ok := m.gone[k]; ok && now.Before(g.until) {
		return g.err
	}
	delete(m.gone, k)
	return nil
}

// markGone remembers a contract series' 404 for goldRollGoneFor.
func (m *goldRollMemo) markGone(code, interval string, err error, now time.Time) {
	m.goneMu.Lock()
	defer m.goneMu.Unlock()
	m.gone[code+"|"+interval] = goldRollGone{err: err, until: now.Add(goldRollGoneFor)}
}

// goldPrevExpired reports whether the previous candidate of this date has
// expired: its delivery month is over, strictly before the date's month.
// On the first days of its own delivery month it has not (it still trades,
// and the daily bar may still be on it).
func goldPrevExpired(day time.Time) bool {
	y, m := goldContractAfter(day.Year(), day.Month())
	py, pm := goldContractBefore(y, m)
	return py < day.Year() || (py == day.Year() && pm < day.Month())
}

// goldRollLogKey is what makes a roll state worth one log line: none and
// window are logged when the state or a contract changes (not on every new
// bar), unknown once per reason and pair of bars — a retry under the same
// bars that fails the same way is not a new line, a new bar still unknown is.
func goldRollLogKey(r GoldRoll, dayT, hourT int64) string {
	reason := ""
	if r.Reason != nil {
		reason = *r.Reason
	}
	if r.State == goldRollUnknown {
		return fmt.Sprintf("%s|%s|%d|%d", r.State, reason, dayT, hourT)
	}
	return fmt.Sprintf("%s|%s|%s|%s", r.State, reason, goldDeref(r.DailyContract), goldDeref(r.HourlyContract))
}

func goldDeref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

// report logs a roll state the first time it is seen (goldRollLogKey).
func (m *goldRollMemo) report(r GoldRoll, dayT, hourT int64, detail string) {
	key := goldRollLogKey(r, dayT, hourT)
	m.logMu.Lock()
	if key == m.lastLog {
		m.logMu.Unlock()
		return
	}
	m.lastLog = key
	m.logMu.Unlock()

	bars := "1d bar " + goldRollStamp(dayT, "2006-01-02") + ", 1h bar " + goldRollStamp(hourT, "2006-01-02 15:04")
	switch r.State {
	case goldRollNone:
		m.logf("[demobot] gold roll: none, both bars on %s (%s)", goldDeref(r.DailyContract), bars)
	case goldRollWindow:
		m.logf("[demobot] gold roll: window %s → %s (%s), 1d bar on %s, 1h bar on %s (%s)",
			goldDeref(r.CurrentContract), goldDeref(r.NextContract), goldDeref(r.Reason), goldDeref(r.DailyContract), goldDeref(r.HourlyContract), bars)
	default:
		why := goldDeref(r.Reason)
		if detail != "" {
			why += ": " + detail
		}
		m.logf("[demobot] gold roll: unknown (%s) (%s)", why, bars)
	}
}

func goldRollStamp(ts int64, layout string) string {
	if ts == 0 {
		return "none"
	}
	return time.Unix(ts, 0).UTC().Format(layout) + " UTC"
}

// goldRollOf establishes the roll state for the card's two bars: the last
// closed daily bar and, with hasPx, the last closed hourly bar.
func (a *Agents) goldRollOf(ctx context.Context, daily []types.OHLCVCandle, hour types.OHLCVCandle, hasPx bool) GoldRoll {
	var dayT, hourT int64
	if len(daily) > 0 {
		dayT = daily[len(daily)-1].Time
	}
	if hasPx {
		hourT = hour.Time
	}
	roll, detail := a.goldRollCheck(ctx, daily, hour, hasPx)
	a.goldRoll.report(roll, dayT, hourT, detail)
	return roll
}

func (a *Agents) goldRollCheck(ctx context.Context, daily []types.OHLCVCandle, hour types.OHLCVCandle, hasPx bool) (GoldRoll, string) {
	if !hasPx {
		return goldRollUnknownOf(goldRollNoPrice), ""
	}
	if len(daily) == 0 {
		return goldRollUnknownOf(goldRollNoDailyBars), ""
	}
	day := daily[len(daily)-1]
	dayAt := time.Unix(day.Time, 0).UTC()
	cands := goldCandidates(dayAt)
	remember := goldPrevExpired(dayAt)
	now := a.clock()
	match := func(bar types.OHLCVCandle, interval string) goldRollSide {
		if !goldOnBarGrid(bar.Time) {
			return goldRollSide{reason: goldRollQuoteRow}
		}
		key := fmt.Sprintf("%d|%s|%s|%s", bar.Time, cands[0], cands[1], cands[2])
		return a.goldRoll.side(interval, key, now, func() goldRollSide {
			rctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), goldRollFetchTimeout)
			defer cancel()
			var got [3]goldRollFetch
			var wg sync.WaitGroup
			for i, code := range cands {
				// An expired previous contract is not asked again while its
				// 404 is remembered: the round sees the same 404 as before.
				if i == goldCandPrev && remember {
					if err := a.goldRoll.goneErr(code, interval, now); err != nil {
						got[i].err = err
						continue
					}
				}
				wg.Add(1)
				go func() {
					defer wg.Done()
					got[i].bars, got[i].err = fetchYahooChart(rctx, goldYahooSymbol(code), interval, goldRollRange)
					if i == goldCandPrev && remember && goldNotServed(got[i].err) {
						a.goldRoll.markGone(code, interval, got[i].err, now)
					}
				}()
			}
			wg.Wait()
			return goldRollSideOf(bar, interval, cands, got)
		})
	}
	// The daily and the hourly checks run side by side: each is its own
	// cached round, and after a daily close both are asked at once.
	var ds, hs goldRollSide
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); ds = match(day, "1d") }()
	go func() { defer wg.Done(); hs = match(hour, "1h") }()
	wg.Wait()
	roll := goldRollFrom(ds, hs)
	detail := ""
	if roll.State == goldRollUnknown {
		detail = ds.detail
		if ds.contract != "" {
			detail = hs.detail
		}
	}
	return roll, detail
}
