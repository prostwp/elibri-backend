package demobot

// sr_text.go — everything the S/R card SAYS.
//
// Presentation only. The rules — swing wing (srWing), cluster tolerance
// (srTolPct), top-3-by-strength selection (supportResistance), the test/break
// logic (breakHoldStats), the established threshold (srStrongTouches) — live
// in agents.go / indicators.go and are the single definition of each rule.
// The Gold Agent shares that clusterizer; nothing here is used by gold.
//
// Honesty rules for every line below:
//   - analytics only: no advice, no targets, no probabilities, no "price will";
//   - every price is printed at the instrument's precision (srPx), and every
//     distance is computed from the PRINTED numbers, so a reader can
//     reproduce each percentage from the card itself;
//   - test counts are neutral: breakHoldStats keeps no approach side, so a
//     level's history is "reactions / breaks", never "support held";
//   - the class word counts pivots only, so the verdict carries the level's
//     "reactions / breaks" beside it ("established" is not "it holds");
//   - "nearest shown" — the card picks the three STRONGEST levels per side,
//     so the nearest of them is not necessarily the nearest cluster overall;
//   - the volume flag is stated as the observation it is (lower volume on
//     the last 3 pivots), never as "weakening", and only for a level whose
//     six compared pivots all carry volume (srVolumeComparable);
//   - scenarios are market events tied to the level's current side of price,
//     never the agent's own counters;
//   - every line — verdict, facts, each content block — fits srFactMaxRunes.

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// srFactMaxRunes is the readability budget for one line of S/R text.
const srFactMaxRunes = 110

// srAssetDecimals is the print precision per served instrument, chosen from
// its quote step: BTC whole dollars (0.001% of price), ETH and gold one
// decimal (gold's COMEX tick is 0.10), EURUSD/GBPUSD the pip (4), USDJPY the
// pip (2). Keyed by the machine asset (spec.Key, else spec.Display).
var srAssetDecimals = map[string]int{
	"BTC":    0,
	"ETH":    1,
	"EURUSD": 4,
	"GBPUSD": 4,
	"USDJPY": 2,
	"XAUUSD": 1,
}

// srDecimals is the precision for one card: the table for served
// instruments, otherwise five significant digits of the reference price
// (clamped to 0..7 decimals) so an unlisted asset never prints a level as a
// bare integer the way the old "≥100 → %d" label did (USDJPY 154.938 → 155).
func srDecimals(spec assetSpec, ref float64) int {
	key := spec.Key
	if key == "" {
		key = spec.Display
	}
	if d, ok := srAssetDecimals[strings.ToUpper(key)]; ok {
		return d
	}
	a := math.Abs(ref)
	if a == 0 || math.IsNaN(a) || math.IsInf(a, 0) {
		return 4
	}
	d := 4 - int(math.Floor(math.Log10(a)))
	if d < 0 {
		return 0
	}
	if d > 7 {
		return 7
	}
	return d
}

// srPx prints prices at one fixed precision and hands back the printed value
// as a number, so distances are computed from exactly what the reader sees.
type srPx struct{ dec int }

func (p srPx) str(v float64) string { return strconv.FormatFloat(v, 'f', p.dec, 64) }

func (p srPx) val(v float64) float64 {
	f, err := strconv.ParseFloat(p.str(v), 64)
	if err != nil {
		return v
	}
	return f
}

// ── level class ──────────────────────────────────────────────────────────────

const (
	srClassEstablished = "established"  // ≥ srStrongTouches swing pivots
	srClassCandidate   = "candidate"    // 2 .. srStrongTouches-1 pivots
	srClassSingle      = "single swing" // one pivot
)

func srClassOf(touches int) string {
	switch {
	case touches >= srStrongTouches:
		return srClassEstablished
	case touches <= 1:
		return srClassSingle
	default:
		return srClassCandidate
	}
}

// srClassKey is the JSON enum form of a class.
func srClassKey(class string) string { return strings.ReplaceAll(class, " ", "_") }

func plural(n int, one, many string) string {
	if n == 1 {
		return "1 " + one
	}
	return fmt.Sprintf("%d %s", n, many)
}

// classPivots: "established, 7 pivots" · "single swing, 1 pivot".
func classPivots(l SRLevel) string {
	return srClassOf(l.Touches) + ", " + plural(l.Touches, "pivot", "pivots")
}

// testsShort: "5 reactions / 1 break" — resolved tests only (unresolved ones
// are dropped by breakHoldStats); "no resolved tests" when there are none.
func testsShort(l SRLevel) string {
	if l.Holds+l.Breaks == 0 {
		return "no resolved tests"
	}
	return plural(l.Holds, "reaction", "reactions") + " / " + plural(l.Breaks, "break", "breaks")
}

// testsLong: "5 reactions / 1 break in 6 resolved tests".
func testsLong(l SRLevel) string {
	n := l.Holds + l.Breaks
	if n == 0 {
		return "no resolved tests"
	}
	return testsShort(l) + " in " + plural(n, "resolved test", "resolved tests")
}

func touchDate(t time.Time) string { return t.UTC().Format("Jan 2") }

// ── the view ─────────────────────────────────────────────────────────────────

// srView is one S/R read plus what the card needs to word it.
type srView struct {
	tf   string
	bars int
	last float64
	px   srPx
	// sup / res: the selected levels in STRENGTH order (supportResistance —
	// the JSON levels order). supOrd / resOrd: the same levels nearest-first,
	// as indices into sup / res — the text order. Both ranks served in JSON
	// come from these two slices, so text and JSON cannot drift apart.
	sup, res       []SRLevel
	supOrd, resOrd []int
	// fullVol: level means (SRLevel.Raw) whose first 3 and last 3 pivots all
	// carry non-zero volume — the only levels the volume line may name.
	fullVol map[float64]bool
}

// srVolumeComparable reads, per clustered level (keyed by its mean, exactly
// as srLevelsOf computes it), whether all six pivots srWeakening compares —
// the first 3 and the last 3 — carry non-zero volume. Read-only: it re-runs
// the same swing and cluster functions on the same candles and changes no
// rule result. The rule's own volume gate needs volume on only half the
// pivots, so a zero (Yahoo: no volume data) can sit inside the compared six
// and be averaged as a real zero; the card states the observation only when
// none does. Levels under 3 pivots are never flagged and are left out.
func srVolumeComparable(candles []types.OHLCVCandle, wing int, tolPct float64) map[float64]bool {
	highs, lows := highsLowsOf(candles)
	sh, sl := swingPointsIdx(highs, lows, wing)
	out := make(map[float64]bool)
	for _, cl := range clusterSwingPoints(append(sh, sl...), tolPct) {
		n := len(cl.idxs)
		if n < 3 {
			continue
		}
		ok := true
		for _, i := range append(append([]int(nil), cl.idxs[:3]...), cl.idxs[n-3:]...) {
			if candles[i].Volume <= 0 {
				ok = false
				break
			}
		}
		out[cl.mean] = ok
	}
	return out
}

// nearestOrder returns indices of levels sorted nearest-to-price first
// (stable, so equal distances keep strength order).
func nearestOrder(levels []SRLevel, last float64) []int {
	idx := make([]int, len(levels))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		return math.Abs(levels[idx[a]].Raw-last) < math.Abs(levels[idx[b]].Raw-last)
	})
	return idx
}

func newSRView(spec assetSpec, sup, res []SRLevel, last float64, bars int, fullVol map[float64]bool) srView {
	return srView{
		tf: spec.Interval, bars: bars, last: last,
		px:  srPx{srDecimals(spec, last)},
		sup: sup, res: res,
		supOrd: nearestOrder(sup, last), resOrd: nearestOrder(res, last),
		fullVol: fullVol,
	}
}

// srSide names a side of price: the card's own split (supportResistance) —
// a level is support when its mean is below the last close, resistance above.
type srSide struct {
	noun   string // "support" | "resistance"
	title  string // "Support" | "Resistance"
	toward string // where the level sits against price: "below" | "above"
	away   string // the other side: "above" | "below"
	short  string // "S" | "R"
}

var (
	sideSupport    = srSide{"support", "Support", "below", "above", "S"}
	sideResistance = srSide{"resistance", "Resistance", "above", "below", "R"}
)

// nearest is the shown level closest to price (raw distance, the split's own
// measure; support wins an exact tie). ok=false when no level is shown.
func (v srView) nearest() (SRLevel, srSide, bool) {
	var best SRLevel
	var side srSide
	found := false
	bestDist := math.Inf(1)
	if len(v.sup) > 0 {
		l := v.sup[v.supOrd[0]]
		best, side, found, bestDist = l, sideSupport, true, math.Abs(v.last-l.Raw)
	}
	if len(v.res) > 0 {
		l := v.res[v.resOrd[0]]
		if d := math.Abs(v.last - l.Raw); d < bestDist {
			best, side, found = l, sideResistance, true
		}
	}
	return best, side, found
}

// dist is the printed distance from price to a level: "0.6%", "<0.1%", or ""
// when both print as the same number (the reader sees them as equal).
func (v srView) dist(l SRLevel) string {
	p, q := v.px.val(v.last), v.px.val(l.Raw)
	if p == q {
		return ""
	}
	return pctAway(p, q)
}

// srTestsLabel is the level's resolved-test history as the verdict words it:
// "tests: 1 reaction / 4 breaks" (the level line's own counts, labeled so they
// read without the method), or "no resolved tests".
func srTestsLabel(l SRLevel) string {
	if l.Holds+l.Breaks == 0 {
		return testsShort(l)
	}
	return "tests: " + testsShort(l)
}

// headline — where price is against the nearest shown level, how confirmed
// that level is and how its tests resolved. Fits srFactMaxRunes.
func (v srView) headline() string { h, _ := v.headlines(); return h }

// headlines returns the verdict and blocks.what_happened, worded alike. The
// class word counts pivots only, so it never stands without the level's test
// history ("established" is not "it holds"). One form is chosen for both:
// the first whose verdict AND "On <tf>: …." sentence fit srFactMaxRunes.
// For width the pivot count goes first, then the class word (both stay in
// the level's fact line); the test history goes last, and only at widths no
// served asset reaches, where the level stands alone.
func (v srView) headlines() (verdict, what string) {
	l, side, _ := v.nearest()
	px, lv := v.px.str(v.last), v.px.str(l.Raw)
	at := fmt.Sprintf("Price %s — at", px)
	if d := v.dist(l); d != "" {
		at = fmt.Sprintf("Price %s — %s %s", px, d, side.away)
	}
	base := fmt.Sprintf("%s nearest shown %s %s", at, side.noun, lv)
	sentence := func(h string) string { return "On " + candleWord(v.tf) + ": " + lowerFirst(h) + "." }
	for _, detail := range []string{
		classPivots(l) + "; " + srTestsLabel(l),
		srClassOf(l.Touches) + "; " + srTestsLabel(l),
		srTestsLabel(l),
	} {
		h := base + " (" + detail + ")"
		w := sentence(h)
		if utf8.RuneCountInString(h) <= srFactMaxRunes && utf8.RuneCountInString(w) <= srFactMaxRunes {
			return h, w
		}
	}
	return base, sentence(base)
}

// nearestSidesLine — the signed distance from price to the nearest shown
// level on each side, from the printed numbers (the level lines' own
// arithmetic); a side with no shown level says so. Only the two numbers side
// by side: no judgement of near or far.
func (v srView) nearestSidesLine() string {
	return srNearestSidesPrefix + v.nearestSides()
}

// srNearestSidesPrefix opens the nearest-sides fact. The /showcase/example
// data block and fallback explanation skip that line (srShowcaseSkip): they
// quote the level lines, as before the line existed.
const srNearestSidesPrefix = "Nearest shown levels: "

func (v srView) nearestSides() string {
	part := func(side srSide, levels []SRLevel, ord []int) string {
		if len(levels) == 0 {
			return "no " + side.noun + " shown " + side.toward + " price"
		}
		l := levels[ord[0]]
		if v.dist(l) == "" {
			return side.noun + " at price"
		}
		return side.noun + " " + signedPct(v.px.val(v.last), v.px.val(l.Raw))
	}
	return part(sideResistance, v.res, v.resOrd) + " · " + part(sideSupport, v.sup, v.supOrd)
}

// levelLine — one shown level: price, signed distance, class, tests, last touch.
func (v srView) levelLine(side srSide, l SRLevel) string {
	d := signedPct(v.px.val(v.last), v.px.val(l.Raw))
	if v.dist(l) == "" {
		d = "at price"
	}
	s := fmt.Sprintf("%s %s (%s) · %s · %s", side.title, v.px.str(l.Raw), d, classPivots(l), testsShort(l))
	if !l.LastTouch.IsZero() {
		s += " · last touch " + touchDate(l.LastTouch)
	}
	return s
}

// emptySideLine — a side with no clustered level (live EURUSD/GBPUSD at the
// window low): said plainly, with the window it was searched in.
func (v srView) emptySideLine(side srSide) string {
	return fmt.Sprintf("No clustered %s %s price in this %d-candle %s window", side.noun, side.toward, v.bars, candleWord(v.tf))
}

// volumeLine — the B4 volume flag as the observation it is, for shown levels
// whose six compared pivots all carry volume (fullVol). "" when none does;
// the JSON weakening field is untouched either way.
func (v srView) volumeLine() string {
	var lv []string
	for _, i := range v.resOrd {
		if l := v.res[i]; l.Weakening && v.fullVol[l.Raw] {
			lv = append(lv, v.px.str(l.Raw))
		}
	}
	for _, i := range v.supOrd {
		if l := v.sup[i]; l.Weakening && v.fullVol[l.Raw] {
			lv = append(lv, v.px.str(l.Raw))
		}
	}
	if len(lv) == 0 {
		return ""
	}
	// Prefix ≤46 runes: six flagged levels at the widest printed number
	// (9 runes) plus separators take 64, and the line must fit 110.
	return "Last 3 pivots on lower volume than first 3: " + strings.Join(lv, ", ")
}

// methodLine — the window and the closed-candle rule, in one line; the full
// method is the how-it-works text.
func (v srView) methodLine() string {
	return fmt.Sprintf("Window: %d closed %s candles · test = a close within 0.25 ATR of a level, resolved within 3 candles",
		v.bars, candleWord(v.tf))
}

// facts is the card body: resistances then supports, each nearest first,
// then the volume observation, the nearest shown distance per side and the
// window/closed-candle line.
func (v srView) facts() []string {
	var out []string
	if len(v.res) == 0 {
		out = append(out, v.emptySideLine(sideResistance))
	}
	for _, i := range v.resOrd {
		out = append(out, v.levelLine(sideResistance, v.res[i]))
	}
	if len(v.sup) == 0 {
		out = append(out, v.emptySideLine(sideSupport))
	}
	for _, i := range v.supOrd {
		out = append(out, v.levelLine(sideSupport, v.sup[i]))
	}
	if vl := v.volumeLine(); vl != "" {
		out = append(out, vl)
	}
	return append(out, v.nearestSidesLine(), v.methodLine())
}

// short is the digest one-liner.
func (v srView) short() string {
	switch {
	case len(v.res) > 0 && len(v.sup) > 0:
		return fmt.Sprintf("nearest shown R %s / S %s", v.px.str(v.res[v.resOrd[0]].Raw), v.px.str(v.sup[v.supOrd[0]].Raw))
	case len(v.res) > 0:
		return fmt.Sprintf("nearest shown R %s · none below", v.px.str(v.res[v.resOrd[0]].Raw))
	default:
		return fmt.Sprintf("nearest shown S %s · none above", v.px.str(v.sup[v.supOrd[0]].Raw))
	}
}

// blocks is the content-ready form of the same card (additive JSON field,
// same ContentBlocks contract as trend). All of it is about the nearest shown
// level; scenarios are the two market events that can follow a test of it,
// worded from the side of price the level sits on now.
func (v srView) blocks() *ContentBlocks {
	l, side, _ := v.nearest()
	tf := candleWord(v.tf)
	lv := v.px.str(l.Raw)
	_, what := v.headlines()
	// "pivots", the card's word for swing pivots throughout (class lines, the
	// how-it-works text): "swing pivots" put the worst case (9999 pivots,
	// 19998 resolved tests, 8-rune price) at 111 runes.
	why := fmt.Sprintf("%s = mean of %s · %s", lv, plural(l.Touches, "pivot", "pivots"), testsLong(l))
	if !l.LastTouch.IsZero() {
		why += " · last touch " + touchDate(l.LastTouch)
	}
	// Price sits on side.away of the level, so the next test comes from there:
	// a close back out on that side is a reaction, beyond the far edge a break.
	// A close beyond the level moves it to the other side of price, where the
	// card's own split (supportResistance) no longer calls it this side.
	inv := fmt.Sprintf("A closed %s candle %s %s puts it %s price: it no longer reads as %s",
		tf, side.toward, lv, side.away, side.noun)
	return &ContentBlocks{
		WhatHappened: what,
		WhyLevel:     why,
		// limitations (additive 2026-09-16): the card's own window/method line,
		// verbatim — it is what the levels are and are not measured over, and
		// it already closes facts[].
		Limitations: v.methodLine(),
		// "its band" = ±0.25 ATR around the level (the window line and the
		// how-it-works text define it). Both events are worded from the level's
		// CURRENT side of price (side comes from the card's own split), so they
		// stay true after a cross: the next card re-derives the side. A close
		// back out on price's side keeps the level where it is; a close past
		// the far edge of the band puts it on the other side of price.
		Scenarios: []string{
			fmt.Sprintf("If a %s close tests %s and a close within 3 candles exits its band %s, the level holds as %s",
				tf, lv, side.away, side.noun),
			fmt.Sprintf("If a %s candle closes %s %s's band, the level is broken and moves %s price",
				tf, side.toward, lv, side.away),
		},
		Invalidates: &inv,
		Regime:      v.regime(l, side),
	}
}

// regime — local level context only: which sides carry shown levels and how
// far the nearest one is. Not macro, not trend.
func (v srView) regime(l SRLevel, side srSide) string {
	d := v.dist(l)
	if d == "" {
		d = "at price"
	} else {
		d += " away"
	}
	tf := candleWord(v.tf)
	switch {
	case len(v.sup) > 0 && len(v.res) > 0:
		return fmt.Sprintf("Levels on both sides · nearest shown: %s, %s · %s", side.noun, d, tf)
	case len(v.res) > 0:
		return fmt.Sprintf("Resistance only, none below price · nearest shown %s · %s", d, tf)
	default:
		return fmt.Sprintf("Support only, none above price · nearest shown %s · %s", d, tf)
	}
}

// points converts one side to the machine form: strength order (the array
// order), with strength_rank = array position and display_rank = the line's
// position in the nearest-first text, both 1-based.
func (v srView) points(levels []SRLevel, ord []int) []SRPoint {
	display := make([]int, len(levels))
	for k, i := range ord {
		display[i] = k + 1
	}
	pts := srPoints(levels, v.px)
	for i := range pts {
		pts[i].DisplayRank = display[i]
		pts[i].StrengthRank = i + 1
	}
	return pts
}

// srPoints converts clustered levels to the machine-readable envelope form:
// raw cluster means at full precision plus the printed label. Always non-nil
// so an empty side serializes as []. Ranks are set by srView.points.
func srPoints(levels []SRLevel, px srPx) []SRPoint {
	pts := make([]SRPoint, 0, len(levels))
	for _, l := range levels {
		p := SRPoint{
			Level:     l.Raw,
			Label:     px.str(l.Raw),
			Class:     srClassKey(srClassOf(l.Touches)),
			Touches:   l.Touches,
			Strength:  l.Strength,
			Weakening: l.Weakening,
			Breaks:    l.Breaks,
			Holds:     l.Holds,
		}
		if !l.LastTouch.IsZero() {
			p.LastTouch = l.LastTouch.Format(time.RFC3339)
		}
		pts = append(pts, p)
	}
	return pts
}

// srNoLevelsCard is the "no significant levels" finding: swings exist, but no
// cluster sits strictly on either side of the last close. An ok reading with
// empty arrays and no blocks (there is no level to word them about).
func srNoLevelsCard(spec assetSpec, swings int, last float64, bars int, dataTime time.Time) Card {
	v := newSRView(spec, nil, nil, last, bars, nil)
	return Card{
		Emoji:      emojiNeutral,
		Agent:      "S/R Agent",
		ShortName:  "S/R",
		Asset:      spec.Display,
		AssetKey:   spec.Key,
		Command:    keySR,
		HowItWorks: howTexts[keySR],
		DataTime:   dataTime,
		Verdict:    "No significant levels detected in the window",
		Short:      "no significant levels",
		Facts: []string{
			fmt.Sprintf("Swing points exist (%d), but no cluster sits clear of the last price %s", swings, v.px.str(last)),
			v.methodLine(),
		},
		Levels: SRLevels{Supports: srPoints(nil, srPx{}), Resistances: srPoints(nil, srPx{})},
	}
}

// srCardFrom is the pure half of SRCard for a read with at least one shown
// level: the selected levels (strength order), the last close, the window,
// and fullVol from srVolumeComparable (nil: the volume line names no level).
func srCardFrom(spec assetSpec, sup, res []SRLevel, last float64, bars int, dataTime time.Time, fullVol map[float64]bool) Card {
	v := newSRView(spec, sup, res, last, bars, fullVol)
	return Card{
		Emoji:      emojiNeutral,
		Agent:      "S/R Agent",
		ShortName:  "S/R",
		Asset:      spec.Display,
		AssetKey:   spec.Key,
		Command:    keySR,
		HowItWorks: howTexts[keySR],
		DataTime:   dataTime,
		Verdict:    v.headline(),
		Short:      v.short(),
		Facts:      v.facts(),
		Levels:     SRLevels{Supports: v.points(sup, v.supOrd), Resistances: v.points(res, v.resOrd)},
		Blocks:     v.blocks(),
	}
}
