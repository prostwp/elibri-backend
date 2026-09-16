package demobot

// macro_text.go — everything the Macro cards SAY: the global regime card and
// the ?asset=btc|gold backdrops.
//
// Presentation, freshness and honesty only. The rules — lamp thresholds (with
// their asymmetry), weights, the composite, the 35/65 bands, the voting
// minimum — live in internal/macro/compute.go and are read here through its
// exported Rule* constants and LampWeight; the gold model's weights and lamp
// mapping live in macroviews.go. Nothing here re-decides a lamp: the backend's
// per-lamp status is authoritative, and every sentence words the rule that
// produced it.
//
// Honesty rules for every line below:
//   - analytics only: no advice, no targets, no probabilities, no forecast for
//     BTC or gold — the regime is a formal state of five tradfin prices;
//   - no causality: a lamp is a positive / neutral / negative contribution in
//     this model; nothing "favors crypto", "pulls money" or "unwinds a haven
//     bid" — the pipeline measures no flows and no later returns;
//   - the score is a "rule score" (gold: "gold score"), never a strength or a
//     confidence;
//   - contributions use the rule's own arithmetic, and are printed only when
//     they reproduce the backend's score (modelRead.pts);
//   - a printed number is never on the wrong side of the threshold it is
//     compared with (sessionPctShown, vixShown);
//   - VIX is read by LEVEL: its session change is never printed beside its
//     status;
//   - dates are the lamps' own session stamps (as_of), never the build time,
//     and "24h" is never said: the change is Close − session Open;
//   - a closed market is only ever a "scheduled tradfin weekend": the week
//     window (macro.TradfinWindowOpen) knows no holidays, so a holiday shows
//     as old dates on the data line, never as a claim;
//   - Crypto Fear & Greed is a separate index, never part of a score, and is
//     marked stale beyond fngStaleAfter;
//   - every visible line fits macroFactMaxRunes.

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/prostwp/elibri-backend/internal/macro"
)

// macroFactMaxRunes is the readability budget for one line of macro text.
const macroFactMaxRunes = 110

// fngStaleAfter: alternative.me publishes one Fear & Greed value per UTC day,
// stamped at the start of that day, so a current value is up to ~24h old;
// 36h leaves 12h for the source's publishing lag. Anything older means at
// least one daily update was missed, and the card says the value is stale.
const fngStaleAfter = 36 * time.Hour

// Contribution words — the only direction vocabulary of the macro cards.
const (
	contribPositive = "positive"
	contribNeutral  = "neutral"
	contribNegative = "negative"
)

// macroWeekendBanner is the first fact while the clock-based tradfin week is
// closed. "Scheduled": the window is the regular Sun 22:00 → Fri 21:00 UTC
// week, so it may be wrong on holidays and never says "market closed".
const macroWeekendBanner = "Scheduled tradfin weekend: session changes are from each lamp's last session"

var macroShortNames = map[string]string{
	"dxy": "DXY", "rates": "US 10Y", "vix": "VIX", "spx": "S&P 500", "gold": "Gold",
}

// lampShort is the lamp's name on the card.
func lampShort(l MacroLamp) string {
	if s, ok := macroShortNames[l.Key]; ok {
		return s
	}
	return truncate(l.Label, 16)
}

// lampLive: the payload carries a value for the lamp (per-lamp ok flag with
// the Value fallback for older payloads — same test as effectiveMacroRegime).
func lampLive(l MacroLamp) bool { return l.OK || l.Value != nil }

// goldInstrument names what the gold lamp's number is, by provider: the
// Yahoo fallback serves COMEX futures, stooq the spot rate — a different
// instrument, not only a different vendor.
func goldInstrument(source string) string {
	switch source {
	case "yahoo":
		return "GC=F futures"
	case "stooq":
		return "XAUUSD spot"
	}
	return ""
}

func fmtNum(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }

func upperFirst(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

func regimeWord(regime string) string {
	switch regime {
	case "risk_on":
		return "risk-on"
	case "risk_off":
		return "risk-off"
	case "mixed":
		return "mixed"
	}
	return "unknown"
}

// ── the rule, worded ─────────────────────────────────────────────────────────

// neutralBand is the session-change interval (in %) the risk rule reads as
// neutral for a directional lamp — below lo is one reading, above hi the
// other (compute.go LampStatus). ok=false for the level-read VIX.
func neutralBand(key string) (lo, hi float64, ok bool) {
	switch key {
	case "dxy", "rates", "gold":
		return 0, macro.RuleStrongMovePct, true
	case "spx":
		return -macro.RuleStrongMovePct, 0, true
	}
	return 0, 0, false
}

// sideSafe rounds v to two decimals, then moves the printed value back to the
// side of [lo, hi] the true value is on: a value below lo never prints at lo,
// above hi never at hi, inside never outside. Integer-cent thresholds make one
// cent always enough.
func sideSafe(v, lo, hi float64) float64 {
	p := math.Round(v*100) / 100
	switch {
	case v < lo && p >= lo:
		p = lo - 0.01
	case v > hi && p <= hi:
		p = hi + 0.01
	case v >= lo && v <= hi:
		p = math.Min(math.Max(p, lo), hi)
	}
	if p == 0 {
		p = 0 // never "-0.00"
	}
	return p
}

// sessionPctShown prints a session change (Close − session Open, %) on the
// same side of the rule's thresholds as the true value. Bounded to eight
// characters (">+999%" beyond ±999.99).
func sessionPctShown(key string, d float64) string {
	p := math.Round(d*100) / 100
	if lo, hi, ok := neutralBand(key); ok {
		p = sideSafe(d, lo, hi)
	}
	switch {
	case math.IsNaN(p):
		return "n/a"
	case p >= 1000:
		return ">+999%"
	case p <= -1000:
		return "<-999%"
	}
	return fmt.Sprintf("%+.2f%%", p)
}

// vixShown prints the VIX level on the same side of 18 / 25 as the true value.
func vixShown(v float64) string {
	p := sideSafe(v, macro.RuleVIXCalmBelow, macro.RuleVIXFearAbove)
	switch {
	case math.IsNaN(p):
		return "n/a"
	case p >= 1000:
		return ">999"
	case p <= -1000:
		return "<-999"
	}
	return fmt.Sprintf("%.2f", p)
}

// lampValueShown prints a non-VIX lamp level (trimFloat precision), bounded.
func lampValueShown(v float64) string {
	switch {
	case math.IsNaN(v):
		return "n/a"
	case v >= 1e7:
		return ">9999999"
	case v <= -1e7:
		return "<-9999999"
	}
	return trimFloat(v)
}

// lampCondition words the rule condition behind a lamp's status, from the
// rule constants: "<18", "fell", "rose >0.5%", "0 to +0.5%". "" without a
// status. The condition is about the lamp, so both models share it.
func lampCondition(key, status string) string {
	strong := fmtNum(macro.RuleStrongMovePct)
	calm, fear := fmtNum(macro.RuleVIXCalmBelow), fmtNum(macro.RuleVIXFearAbove)
	switch key {
	case "vix":
		switch status {
		case "tailwind":
			return "<" + calm
		case "headwind":
			return ">" + fear
		case "neutral":
			return calm + "-" + fear
		}
	case "dxy", "rates", "gold":
		switch status {
		case "tailwind":
			return "fell"
		case "headwind":
			return "rose >" + strong + "%"
		case "neutral":
			return "0 to +" + strong + "%"
		}
	case "spx":
		switch status {
		case "tailwind":
			return "rose"
		case "headwind":
			return "fell >" + strong + "%"
		case "neutral":
			return "-" + strong + "% to 0"
		}
	}
	return ""
}

// lampCause is the compact form used in lists: "VIX 17.10 (<18)",
// "S&P 500 +0.12% (rose)". VIX never shows its session change: the rule
// reads its level.
func lampCause(l MacroLamp) string {
	name := lampShort(l)
	if l.Value == nil {
		return name + " (no data)"
	}
	var s string
	switch {
	case l.Key == "vix":
		s = name + " " + vixShown(*l.Value)
	case l.DeltaPct == nil:
		return name + " (no session change)"
	default:
		s = name + " " + sessionPctShown(l.Key, *l.DeltaPct)
	}
	if c := lampCondition(l.Key, l.Status); c != "" {
		s += " (" + c + ")"
	}
	return s
}

// lampFull is the per-lamp line of the asset views, without the outcome:
// "US 10Y 4.9610, session -0.37% (fell)", "VIX 17.10 (<18)",
// "US 10Y — no data (last seen Sep 11)".
func lampFull(l MacroLamp) string {
	name := lampShort(l)
	if l.Key == "gold" {
		if in := goldInstrument(l.Source); in != "" {
			name += " (" + in + ")"
		}
	}
	if l.Value == nil {
		s := name + " — no data"
		if d, ok := lampStamp(l.AsOf); ok {
			s += " (last seen " + dayShown(d) + ")"
		}
		return s
	}
	if l.Key == "vix" {
		return lampCause(l)
	}
	s := name + " " + lampValueShown(*l.Value)
	if l.DeltaPct == nil {
		return s + " (no session change)"
	}
	s += ", session " + sessionPctShown(l.Key, *l.DeltaPct)
	if c := lampCondition(l.Key, l.Status); c != "" {
		s += " (" + c + ")"
	}
	return s
}

// lampSelfLine is the gold lamp on the gold view: a fact with its instrument,
// no rule condition — it is the asset itself, not an input.
func lampSelfLine(l MacroLamp) string {
	name := lampShort(l)
	if in := goldInstrument(l.Source); in != "" {
		name += " (" + in + ")"
	}
	s := name + " " + lampValueShown(*l.Value)
	if l.DeltaPct != nil {
		s += ", session " + sessionPctShown(l.Key, *l.DeltaPct)
	}
	return s + " — the asset itself, not an input"
}

// MacroLampRule is the machine form of one lamp's rule in one model.
type MacroLampRule struct {
	Input        string `json:"input"`                   // "level" | "session_change_pct"
	PositiveWhen string `json:"positive_when,omitempty"` // e.g. "< 0", "< 18"
	NegativeWhen string `json:"negative_when,omitempty"` // e.g. "> 0.5", "> 25"
}

// riskRule is the risk-appetite rule of a lamp, from the rule constants.
func riskRule(key string) MacroLampRule {
	strong := fmtNum(macro.RuleStrongMovePct)
	switch key {
	case "vix":
		return MacroLampRule{Input: "level", PositiveWhen: "< " + fmtNum(macro.RuleVIXCalmBelow), NegativeWhen: "> " + fmtNum(macro.RuleVIXFearAbove)}
	case "dxy", "rates", "gold":
		return MacroLampRule{Input: "session_change_pct", PositiveWhen: "< 0", NegativeWhen: "> " + strong}
	case "spx":
		return MacroLampRule{Input: "session_change_pct", PositiveWhen: "> 0", NegativeWhen: "< -" + strong}
	}
	return MacroLampRule{}
}

// goldRule re-reads the risk rule through the gold mapping (goldLampView):
// same-sign lamps keep it, inverted lamps swap it, non-voters have none.
func goldRule(key string) MacroLampRule {
	r := riskRule(key)
	switch goldLampView(key, "tailwind") {
	case goldSupport:
		return r
	case goldPressure:
		r.PositiveWhen, r.NegativeWhen = r.NegativeWhen, r.PositiveWhen
		return r
	}
	return MacroLampRule{Input: r.Input}
}

// ── dates ────────────────────────────────────────────────────────────────────

func lampStamp(s string) (time.Time, bool) {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return time.Time{}, false
	}
	return t.UTC(), true
}

func dayOf(t time.Time) time.Time {
	t = t.UTC()
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
}

func dayShown(t time.Time) string { return t.UTC().Format("Jan 2") }

// ── the models ───────────────────────────────────────────────────────────────

// macroModel is one scoring rule over the lamps: the risk-appetite composite
// (internal/macro Composite) or the experimental gold backdrop
// (goldViewScore). Both give a voting lamp +weight (positive), +weight/2
// (neutral) or 0 (negative), renormalised over the voting lamps.
type macroModel struct {
	id           string
	scoreName    string // "rule score" | "gold score"
	hiWord       string // reading above high
	loWord       string // reading below low
	experimental bool
	weight       func(key string) float64
	word         func(l MacroLamp) string // contribution word; "" = not voting
	minVoters    int
	low, high    int
	// total is the rule's own unrounded formula. The two divide in a
	// different order (see goldViewScore); the breakdown must round exactly
	// the way the rule does.
	total func(scoreSum, weightSum float64) float64
}

func riskWord(status string) string {
	switch status {
	case "tailwind":
		return contribPositive
	case "headwind":
		return contribNegative
	case "neutral":
		return contribNeutral
	}
	return ""
}

func goldWord(view string) string {
	switch view {
	case goldSupport:
		return contribPositive
	case goldPressure:
		return contribNegative
	case goldNeutral:
		return contribNeutral
	}
	return ""
}

var riskModel = macroModel{
	id:        "risk_appetite",
	scoreName: "rule score",
	hiWord:    "risk-on",
	loWord:    "risk-off",
	weight:    macro.LampWeight,
	word:      func(l MacroLamp) string { return riskWord(l.Status) },
	minVoters: macro.RuleMinActiveLamps,
	low:       macro.RuleRiskOffBelow,
	high:      macro.RuleRiskOnAbove,
	total:     func(s, w float64) float64 { return s / w * 100 },
}

var goldModel = macroModel{
	id:           "gold_backdrop",
	scoreName:    "gold score",
	hiWord:       contribPositive,
	loWord:       contribNegative,
	experimental: true,
	weight:       goldWeightOf,
	word:         func(l MacroLamp) string { return goldWord(goldLampView(l.Key, l.Status)) },
	minVoters:    goldMinVoters,
	low:          goldPressureBelow,
	high:         goldSupportAbove,
	total:        func(s, w float64) float64 { return s * 100 / w },
}

// lampRead is one lamp under one model.
type lampRead struct {
	l      MacroLamp
	word   string  // positive | neutral | negative | "" (not voting)
	weight float64 // nominal weight in this model (0 = never votes)
	// Renormalised score points, valid when the model scored: points is what
	// the lamp adds (weight, weight/2 or 0 over the voting weights × 100),
	// maxPts its share, vsNeutral = points − maxPts/2 (the signed
	// contribution: score = 50 + Σ vsNeutral).
	points, maxPts, vsNeutral float64
}

// modelRead is the whole model over one payload.
type modelRead struct {
	m         macroModel
	lamps     []lampRead
	voters    int // lamps with a contribution word
	inModel   int // lamps with a weight in this model
	weightSum float64
	exact     float64 // unrounded score (valid when score != nil)
	score     *int    // the rule's rounded score; nil under minVoters
	// pts: the per-lamp numbers may be printed — the model scored and, for
	// the risk model, reproduces the backend's composite exactly.
	pts bool
}

func (m macroModel) read(lamps []MacroLamp) modelRead {
	r := modelRead{m: m}
	var scoreSum float64
	for _, l := range lamps {
		lr := lampRead{l: l, weight: m.weight(l.Key)}
		if lr.weight > 0 {
			r.inModel++
			lr.word = m.word(l)
		}
		if lr.word != "" {
			r.voters++
			r.weightSum += lr.weight
			switch lr.word {
			case contribPositive:
				scoreSum += lr.weight
			case contribNeutral:
				scoreSum += lr.weight / 2
			}
		}
		r.lamps = append(r.lamps, lr)
	}
	if r.voters < m.minVoters || r.weightSum == 0 {
		return r
	}
	r.exact = m.total(scoreSum, r.weightSum)
	v := clampInt(int(math.Round(r.exact)), 0, 100)
	r.score = &v
	r.pts = true
	for i := range r.lamps {
		lr := &r.lamps[i]
		if lr.word == "" {
			continue
		}
		lr.maxPts = lr.weight * 100 / r.weightSum
		switch lr.word {
		case contribPositive:
			lr.points = lr.maxPts
		case contribNeutral:
			lr.points = lr.maxPts / 2
		}
		lr.vsNeutral = lr.points - lr.maxPts/2
	}
	return r
}

// reading is the model's word for its own score: hiWord above high, loWord
// below low, "mixed" between, "" without a score.
func (r modelRead) reading() string {
	switch {
	case r.score == nil:
		return ""
	case *r.score > r.m.high:
		return r.m.hiWord
	case *r.score < r.m.low:
		return r.m.loWord
	}
	return "mixed"
}

// signedPts prints a signed contribution with one decimal; "0" for neutral.
func signedPts(v float64) string {
	if math.Abs(v) < 0.05 {
		return "0"
	}
	return fmt.Sprintf("%+.1f", v)
}

// printedTenths is a contribution exactly as signedPts prints it, in integer
// tenths (parsed back from the printed text, so "%+.1f" tie rounding can never
// disagree with the check).
func printedTenths(v float64) int {
	f, err := strconv.ParseFloat(signedPts(v), 64)
	if err != nil {
		return 0
	}
	return int(math.Round(f * 10))
}

// eqSign is "=" only when 50 plus the PRINTED contributions adds up exactly
// to the PRINTED score, else "≈". Deciding from the unrounded score was wrong:
// three positive 25-weight lamps are exactly 100, but print as
// 50 + 16.7 + 16.7 + 16.7 = 100.1. Every voting lamp's contribution is
// printed (neutral ones as 0), so this is the sum a reader can redo.
func (r modelRead) eqSign(printedScore int) string {
	if !r.pts {
		return "≈"
	}
	tenths := 500
	for _, lr := range r.lamps {
		if lr.word != "" {
			tenths += printedTenths(lr.vsNeutral)
		}
	}
	if tenths == printedScore*10 {
		return "="
	}
	return "≈"
}

// outcome is the tail of an asset-view lamp line: "positive, +7.5",
// "negative for gold, -12.5", "not voting".
func (r modelRead) outcome(lr lampRead, forWhat string) string {
	if lr.word == "" {
		return "not voting"
	}
	s := lr.word + forWhat
	if r.pts {
		s += ", " + signedPts(lr.vsNeutral)
	}
	return s
}

// factorLines are the voting lamps by side: every lamp of the side with the
// larger total contribution, then every lamp of the other side, each side
// largest first. No lamp is hidden behind a count: a side that does not fit
// one line continues on the next (listLines).
func (r modelRead) factorLines() []string {
	var pos, neg []lampRead
	for _, lr := range r.lamps {
		switch lr.word {
		case contribPositive:
			pos = append(pos, lr)
		case contribNegative:
			neg = append(neg, lr)
		}
	}
	size := func(lr lampRead) float64 {
		if r.pts {
			return math.Abs(lr.vsNeutral)
		}
		return lr.weight
	}
	var sp, sn float64
	for _, lr := range pos {
		sp += size(lr)
	}
	for _, lr := range neg {
		sn += size(lr)
	}
	byt := func(s []lampRead) {
		sort.SliceStable(s, func(i, j int) bool { return size(s[i]) > size(s[j]) })
	}
	byt(pos)
	byt(neg)
	if sp >= sn {
		return append(r.listLines(contribPositive, pos), r.listLines(contribNegative, neg)...)
	}
	return append(r.listLines(contribNegative, neg), r.listLines(contribPositive, pos)...)
}

// listLines words one side: "Positive for rule score: <lamp> → +12.5 · …".
// The head names the score the contributions are to (the risk model's rule
// score), so a lamp that is negative here is not read as negative for the
// asset it names — the gold model is a separate reading. Lamps are packed
// into lines of at most macroFactMaxRunes; the rest of the side continues on
// "Positive for rule score (cont.): …" lines, never "· N more".
// macroContMark marks a factor line that continues its side.
const macroContMark = " (cont.): "

// macroFactorHead splits a factor line into its head ("Positive for rule
// score"), whether it continues its side, and its lamp parts; ok=false for
// any other line.
func macroFactorHead(f string) (head string, cont bool, parts []string, ok bool) {
	if !strings.HasPrefix(f, upperFirst(contribPositive)+" for ") && !strings.HasPrefix(f, upperFirst(contribNegative)+" for ") {
		return "", false, nil, false
	}
	if i := strings.Index(f, macroContMark); i > 0 {
		return f[:i], true, strings.Split(f[i+len(macroContMark):], " · "), true
	}
	i := strings.Index(f, ": ")
	if i < 0 {
		return "", false, nil, false
	}
	return f[:i], false, strings.Split(f[i+2:], " · "), true
}

// macroShowcaseFacts folds each factor side into its first line for the
// /showcase/example block: the "(cont.)" lines go, and the first line ends
// with "· +N more on the card" for the lamps it no longer shows. When that
// tail does not fit macroFactMaxRunes, the last shown lamp moves into N.
// Every other line, and every side that fit one line, is unchanged.
func macroShowcaseFacts(facts []string) []string {
	out := make([]string, 0, len(facts))
	for i := 0; i < len(facts); i++ {
		head, cont, parts, ok := macroFactorHead(facts[i])
		if !ok || cont {
			out = append(out, facts[i])
			continue
		}
		hidden := 0
		for i+1 < len(facts) {
			h, c, p, ok := macroFactorHead(facts[i+1])
			if !ok || !c || h != head {
				break
			}
			hidden += len(p)
			i++
		}
		if hidden == 0 {
			out = append(out, facts[i])
			continue
		}
		shown := parts
		for {
			line := fmt.Sprintf("%s: %s · +%d more on the card", head, strings.Join(shown, " · "), hidden)
			if utf8.RuneCountInString(line) <= macroFactMaxRunes || len(shown) == 1 {
				out = append(out, line)
				break
			}
			shown = shown[:len(shown)-1]
			hidden++
		}
	}
	return out
}

func (r modelRead) listLines(word string, items []lampRead) []string {
	head := upperFirst(word) + " for " + r.m.scoreName
	if len(items) == 0 {
		return []string{head + ": none in this model"}
	}
	var out []string
	line := ""
	for _, lr := range items {
		p := lampCause(lr.l)
		if r.pts {
			p += " → " + signedPts(lr.vsNeutral)
		}
		switch {
		case line == "":
			line = head + ": " + p
		case utf8.RuneCountInString(line+" · "+p) <= macroFactMaxRunes:
			line += " · " + p
		default:
			out = append(out, line)
			line = head + macroContMark + p
		}
	}
	return append(out, line)
}

// holdsLine — what keeps the reading and what ends it, from the bands and
// the voting minimum. reading is the card's reading word (for the risk model
// the backend's regime, which is authoritative).
func (r modelRead) holdsLine(reading string) string {
	m := r.m
	switch {
	case reading == m.hiWord:
		return fmt.Sprintf("%s holds while the %s stays above %d with at least %d voting lamps",
			upperFirst(m.hiWord), m.scoreName, m.high, m.minVoters)
	case reading == m.loWord:
		return fmt.Sprintf("%s holds while the %s stays below %d with at least %d voting lamps",
			upperFirst(m.loWord), m.scoreName, m.low, m.minVoters)
	case reading == "mixed" && r.score != nil:
		return fmt.Sprintf("Mixed while the %s is %d-%d: above %d reads %s, below %d %s",
			m.scoreName, m.low, m.high, m.high, m.hiWord, m.low, m.loWord)
	}
	return fmt.Sprintf("No %s: %d of %d lamps vote, the rule needs %d", m.scoreName, r.voters, r.inModel, m.minVoters)
}

// breakdownLine shows the score as 50 plus the signed contributions — the
// rule's own arithmetic, so a reader can re-add it. "≈" when the unrounded
// sum is not a whole number. "" when the numbers may not be printed.
func (r modelRead) breakdownLine() string {
	if !r.pts {
		return ""
	}
	var terms, neutral, idle []string
	for _, lr := range r.lamps {
		if lr.weight == 0 {
			continue // not in this model (gold itself in the gold model)
		}
		switch lr.word {
		case "":
			idle = append(idle, lampShort(lr.l))
		case contribNeutral:
			neutral = append(neutral, lampShort(lr.l))
		default:
			terms = append(terms, lampShort(lr.l)+" "+signedPts(lr.vsNeutral))
		}
	}
	s := fmt.Sprintf("%s %d %s 50", upperFirst(r.m.scoreName), *r.score, r.eqSign(*r.score))
	for _, t := range terms {
		s += " + " + t
	}
	if len(neutral) > 0 {
		s += " · neutral: " + strings.Join(neutral, ", ")
	}
	if len(idle) > 0 {
		s += " · rescaled, not voting: " + strings.Join(idle, ", ")
	}
	return s
}

// ── freshness ────────────────────────────────────────────────────────────────

// macroDataLine: how many lamps are live and which session each comes from —
// one line, grouped by date, so mixed freshness is visible at a glance.
func macroDataLine(lamps []MacroLamp) string {
	live := 0
	byDay := map[time.Time][]string{}
	var days []time.Time
	var unknown, missing []string
	for _, l := range lamps {
		if !lampLive(l) {
			missing = append(missing, lampShort(l))
			continue
		}
		live++
		t, ok := lampStamp(l.AsOf)
		if !ok {
			unknown = append(unknown, lampShort(l))
			continue
		}
		d := dayOf(t)
		if _, seen := byDay[d]; !seen {
			days = append(days, d)
		}
		byDay[d] = append(byDay[d], lampShort(l))
	}
	sort.Slice(days, func(i, j int) bool { return days[i].Before(days[j]) })
	parts := []string{fmt.Sprintf("%d of %d lamps live", live, len(lamps))}
	if len(days) == 1 && len(unknown) == 0 {
		parts = append(parts, "session "+dayShown(days[0]))
	} else {
		for _, d := range days {
			parts = append(parts, dayShown(d)+": "+strings.Join(byDay[d], ", "))
		}
	}
	if len(unknown) > 0 {
		parts = append(parts, "date unknown: "+strings.Join(unknown, ", "))
	}
	if len(missing) > 0 {
		parts = append(parts, "no data: "+strings.Join(missing, ", "))
	}
	return "Data: " + strings.Join(parts, " · ")
}

// macroSources counts the providers behind the live lamps.
func macroSources(lamps []MacroLamp) map[string]int {
	out := map[string]int{}
	for _, l := range lamps {
		if lampLive(l) && l.Source != "" {
			out[l.Source]++
		}
	}
	return out
}

// macroSourceNote is the footer note: "lamps: Yahoo" or "lamps: Yahoo 3 ·
// Stooq 2"; "" when the payload names no provider.
func macroSourceNote(lamps []MacroLamp) string {
	src := macroSources(lamps)
	if len(src) == 0 {
		return ""
	}
	names := make([]string, 0, len(src))
	for k := range src {
		names = append(names, k)
	}
	sort.Strings(names)
	switch len(names) {
	case 1:
		return "lamps: " + upperFirst(truncate(names[0], 12))
	case 2:
	default:
		// Only stooq and yahoo exist today; more providers would not fit the
		// footer — the per-provider counts stay in the JSON (freshness.sources).
		return fmt.Sprintf("lamps: %d sources", len(names))
	}
	parts := make([]string, 0, len(names))
	for _, k := range names {
		parts = append(parts, fmt.Sprintf("%s %d", upperFirst(truncate(k, 12)), src[k]))
	}
	return "lamps: " + strings.Join(parts, " · ")
}

// macroDataTime is the card's data stamp: the OLDEST live lamp's as_of (so a
// card never stamps an older lamp as current), else the backend's freshest
// stamp, else the response time.
func macroDataTime(m *MacroResp) time.Time {
	var oldest time.Time
	for _, l := range m.Lamps {
		if !lampLive(l) {
			continue
		}
		if t, ok := lampStamp(l.AsOf); ok && (oldest.IsZero() || t.Before(oldest)) {
			oldest = t
		}
	}
	if !oldest.IsZero() {
		return oldest
	}
	if t, ok := lampStamp(m.TradfinAsOf); ok {
		return t
	}
	return parseWhen(m.CapturedAt)
}

// macroNoValidator — why every macro card (global, ?asset=btc, ?asset=gold)
// carries no HTTP validator. data_as_of / the footer are the OLDEST live lamp
// (honest about the stalest input), but that stamp can stay put while a newer
// lamp, the score or the regime changes. No lamp stamp versions the body
// either: a lamp's as_of is its session stamp (Yahoo stamps the session
// START), so its value moves during the session under the same as_of. And the
// backend's captured_at is its request time at one-second resolution, so two
// different payloads can share it. Any Last-Modified we could send would
// answer a changed card with a false 304 — so none is sent (Card.noValidator,
// set once in macroBaseCard for every macro path, UNKNOWN included).

// ── Fear & Greed ─────────────────────────────────────────────────────────────

// fngAge is the F&G value's reference time (the source's own date, else our
// fetch time) and its age at now.
func fngAge(f *FearGreedCheck, now time.Time) (ref time.Time, age time.Duration, known bool) {
	ref, known = lampStamp(f.AsOf)
	if !known {
		ref, known = lampStamp(f.FetchedAt)
	}
	if !known {
		return time.Time{}, 0, false
	}
	age = now.Sub(ref)
	if age < 0 {
		age = 0
	}
	return ref, age, true
}

func ageShown(d time.Duration) string {
	h := int(d.Hours())
	switch {
	case h < 48:
		return fmt.Sprintf("%dh", h)
	case h/24 > 999:
		return ">999d"
	}
	return fmt.Sprintf("%dd", h/24)
}

// fngLine is the Fear & Greed fact: a separate index, never in scoreName.
// "" when the payload has no live value.
func fngLine(f *FearGreedCheck, now time.Time, scoreName string) string {
	if f == nil || !f.OK {
		return ""
	}
	head := fmt.Sprintf("Crypto Fear & Greed %d (%s)", clampInt(f.Value, 0, 100), truncate(f.Label, 16))
	tail := " · not in the " + scoreName
	ref, age, known := fngAge(f, now)
	switch {
	case !known:
		return head + ", update time unknown" + tail
	case age > fngStaleAfter:
		return fmt.Sprintf("%s: stale, last update %s (%s ago)%s", head, dayShown(ref), ageShown(age), tail)
	}
	return head + ", " + dayShown(ref) + " · separate index, not in the " + scoreName
}

// ── machine readout ──────────────────────────────────────────────────────────

// MacroReadout is the envelope's "macro" object: the numbers behind a macro
// card, for visualisation (docs/demobot-http.md "Macro card"). Additive.
type MacroReadout struct {
	Model          string             `json:"model"`        // "risk_appetite" | "gold_backdrop"
	Experimental   bool               `json:"experimental"` // true for the gold model
	Reading        string             `json:"reading"`      // risk_on|mixed|risk_off (risk) · positive|mixed|negative|no_read (gold)
	RuleScore      *int               `json:"rule_score"`   // the model's rounded score; null without one
	RuleScoreExact *float64           `json:"rule_score_unrounded"`
	Bands          MacroBands         `json:"bands"`
	MinVotingLamps int                `json:"min_voting_lamps"`
	VotingLamps    int                `json:"voting_lamps"`
	LiveLamps      int                `json:"live_lamps"`
	Lamps          []MacroLampReadout `json:"lamps"`
	Freshness      MacroFreshness     `json:"freshness"`
	FearGreed      *MacroFearGreed    `json:"fear_greed"`
	IsForecast     bool               `json:"is_forecast"` // always false
}

// MacroBands: reading is Low's side below Low, High's side above High, mixed
// in between (both edges inclusive in the middle band).
type MacroBands struct {
	Low  int `json:"low"`
	High int `json:"high"`
}

type MacroLampReadout struct {
	Key          string        `json:"key"`
	Label        string        `json:"label"`
	Instrument   string        `json:"instrument,omitempty"` // gold: "GC=F futures" | "XAUUSD spot"
	Value        *float64      `json:"value"`
	DeltaPct     *float64      `json:"delta_pct"` // Close − session Open, %
	Rule         MacroLampRule `json:"rule"`
	Contribution string        `json:"contribution"` // positive|neutral|negative; "" = not voting
	Voting       bool          `json:"voting"`
	Weight       float64       `json:"weight"`     // nominal weight in this model (0 = never votes)
	Points       *float64      `json:"points"`     // renormalised points the lamp adds; null without a score
	MaxPoints    *float64      `json:"max_points"` // the lamp's renormalised share
	VsNeutral    *float64      `json:"vs_neutral"` // points − max_points/2; score = 50 + Σ vs_neutral
	Source       string        `json:"source"`
	AsOf         string        `json:"as_of"`
}

type MacroFreshness struct {
	OldestAsOf       string         `json:"oldest_as_of"`      // oldest live lamp stamp (= data_as_of)
	TradfinAsOf      string         `json:"tradfin_as_of"`     // freshest lamp stamp (backend)
	SessionDates     []string       `json:"session_dates"`     // distinct live-lamp dates, ascending
	Mixed            bool           `json:"mixed"`             // live lamps come from more than one date
	ScheduledWeekend bool           `json:"scheduled_weekend"` // outside the clock-based tradfin week (no holiday calendar)
	CapturedAt       string         `json:"captured_at"`       // backend response time (card build)
	Sources          map[string]int `json:"sources"`           // provider → live lamp count
}

type MacroFearGreed struct {
	Value           int      `json:"value"`
	Label           string   `json:"label"`
	AsOf            string   `json:"as_of"`
	FetchedAt       string   `json:"fetched_at"`
	AgeHours        *float64 `json:"age_hours"` // null when the backend sent no time
	Stale           bool     `json:"stale"`
	StaleAfterHours int      `json:"stale_after_hours"`
	InScore         bool     `json:"in_score"` // always false
}

func ptrF(v float64) *float64 { return &v }

func round2(v float64) float64 { return math.Round(v*100) / 100 }

func (r modelRead) readout(m *MacroResp, reading string, ruleScore *int, now time.Time) *MacroReadout {
	out := &MacroReadout{
		Model:          r.m.id,
		Experimental:   r.m.experimental,
		Reading:        reading,
		RuleScore:      ruleScore,
		Bands:          MacroBands{Low: r.m.low, High: r.m.high},
		MinVotingLamps: r.m.minVoters,
		VotingLamps:    r.voters,
		Lamps:          make([]MacroLampReadout, 0, len(r.lamps)),
	}
	if r.pts {
		out.RuleScoreExact = ptrF(round2(r.exact))
	}
	rule := riskRule
	if r.m.id == goldModel.id {
		rule = goldRule
	}
	for _, lr := range r.lamps {
		l := lr.l
		if lampLive(l) {
			out.LiveLamps++
		}
		lo := MacroLampReadout{
			Key: l.Key, Label: l.Label, Value: l.Value, DeltaPct: l.DeltaPct,
			Rule: rule(l.Key), Contribution: lr.word, Voting: lr.word != "", Weight: lr.weight,
			Source: l.Source, AsOf: l.AsOf,
		}
		if l.Key == "gold" {
			lo.Instrument = goldInstrument(l.Source)
		}
		if r.pts && lr.word != "" {
			lo.Points, lo.MaxPoints, lo.VsNeutral = ptrF(round2(lr.points)), ptrF(round2(lr.maxPts)), ptrF(round2(lr.vsNeutral))
		}
		out.Lamps = append(out.Lamps, lo)
	}
	out.Freshness = macroFreshness(m)
	if f := m.FNG; f != nil && f.OK {
		fg := &MacroFearGreed{Value: f.Value, Label: f.Label, AsOf: f.AsOf, FetchedAt: f.FetchedAt,
			StaleAfterHours: int(fngStaleAfter.Hours())}
		if _, age, known := fngAge(f, now); known {
			fg.AgeHours = ptrF(round2(age.Hours()))
			fg.Stale = age > fngStaleAfter
		}
		out.FearGreed = fg
	}
	return out
}

func macroFreshness(m *MacroResp) MacroFreshness {
	fr := MacroFreshness{
		TradfinAsOf:      m.TradfinAsOf,
		ScheduledWeekend: !m.TradfinOpen,
		CapturedAt:       m.CapturedAt,
		SessionDates:     []string{},
		Sources:          macroSources(m.Lamps),
	}
	seen := map[string]bool{}
	var oldest time.Time
	for _, l := range m.Lamps {
		if !lampLive(l) {
			continue
		}
		t, ok := lampStamp(l.AsOf)
		if !ok {
			continue
		}
		if oldest.IsZero() || t.Before(oldest) {
			oldest = t
		}
		if d := dayOf(t).Format("2006-01-02"); !seen[d] {
			seen[d] = true
			fr.SessionDates = append(fr.SessionDates, d)
		}
	}
	sort.Strings(fr.SessionDates)
	fr.Mixed = len(fr.SessionDates) > 1
	if !oldest.IsZero() {
		fr.OldestAsOf = oldest.Format(time.RFC3339)
	}
	return fr
}

// ── the cards ────────────────────────────────────────────────────────────────

// macroView is one backend payload, read by both models.
type macroView struct {
	m      *MacroResp
	regime string // effective regime (effectiveMacroRegime)
	now    time.Time
	risk   modelRead
	gold   modelRead
}

func newMacroView(m *MacroResp, regime string) macroView {
	v := macroView{m: m, regime: regime, now: parseWhen(m.CapturedAt),
		risk: riskModel.read(m.Lamps), gold: goldModel.read(m.Lamps)}
	// The risk numbers are printed only when they reproduce the backend's
	// composite: a version skew must never show a sum that does not add up.
	agree := v.risk.score != nil && m.Composite != nil && *v.risk.score == *m.Composite
	v.risk.pts = v.risk.pts && agree
	return v
}

func (v macroView) weekend() bool { return !v.m.TradfinOpen }

// scoreText is the backend composite for the verdicts ("" without one).
func (v macroView) compositeText() string {
	if v.m.Composite == nil {
		return ""
	}
	return fmt.Sprintf("rule score %d/100", *v.m.Composite)
}

// globalVerdict: regime → rule score → the bands.
func (v macroView) globalVerdict() string {
	head := strings.ToUpper(regimeWord(v.regime))
	if c := v.compositeText(); c != "" {
		return fmt.Sprintf("%s — %s (risk-on above %d, risk-off below %d)", head, c, riskModel.high, riskModel.low)
	}
	return fmt.Sprintf("%s — no rule score: %d of %d lamps vote, the rule needs %d",
		head, v.risk.voters, v.risk.inModel, riskModel.minVoters)
}

// btcContext / goldContext are the global card's asset lines: context of the
// regime, not signals on the assets. The gold line calls its model separate:
// the factor lines above are contributions to the rule score (Gold rising is
// negative there), while the gold score is its own model with its own
// mapping, so the two can point different ways without contradicting.
func (v macroView) btcContext() string {
	return fmt.Sprintf("BTC macro backdrop: %s (the regime itself); BTC direction is not inferred", regimeWord(v.regime))
}

func (v macroView) goldContext() string {
	g := v.gold
	if g.score == nil {
		return fmt.Sprintf("Gold macro backdrop: no read, %d of %d lamps vote (the separate gold model needs %d)", g.voters, g.inModel, goldMinVoters)
	}
	return fmt.Sprintf("Gold macro backdrop: %s, gold score %d/100 (separate experimental model, own weights)", g.reading(), *g.score)
}

// fillGlobal writes the global regime card: regime and score → main factors
// → what holds it → data → breakdown → asset context → F&G.
func (v macroView) fillGlobal(c *Card) {
	switch v.regime {
	case "risk_on":
		c.Emoji, c.Short = emojiBull, "risk-on"
	case "risk_off":
		c.Emoji, c.Short = emojiBear, "risk-off"
	default:
		c.Emoji, c.Short = emojiNeutral, "mixed"
	}
	c.Verdict = v.globalVerdict()
	var f []string
	if v.weekend() {
		f = append(f, macroWeekendBanner)
	}
	f = append(f, v.risk.factorLines()...)
	f = append(f, v.risk.holdsLine(regimeWord(v.regime)), macroDataLine(v.m.Lamps))
	if b := v.risk.breakdownLine(); b != "" {
		f = append(f, b)
	}
	f = append(f, v.btcContext(), v.goldContext())
	if l := fngLine(v.m.FNG, v.now, riskModel.scoreName); l != "" {
		f = append(f, l)
	}
	c.Facts = append(c.Facts, f...)
	if v.m.Composite != nil {
		c.Deviation = clampInt(abs(*v.m.Composite-50)*2, 0, 100)
	}
	c.DataTime = macroDataTime(v.m)
	c.SourceNote = macroSourceNote(v.m.Lamps)
	c.Blocks = v.blocks()
	c.Macro = v.risk.readout(v.m, v.regime, v.m.Composite, v.now)
}

// fillBTC writes the BTC backdrop: the same regime, framed as context for BTC.
func (v macroView) fillBTC(c *Card) {
	switch v.regime {
	case "risk_on":
		c.Emoji = emojiBull
	case "risk_off":
		c.Emoji = emojiBear
	default:
		c.Emoji = emojiNeutral
	}
	word := regimeWord(v.regime)
	c.Short = "btc backdrop: " + word
	if ct := v.compositeText(); ct != "" {
		c.Verdict = fmt.Sprintf("BTC MACRO BACKDROP: %s — %s; BTC direction is not inferred", strings.ToUpper(word), ct)
	} else {
		c.Verdict = fmt.Sprintf("BTC MACRO BACKDROP: %s — no rule score (%d of %d lamps vote, needs %d)",
			strings.ToUpper(word), v.risk.voters, v.risk.inModel, riskModel.minVoters)
	}
	var f []string
	if v.weekend() {
		f = append(f, macroWeekendBanner)
	}
	for _, lr := range v.risk.lamps {
		f = append(f, lampFull(lr.l)+" → "+v.risk.outcome(lr, ""))
	}
	f = append(f, v.scoreLine(v.risk), v.risk.holdsLine(word), macroDataLine(v.m.Lamps))
	if l := fngLine(v.m.FNG, v.now, riskModel.scoreName); l != "" {
		f = append(f, l)
	}
	c.Facts = append(c.Facts, f...)
	if v.m.Composite != nil {
		c.Deviation = clampInt(abs(*v.m.Composite-50)*2, 0, 100)
	}
	c.DataTime = macroDataTime(v.m)
	c.SourceNote = macroSourceNote(v.m.Lamps)
	c.Macro = v.risk.readout(v.m, v.regime, v.m.Composite, v.now)
}

// scoreLine sums the asset view's lamp lines: the score as 50 + the printed
// contributions, or the plain score when they may not be printed.
func (v macroView) scoreLine(r modelRead) string {
	bands := fmt.Sprintf("%s above %d, %s below %d", r.m.hiWord, r.m.high, r.m.loWord, r.m.low)
	score := r.score
	if r.m.id == riskModel.id {
		score = v.m.Composite // authoritative
	}
	switch {
	case score == nil:
		return fmt.Sprintf("No %s: %d of %d lamps vote, the rule needs %d", r.m.scoreName, r.voters, r.inModel, r.m.minVoters)
	case r.pts:
		return fmt.Sprintf("%s %d/100 %s 50 + the contributions above (%s)", upperFirst(r.m.scoreName), *score, r.eqSign(*score), bands)
	}
	return fmt.Sprintf("%s %d/100 (%s)", upperFirst(r.m.scoreName), *score, bands)
}

// fillGold writes the experimental gold backdrop.
func (v macroView) fillGold(c *Card) {
	g := v.gold
	reading := g.reading()
	switch reading {
	case "":
		c.Emoji = emojiNeutral
		c.Verdict = fmt.Sprintf("GOLD MACRO BACKDROP: no read — %d of %d lamps vote, the model needs %d", g.voters, g.inModel, goldMinVoters)
		c.Short = "gold backdrop: no read"
		c.Status = statusNoData
	case contribPositive:
		c.Emoji, c.State = emojiBull, goldSupport
	case contribNegative:
		c.Emoji, c.State = emojiBear, goldPressure
	default:
		c.Emoji, c.State = emojiNeutral, goldNeutral
	}
	if reading != "" {
		c.Verdict = fmt.Sprintf("GOLD MACRO BACKDROP: %s — gold score %d/100 (experimental model)", strings.ToUpper(reading), *g.score)
		c.Short = "gold backdrop: " + reading
		c.Deviation = clampInt(abs(*g.score-50)*2, 0, 100)
	}
	var f []string
	if v.weekend() {
		f = append(f, macroWeekendBanner)
	}
	var self *MacroLamp
	for i, lr := range g.lamps {
		if lr.l.Key == "gold" {
			self = &g.lamps[i].l
			continue
		}
		f = append(f, lampFull(lr.l)+" → "+g.outcome(lr, " for gold"))
	}
	if self != nil && self.Value != nil {
		f = append(f, lampSelfLine(*self))
	}
	f = append(f, v.scoreLine(g))
	if reading != "" {
		f = append(f, g.holdsLine(reading))
	}
	f = append(f, goldRuleLines()...)
	f = append(f, macroDataLine(v.m.Lamps))
	if l := fngLine(v.m.FNG, v.now, goldModel.scoreName); l != "" {
		f = append(f, l)
	}
	c.Facts = append(c.Facts, f...)
	c.DataTime = macroDataTime(v.m)
	c.SourceNote = macroSourceNote(v.m.Lamps)
	gr := reading
	if gr == "" {
		gr = "no_read"
	}
	c.Macro = g.readout(v.m, gr, g.score, v.now)
}

// goldRuleLines state the gold model from its own mapping and weights.
func goldRuleLines() []string {
	var same, inv, weights []string
	for _, k := range []string{"dxy", "rates", "vix", "spx"} {
		name := macroShortNames[k]
		switch goldLampView(k, "tailwind") {
		case goldSupport:
			same = append(same, name)
		case goldPressure:
			inv = append(inv, name)
		}
		weights = append(weights, fmt.Sprintf("%s %s", name, fmtNum(goldWeightOf(k))))
	}
	return []string{
		fmt.Sprintf("Rule: %s count as in the risk model; %s count inverted; Gold itself does not vote",
			strings.Join(same, ", "), strings.Join(inv, ", ")),
		"Weights: " + strings.Join(weights, " · ") + " (thresholds are the risk model's)",
	}
}

// ── content blocks (global card) ─────────────────────────────────────────────

// limitations (additive 2026-09-16) is the global card's caveat. It is NOT a
// verbatim fact line: it is assembled from three phrases the card already
// prints elsewhere — "A backdrop, not a forecast" (howTexts[keyMacro]), "BTC
// direction is not inferred" (btcContext) and "experimental model, own
// weights" (goldContext). The gold clause is added only when the gold model
// scored: when goldContext prints "no read" there is no gold score to qualify.
func (v macroView) limitations() string {
	s := "A backdrop, not a forecast: BTC direction is not inferred from the regime"
	if v.gold.score != nil {
		s += ", and the gold score is a separate experimental model with its own weights"
	}
	return s + "."
}

// blocks are the global card's content sentences. why_level stays "": Macro
// has no price level. nil when there is nothing to read.
func (v macroView) blocks() *ContentBlocks {
	r := v.risk
	word := regimeWord(v.regime)
	var pos, neg, neu int
	for _, lr := range r.lamps {
		switch lr.word {
		case contribPositive:
			pos++
		case contribNegative:
			neg++
		case contribNeutral:
			neu++
		}
	}
	score := "no rule score"
	if c := v.compositeText(); c != "" {
		score = c
	}
	b := &ContentBlocks{
		WhatHappened: fmt.Sprintf("Lamps%s: %d positive, %d negative, %d neutral in this model; %s",
			v.sessionSpan(), pos, neg, neu, score),
		Scenarios:   v.scenarios(),
		Regime:      v.regimeBlock(),
		Context:     v.contextBlock(),
		Limitations: v.limitations(),
	}
	hi, lo, min := riskModel.high, riskModel.low, riskModel.minVoters
	var inv string
	switch {
	case v.m.Composite == nil:
		// nothing scored → nothing to invalidate
	case word == "risk-on":
		inv = fmt.Sprintf("Risk-on ends at a rule score of %d or below, or with fewer than %d voting lamps (now %d, %d voting)",
			hi, min, *v.m.Composite, r.voters)
	case word == "risk-off":
		inv = fmt.Sprintf("Risk-off ends at a rule score of %d or above, or with fewer than %d voting lamps (now %d, %d voting)",
			lo, min, *v.m.Composite, r.voters)
	default:
		inv = fmt.Sprintf("Mixed ends when the rule score goes above %d (risk-on) or below %d (risk-off); now %d",
			hi, lo, *v.m.Composite)
	}
	if inv != "" {
		b.Invalidates = &inv
	}
	return b
}

// sessionSpan: " (Sep 14 session)", " (sessions Sep 14 to Sep 15)" or "".
func (v macroView) sessionSpan() string {
	fr := macroFreshness(v.m)
	switch n := len(fr.SessionDates); {
	case n == 0:
		return ""
	case n == 1:
		d, _ := time.Parse("2006-01-02", fr.SessionDates[0])
		return " (" + dayShown(d) + " session)"
	default:
		a, _ := time.Parse("2006-01-02", fr.SessionDates[0])
		z, _ := time.Parse("2006-01-02", fr.SessionDates[n-1])
		return " (sessions " + dayShown(a) + " to " + dayShown(z) + ")"
	}
}

func (v macroView) regimeBlock() string {
	const tail = ": a tradfin backdrop, not a BTC or gold forecast"
	hi, lo := riskModel.high, riskModel.low
	if v.m.Composite == nil {
		return fmt.Sprintf("Mixed with no rule score (%d voting lamps, the rule needs %d)%s", v.risk.voters, riskModel.minVoters, tail)
	}
	s := *v.m.Composite
	switch v.regime {
	case "risk_on":
		return fmt.Sprintf("Risk-on by this model's rule (score %d above %d)%s", s, hi, tail)
	case "risk_off":
		return fmt.Sprintf("Risk-off by this model's rule (score %d below %d)%s", s, lo, tail)
	}
	return fmt.Sprintf("Mixed by this model's rule (score %d within %d-%d)%s", s, lo, hi, tail)
}

// scenarios — the two regime changes the bands define, conditional, no
// probabilities.
func (v macroView) scenarios() []string {
	hi, lo, min := riskModel.high, riskModel.low, riskModel.minVoters
	if v.m.Composite == nil {
		return []string{
			fmt.Sprintf("If at least %d lamps vote, the model computes a rule score and reads a regime", min),
			fmt.Sprintf("If fewer than %d lamps vote, the reading stays mixed with no rule score", min),
		}
	}
	switch v.regime {
	case "risk_on":
		return []string{
			fmt.Sprintf("If the rule score stays above %d with %d+ voting lamps, the model keeps reading risk-on", hi, min),
			fmt.Sprintf("If the rule score falls to %d-%d, the reading turns mixed; below %d, risk-off", lo, hi, lo),
		}
	case "risk_off":
		return []string{
			fmt.Sprintf("If the rule score stays below %d with %d+ voting lamps, the model keeps reading risk-off", lo, min),
			fmt.Sprintf("If the rule score rises to %d-%d, the reading turns mixed; above %d, risk-on", lo, hi, hi),
		}
	}
	return []string{
		fmt.Sprintf("If the rule score rises above %d with %d+ voting lamps, the model reads risk-on", hi, min),
		fmt.Sprintf("If the rule score falls below %d with %d+ voting lamps, the model reads risk-off", lo, min),
	}
}

// contextBlock — the local context beside the regime: the asset backdrops
// and Fear & Greed, each marked for what it is.
func (v macroView) contextBlock() string {
	gold := "no read"
	if v.gold.score != nil {
		gold = v.gold.reading()
	}
	s := fmt.Sprintf("Backdrops: BTC %s (direction not inferred) · gold %s (experimental)", regimeWord(v.regime), gold)
	if f := v.m.FNG; f != nil && f.OK {
		fg := fmt.Sprintf("F&G %d, not scored", clampInt(f.Value, 0, 100))
		if ref, age, known := fngAge(f, v.now); known && age > fngStaleAfter {
			fg = fmt.Sprintf("F&G %d stale (%s)", clampInt(f.Value, 0, 100), dayShown(ref))
		}
		s += " · " + fg
	}
	return s
}

// ── unknown (zero real lamps) ────────────────────────────────────────────────

// macroNoDataNote words the absence: never "market closed" — outside the
// clock-based week it is a scheduled weekend, inside it just missing data.
func macroNoDataNote(tradfinOpen bool) string {
	if tradfinOpen {
		return "no current tradfin data"
	}
	return "no current tradfin data (scheduled tradfin weekend)"
}

// macroLastDataLine: the last date any tradfin lamp carried, when known.
func macroLastDataLine(m *MacroResp) string {
	if t, ok := lampStamp(m.TradfinAsOf); ok {
		return "Last tradfin data in the feed: " + dayShown(t)
	}
	return ""
}
