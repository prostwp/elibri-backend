package demobot

import (
	"strings"
	"time"
)

// Agent keys used by the priority rule and command router.
const (
	keyMacro    = "macro"
	keyWhale    = "whale"
	keyFunding  = "funding"
	keyMomentum = "momentum"
	keyTrend    = "trend"
	keySR       = "sr"
	keyVol      = "vol"
	keyDigest   = "digest"
	keyTop      = "top"
	keyRisk     = "risk"
	keyFX       = "fx"
	keyNews     = "news"
	keyGold     = "gold"
)

// signalOrder is the fixed tie-break order among CONFIRMED readings. Only
// these three compete for the top slot when macro RISK-OFF does not take it.
var signalOrder = []string{keyFunding, keyMomentum, keyTrend}

// unconfirmedTieOrder breaks ties when nothing is confirmed (tier 2). Funding
// goes last: a balanced funding read carries nothing beyond the rate, and it
// used to win the all-neutral case purely by sitting first in signalOrder.
var unconfirmedTieOrder = []string{keyMomentum, keyTrend, keyFunding}

// digestShowNoHighlight switches the DISPLAY of the no_highlight state. Off
// until the product owner decides whether the digest may answer "nothing is
// highlighted" (Digest plan, human decision 2). Off = the digest keeps showing
// the tier-2 card under the usual headline, and the selection line says
// honestly that nothing was confirmed. The state itself is always computed
// and served (digest.selection.state), whatever this flag says.
// A var only so tests can exercise both displays; nothing sets it at runtime.
var digestShowNoHighlight = false

// ── freshness: expected max age per source ───────────────────────────────────
//
// A reading older than its source normally allows is stale: it may still be
// rendered (with its own stamp), but it can NOT take the top slot. The limits
// are expected-cadence bounds, not tuning knobs:
//
//   - Bar-based reads (momentum and trend on Binance 4h bars): the data time is
//     the close of the last CLOSED bar, so a healthy read is always less than
//     one bar old. staleAfterBars = 2 means at least one whole bar was missed.
//   - Funding is a point-in-time read stamped at request time. Anything older
//     than fundingMaxAge is a replayed card, not a current rate.
//   - Macro lamps carry tradfin SESSION stamps (Yahoo stamps the session
//     start). The longest regular gap between two session starts is the
//     weekend, Fri → Mon = 72h; macroRiskOffMaxAge adds 8h for a late feed. A
//     holiday Monday therefore makes risk-off ineligible until the next session
//     — the safe direction: an old backdrop never takes the headline.
const (
	staleAfterBars     = 2
	fundingMaxAge      = 15 * time.Minute
	macroRiskOffMaxAge = 80 * time.Hour
)

// barMaxAge is the stale bound for a read on bars of the given interval.
func barMaxAge(interval string) time.Duration {
	return staleAfterBars * time.Duration(intervalSeconds[interval]) * time.Second
}

// topMaxAge is the eligibility age limit of one priority candidate. The digest
// sweep reads momentum's ranked assets (BTC/ETH) and trend on btcSpec's bars.
func topMaxAge(key string) time.Duration {
	switch key {
	case keyFunding:
		return fundingMaxAge
	case keyMomentum, keyTrend:
		return barMaxAge(btcSpec.Interval)
	}
	return 0
}

// ── candidates ───────────────────────────────────────────────────────────────

// Exclusion reasons served in digest.selection.candidates[].excluded.
const (
	excludedDegraded   = "degraded"     // the agent produced no reading (status != ok)
	excludedStale      = "stale"        // older than topMaxAge for its source
	excludedNoDataTime = "no_data_time" // a live card without a data time: age unknown, never trusted
	// excludedNoRankedRead: a momentum card whose BTC/ETH produced no reading.
	// Gold/FX reads on it are shown but never ranked, so without a crypto read
	// the card has nothing to compete with (and no ranked bar to judge).
	excludedNoRankedRead = "no_ranked_read"
)

// Selection rules served in digest.selection.rule.
const (
	ruleMacroRiskOff  = "macro_risk_off"       // a fresh, fully lit RISK-OFF macro
	ruleConfirmed     = "strongest_confirmed"  // highest score among confirmed readings
	ruleUnconfirmed   = "fallback_unconfirmed" // nothing confirmed: tier-2 card (no_highlight)
	ruleFallbackMacro = "fallback_macro"       // no eligible candidate at all (no_highlight)
)

// Macro risk-off gate outcomes served in digest.selection.macro_risk_off_gate
// (empty when the regime is not risk_off).
const (
	gateEligible     = "eligible"
	gateDegraded     = "degraded"      // card status not ok
	gatePartialLamps = "partial_lamps" // fewer voting lamps than the rule's minimum, or no rule score
	gateStale        = "stale"         // oldest live lamp older than macroRiskOffMaxAge
)

// topCandidate is one priority agent's standing in a sweep.
type topCandidate struct {
	Key       string
	Eligible  bool
	Excluded  string // "" when eligible, else one of the excluded* reasons
	Confirmed bool   // the agent's own rule committed to a finding
	Score     int    // 0..100, meaningful only within the same tier
	AsOf      time.Time
	MaxAge    time.Duration
}

// rankScore is the score a candidate ranks on:
//
//   - funding: the builder's Deviation, relative to the threshold of its own
//     side (see fundingDeviation) — crowded or balanced alike;
//   - momentum: the builder's Deviation, which is non-zero only for a
//     confirmed bullish/bearish crypto read (see momentumRankScore);
//   - trend: ADX×2 for a confirmed trend; 0 for flat / grey / conflict. The
//     raw ADX of an unconfirmed state says how strong a move is that the
//     agent itself declined to call a trend — it must not win on it (live
//     2026-09-15: "Top signal: Trend Agent · BTC — Flat · no trend to read").
func rankScore(key string, c Card) int {
	if key == keyTrend && !c.confirmed {
		return 0
	}
	return clampInt(c.Deviation, 0, 100)
}

// rankCandidate evaluates one priority card at the sweep time now.
func rankCandidate(key string, c Card, now time.Time) topCandidate {
	cand := topCandidate{Key: key, MaxAge: topMaxAge(key), Confirmed: c.confirmed, Score: rankScore(key, c)}
	cand.AsOf = c.DataTime
	if !c.rankAsOf.IsZero() {
		cand.AsOf = c.rankAsOf
	}
	switch {
	case c.effectiveStatus() != statusOK:
		cand.Excluded = excludedDegraded
	case key == keyMomentum && c.noRankedRead:
		// Only BTC/ETH rank; a card that reads just gold/FX would otherwise
		// fall back to the gold bar's DataTime and compete (review 2026-09-15).
		cand.Excluded = excludedNoRankedRead
	case cand.AsOf.IsZero():
		cand.Excluded = excludedNoDataTime
	case now.Sub(cand.AsOf) > cand.MaxAge:
		cand.Excluded = excludedStale
	default:
		cand.Eligible = true
	}
	if !cand.Eligible {
		cand.Confirmed, cand.Score = false, 0
	}
	return cand
}

// macroRiskOffGate decides whether a risk_off regime may take the top slot:
// only a real reading (status ok) with at least the rule's minimum voting
// lamps behind a rule score, on lamps no older than macroRiskOffMaxAge.
// "" when the regime is not risk_off.
func macroRiskOffGate(regime string, c Card, now time.Time) string {
	if regime != "risk_off" {
		return ""
	}
	switch {
	case c.effectiveStatus() != statusOK:
		return gateDegraded
	case c.Macro == nil || c.Macro.RuleScore == nil || c.Macro.VotingLamps < c.Macro.MinVotingLamps:
		return gatePartialLamps
	case c.DataTime.IsZero() || now.Sub(c.DataTime) > macroRiskOffMaxAge:
		return gateStale
	}
	return gateEligible
}

// ── the rule ─────────────────────────────────────────────────────────────────

// pickTop implements the deterministic prioritization rule for /digest, /top
// and the showcase:
//
//  1. A macro RISK-OFF that passes macroRiskOffGate is always the top slot —
//     a hostile big-money backdrop outranks any single crypto reading. Regime
//     "unknown", a degraded card, too few voting lamps or stale lamps never
//     take it: an absence or a remnant of data must not outrank live readings.
//  2. Otherwise the winner is the ELIGIBLE (live and fresh) CONFIRMED reading
//     with the highest score among funding, momentum and trend (rankScore);
//     ties break funding > momentum > trend.
//  3. Nothing confirmed → state no_highlight. Until the product decision the
//     digest still shows a card (digestShowNoHighlight): the eligible
//     unconfirmed candidate with the highest score, ties momentum > trend >
//     funding. Momentum and trend score 0 here, so a balanced funding read wins
//     only on a real non-zero score and never on a tie.
//  4. No eligible candidate at all → macro, whatever its state; the caller
//     renders the honest macro card (offline / unknown included). Also
//     no_highlight: a last resort, not the unknown regime "winning".
//
// The scales of the three agents are NOT calibrated against each other; the
// digest says so on its selection line. FX assets never compete (crypto-only
// trio: momentum ranks its Binance reads only).
func pickTop(macroGate string, cands []topCandidate) (winner, rule string) {
	if macroGate == gateEligible {
		return keyMacro, ruleMacroRiskOff
	}
	by := map[string]topCandidate{}
	for _, c := range cands {
		by[c.Key] = c
	}
	best := func(order []string, confirmed bool) string {
		out, bestScore := "", -1
		for _, k := range order {
			c, ok := by[k]
			if !ok || !c.Eligible || c.Confirmed != confirmed {
				continue
			}
			if c.Score > bestScore { // strict: the earlier key keeps a tie
				out, bestScore = k, c.Score
			}
		}
		return out
	}
	if w := best(signalOrder, true); w != "" {
		return w, ruleConfirmed
	}
	if w := best(unconfirmedTieOrder, false); w != "" {
		return w, ruleUnconfirmed
	}
	return keyMacro, ruleFallbackMacro
}

// topPick is one sweep's full selection: the winner, the rule that chose it,
// and every candidate's standing — the machine-readable "why".
type topPick struct {
	Winner      string
	Rule        string
	NoHighlight bool // nothing confirmed and eligible (rules 3 and 4)
	MacroRegime string
	MacroGate   string // "" unless the regime is risk_off
	Candidates  []topCandidate
	At          time.Time // the sweep time freshness was judged at
}

// selection evaluates the rule over one sweep.
func (g gathered) selection() topPick {
	now := g.at
	if now.IsZero() {
		now = time.Now().UTC()
	}
	p := topPick{MacroRegime: g.regime, At: now}
	for _, k := range signalOrder {
		if c, ok := g.cards[k]; ok {
			p.Candidates = append(p.Candidates, rankCandidate(k, c, now))
		}
	}
	p.MacroGate = macroRiskOffGate(g.regime, g.cards[keyMacro], now)
	p.Winner, p.Rule = pickTop(p.MacroGate, p.Candidates)
	p.NoHighlight = p.Rule == ruleUnconfirmed || p.Rule == ruleFallbackMacro
	return p
}

// topSelection applies the rule to one gather sweep and resolves the winning
// card, falling back to macro when the winner produced no card (belt and
// braces — candidates only exist for keys present in g.cards, and gather
// always stores a macro card). Shared by /digest, /top and the HTTP API.
func topSelection(g gathered) (string, Card) {
	winner := g.selection().Winner
	card, ok := g.cards[winner]
	if !ok {
		winner = keyMacro
		card = g.cards[keyMacro]
	}
	return winner, card
}

// ── the selection line ───────────────────────────────────────────────────────

// signalNames is the reader-facing name of each priority agent, in
// signalOrder, for the selection line.
var signalNames = map[string]string{keyFunding: "Funding", keyMomentum: "Momentum", keyTrend: "Trend"}

// signalScope is what each priority agent's RANKED reading covers in the
// digest sweep (gather): momentum ranks its BTC/ETH reads only — gold and FX
// on the same card are shown, never ranked — and trend reads BTC. Funding is
// market-wide. Named on the selection line (2026-09-15) so "no confirmed
// reading" cannot be read against a card that counts a confirmed gold read.
var signalScope = map[string]string{keyMomentum: "BTC/ETH", keyTrend: "BTC"}

// scopedSignalName is the reader-facing name with its ranked scope:
// "Momentum (BTC/ETH)".
func scopedSignalName(key string) string {
	if s := signalScope[key]; s != "" {
		return signalNames[key] + " (" + s + ")"
	}
	return signalNames[key]
}

func allSignalNames() string {
	names := make([]string, 0, len(signalOrder))
	for _, k := range signalOrder {
		names = append(names, scopedSignalName(k))
	}
	return strings.Join(names, ", ")
}

// confirmedSplit is what the strongest_confirmed rule actually compared:
// pickTop scores only candidates that are BOTH eligible and confirmed
// (best(signalOrder, true)). An eligible but unconfirmed reading is fresh and
// live, yet it never competed — so the selection line and the digest caveat
// list the confirmed ones as the comparison, and name the eligible ones only
// as the field the single confirmed reading stood in.
func confirmedSplit(p topPick) (eligible, confirmed []string) {
	for _, c := range p.Candidates {
		if !c.Eligible {
			continue
		}
		eligible = append(eligible, scopedSignalName(c.Key))
		if c.Confirmed {
			confirmed = append(confirmed, scopedSignalName(c.Key))
		}
	}
	return eligible, confirmed
}

// selectionLine is the one line (≤110 chars) saying how the highlighted card
// was chosen. It explains the RULE, never the market: no "because the market
// is about to…". It says the scales are not calibrated only where scores were
// actually compared — two or more confirmed readings (confirmedSplit).
// Every list names what each agent's ranked reading covers (signalScope).
func selectionLine(p topPick) string {
	all := allSignalNames()
	switch p.Rule {
	case ruleMacroRiskOff:
		return "Macro risk-off tops the digest by rule, ahead of " + all
	case ruleConfirmed:
		eligible, confirmed := confirmedSplit(p)
		switch {
		case len(confirmed) >= 2:
			return "Selected among " + strings.Join(confirmed, ", ") + " by the digest rule; scales not calibrated"
		case len(eligible) >= 2:
			return signalNames[p.Winner] + " is the only confirmed reading among " + strings.Join(eligible, ", ") + "; selected by rule"
		}
		return signalNames[p.Winner] + " is the only fresh live reading among " + all + "; selected by rule"
	case ruleUnconfirmed:
		return "No confirmed reading among " + all + "; shown by the digest's fallback order"
	default:
		return "No fresh live reading among " + all + "; the macro card is shown instead"
	}
}
