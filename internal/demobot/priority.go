package demobot

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
)

// signalOrder is the fixed tie-break order of the "hook" rule. Only these
// three compete for the top slot when macro is not RISK-OFF.
var signalOrder = []string{keyFunding, keyMomentum, keyTrend}

// pickTop implements the deterministic prioritization rule for /digest and
// /top:
//
//  1. If the macro regime is "risk_off", macro is always the top signal —
//     a hostile big-money backdrop outranks any single crypto signal.
//     Regime "unknown" (zero real tradfin lamps — market closed or feed dark)
//     NEVER wins this slot: it is a statement of missing data, not RISK-OFF,
//     and an absence of inputs must not outrank live crypto signals.
//  2. Otherwise the winner is the agent with the highest deviation-from-
//     neutral (0..100) among funding, momentum and trend. Agents whose data
//     source failed are simply absent from `deviations`.
//  3. Ties are broken by the fixed order funding > momentum > trend
//     (strict `>` while scanning in signalOrder keeps the earlier agent).
//  4. If none of the three produced data, fall back to macro — the caller
//     renders whatever macro state it has, including an honest offline or
//     unknown ("no tradfin data") card. That fallback is a last resort, not
//     the unknown regime "winning": with any live crypto signal present, an
//     unknown macro never tops /digest or /top.
//
// FX in v1: FX assets do NOT compete for the top slot. The priority trio is
// crypto-only — the momentum agent's deviation is computed from its
// Binance-sourced reads (BTC/ETH) exclusively, and the /fx overview appears
// in /digest as an informational block below the crypto one-liners.
func pickTop(macroRegime string, deviations map[string]int) string {
	if macroRegime == "risk_off" {
		return keyMacro
	}
	best := ""
	bestDev := -1
	for _, k := range signalOrder {
		d, ok := deviations[k]
		if !ok {
			continue
		}
		if d > bestDev {
			best = k
			bestDev = d
		}
	}
	if best == "" {
		return keyMacro
	}
	return best
}

// topSelection applies pickTop to one gather sweep and resolves the winning
// card, falling back to macro when the winner produced no card (belt and
// braces — deviations only contains keys present in g.cards, and gather
// always stores a macro card). Shared by /digest, /top and the HTTP API.
func topSelection(g gathered) (string, Card) {
	winner := pickTop(g.regime, g.deviations())
	card, ok := g.cards[winner]
	if !ok {
		winner = keyMacro
		card = g.cards[keyMacro]
	}
	return winner, card
}
