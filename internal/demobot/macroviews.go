package demobot

// macroviews.go — per-asset interpretations of the macro lamps (checklist B2).
//
// The backend lamps encode tailwind|neutral|headwind FOR CRYPTO, with tuned
// session-delta thresholds (internal/macro compute.go LampStatus: any relief
// counts, the negative reading needs a strong move). The asset views re-read
// those SAME statuses from each asset's perspective instead of re-deriving
// thresholds — one tuned mapping, two framings.
//
// GOLD verdict mapping (the checklist formula):
//
//	DXY up            = pressure on gold  (dollar-priced metal — tight USD inverse)
//	yields up         = pressure          (higher yields raise the cost of holding zero-yield gold)
//	VIX up / risk-off = support           (fear = safe-haven bid)
//	SPX down          = mild support      (flight to safety; "mild" = lowest weight)
//
// implemented as a crypto-status → gold-status table per lamp:
//
//	dxy   tailwind (dollar down)   → support  | headwind (dollar up strong)   → pressure
//	rates tailwind (yields down)   → support  | headwind (yields up strong)   → pressure
//	vix   tailwind (calm, <18)     → pressure | headwind (fear, >25)          → support   (inverted)
//	spx   tailwind (equities up)   → pressure | headwind (equities down)      → support   (inverted)
//	gold  — excluded: the gold lamp IS the asset; its own session move is
//	        reported as a fact, never as a driver of its own outlook.
//
// Gold composite: support → +weight, neutral → +weight/2, pressure → +0 over
// the VOTING lamps (status known, gold excluded), renormalised. Weights encode
// the formula's emphasis — DXY 40 / rates 25 / VIX 25 / SPX 10 (SPX lowest =
// the checklist's "mild"). Thresholds mirror the crypto composite: >65
// support, <35 pressure, else mixed; fewer than 3 voters → no verdict (an
// asset read off half the map would be a guess).
//
// BTC view = the existing crypto-centric regime, labeled as an asset view
// (checklist: "текущая логика оформляется как asset-view BTC").
//
// Wording note: "support"/"pressure" are demobot card words (the S/R card
// already says "Support:"). The internal/macro banned-word rule guards the
// site frontend's generated_idea enum — none of these strings ship there.

import (
	"context"
	"fmt"
	"math"
	"strings"
)

// Macro asset-view keys accepted by /agents/macro?asset= (strict by design —
// these are lamp re-framings, not candle assets from the trading registry).
const (
	macroAssetBTC  = "btc"
	macroAssetGold = "gold"
)

// macroAssetViews lists the accepted ?asset= values for /agents/macro, in
// listing order.
var macroAssetViews = []string{macroAssetBTC, macroAssetGold}

// Gold-view lamp statuses.
const (
	goldSupport  = "support"
	goldPressure = "pressure"
	goldNeutral  = "neutral"
)

// Gold composite weights (sum 100) and the voter minimum — see the file-top
// formula notes.
const (
	goldWeightDXY   = 40.0
	goldWeightRates = 25.0
	goldWeightVIX   = 25.0
	goldWeightSPX   = 10.0
	goldMinVoters   = 3

	goldPressureBelow = 35 // composite < 35 → PRESSURE
	goldSupportAbove  = 65 // composite > 65 → SUPPORT
)

// goldLampView maps one lamp's crypto status to gold's perspective. "" when
// there is nothing to re-read: unknown crypto status, an unknown key, or the
// gold lamp itself (excluded from its own outlook).
func goldLampView(key, cryptoStatus string) string {
	if cryptoStatus == "" {
		return ""
	}
	switch key {
	case "dxy", "rates": // same sign as crypto: both dislike a firm dollar / rising yields
		switch cryptoStatus {
		case "tailwind":
			return goldSupport
		case "headwind":
			return goldPressure
		default:
			return goldNeutral
		}
	case "vix", "spx": // inverted: fear and falling equities feed the haven bid
		switch cryptoStatus {
		case "tailwind":
			return goldPressure
		case "headwind":
			return goldSupport
		default:
			return goldNeutral
		}
	default: // "gold" and anything unknown
		return ""
	}
}

// goldWeightOf returns the gold-composite weight for a lamp key (0 = non-voter).
func goldWeightOf(key string) float64 {
	switch key {
	case "dxy":
		return goldWeightDXY
	case "rates":
		return goldWeightRates
	case "vix":
		return goldWeightVIX
	case "spx":
		return goldWeightSPX
	default:
		return 0
	}
}

// goldViewScore folds the voting lamps into a 0..100 gold composite
// (support → +w, neutral → +w/2, pressure → +0, renormalised over the voters).
// nil under goldMinVoters voters — no verdict off half the map.
func goldViewScore(lamps []MacroLamp) *int {
	var scoreSum, weightSum float64
	voters := 0
	for _, l := range lamps {
		view := goldLampView(l.Key, l.Status)
		if view == "" {
			continue
		}
		w := goldWeightOf(l.Key)
		if w == 0 {
			continue
		}
		voters++
		weightSum += w
		switch view {
		case goldSupport:
			scoreSum += w
		case goldNeutral:
			scoreSum += w / 2
		}
	}
	if voters < goldMinVoters || weightSum == 0 {
		return nil
	}
	// Multiply before dividing: 57.5/100*100 drifts to 57.4999…, while
	// 57.5*100/100 stays exactly 57.5 — the rounded composite must not lose a
	// point to float ordering.
	v := clampInt(int(math.Round(scoreSum*100/weightSum)), 0, 100)
	return &v
}

// goldViewCounts tallies the voting lamps per gold direction.
func goldViewCounts(lamps []MacroLamp) (sup, prs, neu int) {
	for _, l := range lamps {
		switch goldLampView(l.Key, l.Status) {
		case goldSupport:
			sup++
		case goldPressure:
			prs++
		case goldNeutral:
			neu++
		}
	}
	return sup, prs, neu
}

// goldLampReason words WHY a lamp leans the way it does for gold — the
// documented mapping, verbatim on the card so the formula is visible.
func goldLampReason(key, view string) string {
	switch key + "|" + view {
	case "dxy|" + goldSupport:
		return "softer dollar lifts gold"
	case "dxy|" + goldPressure:
		return "firmer dollar weighs on gold"
	case "rates|" + goldSupport:
		return "falling yields favor holding gold"
	case "rates|" + goldPressure:
		return "rising yields raise the cost of holding gold"
	case "vix|" + goldSupport:
		return "fear = safe-haven bid"
	case "vix|" + goldPressure:
		return "calm tape unwinds the haven bid"
	case "spx|" + goldSupport:
		return "flight to safety"
	case "spx|" + goldPressure:
		return "risk appetite pulls money from havens"
	default:
		return ""
	}
}

// lampValueText renders "Label value" with the session delta when known:
// "Dollar (DXY) 98.94, session -0.21%".
func lampValueText(l MacroLamp) string {
	if l.Value == nil {
		return l.Label + " — no data"
	}
	s := l.Label + " " + trimFloat(*l.Value)
	if l.DeltaPct != nil {
		s += fmt.Sprintf(", session %+.2f%%", *l.DeltaPct)
	}
	return s
}

// effectiveMacroRegime applies the shared honesty rules to a backend macro
// payload: counts the REAL lamps (value present — per-lamp ok flag with the
// Value fallback for older payloads) and reclassifies an old backend's "mixed"
// with zero real lamps to "unknown" (a MIXED claim needs at least one input).
// Shared by the global card and the per-asset views so every framing sees the
// same truth.
func effectiveMacroRegime(m *MacroResp) (regime string, real int) {
	for _, l := range m.Lamps {
		if l.OK || l.Value != nil {
			real++
		}
	}
	regime = m.Regime
	if regime == "mixed" && real == 0 {
		regime = "unknown"
	}
	return regime, real
}

// macroUnknownAssetCard is the honest zero-input state for an asset view:
// the same admission the global card makes, framed per asset.
func macroUnknownAssetCard(c Card, asset string, tradfinOpen bool) Card {
	label := strings.ToUpper(asset)
	c.Emoji = emojiNeutral
	if tradfinOpen {
		c.Verdict = label + " VIEW: UNKNOWN — no tradfin data right now"
		c.Status = statusNoData
	} else {
		c.Verdict = label + " VIEW: UNKNOWN — market closed, no tradfin data"
		c.Status = statusMarketClosed
	}
	c.Short = "unknown (no data)"
	c.Facts = append(c.Facts, "Lamps: no tradfin data right now")
	return c
}

// MacroAssetCard renders the macro lamps re-framed for one asset
// (macroAssetBTC | macroAssetGold). Honesty carries over: source offline →
// offline card; zero real lamps → an UNKNOWN admission; too few voters for
// the gold read → no_data, never a fabricated verdict.
func (a *Agents) MacroAssetCard(ctx context.Context, asset string) Card {
	assetLabel := strings.ToUpper(asset)
	m, err := a.api.Macro(ctx)
	if err != nil {
		return offlineCard("Macro Agent", "Macro", assetLabel, keyMacro, howTexts[keyMacro])
	}
	c := Card{
		Agent:      "Macro Agent",
		ShortName:  "Macro",
		Asset:      assetLabel,
		Command:    keyMacro,
		HowItWorks: howTexts[keyMacro],
		DataTime:   parseWhen(m.CapturedAt),
	}
	regime, real := effectiveMacroRegime(m)
	if regime == "unknown" || real == 0 {
		c = macroUnknownAssetCard(c, asset, m.TradfinOpen)
		if m.FNG != nil && m.FNG.OK {
			c.Facts = append(c.Facts, fmt.Sprintf("Crypto Fear & Greed: %d — %s", m.FNG.Value, m.FNG.Label))
		}
		return c
	}

	switch asset {
	case macroAssetGold:
		buildGoldView(&c, m)
	default:
		buildBTCView(&c, m, regime)
	}
	return c
}

// buildGoldView fills the card with the gold-framed verdict + per-lamp lines.
func buildGoldView(c *Card, m *MacroResp) {
	score := goldViewScore(m.Lamps)
	sup, prs, neu := goldViewCounts(m.Lamps)

	switch {
	case score == nil:
		voters := sup + prs + neu
		c.Emoji = emojiNeutral
		c.Verdict = fmt.Sprintf("GOLD VIEW: not enough live lamps for a read (%d of 4 voting)", voters)
		c.Short = "gold: no read"
		c.Status = statusNoData
	case *score > goldSupportAbove:
		c.Emoji = emojiBull
		c.Verdict = "GOLD VIEW: SUPPORT — macro lamps lean toward gold"
		c.Short = "gold: support"
	case *score < goldPressureBelow:
		c.Emoji = emojiBear
		c.Verdict = "GOLD VIEW: PRESSURE — macro lamps lean against gold"
		c.Short = "gold: pressure"
	default:
		c.Emoji = emojiNeutral
		c.Verdict = "GOLD VIEW: MIXED — lamps split on gold"
		c.Short = "gold: mixed"
	}

	// Per-lamp lines in render order: value + session delta + the gold
	// direction with its documented reason. The gold lamp itself is a fact
	// line, never a vote.
	for _, l := range m.Lamps {
		if l.Key == "gold" {
			if l.Value != nil {
				line := "Gold itself: " + trimFloat(*l.Value)
				if l.DeltaPct != nil {
					line += fmt.Sprintf(" (%+.2f%% session)", *l.DeltaPct)
				}
				c.Facts = append(c.Facts, line)
			}
			continue
		}
		view := goldLampView(l.Key, l.Status)
		switch {
		case l.Value == nil:
			c.Facts = append(c.Facts, lampValueText(l))
		case view == "":
			c.Facts = append(c.Facts, lampValueText(l)+" → no direction read (no session delta)")
		case view == goldNeutral:
			c.Facts = append(c.Facts, lampValueText(l)+" → neutral")
		default:
			word := view
			if l.Key == "spx" {
				word = "mild " + view // the checklist's "mild" — lowest weight
			}
			c.Facts = append(c.Facts, fmt.Sprintf("%s → %s (%s)", lampValueText(l), word, goldLampReason(l.Key, view)))
		}
	}
	if score != nil {
		c.Facts = append(c.Facts, fmt.Sprintf("Gold composite: %d/100 (support above %d, pressure below %d)",
			*score, goldSupportAbove, goldPressureBelow))
		c.Deviation = clampInt(abs(*score-50)*2, 0, 100)
	}
	c.Facts = append(c.Facts,
		"Rule: dollar/yields up = pressure · fear (VIX>25) / equities down = support · weights DXY 40 / 10Y 25 / VIX 25 / SPX 10")
	if m.FNG != nil && m.FNG.OK {
		c.Facts = append(c.Facts, fmt.Sprintf("Crypto Fear & Greed: %d — %s (context, not a gold input)", m.FNG.Value, m.FNG.Label))
	}
}

// buildBTCView fills the card with the crypto-centric read labeled as the BTC
// asset view — the existing regime logic, per-lamp lines included.
func buildBTCView(c *Card, m *MacroResp, regime string) {
	switch regime {
	case "risk_on":
		c.Emoji = emojiBull
		c.Verdict = "BTC VIEW: TAILWIND — risk-on regime favors crypto"
		c.Short = "btc: tailwind"
	case "risk_off":
		c.Emoji = emojiBear
		c.Verdict = "BTC VIEW: HEADWIND — risk-off regime weighs on crypto"
		c.Short = "btc: headwind"
	default:
		c.Emoji = emojiNeutral
		c.Verdict = "BTC VIEW: MIXED — no single regime in control"
		c.Short = "btc: mixed"
	}
	for _, l := range m.Lamps {
		switch {
		case l.Value == nil:
			c.Facts = append(c.Facts, lampValueText(l))
		case l.Status == "":
			c.Facts = append(c.Facts, lampValueText(l)+" → no direction read (no session delta)")
		default:
			c.Facts = append(c.Facts, lampValueText(l)+" → "+l.Status)
		}
	}
	if m.Composite != nil {
		c.Confidence = m.Composite
		c.Deviation = clampInt(abs(*m.Composite-50)*2, 0, 100)
	}
	if m.FNG != nil && m.FNG.OK {
		c.Facts = append(c.Facts, fmt.Sprintf("Crypto Fear & Greed: %d — %s", m.FNG.Value, m.FNG.Label))
	}
}

// macroViewLines renders the global card's signal-map pair — one BTC-view and
// one gold-view one-liner ("карта сигналов": lamps + both asset verdicts in a
// single card). Empty when there is nothing to claim (callers gate on real>0).
func macroViewLines(regime string, lamps []MacroLamp) []string {
	var out []string
	switch regime {
	case "risk_on":
		out = append(out, "BTC view: tailwind — risk-on regime")
	case "risk_off":
		out = append(out, "BTC view: headwind — risk-off regime")
	case "mixed":
		out = append(out, "BTC view: mixed — no single regime")
	}
	if score := goldViewScore(lamps); score != nil {
		sup, prs, neu := goldViewCounts(lamps)
		word := "mixed"
		if *score > goldSupportAbove {
			word = goldSupport
		} else if *score < goldPressureBelow {
			word = goldPressure
		}
		out = append(out, fmt.Sprintf("Gold view: %s — %d lamps for gold / %d against / %d neutral",
			word, sup, prs, neu))
	} else {
		voters := 0
		for _, l := range lamps {
			if goldLampView(l.Key, l.Status) != "" {
				voters++
			}
		}
		out = append(out, fmt.Sprintf("Gold view: not enough live lamps for a read (%d of 4 voting)", voters))
	}
	return out
}
