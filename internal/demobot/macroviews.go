package demobot

// macroviews.go — per-asset framings of the macro lamps (checklist B2):
// /agents/macro?asset=btc|gold. The WORDS live in macro_text.go; this file
// holds the gold model's rule and the card entry points.
//
// The backend lamps carry tailwind|neutral|headwind under the risk-appetite
// rule (internal/macro compute.go LampStatus: any favourable session move
// counts, the unfavourable reading needs a move beyond 0.5%; VIX by level).
// The asset views re-read those SAME statuses instead of re-deriving
// thresholds — one mapping, two framings.
//
// GOLD model (experimental — fixed weights, thresholds borrowed from the risk
// rule, not validated on gold's own history). Crypto status → gold
// contribution per lamp:
//
//	dxy   tailwind (fell)       → positive | headwind (rose >0.5%)  → negative
//	rates tailwind (fell)       → positive | headwind (rose >0.5%)  → negative
//	vix   tailwind (<18)        → negative | headwind (>25)         → positive   (inverted)
//	spx   tailwind (rose)       → negative | headwind (fell >0.5%)  → positive   (inverted)
//	gold  — excluded: the gold lamp IS the asset; its own session move is
//	        reported as a fact, never as an input to its own backdrop.
//
// These are the model's assumptions, not observed causes: the card words each
// lamp as a contribution "in this model", never as a reason gold moves.
// Internally the contributions keep their historical names support /
// pressure / neutral (goldSupport …) because the Gold Agent branches on
// Card.State with them (gold.go).
//
// Gold score: positive → +weight, neutral → +weight/2, negative → +0 over the
// VOTING lamps (status known, gold excluded), renormalised. Weights DXY 40 /
// rates 25 / VIX 25 / SPX 10. >65 positive, <35 negative, else mixed; fewer
// than 3 voters → no read.
//
// BTC view = the risk-appetite regime itself, labeled as BTC's macro
// backdrop; BTC direction is never inferred from it.

import (
	"context"
	"fmt"
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

// Gold-model contribution states (Card.State values of the gold view).
const (
	goldSupport  = "support"
	goldPressure = "pressure"
	goldNeutral  = "neutral"
)

// Gold model weights (sum 100) and the voter minimum — see the file-top notes.
const (
	goldWeightDXY   = 40.0
	goldWeightRates = 25.0
	goldWeightVIX   = 25.0
	goldWeightSPX   = 10.0
	goldMinVoters   = 3

	goldPressureBelow = 35 // gold score < 35 → negative
	goldSupportAbove  = 65 // gold score > 65 → positive
)

// goldLampView maps one lamp's crypto status to the gold model. "" when there
// is nothing to re-read: unknown crypto status, an unknown key, or the gold
// lamp itself (excluded from its own backdrop).
func goldLampView(key, cryptoStatus string) string {
	if cryptoStatus == "" {
		return ""
	}
	switch key {
	case "dxy", "rates": // same sign as the risk rule
		switch cryptoStatus {
		case "tailwind":
			return goldSupport
		case "headwind":
			return goldPressure
		default:
			return goldNeutral
		}
	case "vix", "spx": // inverted in this model
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

// goldWeightOf returns the gold-model weight for a lamp key (0 = non-voter).
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

// goldViewScore is the 0..100 gold score (nil under goldMinVoters voters).
// The arithmetic is goldModel.read: multiply before dividing — 57.5/100*100
// drifts to 57.4999…, while 57.5*100/100 stays exactly 57.5, so the rounded
// score never loses a point to float ordering.
func goldViewScore(lamps []MacroLamp) *int { return goldModel.read(lamps).score }

// effectiveMacroRegime applies the shared honesty rules to a backend macro
// payload: counts the REAL lamps (value present — per-lamp ok flag with the
// Value fallback for older payloads) and reclassifies any regime read off
// zero real lamps to "unknown": a regime is a knowledge claim and needs at
// least one input (an old backend served "mixed" with all-null lamps). Shared
// by the global card and the per-asset views so every framing sees the same
// truth.
func effectiveMacroRegime(m *MacroResp) (regime string, real int) {
	for _, l := range m.Lamps {
		if lampLive(l) {
			real++
		}
	}
	regime = m.Regime
	if real == 0 {
		regime = "unknown"
	}
	return regime, real
}

// macroUnknownFacts are the facts of a zero-input macro card: the absence,
// the last known data date, and the crypto side that is still live.
func macroUnknownFacts(m *MacroResp, scoreName string) []string {
	facts := []string{"Lamps: " + macroNoDataNote(m.TradfinOpen)}
	if l := macroLastDataLine(m); l != "" {
		facts = append(facts, l)
	}
	if l := fngLine(m.FNG, parseWhen(m.CapturedAt), scoreName); l != "" {
		facts = append(facts, l)
	}
	return facts
}

// macroUnknownStatus: outside the clock-based week the machine reason stays
// market_closed (the JSON enum), inside it no_data.
func macroUnknownStatus(tradfinOpen bool) cardStatus {
	if tradfinOpen {
		return statusNoData
	}
	return statusMarketClosed
}

// MacroAssetCard renders the macro lamps framed for one asset
// (macroAssetBTC | macroAssetGold). Honesty carries over: source offline →
// offline card; zero real lamps → an UNKNOWN admission; too few voters for
// the gold read → no_data, never a fabricated reading.
func (a *Agents) MacroAssetCard(ctx context.Context, asset string) Card {
	assetLabel := strings.ToUpper(asset)
	m, err := a.api.Macro(ctx)
	if err != nil {
		return offlineCard("Macro Agent", "Macro", assetLabel, keyMacro, howTexts[keyMacro])
	}
	return macroAssetCardFrom(m, asset)
}

// macroAssetCardFrom is MacroAssetCard's pure half (no network, no clock).
func macroAssetCardFrom(m *MacroResp, asset string) Card {
	assetLabel := strings.ToUpper(asset)
	c := Card{
		Agent:      "Macro Agent",
		ShortName:  "Macro",
		Asset:      assetLabel,
		Command:    keyMacro,
		HowItWorks: howTexts[keyMacro],
		DataTime:   parseWhen(m.CapturedAt),
	}
	regime, _ := effectiveMacroRegime(m)
	scoreName := riskModel.scoreName
	if asset == macroAssetGold {
		scoreName = goldModel.scoreName
	}
	if regime == "unknown" {
		c.Emoji = emojiNeutral
		c.Verdict = fmt.Sprintf("%s MACRO BACKDROP: UNKNOWN — %s", assetLabel, macroNoDataNote(m.TradfinOpen))
		c.Status = macroUnknownStatus(m.TradfinOpen)
		c.Short = "unknown (no data)"
		c.Facts = macroUnknownFacts(m, scoreName)
		return c
	}

	v := newMacroView(m, regime)
	if asset == macroAssetGold {
		v.fillGold(&c)
	} else {
		v.fillBTC(&c)
	}
	return c
}
