package demobot

// macro_text_boundary_test.go — the Macro cards' hard edges, swept over whole
// payloads through the pure card builders (macroCardFrom, macroAssetCardFrom):
// every lamp missing / without a session change / at an extreme value or
// change / at a threshold edge, in every combination; weekend and open week;
// a stale Fear & Greed with the widest label; mixed and unknown sources;
// payloads whose composite matches the rule and payloads skewed from it.
// Every visible line — header, verdict, short, facts, each blocks field, the
// footer — must fit macroFactMaxRunes, print no non-finite or exponent
// number, and carry none of the banned causal / forecast / "24h" wording.

import (
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/prostwp/elibri-backend/internal/macro"
)

// macroVisibleLines is every line a reader can see on one macro card.
func macroVisibleLines(c Card) []string {
	head := c.Emoji + " " + c.Agent
	if c.Asset != "" {
		head += " · " + c.Asset
	}
	html := c.RenderHTML()
	footer := htmlToPlain(html[strings.LastIndex(html, "\n")+1:])
	out := []string{head, footer}
	for _, s := range macroTexts(c) {
		// The how-it-works text is the [ℹ️ How it works] callback alert, not a
		// card line: its budget is Telegram's 200-char cap (checked below).
		if s != c.HowItWorks {
			out = append(out, s)
		}
	}
	return out
}

func checkMacroBudget(t *testing.T, label string, c Card) bool {
	t.Helper()
	ok := true
	for _, s := range macroVisibleLines(c) {
		if n := utf8.RuneCountInString(s); n > macroFactMaxRunes {
			t.Errorf("%s: %d chars (max %d): %q", label, n, macroFactMaxRunes, s)
			ok = false
		}
		if badNumberText.MatchString(s) {
			t.Errorf("%s: non-finite or exponent number: %q", label, s)
			ok = false
		}
	}
	if n := utf8.RuneCountInString(c.HowItWorks); n > 200 {
		t.Errorf("%s: how-it-works %d chars (max 200)", label, n)
		ok = false
	}
	return ok
}

// lampVariants are the worst cases per lamp. Statuses come from the real rule
// (macro.LampStatus), so every payload is one the backend could serve.
func lampVariants(key string, day int, source string) []MacroLamp {
	stamp := time.Date(2026, 9, 10+day, 13, 30, 0, 0, time.UTC).Format(time.RFC3339)
	mk := func(v float64, d *float64) MacroLamp {
		l := MacroLamp{Key: key, Label: key, Value: &v, OK: true, DeltaPct: d, AsOf: stamp, Source: source}
		switch {
		case key == "vix":
			l.Status = macro.LampStatus(key, v, 0)
		case d != nil:
			l.Status = macro.LampStatus(key, v, *d)
		}
		return l
	}
	f := func(v float64) *float64 { return &v }
	missing := MacroLamp{Key: key, Label: key, AsOf: "2026-08-01T20:00:00Z"}
	if key == "vix" {
		return []MacroLamp{missing, mk(17.999, nil), mk(18, f(0)), mk(25.00001, f(-9999)), mk(999999, f(99999.99))}
	}
	edge := 0.49999
	if key == "spx" {
		edge = -0.49999
	}
	return []MacroLamp{missing, mk(99999999, nil), mk(99999999, f(-99999.99)), mk(0.0001, f(99999.99)), mk(9999999.99, f(edge))}
}

func TestMacroCardBoundaryEveryPath(t *testing.T) {
	keys := []string{"dxy", "rates", "vix", "spx", "gold"}
	sources := []string{"yahoo", "stooq", "yahoo", "someverylongprovidername", "stooq"}
	variants := make([][]MacroLamp, len(keys))
	for i, k := range keys {
		variants[i] = lampVariants(k, i, sources[i])
	}
	captured := time.Date(2026, 9, 15, 4, 44, 32, 0, time.UTC)
	fngs := []*FearGreedCheck{
		{Value: 100, Label: "Extreme Greed, very long label", OK: true, AsOf: "2012-01-01T00:00:00Z"}, // stale, >999d
		{Value: 100, Label: "Extreme Greed, very long label", OK: true},                               // time unknown
		{Value: 5, Label: "Extreme Fear", OK: true, AsOf: captured.Add(-2 * time.Hour).Format(time.RFC3339)},
		nil,
	}
	failures := 0
	n := 0
	idx := make([]int, len(keys))
	for {
		lamps := make([]MacroLamp, len(keys))
		for i := range keys {
			lamps[i] = variants[i][idx[i]]
		}
		comp := macro.Composite(toMacroLamps(lamps))
		regime := macro.ClassifyRegime(comp, toMacroLamps(lamps))
		if n%7 == 3 && comp != nil { // version skew: served score ≠ the statuses' score
			skew := (*comp + 37) % 101
			comp = &skew
		}
		m := &MacroResp{
			Regime: regime, Composite: comp, TradfinOpen: n%2 == 0,
			TradfinAsOf: "2026-09-14T13:30:00Z", CapturedAt: captured.Format(time.RFC3339),
			Lamps: lamps, FNG: fngs[n%len(fngs)],
			GeneratedIdea: "The dollar is softening — a backdrop that tends to favor crypto.",
		}
		g, _ := macroCardFrom(m)
		for label, c := range map[string]Card{
			"global": g, "btc": macroAssetCardFrom(m, macroAssetBTC), "gold": macroAssetCardFrom(m, macroAssetGold),
		} {
			if !checkMacroBudget(t, label, c) {
				failures++
			}
			checkMacroWording(t, label, c)
		}
		if failures > 20 {
			t.Fatal("too many failures, stopping")
		}
		n++
		// next combination
		i := 0
		for ; i < len(idx); i++ {
			idx[i]++
			if idx[i] < len(variants[i]) {
				break
			}
			idx[i] = 0
		}
		if i == len(idx) {
			break
		}
	}
	if n != 3125 {
		t.Errorf("swept %d payloads, want 3125", n)
	}
	// And the degraded paths.
	for _, open := range []bool{true, false} {
		m := &MacroResp{Regime: "unknown", TradfinOpen: open, TradfinAsOf: "2026-08-14T20:55:00Z",
			CapturedAt: captured.Format(time.RFC3339), Lamps: []MacroLamp{{Key: "dxy"}}, FNG: fngs[0]}
		g, _ := macroCardFrom(m)
		checkMacroBudget(t, "unknown global", g)
		checkMacroBudget(t, "unknown gold", macroAssetCardFrom(m, macroAssetGold))
		checkMacroBudget(t, "unknown btc", macroAssetCardFrom(m, macroAssetBTC))
	}
	checkMacroBudget(t, "offline", offlineCard("Macro Agent", "Macro", "GOLD", keyMacro, howTexts[keyMacro]))
}
