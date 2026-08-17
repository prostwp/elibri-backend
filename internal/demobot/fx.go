package demobot

import (
	"fmt"
	"strings"
	"time"
)

// ── Asset registry ───────────────────────────────────────────────────────────

const (
	srcBinance = "binance"
	srcYahoo   = "yahoo"
)

// assetSpec describes one tradeable asset the analysis commands accept.
type assetSpec struct {
	Display  string // card label: "BTC", "EURUSD"
	Source   string // srcBinance | srcYahoo
	Symbol   string // exchange-native symbol: "BTCUSDT", "EURUSD=X"
	Fallback string // secondary Yahoo symbol tried when the primary fails
	Interval string // candle interval the indicator math runs on
}

var btcSpec = assetSpec{Display: "BTC", Source: srcBinance, Symbol: "BTCUSDT", Interval: "4h"}

// xauSpec: XAUUSD=X is currently dead on Yahoo ("Not Found", verified live
// 2026-08-18) — GC=F (COMEX gold futures) is the working fallback and in
// practice the actual gold source.
var xauSpec = assetSpec{Display: "XAUUSD", Source: srcYahoo, Symbol: "XAUUSD=X", Fallback: "GC=F", Interval: "1h"}

var assetTable = map[string]assetSpec{
	"btc":    btcSpec,
	"eth":    {Display: "ETH", Source: srcBinance, Symbol: "ETHUSDT", Interval: "4h"},
	"eurusd": {Display: "EURUSD", Source: srcYahoo, Symbol: "EURUSD=X", Interval: "1h"},
	"gbpusd": {Display: "GBPUSD", Source: srcYahoo, Symbol: "GBPUSD=X", Interval: "1h"},
	"usdjpy": {Display: "USDJPY", Source: srcYahoo, Symbol: "USDJPY=X", Interval: "1h"},
	"xauusd": xauSpec,
}

var assetAliases = map[string]string{
	"xau":     "xauusd",
	"gold":    "xauusd",
	"bitcoin": "btc",
}

// fxPairs is the /fx overview set, in render order.
var fxPairs = []string{"eurusd", "gbpusd", "usdjpy", "xauusd"}

// resolveAsset maps a user-typed argument to an assetSpec. Empty argument =
// BTC, matching the original command behavior.
func resolveAsset(arg string) (assetSpec, error) {
	if strings.TrimSpace(arg) == "" {
		return btcSpec, nil
	}
	key := strings.ToLower(strings.TrimSpace(arg))
	if canonical, ok := assetAliases[key]; ok {
		key = canonical
	}
	if spec, ok := assetTable[key]; ok {
		return spec, nil
	}
	return assetSpec{}, fmt.Errorf("unknown asset %q — try: btc, eth, eurusd, gbpusd, usdjpy, xau/gold", arg)
}

// ── Market hours ─────────────────────────────────────────────────────────────

// isForexOpen approximates the FX trading week: closed from Friday 21:00 UTC
// to Sunday 21:00 UTC. Real venue boundaries drift ±1h with US/AU daylight
// saving; the fixed-UTC window is a documented simplification for the demo
// bot, not an execution-grade calendar.
func isForexOpen(now time.Time) bool {
	t := now.UTC()
	switch t.Weekday() {
	case time.Saturday:
		return false
	case time.Friday:
		return t.Hour() < 21
	case time.Sunday:
		return t.Hour() >= 21
	default:
		return true
	}
}

const fxClosedBanner = "⏸ Forex market closed (weekend) — data as of Friday close"

const fxOfflineVerdict = "FX data source unavailable right now"

// ── /fx overview rendering ───────────────────────────────────────────────────

// fxRead is one pair's computed snapshot for the overview.
type fxRead struct {
	Pair         string
	OK           bool
	Insufficient bool   // fetched fine but too little history for EMA200
	Dir          string // up | down | flat (EMA50 vs EMA200)
	RSI          float64
	DayChangePct float64
	HasDay       bool      // false when no bar ~24h back exists to diff against
	CloseAt      time.Time // close time of the last CLOSED bar used
}

// fxLine renders one overview line: "🟢 EURUSD: up · RSI 58.3 · +0.24% 24h".
func fxLine(r fxRead) string {
	if r.Insufficient {
		return emojiNeutral + " " + r.Pair + ": insufficient history"
	}
	if !r.OK {
		return emojiNeutral + " " + r.Pair + ": data unavailable"
	}
	emoji := emojiNeutral
	switch r.Dir {
	case "up":
		emoji = emojiBull
	case "down":
		emoji = emojiBear
	}
	line := fmt.Sprintf("%s %s: %s · RSI %.1f", emoji, r.Pair, r.Dir, r.RSI)
	if r.HasDay {
		line += fmt.Sprintf(" · %+.2f%% 24h", r.DayChangePct)
	}
	return line
}

// fxOverviewCard assembles the /fx card from computed reads (pure —
// golden-tested without network). When the market is closed the banner is
// the first, most prominent fact; data still renders underneath.
func fxOverviewCard(reads []fxRead, open bool, dataTime time.Time) Card {
	var up, down, flat, dead int
	for _, r := range reads {
		switch {
		case !r.OK:
			dead++
		case r.Dir == "up":
			up++
		case r.Dir == "down":
			down++
		default:
			flat++
		}
	}
	var parts, shortParts []string
	add := func(n int, word string, inShort bool) {
		if n == 0 {
			return
		}
		p := fmt.Sprintf("%d %s", n, word)
		parts = append(parts, p)
		if inShort {
			shortParts = append(shortParts, p)
		}
	}
	add(up, "up", true)
	add(down, "down", true)
	add(flat, "flat", true)
	add(dead, "no data", false)

	verdict := "1h trend: " + strings.Join(parts, " · ")
	short := strings.Join(shortParts, " · ")
	if short == "" {
		short = "no data"
	}

	c := Card{
		Emoji:      emojiNeutral,
		Agent:      "FX Agent",
		ShortName:  "FX",
		Command:    keyFX,
		HowItWorks: howTexts[keyFX],
		DataTime:   dataTime,
		SourceNote: "data: Yahoo Finance",
		Verdict:    verdict,
		Short:      short,
	}
	if !open {
		c.Facts = append(c.Facts, fxClosedBanner)
	}
	for _, r := range reads {
		c.Facts = append(c.Facts, fxLine(r))
	}
	return c
}
