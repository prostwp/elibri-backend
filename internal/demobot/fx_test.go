package demobot

import (
	"strings"
	"testing"
	"time"
)

// ── Market hours ─────────────────────────────────────────────────────────────

// Documented approximation: FX closed from Friday 21:00 UTC to Sunday
// 21:00 UTC. 2026-08-12 is a Wednesday, 14th Friday, 15th Saturday,
// 16th Sunday.
func TestIsForexOpen(t *testing.T) {
	utc := func(day, hour, min int) time.Time {
		return time.Date(2026, 8, day, hour, min, 0, 0, time.UTC)
	}
	cases := []struct {
		name string
		at   time.Time
		want bool
	}{
		{"Friday 20:59", utc(14, 20, 59), true},
		{"Friday 21:01", utc(14, 21, 1), false},
		{"Saturday noon", utc(15, 12, 0), false},
		{"Sunday 20:59", utc(16, 20, 59), false},
		{"Sunday 21:01", utc(16, 21, 1), true},
		{"Wednesday", utc(12, 15, 0), true},
	}
	for _, tc := range cases {
		if got := isForexOpen(tc.at); got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
	// Non-UTC input must be normalized before the weekday check:
	// Friday 23:30 Moscow (UTC+3) = Friday 20:30 UTC → still open.
	msk := time.FixedZone("MSK", 3*3600)
	if !isForexOpen(time.Date(2026, 8, 14, 23, 30, 0, 0, msk)) {
		t.Error("Friday 23:30 MSK is 20:30 UTC — must be open")
	}
}

// ── Asset alias mapping ──────────────────────────────────────────────────────

func TestResolveAsset(t *testing.T) {
	cases := []struct {
		arg     string
		display string
		source  string
		symbol  string
	}{
		{"", "BTC", srcBinance, "BTCUSDT"},
		{"btc", "BTC", srcBinance, "BTCUSDT"},
		{"BTC", "BTC", srcBinance, "BTCUSDT"},
		{"eth", "ETH", srcBinance, "ETHUSDT"},
		{"eurusd", "EURUSD", srcYahoo, "EURUSD=X"},
		{"EURUSD", "EURUSD", srcYahoo, "EURUSD=X"},
		{"gbpusd", "GBPUSD", srcYahoo, "GBPUSD=X"},
		{"usdjpy", "USDJPY", srcYahoo, "USDJPY=X"},
		{"xau", "XAUUSD", srcYahoo, "XAUUSD=X"},
		{"gold", "XAUUSD", srcYahoo, "XAUUSD=X"},
		{"xauusd", "XAUUSD", srcYahoo, "XAUUSD=X"},
	}
	for _, tc := range cases {
		spec, err := resolveAsset(tc.arg)
		if err != nil {
			t.Errorf("resolveAsset(%q): unexpected error %v", tc.arg, err)
			continue
		}
		if spec.Display != tc.display || spec.Source != tc.source || spec.Symbol != tc.symbol {
			t.Errorf("resolveAsset(%q): got %+v, want %s/%s/%s", tc.arg, spec, tc.display, tc.source, tc.symbol)
		}
	}
	// Gold must carry the GC=F fallback (XAUUSD=X is dead on Yahoo right now).
	if spec, _ := resolveAsset("gold"); spec.Fallback != "GC=F" {
		t.Errorf("gold fallback: got %q, want GC=F", spec.Fallback)
	}
	if _, err := resolveAsset("doge"); err == nil {
		t.Error("unknown asset must error")
	} else if !strings.Contains(err.Error(), "eurusd") {
		t.Errorf("error should list known assets, got: %v", err)
	}
}

// ── /fx overview card goldens ────────────────────────────────────────────────

func mixedFXReads() []fxRead {
	return []fxRead{
		{Pair: "EURUSD", OK: true, Dir: "up", RSI: 58.3, DayChangePct: 0.24, HasDay: true},
		{Pair: "GBPUSD", OK: true, Dir: "down", RSI: 41.2, DayChangePct: -0.31, HasDay: true},
		{Pair: "USDJPY", OK: false},
		{Pair: "XAUUSD", OK: true, Dir: "up", RSI: 66.0, DayChangePct: 1.05, HasDay: true},
	}
}

func TestFXOverviewCardGoldenOpen(t *testing.T) {
	c := fxOverviewCard(mixedFXReads(), true, goldenTime)
	want := "⚪ <b>FX Agent</b>\n" +
		"<b>1h trend: 2 up · 1 down · 1 no data</b>\n" +
		"• 🟢 EURUSD: up · RSI 58.3 · +0.24% 24h\n" +
		"• 🔴 GBPUSD: down · RSI 41.2 · -0.31% 24h\n" +
		"• ⚪ USDJPY: data unavailable\n" +
		"• 🟢 XAUUSD: up · RSI 66.0 · +1.05% 24h\n" +
		"\n<i>Analytics, not financial advice · AlphaVizor · 2026-08-18 06:00 UTC · data: Yahoo Finance</i>"
	if got := c.RenderHTML(); got != want {
		t.Fatalf("fx open golden mismatch:\ngot:\n%s\nwant:\n%s", got, want)
	}
}

func TestFXOverviewCardGoldenClosed(t *testing.T) {
	c := fxOverviewCard(mixedFXReads(), false, goldenTime)
	got := c.RenderHTML()
	wantBanner := "• ⏸ Forex market closed (weekend) — data as of Friday close\n"
	if !strings.Contains(got, wantBanner) {
		t.Fatalf("closed card must carry the banner line, got:\n%s", got)
	}
	// Banner must be the FIRST fact — prominent, before any pair line.
	bannerIdx := strings.Index(got, "⏸")
	firstPairIdx := strings.Index(got, "EURUSD")
	if bannerIdx < 0 || firstPairIdx < 0 || bannerIdx > firstPairIdx {
		t.Fatalf("banner must precede pair lines:\n%s", got)
	}
	// Levels/trend still render — closed market hides nothing.
	for _, want := range []string{"EURUSD: up", "GBPUSD: down", "USDJPY: data unavailable"} {
		if !strings.Contains(got, want) {
			t.Errorf("closed card missing %q:\n%s", want, got)
		}
	}
}

func TestFXOverviewCardAllDown(t *testing.T) {
	reads := []fxRead{
		{Pair: "EURUSD", OK: true, Dir: "down", RSI: 30, DayChangePct: -1.2, HasDay: true},
		{Pair: "GBPUSD", OK: true, Dir: "down", RSI: 28, DayChangePct: -0.9, HasDay: true},
	}
	c := fxOverviewCard(reads, true, goldenTime)
	if !strings.Contains(c.Verdict, "2 down") || strings.Contains(c.Verdict, "up") {
		t.Errorf("all-down verdict wrong: %q", c.Verdict)
	}
	if c.Short != "2 down" {
		t.Errorf("short: got %q, want \"2 down\"", c.Short)
	}
}

// Day-change omitted when the reference bar is missing.
func TestFXLineWithoutDayChange(t *testing.T) {
	line := fxLine(fxRead{Pair: "EURUSD", OK: true, Dir: "flat", RSI: 50.0})
	want := "⚪ EURUSD: flat · RSI 50.0"
	if line != want {
		t.Errorf("fxLine no-day: got %q, want %q", line, want)
	}
}

// Item 6: too little history is stated explicitly, never rendered as a
// confident flat/neutral read.
func TestFXLineInsufficientHistory(t *testing.T) {
	line := fxLine(fxRead{Pair: "EURUSD", Insufficient: true})
	want := "⚪ EURUSD: insufficient history"
	if line != want {
		t.Errorf("fxLine insufficient: got %q, want %q", line, want)
	}
}
