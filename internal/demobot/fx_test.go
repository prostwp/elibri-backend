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
		// Gold's human label names the CONTRACT: the feed serves COMEX
		// futures, not spot, so a card headed "XAUUSD" would be off by the
		// basis on every level it prints. The machine key stays "XAUUSD" —
		// asserted separately below.
		{"xau", "GOLD · COMEX GC=F", srcYahoo, "XAUUSD=X"},
		{"gold", "GOLD · COMEX GC=F", srcYahoo, "XAUUSD=X"},
		{"xauusd", "GOLD · COMEX GC=F", srcYahoo, "XAUUSD=X"},
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
	// The documented HTTP contract must survive the honest relabelling: every
	// asset serves a plain ticker as its machine key, gold included.
	for _, arg := range []string{"xau", "gold", "xauusd"} {
		if spec, _ := resolveAsset(arg); spec.Key != "XAUUSD" {
			t.Errorf("resolveAsset(%q).Key = %q, want XAUUSD — integrations branch on it", arg, spec.Key)
		}
	}
	for _, arg := range []string{"btc", "eurusd"} {
		spec, _ := resolveAsset(arg)
		if got := (Card{Asset: spec.Display, AssetKey: spec.Key}).assetKey(); got != spec.Display {
			t.Errorf("%s: machine key %q must equal the label %q when they do not differ", arg, got, spec.Display)
		}
	}
	if _, err := resolveAsset("doge"); err == nil {
		t.Error("unknown asset must error")
	} else if !strings.Contains(err.Error(), "eurusd") {
		t.Errorf("error should list known assets, got: %v", err)
	}
}

// ── /fx overview card goldens ────────────────────────────────────────────────

// Stage-1 goldens live in fx_readable_test.go. These keep the older
// guarantees in the new wording.

// Closed market: the banner is the first fact, before any pair line, and the
// data still renders underneath — a closed market hides nothing.
func TestFXOverviewCardClosedBannerFirst(t *testing.T) {
	c := fxCardFromReads(liveFXReads(), fxWeekend)
	got := c.RenderHTML()
	if !strings.Contains(got, "• ⏸ Forex market closed (weekend) — data as of Friday close\n") {
		t.Fatalf("closed card must carry the banner line, got:\n%s", got)
	}
	bannerIdx := strings.Index(got, "⏸")
	firstPairIdx := strings.Index(got, "EURUSD")
	if bannerIdx < 0 || firstPairIdx < 0 || bannerIdx > firstPairIdx {
		t.Fatalf("banner must precede pair lines:\n%s", got)
	}
	for _, want := range []string{"EURUSD · 1.1543 · -0.05% · 43% · below · 40.8", "GOLD · 4320.7 · -0.69%"} {
		if !strings.Contains(got, want) {
			t.Errorf("closed card missing %q:\n%s", want, got)
		}
	}
}

// Every pair on the same side of its EMA200 is still no verdict: the header
// counts coverage, never directions.
func TestFXOverviewCardAllBelowIsNoVerdict(t *testing.T) {
	reads := []fxRead{
		fxOK("eurusd", 1.1, "down", 30, -1.2, 0.1, fxAt(15, 8)),
		fxOK("gbpusd", 1.3, "down", 28, -0.9, 0.1, fxAt(15, 8)),
	}
	c := fxCardFromReads(reads, fxNow)
	if c.Verdict != "FX overview · 1h · 2 pairs read" || c.Short != "2 pairs read" || c.Emoji != emojiNeutral {
		t.Errorf("all-below: verdict %q short %q emoji %q", c.Verdict, c.Short, c.Emoji)
	}
}

// Day change omitted when no reference bar exists — the indicator line keeps
// its labels either way.
func TestFXLineWithoutDayChange(t *testing.T) {
	r := fxOK("eurusd", 1.1, "flat", 50.0, 0, 0.5, fxAt(15, 8))
	r.HasDay = false
	if got := fxMarketLine(r, fxNow); got != "EURUSD 1.1000 · 50% of the 24h range" {
		t.Errorf("no-day line: %q", got)
	}
	if got := fxContextLine(r); got != "EURUSD: EMA50 equal to EMA200 · RSI(1h) 50.0 · last bar Sep 15 08:00 UTC" {
		t.Errorf("flat context line: %q", got)
	}
}

// Too little history is stated explicitly, never rendered as a confident
// flat/neutral read.
func TestFXLineInsufficientHistory(t *testing.T) {
	r := fxRead{Pair: "EURUSD", spec: assetTable["eurusd"], Insufficient: true}
	if got := fxMarketLine(r, fxNow); got != "EURUSD: insufficient history for EMA50/EMA200/RSI(14) on 1h bars" {
		t.Errorf("insufficient: %q", got)
	}
}
