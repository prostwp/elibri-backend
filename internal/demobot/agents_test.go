package demobot

import (
	"strings"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// classifyTrend is a pure state machine — every branch is pinned here.
func TestClassifyTrend(t *testing.T) {
	cases := []struct {
		name  string
		adx   float64
		ema50 float64
		e200  float64
		close float64
		want  string
	}{
		{"flat", 15, 105, 100, 106, trendFlat},
		{"grey zone", 22, 105, 100, 106, trendGrey},
		{"confirmed up", 30, 105, 100, 106, trendUp},
		{"confirmed down", 30, 95, 100, 94, trendDown},
		{"conflict: bullish cross, price below", 30, 105, 100, 104, trendConflict},
		{"conflict: bearish cross, price above", 30, 95, 100, 96, trendConflict},
		{"boundary adx 20 is grey", 20, 105, 100, 106, trendGrey},
		{"boundary adx 25 confirms", 25, 105, 100, 106, trendUp},
	}
	for _, tc := range cases {
		if got := classifyTrend(tc.adx, tc.ema50, tc.e200, tc.close); got != tc.want {
			t.Errorf("%s: got %q, want %q", tc.name, got, tc.want)
		}
	}
}

// The flat state must carry the exact advisory wording from the spec.
func TestTrendVerdictWording(t *testing.T) {
	v := trendVerdict(trendFlat, 0)
	if !strings.Contains(v, "Flat. Trading not advised") {
		t.Fatalf("flat verdict must contain the advisory, got %q", v)
	}
}

// Position-size calculator: risk math is money — pin it hard. The result is
// pure arithmetic: no direction label exists anymore (batch-2 language rule —
// LONG/SHORT read as a trade suggestion; |entry − stop| math needs none).
func TestRiskCalc(t *testing.T) {
	r, err := calcRisk(10000, 1, 100000, 98000)
	if err != nil {
		t.Fatalf("calcRisk: %v", err)
	}
	if !almostEqual(r.RiskAmount, 100, 1e-9) {
		t.Errorf("risk amount: got %.4f, want 100", r.RiskAmount)
	}
	if !almostEqual(r.Size, 0.05, 1e-9) {
		t.Errorf("size: got %.6f, want 0.05", r.Size)
	}
	if !almostEqual(r.Notional, 5000, 1e-6) {
		t.Errorf("notional: got %.4f, want 5000", r.Notional)
	}

	// Entry below stop: same math, |entry − stop| — no direction involved.
	r, err = calcRisk(5000, 2, 3400, 3500)
	if err != nil {
		t.Fatalf("calcRisk entry-below-stop: %v", err)
	}
	if !almostEqual(r.Size, 1.0, 1e-9) { // 100 risk / 100 per-unit
		t.Errorf("size: got %.6f, want 1.0", r.Size)
	}

	for _, bad := range [][4]float64{
		{0, 1, 100, 90},    // zero balance
		{1000, 0, 100, 90}, // zero risk pct
		{1000, 101, 100, 90},
		{1000, 1, 100, 100}, // stop == entry
		{1000, 1, -5, 90},
	} {
		if _, err := calcRisk(bad[0], bad[1], bad[2], bad[3]); err == nil {
			t.Errorf("calcRisk(%v) should error", bad)
		}
	}
}

// Item 6: cards for short-history data carry an explicit degraded verdict —
// neutral emoji plus text, excluded from the priority rule via Offline.
func TestInsufficientCard(t *testing.T) {
	c := insufficientCard(btcSpec, "Trend Agent", "Trend", keyTrend, "how", "EMA200/ADX(14)")
	if c.Emoji != "⚪" {
		t.Errorf("emoji: got %q, want ⚪", c.Emoji)
	}
	if !strings.Contains(c.Verdict, "Insufficient history") {
		t.Errorf("verdict must say insufficient history, got %q", c.Verdict)
	}
	found := false
	for _, f := range c.Facts {
		if strings.Contains(f, "insufficient history for EMA200/ADX(14)") {
			found = true
		}
	}
	if !found {
		t.Errorf("facts must name the starved indicator: %v", c.Facts)
	}
	if c.Confidence != nil {
		t.Error("no confidence bar on a no-verdict card")
	}
	if !c.Offline {
		t.Error("insufficient cards must not feed the priority rule")
	}
	if c.Status != statusInsufficientHistory {
		t.Error("insufficient cards must carry the machine status for ok/reason")
	}
	// FX flavor carries the source note.
	fx := insufficientCard(xauSpec, "Trend Agent", "Trend", keyTrend, "how", "EMA200")
	if fx.SourceNote == "" {
		t.Error("yahoo-sourced insufficient card keeps its source note")
	}
}

// Item 7: forming bars are dropped before any indicator math — ALL trailing
// bars whose close time is in the future, not just one (Yahoo appends both a
// forming hour bar and a live snapshot row). Card timestamps are the close
// time of the last CLOSED bar actually used.
func TestDropUnclosedBarsAndCloseTime(t *testing.T) {
	// 4h bars opening at 0, 14400, 28800 → closing at 14400, 28800, 43200.
	candles := []types.OHLCVCandle{
		{Time: 0, Close: 1}, {Time: 14400, Close: 2}, {Time: 28800, Close: 3},
	}
	at := func(unix int64) time.Time { return time.Unix(unix, 0).UTC() }

	// Mid-bar: the last bar has not closed yet → dropped.
	if got := dropUnclosedBars(candles, "4h", at(43199)); len(got) != 2 || got[1].Time != 14400 {
		t.Fatalf("mid-bar: got %v, want first two bars", got)
	}
	// Exactly at the close boundary the bar IS closed → kept.
	if got := dropUnclosedBars(candles, "4h", at(43200)); len(got) != 3 {
		t.Fatalf("boundary: got %d bars, want 3", len(got))
	}
	// The Yahoo shape: forming bar + snapshot row → BOTH dropped.
	withSnapshot := append(append([]types.OHLCVCandle{}, candles...),
		types.OHLCVCandle{Time: 30000, Close: 4}) // snapshot stamped mid-bar
	if got := dropUnclosedBars(withSnapshot, "4h", at(30100)); len(got) != 2 {
		t.Fatalf("snapshot shape: got %d bars, want 2 (forming + snapshot dropped)", len(got))
	}
	// Everything unclosed → empty, never a fake bar.
	if got := dropUnclosedBars(candles, "4h", at(14000)); len(got) != 0 {
		t.Fatalf("all-unclosed: got %d bars, want 0", len(got))
	}
	if dropUnclosedBars(nil, "4h", at(0)) != nil {
		t.Error("nil in, nil out")
	}

	// Close time = open time + interval.
	closed := candles[:2]
	if want := at(14400 + 14400); !closeTimeOf(closed, "4h").Equal(want) {
		t.Errorf("closeTimeOf 4h: got %v, want %v", closeTimeOf(closed, "4h"), want)
	}
	if want := at(14400 + 3600); !closeTimeOf(closed, "1h").Equal(want) {
		t.Errorf("closeTimeOf 1h: got %v, want %v", closeTimeOf(closed, "1h"), want)
	}
	if closeTimeOf(nil, "4h").IsZero() {
		t.Error("empty series should fall back to a non-zero time, not zero")
	}
}

// Batch-2 regulatory language: momentum verdicts are analytical READINGS.
// "bullish"/"bearish"/"neutral" — never "buy"/"sell" anywhere in the chain.
func TestMomentumVerdictAnalyticalLanguage(t *testing.T) {
	cases := []struct {
		rsi, hist float64
		want      string
	}{
		{62, 120, "bullish"},
		{55, 0.001, "bullish"},
		{40, -3, "bearish"},
		{45, -0.001, "bearish"},
		{50, 1, "neutral"},  // RSI in the dead zone
		{60, -1, "neutral"}, // MACD sign disagrees
		{40, 2, "neutral"},  // MACD sign disagrees
		{55, 0, "neutral"},  // zero histogram confirms nothing
	}
	for _, tc := range cases {
		if got := momentumVerdict(tc.rsi, tc.hist); got != tc.want {
			t.Errorf("momentumVerdict(%v, %v): got %q, want %q", tc.rsi, tc.hist, got, tc.want)
		}
	}
	// The emoji mapping keys on the new words.
	if momentumEmoji("bullish") != emojiBull || momentumEmoji("bearish") != emojiBear || momentumEmoji("neutral") != emojiNeutral {
		t.Error("momentumEmoji must map bullish/bearish/neutral")
	}
	// Advice words must be dead keys — they map to neutral, never a semaphore.
	if momentumEmoji("buy") != emojiNeutral || momentumEmoji("sell") != emojiNeutral {
		t.Error("legacy buy/sell keys must not light a semaphore")
	}
}

// RiskCard output is pure sizing math: no LONG/SHORT, and the card says so.
func TestRiskCardNoDirectionLanguage(t *testing.T) {
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.RiskCard([]float64{10000, 1, 64000, 62500}, false, nil)
	if !strings.HasPrefix(c.Verdict, "Position size: ") {
		t.Errorf("verdict must lead with the size, got %q", c.Verdict)
	}
	joined := c.Verdict + "|" + strings.Join(c.Facts, "|")
	for _, banned := range []string{"LONG", "SHORT"} {
		if strings.Contains(joined, banned) {
			t.Errorf("direction label %q leaked into the risk card: %s", banned, joined)
		}
	}
	var disclaimer, maxLoss bool
	for _, f := range c.Facts {
		if f == "Position sizing math only — not a trade suggestion." {
			disclaimer = true
		}
		if strings.HasPrefix(f, "Max loss at stop:") {
			maxLoss = true
		}
	}
	if !disclaimer {
		t.Errorf("risk card must carry the sizing-only disclaimer: %v", c.Facts)
	}
	if !maxLoss {
		t.Errorf("risk card must name the max loss: %v", c.Facts)
	}
	// The example form keeps its labeling and the disclaimer.
	ex := ag.RiskCard([]float64{10000, 1, 64000, 62500}, true, nil)
	if !strings.HasPrefix(ex.Verdict, "Example — Position size: ") {
		t.Errorf("example verdict: got %q", ex.Verdict)
	}
}

// invalidationLevel: 1 ATR under the lower EMA — pure math, pinned.
func TestInvalidationLevel(t *testing.T) {
	if got := invalidationLevel(105, 100, 2); !almostEqual(got, 98, 1e-9) {
		t.Errorf("uptrend cluster: got %v, want 98 (min(105,100)-2)", got)
	}
	if got := invalidationLevel(95, 100, 3); !almostEqual(got, 92, 1e-9) {
		t.Errorf("downtrend cluster: got %v, want 92 (min(95,100)-3)", got)
	}
}

// Volatility read thresholds.
func TestVolState(t *testing.T) {
	if got := volState(1.3); got != volExpanding {
		t.Errorf("ratio 1.3: got %q, want expanding", got)
	}
	if got := volState(0.7); got != volCompressed {
		t.Errorf("ratio 0.7: got %q, want compressed", got)
	}
	if got := volState(1.0); got != volNormal {
		t.Errorf("ratio 1.0: got %q, want normal", got)
	}
}
