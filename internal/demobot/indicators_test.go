package demobot

import (
	"math"
	"testing"

	"github.com/prostwp/elibri-backend/pkg/types"
)

func almostEqual(a, b, eps float64) bool { return math.Abs(a-b) <= eps }

// ── RSI ──────────────────────────────────────────────────────────────────────

// Classic Wilder/StockCharts fixture: 15 closes → first RSI(14) ≈ 70.46.
// Hand-checked: gains sum 3.34/14=0.238571, losses sum 1.40/14=0.10,
// RS=2.385714, RSI=100-100/3.385714=70.4642.
func TestRSIWilderFixture(t *testing.T) {
	closes := []float64{
		44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
		45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28,
	}
	got, ok := rsiWilder(closes, 14)
	if !ok {
		t.Fatal("fixture has exactly enough history — ok must be true")
	}
	if !almostEqual(got, 70.46, 0.05) {
		t.Fatalf("rsiWilder fixture: got %.4f, want ≈70.46", got)
	}
}

func TestRSIExtremes(t *testing.T) {
	up := make([]float64, 30)
	down := make([]float64, 30)
	for i := range up {
		up[i] = 100 + float64(i)
		down[i] = 100 - float64(i)
	}
	if got, ok := rsiWilder(up, 14); !ok || !almostEqual(got, 100, 1e-9) {
		t.Errorf("monotonic up: got %.4f ok=%v, want 100/true", got, ok)
	}
	if got, ok := rsiWilder(down, 14); !ok || got > 1 {
		t.Errorf("monotonic down: got %.4f ok=%v, want ≈0/true", got, ok)
	}
	// Not enough data → NOT a neutral-looking sentinel: ok=false so the
	// caller renders "insufficient history", never a confident 50.
	if _, ok := rsiWilder([]float64{1, 2, 3}, 14); ok {
		t.Error("short series: ok must be false (no sentinel 50)")
	}
}

// Every indicator must refuse to fake a value on short history (item 6 of
// the adversarial review): (value, ok) with ok=false, no sentinels.
func TestInsufficientHistoryReported(t *testing.T) {
	short := []float64{1, 2, 3}
	if _, ok := rsiWilder(short, 14); ok {
		t.Error("rsiWilder: short series must report ok=false")
	}
	if _, ok := adxWilder(short, short, short, 14); ok {
		t.Error("adxWilder: short series must report ok=false")
	}
	if _, ok := emaLast(short, 200); ok {
		t.Error("emaLast(200): 3 bars must report ok=false")
	}
	if v, ok := emaLast([]float64{7, 7, 7, 7, 7}, 5); !ok || !almostEqual(v, 7, 1e-9) {
		t.Errorf("emaLast: exactly period bars is enough, got %v ok=%v", v, ok)
	}
	if _, _, _, ok := macdLast(short); ok {
		t.Error("macdLast: short series must report ok=false")
	}
}

// ── EMA ──────────────────────────────────────────────────────────────────────

func TestEMASeries(t *testing.T) {
	// Hand-computed: k=2/3; e0=1, e1=5/3, e2=23/9.
	out := emaSeries([]float64{1, 2, 3}, 2)
	want := []float64{1, 5.0 / 3.0, 23.0 / 9.0}
	for i := range want {
		if !almostEqual(out[i], want[i], 1e-9) {
			t.Fatalf("emaSeries[%d]: got %.6f, want %.6f", i, out[i], want[i])
		}
	}
	// Constant series → EMA constant.
	c := emaSeries([]float64{7, 7, 7, 7, 7}, 3)
	for i, v := range c {
		if !almostEqual(v, 7, 1e-9) {
			t.Fatalf("constant EMA[%d]: got %.6f, want 7", i, v)
		}
	}
	if len(emaSeries(nil, 5)) != 0 {
		t.Fatal("nil input should give empty output")
	}
}

// ── ATR (Wilder) ─────────────────────────────────────────────────────────────

// Hand-computed, period=2:
// TR1=max(0.7, |10.5-9.5|, |9.8-9.5|)=1.0; TR2=max(0.8, 0.6, 0.2)=0.8;
// TR3=max(1.1, 1.0, 0.1)=1.1. Seed=(1.0+0.8)/2=0.9; next=(0.9*1+1.1)/2=1.0.
func TestATRWilderFixture(t *testing.T) {
	h := []float64{10.0, 10.5, 10.8, 11.5}
	l := []float64{9.0, 9.8, 10.0, 10.4}
	c := []float64{9.5, 10.2, 10.5, 11.0}
	series := atrSeriesWilder(h, l, c, 2)
	if len(series) != 4 {
		t.Fatalf("series len: got %d, want 4", len(series))
	}
	if !almostEqual(series[2], 0.9, 1e-9) {
		t.Errorf("seed ATR: got %.6f, want 0.9", series[2])
	}
	if !almostEqual(series[3], 1.0, 1e-9) {
		t.Errorf("smoothed ATR: got %.6f, want 1.0", series[3])
	}
}

// ── ADX (Wilder) ─────────────────────────────────────────────────────────────

func TestADXTrendingVsChoppy(t *testing.T) {
	n := 80
	// Steady uptrend: every bar higher high & higher low → +DM only → ADX high.
	upH := make([]float64, n)
	upL := make([]float64, n)
	upC := make([]float64, n)
	for i := 0; i < n; i++ {
		c := 100 + float64(i)*2
		upC[i] = c
		upH[i] = c + 1
		upL[i] = c - 1
	}
	trending, ok := adxWilder(upH, upL, upC, 14)
	if !ok {
		t.Fatal("80 bars is enough history for ADX(14)")
	}
	if trending < 60 {
		t.Errorf("trending ADX: got %.2f, want >60", trending)
	}

	// Perfect chop: alternating up/down bars → +DM ≈ -DM → ADX low.
	chH := make([]float64, n)
	chL := make([]float64, n)
	chC := make([]float64, n)
	for i := 0; i < n; i++ {
		c := 100.0
		if i%2 == 1 {
			c = 101.0
		}
		chC[i] = c
		chH[i] = c + 0.5
		chL[i] = c - 0.5
	}
	choppy, ok := adxWilder(chH, chL, chC, 14)
	if !ok {
		t.Fatal("80 bars is enough history for ADX(14)")
	}
	if choppy > 20 {
		t.Errorf("choppy ADX: got %.2f, want <20", choppy)
	}
	if _, ok := adxWilder(chH[:5], chL[:5], chC[:5], 14); ok {
		t.Error("short series ADX must report ok=false, not a sentinel 0")
	}
}

// ── Swing detection + level clustering ───────────────────────────────────────

func TestSwingHighsLows(t *testing.T) {
	h := []float64{1, 2, 3, 10, 3, 2, 1, 2, 3, 8, 3, 2, 1}
	l := []float64{0.5, 1, 2, 9, 2, 1, 0.2, 1, 2, 7, 2, 1, 0.5}
	sh, sl := swingLevels(h, l, 2)
	if len(sh) != 2 || sh[0] != 10 || sh[1] != 8 {
		t.Fatalf("swing highs: got %v, want [10 8]", sh)
	}
	// Lows: index 6 (0.2) is a swing low with wing 2. Ends can't qualify.
	if len(sl) != 1 || sl[0] != 0.2 {
		t.Fatalf("swing lows: got %v, want [0.2]", sl)
	}
}

func TestClusterLevels(t *testing.T) {
	levels := []float64{110, 110.2, 109.9, 105, 95, 95.1, 94.9, 90}
	got := clusterLevels(levels, 0.5)
	// Expect 4 clusters; strongest first: ~110 (3 touches), ~95 (3), then 105, 90.
	if len(got) != 4 {
		t.Fatalf("clusters: got %d, want 4: %v", len(got), got)
	}
	if got[0].Touches != 3 || got[1].Touches != 3 {
		t.Fatalf("top clusters should have 3 touches: %v", got)
	}
	// Integer levels.
	seen := map[int]bool{}
	for _, c := range got {
		seen[c.Level] = true
	}
	for _, want := range []int{110, 95, 105, 90} {
		if !seen[want] {
			t.Errorf("expected level %d in %v", want, got)
		}
	}
}

func TestSupportResistanceSplit(t *testing.T) {
	// Build candles oscillating between ~90 and ~110 with last close 100.
	var candles []types.OHLCVCandle
	for i := 0; i < 60; i++ {
		base := 100.0
		var h, l float64
		switch i % 8 {
		case 3: // peak
			h, l = 110, 104
		case 7: // trough
			h, l = 96, 90
		default:
			h, l = 102, 98
		}
		candles = append(candles, types.OHLCVCandle{
			Time: int64(i), Open: base, High: h, Low: l, Close: base, Volume: 1,
		})
	}
	sup, res := supportResistance(candles, 2, 0.5)
	if len(sup) == 0 || len(res) == 0 {
		t.Fatalf("expected non-empty S/R, got sup=%v res=%v", sup, res)
	}
	for _, s := range sup {
		if float64(s.Level) >= 100 {
			t.Errorf("support %d should be below last close 100", s.Level)
		}
	}
	for _, r := range res {
		if float64(r.Level) <= 100 {
			t.Errorf("resistance %d should be above last close 100", r.Level)
		}
	}
	if len(sup) > 3 || len(res) > 3 {
		t.Errorf("top-3 cap violated: sup=%d res=%d", len(sup), len(res))
	}
}

// ── MACD ─────────────────────────────────────────────────────────────────────

func TestMACDSignsFollowTrend(t *testing.T) {
	n := 120
	up := make([]float64, n)
	for i := range up {
		up[i] = 100 + float64(i)
	}
	_, _, hist, ok := macdLast(up)
	if !ok || hist <= 0 {
		t.Errorf("uptrend MACD hist: got %.4f ok=%v, want >0/true", hist, ok)
	}
	down := make([]float64, n)
	for i := range down {
		down[i] = 300 - float64(i)
	}
	_, _, hist, ok = macdLast(down)
	if !ok || hist >= 0 {
		t.Errorf("downtrend MACD hist: got %.4f ok=%v, want <0/true", hist, ok)
	}
}
