package demobot

// sr_b4_test.go — B4 S/R upgrades: volume-weighted strength, the weakening
// detector, break/hold frequency over the window and last-touch dates.

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// ── break/hold counting on a known history (pure) ────────────────────────────

// constATR builds an ATR slice of n entries all equal to v (warmup handled by
// the caller via leading zeros).
func constATR(n int, v float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func TestBreakHoldStatsKnownHistory(t *testing.T) {
	const level = 100.0
	// ATR 4 everywhere → band = 0.25×4 = 1.0 around the level.
	t.Run("approach from above, break below", func(t *testing.T) {
		closes := []float64{105, 100.5, 99.5, 98.5, 98}
		//                   out↑  test   in    BREAK (98.5 < 99 by >band on the far side)
		r := breakHoldStats(level, closes, constATR(len(closes), 4))
		if r.tests != 1 || r.breaks != 1 || r.holds != 0 {
			t.Errorf("got %+v, want tests/breaks/holds 1/1/0", r)
		}
	})

	t.Run("approach from above, clean rejection = hold", func(t *testing.T) {
		closes := []float64{105, 100.8, 103, 104, 105}
		r := breakHoldStats(level, closes, constATR(len(closes), 4))
		if r.tests != 1 || r.breaks != 0 || r.holds != 1 {
			t.Errorf("got %+v, want 1/0/1", r)
		}
	})

	// Review blocker: three bars sitting INSIDE the band resolve nothing — an
	// unfinished consolidation is not a rejection. Unresolved → dropped from
	// the counts entirely, never a hold.
	t.Run("stalls inside the band for 3 bars = unresolved, dropped", func(t *testing.T) {
		closes := []float64{105, 100.5, 100.2, 99.8, 100.1, 100.4}
		r := breakHoldStats(level, closes, constATR(len(closes), 4))
		if r.tests != 0 || r.breaks != 0 || r.holds != 0 {
			t.Errorf("got %+v, want 0/0/0 (consolidation is not a hold)", r)
		}
	})

	// After a dropped consolidation, a FRESH exit + re-entry later still
	// counts as a new test.
	t.Run("dropped consolidation then a real hold", func(t *testing.T) {
		closes := []float64{105, 100.5, 100.2, 99.8, 100.1, 105, 100.5, 103, 104}
		//                   out   drop:3 bars inside............  out  test  HOLD
		r := breakHoldStats(level, closes, constATR(len(closes), 4))
		if r.tests != 1 || r.breaks != 0 || r.holds != 1 {
			t.Errorf("got %+v, want 1/0/1 (only the resolved test counts)", r)
		}
	})

	t.Run("two separate tests: hold then break", func(t *testing.T) {
		closes := []float64{105, 100.5, 103, 100.9, 98.5}
		//                   out   test1  HOLD  test2  BREAK
		r := breakHoldStats(level, closes, constATR(len(closes), 4))
		if r.tests != 2 || r.breaks != 1 || r.holds != 1 {
			t.Errorf("got %+v, want 2/1/1", r)
		}
	})

	t.Run("approach from below mirrors", func(t *testing.T) {
		closes := []float64{95, 99.5, 101.5, 102}
		//                   out↓  test  BREAK above
		r := breakHoldStats(level, closes, constATR(len(closes), 4))
		if r.tests != 1 || r.breaks != 1 || r.holds != 0 {
			t.Errorf("got %+v, want 1/1/0", r)
		}
	})

	t.Run("unresolved trailing test is dropped", func(t *testing.T) {
		closes := []float64{105, 100.5, 100.2}
		// test opens at the 2nd bar but the window ends before 3 bars pass.
		r := breakHoldStats(level, closes, constATR(len(closes), 4))
		if r.tests != 0 || r.breaks != 0 || r.holds != 0 {
			t.Errorf("got %+v, want 0/0/0 (outcome unknown)", r)
		}
	})

	t.Run("no approach known → no test", func(t *testing.T) {
		closes := []float64{100.2, 100.5, 99.8, 100.1, 100.3, 100.0}
		// The series never leaves the band → no approach side → nothing counted.
		r := breakHoldStats(level, closes, constATR(len(closes), 4))
		if r.tests != 0 || r.breaks != 0 || r.holds != 0 {
			t.Errorf("got %+v, want 0/0/0", r)
		}
	})

	t.Run("warmup bars without ATR are skipped", func(t *testing.T) {
		closes := []float64{98.5, 105, 100.5, 103, 104}
		atr := []float64{0, 4, 4, 4, 4} // bar 0 has no ATR yet
		r := breakHoldStats(level, closes, atr)
		if r.tests != 1 || r.breaks != 0 || r.holds != 1 {
			t.Errorf("got %+v, want 1/0/1", r)
		}
	})

	// Review fix 9: the band is FROZEN at the entry bar's ATR — a volatility
	// spike after the touch must not reclassify the same move.
	t.Run("band frozen at the test bar's ATR", func(t *testing.T) {
		closes := []float64{105, 100.5, 98.5}
		atr := []float64{4, 4, 40} // entry band 1.0; bar2's own band would be 10
		r := breakHoldStats(level, closes, atr)
		if r.tests != 1 || r.breaks != 1 || r.holds != 0 {
			t.Errorf("got %+v, want 1/1/0 (98.5 is beyond the FROZEN band)", r)
		}
	})

	// Review fix 6: lastTestIdx tracks the newest bar whose close entered the
	// band with a known approach — including an unresolved trailing entry
	// (the touch is a fact even when its outcome is unknown).
	t.Run("last test index reported", func(t *testing.T) {
		closes := []float64{105, 100.5, 103, 104, 105, 100.4}
		//                   out   test@1 HOLD             entry@5 (unresolved)
		r := breakHoldStats(level, closes, constATR(len(closes), 4))
		if r.tests != 1 || r.holds != 1 {
			t.Errorf("got %+v, want 1 resolved hold", r)
		}
		if r.lastTestIdx != 5 {
			t.Errorf("lastTestIdx = %d, want 5 (the unresolved touch still dates the level)", r.lastTestIdx)
		}
	})

	t.Run("no touches → lastTestIdx -1", func(t *testing.T) {
		closes := []float64{105, 106, 107}
		r := breakHoldStats(level, closes, constATR(len(closes), 4))
		if r.lastTestIdx != -1 {
			t.Errorf("lastTestIdx = %d, want -1", r.lastTestIdx)
		}
	})
}

// ── volume-weighted strength + weakening through supportResistance ───────────

// srFixtureCandles builds an alternating series: even bars at 100 (swing
// lows), odd bars at 110 (swing highs), wing=1 finds every one. Wickless bars
// (High == Low == Close) keep the cluster means at exactly 110/100. The last
// bar closes mid-way (105) so 110 splits as resistance and 100 as support.
// Volumes are caller-driven per bar.
func srFixtureCandles(bars int, volume func(i int) float64) []types.OHLCVCandle {
	start := time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC).Unix()
	out := make([]types.OHLCVCandle, bars)
	for i := range out {
		p := 100.0
		if i%2 == 1 {
			p = 110.0
		}
		if i == bars-1 {
			p = 105.0
		}
		out[i] = types.OHLCVCandle{
			Time: start + int64(i)*14400,
			Open: p, High: p, Low: p, Close: p,
			Volume: volume(i),
		}
	}
	return out
}

func TestSRVolumeWeightedStrength(t *testing.T) {
	// 21 bars: 10 swing highs at 110 (odd i=1..19), 9 swing lows at 100
	// (even i=2..18; i=0 has no left wing, i=20 is the 105 closer).
	// Volumes: swing-high bars get 1000 (above median), swing-low bars 10,
	// everything else 10 → median = 10 → each 110-touch earns +0.5.
	candles := srFixtureCandles(21, func(i int) float64 {
		if i%2 == 1 {
			return 1000
		}
		return 10
	})
	sup, res := supportResistance(candles, 1, 0.5)
	if len(res) == 0 || len(sup) == 0 {
		t.Fatalf("expected both sides, got sup=%v res=%v", sup, res)
	}
	r, s := res[0], sup[0]
	if r.Level != 110 || s.Level != 100 {
		t.Fatalf("levels: R %d S %d, want 110/100", r.Level, s.Level)
	}
	if r.Touches != 10 || s.Touches != 9 {
		t.Errorf("touches: R %d S %d, want 10/9", r.Touches, s.Touches)
	}
	// Strength formula: touches + 0.5 per above-median-volume touch.
	// R: 10 touches, all 1000 > 10 → 10 + 5.0 = 15. S: volumes at median (10),
	// not ABOVE it → 9 + 0 = 9.
	if r.Strength != 15 {
		t.Errorf("R strength = %v, want 15 (10 touches + 10×0.5)", r.Strength)
	}
	if s.Strength != 9 {
		t.Errorf("S strength = %v, want 9 (no above-median touches)", s.Strength)
	}
	// Last-touch dates: R last touched at bar 19, S at bar 18.
	base := time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC)
	if want := base.Add(19 * 4 * time.Hour); !r.LastTouch.Equal(want) {
		t.Errorf("R last touch = %v, want %v", r.LastTouch, want)
	}
	if want := base.Add(18 * 4 * time.Hour); !s.LastTouch.Equal(want) {
		t.Errorf("S last touch = %v, want %v", s.LastTouch, want)
	}
}

func TestSRWeakeningDetector(t *testing.T) {
	// Same alternating shape; the 110 level has 10 touches (≥7) whose volumes
	// FADE: first three touches 1000, last three 50 → weakening. The 100 level
	// keeps flat volumes → not weakening.
	candles := srFixtureCandles(21, func(i int) float64 {
		if i%2 == 1 {
			switch {
			case i <= 5:
				return 1000 // first three 110-touches: i = 1,3,5
			case i >= 15:
				return 50 // last three: i = 15,17,19
			default:
				return 500
			}
		}
		return 100
	})
	sup, res := supportResistance(candles, 1, 0.5)
	if len(res) == 0 || len(sup) == 0 {
		t.Fatal("expected both sides")
	}
	if !res[0].Weakening {
		t.Errorf("110 level: weakening = false, want true (mean last3 50 < mean first3 1000)")
	}
	if sup[0].Weakening {
		t.Errorf("100 level: weakening = true, want false (flat volumes)")
	}

	// Fewer than 7 touches never flags, no matter the fade.
	small := srFixtureCandles(9, func(i int) float64 { // four 110-touches
		if i%2 == 1 {
			return 1000 - float64(i)*100
		}
		return 100
	})
	_, resSmall := supportResistance(small, 1, 0.5)
	if len(resSmall) > 0 && resSmall[0].Weakening {
		t.Errorf("4-touch level flagged weakening — the 7-touch floor must gate it")
	}
}

// Review fix 11: the median is taken over NON-ZERO volumes, and a level whose
// touches are mostly volume-less disables the volume features instead of
// pretending zeros are information.
func TestSRVolumeFeaturesMixedSeries(t *testing.T) {
	// Odd (110-touch) bars carry real volume: five at 2000, five at 500.
	// Even (100-touch) bars carry zero volume (FX-style holes).
	candles := srFixtureCandles(21, func(i int) float64 {
		if i%2 == 1 {
			if i <= 9 {
				return 2000
			}
			return 500
		}
		return 0
	})
	sup, res := supportResistance(candles, 1, 0.5)
	if len(res) == 0 || len(sup) == 0 {
		t.Fatal("expected both sides")
	}
	// Non-zero median = median(5×2000, 5×500) = 1250 → the 2000-volume touches
	// earn +0.5 each: strength = 10 + 5×0.5 = 12.5. A zero-dragged median
	// (old bug) would have been ~500 and given +0.5 to every touch.
	if res[0].Strength != 12.5 {
		t.Errorf("R strength = %v, want 12.5 (median over non-zero volumes)", res[0].Strength)
	}
	// The 100 level: every touch volume-less → fewer than half carry volume →
	// features disabled: strength == touches, no weakening claim.
	if sup[0].Strength != float64(sup[0].Touches) {
		t.Errorf("S strength = %v, want plain touches %d (volume features disabled)", sup[0].Strength, sup[0].Touches)
	}
	if sup[0].Weakening {
		t.Error("S weakening claimed with zero-volume touches")
	}
}

// Review fix 12: a bar that is both a swing high and a swing low (an outside
// bar) must count as ONE touch when both points land in the same cluster.
func TestSRSameBarDoubleCountGuard(t *testing.T) {
	start := time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC).Unix()
	mk := func(h, l float64) types.OHLCVCandle {
		return types.OHLCVCandle{High: h, Low: l, Open: (h + l) / 2, Close: (h + l) / 2, Volume: 10}
	}
	candles := []types.OHLCVCandle{
		mk(100.0, 99.95),
		mk(100.2, 99.9), // outside bar: swing high AND swing low, 0.3 apart (< 0.5% tol)
		mk(100.0, 99.95),
		mk(105, 105), // closer, so the cluster splits as support
	}
	for i := range candles {
		candles[i].Time = start + int64(i)*14400
	}
	sup, _ := supportResistance(candles, 1, 0.5)
	if len(sup) != 1 {
		t.Fatalf("supports = %+v, want the one merged cluster", sup)
	}
	if sup[0].Touches != 1 {
		t.Errorf("touches = %d, want 1 (same-bar high+low is one touch)", sup[0].Touches)
	}
}

// Review fix 6: last_touch = the newest of (last swing in the cluster, last
// close-test of the level) — a recent test of an old level refreshes it.
func TestSRLastTouchRefreshedByTest(t *testing.T) {
	base := time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC)
	start := base.Unix()
	// Bars 0-10 alternate 100/110 (the 110 cluster's last SWING is bar 9),
	// bars 11-25 drift flat at 104 (ATR decays, no swings), then a late
	// three-bar push through 110: 109.6 enters the band at bar 26 (a test —
	// NOT a swing high, bar 27 is higher), and bar 28 closes beyond → break.
	// The close-test at bar 26 must refresh the 110 level's last_touch.
	prices := []float64{
		100, 110, 100, 110, 100, 110, 100, 110, 100, 110, 100,
		104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104,
		109.6, 110.4, 111.0, 105,
	}
	candles := make([]types.OHLCVCandle, len(prices))
	for i, p := range prices {
		candles[i] = types.OHLCVCandle{
			Time: start + int64(i)*14400,
			Open: p, High: p, Low: p, Close: p, Volume: 10,
		}
	}
	_, res := supportResistance(candles, 1, 0.5)
	if len(res) == 0 {
		t.Fatalf("expected the 110 resistance, got res=%+v", res)
	}
	r := res[0]
	if r.Level != 110 || r.Touches != 5 {
		t.Fatalf("R = level %d touches %d, want the 5-touch 110 cluster", r.Level, r.Touches)
	}
	if r.Breaks != 1 {
		t.Errorf("breaks = %d, want 1 (the late push through)", r.Breaks)
	}
	// Last swing touch was bar 9; the close-test entry at bar 26 refreshes it.
	want := base.Add(26 * 4 * time.Hour)
	if !r.LastTouch.Equal(want) {
		t.Errorf("last touch = %v, want %v (refreshed by the close-test)", r.LastTouch, want)
	}
}

// Review fix 2: both-empty honesty. Zero swing points in a long-enough window
// → insufficient-history-style degrade; swings present but no level on either
// side of price → an explicit ok "no significant levels" finding.
func TestSRCardEmptyLevelHonesty(t *testing.T) {
	// (a) Monotone series: enough bars, zero swings → 503 insufficient_history.
	stubBinanceKlines(t, 60, func(int) float64 { return 100 })
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.SRCard(context.Background(), btcSpec)
	if c.Status != statusInsufficientHistory || !c.Offline {
		t.Errorf("monotone window: status=%v offline=%v, want insufficient_history 503-path", c.Status, c.Offline)
	}
	if strings.Contains(c.Verdict, "Key levels around") {
		t.Errorf("monotone window must not claim levels: %q", c.Verdict)
	}
}

func TestSRCardNoLevelsEitherSideOfPrice(t *testing.T) {
	// (b) Swings exist but the only cluster sits exactly AT the last close →
	// neither side gets a level. That is a real, honest finding: ok:true with
	// the explicit "no significant levels" verdict and empty arrays.
	// Construction (wing=3): a monotone rise (no swing lows, no swing highs)
	// with ONE wick spike to 150 at bar 10 — the sole swing point — and a
	// final close at exactly 150.
	start := time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC).Unix()
	candles := make([]types.OHLCVCandle, 30)
	for i := range candles {
		p := 100.0 + float64(i)
		if i == len(candles)-1 {
			p = 150 // final close ON the cluster mean
		}
		c := types.OHLCVCandle{Time: start + int64(i)*14400, Open: p, High: p, Low: p, Close: p, Volume: 10}
		if i == 10 {
			c.High = 150 // the spike wick — the only swing point in the window
		}
		candles[i] = c
	}
	stubBinanceCandles(t, candles)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.SRCard(context.Background(), btcSpec)
	if c.Status != statusOK || c.Offline {
		t.Fatalf("swings-present empty split must stay a real reading: status=%v offline=%v", c.Status, c.Offline)
	}
	if !strings.Contains(c.Verdict, "No significant levels detected in the window") {
		t.Errorf("verdict = %q, want the explicit no-levels finding", c.Verdict)
	}
	lv, ok := c.Levels.(SRLevels)
	if !ok || len(lv.Supports) != 0 || len(lv.Resistances) != 0 {
		t.Errorf("levels = %+v, want empty arrays", c.Levels)
	}
}

func TestSRZeroVolumeSeriesStaysHonest(t *testing.T) {
	// FX via Yahoo has null volume throughout → median 0. No volume bonus, no
	// weakening claims — strength must equal the plain touch count.
	candles := srFixtureCandles(21, func(int) float64 { return 0 })
	sup, res := supportResistance(candles, 1, 0.5)
	for _, l := range append(sup, res...) {
		if l.Strength != float64(l.Touches) {
			t.Errorf("level %d: strength %v != touches %d on a volume-less series", l.Level, l.Strength, l.Touches)
		}
		if l.Weakening {
			t.Errorf("level %d: weakening claimed with zero volume data", l.Level)
		}
	}
}

// ── card + levels wiring ─────────────────────────────────────────────────────

// srCardCycleCandles is a period-12 zigzag between 90 and 110 with small
// (±0.5) wicks: swing highs of exactly 110 every cycle (phase 3), swing lows
// of exactly 90 (phase 9) — clean clusters at wing=3, and closes that leave
// the ±0.25×ATR band between touches so break/hold tests actually register.
// The last bar closes at 105 → 110 splits as resistance, 90 as support.
func srCardCycleCandles(bars int, volume func(i int) float64) []types.OHLCVCandle {
	offsets := []float64{100, 103, 106, 110, 106, 103, 100, 97, 94, 90, 94, 97}
	start := time.Now().Unix() - int64(bars+2)*14400 // all bars closed
	out := make([]types.OHLCVCandle, bars)
	for i := range out {
		p := offsets[i%12]
		if i == bars-1 {
			p = 105
		}
		out[i] = types.OHLCVCandle{
			Time: start + int64(i)*14400,
			Open: p, High: p + 0.5, Low: p - 0.5, Close: p,
			Volume: volume(i),
		}
	}
	return out
}

func TestSRCardB4FactsAndLevels(t *testing.T) {
	stubBinanceCandles(t, srCardCycleCandles(250, func(int) float64 { return 100 }))
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.SRCard(context.Background(), btcSpec)

	joined := strings.Join(c.Facts, "|")
	// The strength formula + the frequency wording must be documented on the
	// card itself ("frequency", never "probability").
	if !strings.Contains(joined, "strength = touches + 0.5 per above-median-volume touch") {
		t.Errorf("strength formula line missing: %v", c.Facts)
	}
	if !strings.Contains(joined, "held") || !strings.Contains(joined, "of") {
		t.Errorf("held-of-tests wording missing: %v", c.Facts)
	}
	if strings.Contains(strings.ToLower(joined), "probability") {
		t.Errorf("probability language is banned — frequency only: %v", c.Facts)
	}

	lv, ok := c.Levels.(SRLevels)
	if !ok {
		t.Fatalf("levels = %T, want SRLevels", c.Levels)
	}
	pts := append(append([]SRPoint{}, lv.Supports...), lv.Resistances...)
	if len(pts) == 0 {
		t.Fatal("no levels produced from the zigzag")
	}
	for _, p := range pts {
		if p.Strength < float64(p.Touches) {
			t.Errorf("level %v: strength %v < touches %d", p.Level, p.Strength, p.Touches)
		}
		if p.Breaks < 0 || p.Holds < 0 {
			t.Errorf("level %v: negative break/hold counts", p.Level)
		}
		if p.LastTouch == "" {
			t.Errorf("level %v: last_touch missing", p.Level)
		} else if _, err := time.Parse(time.RFC3339, p.LastTouch); err != nil {
			t.Errorf("level %v: last_touch %q not RFC3339: %v", p.Level, p.LastTouch, err)
		}
	}
}

func TestSRCardWeakeningLine(t *testing.T) {
	// The 110 level's touch bars sit at i%12==3 (i = 3, 15, 27, …, 243). Give
	// the first three 1000, the last three 50 → the fade must surface as
	// "weakening: volume fading" on the card and weakening:true in levels.
	vols := func(i int) float64 {
		if i%12 == 3 {
			if i <= 27 {
				return 1000
			}
			if i >= 219 {
				return 50
			}
			return 500
		}
		return 100
	}
	stubBinanceCandles(t, srCardCycleCandles(250, vols))
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.SRCard(context.Background(), btcSpec)

	if !strings.Contains(strings.Join(c.Facts, "|"), "weakening: volume fading") {
		t.Errorf("weakening marker missing from the card: %v", c.Facts)
	}
	lv, ok := c.Levels.(SRLevels)
	if !ok {
		t.Fatalf("levels = %T, want SRLevels", c.Levels)
	}
	found := false
	for _, p := range append(append([]SRPoint{}, lv.Supports...), lv.Resistances...) {
		if p.Weakening {
			found = true
		}
	}
	if !found {
		t.Errorf("no level carries weakening:true in levels: %+v", lv)
	}
}
