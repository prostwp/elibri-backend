package demobot

// trend_b3_test.go — B3: the pullback zone (EMA20-EMA50 band, confirmed-trend
// states only) and the HH/HL swing-structure fact.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// stubBinanceCandles serves the EXACT given candles for any symbol — for
// fixtures whose wick sizes matter (B4 band math).
func stubBinanceCandles(t *testing.T, candles []types.OHLCVCandle) {
	t.Helper()
	rows := make([][]any, len(candles))
	for i, c := range candles {
		rows[i] = []any{
			float64(c.Time) * 1000,
			fmt.Sprintf("%f", c.Open), fmt.Sprintf("%f", c.High),
			fmt.Sprintf("%f", c.Low), fmt.Sprintf("%f", c.Close),
			fmt.Sprintf("%f", c.Volume),
		}
	}
	body, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	orig := binanceKlinesBase
	binanceKlinesBase = srv.URL
	t.Cleanup(func() { binanceKlinesBase = orig })
}

// stubBinanceSeries serves `bars` synthetic closed 4h candles with
// caller-driven closes/volumes and the classic ±100 wick shape
// (stubBinanceKlines keeps its fixed uptrend for the older tests).
func stubBinanceSeries(t *testing.T, bars int, price func(i int) float64, volume func(i int) float64) {
	t.Helper()
	start := time.Now().Unix() - int64(bars+2)*14400 // every bar closed
	candles := make([]types.OHLCVCandle, bars)
	for i := range candles {
		p := price(i)
		candles[i] = types.OHLCVCandle{
			Time: start + int64(i)*14400,
			Open: p - 25, High: p + 100, Low: p - 100, Close: p,
			Volume: volume(i),
		}
	}
	stubBinanceCandles(t, candles)
}

// zigzagUp is a rising sawtooth: +100/bar drift with a 10-bar cycle of ±600
// wiggle — swing highs at i%10==3 and swing lows at i%10==8, both rising
// every cycle (verified by hand in the B3 design notes). Confirmed uptrend
// with clean HH/HL structure.
func zigzagUp(i int) float64 {
	offsets := []float64{0, 200, 400, 600, 400, 200, 0, -200, -400, -200}
	return 60000 + float64(i)*100 + offsets[i%10]
}

func flatVol(int) float64 { return 100 }

// ── pure gating: zone only in confirmed states ───────────────────────────────

func TestPullbackZoneGating(t *testing.T) {
	for _, state := range []string{trendFlat, trendGrey, trendConflict} {
		if z := pullbackZoneFor(state, 63200, 63900); z != nil {
			t.Errorf("state %s: zone = %+v, want nil (confirmed states only)", state, z)
		}
	}
	up := pullbackZoneFor(trendUp, 63900, 63200)
	if up == nil || up.From != 63900 || up.To != 63200 {
		t.Errorf("uptrend zone = %+v, want from=EMA20 63900 to=EMA50 63200", up)
	}
	down := pullbackZoneFor(trendDown, 61100, 61800)
	if down == nil || down.From != 61100 || down.To != 61800 {
		t.Errorf("downtrend zone = %+v, want from=EMA20 61100 to=EMA50 61800", down)
	}
}

// ── pure HH/HL classification (alternation-aware, review fix 7) ──────────────

// pts builds swing points from (idx, price) pairs.
func pts(pairs ...float64) []swingPoint {
	out := make([]swingPoint, 0, len(pairs)/2)
	for i := 0; i+1 < len(pairs); i += 2 {
		out = append(out, swingPoint{idx: int(pairs[i]), price: pairs[i+1]})
	}
	return out
}

func TestHHHLStructure(t *testing.T) {
	cases := []struct {
		name        string
		highs, lows []swingPoint
		want        string
	}{
		{
			"alternating rising pivots",
			pts(1, 10, 5, 11, 9, 12), pts(3, 5, 7, 6, 11, 7),
			"hh_hl", // H1 L3 H5 L7 H9 L11 — proper structure, both rising
		},
		{
			"alternating falling pivots",
			pts(1, 12, 5, 11, 9, 10), pts(3, 7, 7, 6, 11, 5),
			"lh_ll",
		},
		{
			"expanding (HH but LL)",
			pts(1, 10, 5, 11, 9, 12), pts(3, 7, 7, 6, 11, 5),
			"mixed",
		},
		{
			"flat highs",
			pts(1, 10, 5, 10, 9, 10), pts(3, 5, 7, 6, 11, 7),
			"mixed",
		},
		{
			// Two consecutive highs without a low between them (a double top
			// inside the window) is NOT an alternating structure.
			"non-alternating: adjacent highs",
			pts(1, 10, 3, 11, 9, 12), pts(5, 5, 7, 6, 11, 7),
			"mixed", // order: H1 H3 L5 L7? → H1,H3 adjacent → mixed
		},
		{
			// A same-bar high+low (outside bar) cannot alternate.
			"non-alternating: same-bar pivot pair",
			pts(1, 10, 5, 11, 9, 12), pts(3, 5, 7, 6, 9, 7),
			"mixed",
		},
		{
			// Only the LAST six pivots are read: early junk is ignored.
			"only the last six pivots count",
			pts(0, 99, 11, 10, 15, 11, 19, 12), pts(13, 5, 17, 6, 21, 7),
			"hh_hl", // tail: H11 L13 H15 L17 H19 L21
		},
		{"too few pivots", pts(1, 10, 5, 11), pts(3, 5, 7, 6), ""},
		{"empty", nil, nil, ""},
	}
	for _, tc := range cases {
		if got := hhhlStructure(tc.highs, tc.lows); got != tc.want {
			t.Errorf("%s: got %q, want %q", tc.name, got, tc.want)
		}
	}
}

// ── pure invalidation direction (review fix 8) ───────────────────────────────

func TestInvalidationFor(t *testing.T) {
	// Uptrend (and non-directional states): below min(EMA) − 1 ATR.
	for _, state := range []string{trendUp, trendFlat, trendGrey, trendConflict} {
		lvl, side := invalidationFor(state, 105, 100, 4)
		if lvl != 96 || side != "below" {
			t.Errorf("%s: got %v/%s, want 96/below", state, lvl, side)
		}
	}
	// Downtrend: the structure breaks UPWARD — max(EMA) + 1 ATR.
	lvl, side := invalidationFor(trendDown, 100, 105, 4)
	if lvl != 109 || side != "above" {
		t.Errorf("down: got %v/%s, want 109/above", lvl, side)
	}
}

// ── card integration ─────────────────────────────────────────────────────────

func TestTrendCardPullbackZoneAndStructure(t *testing.T) {
	stubBinanceSeries(t, 250, zigzagUp, flatVol)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.TrendCard(context.Background(), btcSpec)

	if !strings.Contains(c.Verdict, "UPTREND") {
		t.Fatalf("zigzag stub must confirm an uptrend, got %q", c.Verdict)
	}
	joined := strings.Join(c.Facts, "|")

	// The zone fact: "Pullback zone: 63200-63900 (EMA20-EMA50 band)" wording,
	// rendered low-high.
	var zoneLine string
	for _, f := range c.Facts {
		if strings.HasPrefix(f, "Pullback zone: ") {
			zoneLine = f
		}
	}
	if zoneLine == "" {
		t.Fatalf("pullback zone fact missing in a confirmed trend: %v", c.Facts)
	}
	if !strings.HasSuffix(zoneLine, "(EMA20-EMA50 band)") {
		t.Errorf("zone wording: %q", zoneLine)
	}
	var lo, hi float64
	if _, err := fmt.Sscanf(zoneLine, "Pullback zone: %f-%f (EMA20-EMA50 band)", &lo, &hi); err != nil {
		t.Fatalf("cannot parse zone from %q: %v", zoneLine, err)
	}
	if !(lo < hi) {
		t.Errorf("zone rendered %v-%v, want low-high order", lo, hi)
	}

	// levels.pullback_zone carries the RAW EMAs: from = EMA20 (nearer price in
	// an uptrend), to = EMA50 — so from > to here, order preserved in JSON.
	lv, ok := c.Levels.(TrendLevels)
	if !ok {
		t.Fatalf("levels = %T, want TrendLevels", c.Levels)
	}
	if lv.PullbackZone == nil {
		t.Fatal("levels.pullback_zone missing in a confirmed trend")
	}
	if !(lv.PullbackZone.From > lv.PullbackZone.To) {
		t.Errorf("uptrend zone from(EMA20)=%v to(EMA50)=%v, want EMA20 above EMA50",
			lv.PullbackZone.From, lv.PullbackZone.To)
	}
	if lv.Invalidation == 0 {
		t.Error("invalidation must still ride along")
	}

	// HH/HL structure fact from the same swing points S/R uses.
	if !strings.Contains(joined, "Structure: HH/HL confirmed") {
		t.Errorf("structure fact missing or wrong: %v", c.Facts)
	}

	// Envelope shape: pullback_zone serialized with from/to.
	env, err := encodeJSON(cardEnvelope(c))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(env), `"pullback_zone":{"from":`) {
		t.Errorf("envelope levels must carry pullback_zone: %s", env)
	}
}

func TestTrendCardNoZoneOutsideConfirmed(t *testing.T) {
	// Constant price → ADX ~0 → flat. No zone fact, no pullback_zone in
	// levels, and no fabricated Structure line (a monotone-flat series has no
	// swings to read).
	stubBinanceSeries(t, 250, func(int) float64 { return 60000 }, flatVol)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.TrendCard(context.Background(), btcSpec)

	if !strings.HasPrefix(c.Verdict, "Flat") {
		t.Fatalf("constant stub must read flat, got %q", c.Verdict)
	}
	joined := strings.Join(c.Facts, "|")
	if strings.Contains(joined, "Pullback zone") {
		t.Errorf("flat state must not offer a pullback zone: %v", c.Facts)
	}
	if lv, ok := c.Levels.(TrendLevels); ok && lv.PullbackZone != nil {
		t.Errorf("levels.pullback_zone = %+v, want nil outside confirmed trends", lv.PullbackZone)
	}
	env, err := encodeJSON(cardEnvelope(c))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(env), "pullback_zone") {
		t.Errorf("envelope must omit pullback_zone outside confirmed trends: %s", env)
	}
}

// zigzagDown mirrors zigzagUp: falling drift with alternating pivots →
// confirmed downtrend with LH/LL agreement.
func zigzagDown(i int) float64 {
	return 160000 - zigzagUp(i)
}

// Review fix 8: a confirmed DOWNTREND's invalidation sits ABOVE the EMA
// cluster (max(EMA50,EMA200) + 1 ATR), worded "above".
func TestTrendCardDowntrendInvalidationAbove(t *testing.T) {
	stubBinanceSeries(t, 250, zigzagDown, flatVol)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.TrendCard(context.Background(), btcSpec)

	if !strings.Contains(c.Verdict, "DOWNTREND") {
		t.Fatalf("mirrored zigzag must confirm a downtrend, got %q", c.Verdict)
	}
	joined := strings.Join(c.Facts, "|")
	if !strings.Contains(joined, "Structure: LH/LL") {
		t.Errorf("downtrend agreement fact missing: %v", c.Facts)
	}
	var invLine string
	for _, f := range c.Facts {
		if strings.HasPrefix(f, "Invalidation: above ") {
			invLine = f
		}
	}
	if invLine == "" {
		t.Fatalf("downtrend invalidation must be worded 'above': %v", c.Facts)
	}
	if !strings.HasSuffix(invLine, "(1 ATR over the EMA cluster)") {
		t.Errorf("invalidation wording: %q", invLine)
	}
	lv, ok := c.Levels.(TrendLevels)
	if !ok {
		t.Fatalf("levels = %T, want TrendLevels", c.Levels)
	}
	if lv.InvalidationSide != "above" {
		t.Errorf("invalidation_side = %q, want above", lv.InvalidationSide)
	}
	last := zigzagDown(249)
	if lv.Invalidation <= last {
		t.Errorf("downtrend invalidation %v must sit above the last close %v", lv.Invalidation, last)
	}
	// The pullback zone still rides along (confirmed state, agreement).
	if lv.PullbackZone == nil || !(lv.PullbackZone.From < lv.PullbackZone.To) {
		t.Errorf("downtrend zone = %+v, want from(EMA20) below to(EMA50)", lv.PullbackZone)
	}
}

// Review fix 7: EMA/ADX-confirmed direction with CONTRADICTING swing
// structure demotes to grey zone — the state machine now requires agreement.
// Fixture: monotone rising closes (EMA/ADX scream uptrend) with six wick-only
// pivots whose highs AND lows fall (LH/LL).
func lhllOverUptrend(bars int) []types.OHLCVCandle {
	start := time.Now().Unix() - int64(bars+2)*14400
	out := make([]types.OHLCVCandle, bars)
	upSpike := map[int]float64{220: 6000, 230: 4000, 240: 2000}   // falling swing highs
	downSpike := map[int]float64{225: 2000, 235: 4000, 245: 6000} // falling swing lows
	for i := range out {
		p := 60000 + 150*float64(i)
		c := types.OHLCVCandle{
			Time: start + int64(i)*14400,
			Open: p - 25, High: p + 100, Low: p - 100, Close: p, Volume: 100,
		}
		if s, ok := upSpike[i]; ok {
			c.High = p + s
		}
		if s, ok := downSpike[i]; ok {
			c.Low = p - s
		}
		out[i] = c
	}
	return out
}

func TestTrendCardStructureDisagreementDemotes(t *testing.T) {
	candles := lhllOverUptrend(250)

	// Precondition asserts: the fixture must genuinely read as an EMA/ADX
	// uptrend with LH/LL pivots — if construction drifts, fail HERE loudly.
	closes := closesOf(candles)
	highs, lows := highsLowsOf(candles)
	ema50, _ := emaLast(closes, 50)
	ema200, _ := emaLast(closes, 200)
	adx, _ := adxWilder(highs, lows, closes, 14)
	if got := classifyTrend(adx, ema50, ema200, closes[len(closes)-1]); got != trendUp {
		t.Fatalf("fixture precondition: classifyTrend = %q (adx %.1f), want up", got, adx)
	}
	sh, sl := swingPointsIdx(highs, lows, 3)
	if got := hhhlStructure(sh, sl); got != "lh_ll" {
		t.Fatalf("fixture precondition: structure = %q, want lh_ll (pivots %d/%d)", got, len(sh), len(sl))
	}

	stubBinanceCandles(t, candles)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.TrendCard(context.Background(), btcSpec)

	if !strings.Contains(c.Verdict, "Grey zone") {
		t.Errorf("verdict = %q, want grey (EMA-up + LH/LL must not confirm)", c.Verdict)
	}
	joined := strings.Join(c.Facts, "|")
	if !strings.Contains(joined, "Structure disagrees (LH/LL) — trend not confirmed") {
		t.Errorf("disagreement fact missing: %v", c.Facts)
	}
	if strings.Contains(joined, "Pullback zone") {
		t.Errorf("demoted state must not offer a pullback zone: %v", c.Facts)
	}
	if lv, ok := c.Levels.(TrendLevels); ok && lv.PullbackZone != nil {
		t.Errorf("levels.pullback_zone = %+v, want nil after demotion", lv.PullbackZone)
	}
}

func TestTrendCardMonotoneHasNoStructureClaim(t *testing.T) {
	// The plain rising stub is monotone — zero swing points, so the card must
	// stay silent about HH/HL rather than invent a claim.
	stubBinanceKlines(t, 250, flatVol)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.TrendCard(context.Background(), btcSpec)
	if strings.Contains(strings.Join(c.Facts, "|"), "Structure:") {
		t.Errorf("monotone series has no swings — no structure claim allowed: %v", c.Facts)
	}
}
