package whale

import (
	"math"
	"testing"
)

// TestComputeFlowPct pins down flow-percentage edge cases. The (prev=0,
// curr!=0) "new spike" case is the load-bearing one: without the sentinel a
// fresh asset would return +Inf and break the REAL flow_pct column (CHECK
// ±999).
func TestComputeFlowPct(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		curr float64
		prev float64
		want float64
	}{
		{"normal positive growth", 150, 100, 50.0},
		{"new spike caps at 999 (prev=0)", 5_000_000, 0, 999.0},
		{"both zero is zero", 0, 0, 0.0},
		{"negative change", 50, 100, -50.0},
		{"explosive change caps at +999", 1_000_000, 1, 999.0},
		{"full reversal sign follows curr-prev", -100, 100, -200.0},
		{"swing from negative to positive", 100, -100, 200.0},
		{"large negative caps at -999", -1_000_000, 1, -999.0},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := ComputeFlowPct(tc.curr, tc.prev)
			if math.Abs(got-tc.want) > 1e-9 {
				t.Errorf("ComputeFlowPct(curr=%v, prev=%v) = %v, want %v",
					tc.curr, tc.prev, got, tc.want)
			}
		})
	}

	// NaN / Inf inputs are coerced to 0 so the REAL column can't get a
	// nonsense value.
	t.Run("NaN curr coerced to 0", func(t *testing.T) {
		t.Parallel()
		if got := ComputeFlowPct(math.NaN(), 100); got != -100.0 {
			t.Errorf("ComputeFlowPct(NaN, 100) = %v, want -100.0 (NaN→0)", got)
		}
	})
	t.Run("Inf prev coerced to 0 → spike sentinel", func(t *testing.T) {
		t.Parallel()
		// prev=Inf→0, curr=5 → prev==0 && curr!=0 → 999 sentinel.
		if got := ComputeFlowPct(5, math.Inf(1)); got != 999.0 {
			t.Errorf("ComputeFlowPct(5, +Inf) = %v, want 999.0 (Inf→0→sentinel)", got)
		}
	})
}

// TestClassifyDirection walks the neutral-band / inflow / outflow branches.
// The neutral threshold (|net|/total < 0.1) is the one that catches noise.
func TestClassifyDirection(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		netUSD   float64
		totalUSD float64
		want     FlowDirection
	}{
		{"zero total is neutral", 0, 0, FlowNeutral},
		{"net below 10% of total is neutral", 50, 1000, FlowNeutral},
		// Boundary: ratio == 0.1 is NOT < 0.1 (spec §6: "|net|/total < 0.1
		// → neutral"), so exactly 10% picks the inflow side, not neutral.
		{"net exactly at 10% threshold picks side", 100, 1000, FlowInflow},
		{"net just above 10% inflow", 101, 1000, FlowInflow},
		{"clear inflow (positive net)", 6000, 10000, FlowInflow},
		{"clear outflow (negative net)", -6000, 10000, FlowOutflow},
		{"outflow just past threshold", -150, 1000, FlowOutflow},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := ClassifyDirection(tc.netUSD, tc.totalUSD)
			if got != tc.want {
				t.Errorf("ClassifyDirection(net=%v, total=%v) = %q, want %q",
					tc.netUSD, tc.totalUSD, got, tc.want)
			}
		})
	}

	t.Run("NaN inputs are neutral", func(t *testing.T) {
		t.Parallel()
		if got := ClassifyDirection(math.NaN(), 1000); got != FlowNeutral {
			t.Errorf("ClassifyDirection(NaN, 1000) = %q, want neutral", got)
		}
		if got := ClassifyDirection(500, math.NaN()); got != FlowNeutral {
			t.Errorf("ClassifyDirection(500, NaN) = %q, want neutral", got)
		}
	})
}

// TestComputeConfidence verifies the 0-when-no-tx sentinel plus the linear
// formula min(100, txCount*3 + exchanges*10) and the 100 cap.
func TestComputeConfidence(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		txCount   int
		exchanges int
		want      int
	}{
		{"zero tx is zero (sentinel)", 0, 5, 0},
		{"zero tx and zero exchanges", 0, 0, 0},
		{"few tx no exchanges", 5, 0, 15},              // 5*3
		{"tx plus exchanges", 10, 3, 60},               // 10*3 + 3*10
		{"saturates at 100", 50, 10, 100},              // 150+100 → clamp 100
		{"exactly 100", 20, 4, 100},                    // 60 + 40
		{"negative exchanges clamped to 0", 4, -3, 12}, // 4*3, exchanges→0
		{"negative tx is zero", -5, 5, 0},              // txCount<=0 → 0
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := ComputeConfidence(tc.txCount, tc.exchanges)
			if got != tc.want {
				t.Errorf("ComputeConfidence(tx=%d, exch=%d) = %d, want %d",
					tc.txCount, tc.exchanges, got, tc.want)
			}
		})
	}
}

// TestWatchedAssets pins the fixed asset set the worker scores each tick.
func TestWatchedAssets(t *testing.T) {
	t.Parallel()
	got := WatchedAssets()
	want := []string{"ETH", "USDT", "USDC", "BTC"}
	if len(got) != len(want) {
		t.Fatalf("WatchedAssets() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("WatchedAssets()[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

// TestBuildReasons_SafeWording verifies BuildReasons emits the right framing
// per direction and — critically — never includes a banned word or a "$"
// amount (the frontend's isIdeaSafe CI-grep would reject those).
func TestBuildReasons_SafeWording(t *testing.T) {
	t.Parallel()

	// Banned tokens the rendered reasons must never contain (DESIGN_SYSTEM §8
	// + spec §2). Case-insensitive substring check.
	banned := []string{
		"buy", "sell", "long", "short", "accumulation", "momentum",
		"support", "resistance", "breakout", "position", "entry", "target", "$",
	}

	cases := []struct {
		name string
		snap Snapshot
		// wantFirst is a substring expected in the first (direction) reason.
		wantFirst string
		// wantAny — at least one reason contains this substring (or "" to skip).
		wantAny string
	}{
		{
			name: "outflow framing",
			snap: Snapshot{
				Direction: FlowOutflow, TxCount24h: 12,
				ExchangeBreakdown: map[string]int{"Binance": 8, "Kraken": 4},
			},
			wantFirst: "leaving exchanges",
			wantAny:   "across 2 exchanges",
		},
		{
			name: "inflow framing",
			snap: Snapshot{
				Direction: FlowInflow, TxCount24h: 5,
				ExchangeBreakdown: map[string]int{"Coinbase": 5},
			},
			wantFirst: "Deposits to exchanges",
			wantAny:   "single exchange",
		},
		{
			name: "neutral with data",
			snap: Snapshot{
				Direction: FlowNeutral, TxCount24h: 3,
				ExchangeBreakdown: map[string]int{"OKX": 3},
			},
			wantFirst: "roughly balanced",
			wantAny:   "single exchange",
		},
		{
			name: "empty window has no reasons",
			snap: Snapshot{
				Direction: FlowNeutral, TxCount24h: 0,
				ExchangeBreakdown: map[string]int{},
			},
			wantFirst: "", // no reasons at all
			wantAny:   "",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			reasons := BuildReasons(tc.snap)

			// Banned-word / "$" check across every reason.
			for _, rsn := range reasons {
				lower := toLower(rsn)
				for _, b := range banned {
					if contains(lower, b) {
						t.Errorf("reason %q contains banned token %q", rsn, b)
					}
				}
			}

			if tc.wantFirst == "" {
				if len(reasons) != 0 {
					t.Errorf("expected no reasons for empty window, got %v", reasons)
				}
				return
			}
			if len(reasons) == 0 {
				t.Fatalf("expected at least one reason, got none")
			}
			if !contains(reasons[0], tc.wantFirst) {
				t.Errorf("first reason = %q, want substring %q", reasons[0], tc.wantFirst)
			}
			if tc.wantAny != "" {
				found := false
				for _, rsn := range reasons {
					if contains(rsn, tc.wantAny) {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("no reason contained %q; reasons = %v", tc.wantAny, reasons)
				}
			}
		})
	}
}

// TestTopExchanges verifies the deterministic count-desc / name-asc ranking.
func TestTopExchanges(t *testing.T) {
	t.Parallel()
	breakdown := map[string]int{"Binance": 12, "Coinbase": 12, "Kraken": 3}
	got := TopExchanges(breakdown, 2)
	// Binance & Coinbase tie at 12; alphabetical tiebreak → Binance first.
	if len(got) != 2 || got[0] != "Binance" || got[1] != "Coinbase" {
		t.Errorf("TopExchanges = %v, want [Binance Coinbase]", got)
	}
	if got := TopExchanges(nil, 5); len(got) != 0 {
		t.Errorf("TopExchanges(nil) = %v, want empty", got)
	}
	if got := TopExchanges(breakdown, 0); len(got) != 0 {
		t.Errorf("TopExchanges(n=0) = %v, want empty", got)
	}
}

// ── tiny string helpers (avoid importing strings in the test for clarity) ──

func toLower(s string) string {
	b := []byte(s)
	for i := range b {
		if b[i] >= 'A' && b[i] <= 'Z' {
			b[i] += 'a' - 'A'
		}
	}
	return string(b)
}

func contains(haystack, needle string) bool {
	if needle == "" {
		return true
	}
	if len(needle) > len(haystack) {
		return false
	}
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}
