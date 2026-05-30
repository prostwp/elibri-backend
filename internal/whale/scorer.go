package whale

import (
	"fmt"
	"math"
	"sort"
)

// scorer.go — pure functions that turn raw inflow/outflow sums into the
// Snapshot fields (flow_pct, direction, confidence, reasons). No I/O, no
// network — fully unit-testable. All wording emitted here uses ONLY the safe
// vocabulary (deposits / withdrawals / coins leaving|entering exchanges /
// across N exchanges / largest single move). NONE of the banned words
// (buy, sell, long, short, accumulation, momentum, support, resistance,
// breakout, position, entry, target) and NO "$" amounts appear in reasons —
// the frontend's isIdeaSafe CI-grep depends on this.

const (
	// flowCap is the absolute cap on ComputeFlowPct so a fresh asset (prev=0)
	// doesn't return +Inf and break the REAL flow_pct column (CHECK ±999).
	flowCap = 999.0

	// flowSpikeSentinel is returned when prev==0 && curr!=0 — treat the first
	// nonzero window as a saturated "new spike". Same magnitude as flowCap so
	// the UI can render one "New" chip off this value.
	flowSpikeSentinel = 999.0

	// neutralThreshold: when |netUSD| is below this fraction of the total
	// gross flow, the direction is "neutral" (the net is noise relative to
	// the churn). 0.1 = net must be at least 10% of gross to pick a side.
	neutralThreshold = 0.1

	// confidencePerTx / confidencePerExchange: linear contributors to the
	// data-quality confidence. min(100, txCount*3 + exchanges*10). A handful
	// of txns across a couple of exchanges already reads as "we have data".
	confidencePerTx       = 3
	confidencePerExchange = 10
)

// ComputeFlowPct returns the percentage change from prev to curr, capped at
// ±flowCap. Special cases (spec §6):
//   - prev == 0 && curr == 0: returns 0 (no signal either way).
//   - prev == 0 && curr != 0: returns flowSpikeSentinel (+999) — "new spike".
//   - otherwise: ((curr-prev) / |prev|) * 100, clamped to [-999, 999].
//
// Using |prev| in the denominator (not prev) keeps the sign of the change
// driven by (curr-prev), so a netflow swinging from -100 to +100 reports a
// positive change. NaN/Inf inputs are coerced to 0 first so a malformed
// upstream sum can't poison the REAL column.
func ComputeFlowPct(curr, prev float64) float64 {
	if math.IsNaN(curr) || math.IsInf(curr, 0) {
		curr = 0
	}
	if math.IsNaN(prev) || math.IsInf(prev, 0) {
		prev = 0
	}
	if prev == 0 {
		if curr == 0 {
			return 0
		}
		return flowSpikeSentinel
	}
	pct := ((curr - prev) / math.Abs(prev)) * 100.0
	if pct > flowCap {
		return flowCap
	}
	if pct < -flowCap {
		return -flowCap
	}
	return pct
}

// ClassifyDirection buckets a (netUSD, totalUSD) pair into a FlowDirection.
//   - totalUSD == 0 OR |netUSD|/totalUSD < neutralThreshold → FlowNeutral.
//   - netUSD > 0 → FlowInflow  (coins moved TO exchanges; bearish ▼).
//   - netUSD < 0 → FlowOutflow (coins moved FROM exchanges; bullish ▲).
//
// totalUSD is the GROSS flow (inflow + outflow); netUSD is inflow - outflow.
// The neutral band suppresses a misleading direction verdict when the net is
// just churn-noise relative to the gross. NaN inputs → neutral.
func ClassifyDirection(netUSD, totalUSD float64) FlowDirection {
	if math.IsNaN(netUSD) || math.IsNaN(totalUSD) {
		return FlowNeutral
	}
	if totalUSD == 0 || math.Abs(netUSD)/totalUSD < neutralThreshold {
		return FlowNeutral
	}
	if netUSD > 0 {
		return FlowInflow
	}
	return FlowOutflow
}

// ComputeConfidence returns a 0..100 data-quality score (NOT a prediction-
// strength score). min(100, txCount*confidencePerTx + exchanges*confidencePerExchange).
// Returns 0 when txCount == 0 — the "too little data" sentinel the UI renders
// as "—". Negative inputs are clamped to 0.
func ComputeConfidence(txCount, exchanges int) int {
	if txCount <= 0 {
		return 0
	}
	if exchanges < 0 {
		exchanges = 0
	}
	score := txCount*confidencePerTx + exchanges*confidencePerExchange
	if score > 100 {
		return 100
	}
	if score < 0 {
		return 0
	}
	return score
}

// WatchedAssets is the fixed set of assets the worker scores each tick. ETH +
// the two major stablecoins carry the bulk of labeled exchange flow we can see
// via Etherscan; BTC is included for the live feed (its snapshot is Partial /
// neutral since BTC netflow isn't computed in the MVP).
func WatchedAssets() []string {
	return []string{"ETH", "USDT", "USDC", "BTC"}
}

// BuildReasons assembles short, safe-worded bullet phrases for the UI from a
// Snapshot. ONLY the safe vocabulary is used — no banned words, no "$"
// amounts. Order matches how a reader scans: direction framing → diversity →
// largest move.
//
// The function reads from the snapshot's already-computed fields (Direction,
// InflowUSD24h, OutflowUSD24h, ExchangeBreakdown, TxCount24h) rather than
// recomputing — it's a pure formatter. It never emits a numeric dollar value:
// "largest single move" is mentioned as a qualitative phrase only, because the
// actual figure would be a banned "$N" token in the rendered string.
func BuildReasons(snap Snapshot) []string {
	out := make([]string, 0, 3)

	// Direction framing — the headline. Uses deposits/withdrawals + the
	// "coins leaving/entering exchanges" safe phrasing.
	switch snap.Direction {
	case FlowOutflow:
		out = append(out, "Coins leaving exchanges outpaced deposits over 24h.")
	case FlowInflow:
		out = append(out, "Deposits to exchanges outpaced withdrawals over 24h.")
	default: // FlowNeutral
		// Only frame neutrality when there's actually data; an empty window
		// gets no bullet (the confidence=0 sentinel speaks for it).
		if snap.TxCount24h > 0 {
			out = append(out, "Deposits and withdrawals were roughly balanced over 24h.")
		}
	}

	// Exchange diversity — "across N exchanges". Skip when we saw none.
	nExch := len(snap.ExchangeBreakdown)
	switch {
	case nExch == 0:
		// silent
	case nExch == 1:
		out = append(out, "Activity seen at a single exchange.")
	default:
		out = append(out, fmt.Sprintf("Activity seen across %d exchanges.", nExch))
	}

	// Largest single move — qualitative only (no figure → no "$N" token).
	// Emitted when there's at least one transfer this window so the bullet
	// has a referent.
	if snap.TxCount24h > 0 {
		out = append(out, "Includes at least one large single move.")
	}

	return out
}

// TopExchanges returns up to n exchange labels from a breakdown map, ranked by
// count DESC, ties broken alphabetically ASC (deterministic). Helper for any
// caller that wants a stable "top exchanges" list off ExchangeBreakdown — not
// used by BuildReasons (which deliberately avoids naming exchanges to keep the
// safe-word surface minimal) but provided for the worker's logging / a future
// detail view.
func TopExchanges(breakdown map[string]int, n int) []string {
	if n <= 0 || len(breakdown) == 0 {
		return []string{}
	}
	type kv struct {
		name  string
		count int
	}
	pairs := make([]kv, 0, len(breakdown))
	for k, v := range breakdown {
		pairs = append(pairs, kv{name: k, count: v})
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].count != pairs[j].count {
			return pairs[i].count > pairs[j].count
		}
		return pairs[i].name < pairs[j].name
	})
	if n > len(pairs) {
		n = len(pairs)
	}
	out := make([]string, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, pairs[i].name)
	}
	return out
}
