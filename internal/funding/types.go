// Package funding implements the liquidations layer for the Funding Rate /
// Perp Pressure scenario: a long-lived WebSocket consumer of Binance Futures'
// public all-market force-order stream, a thread-safe in-memory rolling window,
// price-magnet zone aggregation, and the public REST handler interface.
//
// Unlike internal/whale (a 10-min HTTP poll → Postgres snapshot leaderboard),
// funding is a PUSH consumer with NO database in the primary path: events
// arrive continuously over a persistent WS, accumulate in a ~1h ring, and the
// endpoint serves the live feed + aggregated zones. Cold-restart loses the
// window — acceptable, the feed naturally refills within minutes (the source
// has no public REST backfill).
//
// Side mapping (Binance forceOrder semantics, curl-verified 2026-05-30):
//   o.S == "SELL" ⇒ a LONG position was force-closed → side "long_liq" (▼ flush)
//   o.S == "BUY"  ⇒ a SHORT position was force-closed → side "short_liq" (▲ flush)
//
// This file holds types only. The ring lives in store.go; the WS consumer in
// worker.go; the read interface + handler glue in handler.go.
package funding

// Side labels which kind of position was liquidated. Rendered ▲/▼ on the
// frontend — never as long/short trade language (DESIGN_SYSTEM §6 / spec §5).
//
// Declared as a string-backed alias rather than a distinct type so the JSON
// contract stays plain strings and callers (handler, tests) can compare
// against the SideLongLiq / SideShortLiq constants without conversions.
const (
	// SideLongLiq: a long was force-closed (forceOrder S=SELL). Reads as a ▼
	// downward flush.
	SideLongLiq = "long_liq"
	// SideShortLiq: a short was force-closed (forceOrder S=BUY). Reads as a ▲
	// upward flush.
	SideShortLiq = "short_liq"
)

// Liquidation is one force-order event, normalized from the raw
// !forceOrder@arr payload. JSON tags match the data contract (spec §5) exactly.
type Liquidation struct {
	Symbol string `json:"symbol"`
	// Side is "long_liq" | "short_liq" (see the Side constants).
	Side string `json:"side"`
	// Qty is the filled quantity (forceOrder o.z).
	Qty float64 `json:"qty"`
	// Price is the average fill price (forceOrder o.ap, fallback o.p).
	Price float64 `json:"price"`
	// USDValue = Price × Qty; 0 when the price was unavailable → UI renders "—".
	USDValue float64 `json:"usd_value"`
	// TS is the event time (forceOrder o.T) as a UTC RFC3339 string. Stamped at
	// parse time so the wire payload is a stable ISO string, not a time.Time
	// zero-value when the field is absent.
	TS string `json:"ts"`
}

// Zone is a price-magnet band: liquidations for one symbol bucketed into a
// price range, with the summed USD value, event count, and the dominant side.
// Magnet zones are a non-interactive display (discovery §3). JSON tags match
// the data contract (spec §5).
type Zone struct {
	Symbol string `json:"symbol"`
	// PriceBand is a human-readable range, e.g. "9900-9950".
	PriceBand string `json:"price_band"`
	// TotalUSD is the summed usd_value of every liquidation in the band.
	TotalUSD float64 `json:"total_usd"`
	// Count is the number of liquidation events in the band.
	Count int `json:"count"`
	// Side is the dominant side (by summed USD) in the band.
	Side string `json:"side"`
}

// Response is the JSON body of GET /api/v1/funding/liquidations. Slices are
// always non-nil (the handler / store guarantee `[]`, never `null`) so the
// frontend's `.map()` / `.length` checks hold even before the WS delivers
// anything.
type Response struct {
	// CapturedAt is the server time the response was assembled (UTC RFC3339),
	// or "" when the window is empty. Events are continuous, so unlike whale
	// (which uses the newest snapshot's timestamp) this is simply now().
	CapturedAt string        `json:"captured_at"`
	Feed       []Liquidation `json:"feed"`
	Zones      []Zone        `json:"zones"`
}
