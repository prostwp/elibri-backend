package api

// funding_handlers.go — Funding Rate / liquidations REST surface.
//
// AUTH: GET /api/v1/funding/liquidations is registered on the same mux that
// auth.Middleware wraps in NewRouter, and is allowlisted in
// internal/auth/jwt.go publicPrefixes ("/api/v1/funding/") — same public tier
// as whale-flow, no per-user data.
//
// WIRING: unlike whale (a poolbacked store the router builds inline as
// whale.NewStore(store.Pool)), the funding store is a PROCESS SINGLETON — the
// WS worker writes to it, this handler reads from it, and it has no DB. So it's
// threaded in via SetFundingStore() from cmd/server/main.go, exactly mirroring
// SetIdeaGenerator. The handler reads the funding.LiquidationReader interface
// (not *Store) so internal/api/funding_handlers_test.go can pass a fake and
// exercise the limit-clamping / empty paths offline.

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prostwp/elibri-backend/internal/funding"
)

const (
	// fundingFeedDefaultLimit is the live-feed page size when ?limit is omitted.
	fundingFeedDefaultLimit = 40

	// fundingFeedMaxLimit caps the per-request feed. 200 per the task contract —
	// bounds the row count returned per public (no-auth) request.
	fundingFeedMaxLimit = 200
)

// fundingStore is the package-level liquidation reader wired in from
// cmd/server/main.go via SetFundingStore. nil → the handler serves an empty
// (but well-formed) response, so unit tests and a DB-less / worker-less boot
// don't have to plumb anything.
//
// Mirrors the narrativeIdeaGen / SetIdeaGenerator pattern: a single global
// wired once at boot, never mutated thereafter.
var fundingStore funding.LiquidationReader

// SetFundingStore wires the liquidation store the funding handler reads from.
// Called once from cmd/server/main.go with the same *funding.Store the WS
// worker writes to. nil leaves the handler serving an empty feed (200), which
// is the correct cold/degraded behaviour — the frontend's liquidation section
// shows its own empty state while the rest of the page works.
func SetFundingStore(r funding.LiquidationReader) { fundingStore = r }

// handleFundingLiquidations serves GET /api/v1/funding/liquidations.
//
// One optional query parameter:
//
//	limit  int 1..200  — number of liquidations in the feed (default 40).
//
// Response shape (spec §5):
//
//	{
//	  "captured_at": "2026-05-30T00:49:00Z",   // server time, UTC RFC3339
//	  "feed":  [ {Liquidation}, ... ],          // newest-first
//	  "zones": [ {Zone}, ... ]                  // top bands by total USD
//	}
//
// Empty store (worker not connected yet, or no liquidations in the window) →
// 200 with feed:[] / zones:[] and captured_at = "" (NOT 500, NOT skeleton-
// forever). This keeps section #4 degrading gracefully on its own.
func handleFundingLiquidations(w http.ResponseWriter, r *http.Request) {
	// limit: default 40, clamped to [1, 200]. Out-of-range / non-integer
	// silently falls back to the default (same tolerance the whale + narrative
	// endpoints apply to malformed query strings).
	limit := fundingFeedDefaultLimit
	if v := r.URL.Query().Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 && n <= fundingFeedMaxLimit {
			limit = n
		}
	}

	// nil store (not wired / DB-less boot) → empty but well-formed response.
	if fundingStore == nil {
		writeJSON(w, funding.Response{
			CapturedAt: "",
			Feed:       []funding.Liquidation{},
			Zones:      []funding.Zone{},
		})
		return
	}

	feed := fundingStore.Recent(limit)
	zones := fundingStore.Zones()

	// Defensive: never emit a nil slice — the frontend's `.map()` / `.length`
	// checks depend on `[]` not `null`. (The store already returns non-nil, but
	// a future reader impl might not.)
	if feed == nil {
		feed = []funding.Liquidation{}
	}
	if zones == nil {
		zones = []funding.Zone{}
	}

	// captured_at = server time the response was assembled. The store is wired
	// (non-nil, checked above), so the worker is running and this is a live
	// push feed — stamp now() even on a quiet (empty) window so the frontend
	// can tell "live but quiet" from "worker never connected" (the nil-store
	// path above returns ""). The frontend gates its LIVE pill on hasFeed, so
	// this never claims a false "live" on an empty feed.
	writeJSON(w, funding.Response{
		CapturedAt: time.Now().UTC().Format(time.RFC3339),
		Feed:       feed,
		Zones:      zones,
	})
}
