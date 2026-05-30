package api

// whale_handlers.go — Whale Flow REST surface.
//
// AUTH: GET /api/v1/whale-flow is registered on the same mux that
// auth.Middleware wraps in NewRouter, and is allowlisted in
// internal/auth/jwt.go publicPaths (no per-user data — same public tier as
// /api/v1/narratives). No extra wiring needed here.
//
// The handler reads the whale.SnapshotReader interface (not *Store directly)
// so internal/api/whale_handlers_test.go can pass a fake reader and exercise
// the limit-clamping / empty-store paths offline — exactly like
// handleNarrativesList.

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prostwp/elibri-backend/internal/whale"
)

const (
	// whaleTransfersDefaultLimit is the live-feed page size when ?limit is
	// omitted. 50 matches the spec §1 default.
	whaleTransfersDefaultLimit = 50

	// whaleTransfersMaxLimit caps the per-request live-feed page. 100 per the
	// spec — bounds the row count returned per public (no-auth) request.
	whaleTransfersMaxLimit = 100
)

// handleWhaleFlowList serves GET /api/v1/whale-flow.
//
// Returns the latest snapshot per asset (sorted by |net_flow_usd_24h| DESC by
// the store) plus the live transfer feed (newest-first). One optional query
// parameter:
//
//	limit  int 1..100  — number of transfers in the feed (default 50).
//
// Response shape (spec §1):
//
//	{
//	  "captured_at": "2026-05-30T00:49:00Z",   // newest snapshot's captured_at, or ""
//	  "flows": [ {Snapshot}, ... ],
//	  "transfers": [ {WhaleTransfer}, ... ]
//	}
//
// When the store has no rows yet (worker hasn't run, or first deploy), returns
// 200 with empty flows + transfers slices and captured_at = "".
func handleWhaleFlowList(reader whale.SnapshotReader) http.HandlerFunc {
	type response struct {
		CapturedAt string                `json:"captured_at"`
		Flows      []whale.Snapshot      `json:"flows"`
		Transfers  []whale.WhaleTransfer `json:"transfers"`
	}

	return func(w http.ResponseWriter, r *http.Request) {
		// limit: default 50, clamped to [1, 100]. Out-of-range / non-integer
		// silently falls back to the default (matches the tolerance the
		// narrative endpoints apply to malformed query strings).
		limit := whaleTransfersDefaultLimit
		if v := r.URL.Query().Get("limit"); v != "" {
			if n, err := strconv.Atoi(v); err == nil && n >= 1 && n <= whaleTransfersMaxLimit {
				limit = n
			}
		}

		flows, err := reader.LatestSnapshots(r.Context())
		if err != nil {
			writeError(w, http.StatusInternalServerError, "failed to load whale flows")
			return
		}

		transfers, err := reader.RecentTransfers(r.Context(), limit)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "failed to load whale transfers")
			return
		}

		// Defensive: never emit a nil slice — the frontend's `.map()` / `.length`
		// checks depend on `[]` not `null`.
		if flows == nil {
			flows = []whale.Snapshot{}
		}
		if transfers == nil {
			transfers = []whale.WhaleTransfer{}
		}

		// captured_at — newest snapshot's captured_at across the flows. This
		// is the value the frontend uses for "last refreshed Xm ago" display.
		var latest time.Time
		for _, s := range flows {
			if !s.CapturedAt.IsZero() && s.CapturedAt.After(latest) {
				latest = s.CapturedAt
			}
		}
		capturedAt := ""
		if !latest.IsZero() {
			capturedAt = latest.UTC().Format(time.RFC3339)
		}

		writeJSON(w, response{
			CapturedAt: capturedAt,
			Flows:      flows,
			Transfers:  transfers,
		})
	}
}
