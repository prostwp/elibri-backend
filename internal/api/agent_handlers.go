package api

// agent_handlers.go — Public agent output endpoints (MVP-mock).
//
// This file owns the read-only "what does the agent produce" surface that the
// catalog detail page hits to render the "Live output preview" section. It is
// deliberately separate from the live runner / narrative pipeline files —
// nothing here touches Redis Streams, Postgres queries against narrative
// snapshots, or any background worker. For the May-19 demo we serve a
// hand-crafted JSON shaped exactly to the spec so the frontend can wire
// against a stable contract before the real data feeds (X / Reddit / Telegram
// / on-chain) are integrated.
//
// Endpoint:
//   GET /api/v1/agents/:slug/output   (public, no auth)
//
// The slug is currently restricted to a known set (only "narrative_radar"
// today). Adding a second agent is an `agentSamples[slug] = ...` line plus
// embedding another JSON file — no router change needed.
//
// JWT bypass for the prefix /api/v1/agents/ lives in internal/auth/jwt.go's
// publicPrefixes — anonymous GETs are accepted, JWT-bearing GETs are accepted
// but the user_id is unused here (no per-user customisation in MVP).

import (
	_ "embed"
	"encoding/json"
	"net/http"
	"time"
)

// narrativeRadarSampleJSON is the embedded sample JSON for the Narrative
// Radar agent's output. The file holds the static fields (narratives,
// historical_stats, agent_name, agent_version, timeframe, confidence, risk,
// disclaimer); the time-sensitive fields (generated_at, window_start,
// window_end, next_refresh_at) are computed per request in serveAgentOutput.
//
// We use //go:embed (Go 1.16+) so the binary is self-contained — no runtime
// filesystem dependency, no "where's my JSON" surprise during a redeploy.
//
//go:embed agents/narrative_radar_sample.json
var narrativeRadarSampleJSON []byte

// nextRefreshInterval is the gap between generated_at and next_refresh_at in
// the response. 15 minutes matches the spec ("Refresh cadence: next_refresh_at
// in JSON — каждые 15 минут реалистично"). Kept as a const so a v1.1 change
// to cadence is one line, not a code search.
const nextRefreshInterval = 15 * time.Minute

// windowDuration is how far back the 48h "observation window" extends from
// generated_at. The spec's `timeframe` field ("48h") is hard-coded in the
// embedded JSON; this constant only drives the computed window_start. Keeping
// the two in sync is a manual responsibility — if we ever ship multiple
// timeframes we'll move both into a per-agent config struct.
const windowDuration = 48 * time.Hour

// agentOutputEnvelope is the wire shape merged from the embedded sample plus
// the computed time-sensitive fields. We model it as a map for two reasons:
//
//  1. The sample JSON is authored by product (it changes shape regularly while
//     the agent spec is still in flux). A typed struct would force a Go-side
//     update every time a new field appears in the spec — a friction point we
//     don't want during the pre-demo iteration window.
//
//  2. The computed fields (generated_at, window_*, next_refresh_at) are
//     conceptually injected on top of the static payload. A merge is cleaner
//     than redeclaring every field in a struct.
//
// The exported handler is the only writer; we never decode a request body
// into this shape, so the map-vs-struct flexibility comes at zero risk.

// serveAgentOutput is wired in NewRouter as GET /api/v1/agents/{slug}/output.
//
// Behaviour:
//   - Looks up the slug in agentSamples; 404 if unknown.
//   - Decodes the embedded sample JSON (cheap — ~2 KB).
//   - Stamps generated_at = now, window_end = now, window_start = now - 48h,
//     next_refresh_at = now + 15m. All in UTC, RFC3339.
//   - Returns the merged object as application/json.
//
// The decode-merge-encode pattern adds ~50µs per request — not a bottleneck
// at the scale we expect (the detail page is the only caller for now). A
// premature optimisation here would be to pre-parse the sample at init time;
// we keep the per-request decode so a hot-reload of the embedded file (during
// dev with -ldflags or a future runtime override) Just Works.
func serveAgentOutput(w http.ResponseWriter, r *http.Request) {
	slug := r.PathValue("slug")
	raw, ok := agentSamples[slug]
	if !ok {
		writeError(w, http.StatusNotFound, "agent not found")
		return
	}

	// Decode into a generic map so we can layer computed time fields on top
	// without enumerating every static field. json.Decoder over a Reader
	// would also work but Unmarshal is fine for a ~2 KB blob.
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		// This is a "we shipped a broken JSON file" error — show 500 so
		// the operator notices in logs / Sentry, not a 200 with garbage.
		writeError(w, http.StatusInternalServerError, "failed to decode agent sample")
		return
	}

	now := time.Now().UTC()
	payload["generated_at"] = now.Format(time.RFC3339)
	payload["window_end"] = now.Format(time.RFC3339)
	payload["window_start"] = now.Add(-windowDuration).Format(time.RFC3339)
	payload["next_refresh_at"] = now.Add(nextRefreshInterval).Format(time.RFC3339)

	writeJSON(w, payload)
}

// agentSamples maps a public agent slug to its embedded sample JSON blob.
// New agents are added by:
//  1. Drop a JSON file into internal/api/agents/.
//  2. Add a //go:embed directive + var at the top of this file.
//  3. Register the var here against the public slug.
//
// We deliberately keep this map small and explicit instead of auto-loading
// every JSON in the directory — the public slug is part of the URL contract
// and must be reviewable in code review, not implied by a filename.
var agentSamples = map[string][]byte{
	"narrative_radar": narrativeRadarSampleJSON,
}
