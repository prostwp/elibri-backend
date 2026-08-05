package api

// prefs_handlers.go — watchlist + notification preferences (sub-chat C).
//
// Endpoints (all auth-required — NOT in publicPaths):
//   GET /api/v1/watchlist            -> {"symbols":[...]}
//   PUT /api/v1/watchlist            {symbols:[...]} -> {"symbols":[...]} (normalised)
//   GET /api/v1/notification-prefs   -> alert_settings object (defaults if none)
//   PUT /api/v1/notification-prefs   object -> merged+saved object
//
// Both back onto the existing user_preferences table (migration 002): PK
// user_id, watchlist TEXT[], alert_settings JSONB. The row may not exist yet —
// every write is an upsert (INSERT ... ON CONFLICT (user_id) DO UPDATE), every
// read tolerates a missing row by returning defaults.
//
// Pure helpers (normalizeWatchlist, mergeNotificationPrefs, defaultNotificationPrefs,
// isValidFrequency) are extracted so they can be table-tested without Postgres
// — see prefs_handlers_test.go.

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/prostwp/elibri-backend/internal/auth"
	"github.com/prostwp/elibri-backend/internal/store"
)

// Watchlist validation limits (contract §B).
const (
	maxWatchlistSymbols   = 50
	maxWatchlistSymbolLen = 20
)

// validFrequencies is the closed enum for notification-prefs.frequency.
var validFrequencies = map[string]struct{}{
	"realtime": {},
	"hourly":   {},
	"daily":    {},
}

func isValidFrequency(s string) bool { _, ok := validFrequencies[s]; return ok }

// objectFields are the notification-pref keys that MUST be JSON objects when
// present. A PUT may legally omit them (partial update), but if supplied they
// cannot be null/scalar/array — otherwise mergeObject would persist the bad
// value verbatim (e.g. "channels": null), and the next GET would crash the
// frontend's prefs.channels.email dereference on every load.
var objectFields = []string{"channels", "per_scenario"}

// validateObjectFields checks that every objectFields key, IF present in the
// incoming map, decodes to a JSON object (map[string]any). Returns the first
// offending key and ok=false on violation; ("", true) when all present ones
// are objects (or absent). Pure — no DB.
func validateObjectFields(incoming map[string]any) (badKey string, ok bool) {
	for _, key := range objectFields {
		if v, present := incoming[key]; present {
			if _, isObj := v.(map[string]any); !isObj {
				return key, false
			}
		}
	}
	return "", true
}

// normalizeWatchlist trims, uppercases, drops empty/oversized symbols, dedups
// (preserving first-seen order), and caps the list at maxWatchlistSymbols.
// Pure — no DB. Symbols longer than maxWatchlistSymbolLen are dropped (not
// truncated) so we never silently store a mangled ticker.
func normalizeWatchlist(in []string) []string {
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, raw := range in {
		s := strings.ToUpper(strings.TrimSpace(raw))
		if s == "" || len(s) > maxWatchlistSymbolLen {
			continue
		}
		if _, dup := seen[s]; dup {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
		if len(out) >= maxWatchlistSymbols {
			break
		}
	}
	return out
}

// defaultNotificationPrefs is the canonical default object (contract §B):
// every channel on except telegram, no digest, critical alerts on, realtime
// frequency, no per-scenario overrides. Returned when the row/JSONB is absent.
func defaultNotificationPrefs() map[string]any {
	return map[string]any{
		"channels": map[string]any{
			"email":     true,
			"telegram":  false,
			"dashboard": true,
		},
		"daily_digest":    false,
		"critical_alerts": true,
		"frequency":       "realtime",
		"per_scenario":    map[string]any{},
	}
}

// mergeNotificationPrefs overlays `incoming` onto `base` and returns the
// result. Semantics (contract §B — "PUT merges, doesn't clobber unknown
// keys"):
//   - Unknown/forward-compat top-level keys in base survive untouched.
//   - The `channels` and `per_scenario` objects are merged per-key (a PUT of
//     just one channel doesn't wipe the others).
//   - `frequency`, if present, must already be validated by the caller; an
//     absent/empty frequency leaves base's value intact.
//   - Other scalar keys overwrite.
//
// Pure — no DB. Always returns a fresh map (does not mutate base in place for
// the nested objects).
func mergeNotificationPrefs(base, incoming map[string]any) map[string]any {
	out := make(map[string]any, len(base)+len(incoming))
	for k, v := range base {
		out[k] = v
	}
	for k, v := range incoming {
		switch k {
		case "channels", "per_scenario":
			out[k] = mergeObject(out[k], v)
		default:
			out[k] = v
		}
	}
	return out
}

// mergeObject shallow-merges two values that are expected to be JSON objects.
// If either side isn't an object, `incoming` wins (so a client can replace a
// scalar with an object or vice-versa). Returns a fresh map when merging.
func mergeObject(baseVal, incomingVal any) any {
	baseMap, baseOK := baseVal.(map[string]any)
	incMap, incOK := incomingVal.(map[string]any)
	if !incOK {
		return incomingVal
	}
	if !baseOK {
		return incMap
	}
	merged := make(map[string]any, len(baseMap)+len(incMap))
	for k, v := range baseMap {
		merged[k] = v
	}
	for k, v := range incMap {
		merged[k] = v
	}
	return merged
}

// ── Watchlist ──────────────────────────────────────────────────────────────

type watchlistReq struct {
	Symbols []string `json:"symbols"`
}

func handleWatchlistRouter(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		handleWatchlistGet(w, r)
	case http.MethodPut:
		handleWatchlistPut(w, r)
	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func handleWatchlistGet(w http.ResponseWriter, r *http.Request) {
	if !requireDB(w) {
		return
	}
	userID := auth.GetUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	var symbols []string
	err := store.Pool.QueryRow(r.Context(),
		`SELECT COALESCE(watchlist, '{}') FROM user_preferences WHERE user_id = $1`, userID,
	).Scan(&symbols)
	if err == pgx.ErrNoRows {
		writeJSON(w, map[string]any{"symbols": []string{}})
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to load watchlist")
		return
	}
	if symbols == nil {
		symbols = []string{}
	}
	writeJSON(w, map[string]any{"symbols": symbols})
}

func handleWatchlistPut(w http.ResponseWriter, r *http.Request) {
	if !requireDB(w) {
		return
	}
	userID := auth.GetUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	var req watchlistReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	symbols := normalizeWatchlist(req.Symbols)

	// Upsert: the row may not exist yet. ON CONFLICT keys on the user_id PK.
	if _, err := store.Pool.Exec(r.Context(), `
		INSERT INTO user_preferences (user_id, watchlist, updated_at)
		VALUES ($1, $2, NOW())
		ON CONFLICT (user_id) DO UPDATE
		   SET watchlist = EXCLUDED.watchlist, updated_at = NOW()
	`, userID, symbols); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to save watchlist")
		return
	}
	writeJSON(w, map[string]any{"symbols": symbols})
}

// ── Notification prefs ─────────────────────────────────────────────────────

func handleNotificationPrefsRouter(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		handleNotificationPrefsGet(w, r)
	case http.MethodPut:
		handleNotificationPrefsPut(w, r)
	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

// loadAlertSettings reads alert_settings JSONB for a user, returning the
// parsed map merged onto defaults (so missing keys are always populated).
// A missing row or empty {} yields the full default object.
func loadAlertSettings(r *http.Request, userID string) (map[string]any, error) {
	var raw []byte
	err := store.Pool.QueryRow(r.Context(),
		`SELECT COALESCE(alert_settings, '{}'::jsonb) FROM user_preferences WHERE user_id = $1`, userID,
	).Scan(&raw)
	if err == pgx.ErrNoRows {
		return defaultNotificationPrefs(), nil
	}
	if err != nil {
		return nil, err
	}
	stored := map[string]any{}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &stored); err != nil {
			// Corrupt JSON on the row — fall back to defaults rather than 500.
			return defaultNotificationPrefs(), nil
		}
	}
	// Merge stored over defaults so the response always has the full shape.
	return mergeNotificationPrefs(defaultNotificationPrefs(), stored), nil
}

func handleNotificationPrefsGet(w http.ResponseWriter, r *http.Request) {
	if !requireDB(w) {
		return
	}
	userID := auth.GetUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	prefs, err := loadAlertSettings(r, userID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to load notification prefs")
		return
	}
	writeJSON(w, prefs)
}

func handleNotificationPrefsPut(w http.ResponseWriter, r *http.Request) {
	if !requireDB(w) {
		return
	}
	userID := auth.GetUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	incoming := map[string]any{}
	if err := json.NewDecoder(r.Body).Decode(&incoming); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}

	// Validate frequency if the client sent one. Anything else is allowed
	// through (forward-compat keys for sub-chat F).
	if f, ok := incoming["frequency"]; ok {
		fs, isStr := f.(string)
		if !isStr || !isValidFrequency(fs) {
			writeError(w, http.StatusBadRequest, "frequency must be one of: realtime, hourly, daily")
			return
		}
	}

	// Reject a present-but-non-object channels/per_scenario before merging
	// (see validateObjectFields for why a null/scalar here corrupts the JSONB
	// and crashes the frontend on the next GET).
	if badKey, ok := validateObjectFields(incoming); !ok {
		writeError(w, http.StatusBadRequest, badKey+" must be an object")
		return
	}

	// Merge onto the user's CURRENT stored prefs (already merged over defaults
	// by loadAlertSettings), so a partial PUT preserves untouched keys.
	current, err := loadAlertSettings(r, userID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to load notification prefs")
		return
	}
	merged := mergeNotificationPrefs(current, incoming)

	encoded, err := json.Marshal(merged)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to encode notification prefs")
		return
	}

	if _, err := store.Pool.Exec(r.Context(), `
		INSERT INTO user_preferences (user_id, alert_settings, updated_at)
		VALUES ($1, $2::jsonb, NOW())
		ON CONFLICT (user_id) DO UPDATE
		   SET alert_settings = EXCLUDED.alert_settings, updated_at = NOW()
	`, userID, string(encoded)); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to save notification prefs")
		return
	}
	writeJSON(w, merged)
}
