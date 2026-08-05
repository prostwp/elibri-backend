package api

// follows_handlers.go — user "follows a public scenario" (sub-chat C).
//
// Endpoints (all auth-required — NOT in publicPaths):
//   POST   /api/v1/follows                 {strategy_id} -> {"status":"ok"}
//   DELETE /api/v1/follows/{strategy_id}                 -> {"status":"ok"}
//   GET    /api/v1/follows                               -> {"follows":[card,...]}
//
// Follow != clone. A follow is a lightweight bookmark row in user_follows
// (migration 014). The card returned by GET joins user_follows ⋈ strategies
// and surfaces only PUBLIC scenarios (is_public = TRUE) — a scenario that was
// unpublished or deleted simply drops out of the list (CASCADE removes the
// follow on delete; the WHERE filter hides unpublished ones).

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/prostwp/elibri-backend/internal/auth"
	"github.com/prostwp/elibri-backend/internal/store"
)

type followReq struct {
	StrategyID string `json:"strategy_id"`
}

// followCard is the GET /follows row shape (contract §B). slug == author_slug
// so the frontend can route to /scenarios/{slug}.
type followCard struct {
	StrategyID  string `json:"strategy_id"`
	Slug        string `json:"slug"`
	Name        string `json:"name"`
	AuthorName  string `json:"author_name"`
	PreviewChip string `json:"preview_chip"`
	RiskTier    string `json:"risk_tier"`
	Interval    string `json:"interval"`
	IsPremium   bool   `json:"is_premium"`
	CloneCount  int    `json:"clone_count"`
	FollowedAt  string `json:"followed_at"`
}

// handleFollowsRouter dispatches the /api/v1/follows{,/} family. Go's mux
// gives us POST/GET on the exact path and DELETE on the .../{id} subpath via
// separate registrations in router.go, so this single entry point only needs
// to branch on method for the collection path. DELETE-with-id is handled by
// handleUnfollow (registered on the subpath pattern).
func handleFollowsRouter(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		handleFollowCreate(w, r)
	case http.MethodGet:
		handleFollowsList(w, r)
	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

// handleFollowCreate inserts a follow. Idempotent: following an
// already-followed scenario returns 200 (ON CONFLICT DO NOTHING) rather than
// 409 — re-clicking "follow" should never error. 404 if the target is not a
// public scenario (private/nonexistent strategy_id).
func handleFollowCreate(w http.ResponseWriter, r *http.Request) {
	if !requireDB(w) {
		return
	}
	userID := auth.GetUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	var req followReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	req.StrategyID = strings.TrimSpace(req.StrategyID)
	if req.StrategyID == "" {
		writeError(w, http.StatusBadRequest, "strategy_id required")
		return
	}

	// Only public scenarios are followable. Verify existence + public flag
	// first so we return a clean 404 (and never insert a follow pointing at a
	// private row). A malformed UUID also lands here as 404 via the type
	// mismatch path — we guard it explicitly below to avoid a 500.
	var exists bool
	if err := store.Pool.QueryRow(r.Context(),
		`SELECT EXISTS(SELECT 1 FROM strategies WHERE id = $1 AND is_public = TRUE)`,
		req.StrategyID,
	).Scan(&exists); err != nil {
		// Invalid UUID syntax (or any other lookup error) -> treat as not
		// found rather than leaking a 500 for a bad client-supplied id.
		writeError(w, http.StatusNotFound, "scenario not found")
		return
	}
	if !exists {
		writeError(w, http.StatusNotFound, "scenario not found")
		return
	}

	if _, err := store.Pool.Exec(r.Context(),
		`INSERT INTO user_follows (user_id, strategy_id)
		 VALUES ($1, $2)
		 ON CONFLICT ON CONSTRAINT user_follows_uq DO NOTHING`,
		userID, req.StrategyID,
	); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to follow scenario")
		return
	}
	writeJSON(w, map[string]string{"status": "ok"})
}

// handleUnfollow removes a follow by strategy_id (path param). 404 if there
// was no such follow for this user. Registered on the /api/v1/follows/{id}
// subpath (DELETE).
func handleUnfollow(w http.ResponseWriter, r *http.Request) {
	if !requireDB(w) {
		return
	}
	userID := auth.GetUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	strategyID := strings.TrimSpace(r.PathValue("strategy_id"))
	if strategyID == "" {
		writeError(w, http.StatusBadRequest, "strategy_id required")
		return
	}

	tag, err := store.Pool.Exec(r.Context(),
		`DELETE FROM user_follows WHERE user_id = $1 AND strategy_id = $2`,
		userID, strategyID,
	)
	if err != nil {
		// Bad UUID etc. — nothing to delete, report 404 not 500.
		writeError(w, http.StatusNotFound, "follow not found")
		return
	}
	if tag.RowsAffected() == 0 {
		writeError(w, http.StatusNotFound, "follow not found")
		return
	}
	writeJSON(w, map[string]string{"status": "ok"})
}

// handleFollowsList returns the user's followed PUBLIC scenarios as cards,
// newest-followed first. Unpublished/deleted scenarios are filtered out by
// is_public = TRUE (deleted ones are already gone via CASCADE).
func handleFollowsList(w http.ResponseWriter, r *http.Request) {
	if !requireDB(w) {
		return
	}
	userID := auth.GetUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	rows, err := store.Pool.Query(r.Context(), `
		SELECT s.id,
		       COALESCE(s.author_slug, ''),
		       s.name,
		       COALESCE(s.author_name, ''),
		       COALESCE(s.preview_chip, ''),
		       COALESCE(s.risk_tier, 'balanced'),
		       COALESCE(s.interval, '1h'),
		       COALESCE(s.is_premium, false),
		       COALESCE(s.clone_count, 0),
		       uf.created_at
		FROM user_follows uf
		JOIN strategies s ON s.id = uf.strategy_id
		WHERE uf.user_id = $1 AND s.is_public = TRUE
		ORDER BY uf.created_at DESC
	`, userID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to list follows")
		return
	}
	defer rows.Close()

	out := []followCard{}
	for rows.Next() {
		var (
			c          followCard
			followedAt time.Time
		)
		if err := rows.Scan(
			&c.StrategyID, &c.Slug, &c.Name, &c.AuthorName, &c.PreviewChip,
			&c.RiskTier, &c.Interval, &c.IsPremium, &c.CloneCount, &followedAt,
		); err != nil {
			continue
		}
		c.FollowedAt = followedAt.UTC().Format(time.RFC3339)
		out = append(out, c)
	}
	writeJSON(w, map[string]any{"follows": out})
}
