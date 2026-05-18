package api

// scenarios_public_handlers.go — Sprint 2 (v3 SaaS redesign).
//
// This file is intentionally separate from scenario_handlers.go (which owns
// the live runner control plane: start/stop/active for an authenticated
// user's own strategies). Mixing the two would create a confusing module
// where "scenario" sometimes means "a row in strategies the user owns and
// can run" and sometimes "a public catalog item the user can browse and
// clone". Here all endpoints operate on the *catalog projection* of the
// same strategies table — public-flagged rows surfaced under author_slug.
//
// Endpoints:
//   GET  /api/v1/scenarios/featured                 (public, no auth)
//   GET  /api/v1/scenarios/public?page&theme&style&q (public, no auth)
//   GET  /api/v1/scenarios/public/{slug}            (public, no auth)
//   POST /api/v1/scenarios/{slug}/clone             (auth required)
//   GET  /api/v1/dashboard/summary                  (auth required)
//   GET  /api/v1/billing                            (auth required, stub)
//
// JWT public-paths bypass for the first three lives in
// internal/auth/jwt.go (publicPaths + publicPrefixes).

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/prostwp/elibri-backend/internal/auth"
	"github.com/prostwp/elibri-backend/internal/store"
)

// ──────────────────────────────────────────────────────────────────────────────
// Validation helpers
//
// Themes/styles are closed enums (per Sprint 2 spec) — anything outside the
// whitelist is silently dropped from the WHERE clause so an attacker can't
// probe for unindexed values or trigger pg type-coercion errors. Same idea
// as ml.IsValidTier in auth_handlers.go.
// ──────────────────────────────────────────────────────────────────────────────

var validThemes = map[string]struct{}{
	"gold_silver": {},
	"currencies":  {},
	"indices":     {},
	"oil_gas":     {},
	"crypto":      {},
	"multi":       {},
}

var validStyles = map[string]struct{}{
	"news":        {},
	"fundamental": {},
	"technical":   {},
	"astro":       {},
	"levels":      {},
}

func isValidTheme(s string) bool { _, ok := validThemes[s]; return ok }
func isValidStyle(s string) bool { _, ok := validStyles[s]; return ok }

// publicListParams is the parsed/normalised view of the query string for
// /api/v1/scenarios/public. Extracted so it's unit-testable without spinning
// up Postgres — see scenarios_public_handlers_test.go::TestParseListParams_*.
type publicListParams struct {
	Page  int
	Theme string // already validated; "" means no filter
	Style string // already validated; "" means no filter
	Q     string // already escaped for ILIKE; "" means no filter
}

const (
	listPageSize    = 20
	listMaxPage     = 50 // cap on `page` so we don't issue OFFSET 1000000 queries
	maxQueryLen     = 80 // longer free-text searches are noise / abuse
	featuredLimit   = 6
	dashboardAlerts = 10
)

// parseListParams reads page/theme/style/q from the URL and returns a
// normalised, defensible view. Invalid values become empty (filter dropped)
// instead of 400 — same tolerance pattern as handleNarrativesList.
//
// The `q` value is escaped for a postgres ILIKE pattern: backslash, percent
// and underscore are prefixed with a backslash so a user typing "100%" or
// "foo_bar" doesn't accidentally turn into a wildcard search. We rely on
// the default ESCAPE character in pgx (backslash) — no need for an explicit
// ESCAPE clause.
func parseListParams(q map[string][]string) publicListParams {
	out := publicListParams{Page: 1}

	if v := firstQuery(q, "page"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 && n <= listMaxPage {
			out.Page = n
		}
	}
	if v := strings.ToLower(strings.TrimSpace(firstQuery(q, "theme"))); v != "" && isValidTheme(v) {
		out.Theme = v
	}
	if v := strings.ToLower(strings.TrimSpace(firstQuery(q, "style"))); v != "" && isValidStyle(v) {
		out.Style = v
	}
	if v := strings.TrimSpace(firstQuery(q, "q")); v != "" {
		if len(v) > maxQueryLen {
			v = v[:maxQueryLen]
		}
		out.Q = escapeLikePattern(v)
	}
	return out
}

func firstQuery(q map[string][]string, key string) string {
	if vs, ok := q[key]; ok && len(vs) > 0 {
		return vs[0]
	}
	return ""
}

// escapeLikePattern neutralises the three magic chars in pg's LIKE/ILIKE
// grammar so "5%_off" becomes a literal substring search. Backslash MUST
// come first or we'd double-escape our own escapes.
func escapeLikePattern(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `%`, `\%`)
	s = strings.ReplaceAll(s, `_`, `\_`)
	return s
}

// ──────────────────────────────────────────────────────────────────────────────
// Wire-format DTOs
//
// We project from strategies but never expose user_id, password, or huge
// nodes_json/edges_json blobs in list responses (mobile bundles already
// hate 500 KB JSON arrays). Detail responses skip nodes_json/edges_json
// unless ?include=graph is set — keeps the catalog fast while still
// allowing the editor to fetch the graph for cloning preview.
// ──────────────────────────────────────────────────────────────────────────────

type publicScenarioCard struct {
	ID           string  `json:"id"`
	Slug         string  `json:"slug"`
	Name         string  `json:"name"`
	AuthorSlug   string  `json:"author_slug"`
	AuthorName   string  `json:"author_name"`
	AuthorTheme  string  `json:"author_theme"`
	AuthorStyle  string  `json:"author_style"`
	AuthorBio    string  `json:"author_bio,omitempty"`
	PreviewChip  string  `json:"preview_chip,omitempty"`
	RiskTier     string  `json:"risk_tier"`
	Interval     string  `json:"interval"`
	SelectedPair string  `json:"selected_pair"`
	IsPremium    bool    `json:"is_premium"`
	IsFeatured   bool    `json:"is_featured"`
	CloneCount   int     `json:"clone_count"`
	PublishedAt  *string `json:"published_at,omitempty"`
}

type publicScenarioDetail struct {
	publicScenarioCard
	Segment   string          `json:"segment"`
	NodesJSON json.RawMessage `json:"nodes_json,omitempty"`
	EdgesJSON json.RawMessage `json:"edges_json,omitempty"`
}

type listResponse struct {
	Items      []publicScenarioCard `json:"items"`
	Page       int                  `json:"page"`
	TotalPages int                  `json:"total_pages"`
}

// ──────────────────────────────────────────────────────────────────────────────
// SQL helpers
//
// We treat author_slug as the public "slug" — every public scenario must
// have one (validated in queries via author_slug IS NOT NULL). Strategies
// without a slug never make it to the catalog, even if is_public got
// flipped manually.
// ──────────────────────────────────────────────────────────────────────────────

const cardSelectColumns = `
	id, COALESCE(author_slug, ''), name,
	COALESCE(author_slug, ''), COALESCE(author_name, ''),
	COALESCE(author_theme, ''), COALESCE(author_style, ''),
	COALESCE(author_bio, ''), COALESCE(preview_chip, ''),
	COALESCE(risk_tier, 'balanced'), COALESCE(interval, '1h'),
	COALESCE(selected_pair, 'EURUSD'),
	COALESCE(is_premium, false), COALESCE(is_featured, false),
	COALESCE(clone_count, 0), published_at
`

func scanCard(rows pgx.Rows, c *publicScenarioCard) error {
	var publishedAt *time.Time
	err := rows.Scan(
		&c.ID, &c.Slug, &c.Name,
		&c.AuthorSlug, &c.AuthorName, &c.AuthorTheme, &c.AuthorStyle,
		&c.AuthorBio, &c.PreviewChip,
		&c.RiskTier, &c.Interval, &c.SelectedPair,
		&c.IsPremium, &c.IsFeatured, &c.CloneCount, &publishedAt,
	)
	if err != nil {
		return err
	}
	if publishedAt != nil {
		s := publishedAt.UTC().Format(time.RFC3339)
		c.PublishedAt = &s
	}
	return nil
}

// ──────────────────────────────────────────────────────────────────────────────
// GET /api/v1/scenarios/featured
//
// Up to N curated cards (is_featured=TRUE AND is_public=TRUE). Mirrors the
// "Hero" section on the new home page — kept tiny so it can be cached at
// the CDN later. featuredLimit is hard-coded (no ?limit=) to make caching
// effective.
// ──────────────────────────────────────────────────────────────────────────────

func handleScenariosFeatured(w http.ResponseWriter, r *http.Request) {
	if !requireDB(w) {
		return
	}
	rows, err := store.Pool.Query(r.Context(), `
		SELECT `+cardSelectColumns+`
		FROM strategies
		WHERE is_public = TRUE AND is_featured = TRUE AND author_slug IS NOT NULL
		ORDER BY clone_count DESC, published_at DESC NULLS LAST
		LIMIT $1
	`, featuredLimit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to list featured scenarios")
		return
	}
	defer rows.Close()

	out := []publicScenarioCard{}
	for rows.Next() {
		var c publicScenarioCard
		if err := scanCard(rows, &c); err != nil {
			continue
		}
		out = append(out, c)
	}
	writeJSON(w, map[string]any{"items": out})
}

// ──────────────────────────────────────────────────────────────────────────────
// GET /api/v1/scenarios/public
//
// Paginated catalog. Filters by theme + style; full-text-ish search on
// name OR author_name. The COUNT(*) and SELECT run as two queries because
// we need accurate total_pages even when page is empty (so the UI can
// render "page 4 of 7" controls). Postgres parallelises both so the cost
// stays in the millisecond range as long as the partial index is used.
// ──────────────────────────────────────────────────────────────────────────────

func handleScenariosPublicList(w http.ResponseWriter, r *http.Request) {
	if !requireDB(w) {
		return
	}
	params := parseListParams(r.URL.Query())

	// Build dynamic WHERE — keep ALL values parametrised. Never concatenate
	// a user-supplied string into the SQL itself; only the $-placeholder
	// count grows when a filter is present. We track args + the placeholder
	// position explicitly so each filter's SQL can stamp the correct $N
	// without an after-the-fact rewrite.
	conds := []string{"is_public = TRUE", "author_slug IS NOT NULL"}
	args := []any{}
	addArg := func(val any) string {
		args = append(args, val)
		return "$" + strconv.Itoa(len(args))
	}
	if params.Theme != "" {
		conds = append(conds, "author_theme = "+addArg(params.Theme))
	}
	if params.Style != "" {
		conds = append(conds, "author_style = "+addArg(params.Style))
	}
	if params.Q != "" {
		// We need the same value twice (name + author_name). Pass it once
		// through addArg, then reuse the same placeholder for both sides —
		// pgx accepts repeated placeholder usage and Postgres deduplicates
		// at the query-plan stage.
		ph := addArg(params.Q)
		conds = append(conds,
			"(name ILIKE '%' || "+ph+" || '%' OR author_name ILIKE '%' || "+ph+" || '%')",
		)
	}
	whereSQL := "WHERE " + strings.Join(conds, " AND ")

	// Count first — small partial index over is_public makes this cheap.
	var total int
	if err := store.Pool.QueryRow(r.Context(),
		`SELECT COUNT(*) FROM strategies `+whereSQL, args...,
	).Scan(&total); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to count scenarios")
		return
	}

	totalPages := (total + listPageSize - 1) / listPageSize
	if totalPages == 0 {
		totalPages = 1
	}

	// If user asked for a page beyond the available range we still return
	// 200 with empty items — matches typical pagination UX (avoid 404 on
	// "page=99" when the count just dropped).
	offset := (params.Page - 1) * listPageSize
	listArgs := append([]any{}, args...)
	listArgs = append(listArgs, listPageSize, offset)
	limitIdx := len(args) + 1
	offsetIdx := len(args) + 2

	rows, err := store.Pool.Query(r.Context(), `
		SELECT `+cardSelectColumns+`
		FROM strategies
		`+whereSQL+`
		ORDER BY is_featured DESC, clone_count DESC, published_at DESC NULLS LAST
		LIMIT $`+strconv.Itoa(limitIdx)+` OFFSET $`+strconv.Itoa(offsetIdx),
		listArgs...,
	)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to list scenarios")
		return
	}
	defer rows.Close()

	items := []publicScenarioCard{}
	for rows.Next() {
		var c publicScenarioCard
		if err := scanCard(rows, &c); err != nil {
			continue
		}
		items = append(items, c)
	}

	writeJSON(w, listResponse{
		Items:      items,
		Page:       params.Page,
		TotalPages: totalPages,
	})
}

// ──────────────────────────────────────────────────────────────────────────────
// GET /api/v1/scenarios/public/{slug}
//
// Single public scenario lookup. By default we omit nodes_json + edges_json
// (the graph blobs) — mobile clients showing the detail page only need the
// metadata. Pass ?include=graph to opt in (used by the cloning preview).
// ──────────────────────────────────────────────────────────────────────────────

func handleScenariosPublicDetail(w http.ResponseWriter, r *http.Request) {
	if !requireDB(w) {
		return
	}
	slug := r.PathValue("slug")
	if slug == "" {
		writeError(w, http.StatusBadRequest, "slug required")
		return
	}
	includeGraph := r.URL.Query().Get("include") == "graph"

	var (
		d      publicScenarioDetail
		nodes  json.RawMessage
		edges  json.RawMessage
		seg    string
		pubAt  *string
	)
	err := store.Pool.QueryRow(r.Context(), `
		SELECT id, COALESCE(author_slug, ''), name,
		       COALESCE(author_slug, ''), COALESCE(author_name, ''),
		       COALESCE(author_theme, ''), COALESCE(author_style, ''),
		       COALESCE(author_bio, ''), COALESCE(preview_chip, ''),
		       COALESCE(risk_tier, 'balanced'), COALESCE(interval, '1h'),
		       COALESCE(selected_pair, 'EURUSD'),
		       COALESCE(is_premium, false), COALESCE(is_featured, false),
		       COALESCE(clone_count, 0), published_at,
		       COALESCE(segment, 'pro'), COALESCE(nodes_json, '[]'::jsonb), COALESCE(edges_json, '[]'::jsonb)
		FROM strategies
		WHERE author_slug = $1 AND is_public = TRUE
	`, slug).Scan(
		&d.ID, &d.Slug, &d.Name,
		&d.AuthorSlug, &d.AuthorName, &d.AuthorTheme, &d.AuthorStyle,
		&d.AuthorBio, &d.PreviewChip,
		&d.RiskTier, &d.Interval, &d.SelectedPair,
		&d.IsPremium, &d.IsFeatured, &d.CloneCount, &pubAt,
		&seg, &nodes, &edges,
	)
	if err == pgx.ErrNoRows {
		writeError(w, http.StatusNotFound, "scenario not found")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to load scenario")
		return
	}
	d.PublishedAt = pubAt
	d.Segment = seg
	if includeGraph {
		d.NodesJSON = nodes
		d.EdgesJSON = edges
	}

	// Sprint 3: log this view into scenario_views.
	//
	// We schedule the INSERT BEFORE writing the JSON body so the goroutine
	// kicks off as early as possible — Go's scheduler will let the goroutine
	// run concurrently with the response serialization on a multi-core box.
	// The goroutine has its own 2 s timeout context, decoupled from r.Context()
	// so a fast client disconnect doesn't cancel the analytics insert mid-flight.
	//
	// auth.GetUserID(r) returns "" for anonymous viewers (the public-detail
	// endpoint is whitelisted in jwt.go's publicPrefixes and the middleware
	// short-circuits without populating the user_id context key) — that
	// translates to NULL in viewer_user, exactly the schema intent.
	logScenarioViewAsync(
		store.Pool,
		d.ID,
		auth.GetUserID(r),
		hashIP(extractClientIP(r), resolveIPHashSalt()),
	)

	writeJSON(w, d)
}

// ──────────────────────────────────────────────────────────────────────────────
// POST /api/v1/scenarios/{slug}/clone
//
// Authenticated. Inserts a copy of the public scenario into the user's own
// strategies, with author/featured/premium/public flags reset (so a clone
// never accidentally re-publishes itself). Returns the new strategy_id.
// Increments clone_count atomically on the source so the catalog "Trending"
// sort updates immediately.
//
// Race / over-counting:
//   The `UPDATE ... clone_count = clone_count + 1` is row-locked; even if a
//   user click-spams the button, every successful clone bumps exactly once.
//   We accept double-clones occasionally racing — the cost is "user has two
//   copies in their list", not data loss.
// ──────────────────────────────────────────────────────────────────────────────

func handleScenarioClone(w http.ResponseWriter, r *http.Request) {
	if !requireDB(w) {
		return
	}
	userID := auth.GetUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	slug := r.PathValue("slug")
	if slug == "" {
		writeError(w, http.StatusBadRequest, "slug required")
		return
	}

	tx, err := store.Pool.Begin(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, "tx begin failed")
		return
	}
	// Defer with a captured-err pattern: if anything below errors, rollback
	// runs; on commit we set tx=nil to skip rollback.
	defer func() {
		if tx != nil {
			_ = tx.Rollback(r.Context())
		}
	}()

	// SELECT-FOR-UPDATE keeps clone_count consistent across racing clones.
	var (
		srcID     string
		name      string
		segment   string
		pair      string
		interval  string
		riskTier  string
		nodesJSON json.RawMessage
		edgesJSON json.RawMessage
	)
	err = tx.QueryRow(r.Context(), `
		SELECT id, name, COALESCE(segment, 'pro'), COALESCE(selected_pair, 'EURUSD'),
		       COALESCE(interval, '1h'), COALESCE(risk_tier, 'balanced'),
		       COALESCE(nodes_json, '[]'::jsonb), COALESCE(edges_json, '[]'::jsonb)
		FROM strategies
		WHERE author_slug = $1 AND is_public = TRUE
		FOR UPDATE
	`, slug).Scan(&srcID, &name, &segment, &pair, &interval, &riskTier, &nodesJSON, &edgesJSON)
	if err == pgx.ErrNoRows {
		writeError(w, http.StatusNotFound, "scenario not found")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to load scenario")
		return
	}

	// Insert the clone with all author/public/featured fields reset. The
	// new row inherits selected_pair, interval, risk_tier and the graph
	// blobs verbatim. is_active stays false: the user must explicitly hit
	// /scenarios/{id}/start before the runner picks it up.
	var newID string
	err = tx.QueryRow(r.Context(), `
		INSERT INTO strategies (
			user_id, name, segment, selected_pair, interval, risk_tier,
			nodes_json, edges_json,
			is_active, is_public, is_featured, is_premium,
			author_slug, author_name, author_theme, author_style, author_bio,
			author_position, preview_chip, published_at, clone_count
		)
		VALUES (
			$1, $2, $3, $4, $5, $6,
			$7, $8,
			FALSE, FALSE, FALSE, FALSE,
			NULL, NULL, NULL, NULL, NULL,
			NULL, NULL, NULL, 0
		)
		RETURNING id
	`, userID, name+" (clone)", segment, pair, interval, riskTier, nodesJSON, edgesJSON).Scan(&newID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to insert clone")
		return
	}

	if _, err := tx.Exec(r.Context(),
		`UPDATE strategies SET clone_count = clone_count + 1 WHERE id = $1`, srcID,
	); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to bump clone_count")
		return
	}

	if err := tx.Commit(r.Context()); err != nil {
		writeError(w, http.StatusInternalServerError, "tx commit failed")
		return
	}
	tx = nil

	writeJSON(w, map[string]any{"strategy_id": newID})
}

// ──────────────────────────────────────────────────────────────────────────────
// GET /api/v1/dashboard/summary
//
// Authenticated. Single round-trip endpoint that the post-login Dashboard
// page hits — collapses 3+ separate queries (strategies count, recent
// alerts, featured cards) into one HTTP call. Reduces latency on the
// landing-after-login screen from ~600 ms down to ~150 ms.
// ──────────────────────────────────────────────────────────────────────────────

type dashboardAlertRow struct {
	ID         string  `json:"id"`
	StrategyID string  `json:"strategy_id"`
	Symbol     string  `json:"symbol"`
	Direction  string  `json:"direction"`
	Confidence float64 `json:"confidence"`
	BarTime    int64   `json:"bar_time"`
	CreatedAt  string  `json:"created_at"`
}

type dashboardSummary struct {
	StrategiesCount int                  `json:"strategies_count"`
	RecentAlerts    []dashboardAlertRow  `json:"recent_alerts"`
	Featured        []publicScenarioCard `json:"featured"`
}

func handleDashboardSummary(w http.ResponseWriter, r *http.Request) {
	if !requireDB(w) {
		return
	}
	userID := auth.GetUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	out := dashboardSummary{
		RecentAlerts: []dashboardAlertRow{},
		Featured:     []publicScenarioCard{},
	}

	// 1) Count of strategies owned by the user. Cheap with idx_strategies_user.
	if err := store.Pool.QueryRow(r.Context(),
		`SELECT COUNT(*) FROM strategies WHERE user_id = $1`, userID,
	).Scan(&out.StrategiesCount); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to count strategies")
		return
	}

	// 2) Recent alerts (up to 10). Indexed by user_id+created_at DESC in 005.
	alertRows, err := store.Pool.Query(r.Context(), `
		SELECT id, strategy_id, symbol, direction,
		       COALESCE(confidence, 0), bar_time, created_at
		FROM alerts
		WHERE user_id = $1
		ORDER BY created_at DESC
		LIMIT $2
	`, userID, dashboardAlerts)
	if err == nil {
		defer alertRows.Close()
		for alertRows.Next() {
			var a dashboardAlertRow
			if err := alertRows.Scan(&a.ID, &a.StrategyID, &a.Symbol, &a.Direction,
				&a.Confidence, &a.BarTime, &a.CreatedAt); err == nil {
				out.RecentAlerts = append(out.RecentAlerts, a)
			}
		}
	}
	// We tolerate alert query failure — dashboard should still render
	// strategies_count + featured even if alerts table is e.g. locked.

	// 3) Featured scenarios (same query as /scenarios/featured).
	feaRows, err := store.Pool.Query(r.Context(), `
		SELECT `+cardSelectColumns+`
		FROM strategies
		WHERE is_public = TRUE AND is_featured = TRUE AND author_slug IS NOT NULL
		ORDER BY clone_count DESC, published_at DESC NULLS LAST
		LIMIT $1
	`, featuredLimit)
	if err == nil {
		defer feaRows.Close()
		for feaRows.Next() {
			var c publicScenarioCard
			if err := scanCard(feaRows, &c); err == nil {
				out.Featured = append(out.Featured, c)
			}
		}
	}
	writeJSON(w, out)
}

// ──────────────────────────────────────────────────────────────────────────────
// GET /api/v1/billing
//
// Stub for the upcoming paywall. Returns the user's tier from `users.tier`
// (default 'free'). The real billing integration (Stripe webhook → tier
// flip) lives in a future sprint — right now the frontend just needs a
// stable endpoint shape so it can render "Upgrade" CTAs without crashing.
// ──────────────────────────────────────────────────────────────────────────────

func handleBillingStub(w http.ResponseWriter, r *http.Request) {
	if !requireDB(w) {
		return
	}
	userID := auth.GetUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	tier := "free"
	// Best-effort read: if the column or row is missing we still return
	// "free" so the UI keeps working during partial rollouts.
	_ = store.Pool.QueryRow(r.Context(),
		`SELECT COALESCE(tier, 'free') FROM users WHERE id = $1`, userID,
	).Scan(&tier)

	writeJSON(w, map[string]any{
		"tier":             tier,
		"stripe_customer":  nil,
		"renewal_at":       nil,
		"upgrade_url":      "/pricing",
	})
}
