package api

// scenarios_public_handlers_test.go — offline coverage for the Sprint 2
// public-catalog endpoints.
//
// What we test offline (no Postgres needed):
//   1. parseListParams — pagination clamping, theme/style enum filtering,
//      query-string trimming, q-param LIKE-pattern escaping.
//   2. escapeLikePattern — backslash, %, and _ are neutralised.
//   3. JWT public-path whitelist — featured/public/public/{slug} bypass auth;
//      clone/dashboard/billing demand a token; the bare "/api/v1/scenarios/public/"
//      (empty slug) does NOT bypass.
//   4. DB-503 fall-through — every handler short-circuits to 503 when
//      store.Pool is nil, never panics.
//   5. handleBillingStub response shape (DB-503 envelope is the deepest we
//      can go without PG; the success-path tier read needs a real row).
//
// What we DON'T test here (TODO(integration)):
//   - 200/404 for handleScenariosPublicDetail with real rows.
//   - Featured ORDER BY behaviour.
//   - Clone-count increment + transaction commit.
//   - Filter SQL actually constraining the WHERE clause beyond what
//     parseListParams already verifies in isolation.
//   These need a Postgres test container; tracked separately so this file
//   stays runnable in `go test ./...` without docker.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/prostwp/elibri-backend/internal/auth"
	"github.com/prostwp/elibri-backend/internal/config"
)

const testJWTSecret = "test-secret-please-do-not-deploy-32-bytes-min"

// newTestRouter spins up a fresh router with the test JWT secret. CORS is
// permissive (set to *) so origin checks don't interfere with httptest.
func newTestRouter(t *testing.T) http.Handler {
	t.Helper()
	cfg := &config.Config{
		JWTSecret:   testJWTSecret,
		CORSOrigins: "*",
	}
	return NewRouter(cfg)
}

// validUserToken issues an access token for a stub user — enough to pass
// auth.Middleware, since we only assert "auth gate cleared, then 503"
// downstream.
func validUserToken(t *testing.T) string {
	t.Helper()
	tok, err := auth.IssueToken(testJWTSecret,
		"00000000-0000-0000-0000-000000000001",
		"tester@elibri.local",
		"user",
	)
	if err != nil {
		t.Fatalf("issue token: %v", err)
	}
	return tok
}

// ─── parseListParams ─────────────────────────────────────────────────────────

func TestParseListParams_Defaults(t *testing.T) {
	got := parseListParams(map[string][]string{})
	if got.Page != 1 {
		t.Errorf("Page = %d, want 1", got.Page)
	}
	if got.Theme != "" || got.Style != "" || got.Q != "" {
		t.Errorf("filters not zero by default: %+v", got)
	}
}

func TestParseListParams_Pagination(t *testing.T) {
	cases := []struct {
		raw     string
		want    int
		comment string
	}{
		{"1", 1, "min legal"},
		{"2", 2, "page 2 → offset 20"},
		{"50", 50, "max legal"},
		{"0", 1, "zero clamps back to 1"},
		{"-5", 1, "negative ignored"},
		{"51", 1, "over-cap ignored"},
		{"abc", 1, "non-numeric ignored"},
		{"", 1, "empty falls back"},
	}
	for _, tc := range cases {
		t.Run(tc.raw+"_"+tc.comment, func(t *testing.T) {
			got := parseListParams(map[string][]string{"page": {tc.raw}})
			if got.Page != tc.want {
				t.Errorf("page=%q → Page=%d, want %d", tc.raw, got.Page, tc.want)
			}
		})
	}
}

func TestParseListParams_OffsetMath(t *testing.T) {
	// page=2 should produce offset 20 (page-1 * pageSize).
	// We don't compute offset inside parseListParams (handler does it), but
	// we can assert the page value and the constant we'll multiply by.
	got := parseListParams(map[string][]string{"page": {"2"}})
	if got.Page != 2 {
		t.Fatalf("Page = %d, want 2", got.Page)
	}
	offset := (got.Page - 1) * listPageSize
	if offset != 20 {
		t.Errorf("computed offset = %d, want 20", offset)
	}
}

func TestParseListParams_ThemeWhitelist(t *testing.T) {
	for theme := range validThemes {
		t.Run(theme, func(t *testing.T) {
			got := parseListParams(map[string][]string{"theme": {theme}})
			if got.Theme != theme {
				t.Errorf("Theme = %q, want %q", got.Theme, theme)
			}
		})
	}
	bad := []string{"junk", "stocks", "GOLD_SILVER ", " "}
	for _, b := range bad {
		t.Run("bad/"+b, func(t *testing.T) {
			got := parseListParams(map[string][]string{"theme": {b}})
			// Special case: "GOLD_SILVER " gets trimmed + lowercased then
			// passes the whitelist — that's INTENDED tolerance. The remaining
			// bad inputs ("junk", "stocks", " ") must drop to "".
			lower := strings.ToLower(strings.TrimSpace(b))
			if _, ok := validThemes[lower]; ok {
				if got.Theme != lower {
					t.Errorf("Theme = %q, want %q (after normalisation)", got.Theme, lower)
				}
			} else if got.Theme != "" {
				t.Errorf("Theme = %q, want empty (input %q invalid)", got.Theme, b)
			}
		})
	}
}

func TestParseListParams_StyleWhitelist(t *testing.T) {
	for style := range validStyles {
		t.Run(style, func(t *testing.T) {
			got := parseListParams(map[string][]string{"style": {style}})
			if got.Style != style {
				t.Errorf("Style = %q, want %q", got.Style, style)
			}
		})
	}
	got := parseListParams(map[string][]string{"style": {"meme"}})
	if got.Style != "" {
		t.Errorf("invalid style accepted: %q", got.Style)
	}
}

func TestParseListParams_QueryEscape(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"hello", "hello"},
		{"50%", `50\%`},
		{"foo_bar", `foo\_bar`},
		{`back\slash`, `back\\slash`},
		{`mix_of%special\things`, `mix\_of\%special\\things`},
		{"   trim_me   ", `trim\_me`},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			got := parseListParams(map[string][]string{"q": {tc.in}})
			if got.Q != tc.want {
				t.Errorf("Q = %q, want %q", got.Q, tc.want)
			}
		})
	}
}

func TestParseListParams_QueryLengthCap(t *testing.T) {
	long := strings.Repeat("a", maxQueryLen+50)
	got := parseListParams(map[string][]string{"q": {long}})
	if len(got.Q) > maxQueryLen {
		t.Errorf("Q len = %d, expected <= %d", len(got.Q), maxQueryLen)
	}
}

// ─── escapeLikePattern (low-level) ───────────────────────────────────────────

func TestEscapeLikePattern(t *testing.T) {
	if escapeLikePattern("") != "" {
		t.Error("empty in → empty out")
	}
	if escapeLikePattern("plain text") != "plain text" {
		t.Error("plain text should pass through unchanged")
	}
	// Order check: backslash must be escaped FIRST, otherwise we'd
	// double-escape our own escape characters.
	if got := escapeLikePattern(`\%`); got != `\\\%` {
		t.Errorf("escapeLikePattern(`\\%%`) = %q, want %q", got, `\\\%`)
	}
}

// ─── JWT public-path matrix ──────────────────────────────────────────────────
//
// These tests exercise the *whole* router — auth.Middleware sits in front of
// the mux. We don't have Postgres, so the success path inside each public
// handler will return 503; the *important* thing is that 503 means the
// auth gate cleared. If auth blocked the request we'd see 401 instead.

func TestPublicPaths_FeaturedNoAuth(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/scenarios/featured", nil)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code == http.StatusUnauthorized {
		t.Errorf("featured returned 401 — should be public")
	}
}

func TestPublicPaths_PublicListNoAuth(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/scenarios/public", nil)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code == http.StatusUnauthorized {
		t.Errorf("public list returned 401 — should be public")
	}
}

func TestPublicPaths_PublicDetailNoAuth(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/scenarios/public/news_pivot", nil)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code == http.StatusUnauthorized {
		t.Errorf("public detail with slug returned 401 — should be public")
	}
}

func TestPublicPaths_PublicDetailEmptySlugRequiresAuth(t *testing.T) {
	// Path "/api/v1/scenarios/public/" has nothing AFTER the prefix — by
	// design it must NOT bypass auth (avoids confusion-by-empty-slug bugs).
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/scenarios/public/", nil)
	newTestRouter(t).ServeHTTP(rec, req)
	// Without a JWT we expect 401 from the middleware. (The mux might also
	// route this to the public-list handler — even then the prefix-bypass
	// rule says: don't grant public access. So 401 is the contract.)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401 (empty slug must NOT bypass auth)", rec.Code)
	}
}

func TestProtectedPaths_CloneRequiresAuth(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/scenarios/news_pivot/clone", nil)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("clone without JWT: status = %d, want 401", rec.Code)
	}
}

func TestProtectedPaths_DashboardRequiresAuth(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/dashboard/summary", nil)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("dashboard without JWT: status = %d, want 401", rec.Code)
	}
}

func TestProtectedPaths_BillingRequiresAuth(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/billing", nil)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("billing without JWT: status = %d, want 401", rec.Code)
	}
}

// ─── DB-503 fall-through ─────────────────────────────────────────────────────
//
// All handlers call requireDB(w) first. With store.Pool nil (which is the
// case in this test process — InitPostgres is never called), they must
// short-circuit to 503 instead of nil-derefing the pool.

func TestHandlers_DB503_AuthedClone(t *testing.T) {
	tok := validUserToken(t)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/scenarios/news_pivot/clone", nil)
	req.Header.Set("Authorization", "Bearer "+tok)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("clone w/ auth, no DB: status = %d, want 503", rec.Code)
	}
}

func TestHandlers_DB503_AuthedDashboard(t *testing.T) {
	tok := validUserToken(t)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/dashboard/summary", nil)
	req.Header.Set("Authorization", "Bearer "+tok)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("dashboard w/ auth, no DB: status = %d, want 503", rec.Code)
	}
}

func TestHandlers_DB503_AuthedBilling(t *testing.T) {
	tok := validUserToken(t)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/billing", nil)
	req.Header.Set("Authorization", "Bearer "+tok)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("billing w/ auth, no DB: status = %d, want 503", rec.Code)
	}
	// Body should be a JSON error envelope, not a panic stack.
	var env map[string]string
	if err := json.NewDecoder(rec.Body).Decode(&env); err != nil {
		t.Errorf("billing 503 body not JSON: %v", err)
	}
	if env["error"] == "" {
		t.Errorf("billing 503 missing error key: %+v", env)
	}
}

func TestHandlers_DB503_PublicFeatured(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/scenarios/featured", nil)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("featured no DB: status = %d, want 503", rec.Code)
	}
}

func TestHandlers_DB503_PublicList(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/scenarios/public?page=2&theme=crypto", nil)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("public list no DB: status = %d, want 503", rec.Code)
	}
}

func TestHandlers_DB503_PublicDetail(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/scenarios/public/some_slug", nil)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("public detail no DB: status = %d, want 503", rec.Code)
	}
}

// ─── Filter parameter parity (pure function) ─────────────────────────────────
//
// Smoke check that q + theme + style + page combine without losing values.
// Catches regressions where one filter accidentally clobbers another.

func TestParseListParams_AllFiltersPresent(t *testing.T) {
	q := url.Values{}
	q.Set("page", "3")
	q.Set("theme", "crypto")
	q.Set("style", "news")
	q.Set("q", "btc dominance")
	got := parseListParams(q)
	if got.Page != 3 {
		t.Errorf("Page = %d, want 3", got.Page)
	}
	if got.Theme != "crypto" {
		t.Errorf("Theme = %q, want crypto", got.Theme)
	}
	if got.Style != "news" {
		t.Errorf("Style = %q, want news", got.Style)
	}
	if got.Q != "btc dominance" {
		t.Errorf("Q = %q, want btc dominance", got.Q)
	}
	// Offset for page=3 → 40.
	if off := (got.Page - 1) * listPageSize; off != 40 {
		t.Errorf("offset for page=3 = %d, want 40", off)
	}
}

// TODO(integration): once PG test infra lands, add tests for:
//   - handleScenariosPublicDetail returns 404 for missing slug
//   - handleScenariosPublicDetail returns 404 for is_public=FALSE row
//   - handleScenarioClone copies graph, resets author/featured/public/premium
//   - handleScenarioClone bumps clone_count by exactly 1 (in TX)
//   - handleScenariosPublicList ORDER BY hits the partial index
//   - handleScenariosFeatured limits to featuredLimit rows
