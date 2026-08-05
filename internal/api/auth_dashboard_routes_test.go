package api

// auth_dashboard_routes_test.go — offline routing/auth-gate coverage for the
// sub-chat C endpoints (verify/reset, follows, watchlist, notification-prefs).
//
// Like scenarios_public_handlers_test.go, these run the WHOLE router with no
// Postgres. Two things are checked without a DB:
//   1. Auth gating: public endpoints (verify-email/forgot-password/
//      reset-password) must NOT 401 without a token; protected endpoints
//      (resend-verification, follows, watchlist, notification-prefs) MUST 401.
//   2. DB-503 fall-through: every handler calls requireDB / nil-pool-checks
//      first, so with store.Pool nil they return 503 (never panic) once auth
//      has cleared.
//
// What is NOT covered here (needs a Postgres test container, tracked for the
// integration tier): the 400 validation branches (accepted_terms=false, bad
// frequency, password<6, invalid token) — they sit AFTER the DB gate in the
// handler, so a nil pool short-circuits to 503 before they can run. The pure
// validation logic those branches rely on is unit-tested directly in
// prefs_handlers_test.go (watchlist/frequency/merge) and the email/token
// packages.

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ─── Public auth endpoints must bypass the JWT gate ─────────────────────────

func TestPublicPaths_VerifyEmailNoAuth(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/auth/verify-email",
		strings.NewReader(`{"token":"x"}`))
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code == http.StatusUnauthorized {
		t.Errorf("verify-email returned 401 — should be public")
	}
}

func TestPublicPaths_ForgotPasswordNoAuth(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/auth/forgot-password",
		strings.NewReader(`{"email":"a@b.com"}`))
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code == http.StatusUnauthorized {
		t.Errorf("forgot-password returned 401 — should be public")
	}
}

func TestPublicPaths_ResetPasswordNoAuth(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/auth/reset-password",
		strings.NewReader(`{"token":"x","new_password":"secret1"}`))
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code == http.StatusUnauthorized {
		t.Errorf("reset-password returned 401 — should be public")
	}
}

// ─── Protected endpoints must require a token ───────────────────────────────

func TestProtectedPaths_ResendVerificationRequiresAuth(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/auth/resend-verification", nil)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("resend-verification without JWT: status = %d, want 401", rec.Code)
	}
}

func TestProtectedPaths_FollowsRequireAuth(t *testing.T) {
	cases := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/api/v1/follows"},
		{http.MethodPost, "/api/v1/follows"},
		{http.MethodDelete, "/api/v1/follows/00000000-0000-0000-0000-000000000001"},
	}
	for _, tc := range cases {
		t.Run(tc.method+"_"+tc.path, func(t *testing.T) {
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(tc.method, tc.path, nil)
			newTestRouter(t).ServeHTTP(rec, req)
			if rec.Code != http.StatusUnauthorized {
				t.Errorf("%s %s without JWT: status = %d, want 401", tc.method, tc.path, rec.Code)
			}
		})
	}
}

func TestProtectedPaths_WatchlistRequiresAuth(t *testing.T) {
	for _, m := range []string{http.MethodGet, http.MethodPut} {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(m, "/api/v1/watchlist", nil)
		newTestRouter(t).ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("%s /watchlist without JWT: status = %d, want 401", m, rec.Code)
		}
	}
}

func TestProtectedPaths_NotificationPrefsRequiresAuth(t *testing.T) {
	for _, m := range []string{http.MethodGet, http.MethodPut} {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(m, "/api/v1/notification-prefs", nil)
		newTestRouter(t).ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("%s /notification-prefs without JWT: status = %d, want 401", m, rec.Code)
		}
	}
}

// ─── DB-503 fall-through (auth cleared, no Postgres) ────────────────────────

func TestHandlers_DB503_VerifyEmail(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/auth/verify-email",
		strings.NewReader(`{"token":"x"}`))
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("verify-email no DB: status = %d, want 503", rec.Code)
	}
}

func TestHandlers_DB503_ForgotPassword(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/auth/forgot-password",
		strings.NewReader(`{"email":"a@b.com"}`))
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("forgot-password no DB: status = %d, want 503", rec.Code)
	}
}

func TestHandlers_DB503_AuthedFollowsList(t *testing.T) {
	tok := validUserToken(t)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/follows", nil)
	req.Header.Set("Authorization", "Bearer "+tok)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("follows list w/ auth, no DB: status = %d, want 503", rec.Code)
	}
}

func TestHandlers_DB503_AuthedWatchlistGet(t *testing.T) {
	tok := validUserToken(t)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/watchlist", nil)
	req.Header.Set("Authorization", "Bearer "+tok)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("watchlist get w/ auth, no DB: status = %d, want 503", rec.Code)
	}
}

func TestHandlers_DB503_AuthedNotificationPrefsGet(t *testing.T) {
	tok := validUserToken(t)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/notification-prefs", nil)
	req.Header.Set("Authorization", "Bearer "+tok)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("notification-prefs get w/ auth, no DB: status = %d, want 503", rec.Code)
	}
}
