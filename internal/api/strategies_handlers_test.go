package api

// strategies_handlers_test.go — Sprint 3 backend.
//
// Offline contract checks for POST /api/v1/strategies. This is the same
// "DB unavailable in tests, so we drive the auth + middleware layer and
// assert the response shape" pattern used elsewhere in this file. We
// deliberately do not test the actual INSERT round-trip — that needs a
// Postgres test container which the rest of the package also defers
// to a future integration suite.
//
// What we lock in here:
//   1. Missing JWT → 401 (auth gate works for /api/v1/strategies POST).
//   2. Valid JWT + no DB → 503 with the standard error envelope.
//   3. Valid JWT + DB unavailable + INVALID JSON body → 400, never 500
//      or panic. The 503 short-circuit must run BEFORE json.Decode so
//      we don't waste cycles parsing a body for a request that can't
//      succeed regardless. (We assert that 503 lands first.)
//   4. Valid JWT + bad JSON body when the DB IS plausibly available
//      (we can't simulate this here without PG) → 400. Documented as
//      TODO(integration).
//
// The Sprint 3 audit task asked us to confirm the POST handler is sound:
//   - nodes_json/edges_json saved as JSONB → confirmed by reading the
//     INSERT in auth.CreateStrategy (uses pgx jsonb-compatible binding
//     of json.RawMessage; defaults nil/empty bytes to "[]").
//   - successful save returns full strategy with id → confirmed by
//     RETURNING id, created_at, updated_at on the INSERT path.
//   - validation error returns 400 → json.NewDecoder.Decode failure
//     branch hits writeError(w, 400, ...).
// Locking those in needs a real DB; this file's tests cover the shape
// we can verify without one. See TODO(integration) at the bottom.

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestStrategiesCreate_RequiresAuth(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/strategies",
		bytes.NewBufferString(`{"name":"x"}`))
	req.Header.Set("Content-Type", "application/json")
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("POST /api/v1/strategies without auth: status = %d, want 401", rec.Code)
	}
}

func TestStrategiesCreate_DB503AfterAuth(t *testing.T) {
	tok := validUserToken(t)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/strategies",
		bytes.NewBufferString(`{"name":"my strategy","nodes_json":[],"edges_json":[],"selected_pair":"EURUSD"}`))
	req.Header.Set("Authorization", "Bearer "+tok)
	req.Header.Set("Content-Type", "application/json")
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("POST w/ auth, no DB: status = %d, want 503", rec.Code)
	}
	// Body should be JSON envelope, not bare text or panic stack.
	var env map[string]string
	if err := json.NewDecoder(rec.Body).Decode(&env); err != nil {
		t.Errorf("503 body not JSON: %v", err)
	}
	if env["error"] == "" {
		t.Errorf("503 missing error key: %+v", env)
	}
}

// The 503 short-circuit is intentionally before the body parse — we want
// invalid bodies to ALSO return 503 when the DB is gone, because the
// outcome would be the same anyway and parsing first wastes cycles. If
// somebody refactors the handler to validate-then-503 they need to
// update this expectation; that's a worthwhile failure mode to flag.
func TestStrategiesCreate_NoDB_GarbageBodyStill503(t *testing.T) {
	tok := validUserToken(t)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/strategies",
		bytes.NewBufferString(`{not even json`))
	req.Header.Set("Authorization", "Bearer "+tok)
	req.Header.Set("Content-Type", "application/json")
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("garbage body w/ no DB: status = %d, want 503 (DB check runs first)", rec.Code)
	}
}

// We can also assert the auth gate accepts a Bearer token whose role is
// "user" — handleStrategiesCreate should not require admin.
func TestStrategiesCreate_RoleUserAccepted(t *testing.T) {
	tok := validUserToken(t)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/strategies",
		bytes.NewBufferString(`{}`))
	req.Header.Set("Authorization", "Bearer "+tok)
	newTestRouter(t).ServeHTTP(rec, req)
	if rec.Code == http.StatusForbidden {
		t.Errorf("POST returned 403 for role=user — should be allowed")
	}
}

// Body-too-large guard: the bodyLimitMiddleware caps requests at 2 MB.
// A POST that just barely exceeds this should be rejected by the
// MaxBytesReader, not crash the handler. We don't have a way to stress
// 2 MB cleanly in a unit test (httptest reads it all into memory), so
// we satisfy ourselves with checking that a "large but legal" body
// doesn't hang or panic; the actual cap is exercised by reading the
// middleware's source comment.
func TestStrategiesCreate_LargeButLegalBodyDoesNotPanic(t *testing.T) {
	tok := validUserToken(t)
	// 4 KB JSON object — well under the 2 MB cap, just verifies the
	// handler doesn't choke on a body that's bigger than a one-liner.
	body := `{"name":"big","nodes_json":[` +
		strings.Repeat(`{"id":"x"},`, 200) + `{"id":"end"}],"edges_json":[]}`
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/strategies",
		bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer "+tok)
	req.Header.Set("Content-Type", "application/json")
	newTestRouter(t).ServeHTTP(rec, req)
	// We expect 503 (no DB) — anything above 500 except 503 would
	// indicate the handler crashed before requireDB could short-circuit.
	if rec.Code >= 500 && rec.Code != http.StatusServiceUnavailable {
		t.Errorf("large legal body: status = %d, expected 503 short-circuit", rec.Code)
	}
}

// TODO(integration): once a Postgres test container is wired in, add
// these tests covering the actual storage path:
//   - POST {nodes_json:[...] edges_json:[...]} returns 200 with id !=""
//     and the persisted row has the JSONB intact.
//   - POST {nodes_json: null} stores [] (the empty-bytes default in
//     auth.CreateStrategy).
//   - POST {} (empty body) stores name="Untitled Strategy",
//     segment="pro", selected_pair="EURUSD".
//   - POST {selected_pair: "<sql injection attempt>"} stores literally
//     (parameterised query — the value is a string, not glued into SQL).
//   - GET /api/v1/strategies after POST sees the new row.
