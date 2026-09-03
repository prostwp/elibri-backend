package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// stubDemobot points the proxy at a scripted upstream and records what it asked for.
func stubDemobot(t *testing.T, h http.HandlerFunc) *string {
	t.Helper()
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.URL.String()
		h(w, r)
	}))
	t.Cleanup(srv.Close)
	t.Setenv("DEMOBOT_URL", srv.URL)
	return &got
}

func proxyGet(t *testing.T, path string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	rec := httptest.NewRecorder()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/v1/demobot/{path...}", handleDemobotProxy)
	mux.ServeHTTP(rec, req)
	return rec
}

func TestDemobotProxyForwardsAllowedPaths(t *testing.T) {
	cases := map[string]string{
		"showcase":         "/showcase",
		"agents":           "/agents",
		"agents/gold":      "/agents/gold",
		"agents/trend":     "/agents/trend",
		"showcase/example": "/showcase/example",
	}
	for suffix, wantUpstream := range cases {
		t.Run(suffix, func(t *testing.T) {
			got := stubDemobot(t, func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write([]byte(`{"ok":true}`))
			})
			rec := proxyGet(t, "/api/v1/demobot/"+suffix)
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200", rec.Code)
			}
			if *got != wantUpstream {
				t.Errorf("upstream path = %q, want %q", *got, wantUpstream)
			}
		})
	}
}

// An unknown suffix must never be forwarded: the whole point of the allowlist
// is that this proxy cannot be walked into another service.
//
// Two properties are asserted, and only these two. Traversal inputs never
// reach the handler at all — Go's ServeMux normalises "…/../…" and answers
// 307 first — so pinning an exact status here would be testing the standard
// library's routing, not this proxy. What matters is that nothing goes
// upstream and that no redirect leaves the origin.
func TestDemobotProxyRejectsUnknownPaths(t *testing.T) {
	for _, bad := range []string{
		"agents/../../admin", "internal", "",
		"http://evil.example.com", "agents/gold/../../../etc/passwd",
	} {
		t.Run(bad, func(t *testing.T) {
			reached := false
			stubDemobot(t, func(w http.ResponseWriter, r *http.Request) { reached = true })
			rec := proxyGet(t, "/api/v1/demobot/"+bad)

			if reached {
				t.Errorf("proxy forwarded %q upstream — the allowlist did not hold", bad)
			}
			if rec.Code == http.StatusOK {
				t.Errorf("unknown path %q answered 200", bad)
			}
			// No open redirect: a Location, if any, stays on this origin.
			if loc := rec.Header().Get("Location"); loc != "" && !strings.HasPrefix(loc, "/") {
				t.Errorf("redirect leaves the origin: %q -> %q", bad, loc)
			}
		})
	}
}

// An unknown agent NAME is the demobot's call, not ours: it is forwarded and
// its own 404 comes back. Pinning a list here would be a second definition of
// which agents exist, and it would silently drop any agent added later.
func TestDemobotProxyForwardsUnknownAgentNameAndReturnsUpstream404(t *testing.T) {
	var seen string
	stubDemobot(t, func(w http.ResponseWriter, r *http.Request) {
		seen = r.URL.Path
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"unknown agent"}`))
	})
	rec := proxyGet(t, "/api/v1/demobot/agents/brandnewagent")
	if seen != "/agents/brandnewagent" {
		t.Errorf("upstream path = %q, want the name forwarded verbatim", seen)
	}
	if rec.Code != http.StatusNotFound {
		t.Errorf("status = %d, want the upstream 404", rec.Code)
	}
}

// The segment must still be unable to carry anything that escapes the path.
func TestDemobotProxyRejectsMalformedAgentNames(t *testing.T) {
	for _, bad := range []string{
		// %20, not a raw space: a raw one cannot reach a server at all, and
		// httptest.NewRequest refuses to build such a URL.
		"agents/UPPER", "agents/has%20space", "agents/a:b", "agents/a@b",
		"agents/a/b", "agents/", "agents/" + strings.Repeat("x", 40),
	} {
		t.Run(bad, func(t *testing.T) {
			reached := false
			stubDemobot(t, func(w http.ResponseWriter, r *http.Request) { reached = true })
			rec := proxyGet(t, "/api/v1/demobot/"+bad)
			if reached {
				t.Errorf("forwarded malformed name %q upstream", bad)
			}
			if rec.Code == http.StatusOK {
				t.Errorf("malformed name %q answered 200", bad)
			}
		})
	}
}

// Only asset/assets/tf travel; anything else a caller appends is dropped.
func TestDemobotProxyForwardsOnlyKnownQuery(t *testing.T) {
	got := stubDemobot(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{}`))
	})
	proxyGet(t, "/api/v1/demobot/agents/momentum?asset=gold&tf=1d&token=secret&redirect=http://evil")
	if strings.Contains(*got, "token") || strings.Contains(*got, "redirect") {
		t.Errorf("unexpected query forwarded: %q", *got)
	}
	for _, want := range []string{"asset=gold", "tf=1d"} {
		if !strings.Contains(*got, want) {
			t.Errorf("query %q missing from %q", want, *got)
		}
	}
}

// An honest degraded card (200 with ok=false) must arrive unchanged: the UI
// branches on it, so the proxy must not "helpfully" turn it into an error.
func TestDemobotProxyPassesDegradedCardThrough(t *testing.T) {
	stubDemobot(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"ok":false,"reason":"source_offline","verdict":"offline"}`))
	})
	rec := proxyGet(t, "/api/v1/demobot/agents/gold")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want the upstream 200", rec.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("body not JSON: %v", err)
	}
	if body["ok"] != false || body["reason"] != "source_offline" {
		t.Errorf("degraded fields altered in transit: %v", body)
	}
}

// Upstream failure states must be distinguishable by the UI.
func TestDemobotProxyReportsUpstreamDown(t *testing.T) {
	t.Setenv("DEMOBOT_URL", "http://127.0.0.1:1") // nothing listens here
	rec := proxyGet(t, "/api/v1/demobot/showcase")
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503 when the demobot is down", rec.Code)
	}
}

// A 404 from the demobot must stay a 404, not become our 200.
func TestDemobotProxyPreservesUpstreamStatus(t *testing.T) {
	stubDemobot(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"nope"}`))
	})
	rec := proxyGet(t, "/api/v1/demobot/agents/fx")
	if rec.Code != http.StatusNotFound {
		t.Errorf("status = %d, want the upstream 404", rec.Code)
	}
}

// Codex review #11/#12: a proxy may refuse a request; it may not quietly
// answer a different one. Get+Set collapsed repeated values and dropped
// present-but-empty ones, so "assets=" reached upstream as no parameter at
// all and came back with the DEFAULT scan — real data for a query nobody made.
func TestDemobotProxyPreservesQueryVerbatim(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"repeated values kept", "asset=btc&asset=eth", "asset=btc&asset=eth"},
		{"present but empty kept", "assets=", "assets="},
		{"risk params forwarded", "balance=10000&risk=1&entry=64000&stop=62500",
			"balance=10000&entry=64000&risk=1&stop=62500"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := stubDemobot(t, func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write([]byte(`{}`))
			})
			proxyGet(t, "/api/v1/demobot/agents/momentum?"+tc.in)
			if !strings.Contains(*got, tc.want) {
				t.Errorf("upstream query = %q, want it to contain %q", *got, tc.want)
			}
		})
	}
}

// Codex review #21: an oversized body must become an error, never a truncated
// document forwarded with the upstream's 200 — the client would parse garbage
// as success.
func TestDemobotProxyRejectsOversizedBodyInsteadOfTruncating(t *testing.T) {
	stubDemobot(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(make([]byte, demobotMaxBytes+2048))
	})
	rec := proxyGet(t, "/api/v1/demobot/showcase")
	if rec.Code == http.StatusOK {
		t.Errorf("oversized body answered 200 with %d bytes — truncated JSON passed as success", rec.Body.Len())
	}
	if rec.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", rec.Code)
	}
}
