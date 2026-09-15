package demobot

// composite_modtime_test.go — validators of the COMPOSITE endpoints, never a
// component's data time (a component can change while it is neither the
// newest nor the oldest reading, and any component stamp would then answer a
// conditional GET with a false 304). data_as_of stays the oldest reading.
//   - /showcase: Last-Modified = the sweep time; 304 only within the
//     memoized sweep, whose body cannot change.
//   - /agents/digest (rebuilt per request, one-second header resolution),
//     /agents/top and /showcase/example (per-request AI parts reading the
//     whole sweep): no Last-Modified at all, If-Modified-Since ignored,
//     always 200.
// Codex reviews of ad77977, 02d6cb0 and 37c3f52.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// stepClock is a settable Agents.now.
type stepClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *stepClock) now() time.Time  { c.mu.Lock(); defer c.mu.Unlock(); return c.t }
func (c *stepClock) set(t time.Time) { c.mu.Lock(); c.t = t; c.mu.Unlock() }

// mutableMacroAgents serves /api/v1/macro from a swappable payload and a live
// whale snapshot (an OLDER reading than macro, so macro is never the oldest
// component); every other upstream is dead.
func mutableMacroAgents(t *testing.T) (*Agents, func(string)) {
	t.Helper()
	stubExternalBases(t)
	var mu sync.Mutex
	macroBody := macroLiveFixture
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/macro":
			mu.Lock()
			b := macroBody
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(b))
		case "/api/v1/whale-flow":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(whaleLiveFixture))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return NewAgents(NewBackendClient(srv.URL)), func(s string) { mu.Lock(); macroBody = s; mu.Unlock() }
}

// macroLiveChanged is the live payload with a newer DXY reading (score 83 →
// 95); the oldest lamp (VIX, Sep 14 07:00) is unchanged.
func macroLiveChanged(t *testing.T) string {
	t.Helper()
	s := strings.Replace(macroLiveFixture,
		`"value":99.61299896240234,"ok":true,"delta_pct":0.13369079310498477,"status":"neutral","as_of":"2026-09-15T04:00:00Z"`,
		`"value":99.2,"ok":true,"delta_pct":-0.28,"status":"tailwind","as_of":"2026-09-15T05:00:00Z"`, 1)
	s = strings.Replace(s, `"composite":83`, `"composite":95`, 1)
	if s == macroLiveFixture {
		t.Fatal("fixture replace did not apply")
	}
	return s
}

// rawGet is a GET with an optional If-Modified-Since: status, the
// Last-Modified header as sent ("" when absent) and the JSON body on 200.
func rawGet(t *testing.T, url, ims string) (int, string, []byte) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodGet, url, nil)
	if ims != "" {
		req.Header.Set("If-Modified-Since", ims)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var body []byte
	if resp.StatusCode == http.StatusOK {
		var raw json.RawMessage
		if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
			t.Fatal(err)
		}
		body = raw
	}
	return resp.StatusCode, resp.Header.Get("Last-Modified"), body
}

func condGet(t *testing.T, url, ims string) (int, time.Time, []byte) {
	t.Helper()
	status, lmRaw, body := rawGet(t, url, ims)
	lm, err := http.ParseTime(lmRaw)
	if err != nil {
		t.Fatalf("%s: Last-Modified %q: %v", url, lmRaw, err)
	}
	return status, lm.UTC(), body
}

func httpStamp(tm time.Time) string { return tm.UTC().Format(http.TimeFormat) }

func TestShowcaseLastModifiedIsSweepTime(t *testing.T) {
	ag, setMacro := mutableMacroAgents(t)
	t0 := time.Date(2026, 9, 15, 5, 0, 0, 0, time.UTC)
	clk := &stepClock{t: t0}
	ag.now = clk.now
	s, srv := newTestAPI(t, ag, true)

	for _, path := range []string{"/showcase"} {
		setMacro(macroLiveFixture)
		s.sc.mu.Lock()
		s.sc.cur = nil // a fresh sweep for this path
		s.sc.mu.Unlock()
		clk.set(t0)

		st1, lm1, body1 := condGet(t, srv.URL+path, "")
		if st1 != 200 || !lm1.Equal(t0) {
			t.Fatalf("%s: first GET %d, Last-Modified %s, want 200 at the sweep time %s", path, st1, lm1, t0)
		}
		// The same memoized sweep (inside its TTL; the clock moving does not
		// matter) still answers its own stamp with 304.
		clk.set(t0.Add(30 * time.Second))
		if st, _, _ := condGet(t, srv.URL+path, httpStamp(lm1)); st != http.StatusNotModified {
			t.Errorf("%s: identical memoized sweep with its own stamp: %d, want 304", path, st)
		}

		// A non-oldest component changes (macro; whale's reading is older),
		// and the next sweep runs.
		setMacro(macroLiveChanged(t))
		s.sc.mu.Lock()
		s.sc.cur = nil // the memo expired
		s.sc.mu.Unlock()
		clk.set(t0.Add(90 * time.Second))
		st2, lm2, body2 := condGet(t, srv.URL+path, httpStamp(lm1))
		if st2 != http.StatusOK {
			t.Errorf("%s: changed sweep with the first stamp: %d, want 200 (false 304)", path, st2)
		}
		if !lm2.After(lm1) {
			t.Errorf("%s: Last-Modified %s → %s must advance", path, lm1, lm2)
		}
		if path == "/showcase" && (string(body1) == string(body2) || !strings.Contains(string(body2), "rule score 95/100")) {
			t.Errorf("%s: the second sweep must carry the changed macro card", path)
		}
	}
}

// /agents/digest, /agents/top and /showcase/example carry no validator: no Last-Modified,
// and If-Modified-Since — even a date far in the future, which any stamp would
// satisfy — never turns a changed (or unchanged) body into a 304.
func TestDigestAndExampleSendNoValidator(t *testing.T) {
	ag, setMacro := mutableMacroAgents(t)
	t0 := time.Date(2026, 9, 15, 5, 0, 0, 0, time.UTC)
	clk := &stepClock{t: t0}
	ag.now = clk.now
	s, srv := newTestAPI(t, ag, true)

	dataAsOf := func(body []byte) string {
		var env struct {
			DataAsOf string `json:"data_as_of"`
		}
		if err := json.Unmarshal(body, &env); err != nil {
			t.Fatal(err)
		}
		return env.DataAsOf
	}
	future := httpStamp(t0.Add(24 * time.Hour))

	for _, path := range []string{"/agents/digest", "/agents/top", "/showcase/example"} {
		setMacro(macroLiveFixture)
		s.sc.mu.Lock()
		s.sc.cur = nil
		s.sc.mu.Unlock()

		st1, lm1, body1 := rawGet(t, srv.URL+path, "")
		if st1 != 200 || lm1 != "" {
			t.Fatalf("%s: first GET %d, Last-Modified %q; want 200 and no Last-Modified", path, st1, lm1)
		}
		// Same body, same memoized sweep, a conditional GET any stamp would
		// satisfy → still 200 with the body.
		if st, lm, body := rawGet(t, srv.URL+path, future); st != 200 || lm != "" || len(body) == 0 {
			t.Errorf("%s: conditional GET → %d, Last-Modified %q, body %d bytes; want 200 with the body", path, st, lm, len(body))
		}
		// A changed component (same second of the clock) → 200, never 304.
		setMacro(macroLiveChanged(t))
		s.sc.mu.Lock()
		s.sc.cur = nil
		s.sc.mu.Unlock()
		st2, lm2, body2 := rawGet(t, srv.URL+path, future)
		if st2 != 200 || lm2 != "" {
			t.Errorf("%s: changed body with a conditional GET → %d, Last-Modified %q; want 200, none", path, st2, lm2)
		}
		// The body really changed, and carries the changed macro read (the
		// macro card is the winner here: every priority agent is dead).
		if string(body1) == string(body2) || !strings.Contains(string(body2), "95/100") ||
			strings.Contains(string(body2), "83/100") {
			t.Errorf("%s: the second body must carry the changed macro card (95, not 83):\n%s", path, body2)
		}
		if path == "/agents/digest" {
			// data_as_of stays the OLDEST reading (the whale snapshot) and
			// did not move while the macro card changed.
			if a, b := dataAsOf(body1), dataAsOf(body2); a != "2026-08-18T06:00:00Z" || a != b {
				t.Errorf("digest data_as_of %q → %q, want the stable oldest reading", a, b)
			}
		}
	}
}
