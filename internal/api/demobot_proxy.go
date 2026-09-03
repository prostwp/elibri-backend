package api

// demobot_proxy.go — read-only pass-through to the demobot agents API.
//
// WHY A PROXY AT ALL. The demobot serves 13 live agents over HTTP, but a
// browser cannot reach it: the port is bound to the host and closed from the
// outside on purpose (the team asked for that), and the server's nginx is not
// running. The Go backend is the one process the frontend already talks to,
// so it is the natural place to expose the agents without opening a port.
//
// It is NOT the mock at /api/v1/agents/{slug}/output. That one serves embedded
// sample JSON authored by product; this one forwards to the running demobot
// and returns whatever the live agent actually says, degraded states included.
//
// AUTHENTICATED. This prefix is deliberately NOT in publicPrefixes: the
// demobot's own rate limiter is switched off (its container consumer polices
// itself at 8 rps, and a 429 to that consumer is worse than no limiter), so an
// anonymous public route here would hand anyone unmetered access to sweeps
// that fan out to Binance, Yahoo and the AI provider. Each decision was right
// alone; together they were a hole. Auth is what closes it.
//
// Deliberately narrow:
//   - GET only. The demobot API is read-only; anything else is a 405 here
//     rather than something the proxy has to reason about.
//   - A fixed allowlist of upstream paths. No arbitrary path pass-through, so
//     this can never be walked into an SSRF against another service.
//   - Upstream body capped and timed out: a hung demobot must degrade this
//     endpoint, not pin a backend goroutine.

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"
)

// demobotBaseURL is where the demobot listens. Same host in every deployment
// we run, so localhost is the honest default; DEMOBOT_URL overrides it.
func demobotBaseURL() string {
	if v := strings.TrimSpace(os.Getenv("DEMOBOT_URL")); v != "" {
		return strings.TrimRight(v, "/")
	}
	return "http://127.0.0.1:8090"
}

const (
	// Longer than the upstream's own worst case, not equal to it: a digest
	// sweep budgets 25s for gathering and may add an AI call on top, which is
	// why the demobot's own write timeout is 60s. Matching 25s here cut
	// healthy slow requests off as 504 before upstream was allowed to finish.
	demobotTimeout  = 45 * time.Second
	demobotMaxBytes = 2 << 20 // 2 MB — a showcase sweep is ~40 KB
)

// demobotAllowed maps the public suffix to the upstream path. An explicit map,
// not string concatenation: the proxy must never forward a path a caller
// composed.
var demobotAllowed = map[string]string{
	"agents":           "/agents",
	"showcase":         "/showcase",
	"showcase/example": "/showcase/example",
	"digest":           "/agents/digest",
	"top":              "/agents/top",
}

// demobotAgentName is the shape a per-agent path segment may take:
// /api/v1/demobot/agents/{name}.
//
// A CHARACTER CLASS, not a list of known agents. An explicit list looks safer
// and is actually a second definition of "which agents exist" — it drifts the
// day someone adds one to the demobot, and the new agent 404s here for no
// visible reason. The security property never came from knowing the names: it
// comes from the segment being unable to contain "/", "..", ":" or "@", so it
// cannot escape the path or redirect the host. What exists is the demobot's
// answer to give, and its own 404 travels back untouched.
var demobotAgentName = regexp.MustCompile(`^[a-z][a-z0-9_-]{0,31}$`)

// demobotQueryKeys are the only query parameters forwarded upstream;
// everything else is dropped rather than passed along.
//
// The risk calculator's four inputs belong here too. They were missing, so
// /agents/risk was reachable through the proxy but never answerable: the
// params were stripped and upstream replied "missing ?balance=".
var demobotQueryKeys = []string{
	"asset", "assets", "tf", // analysis agents
	"balance", "risk", "entry", "stop", // risk calculator
}

var demobotClient = &http.Client{Timeout: demobotTimeout}

// handleDemobotProxy is wired as GET /api/v1/demobot/{path...}.
func handleDemobotProxy(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "read-only endpoint")
		return
	}
	// Leading slash only. Trimming the TRAILING one collapsed "agents/" into
	// "agents", so a caller that built "agents/${name}" with an empty name
	// silently received the whole agent LIST — a different shape, answered
	// 200, hiding the caller's bug instead of surfacing it as a 404.
	suffix := strings.TrimPrefix(r.PathValue("path"), "/")

	upstream, ok := demobotAllowed[suffix]
	if !ok {
		// /agents/{name} — the only parameterised shape.
		if name, found := strings.CutPrefix(suffix, "agents/"); found && demobotAgentName.MatchString(name) {
			upstream = "/agents/" + name
		} else {
			writeError(w, http.StatusNotFound, "unknown demobot endpoint")
			return
		}
	}

	// Copy each allowed key VERBATIM: every value, and present-but-empty ones
	// too. Get+Set collapsed both, and each collapse silently changed the
	// request: "asset=btc&asset=eth" became a single asset, and an empty
	// "assets=" vanished entirely so upstream ran its DEFAULT scan and
	// returned real data for a query nobody made instead of the 400 it owes.
	// A proxy may refuse a request; it may not quietly answer a different one.
	src := r.URL.Query()
	q := url.Values{}
	for _, k := range demobotQueryKeys {
		if vs, ok := src[k]; ok {
			q[k] = append([]string(nil), vs...)
		}
	}
	target := demobotBaseURL() + upstream
	if len(q) > 0 {
		target += "?" + q.Encode()
	}

	ctx, cancel := context.WithTimeout(r.Context(), demobotTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to build upstream request")
		return
	}

	resp, err := demobotClient.Do(req)
	if err != nil {
		// The demobot being down is an expected operational state, not a bug
		// in this handler: 503 says "upstream unavailable", and the UI can
		// tell that apart from a 500.
		status := http.StatusServiceUnavailable
		if errors.Is(err, context.DeadlineExceeded) {
			status = http.StatusGatewayTimeout
		}
		writeError(w, status, "demobot unavailable")
		return
	}
	defer resp.Body.Close()

	// One byte past the cap, so an oversized body is DETECTED rather than
	// silently truncated. Truncated JSON forwarded with the upstream's 200 is
	// the worst outcome available: the client parses garbage as success.
	body, err := io.ReadAll(io.LimitReader(resp.Body, demobotMaxBytes+1))
	if err != nil {
		writeError(w, http.StatusBadGateway, "failed to read demobot response")
		return
	}
	if len(body) > demobotMaxBytes {
		writeError(w, http.StatusBadGateway, "demobot response too large")
		return
	}

	// The demobot's own status is passed through verbatim: an agent that
	// answers 200 with ok=false (an honest degraded card) must reach the UI
	// unchanged, and its 404s must not become our 200s.
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(body)
}
