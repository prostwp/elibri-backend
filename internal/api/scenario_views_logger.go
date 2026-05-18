package api

// scenario_views_logger.go — Sprint 3 backend.
//
// Helpers for logging public-scenario detail views into scenario_views
// (table created in migration 011). The actual INSERT is fire-and-forget:
// handleScenariosPublicDetail kicks off a goroutine after the snapshot has
// been written so a slow analytics write never blocks the user's request.
//
// Why a separate file:
//   scenarios_public_handlers.go is already large (~640 LOC). Pulling the
//   view-logging concern out keeps the catalog handlers focused on data
//   shaping and lets the IP-extraction / hashing helpers be unit-tested
//   without touching the HTTP wiring.
//
// Design choices documented in-line where it matters:
//   - extractClientIP priorities: X-Forwarded-For first hop > X-Real-IP >
//     RemoteAddr. We parse the *first* X-Forwarded-For hop only — the rest
//     of the chain is whatever proxies appended and is not the actual user.
//   - ip_hash uses SHA-256 over (ip + salt) and is stored as a 64-char hex
//     string (matches the CHAR(64) column). The salt comes from
//     IP_HASH_SALT env var; if missing we fall back to a known constant
//     and warn ONCE so the operator notices in logs but the analytics
//     pipeline doesn't break.
//   - The INSERT runs in its own goroutine with a 2 s context timeout so a
//     slow write never holds the user's response. INSERT errors are
//     logged at debug level and dropped — view counts are advisory.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// defaultIPHashSalt is used when IP_HASH_SALT is not set. The string is
// deliberately ugly so a `grep` over the deployed config catches it
// before production. It is NOT a secret — anyone with read access to
// scenario_views can grind a small list of IPs against it. The mitigation
// is to set a real salt in production; the fallback exists only to keep
// the analytics pipeline functional in dev / smoke environments.
const defaultIPHashSalt = "elibri-default-salt-do-not-use-in-prod"

// ipSaltOnce + ipSaltWarned ensure the "no salt set" warning prints once
// per process. Without this, every public-scenario detail request would
// emit a log line — noisy in dev, useless in prod.
var (
	ipSaltOnce    sync.Once
	ipSaltWarned  bool
	ipSaltCached  string
)

// resolveIPHashSalt reads IP_HASH_SALT and caches the result. The first
// call may print a warning if the salt is unset; subsequent calls are
// silent and just return the cached value.
func resolveIPHashSalt() string {
	ipSaltOnce.Do(func() {
		v := strings.TrimSpace(os.Getenv("IP_HASH_SALT"))
		if v == "" {
			log.Printf("WARNING: IP_HASH_SALT not set — falling back to default salt for scenario_views.ip_hash. " +
				"Set IP_HASH_SALT to a strong random string in production.")
			ipSaltWarned = true
			ipSaltCached = defaultIPHashSalt
			return
		}
		ipSaltCached = v
	})
	return ipSaltCached
}

// extractClientIP picks the most-trustworthy client IP from a request.
//
// Priority order (highest first):
//   1. First hop of X-Forwarded-For (the originally-connecting client as
//      reported by the closest reverse proxy — assuming the proxy is
//      trusted; in our deployment Cloudflare → Caddy → Go, both rewrite
//      this header)
//   2. X-Real-IP (commonly set by single-hop proxies like nginx)
//   3. r.RemoteAddr — the connecting socket address
//
// We strip ports off RemoteAddr (it always carries one in net/http) but
// we trust X-Forwarded-For / X-Real-IP to be IP-only. If a header value
// is malformed (e.g. empty hop, whitespace, "unknown") we silently fall
// through to the next priority — better to log a slightly less precise
// IP than to drop the row.
func extractClientIP(r *http.Request) string {
	if v := r.Header.Get("X-Forwarded-For"); v != "" {
		// The header looks like "client, proxy1, proxy2". The first hop
		// is the actual client (assuming the closest proxy is trusted).
		// strings.Split + trim handles the typical comma-space format
		// and the rare comma-only one without an extra regexp.
		first := strings.TrimSpace(strings.SplitN(v, ",", 2)[0])
		if first != "" && strings.ToLower(first) != "unknown" {
			return first
		}
	}
	if v := strings.TrimSpace(r.Header.Get("X-Real-IP")); v != "" {
		return v
	}
	// RemoteAddr is "ip:port" for TCP — net.SplitHostPort handles both
	// IPv4 (1.2.3.4:5678) and IPv6 ([::1]:5678). On parse failure we
	// fall back to the raw string so we never lose the value entirely.
	if r.RemoteAddr != "" {
		host, _, err := net.SplitHostPort(r.RemoteAddr)
		if err == nil {
			return host
		}
		return r.RemoteAddr
	}
	return ""
}

// hashIP returns a hex-encoded SHA-256 of (ip + salt), exactly 64 chars
// wide. Deterministic for the same (ip, salt) pair so we can build
// "unique viewers per scenario" counters later by COUNT(DISTINCT ip_hash).
//
// Empty inputs are accepted to keep the call site simple — if the IP is
// unknown we still record a row, just with a hash of (empty + salt). The
// analytics layer can filter those out by joining against the column
// length / a known marker if it ever matters.
func hashIP(ip, salt string) string {
	h := sha256.Sum256([]byte(ip + salt))
	return hex.EncodeToString(h[:])
}

// logScenarioViewAsync schedules a fire-and-forget INSERT into
// scenario_views. Returns immediately — the actual write runs in a new
// goroutine with a 2 s timeout context so a hung pool never piles up
// goroutines indefinitely.
//
// Failure modes (all non-fatal):
//   - pool nil: skip silently. Caller should already have requireDB'd.
//   - INSERT errors: logged + dropped. View counts are analytics, not
//     a transactional concern.
//   - Context timeout (>2 s): Postgres rolls the statement back; the
//     user's request was already served by the parent handler.
//
// We capture only primitives (strategyID, viewerUserID, ipHash) — never
// the *http.Request — so the goroutine can outlive the handler stack
// without holding onto the request body / connection.
func logScenarioViewAsync(pool *pgxpool.Pool, strategyID, viewerUserID, ipHash string) {
	if pool == nil || strategyID == "" {
		return
	}
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		// viewer_user is nullable in the schema; pass *string so a
		// missing user id (anonymous browser) goes in as NULL rather
		// than the empty UUID, which would FK-violate users(id).
		var viewerArg any
		if viewerUserID != "" {
			viewerArg = viewerUserID
		} else {
			viewerArg = nil
		}
		// ip_hash is nullable too; we always pass a 64-char hash, but
		// guarding against an unexpected empty hash here keeps the
		// schema invariant (CHAR(64)) intact.
		var hashArg any
		if len(ipHash) == 64 {
			hashArg = ipHash
		} else {
			hashArg = nil
		}
		if _, err := pool.Exec(ctx,
			`INSERT INTO scenario_views (strategy_id, viewer_user, ip_hash) VALUES ($1, $2, $3)`,
			strategyID, viewerArg, hashArg,
		); err != nil {
			// Debug-level event: routine analytics noise (pool busy,
			// row already collected by partition GC, etc). Keep the
			// log line short so it doesn't drown out scenario-runner
			// output in journalctl.
			log.Printf("scenario_views insert dropped (strategy=%s): %v", strategyID, err)
		}
	}()
}
