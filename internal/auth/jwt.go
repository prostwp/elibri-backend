package auth

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

type contextKey string

const UserIDKey contextKey = "user_id"
const UserEmailKey contextKey = "user_email"
const UserRoleKey contextKey = "user_role"

type Claims struct {
	UserID string `json:"sub"`
	Email  string `json:"email"`
	Role   string `json:"role"`
	jwt.RegisteredClaims
}

const (
	accessTokenTTL = 24 * time.Hour
	// Issuer is stamped on every token we issue and verified on every parse.
	// Prevents a sibling service (or leaked dev secret) from minting tokens
	// acceptable to this backend.
	issuerName = "elibri-backend"
)

func IssueToken(secret, userID, email, role string) (string, error) {
	now := time.Now()
	claims := Claims{
		UserID: userID,
		Email:  email,
		Role:   role,
		RegisteredClaims: jwt.RegisteredClaims{
			IssuedAt:  jwt.NewNumericDate(now),
			ExpiresAt: jwt.NewNumericDate(now.Add(accessTokenTTL)),
			Issuer:    issuerName,
			Subject:   userID,
		},
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(secret))
}

func parseToken(tokenStr, secret string) (*Claims, error) {
	parsed, err := jwt.ParseWithClaims(tokenStr, &Claims{}, func(t *jwt.Token) (interface{}, error) {
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", t.Header["alg"])
		}
		return []byte(secret), nil
	})
	if err != nil {
		return nil, err
	}
	claims, ok := parsed.Claims.(*Claims)
	if !ok || !parsed.Valid {
		return nil, errors.New("invalid token")
	}
	if claims.UserID == "" {
		return nil, errors.New("missing user_id")
	}
	// Reject tokens issued by a different service that happens to share the
	// HMAC secret. The ParseWithClaims chain does not validate Issuer by
	// default — we enforce it explicitly.
	if claims.Issuer != issuerName {
		return nil, fmt.Errorf("invalid issuer: %s", claims.Issuer)
	}
	return claims, nil
}

// Middleware validates own JWT and injects user info into context.
// Public paths (health, auth/register, auth/login) bypass auth.
func Middleware(secret string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Public paths bypass the missing-token rejection but still
			// parse a Bearer token if one was provided. This lets the
			// public-detail handler (Sprint 3) attribute scenario views
			// to a known user when the browser carries a JWT, without
			// blocking anonymous requests that just want to read.
			if isPublicPath(r.URL.Path) {
				if tokenStr := extractToken(r); tokenStr != "" {
					if claims, err := parseToken(tokenStr, secret); err == nil {
						ctx := context.WithValue(r.Context(), UserIDKey, claims.UserID)
						ctx = context.WithValue(ctx, UserEmailKey, claims.Email)
						ctx = context.WithValue(ctx, UserRoleKey, claims.Role)
						r = r.WithContext(ctx)
					}
					// Invalid tokens on public paths are silently
					// ignored — we don't want to 401 a legitimate
					// public-page load just because a stale JWT is in
					// localStorage. The user still gets a response;
					// the view row just lands as anonymous.
				}
				next.ServeHTTP(w, r)
				return
			}

			tokenStr := extractToken(r)
			if tokenStr == "" {
				http.Error(w, `{"error":"missing authorization"}`, http.StatusUnauthorized)
				return
			}

			claims, err := parseToken(tokenStr, secret)
			if err != nil {
				http.Error(w, `{"error":"invalid token"}`, http.StatusUnauthorized)
				return
			}

			ctx := context.WithValue(r.Context(), UserIDKey, claims.UserID)
			ctx = context.WithValue(ctx, UserEmailKey, claims.Email)
			ctx = context.WithValue(ctx, UserRoleKey, claims.Role)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

var publicPaths = []string{
	"/health",
	"/ready",
	"/api/v1/auth/register",
	"/api/v1/auth/login",
	// Email verification + password reset (sub-chat C). These run pre-auth:
	// the user clicks a link from an email and has no Bearer token. The token
	// in the request body is the credential. resend-verification is NOT here
	// (it reads the authenticated user). follows/watchlist/notification-prefs
	// are NOT here either — they are per-user and stay protected.
	"/api/v1/auth/verify-email",
	"/api/v1/auth/forgot-password",
	"/api/v1/auth/reset-password",
	// Public catalog (Sprint 2 v3 SaaS redesign). Featured + list are exact
	// matches; detail uses prefix matching below because it carries a slug.
	"/api/v1/scenarios/featured",
	"/api/v1/scenarios/public",
	// Narrative Radar feed — no private user data, used by guest landing
	// and by the Builder dashboard. Keeping it auth-gated burned smoke
	// time for curl tests with no upside (worker output is the same for
	// all users — there is no per-user filtering yet).
	"/api/v1/narratives",
	// Fear & Greed proxy — no per-user data; guest landing reads it. Server
	// proxies CMC because CMC blocks browser CORS.
	"/api/v1/market/fear-greed",
	// Relative-strength + volume trend for the Crypto Sentiment liquidity
	// panel. No per-user data; same public tier as the F&G proxy.
	"/api/v1/market/momentum",
	// AlphaVizor AI market-mood read — no per-user data; same public tier
	// as F&G and momentum. Frontend hides the block when "read" is empty.
	"/api/v1/market/mood-read",
	// Whale Flow feed — latest snapshot per asset + live transfer feed. No
	// per-user data; same public tier as the Narrative Radar feed. Used by
	// the scenario detail page and guest landing.
	"/api/v1/whale-flow",
	// Macro Sentiment regime — no per-user data; same public tier as whale-flow
	// and narratives. Exact match (leaf path, no slug). The macro calendar is
	// embedded in the response, so one allowlist line suffices. Distinct from
	// "/api/v1/macrocal" (the blackout calendar) — that resolves to its own route.
	"/api/v1/macro",
}

// publicPrefixes lets unauth GETs through when the path STARTS WITH one of
// these strings AND extends past the prefix (so we don't accidentally
// accept "/api/v1/scenarios/public/" with an empty slug — that should still
// 401 to make the bug obvious instead of returning a confusing 404 from
// the handler). Order is irrelevant: prefixes are mutually disjoint.
var publicPrefixes = []string{
	"/api/v1/scenarios/public/",
	"/api/v1/agents/",
	// Per-narrative click-target endpoint (.../narratives/{slug}/mentions).
	// Same public tier as /api/v1/narratives itself — no per-user data,
	// just article references for the Radar UI's source chips.
	"/api/v1/narratives/",
	// Funding Rate liquidations feed (.../funding/liquidations). No per-user
	// data; same public tier as whale-flow. Prefix form because the path
	// extends past the prefix with "liquidations" (the len>prefix check below
	// then lets the unauth GET through).
	"/api/v1/funding/",
}

func isPublicPath(path string) bool {
	for _, p := range publicPaths {
		if path == p {
			return true
		}
	}
	for _, prefix := range publicPrefixes {
		// HasPrefix + length check: "/api/v1/scenarios/public/abc" passes,
		// "/api/v1/scenarios/public/" alone does not (slug missing).
		if strings.HasPrefix(path, prefix) && len(path) > len(prefix) {
			return true
		}
	}
	return false
}

func extractToken(r *http.Request) string {
	h := r.Header.Get("Authorization")
	if strings.HasPrefix(h, "Bearer ") {
		return strings.TrimPrefix(h, "Bearer ")
	}
	return r.URL.Query().Get("token")
}

func GetUserID(r *http.Request) string {
	if v, ok := r.Context().Value(UserIDKey).(string); ok {
		return v
	}
	return ""
}

func GetUserRole(r *http.Request) string {
	if v, ok := r.Context().Value(UserRoleKey).(string); ok {
		return v
	}
	return ""
}

func IsAdmin(r *http.Request) bool {
	return GetUserRole(r) == "admin"
}
