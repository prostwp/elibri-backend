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
			if isPublicPath(r.URL.Path) {
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
}

func isPublicPath(path string) bool {
	for _, p := range publicPaths {
		if path == p {
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
