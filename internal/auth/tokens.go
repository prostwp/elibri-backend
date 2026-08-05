package auth

// tokens.go — single-use, hashed action tokens for email verification and
// password reset (sub-chat C, schema in migration 014_auth_dashboard.sql).
//
// Security model:
//   - The RAW token is what ships in the email link. We NEVER store it.
//   - The DB stores only hex(sha256(raw)) in auth_tokens.token_hash, so a
//     read-only DB leak can't be replayed against the verify/reset endpoints
//     (an attacker would need a sha256 preimage).
//   - A token is valid iff used_at IS NULL AND expires_at > NOW(). On use we
//     stamp used_at = NOW() so it can't be replayed.
//   - "One active token per (user, purpose)": issuing a fresh token first
//     marks every prior un-used token of the same purpose as used. This caps
//     the blast radius of repeated forgot-password / resend requests (only
//     the latest link ever works).

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Token purposes — mirror the CHECK constraint in 014_auth_dashboard.sql.
const (
	PurposeEmailVerify   = "email_verify"
	PurposePasswordReset = "password_reset"
)

// Token TTLs (contract §A): verify links live a day, reset links an hour.
const (
	EmailVerifyTTL   = 24 * time.Hour
	PasswordResetTTL = 1 * time.Hour
)

// ErrTokenInvalid is returned by ConsumeAuthToken when the supplied raw token
// does not match an active (un-used, un-expired) row for the given purpose.
// The handler maps this to a 400 — deliberately not distinguishing
// invalid/expired/used so we don't leak token state to a guesser.
var ErrTokenInvalid = errors.New("token invalid, expired, or already used")

// GenerateAuthToken returns a freshly minted action token.
//
//	raw       — 32 random bytes, base64-RawURL encoded (goes in the email link)
//	sha256hex — hex(sha256(raw)), 64 chars (goes in auth_tokens.token_hash)
//
// RawURLEncoding keeps the token URL-safe (no +,/,= to percent-escape).
func GenerateAuthToken() (raw, sha256hex string, err error) {
	b := make([]byte, 32)
	if _, err = rand.Read(b); err != nil {
		return "", "", err
	}
	raw = base64.RawURLEncoding.EncodeToString(b)
	sha256hex = hashToken(raw)
	return raw, sha256hex, nil
}

// hashToken returns hex(sha256(raw)) — the canonical stored form. Used both
// when issuing (store this) and when consuming (hash the input, look it up).
func hashToken(raw string) string {
	sum := sha256.Sum256([]byte(raw))
	return hex.EncodeToString(sum[:])
}

// IssueAuthToken invalidates any prior active token of the same purpose for
// the user, then inserts a new one. Returns the RAW token to embed in the
// email link. ttl picks expiry (EmailVerifyTTL / PasswordResetTTL).
//
// Runs in a transaction so the invalidate+insert is atomic — a crash between
// the two can't leave two active tokens.
func IssueAuthToken(ctx context.Context, pool *pgxpool.Pool, userID, purpose string, ttl time.Duration) (raw string, err error) {
	raw, hash, err := GenerateAuthToken()
	if err != nil {
		return "", err
	}

	tx, err := pool.Begin(ctx)
	if err != nil {
		return "", err
	}
	defer func() {
		if tx != nil {
			_ = tx.Rollback(ctx)
		}
	}()

	// One active token per (user, purpose): retire the old ones first.
	if _, err = tx.Exec(ctx,
		`UPDATE auth_tokens SET used_at = NOW()
		 WHERE user_id = $1 AND purpose = $2 AND used_at IS NULL`,
		userID, purpose,
	); err != nil {
		return "", err
	}

	if _, err = tx.Exec(ctx,
		`INSERT INTO auth_tokens (user_id, token_hash, purpose, expires_at)
		 VALUES ($1, $2, $3, NOW() + $4::interval)`,
		userID, hash, purpose, ttl.String(),
	); err != nil {
		return "", err
	}

	if err = tx.Commit(ctx); err != nil {
		return "", err
	}
	tx = nil
	return raw, nil
}

// ConsumeAuthToken looks up the raw token by its hash for the given purpose,
// verifies it is active, marks it used, and returns the owning user_id.
//
// The UPDATE ... WHERE used_at IS NULL AND expires_at > NOW() ... RETURNING
// is a single atomic statement: two concurrent consume attempts can't both
// succeed (the second sees used_at already set and affects zero rows). Any
// zero-row outcome maps to ErrTokenInvalid.
func ConsumeAuthToken(ctx context.Context, pool *pgxpool.Pool, rawToken, purpose string) (userID string, err error) {
	hash := hashToken(rawToken)
	err = pool.QueryRow(ctx, `
		UPDATE auth_tokens
		   SET used_at = NOW()
		 WHERE token_hash = $1
		   AND purpose    = $2
		   AND used_at IS NULL
		   AND expires_at > NOW()
		RETURNING user_id
	`, hash, purpose).Scan(&userID)
	if err == pgx.ErrNoRows {
		return "", ErrTokenInvalid
	}
	if err != nil {
		return "", err
	}
	return userID, nil
}
