package api

// verify_handlers.go — email verification + password reset (sub-chat C).
//
// Endpoints:
//   POST /api/v1/auth/verify-email         (public)  {token}                -> {"status":"ok"}
//   POST /api/v1/auth/resend-verification  (auth)    {}                     -> {"status":"ok"}
//   POST /api/v1/auth/forgot-password      (public)  {email}                -> ALWAYS {"status":"ok"}
//   POST /api/v1/auth/reset-password       (public)  {token,new_password}   -> {"status":"ok"}
//
// Token mechanics live in internal/auth/tokens.go (hashed, single-use,
// one-active-per-(user,purpose)). Email rendering/sending lives in
// internal/email. Public-path bypass for verify/forgot/reset is wired in
// internal/auth/jwt.go publicPaths — resend stays protected.

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/prostwp/elibri-backend/internal/auth"
	"github.com/prostwp/elibri-backend/internal/config"
	"github.com/prostwp/elibri-backend/internal/email"
	"github.com/prostwp/elibri-backend/internal/store"
)

type verifyEmailReq struct {
	Token string `json:"token"`
}

// handleVerifyEmail consumes an email_verify token and flips
// users.email_verified = TRUE. Idempotent at the token level: a token is
// single-use, so a double-submit of the same link yields 400 the second time.
// We deliberately do not distinguish invalid/expired/used (anti-probe).
func handleVerifyEmail(w http.ResponseWriter, r *http.Request) {
	if !requireDB(w) {
		return
	}
	var req verifyEmailReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	req.Token = strings.TrimSpace(req.Token)
	if req.Token == "" {
		writeError(w, http.StatusBadRequest, "token required")
		return
	}

	userID, err := auth.ConsumeAuthToken(r.Context(), store.Pool, req.Token, auth.PurposeEmailVerify)
	if err == auth.ErrTokenInvalid {
		writeError(w, http.StatusBadRequest, "invalid or expired token")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to verify email")
		return
	}

	if _, err := store.Pool.Exec(r.Context(),
		`UPDATE users SET email_verified = TRUE, updated_at = NOW() WHERE id = $1`, userID,
	); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to mark email verified")
		return
	}
	writeJSON(w, map[string]string{"status": "ok"})
}

// handleResendVerification re-issues a verify link for the authenticated
// user. 409 if already verified (no point emailing). Requires auth (NOT in
// publicPaths). Email-send failure is non-fatal — same rationale as register.
func handleResendVerification(cfg *config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !requireDB(w) {
			return
		}
		userID := auth.GetUserID(r)
		if userID == "" {
			writeError(w, http.StatusUnauthorized, "unauthorized")
			return
		}

		u, err := auth.GetUserByID(r.Context(), store.Pool, userID)
		if err == auth.ErrUserNotFound {
			writeError(w, http.StatusUnauthorized, "unauthorized")
			return
		}
		if err != nil {
			writeError(w, http.StatusInternalServerError, "failed to load user")
			return
		}
		if u.EmailVerified {
			writeError(w, http.StatusConflict, "email already verified")
			return
		}

		rawTok, err := auth.IssueAuthToken(r.Context(), store.Pool, u.ID, auth.PurposeEmailVerify, auth.EmailVerifyTTL)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "failed to issue token")
			return
		}
		subject, html := email.VerifyEmailHTML(cfg.AppBaseURL, rawTok)
		if serr := email.SendEmail(r.Context(), cfg.ResendAPIKey, cfg.ResendFrom, u.Email, subject, html); serr != nil {
			log.Printf("WARN: failed to resend verification email to %s: %v", u.Email, serr)
		}
		writeJSON(w, map[string]string{"status": "ok"})
	}
}

type forgotPasswordReq struct {
	Email string `json:"email"`
}

// handleForgotPassword is anti-enumeration: it ALWAYS returns 200
// {"status":"ok"} for a well-formed request, whether or not the email maps to
// a real account. Only a missing email / malformed JSON yields 400. When the
// user does exist we issue a reset token + send the email; otherwise we do
// nothing — but the response is byte-identical either way.
func handleForgotPassword(cfg *config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !requireDB(w) {
			return
		}
		var req forgotPasswordReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON")
			return
		}
		req.Email = strings.ToLower(strings.TrimSpace(req.Email))
		if req.Email == "" {
			writeError(w, http.StatusBadRequest, "email required")
			return
		}

		// Look the user up; on miss we fall straight through to the neutral
		// 200. On hit we best-effort issue+send — a failure there is ALSO
		// swallowed so timing/observable behaviour stays uniform.
		u, err := auth.GetUserByEmail(r.Context(), store.Pool, req.Email)
		if err == nil && u != nil {
			if rawTok, terr := auth.IssueAuthToken(r.Context(), store.Pool, u.ID, auth.PurposePasswordReset, auth.PasswordResetTTL); terr != nil {
				log.Printf("WARN: failed to issue reset token for user %s: %v", u.ID, terr)
			} else {
				subject, html := email.ResetPasswordHTML(cfg.AppBaseURL, rawTok)
				if serr := email.SendEmail(r.Context(), cfg.ResendAPIKey, cfg.ResendFrom, u.Email, subject, html); serr != nil {
					log.Printf("WARN: failed to send reset email to %s: %v", u.Email, serr)
				}
			}
		} else if err != nil && err != auth.ErrUserNotFound {
			// A real DB error (not "no such user") — log it, but STILL return
			// the neutral 200 so the response can't be used to probe accounts.
			log.Printf("WARN: forgot-password lookup error for %s: %v", req.Email, err)
		}

		writeJSON(w, map[string]string{"status": "ok"})
	}
}

type resetPwReq struct {
	Token       string `json:"token"`
	NewPassword string `json:"new_password"`
}

// handleResetPassword consumes a password_reset token and sets a new bcrypt
// hash. Password min length 6 (consistent with register). Invalid/expired/
// used token -> 400. Public path.
func handleResetPassword(w http.ResponseWriter, r *http.Request) {
	if !requireDB(w) {
		return
	}
	var req resetPwReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	req.Token = strings.TrimSpace(req.Token)
	if req.Token == "" {
		writeError(w, http.StatusBadRequest, "token required")
		return
	}
	if len(req.NewPassword) < 6 {
		writeError(w, http.StatusBadRequest, "password (min 6 chars) required")
		return
	}

	userID, err := auth.ConsumeAuthToken(r.Context(), store.Pool, req.Token, auth.PurposePasswordReset)
	if err == auth.ErrTokenInvalid {
		writeError(w, http.StatusBadRequest, "invalid or expired token")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to reset password")
		return
	}

	// ResetUserPassword bcrypts + updates. ErrUserNotFound here would mean the
	// user was deleted between token issue and use (CASCADE would normally
	// have nuked the token too) — treat as an invalid token to the client.
	if err := auth.ResetUserPassword(r.Context(), store.Pool, userID, req.NewPassword); err != nil {
		if err == auth.ErrUserNotFound || err == pgx.ErrNoRows {
			writeError(w, http.StatusBadRequest, "invalid or expired token")
			return
		}
		writeError(w, http.StatusInternalServerError, "failed to reset password")
		return
	}
	writeJSON(w, map[string]string{"status": "ok"})
}
