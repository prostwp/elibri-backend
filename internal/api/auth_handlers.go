package api

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/prostwp/elibri-backend/internal/auth"
	"github.com/prostwp/elibri-backend/internal/config"
	"github.com/prostwp/elibri-backend/internal/email"
	"github.com/prostwp/elibri-backend/internal/ml"
	"github.com/prostwp/elibri-backend/internal/store"
)

// defaultTermsVersion is the backend fallback stamped on consent when the
// client sends an empty terms_version. The frontend's TERMS_VERSION
// (src/lib/legalVersion.ts) is the canonical source; this only guards the
// "client omitted it" path so the column is never silently NULL on consent.
const defaultTermsVersion = "2026-06"

type registerReq struct {
	Email       string `json:"email"`
	Password    string `json:"password"`
	DisplayName string `json:"display_name"`
	// Sub-chat C — consent gate. AcceptedTerms MUST be true (one checkbox
	// covering Terms + Privacy + Risk). TermsVersion records which revision
	// the user agreed to; empty falls back to defaultTermsVersion.
	AcceptedTerms bool   `json:"accepted_terms"`
	TermsVersion  string `json:"terms_version"`
}

type authResponse struct {
	Token string    `json:"token"`
	User  auth.User `json:"user"`
}

func handleRegister(cfg *config.Config) http.HandlerFunc {
	adminEmails := splitCSV(cfg.AdminEmails)
	return func(w http.ResponseWriter, r *http.Request) {
		if store.Pool == nil {
			writeError(w, http.StatusServiceUnavailable, "database unavailable")
			return
		}
		var req registerReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON")
			return
		}
		req.Email = strings.ToLower(strings.TrimSpace(req.Email))
		if req.Email == "" || len(req.Password) < 6 {
			writeError(w, http.StatusBadRequest, "email and password (min 6 chars) required")
			return
		}
		// Consent gate (sub-chat C): the single Terms+Privacy+Risk checkbox
		// must be ticked. We reject BEFORE creating the user so a declined
		// consent never leaves a half-registered row.
		if !req.AcceptedTerms {
			writeError(w, http.StatusBadRequest, "you must accept the terms")
			return
		}
		termsVersion := strings.TrimSpace(req.TermsVersion)
		if termsVersion == "" {
			termsVersion = defaultTermsVersion
		}

		role := "user"
		for _, a := range adminEmails {
			if strings.EqualFold(a, req.Email) {
				role = "admin"
				break
			}
		}

		// New users start unverified (email_verified defaults FALSE in the
		// schema). Login still works — the frontend shows a "verify" banner.
		u, err := auth.CreateUser(r.Context(), store.Pool, req.Email, req.Password, req.DisplayName, role)
		if err == auth.ErrUserExists {
			writeError(w, http.StatusConflict, "email already registered")
			return
		}
		if err != nil {
			writeError(w, http.StatusInternalServerError, "failed to create user")
			return
		}

		// Record consent separately so CreateUser's signature stays untouched
		// (it is called from other paths). Best-effort: a failed consent write
		// is logged but does not abort registration — the account exists and
		// the user can re-consent later if we ever surface it.
		if _, err := store.Pool.Exec(r.Context(),
			`UPDATE users SET terms_accepted_at = NOW(), terms_version = $1, updated_at = NOW() WHERE id = $2`,
			termsVersion, u.ID,
		); err != nil {
			log.Printf("WARN: failed to record consent for user %s: %v", u.ID, err)
		} else {
			// Reflect the consent in the response object the client gets back.
			now := time.Now()
			u.TermsAcceptedAt = &now
		}

		// Issue an email-verification token and fire the Resend email. A send
		// failure is NON-FATAL: log WARN and still return 200 — the user can
		// hit /auth/resend-verification. Dev mode (no RESEND_API_KEY) logs the
		// link to stdout instead of calling the API.
		if rawTok, terr := auth.IssueAuthToken(r.Context(), store.Pool, u.ID, auth.PurposeEmailVerify, auth.EmailVerifyTTL); terr != nil {
			log.Printf("WARN: failed to issue verify token for user %s: %v", u.ID, terr)
		} else {
			subject, html := email.VerifyEmailHTML(cfg.AppBaseURL, rawTok)
			if serr := email.SendEmail(r.Context(), cfg.ResendAPIKey, cfg.ResendFrom, u.Email, subject, html); serr != nil {
				log.Printf("WARN: failed to send verification email to %s: %v", u.Email, serr)
			}
		}

		token, err := auth.IssueToken(cfg.JWTSecret, u.ID, u.Email, u.Role)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "failed to issue token")
			return
		}
		writeJSON(w, authResponse{Token: token, User: *u})
	}
}

type loginReq struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

func handleLogin(cfg *config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if store.Pool == nil {
			writeError(w, http.StatusServiceUnavailable, "database unavailable")
			return
		}
		var req loginReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON")
			return
		}
		u, err := auth.Authenticate(r.Context(), store.Pool, req.Email, req.Password)
		if err == auth.ErrInvalidCredentials {
			writeError(w, http.StatusUnauthorized, "invalid credentials")
			return
		}
		if err != nil {
			writeError(w, http.StatusInternalServerError, "auth error")
			return
		}
		token, err := auth.IssueToken(cfg.JWTSecret, u.ID, u.Email, u.Role)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "failed to issue token")
			return
		}
		writeJSON(w, authResponse{Token: token, User: *u})
	}
}

func handleMe(w http.ResponseWriter, r *http.Request) {
	if store.Pool == nil {
		writeError(w, http.StatusServiceUnavailable, "database unavailable")
		return
	}
	userID := auth.GetUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	u, err := auth.GetUserByID(r.Context(), store.Pool, userID)
	if err == auth.ErrUserNotFound {
		writeError(w, http.StatusNotFound, "user not found")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to load user")
		return
	}
	writeJSON(w, u)
}

type updateMeReq struct {
	DisplayName *string `json:"display_name"`
	// RiskTier (Patch 2C): optional risk-tier change. When nil, column untouched.
	// Validated against ml.IsValidTier — only "conservative"|"balanced"|"aggressive".
	RiskTier *string `json:"risk_tier,omitempty"`
}

func handleUpdateMe(w http.ResponseWriter, r *http.Request) {
	if store.Pool == nil {
		writeError(w, http.StatusServiceUnavailable, "database unavailable")
		return
	}
	userID := auth.GetUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	var req updateMeReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	if req.DisplayName != nil {
		name := strings.TrimSpace(*req.DisplayName)
		if len(name) > 100 {
			writeError(w, http.StatusBadRequest, "display_name must be under 100 characters")
			return
		}
		_, err := store.Pool.Exec(r.Context(),
			`UPDATE users SET display_name = $1, updated_at = NOW() WHERE id = $2`,
			name, userID)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "failed to update")
			return
		}
	}
	if req.RiskTier != nil {
		tier := strings.ToLower(strings.TrimSpace(*req.RiskTier))
		if !ml.IsValidTier(tier) {
			writeError(w, http.StatusBadRequest, "risk_tier must be one of: conservative, balanced, aggressive")
			return
		}
		_, err := store.Pool.Exec(r.Context(),
			`UPDATE users SET risk_tier = $1, updated_at = NOW() WHERE id = $2`,
			tier, userID)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "failed to update risk_tier")
			return
		}
	}
	u, err := auth.GetUserByID(r.Context(), store.Pool, userID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to reload user")
		return
	}
	writeJSON(w, u)
}

func splitCSV(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
