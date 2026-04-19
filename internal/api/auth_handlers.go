package api

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/prostwp/elibri-backend/internal/auth"
	"github.com/prostwp/elibri-backend/internal/config"
	"github.com/prostwp/elibri-backend/internal/store"
)

type registerReq struct {
	Email       string `json:"email"`
	Password    string `json:"password"`
	DisplayName string `json:"display_name"`
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

		role := "user"
		for _, a := range adminEmails {
			if strings.EqualFold(a, req.Email) {
				role = "admin"
				break
			}
		}

		u, err := auth.CreateUser(r.Context(), store.Pool, req.Email, req.Password, req.DisplayName, role)
		if err == auth.ErrUserExists {
			writeError(w, http.StatusConflict, "email already registered")
			return
		}
		if err != nil {
			writeError(w, http.StatusInternalServerError, "failed to create user")
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
