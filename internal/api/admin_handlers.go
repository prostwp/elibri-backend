package api

import (
	"encoding/json"
	"net/http"

	"github.com/prostwp/elibri-backend/internal/auth"
	"github.com/prostwp/elibri-backend/internal/store"
)

func requireAdmin(w http.ResponseWriter, r *http.Request) bool {
	if !auth.IsAdmin(r) {
		writeError(w, http.StatusForbidden, "admin access required")
		return false
	}
	return true
}

func handleAdminListUsers(w http.ResponseWriter, r *http.Request) {
	if !requireAdmin(w, r) {
		return
	}
	if store.Pool == nil {
		writeError(w, http.StatusServiceUnavailable, "database unavailable")
		return
	}
	users, err := auth.ListUsers(r.Context(), store.Pool)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to list users")
		return
	}
	writeJSON(w, map[string]any{"users": users})
}

type resetPasswordReq struct {
	NewPassword string `json:"new_password"`
}

// POST /api/v1/admin/users/{id}/reset-password
// body: { "new_password": "..." } — OR empty → generates temp password
func handleAdminResetPassword(w http.ResponseWriter, r *http.Request) {
	if !requireAdmin(w, r) {
		return
	}
	if store.Pool == nil {
		writeError(w, http.StatusServiceUnavailable, "database unavailable")
		return
	}

	userID := r.PathValue("id")
	if userID == "" {
		writeError(w, http.StatusBadRequest, "missing user id")
		return
	}

	var req resetPasswordReq
	_ = json.NewDecoder(r.Body).Decode(&req)

	newPassword := req.NewPassword
	generated := false
	if newPassword == "" {
		tmp, err := auth.GenerateTempPassword()
		if err != nil {
			writeError(w, http.StatusInternalServerError, "failed to generate password")
			return
		}
		newPassword = tmp
		generated = true
	} else if len(newPassword) < 6 {
		writeError(w, http.StatusBadRequest, "password must be at least 6 characters")
		return
	}

	if err := auth.ResetUserPassword(r.Context(), store.Pool, userID, newPassword); err != nil {
		if err == auth.ErrUserNotFound {
			writeError(w, http.StatusNotFound, "user not found")
			return
		}
		writeError(w, http.StatusInternalServerError, "failed to reset password")
		return
	}

	resp := map[string]any{"status": "ok"}
	if generated {
		resp["temp_password"] = newPassword
	}
	w.Header().Set("Cache-Control", "no-store")
	writeJSON(w, resp)
}
