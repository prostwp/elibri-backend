package api

import (
	"encoding/json"
	"net/http"

	"github.com/prostwp/elibri-backend/internal/auth"
	"github.com/prostwp/elibri-backend/internal/config"
	"github.com/prostwp/elibri-backend/internal/store"
	"github.com/prostwp/elibri-backend/internal/telegram"
)

// POST /api/v1/telegram/link
// Issues a short-lived code. User types `/link <code>` in the bot to bind.
func handleTelegramLinkCreate(cfg *config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID := auth.GetUserID(r)
		if userID == "" {
			http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
			return
		}
		code, expiresAt, err := telegram.IssueLinkCode(r.Context(), store.Pool, userID)
		if err != nil {
			http.Error(w, `{"error":"failed to issue code"}`, http.StatusInternalServerError)
			return
		}

		resp := map[string]any{
			"code":         code,
			"expires_at":   expiresAt.UTC().Format("2006-01-02T15:04:05Z"),
			"bot_username": cfg.TelegramBotUsername,
		}
		if cfg.TelegramBotUsername != "" {
			resp["deeplink"] = "https://t.me/" + cfg.TelegramBotUsername + "?start=link_" + code
		}
		_ = json.NewEncoder(w).Encode(resp)
	}
}

// DELETE /api/v1/telegram/link
func handleTelegramUnlink(w http.ResponseWriter, r *http.Request) {
	userID := auth.GetUserID(r)
	if userID == "" {
		http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
		return
	}
	_, err := store.Pool.Exec(r.Context(), `
		UPDATE users SET telegram_chat_id = NULL, telegram_linked_at = NULL
		WHERE id = $1
	`, userID)
	if err != nil {
		http.Error(w, `{"error":"db error"}`, http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]any{"status": "unlinked"})
}

// Route multiplexer for /api/v1/telegram/link (POST creates, DELETE removes).
func handleTelegramLinkRouter(cfg *config.Config) http.HandlerFunc {
	create := handleTelegramLinkCreate(cfg)
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			create(w, r)
		case http.MethodDelete:
			handleTelegramUnlink(w, r)
		default:
			http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		}
	}
}
