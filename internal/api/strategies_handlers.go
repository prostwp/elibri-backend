package api

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/prostwp/elibri-backend/internal/auth"
	"github.com/prostwp/elibri-backend/internal/store"
)

func handleStrategiesList(w http.ResponseWriter, r *http.Request) {
	if store.Pool == nil {
		writeError(w, http.StatusServiceUnavailable, "database unavailable")
		return
	}
	userID := auth.GetUserID(r)
	items, err := auth.ListStrategies(r.Context(), store.Pool, userID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to list strategies")
		return
	}
	writeJSON(w, map[string]any{"strategies": items})
}

func handleStrategiesCreate(w http.ResponseWriter, r *http.Request) {
	if store.Pool == nil {
		writeError(w, http.StatusServiceUnavailable, "database unavailable")
		return
	}
	userID := auth.GetUserID(r)
	var s auth.Strategy
	if err := json.NewDecoder(r.Body).Decode(&s); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	s.UserID = userID
	if s.Name == "" {
		s.Name = "Untitled Strategy"
	}
	if s.Segment == "" {
		s.Segment = "pro"
	}
	if s.SelectedPair == "" {
		s.SelectedPair = "EURUSD"
	}
	if err := auth.CreateStrategy(r.Context(), store.Pool, &s); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to create strategy")
		return
	}
	writeJSON(w, s)
}

// GET/PUT/DELETE /api/v1/strategies/{id}
func handleStrategyByID(w http.ResponseWriter, r *http.Request) {
	if store.Pool == nil {
		writeError(w, http.StatusServiceUnavailable, "database unavailable")
		return
	}
	userID := auth.GetUserID(r)
	id := strings.TrimPrefix(r.URL.Path, "/api/v1/strategies/")
	if id == "" || strings.Contains(id, "/") {
		writeError(w, http.StatusBadRequest, "invalid strategy id")
		return
	}

	switch r.Method {
	case http.MethodGet:
		s, err := auth.GetStrategy(r.Context(), store.Pool, userID, id)
		if err == auth.ErrStrategyNotFound {
			writeError(w, http.StatusNotFound, "strategy not found")
			return
		}
		if err != nil {
			writeError(w, http.StatusInternalServerError, "failed to load strategy")
			return
		}
		writeJSON(w, s)

	case http.MethodPut:
		var s auth.Strategy
		if err := json.NewDecoder(r.Body).Decode(&s); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON")
			return
		}
		s.ID = id
		s.UserID = userID
		if err := auth.UpdateStrategy(r.Context(), store.Pool, &s); err != nil {
			if err == auth.ErrStrategyNotFound {
				writeError(w, http.StatusNotFound, "strategy not found")
				return
			}
			writeError(w, http.StatusInternalServerError, "failed to update strategy")
			return
		}
		writeJSON(w, s)

	case http.MethodDelete:
		if err := auth.DeleteStrategy(r.Context(), store.Pool, userID, id); err != nil {
			if err == auth.ErrStrategyNotFound {
				writeError(w, http.StatusNotFound, "strategy not found")
				return
			}
			writeError(w, http.StatusInternalServerError, "failed to delete strategy")
			return
		}
		writeJSON(w, map[string]string{"status": "deleted"})

	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}
