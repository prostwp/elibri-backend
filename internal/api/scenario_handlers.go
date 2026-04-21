package api

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/prostwp/elibri-backend/internal/auth"
	"github.com/prostwp/elibri-backend/internal/scenario"
	"github.com/prostwp/elibri-backend/internal/store"
)

// runner is wired in from main.go via SetRunner so handlers can control the
// live scenario goroutine pool. nil-safe: endpoints fall back to DB-only
// behaviour when the runner isn't ready.
var runner *scenario.Runner

// SetRunner is called from cmd/server/main.go after Runner is constructed.
func SetRunner(r *scenario.Runner) { runner = r }

// POST /api/v1/scenarios/{id}/start
// Marks the strategy is_active=true, asks runner to spawn the loop.
func handleScenarioStart(w http.ResponseWriter, r *http.Request) {
	userID := auth.GetUserID(r)
	if userID == "" {
		http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
		return
	}
	id := extractScenarioID(r.URL.Path)
	if id == "" {
		http.Error(w, `{"error":"id required"}`, http.StatusBadRequest)
		return
	}

	// Ownership check + toggle active.
	ct, err := store.Pool.Exec(r.Context(), `
		UPDATE strategies SET is_active = true, paused_until = NULL, updated_at = NOW()
		WHERE id = $1 AND user_id = $2
	`, id, userID)
	if err != nil {
		http.Error(w, `{"error":"db error"}`, http.StatusInternalServerError)
		return
	}
	if ct.RowsAffected() == 0 {
		http.Error(w, `{"error":"not found"}`, http.StatusNotFound)
		return
	}

	if runner != nil {
		if err := runner.Start(r.Context(), id); err != nil {
			// Don't fail the HTTP call — strategy is active in DB; runner
			// will pick it up on next StartAllActive or restart.
			writeJSON(w, map[string]any{"status": "active_in_db", "runner_error": err.Error()})
			return
		}
	}
	writeJSON(w, map[string]any{"status": "started", "id": id})
}

// POST /api/v1/scenarios/{id}/stop
func handleScenarioStop(w http.ResponseWriter, r *http.Request) {
	userID := auth.GetUserID(r)
	if userID == "" {
		http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
		return
	}
	id := extractScenarioID(r.URL.Path)
	if id == "" {
		http.Error(w, `{"error":"id required"}`, http.StatusBadRequest)
		return
	}

	ct, err := store.Pool.Exec(r.Context(), `
		UPDATE strategies SET is_active = false, updated_at = NOW()
		WHERE id = $1 AND user_id = $2
	`, id, userID)
	if err != nil {
		http.Error(w, `{"error":"db error"}`, http.StatusInternalServerError)
		return
	}
	if ct.RowsAffected() == 0 {
		http.Error(w, `{"error":"not found"}`, http.StatusNotFound)
		return
	}

	if runner != nil {
		runner.Stop(id)
	}
	writeJSON(w, map[string]any{"status": "stopped", "id": id})
}

// GET /api/v1/scenarios/active
// Returns user's active scenarios with runtime status (running/pending).
func handleScenariosActive(w http.ResponseWriter, r *http.Request) {
	userID := auth.GetUserID(r)
	if userID == "" {
		http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
		return
	}

	rows, err := store.Pool.Query(r.Context(), `
		SELECT id, name, selected_pair, interval, risk_tier,
		       is_active, paused_until, last_signal_bar_time, last_signal_direction
		FROM strategies
		WHERE user_id = $1 AND is_active = true
		ORDER BY updated_at DESC
	`, userID)
	if err != nil {
		http.Error(w, `{"error":"db error"}`, http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	type item struct {
		ID                  string  `json:"id"`
		Name                string  `json:"name"`
		Symbol              string  `json:"symbol"`
		Interval            string  `json:"interval"`
		RiskTier            string  `json:"risk_tier"`
		IsActive            bool    `json:"is_active"`
		PausedUntil         *string `json:"paused_until,omitempty"`
		LastSignalBarTime   int64   `json:"last_signal_bar_time"`
		LastSignalDirection string  `json:"last_signal_direction,omitempty"`
		Running             bool    `json:"running"`
	}
	out := []item{}
	for rows.Next() {
		var it item
		var paused *string
		var lastBar *int64
		var lastDir *string
		if err := rows.Scan(&it.ID, &it.Name, &it.Symbol, &it.Interval, &it.RiskTier,
			&it.IsActive, &paused, &lastBar, &lastDir); err != nil {
			continue
		}
		it.PausedUntil = paused
		if lastBar != nil {
			it.LastSignalBarTime = *lastBar
		}
		if lastDir != nil {
			it.LastSignalDirection = *lastDir
		}
		it.Running = runner != nil && runner.IsRunning(it.ID)
		out = append(out, it)
	}
	writeJSON(w, map[string]any{"scenarios": out})
}

// GET /api/v1/alerts?limit=50&strategy_id=...
func handleAlertsList(w http.ResponseWriter, r *http.Request) {
	userID := auth.GetUserID(r)
	if userID == "" {
		http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
		return
	}

	q := r.URL.Query()
	limit := 50
	if s := q.Get("limit"); s != "" {
		var n int
		if _, err := intParse(s, &n); err == nil && n > 0 && n <= 500 {
			limit = n
		}
	}
	strategyID := q.Get("strategy_id")

	sql := `
		SELECT id, strategy_id, symbol, interval, direction, label,
		       COALESCE(confidence, 0), COALESCE(entry_price, 0),
		       COALESCE(stop_loss, 0), COALESCE(take_profit, 0),
		       COALESCE(position_size_usd, 0), bar_time, created_at,
		       telegram_sent_at
		FROM alerts
		WHERE user_id = $1
	`
	args := []any{userID}
	if strategyID != "" {
		sql += " AND strategy_id = $2"
		args = append(args, strategyID)
	}
	sql += " ORDER BY created_at DESC LIMIT " + itoa(limit)

	rows, err := store.Pool.Query(r.Context(), sql, args...)
	if err != nil {
		http.Error(w, `{"error":"db error"}`, http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	type row struct {
		ID             string  `json:"id"`
		StrategyID     string  `json:"strategy_id"`
		Symbol         string  `json:"symbol"`
		Interval       string  `json:"interval"`
		Direction      string  `json:"direction"`
		Label          string  `json:"label,omitempty"`
		Confidence     float64 `json:"confidence"`
		EntryPrice     float64 `json:"entry_price"`
		StopLoss       float64 `json:"stop_loss"`
		TakeProfit     float64 `json:"take_profit"`
		PositionSize   float64 `json:"position_size_usd"`
		BarTime        int64   `json:"bar_time"`
		CreatedAt      string  `json:"created_at"`
		TelegramSentAt *string `json:"telegram_sent_at,omitempty"`
	}
	out := []row{}
	for rows.Next() {
		var it row
		var created string
		var tgSent *string
		var label *string
		if err := rows.Scan(&it.ID, &it.StrategyID, &it.Symbol, &it.Interval,
			&it.Direction, &label, &it.Confidence, &it.EntryPrice, &it.StopLoss,
			&it.TakeProfit, &it.PositionSize, &it.BarTime, &created, &tgSent); err != nil {
			continue
		}
		it.CreatedAt = created
		it.TelegramSentAt = tgSent
		if label != nil {
			it.Label = *label
		}
		out = append(out, it)
	}
	writeJSON(w, map[string]any{"alerts": out})
}

// extractScenarioID pulls id from `/api/v1/scenarios/{id}/start` etc.
func extractScenarioID(path string) string {
	path = strings.TrimPrefix(path, "/api/v1/scenarios/")
	idx := strings.Index(path, "/")
	if idx == -1 {
		return path
	}
	return path[:idx]
}

// itoa is a tiny alloc-free int→string (len<=10) used in query building.
func itoa(n int) string {
	b := make([]byte, 0, 10)
	if n == 0 {
		return "0"
	}
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

func intParse(s string, out *int) (int, error) {
	// JSON-based parse keeps import surface small.
	var n int
	err := json.Unmarshal([]byte(s), &n)
	if err == nil {
		*out = n
	}
	return n, err
}

// Dispatches /api/v1/scenarios/... subroutes based on suffix.
func handleScenarioSubroute(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	switch {
	case strings.HasSuffix(path, "/start") && r.Method == http.MethodPost:
		handleScenarioStart(w, r)
	case strings.HasSuffix(path, "/stop") && r.Method == http.MethodPost:
		handleScenarioStop(w, r)
	case strings.HasSuffix(path, "/active") && r.Method == http.MethodGet:
		handleScenariosActive(w, r)
	default:
		http.NotFound(w, r)
	}
}
