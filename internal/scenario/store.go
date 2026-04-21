package scenario

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// ActiveScenario is the subset of the strategies table needed by the runner:
// everything required to tick, evaluate, and log an alert.
type ActiveScenario struct {
	ID                  string
	UserID              string
	Name                string
	Symbol              string // selected_pair
	Interval            string
	RiskTier            string
	NodesJSON           json.RawMessage
	EdgesJSON           json.RawMessage
	TelegramEnabled     bool
	AutoExecute         bool
	LastSignalBarTime   int64  // unix seconds; 0 if never fired
	LastSignalDirection string // "buy" | "sell" | ""
	PausedUntil         *time.Time
}

// ListActiveScenarios returns every scenario flagged is_active=true and not
// currently paused. Called once on startup + whenever the runner needs to
// resync (e.g. after a user flips a switch via /scenarios/{id}/start).
func ListActiveScenarios(ctx context.Context, pool *pgxpool.Pool) ([]ActiveScenario, error) {
	rows, err := pool.Query(ctx, `
		SELECT id, user_id, name, selected_pair, interval, risk_tier,
		       nodes_json, edges_json, telegram_enabled, auto_execute,
		       COALESCE(last_signal_bar_time, 0), COALESCE(last_signal_direction, ''),
		       paused_until
		FROM strategies
		WHERE is_active = true
		  AND (paused_until IS NULL OR paused_until < NOW())
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []ActiveScenario{}
	for rows.Next() {
		var s ActiveScenario
		if err := rows.Scan(
			&s.ID, &s.UserID, &s.Name, &s.Symbol, &s.Interval, &s.RiskTier,
			&s.NodesJSON, &s.EdgesJSON, &s.TelegramEnabled, &s.AutoExecute,
			&s.LastSignalBarTime, &s.LastSignalDirection, &s.PausedUntil,
		); err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

// MarkLastSignal updates last_signal_bar_time / last_signal_direction so the
// runner skips the same bar on the next tick. Cheap UPDATE — called once per
// fired signal, not every tick.
func MarkLastSignal(ctx context.Context, pool *pgxpool.Pool, scenarioID string, barTime int64, direction string) error {
	_, err := pool.Exec(ctx, `
		UPDATE strategies
		SET last_signal_bar_time = $2, last_signal_direction = $3, updated_at = NOW()
		WHERE id = $1
	`, scenarioID, barTime, direction)
	return err
}

// Alert is a pending delivery unit: runner → Telegram queue.
type Alert struct {
	ID               string
	UserID           string
	StrategyID       string
	Symbol           string
	Interval         string
	Direction        string // "buy" | "sell"
	Label            string // "trend_aligned" | "mean_reversion" | "random"
	Confidence       float64
	EntryPrice       float64
	StopLoss         float64
	TakeProfit       float64
	PositionSizeUSD  float64
	BarTime          int64
	CreatedAt        time.Time
	TelegramChatID   int64 // 0 if user didn't link TG
	Meta             json.RawMessage
}

// InsertAlert writes the alert row with ON CONFLICT DO NOTHING on the dedup
// index — guarantees at-most-once per (strategy, bar, direction) even across
// server restarts mid-tick. Returns (inserted: bool, id: string).
func InsertAlert(ctx context.Context, pool *pgxpool.Pool, a *Alert) (bool, error) {
	var id string
	err := pool.QueryRow(ctx, `
		INSERT INTO alerts (
			user_id, strategy_id, symbol, interval, direction, label,
			confidence, entry_price, stop_loss, take_profit, position_size_usd,
			bar_time, meta
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
		ON CONFLICT (strategy_id, bar_time, direction) DO NOTHING
		RETURNING id
	`,
		a.UserID, a.StrategyID, a.Symbol, a.Interval, a.Direction, a.Label,
		a.Confidence, a.EntryPrice, a.StopLoss, a.TakeProfit, a.PositionSizeUSD,
		a.BarTime, a.Meta,
	).Scan(&id)
	if err != nil {
		// pgx returns pgx.ErrNoRows on ON CONFLICT DO NOTHING + RETURNING
		// when the dedup index blocked insertion. Treat as non-error.
		// PHASE 1 fix: use errors.Is, not string compare — future library
		// upgrades could change the message silently.
		if errors.Is(err, pgx.ErrNoRows) {
			return false, nil
		}
		return false, err
	}
	a.ID = id
	return true, nil
}

// MarkAlertSent records Telegram delivery success.
func MarkAlertSent(ctx context.Context, pool *pgxpool.Pool, alertID string, tgMessageID int64) error {
	_, err := pool.Exec(ctx, `
		UPDATE alerts
		SET telegram_sent_at = NOW(), telegram_message_id = $2
		WHERE id = $1
	`, alertID, tgMessageID)
	return err
}

// CountAlertsLast24h counts how many alerts a user has received in the last
// 24h — used to enforce AlertsMaxPerDayPerUser quota.
func CountAlertsLast24h(ctx context.Context, pool *pgxpool.Pool, userID string) (int, error) {
	var n int
	err := pool.QueryRow(ctx, `
		SELECT COUNT(*) FROM alerts
		WHERE user_id = $1 AND created_at > NOW() - INTERVAL '24 hours'
	`, userID).Scan(&n)
	return n, err
}

// GetUserTelegramChatID returns (chat_id, linked). chat_id is 0 when user
// hasn't linked Telegram yet.
func GetUserTelegramChatID(ctx context.Context, pool *pgxpool.Pool, userID string) (int64, bool, error) {
	var chatID *int64
	err := pool.QueryRow(ctx, `
		SELECT telegram_chat_id FROM users WHERE id = $1
	`, userID).Scan(&chatID)
	if err != nil {
		return 0, false, err
	}
	if chatID == nil {
		return 0, false, nil
	}
	return *chatID, true, nil
}
