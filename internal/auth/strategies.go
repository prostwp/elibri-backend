package auth

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Strategy struct {
	ID            string          `json:"id"`
	UserID        string          `json:"user_id"`
	Name          string          `json:"name"`
	NodesJSON     json.RawMessage `json:"nodes_json"`
	EdgesJSON     json.RawMessage `json:"edges_json"`
	Segment       string          `json:"segment"`
	SelectedPair  string          `json:"selected_pair"`
	CreatedAt     time.Time       `json:"created_at"`
	UpdatedAt     time.Time       `json:"updated_at"`
}

var ErrStrategyNotFound = errors.New("strategy not found")

func ListStrategies(ctx context.Context, pool *pgxpool.Pool, userID string) ([]Strategy, error) {
	rows, err := pool.Query(ctx, `
		SELECT id, user_id, name, nodes_json, edges_json, segment, selected_pair, created_at, updated_at
		FROM strategies WHERE user_id = $1 ORDER BY updated_at DESC
	`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []Strategy{}
	for rows.Next() {
		var s Strategy
		if err := rows.Scan(&s.ID, &s.UserID, &s.Name, &s.NodesJSON, &s.EdgesJSON, &s.Segment, &s.SelectedPair, &s.CreatedAt, &s.UpdatedAt); err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

func GetStrategy(ctx context.Context, pool *pgxpool.Pool, userID, id string) (*Strategy, error) {
	var s Strategy
	err := pool.QueryRow(ctx, `
		SELECT id, user_id, name, nodes_json, edges_json, segment, selected_pair, created_at, updated_at
		FROM strategies WHERE id = $1 AND user_id = $2
	`, id, userID).Scan(&s.ID, &s.UserID, &s.Name, &s.NodesJSON, &s.EdgesJSON, &s.Segment, &s.SelectedPair, &s.CreatedAt, &s.UpdatedAt)
	if err == pgx.ErrNoRows {
		return nil, ErrStrategyNotFound
	}
	if err != nil {
		return nil, err
	}
	return &s, nil
}

func CreateStrategy(ctx context.Context, pool *pgxpool.Pool, s *Strategy) error {
	if len(s.NodesJSON) == 0 {
		s.NodesJSON = json.RawMessage("[]")
	}
	if len(s.EdgesJSON) == 0 {
		s.EdgesJSON = json.RawMessage("[]")
	}
	return pool.QueryRow(ctx, `
		INSERT INTO strategies (user_id, name, nodes_json, edges_json, segment, selected_pair)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING id, created_at, updated_at
	`, s.UserID, s.Name, s.NodesJSON, s.EdgesJSON, s.Segment, s.SelectedPair).Scan(
		&s.ID, &s.CreatedAt, &s.UpdatedAt,
	)
}

func UpdateStrategy(ctx context.Context, pool *pgxpool.Pool, s *Strategy) error {
	tag, err := pool.Exec(ctx, `
		UPDATE strategies
		SET name = $1, nodes_json = $2, edges_json = $3, segment = $4, selected_pair = $5, updated_at = NOW()
		WHERE id = $6 AND user_id = $7
	`, s.Name, s.NodesJSON, s.EdgesJSON, s.Segment, s.SelectedPair, s.ID, s.UserID)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrStrategyNotFound
	}
	return nil
}

func DeleteStrategy(ctx context.Context, pool *pgxpool.Pool, userID, id string) error {
	tag, err := pool.Exec(ctx, `DELETE FROM strategies WHERE id = $1 AND user_id = $2`, id, userID)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrStrategyNotFound
	}
	return nil
}
