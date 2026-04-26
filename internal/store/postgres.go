package store

import (
	"context"
	"log"
	"os"

	"github.com/jackc/pgx/v5/pgxpool"
)

var Pool *pgxpool.Pool

func InitPostgres(databaseURL string) {
	var err error
	Pool, err = pgxpool.New(context.Background(), databaseURL)
	if err != nil {
		log.Printf("WARNING: PostgreSQL not available: %v", err)
		return
	}

	if err := Pool.Ping(context.Background()); err != nil {
		log.Printf("WARNING: PostgreSQL ping failed: %v", err)
		Pool = nil
		return
	}

	log.Println("PostgreSQL connected")

	// Run migrations
	runMigrations()
}

func runMigrations() {
	if Pool == nil {
		return
	}

	files := []string{
		"internal/store/migrations/001_market_data.sql",
		"internal/store/migrations/002_ml_signals.sql",
		"internal/store/migrations/003_users_and_strategies.sql",
		"internal/store/migrations/004_risk_tier.sql",
		"internal/store/migrations/005_scenarios_and_telegram.sql",
		"internal/store/migrations/006_strategies_risk_tier_check.sql",
		"internal/store/migrations/007_strategies_interval_check_and_alerts_indexes.sql",
		"internal/store/migrations/008_strategies_authors.sql",
	}

	for _, f := range files {
		sql, err := os.ReadFile(f)
		if err != nil {
			log.Printf("Migration file not found: %s", f)
			continue
		}

		_, err = Pool.Exec(context.Background(), string(sql))
		if err != nil {
			log.Printf("Migration error (%s): %v", f, err)
		} else {
			log.Printf("Migration applied: %s", f)
		}
	}
}

func ClosePostgres() {
	if Pool != nil {
		Pool.Close()
	}
}
