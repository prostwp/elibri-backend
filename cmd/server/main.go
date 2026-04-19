package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/prostwp/elibri-backend/internal/api"
	"github.com/prostwp/elibri-backend/internal/config"
	"github.com/prostwp/elibri-backend/internal/ml"
	"github.com/prostwp/elibri-backend/internal/store"
)

func main() {
	cfg := config.Load()

	// Initialize stores (non-fatal if unavailable)
	store.InitPostgres(cfg.DatabaseURL)
	defer store.ClosePostgres()

	store.InitRedis(cfg.RedisURL)
	defer store.CloseRedis()

	// Load ML v2 models from ml-training/models/ (Python training output).
	// Non-fatal if missing — predict_v2 falls back to legacy v1.
	ml.SetModelsDir("ml-training/models")
	if n, err := ml.LoadModelsV2(); err != nil {
		log.Printf("ML v2 models: %v (continuing with fallback)", err)
	} else {
		log.Printf("ML v2: %d models loaded", n)
		ml.MarkLoaded()
	}

	// Load per-model HC thresholds tuned by analyze_thresholds.py.
	if n, err := ml.LoadThresholds("ml-training"); err != nil {
		log.Printf("ML thresholds: %v (using default 0.80/0.20)", err)
	} else {
		log.Printf("ML thresholds: %d loaded", n)
	}

	router := api.NewRouter(cfg)

	addr := fmt.Sprintf(":%s", cfg.Port)
	log.Printf("Elibri Backend starting on %s", addr)

	if err := http.ListenAndServe(addr, router); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
