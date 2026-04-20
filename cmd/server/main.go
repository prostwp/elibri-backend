package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/prostwp/elibri-backend/internal/api"
	"github.com/prostwp/elibri-backend/internal/config"
	"github.com/prostwp/elibri-backend/internal/ml"
	"github.com/prostwp/elibri-backend/internal/scenario"
	"github.com/prostwp/elibri-backend/internal/store"
	"github.com/prostwp/elibri-backend/internal/telegram"
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

	// Parent context for live-runner + Telegram bot. Cancelled on SIGINT/SIGTERM.
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// Telegram bot (nil if TELEGRAM_BOT_TOKEN empty — runner still logs alerts in DB).
	tgBot, err := telegram.NewBot(cfg.TelegramBotToken, cfg.TelegramBotUsername, store.Pool)
	if err != nil {
		log.Printf("Telegram disabled: %v", err)
	} else if tgBot != nil {
		log.Printf("Telegram bot initialized (@%s)", cfg.TelegramBotUsername)
	}

	// Alert delivery queue + drain goroutine.
	alertQ := telegram.NewAlertQueue(tgBot, store.Pool)
	go alertQ.Run(ctx)

	// Scenario runner — polls active strategies, emits alerts.
	runner := scenario.NewRunner(ctx, store.Pool, alertQ)
	if err := runner.StartAllActive(ctx); err != nil {
		log.Printf("Scenario runner hydrate error: %v", err)
	}
	api.SetRunner(runner)
	defer runner.Shutdown()

	// Start Telegram long-poll last (it blocks on bot.Start).
	if tgBot != nil {
		go tgBot.Start(ctx)
	}

	router := api.NewRouter(cfg)

	addr := fmt.Sprintf(":%s", cfg.Port)
	log.Printf("Elibri Backend starting on %s", addr)

	server := &http.Server{Addr: addr, Handler: router}
	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("Shutdown signal received, stopping scenario runner and HTTP server...")
	shutdownCtx, shutdownCancel := context.WithCancel(context.Background())
	defer shutdownCancel()
	_ = server.Shutdown(shutdownCtx)
}
