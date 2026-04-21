package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

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
	// Pass AlertsMaxPerDayPerUser so deliver() enforces per-user quota.
	alertQ := telegram.NewAlertQueue(tgBot, store.Pool, cfg.AlertsMaxPerDayPerUser)
	go alertQ.Run(ctx)

	// Scenario runner — polls active strategies, emits alerts.
	runner := scenario.NewRunner(ctx, store.Pool, alertQ)
	if err := runner.StartAllActive(ctx); err != nil {
		log.Printf("Scenario runner hydrate error: %v", err)
	}
	api.SetRunner(runner)
	api.SetScenarioConfig(cfg) // wire quotas + kill-switch into handlers

	// Start Telegram long-poll last (it blocks on bot.Start).
	if tgBot != nil {
		go tgBot.Start(ctx)
	}

	router := api.NewRouter(cfg)

	addr := fmt.Sprintf(":%s", cfg.Port)
	log.Printf("Elibri Backend starting on %s", addr)

	// Patch 2L: HTTP timeouts to kill Slowloris / stuck clients.
	// Without these, one idle TCP connection holds a goroutine forever.
	server := &http.Server{
		Addr:              addr,
		Handler:           router,
		ReadHeaderTimeout: 10 * time.Second, // attacker can't dribble headers
		ReadTimeout:       30 * time.Second, // body must arrive in 30s
		WriteTimeout:      60 * time.Second, // slow ML predicts can take ~5-10s; 60s headroom
		IdleTimeout:       120 * time.Second,
	}
	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("Shutdown signal received — draining runner + alert queue + HTTP server...")

	// PHASE 1 fix #2: explicit shutdown ordering so in-flight work completes.
	//   1) HTTP server stops accepting new requests (existing ones get 20s to finish).
	//   2) Scenario runner stops ticking (goroutines exit on parent ctx.Done).
	//   3) Alert queue drains any buffered alerts to Telegram + DB.
	//   4) Store pool closes (via deferred ClosePostgres).
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer shutdownCancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("HTTP server shutdown error: %v", err)
	}

	runner.Shutdown()

	// Block until AlertQueue.Run returns (buffered alerts flushed).
	select {
	case <-alertQ.Done():
		log.Println("Shutdown complete")
	case <-time.After(15 * time.Second):
		log.Println("Shutdown hit 15s deadline — some alerts may be in DB but not delivered")
	}
}
