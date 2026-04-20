package config

import (
	"fmt"
	"log"
	"os"
)

const devJWTSecret = "dev-secret-change-in-production-please-32-bytes-min"

type Config struct {
	Port              string
	DatabaseURL       string
	RedisURL          string
	JWTSecret         string
	AdminEmails       string
	BinanceAPIKey     string
	FinnhubAPIKey     string
	AlphaVantageAPIKey string
	LunarCrushAPIKey  string // crypto social (Twitter/Reddit KOL posts)
	CORSOrigins       string
	Env               string

	// Patch 3 — live scenario runner + Telegram alerts.
	TelegramBotToken       string
	TelegramBotUsername    string
	ScenarioMaxPerUser     int
	AlertsMaxPerDayPerUser int
	ScenarioKillSwitch     bool
}

func Load() *Config {
	cfg := &Config{
		Port:          getEnv("PORT", "8080"),
		DatabaseURL:   getEnv("DATABASE_URL", "postgresql://elibri:elibri@localhost:5432/elibri?sslmode=disable"),
		RedisURL:      getEnv("REDIS_URL", "redis://localhost:6379/0"),
		JWTSecret:     getEnv("JWT_SECRET", devJWTSecret),
		AdminEmails:   getEnv("ADMIN_EMAILS", ""),
		BinanceAPIKey:     getEnv("BINANCE_API_KEY", ""),
		FinnhubAPIKey:     getEnv("FINNHUB_API_KEY", ""),
		AlphaVantageAPIKey: getEnv("ALPHA_VANTAGE_API_KEY", ""),
		LunarCrushAPIKey:  getEnv("LUNARCRUSH_API_KEY", ""),
		CORSOrigins:       getEnv("CORS_ORIGINS", "https://prostwp.github.io,http://localhost:5173"),
		Env:           getEnv("GO_ENV", "development"),

		TelegramBotToken:       getEnv("TELEGRAM_BOT_TOKEN", ""),
		TelegramBotUsername:    getEnv("TELEGRAM_BOT_USERNAME", ""),
		ScenarioMaxPerUser:     getEnvInt("SCENARIO_MAX_PER_USER", 5),
		AlertsMaxPerDayPerUser: getEnvInt("ALERTS_MAX_PER_DAY", 100),
		ScenarioKillSwitch:     getEnv("SCENARIO_KILL_SWITCH", "0") == "1",
	}
	// Refuse to start in production with the dev secret.
	if cfg.JWTSecret == devJWTSecret && cfg.Env != "development" {
		log.Fatal("FATAL: JWT_SECRET is the dev default in non-development env. Set JWT_SECRET to a strong random string (min 32 bytes).")
	}
	if cfg.JWTSecret == devJWTSecret {
		log.Println("WARNING: using dev JWT_SECRET — set JWT_SECRET env var before going to production")
	}
	return cfg
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		var n int
		if _, err := fmt.Sscanf(v, "%d", &n); err == nil {
			return n
		}
	}
	return fallback
}
