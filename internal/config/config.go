package config

import (
	"fmt"
	"log"
	"os"
)

const devJWTSecret = "dev-secret-change-in-production-please-32-bytes-min"

type Config struct {
	Port               string
	DatabaseURL        string
	RedisURL           string
	JWTSecret          string
	AdminEmails        string
	BinanceAPIKey      string
	FinnhubAPIKey      string
	AlphaVantageAPIKey string
	LunarCrushAPIKey   string // crypto social (Twitter/Reddit KOL posts)
	CORSOrigins        string
	Env                string

	// Patch 3 — live scenario runner + Telegram alerts.
	TelegramBotToken       string
	TelegramBotUsername    string
	ScenarioMaxPerUser     int
	AlertsMaxPerDayPerUser int
	ScenarioKillSwitch     bool

	// Patch 3A — macro blackout gate. Blocks scenario alerts around
	// high-impact USD releases (FOMC, CPI, NFP, PCE, Fed speakers) where
	// volatility expands and the ML edge collapses.
	MacroBlackoutEnabled bool
	MacroBlackoutBefore  int    // minutes before event to start blocking
	MacroBlackoutAfter   int    // minutes after event to keep blocking
	MacroImpactFilter    string // "low" | "medium" | "high" — min impact to block

	// Sub-chat C — Resend transactional email (verify / password reset).
	// Secrets come from env only; empty ResendAPIKey switches the email
	// util into dev mode (log link to stdout, never call the API).
	ResendAPIKey string // RESEND_API_KEY — empty => dev mode (links logged)
	AppBaseURL   string // APP_BASE_URL — frontend origin incl. Vite base (local: /elibri-fx)
	ResendFrom   string // RESEND_FROM — verified sender address (SPF/DKIM via DevOps)
}

func Load() *Config {
	cfg := &Config{
		Port:               getEnv("PORT", "8080"),
		DatabaseURL:        getEnv("DATABASE_URL", "postgresql://elibri:elibri@localhost:5432/elibri?sslmode=disable"),
		RedisURL:           getEnv("REDIS_URL", "redis://localhost:6379/0"),
		JWTSecret:          getEnv("JWT_SECRET", devJWTSecret),
		AdminEmails:        getEnv("ADMIN_EMAILS", ""),
		BinanceAPIKey:      getEnv("BINANCE_API_KEY", ""),
		FinnhubAPIKey:      getEnv("FINNHUB_API_KEY", ""),
		AlphaVantageAPIKey: getEnv("ALPHA_VANTAGE_API_KEY", ""),
		LunarCrushAPIKey:   getEnv("LUNARCRUSH_API_KEY", ""),
		CORSOrigins:        getEnv("CORS_ORIGINS", "https://prostwp.github.io,http://localhost:5173"),
		Env:                getEnv("GO_ENV", "development"),

		TelegramBotToken:       getEnv("TELEGRAM_BOT_TOKEN", ""),
		TelegramBotUsername:    getEnv("TELEGRAM_BOT_USERNAME", ""),
		ScenarioMaxPerUser:     getEnvInt("SCENARIO_MAX_PER_USER", 5),
		AlertsMaxPerDayPerUser: getEnvInt("ALERTS_MAX_PER_DAY", 100),
		ScenarioKillSwitch:     getEnv("SCENARIO_KILL_SWITCH", "0") == "1",

		// Macro blackout gate — ON by default so paper trading respects
		// high-impact USD releases without user action.
		MacroBlackoutEnabled: getEnv("MACRO_BLACKOUT_ENABLED", "1") == "1",
		MacroBlackoutBefore:  getEnvInt("MACRO_BLACKOUT_MINUTES_BEFORE", 30),
		MacroBlackoutAfter:   getEnvInt("MACRO_BLACKOUT_MINUTES_AFTER", 15),
		MacroImpactFilter:    getEnv("MACRO_IMPACT_FILTER", "high"),

		ResendAPIKey: getEnv("RESEND_API_KEY", ""),
		// Default matches the current local Vite base (vite.config.ts: /elibri-fx/).
		// PROD must set APP_BASE_URL to the deployed origin+base so verify/reset
		// email links resolve (else they 404). Canonical prod base TBD — see dnevnik.
		AppBaseURL:   getEnv("APP_BASE_URL", "http://localhost:5173/elibri-fx"),
		ResendFrom:   getEnv("RESEND_FROM", "no-reply@alphavizor.com"),
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
