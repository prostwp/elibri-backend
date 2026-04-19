package config

import (
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
	CORSOrigins       string
	Env               string
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
		CORSOrigins:       getEnv("CORS_ORIGINS", "https://prostwp.github.io,http://localhost:5173"),
		Env:           getEnv("GO_ENV", "development"),
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
