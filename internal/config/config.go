package config

import "os"

type Config struct {
	Port              string
	DatabaseURL       string
	RedisURL          string
	SupabaseJWTSecret string
	SupabaseURL       string
	BinanceAPIKey     string
	FinnhubAPIKey     string
	CORSOrigins       string
}

func Load() *Config {
	return &Config{
		Port:              getEnv("PORT", "8080"),
		DatabaseURL:       getEnv("DATABASE_URL", "postgresql://elibri:elibri@localhost:5432/elibri?sslmode=disable"),
		RedisURL:          getEnv("REDIS_URL", "redis://localhost:6379/0"),
		SupabaseJWTSecret: getEnv("SUPABASE_JWT_SECRET", ""),
		SupabaseURL:       getEnv("SUPABASE_URL", ""),
		BinanceAPIKey:     getEnv("BINANCE_API_KEY", ""),
		FinnhubAPIKey:     getEnv("FINNHUB_API_KEY", ""),
		CORSOrigins:       getEnv("CORS_ORIGINS", "https://prostwp.github.io,http://localhost:5173"),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
