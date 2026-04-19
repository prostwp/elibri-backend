package api

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/prostwp/elibri-backend/internal/auth"
	"github.com/prostwp/elibri-backend/internal/config"
	"github.com/prostwp/elibri-backend/internal/ws"
)

var WsHub *ws.Hub

func NewRouter(cfg *config.Config) http.Handler {
	mux := http.NewServeMux()

	// WebSocket hub
	WsHub = ws.NewHub()
	go WsHub.Run()

	// Health
	mux.HandleFunc("GET /health", handleHealth)

	// WebSocket
	mux.HandleFunc("/ws", WsHub.HandleWS)

	// Auth (public: register/login, protected: me)
	mux.HandleFunc("POST /api/v1/auth/register", handleRegister(cfg))
	mux.HandleFunc("POST /api/v1/auth/login", handleLogin(cfg))
	mux.HandleFunc("GET /api/v1/auth/me", handleMe)
	mux.HandleFunc("PATCH /api/v1/auth/me", handleUpdateMe)

	// Strategies (protected)
	mux.HandleFunc("GET /api/v1/strategies", handleStrategiesList)
	mux.HandleFunc("POST /api/v1/strategies", handleStrategiesCreate)
	mux.HandleFunc("/api/v1/strategies/", handleStrategyByID)

	// Admin (protected + role check inside handler)
	mux.HandleFunc("GET /api/v1/admin/users", handleAdminListUsers)
	mux.HandleFunc("POST /api/v1/admin/users/{id}/reset-password", handleAdminResetPassword)

	// Market data (real)
	mux.HandleFunc("GET /api/v1/market/candles", handleMarketCandlesReal)
	mux.HandleFunc("GET /api/v1/market/quotes", handleMarketQuotesReal)
	mux.HandleFunc("GET /api/v1/market/fundamentals", handleMarketFundamentals)

	// ML (real)
	mux.HandleFunc("POST /api/v1/ml/predict", handleMLPredictV2)
	mux.HandleFunc("POST /api/v1/ml/predict/legacy", handleMLPredictReal)
	mux.HandleFunc("GET /api/v1/ml/models", handleMLModels)
	mux.HandleFunc("POST /api/v1/ml/train", handleMLTrain)
	mux.HandleFunc("POST /api/v1/ml/reload", handleMLReload)
	mux.HandleFunc("GET /api/v1/ml/backtest", handleMLBacktest)
	mux.HandleFunc("GET /api/v1/ml/backtest/strategy", handleMLStrategyBacktest)
	mux.HandleFunc("GET /api/v1/ml/paper-trades", handleMLPaperTrades)
	mux.HandleFunc("GET /api/v1/ml/paper/status", handleMLPaperStatus)

	// Crypto (real)
	mux.HandleFunc("GET /api/v1/crypto/scan", handleCryptoScanReal)
	mux.HandleFunc("GET /api/v1/crypto/listings", handleCryptoListingsReal)

	// News / Fundamental (Finnhub general+crypto, CoinDesk RSS, Cointelegraph RSS, Alpha Vantage)
	mux.HandleFunc("GET /api/v1/news/fundamental", handleFundamentalNews(cfg))

	// Middleware chain: CORS → Auth (JWT)
	var handler http.Handler = mux
	handler = auth.Middleware(cfg.JWTSecret)(handler)
	return corsMiddleware(cfg.CORSOrigins, handler)
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]string{
		"status":  "ok",
		"version": "0.2.0",
		"service": "elibri-backend",
	})
}

func handleMarketFundamentals(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]string{"status": "not_implemented"})
}

// --- Helpers ---

func writeJSON(w http.ResponseWriter, data any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

func corsMiddleware(origins string, next http.Handler) http.Handler {
	allowedOrigins := strings.Split(origins, ",")

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		for _, allowed := range allowedOrigins {
			if strings.TrimSpace(allowed) == origin {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				break
			}
		}
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Max-Age", "86400")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}
