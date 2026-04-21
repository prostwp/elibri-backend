package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/prostwp/elibri-backend/internal/auth"
	"github.com/prostwp/elibri-backend/internal/config"
	"github.com/prostwp/elibri-backend/internal/ml"
	"github.com/prostwp/elibri-backend/internal/store"
	"github.com/prostwp/elibri-backend/internal/ws"
)

var WsHub *ws.Hub

func NewRouter(cfg *config.Config) http.Handler {
	mux := http.NewServeMux()

	// WebSocket hub
	WsHub = ws.NewHub()
	go WsHub.Run()

	// Health + readiness (liveness = /health always 200, readiness = /ready)
	mux.HandleFunc("GET /health", handleHealth)
	mux.HandleFunc("GET /ready", handleReady)

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

	// Scenarios (live runner control) — Patch 3
	mux.HandleFunc("GET /api/v1/scenarios/active", handleScenariosActive)
	mux.HandleFunc("/api/v1/scenarios/", handleScenarioSubroute)

	// Macro calendar (Patch 3A) — upcoming high-impact events + blackout status
	mux.HandleFunc("GET /api/v1/macrocal", handleMacroCalendar)

	// Telegram link + alerts history — Patch 3
	mux.HandleFunc("/api/v1/telegram/link", handleTelegramLinkRouter(cfg))
	mux.HandleFunc("GET /api/v1/alerts", handleAlertsList)

	// Admin (protected + role check inside handler)
	mux.HandleFunc("GET /api/v1/admin/users", handleAdminListUsers)
	mux.HandleFunc("POST /api/v1/admin/users/{id}/reset-password", handleAdminResetPassword)

	// Market data (real)
	mux.HandleFunc("GET /api/v1/market/candles", handleMarketCandlesReal)
	mux.HandleFunc("GET /api/v1/market/quotes", handleMarketQuotesReal)
	mux.HandleFunc("GET /api/v1/market/fundamentals", handleMarketFundamentals)

	// ML (real)
	mux.HandleFunc("POST /api/v1/ml/predict", handleMLPredictV2)
	mux.HandleFunc("POST /api/v1/ml/predict/multi", handleMLPredictMulti)
	mux.HandleFunc("POST /api/v1/ml/predict/legacy", handleMLPredictReal)
	mux.HandleFunc("GET /api/v1/ml/models", handleMLModels)
	mux.HandleFunc("POST /api/v1/ml/train", handleMLTrain)
	mux.HandleFunc("POST /api/v1/ml/reload", handleMLReload)
	mux.HandleFunc("GET /api/v1/ml/backtest", handleMLBacktest)
	mux.HandleFunc("GET /api/v1/ml/backtest/strategy", handleMLStrategyBacktest)
	mux.HandleFunc("POST /api/v1/ml/backtest/run", handleMLRunBacktest)
	mux.HandleFunc("GET /api/v1/ml/paper-trades", handleMLPaperTrades)
	mux.HandleFunc("POST /api/v1/ml/paper-trades/run", handleMLRunPaperTrades)
	mux.HandleFunc("GET /api/v1/ml/paper/status", handleMLPaperStatus)

	// Crypto (real)
	mux.HandleFunc("GET /api/v1/crypto/scan", handleCryptoScanReal)
	mux.HandleFunc("GET /api/v1/crypto/listings", handleCryptoListingsReal)

	// News / Fundamental (Finnhub general+crypto, CoinDesk RSS, Cointelegraph RSS, Alpha Vantage)
	mux.HandleFunc("GET /api/v1/news/fundamental", handleFundamentalNews(cfg))

	// Middleware chain: CORS → Auth (JWT)
	var handler http.Handler = mux
	handler = auth.Middleware(cfg.JWTSecret)(handler)
	handler = bodyLimitMiddleware(handler)
	return corsMiddleware(cfg.CORSOrigins, handler)
}

// bodyLimitMiddleware caps request bodies at 2 MB. Strategies with large
// nodes_json are the only heavy payload (~50 KB typical, 500 KB worst case)
// — 2 MB is ample for that without letting an attacker POST 50 MB JSON
// blobs that would end up in Postgres JSONB columns and OOM the pool.
// Must run BEFORE auth.Middleware so an auth-failed request still has the
// body capped (MaxBytesReader only measures once the body is read).
func bodyLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// GET/HEAD/DELETE/OPTIONS don't carry bodies — skip the wrapper to
		// avoid extra allocations on the hot read path.
		if r.Body != nil && r.Method != http.MethodGet && r.Method != http.MethodHead &&
			r.Method != http.MethodDelete && r.Method != http.MethodOptions {
			r.Body = http.MaxBytesReader(w, r.Body, 2<<20) // 2 MB
		}
		next.ServeHTTP(w, r)
	})
}

// handleHealth is a lightweight liveness probe — answers 200 OK even when
// the DB is down, so systemd/LB can tell "process is alive" vs "process
// crashed." For dependency checks use /ready.
func handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]string{
		"status":  "ok",
		"version": "0.2.0",
		"service": "elibri-backend",
	})
}

// handleReady is the readiness probe — returns 503 when a required
// dependency is missing, so K8s/LB rotates the instance out of rotation
// instead of serving 500s. Lightweight: just checks store.Pool presence
// and a fast `SELECT 1` round-trip.
func handleReady(w http.ResponseWriter, r *http.Request) {
	if store.Pool == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"status":"not_ready","reason":"database pool uninitialized"}`))
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	var n int
	if err := store.Pool.QueryRow(ctx, `SELECT 1`).Scan(&n); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"status":"not_ready","reason":"database ping failed"}`))
		return
	}
	// ML models optional — report as advisory, not blocking.
	writeJSON(w, map[string]any{
		"status":     "ready",
		"ml_loaded":  ml.V2Health().NModels,
		"timestamp":  time.Now().UTC().Format(time.RFC3339),
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
