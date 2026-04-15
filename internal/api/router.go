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

	// Market data (real)
	mux.HandleFunc("GET /api/v1/market/candles", handleMarketCandlesReal)
	mux.HandleFunc("GET /api/v1/market/quotes", handleMarketQuotesReal)
	mux.HandleFunc("GET /api/v1/market/fundamentals", handleMarketFundamentals)

	// ML (real)
	mux.HandleFunc("POST /api/v1/ml/predict", handleMLPredictReal)

	// Crypto (real)
	mux.HandleFunc("GET /api/v1/crypto/scan", handleCryptoScanReal)
	mux.HandleFunc("GET /api/v1/crypto/listings", handleCryptoListingsReal)

	// Wrap with middleware chain: CORS → Auth
	var handler http.Handler = mux
	if cfg.SupabaseJWTSecret != "" {
		handler = auth.Middleware(cfg.SupabaseJWTSecret)(handler)
	}
	return corsMiddleware(cfg.CORSOrigins, handler)
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]string{
		"status":  "ok",
		"version": "0.1.0",
		"service": "elibri-backend",
	})
}

// Placeholder handlers — will be implemented in subsequent phases
func handleMarketCandles(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]string{"status": "not_implemented"})
}

func handleMarketQuotes(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]string{"status": "not_implemented"})
}

func handleMarketFundamentals(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]string{"status": "not_implemented"})
}

func handleMLPredict(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]string{"status": "not_implemented"})
}

func handleCryptoScan(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]string{"status": "not_implemented"})
}

func handleCryptoListings(w http.ResponseWriter, r *http.Request) {
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
