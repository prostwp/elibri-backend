package api

import (
	"log"
	"net/http"
	"strconv"
	"sync/atomic"

	"github.com/prostwp/elibri-backend/internal/config"
	"github.com/prostwp/elibri-backend/internal/news"
)

// disabledLogged guards a one-shot log line so we don't spam every request when
// FINNHUB_API_KEY is unset. Mirrors the pattern in internal/macrocal/calendar.go
// which logs exactly once on startup. Here we log on first request because
// config isn't read during startup wiring — avoids log noise on healthy deploys.
var newsDisabledLogged atomic.Bool

// handleFundamentalNews aggregates news from Finnhub + RSS + Alpha Vantage + LunarCrush social.
//
// Patch 2N+1 (H2 fundamental recon): warn once when FINNHUB_API_KEY is empty
// so operators can diagnose "no news" without reading code. Without this guard
// the endpoint silently returned an empty aggregate and the CryptoFundamental
// node showed "No news matching filter" with no clue why.
func handleFundamentalNews(cfg *config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if cfg.FinnhubAPIKey == "" && !newsDisabledLogged.Swap(true) {
			log.Printf("[news] FINNHUB_API_KEY not set — /news/fundamental will return mostly empty aggregates (RSS-only). Set it in env to enable Finnhub sources.")
		}

		symbol := r.URL.Query().Get("symbol")
		hoursStr := r.URL.Query().Get("hours")
		hours := 24
		if v, err := strconv.Atoi(hoursStr); err == nil && v > 0 && v <= 168 {
			hours = v
		}

		agg, err := news.Fetch(
			r.Context(),
			cfg.FinnhubAPIKey,
			cfg.AlphaVantageAPIKey,
			cfg.LunarCrushAPIKey,
			symbol,
			hours,
		)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "failed to fetch news")
			return
		}
		writeJSON(w, agg)
	}
}
