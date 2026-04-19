package api

import (
	"net/http"
	"strconv"

	"github.com/prostwp/elibri-backend/internal/config"
	"github.com/prostwp/elibri-backend/internal/news"
)

// handleFundamentalNews aggregates news from Finnhub + RSS + Alpha Vantage + LunarCrush social.
func handleFundamentalNews(cfg *config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
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
