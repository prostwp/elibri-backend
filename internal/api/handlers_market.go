package api

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/prostwp/elibri-backend/internal/market"
)

func handleMarketCandlesReal(w http.ResponseWriter, r *http.Request) {
	symbol := r.URL.Query().Get("symbol")
	source := r.URL.Query().Get("source") // "moex" or "binance"
	interval := r.URL.Query().Get("interval")
	limitStr := r.URL.Query().Get("limit")

	if symbol == "" {
		http.Error(w, `{"error":"symbol required"}`, http.StatusBadRequest)
		return
	}

	limit := 200
	if limitStr != "" {
		if n, err := strconv.Atoi(limitStr); err == nil {
			limit = n
		}
	}

	if source == "binance" {
		if interval == "" {
			interval = "1d"
		}
		candles, err := market.FetchCryptoCandles(symbol, interval, limit)
		if err != nil {
			http.Error(w, `{"error":"`+err.Error()+`"}`, http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]interface{}{
			"candles": candles,
			"source":  "binance",
			"symbol":  symbol,
		})
		return
	}

	// Default: MOEX
	candles, err := market.FetchCandles(symbol, limit)
	if err != nil {
		http.Error(w, `{"error":"`+err.Error()+`"}`, http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]interface{}{
		"candles": candles,
		"source":  "moex",
		"symbol":  symbol,
	})
}

func handleMarketQuotesReal(w http.ResponseWriter, r *http.Request) {
	symbolsStr := r.URL.Query().Get("symbols")
	source := r.URL.Query().Get("source")

	if symbolsStr == "" {
		http.Error(w, `{"error":"symbols required"}`, http.StatusBadRequest)
		return
	}

	symbols := strings.Split(symbolsStr, ",")

	if source == "binance" {
		quotes, err := market.FetchCryptoQuotes(symbols)
		if err != nil {
			http.Error(w, `{"error":"`+err.Error()+`"}`, http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]interface{}{
			"quotes": quotes,
			"source": "binance",
		})
		return
	}

	// Default: MOEX
	quotes, err := market.FetchQuotes(symbols)
	if err != nil {
		http.Error(w, `{"error":"`+err.Error()+`"}`, http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]interface{}{
		"quotes": quotes,
		"source": "moex",
	})
}
