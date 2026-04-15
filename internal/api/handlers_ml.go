package api

import (
	"encoding/json"
	"net/http"

	"github.com/prostwp/elibri-backend/internal/market"
	"github.com/prostwp/elibri-backend/internal/ml"
	"github.com/prostwp/elibri-backend/pkg/types"
)

type mlPredictRequest struct {
	Symbol  string              `json:"symbol"`
	Source  string              `json:"source"` // "moex" or "binance"
	Candles []types.OHLCVCandle `json:"candles,omitempty"`
}

func handleMLPredictReal(w http.ResponseWriter, r *http.Request) {
	var req mlPredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
		return
	}

	if req.Symbol == "" {
		http.Error(w, `{"error":"symbol required"}`, http.StatusBadRequest)
		return
	}

	// Fetch candles if not provided
	candles := req.Candles
	if len(candles) == 0 {
		var err error
		if req.Source == "binance" {
			candles, err = market.FetchCryptoCandles(req.Symbol, "1h", 200)
		} else {
			candles, err = market.FetchCandles(req.Symbol, 200)
		}
		if err != nil {
			http.Error(w, `{"error":"failed to fetch candles: `+err.Error()+`"}`, http.StatusInternalServerError)
			return
		}
	}

	if len(candles) < 30 {
		http.Error(w, `{"error":"not enough candle data (need 30+)"}`, http.StatusBadRequest)
		return
	}

	prediction := ml.PredictSymbol(req.Symbol, candles)

	// Also return features for transparency
	features := ml.ExtractFeatures(candles)

	writeJSON(w, map[string]interface{}{
		"prediction": prediction,
		"features":   features,
	})
}
