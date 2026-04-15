package ml

import (
	"sync"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

var (
	models   = make(map[string]*GBDTModel) // symbol → model
	modelsMu sync.RWMutex
)

// PredictSymbol generates ML prediction for a symbol
func PredictSymbol(symbol string, candles []types.OHLCVCandle) types.MLPrediction {
	features := ExtractFeatures(candles)

	modelsMu.RLock()
	model, ok := models[symbol]
	modelsMu.RUnlock()

	if !ok {
		// Train on-the-fly if no model exists
		model = Train(candles, 5, 50, 0.1)
		modelsMu.Lock()
		models[symbol] = model
		modelsMu.Unlock()
	}

	prob := model.Predict(features)

	direction := "neutral"
	if prob > 0.6 {
		direction = "buy"
	} else if prob < 0.4 {
		direction = "sell"
	}

	confidence := prob * 100
	if direction == "sell" {
		confidence = (1 - prob) * 100
	}
	if direction == "neutral" {
		confidence = 50
	}

	// Price target based on ATR
	lastPrice := candles[len(candles)-1].Close
	atr := calcATR(candles, 14)
	priceTarget := lastPrice
	if direction == "buy" {
		priceTarget = lastPrice + atr*2
	} else if direction == "sell" {
		priceTarget = lastPrice - atr*2
	}

	return types.MLPrediction{
		Symbol:       symbol,
		Direction:    direction,
		Confidence:   confidence,
		PriceTarget:  priceTarget,
		Timeframe:    "5D",
		ModelVersion: "gbdt_v" + itoa(model.Version),
		PredictedAt:  time.Now().Unix(),
	}
}

// TrainModel explicitly trains a model for a symbol
func TrainModel(symbol string, candles []types.OHLCVCandle) *GBDTModel {
	model := Train(candles, 5, 100, 0.1)
	model.Version = int(time.Now().Unix())

	modelsMu.Lock()
	models[symbol] = model
	modelsMu.Unlock()

	return model
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	s := ""
	for n > 0 {
		s = string(rune('0'+n%10)) + s
		n /= 10
	}
	return s
}
