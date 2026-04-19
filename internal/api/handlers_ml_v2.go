package api

import (
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/prostwp/elibri-backend/internal/market"
	"github.com/prostwp/elibri-backend/internal/ml"
	"github.com/prostwp/elibri-backend/pkg/types"
)

// mlPredictV2Request is the enriched V2 predict payload.
// Backwards-compatible: if v2 fields missing, falls back to legacy behaviour.
type mlPredictV2Request struct {
	Symbol       string              `json:"symbol"`
	Interval     string              `json:"interval"`      // 1h|4h|1d
	Source       string              `json:"source"`        // binance|moex
	TradingStyle string              `json:"trading_style"` // scalp|day|swing|position
	Candles      []types.OHLCVCandle `json:"candles,omitempty"`
}

// handleMLPredictV2 handles POST /api/v1/ml/predict with richer output.
// Supports both V1 (legacy) and V2 (ensemble) via presence of interval/tradingStyle.
func handleMLPredictV2(w http.ResponseWriter, r *http.Request) {
	var req mlPredictV2Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
		return
	}

	if req.Symbol == "" {
		http.Error(w, `{"error":"symbol required"}`, http.StatusBadRequest)
		return
	}
	if req.Interval == "" {
		req.Interval = "4h"
	}
	if req.TradingStyle == "" {
		req.TradingStyle = "swing"
	}
	if req.Source == "" {
		req.Source = "binance"
	}

	// Fetch candles if not provided.
	candles := req.Candles
	if len(candles) == 0 {
		var err error
		if req.Source == "binance" {
			candles, err = market.FetchCryptoCandles(req.Symbol, req.Interval, 500)
		} else {
			candles, err = market.FetchCandles(req.Symbol, 500)
		}
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"error": "failed to fetch candles",
				"detail": err.Error(),
			})
			return
		}
	}
	if len(candles) < 30 {
		http.Error(w, `{"error":"not enough candle data (need 30+)"}`, http.StatusBadRequest)
		return
	}

	// BTC context: fetch when symbol != BTCUSDT for cross-asset features.
	var btcCloses []float64
	if req.Symbol != "BTCUSDT" {
		btc, err := market.FetchCryptoCandles("BTCUSDT", req.Interval, len(candles))
		if err == nil && len(btc) == len(candles) {
			btcCloses = make([]float64, len(btc))
			for i, c := range btc {
				btcCloses[i] = c.Close
			}
		}
	}

	// Taker-buy volumes: not yet wired through market layer → pass nil.
	prediction := ml.PredictV2(req.Symbol, req.Interval, req.TradingStyle, candles, nil, btcCloses)
	writeJSON(w, prediction)
}

// mlPredictMultiResponse bundles predictions for the same symbol across
// several intervals. The MTF "verdict" is the alignment of direction/confidence
// — when all intervals agree we flag it as high-quality.
type mlPredictMultiResponse struct {
	Symbol     string                      `json:"symbol"`
	Primary    string                      `json:"primary_interval"`
	Predictions map[string]*ml.PredictionV2 `json:"predictions"` // interval → prediction
	Consensus  struct {
		Direction    string  `json:"direction"`      // aligned direction or "mixed"
		Alignment    float64 `json:"alignment"`      // 0..1, fraction of TFs agreeing with majority
		HighQuality  bool    `json:"high_quality"`   // true if 100% aligned AND majority has HC
		AvgConfidence float64 `json:"avg_confidence"`
	} `json:"consensus"`
}

type mlPredictMultiRequest struct {
	Symbol       string   `json:"symbol"`
	Intervals    []string `json:"intervals"` // ["1h","4h","1d"]
	TradingStyle string   `json:"trading_style"`
	Source       string   `json:"source"`
}

func handleMLPredictMulti(w http.ResponseWriter, r *http.Request) {
	var req mlPredictMultiRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid request"}`, http.StatusBadRequest)
		return
	}
	if req.Symbol == "" {
		http.Error(w, `{"error":"symbol required"}`, http.StatusBadRequest)
		return
	}
	if len(req.Intervals) == 0 {
		req.Intervals = []string{"1h", "4h", "1d"}
	}
	if req.TradingStyle == "" {
		req.TradingStyle = "swing"
	}
	if req.Source == "" {
		req.Source = "binance"
	}

	// Fetch candles for each requested interval, pass to PredictV2 per-TF.
	// BTC cross-asset context reused across TFs when symbol != BTC.
	predictions := make(map[string]*ml.PredictionV2)
	for _, iv := range req.Intervals {
		candles, err := market.FetchCryptoCandles(req.Symbol, iv, 500)
		if err != nil || len(candles) < 30 {
			predictions[iv] = nil
			continue
		}
		var btcCloses []float64
		if req.Symbol != "BTCUSDT" {
			btc, err := market.FetchCryptoCandles("BTCUSDT", iv, len(candles))
			if err == nil && len(btc) == len(candles) {
				btcCloses = make([]float64, len(btc))
				for i, c := range btc {
					btcCloses[i] = c.Close
				}
			}
		}
		p := ml.PredictV2(req.Symbol, iv, req.TradingStyle, candles, nil, btcCloses)
		predictions[iv] = &p
	}

	// Compute consensus: count directions, check HC alignment.
	dirCount := map[string]int{"buy": 0, "sell": 0, "neutral": 0}
	var sumConf float64
	nValid := 0
	hcAligned := 0
	for _, p := range predictions {
		if p == nil {
			continue
		}
		dirCount[p.Direction]++
		sumConf += p.Confidence
		nValid++
		if p.Metrics.HighConfidence {
			hcAligned++
		}
	}

	resp := mlPredictMultiResponse{
		Symbol:      req.Symbol,
		Primary:     req.Intervals[len(req.Intervals)-1], // highest TF is primary (trend)
		Predictions: predictions,
	}

	if nValid > 0 {
		// Majority direction.
		majority := "neutral"
		maxCount := 0
		for dir, c := range dirCount {
			if c > maxCount {
				maxCount = c
				majority = dir
			}
		}
		alignment := float64(maxCount) / float64(nValid)
		resp.Consensus.Direction = majority
		if alignment < 1.0 && dirCount["buy"] > 0 && dirCount["sell"] > 0 {
			resp.Consensus.Direction = "mixed" // explicit conflict
		}
		resp.Consensus.Alignment = alignment
		resp.Consensus.AvgConfidence = sumConf / float64(nValid)
		// High-quality signal: ALL intervals agree + at least one HC on any TF.
		resp.Consensus.HighQuality = alignment == 1.0 && hcAligned > 0 && majority != "neutral"
	}

	writeJSON(w, resp)
}

// handleMLModels returns metadata about loaded V2 models.
func handleMLModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]any{
		"health":     ml.V2Health(),
		"models":     ml.ListLoadedModels(),
		"thresholds": ml.ListThresholds(),
	})
}

// handleMLPaperStatus returns current paper-trading state for UI.
func handleMLPaperStatus(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]any{
		"mode":      "paper", // TODO wire crypto.Mode()
		"balance":   10000.0,
		"positions": []any{},
	})
}

// handleMLBacktest reads backtest_summary.json + per-strategy files.
func handleMLBacktest(w http.ResponseWriter, r *http.Request) {
	summaryPath := filepath.Join("ml-training", "logs", "backtest_summary.json")
	data, err := os.ReadFile(summaryPath)
	if err != nil {
		writeJSON(w, map[string]any{"error": "no backtest data yet", "hint": "run backtest.py"})
		return
	}
	var summary any
	_ = json.Unmarshal(data, &summary)
	writeJSON(w, summary)
}

// handleMLPaperTrades reads logs/paper_trades.json.
func handleMLPaperTrades(w http.ResponseWriter, r *http.Request) {
	path := filepath.Join("ml-training", "logs", "paper_trades.json")
	data, err := os.ReadFile(path)
	if err != nil {
		writeJSON(w, map[string]any{"error": "no paper-trade data yet", "hint": "run paper_trade.py"})
		return
	}
	var result any
	_ = json.Unmarshal(data, &result)
	writeJSON(w, result)
}

// handleMLRunBacktest triggers backtest.py in background.
func handleMLRunBacktest(w http.ResponseWriter, r *http.Request) {
	go func() {
		trainingDir, _ := filepath.Abs("ml-training")
		cmd := exec.Command("./.venv/bin/python", "-u", "backtest.py")
		cmd.Dir = trainingDir
		_ = cmd.Run()
	}()
	writeJSON(w, map[string]any{
		"status":  "started",
		"message": "Backtest running — poll /ml/backtest in ~2-5 min",
	})
}

// handleMLRunPaperTrades triggers paper_trade.py in background.
func handleMLRunPaperTrades(w http.ResponseWriter, r *http.Request) {
	go func() {
		trainingDir, _ := filepath.Abs("ml-training")
		cmd := exec.Command("./.venv/bin/python", "-u", "paper_trade.py")
		cmd.Dir = trainingDir
		_ = cmd.Run()
	}()
	writeJSON(w, map[string]any{
		"status":  "started",
		"message": "Paper trading running — poll /ml/paper-trades in ~2-5 min",
	})
}

// handleMLStrategyBacktest returns one strategy's detailed backtest (trades + equity curve).
func handleMLStrategyBacktest(w http.ResponseWriter, r *http.Request) {
	symbol := r.URL.Query().Get("symbol")
	interval := r.URL.Query().Get("interval")
	if symbol == "" || interval == "" {
		http.Error(w, `{"error":"symbol and interval required"}`, http.StatusBadRequest)
		return
	}
	path := filepath.Join("ml-training", "logs", "backtest_"+symbol+"_"+interval+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		writeJSON(w, map[string]any{"error": "no backtest for this strategy"})
		return
	}
	var result any
	_ = json.Unmarshal(data, &result)
	writeJSON(w, result)
}

// handleMLTrain triggers a Python training run in-process (admin-only).
// POST body: {"symbol":"BTCUSDT","interval":"4h","quick":true}
// Returns immediately with task ID; actual training happens in a goroutine.
// Caller should poll /api/v1/ml/models afterwards.
type mlTrainRequest struct {
	Symbol   string `json:"symbol"`
	Interval string `json:"interval"`
	Quick    bool   `json:"quick"`
}

func handleMLTrain(w http.ResponseWriter, r *http.Request) {
	var req mlTrainRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
		return
	}
	if req.Symbol == "" || req.Interval == "" {
		http.Error(w, `{"error":"symbol and interval required"}`, http.StatusBadRequest)
		return
	}

	taskID := time.Now().Format("20060102-150405")

	go func(symbol, interval string, quick bool) {
		trainingDir, _ := filepath.Abs("ml-training")
		args := []string{
			"train.py",
			"--symbols", symbol,
			"--intervals", interval,
		}
		if quick {
			args = append(args, "--quick")
		}
		cmd := exec.Command("./.venv/bin/python", args...)
		cmd.Dir = trainingDir
		_ = cmd.Run() // errors logged to stdout by training script
		// Re-load models after training completes.
		_, _ = ml.LoadModelsV2()
		ml.MarkLoaded()
	}(req.Symbol, req.Interval, req.Quick)

	writeJSON(w, map[string]any{
		"task_id": taskID,
		"status":  "started",
		"message": "Training in background — poll GET /api/v1/ml/models after ~5 min",
	})
}

// handleMLReload forces re-scan of the models directory (for dev hot-reload).
func handleMLReload(w http.ResponseWriter, r *http.Request) {
	n, err := ml.LoadModelsV2()
	if err != nil {
		http.Error(w, `{"error":"`+err.Error()+`"}`, http.StatusInternalServerError)
		return
	}
	ml.MarkLoaded()
	writeJSON(w, map[string]any{"loaded": n})
}
