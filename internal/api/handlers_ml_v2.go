package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"time"

	"github.com/prostwp/elibri-backend/internal/auth"
	"github.com/prostwp/elibri-backend/internal/market"
	"github.com/prostwp/elibri-backend/internal/ml"
	"github.com/prostwp/elibri-backend/internal/store"
	"github.com/prostwp/elibri-backend/pkg/types"
)

// Only 1 concurrent Python-backed training/backtest job at a time.
// Protects against /ml/train spam DoS (each run spawns XGBoost that eats all
// cores + 600MB RAM for 30+ minutes).
var mlJobSlot = make(chan struct{}, 1)

// Whitelist symbol/interval so user-supplied values can't escape exec args
// (path traversal, flag injection via train.py --eval).
var (
	symbolPattern   = regexp.MustCompile(`^[A-Z0-9]{2,20}$`)
	intervalPattern = regexp.MustCompile(`^(5m|15m|1h|4h|1d)$`)
)

func validateSymbol(s string) error {
	if !symbolPattern.MatchString(s) {
		return fmt.Errorf("invalid symbol %q (must match %s)", s, symbolPattern)
	}
	return nil
}

func validateInterval(s string) error {
	if !intervalPattern.MatchString(s) {
		return fmt.Errorf("invalid interval %q (allowed: 5m, 15m, 1h, 4h, 1d)", s)
	}
	return nil
}

// mlPredictV2Request is the enriched V2 predict payload.
// Backwards-compatible: if v2 fields missing, falls back to legacy behaviour.
type mlPredictV2Request struct {
	Symbol       string              `json:"symbol"`
	Interval     string              `json:"interval"`      // 1h|4h|1d
	Source       string              `json:"source"`        // binance|moex
	TradingStyle string              `json:"trading_style"` // scalp|day|swing|position
	// RiskTier (Patch 2C): optional override. Empty → inherit user's
	// saved tier from JWT context; if that's also empty → "balanced".
	RiskTier string              `json:"risk_tier,omitempty"`
	Candles  []types.OHLCVCandle `json:"candles,omitempty"`
}

// resolveRiskTier picks the effective tier from (request → user row → default).
// Called per /predict request. Keeps DB lookup out of the ml package.
func resolveRiskTier(r *http.Request, explicit string) string {
	if explicit != "" && ml.IsValidTier(explicit) {
		return explicit
	}
	// Lazy-load user's stored tier from DB using the JWT-injected user_id.
	if store.Pool != nil {
		if userID := auth.GetUserID(r); userID != "" {
			if u, err := auth.GetUserByID(r.Context(), store.Pool, userID); err == nil && ml.IsValidTier(u.RiskTier) {
				return u.RiskTier
			}
		}
	}
	return string(ml.TierBalanced)
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
	// Risk tier resolution: explicit request > user's saved tier > balanced.
	tier := resolveRiskTier(r, req.RiskTier)
	prediction := ml.PredictV2WithTier(req.Symbol, req.Interval, req.TradingStyle, tier, candles, nil, btcCloses)
	writeJSON(w, prediction)
}

// mlPredictMultiResponse bundles predictions for the same symbol across
// several intervals. The MTF "verdict" is the alignment of direction/confidence
// — when all intervals agree we flag it as high-quality.
//
// Patch 2C adds Consensus.Label/LabelReason/Blocked so the frontend can
// show trend_aligned / mean_reversion / random badges and know when a
// tier gate downgraded the signal.
type mlPredictMultiResponse struct {
	Symbol      string                      `json:"symbol"`
	Primary     string                      `json:"primary_interval"`
	Predictions map[string]*ml.PredictionV2 `json:"predictions"` // interval → prediction
	Consensus   struct {
		Direction     string  `json:"direction"`     // aligned direction or "mixed"
		Alignment     float64 `json:"alignment"`     // 0..1, fraction of TFs agreeing with majority
		HighQuality   bool    `json:"high_quality"`  // true if 100% aligned AND majority has HC
		AvgConfidence float64 `json:"avg_confidence"`
		// Label is the risk-tier classification applied to the consensus
		// direction: "trend_aligned" | "mean_reversion" | "random".
		Label string `json:"label,omitempty"`
		// LabelReason is a short human-readable explanation (e.g.
		// "1d trend aligned, adx=24.3"). Surfaces in UI tooltips.
		LabelReason string `json:"label_reason,omitempty"`
		// Blocked is true when the user's risk tier does not allow this label
		// (e.g. conservative tier receiving mean_reversion). When set, the
		// effective Direction is forced to "neutral".
		Blocked bool `json:"blocked,omitempty"`
		// RiskTier echoes the tier applied (for UI badges).
		RiskTier string `json:"risk_tier,omitempty"`
	} `json:"consensus"`
}

type mlPredictMultiRequest struct {
	Symbol       string   `json:"symbol"`
	Intervals    []string `json:"intervals"` // ["1h","4h","1d"]
	TradingStyle string   `json:"trading_style"`
	Source       string   `json:"source"`
	// RiskTier (Patch 2C): optional per-request override.
	RiskTier string `json:"risk_tier,omitempty"`
}

// ─── Signal classifier (Patch 2C) ────────────────────────────────
// Pure rule-based — no ML retraining. Relies on features already
// exposed by PredictionV2.Features, plus the 1d trend-anchor
// direction cached for 15 min to avoid spawning a 1d predict on
// every classify call (handy for scalp/day signals).
//
// PHASE 1 fix: daily-direction cache moved to internal/ml so the
// scenario runner shares the same cache — previously it bypassed
// and hammered 1d predict on every tick.

// ClassifySignal is re-exported from internal/ml so the scenario package can
// reuse the same rules without an import cycle. Kept as a thin alias here to
// avoid breaking existing callers in this file.
func ClassifySignal(signalDir, dailyDir, interval string, features map[string]float64) (string, string) {
	return ml.ClassifySignal(signalDir, dailyDir, interval, features)
}

// intervalRank sorts interval strings by actual duration (regardless of
// request order). Used to pick the highest TF as trend anchor.
var intervalRank = map[string]int{
	"1m": 1, "5m": 2, "15m": 3, "30m": 4, "1h": 5, "4h": 6, "1d": 7, "1w": 8,
}

func pickHighestInterval(intervals []string) string {
	best := ""
	bestRank := -1
	for _, iv := range intervals {
		if r, ok := intervalRank[iv]; ok && r > bestRank {
			bestRank = r
			best = iv
		}
	}
	if best == "" && len(intervals) > 0 {
		return intervals[0] // fallback
	}
	return best
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

	// Resolve tier once; reused for per-TF vol gates AND post-consensus label gate.
	tier := resolveRiskTier(r, req.RiskTier)

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
		p := ml.PredictV2WithTier(req.Symbol, iv, req.TradingStyle, tier, candles, nil, btcCloses)
		predictions[iv] = &p
	}

	// Compute consensus: count directions, check HC alignment.
	// Two vote pools: all predictions + HC-only. When HC signals exist,
	// they outvote non-HC neutrals (fixes case where 2 HC SELLs lose to
	// 3 neutral-but-biased predictions and the consensus reports flat).
	dirCount := map[string]int{"buy": 0, "sell": 0, "neutral": 0}
	hcDirCount := map[string]int{"buy": 0, "sell": 0, "neutral": 0}
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
			hcDirCount[p.Direction]++
		}
	}

	// Pick primary = highest-TF (trend anchor), regardless of request order.
	primary := pickHighestInterval(req.Intervals)
	resp := mlPredictMultiResponse{
		Symbol:      req.Symbol,
		Primary:     primary,
		Predictions: predictions,
	}

	resp.Consensus.RiskTier = tier

	if nValid > 0 {
		// Prefer HC-weighted majority when any HC signals exist.
		// HC predictions are the ones that actually passed the model's
		// confidence gate — they carry more information than neutral
		// predictions that fell through by default.
		var majority string
		var maxCount int
		hcBuy, hcSell := hcDirCount["buy"], hcDirCount["sell"]
		switch {
		case hcBuy > hcSell:
			majority = "buy"
			maxCount = hcBuy
		case hcSell > hcBuy:
			majority = "sell"
			maxCount = hcSell
		default:
			// No HC signals or HC tie — fall back to raw majority.
			majority = "neutral"
			maxCount = 0
			for dir, c := range dirCount {
				if c > maxCount {
					maxCount = c
					majority = dir
				}
			}
		}
		alignment := float64(maxCount) / float64(nValid)
		resp.Consensus.Direction = majority
		// Explicit conflict: at least one HC buy AND one HC sell.
		if hcBuy > 0 && hcSell > 0 {
			resp.Consensus.Direction = "mixed"
		}
		resp.Consensus.Alignment = alignment
		resp.Consensus.AvgConfidence = sumConf / float64(nValid)
		// High-quality signal: ALL intervals agree + at least one HC on any TF.
		resp.Consensus.HighQuality = alignment == 1.0 && hcAligned > 0 && majority != "neutral"

		// Patch 2C: classify consensus + apply tier label gate.
		// Use primary-interval features as the signal's feature snapshot.
		// Daily direction comes from the matching prediction if requested;
		// else lazy-fetch with a 15-min cache.
		var primaryFeatures map[string]float64
		if pp, ok := predictions[primary]; ok && pp != nil {
			primaryFeatures = pp.Features
		}
		var dailyDir string
		if pd, ok := predictions["1d"]; ok && pd != nil {
			dailyDir = pd.Direction
		} else if req.Symbol != "" {
			dailyDir = ml.GetDailyDirectionCached(req.Symbol, func() string {
				cd, err := market.FetchCryptoCandlesCached(req.Symbol, "1d", 500)
				if err != nil || len(cd) < 30 {
					return ""
				}
				// No BTC context on cold classify → acceptable; direction is all we need.
				dp := ml.PredictV2WithTier(req.Symbol, "1d", req.TradingStyle, tier, cd, nil, nil)
				return dp.Direction
			})
		}
		if primaryFeatures != nil {
			label, reason := ClassifySignal(resp.Consensus.Direction, dailyDir, primary, primaryFeatures)
			resp.Consensus.Label = label
			resp.Consensus.LabelReason = reason
			policy := ml.GetTier(tier)
			if !policy.LabelAllowed(label) {
				resp.Consensus.Blocked = true
				resp.Consensus.Direction = "neutral"
			}
		}
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

// handleMLRunBacktest triggers backtest.py in background (guarded by slot).
func handleMLRunBacktest(w http.ResponseWriter, r *http.Request) {
	select {
	case mlJobSlot <- struct{}{}:
	default:
		http.Error(w, `{"error":"ML job already in progress"}`, http.StatusTooManyRequests)
		return
	}
	go func() {
		defer func() { <-mlJobSlot }()
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
		defer cancel()
		trainingDir, _ := filepath.Abs("ml-training")
		cmd := exec.CommandContext(ctx, "./.venv/bin/python", "-u", "backtest.py")
		cmd.Dir = trainingDir
		_ = cmd.Run()
	}()
	writeJSON(w, map[string]any{
		"status":  "started",
		"message": "Backtest running — poll /ml/backtest in ~2-5 min",
	})
}

// handleMLRunPaperTrades triggers paper_trade.py in background (guarded).
func handleMLRunPaperTrades(w http.ResponseWriter, r *http.Request) {
	select {
	case mlJobSlot <- struct{}{}:
	default:
		http.Error(w, `{"error":"ML job already in progress"}`, http.StatusTooManyRequests)
		return
	}
	go func() {
		defer func() { <-mlJobSlot }()
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
		defer cancel()
		trainingDir, _ := filepath.Abs("ml-training")
		cmd := exec.CommandContext(ctx, "./.venv/bin/python", "-u", "paper_trade.py")
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
	if err := validateSymbol(req.Symbol); err != nil {
		http.Error(w, `{"error":"`+err.Error()+`"}`, http.StatusBadRequest)
		return
	}
	if err := validateInterval(req.Interval); err != nil {
		http.Error(w, `{"error":"`+err.Error()+`"}`, http.StatusBadRequest)
		return
	}

	// Reserve the single training slot. If busy, reject — avoid spawning
	// multiple XGBoost processes that would saturate CPU + RAM.
	select {
	case mlJobSlot <- struct{}{}:
	default:
		http.Error(w, `{"error":"training already in progress; retry in a few minutes"}`, http.StatusTooManyRequests)
		return
	}

	taskID := time.Now().Format("20060102-150405")

	go func(symbol, interval string, quick bool) {
		defer func() { <-mlJobSlot }() // release slot when done
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Minute)
		defer cancel()

		trainingDir, _ := filepath.Abs("ml-training")
		args := []string{
			"train.py",
			"--symbols", symbol,
			"--intervals", interval,
		}
		if quick {
			args = append(args, "--quick")
		}
		cmd := exec.CommandContext(ctx, "./.venv/bin/python", args...)
		cmd.Dir = trainingDir
		_ = cmd.Run()
		_, _ = ml.LoadModelsV2()
		ml.MarkLoaded()
	}(req.Symbol, req.Interval, req.Quick)

	writeJSON(w, map[string]any{
		"task_id": taskID,
		"status":  "started",
		"message": "Training in background — poll GET /api/v1/ml/models after ~5 min",
	})
}

// handleMLReload forces re-scan of the models directory + threshold file.
// Call after train.py + analyze_thresholds.py produce fresh artifacts.
func handleMLReload(w http.ResponseWriter, r *http.Request) {
	n, err := ml.LoadModelsV2()
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	ml.MarkLoaded()
	// Also hot-reload thresholds (previously main.go did this once at startup,
	// meaning fresh analyze_thresholds output was stale until restart).
	nThr, _ := ml.LoadThresholds("ml-training")
	writeJSON(w, map[string]any{"models_loaded": n, "thresholds_loaded": nThr})
}
