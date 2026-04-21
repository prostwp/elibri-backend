package ml

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

// HCThreshold is a per-model confidence threshold learned post-hoc by
// analyze_thresholds.py. Key: "{symbol}_{interval}". If absent, fall back
// to the default 0.80/0.20 filter.
type HCThreshold struct {
	Symbol        string  `json:"symbol"`
	Interval      string  `json:"interval"`
	Key           string  `json:"key"`       // e.g. "thr_0.60" or "top_10pct"
	ThresholdHigh float64 `json:"threshold_high"` // prob > this = long signal
	ThresholdLow  float64 `json:"threshold_low"`  // prob < this = short signal
	Precision     float64 `json:"precision"`
	NSignals      int     `json:"n_signals"`
	Fraction      float64 `json:"fraction"` // fraction of test bars that pass
}

var (
	thresholdsMu sync.RWMutex
	thresholds   = make(map[string]HCThreshold)
)

// LoadThresholds reads logs/best_thresholds.json emitted by analyze_thresholds.py.
func LoadThresholds(dir string) (int, error) {
	path := filepath.Join(dir, "logs", "best_thresholds.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	var raw struct {
		Results []struct {
			Symbol   string `json:"symbol"`
			Interval string `json:"interval"`
			Best     struct {
				Key       string  `json:"key"`
				Precision float64 `json:"precision"`
				NSignals  int     `json:"n_signals"`
				Fraction  float64 `json:"fraction"`
			} `json:"best"`
			ProbaMin  float64 `json:"proba_min"`
			ProbaMax  float64 `json:"proba_max"`
			ProbaMean float64 `json:"proba_mean"`
			ProbaStd  float64 `json:"proba_std"`
		} `json:"results"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return 0, err
	}

	thresholdsMu.Lock()
	for _, r := range raw.Results {
		if r.Best.Key == "" {
			// No valid threshold was picked (e.g. pre-Patch 2G 1d models).
			// Skip — GetThreshold returns the safe default 0.80/0.20.
			continue
		}
		key := r.Symbol + "_" + r.Interval
		high, low := deriveAbsoluteThresholds(r.Best.Key, r.ProbaMean, r.ProbaStd)
		thresholds[key] = HCThreshold{
			Symbol:        r.Symbol,
			Interval:      r.Interval,
			Key:           r.Best.Key,
			ThresholdHigh: high,
			ThresholdLow:  low,
			Precision:     r.Best.Precision,
			NSignals:      r.Best.NSignals,
			Fraction:      r.Best.Fraction,
		}
	}
	count := len(thresholds)
	thresholdsMu.Unlock()

	// Env-based hotfix overrides (Patch 2G).
	if nOver := applyEnvOverrides(); nOver > 0 {
		log.Printf("ML thresholds: %d env override(s) applied", nOver)
	}

	return count, nil
}

// deriveAbsoluteThresholds turns a key like "thr_0.675" or "top_10pct" into
// concrete high/low probability boundaries.
//
// Patch 2G fix: previously this was a static switch matching only 5 hardcoded
// keys (thr_0.55/0.60/0.65/0.70/0.80). Any other threshold from
// analyze_thresholds.py (e.g. thr_0.775 picked by sweep) silently fell back
// to default 0.80/0.20 — meaning the backend IGNORED the fine-tuned
// thresholds and everything was gated at 0.80. This is why no signals fired
// on 5m/1h/4h in production despite backtest v2 showing 3-35 trades per
// period with effective thresholds 0.59-0.63.
func deriveAbsoluteThresholds(key string, mean, std float64) (high, low float64) {
	// thr_X.XXX form — parse any decimal threshold.
	if strings.HasPrefix(key, "thr_") {
		if v, err := strconv.ParseFloat(key[4:], 64); err == nil {
			if v > 0.5 && v <= 1.0 {
				return v, 1.0 - v // symmetric gate
			}
		}
	}
	if key == "top_10pct" {
		// Approximate top-10pct band using empirical mean ± 1.28 × std
		// (1.28 ≈ 90th percentile of standard normal).
		high = mean + 1.28*std
		low = mean - 1.28*std
		if high < 0.51 {
			high = 0.51
		}
		if low > 0.49 {
			low = 0.49
		}
		return high, low
	}
	// Unknown — safe default.
	return 0.80, 0.20
}

// applyEnvOverrides lets ops temporarily tune thresholds without regenerating
// best_thresholds.json — useful for paper-trading experiments.
//
// Format: ML_THRESHOLD_OVERRIDES="BTCUSDT_4h:0.60,BTCUSDT_1h:0.65"
// Each override sets both ThresholdHigh=v and ThresholdLow=1-v.
// Must be called AFTER LoadThresholds so it overwrites JSON-loaded values.
func applyEnvOverrides() int {
	env := os.Getenv("ML_THRESHOLD_OVERRIDES")
	if env == "" {
		return 0
	}
	n := 0
	thresholdsMu.Lock()
	defer thresholdsMu.Unlock()
	for _, pair := range strings.Split(env, ",") {
		pair = strings.TrimSpace(pair)
		kv := strings.SplitN(pair, ":", 2)
		if len(kv) != 2 {
			continue
		}
		key := strings.TrimSpace(kv[0])
		val, err := strconv.ParseFloat(strings.TrimSpace(kv[1]), 64)
		if err != nil || val <= 0.5 || val > 1.0 {
			log.Printf("ML_THRESHOLD_OVERRIDES: skipping invalid %q", pair)
			continue
		}
		existing := thresholds[key]
		if existing.Symbol == "" {
			// Key format SYMBOL_INTERVAL — split by last underscore.
			if idx := strings.LastIndex(key, "_"); idx > 0 {
				existing.Symbol = key[:idx]
				existing.Interval = key[idx+1:]
			}
		}
		existing.Key = "env_override"
		existing.ThresholdHigh = val
		existing.ThresholdLow = 1.0 - val
		thresholds[key] = existing
		log.Printf("ML threshold override: %s → HC=%.3f (low=%.3f)", key, val, 1.0-val)
		n++
	}
	return n
}

// GetThreshold returns the per-model threshold pair, or defaults if unknown.
func GetThreshold(symbol, interval string) HCThreshold {
	key := symbol + "_" + interval
	thresholdsMu.RLock()
	defer thresholdsMu.RUnlock()
	if t, ok := thresholds[key]; ok {
		return t
	}
	return HCThreshold{
		Symbol:        symbol,
		Interval:      interval,
		Key:           "default_0.80",
		ThresholdHigh: 0.80,
		ThresholdLow:  0.20,
	}
}

// ListThresholds returns all loaded thresholds — exposed via API for debug.
func ListThresholds() []HCThreshold {
	thresholdsMu.RLock()
	defer thresholdsMu.RUnlock()
	out := make([]HCThreshold, 0, len(thresholds))
	for _, t := range thresholds {
		out = append(out, t)
	}
	return out
}
