package ml

import (
	"encoding/json"
	"os"
	"path/filepath"
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
	defer thresholdsMu.Unlock()
	for _, r := range raw.Results {
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
	return len(raw.Results), nil
}

// deriveAbsoluteThresholds turns a key like "thr_0.65" or "top_10pct" into
// concrete high/low probability boundaries.
func deriveAbsoluteThresholds(key string, mean, std float64) (high, low float64) {
	switch key {
	case "thr_0.55":
		return 0.55, 0.45
	case "thr_0.60":
		return 0.60, 0.40
	case "thr_0.65":
		return 0.65, 0.35
	case "thr_0.70":
		return 0.70, 0.30
	case "thr_0.80":
		return 0.80, 0.20
	case "top_10pct":
		// Approximate top-10pct band using empirical mean ± 1.28 × std
		// (1.28 ≈ 90th percentile of standard normal).
		high = mean + 1.28*std
		low = mean - 1.28*std
		// Clamp sanity.
		if high < 0.51 {
			high = 0.51
		}
		if low > 0.49 {
			low = 0.49
		}
		return high, low
	default:
		return 0.80, 0.20
	}
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
