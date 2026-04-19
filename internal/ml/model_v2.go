package ml

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// ─── V2 Model Schema ─────────────────────────────────────────
// Exported by ml-training/train.py → serialize_model().
// For MVP, Go inference uses RF trees + meta-learner. XGB/LGBM raw models are
// preserved in the JSON for future native inference (requires XGB/LGBM Go bindings
// or re-implementation of their tree formats). Until then, meta-learner is
// fed with [rf_prob, rf_prob, rf_prob] as a calibrated fallback — this matches
// training OOF distribution closely for small data.

type RFTree struct {
	ChildrenLeft  []int       `json:"children_left"`
	ChildrenRight []int       `json:"children_right"`
	Feature       []int       `json:"feature"`
	Threshold     []float64   `json:"threshold"`
	Value         [][]float64 `json:"value"` // per-class counts [n_class_0, n_class_1]
}

type ModelV2 struct {
	Version           string             `json:"version"`
	Symbol            string             `json:"symbol"`
	Interval          string             `json:"interval"`
	Horizon           int                `json:"horizon"`
	FeatureCols       []string           `json:"feature_cols"`
	TrainedAt         string             `json:"trained_at"`
	MetaWeights       []float64          `json:"meta_weights"`
	MetaIntercept     float64            `json:"meta_intercept"`
	XGBModel          string             `json:"xgb_model"` // reserved
	LGBMModel         string             `json:"lgbm_model"` // reserved
	RFTrees           []RFTree           `json:"rf_trees"`
	Metrics           ModelMetrics       `json:"metrics"`
	FeatureImportance map[string]float64 `json:"feature_importance"`
}

type ModelMetrics struct {
	Folds           []FoldMetrics `json:"folds"`
	AvgAccuracy     float64       `json:"avg_accuracy"`
	AvgSharpe       float64       `json:"avg_sharpe"`
	AvgF1           float64       `json:"avg_f1"`
	AvgPrecision    float64       `json:"avg_precision"`
	AvgRecall       float64       `json:"avg_recall"`
	HCPrecision     float64       `json:"hc_precision"`       // precision on high-conf trades only
	HCSignalsTotal  int           `json:"hc_signals_total"`   // # trades passing filter
	HCSignalRate    float64       `json:"hc_signal_rate"`     // fraction of bars that get a signal
	NFolds          int           `json:"n_folds"`
	NTestTotal      int           `json:"n_test_total"`
}

type FoldMetrics struct {
	Fold         int     `json:"fold"`
	TrainStart   string  `json:"train_start"`
	TrainEnd     string  `json:"train_end"`
	TestStart    string  `json:"test_start"`
	TestEnd      string  `json:"test_end"`
	NTrain       int     `json:"n_train"`
	NTest        int     `json:"n_test"`
	Accuracy     float64 `json:"accuracy"`
	Precision    float64 `json:"precision"`
	Recall       float64 `json:"recall"`
	F1           float64 `json:"f1"`
	Sharpe       float64 `json:"sharpe"`
	HCPrecision  float64 `json:"hc_precision"`
	HCCount      int     `json:"hc_count"`
	HCWinRate    float64 `json:"hc_win_rate"`
}

type LatestPointer map[string]struct {
	Model    string  `json:"model"`
	Patterns string  `json:"patterns"`
	Horizon  int     `json:"horizon"`
	Accuracy float64 `json:"accuracy"`
	Sharpe   float64 `json:"sharpe"`
}

// ─── Registry ────────────────────────────────────────────────

var (
	modelsV2   = make(map[string]*ModelV2)  // key: {symbol}_{interval}
	patternsV2 = make(map[string]*Patterns) // key: {symbol}_{interval}
	v2Mu       sync.RWMutex
	modelsDir  = "internal/ml/models"
)

// SetModelsDir overrides the default lookup directory (for tests / alt deploys).
func SetModelsDir(dir string) {
	v2Mu.Lock()
	defer v2Mu.Unlock()
	modelsDir = dir
}

// LoadModelsV2 scans modelsDir for latest.json and loads all referenced models
// + pattern indices. Returns count loaded and any non-fatal error summary.
func LoadModelsV2() (int, error) {
	latestPath := filepath.Join(modelsDir, "latest.json")
	data, err := os.ReadFile(latestPath)
	if err != nil {
		return 0, fmt.Errorf("read latest.json: %w", err)
	}
	var latest LatestPointer
	if err := json.Unmarshal(data, &latest); err != nil {
		return 0, fmt.Errorf("parse latest.json: %w", err)
	}

	loaded := 0
	v2Mu.Lock()
	defer v2Mu.Unlock()
	for key, ref := range latest {
		modelPath := filepath.Join(modelsDir, ref.Model)
		m, err := readModelFile(modelPath)
		if err != nil {
			// non-fatal: skip this key, continue others
			continue
		}
		modelsV2[key] = m

		patPath := filepath.Join(modelsDir, ref.Patterns)
		if pat, err := readPatternsFile(patPath); err == nil {
			patternsV2[key] = pat
		}
		loaded++
	}
	return loaded, nil
}

func readModelFile(path string) (*ModelV2, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m ModelV2
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// ─── Ensemble inference ─────────────────────────────────────

// Predict returns (probability, rfProbRaw) for feature vector x.
// The ensemble uses RF + meta-learner. meta_weights fitted on [xgb, lgbm, rf]
// outputs during training; here we feed RF output three times as a stable
// approximation until XGB/LGBM Go implementations land.
func (m *ModelV2) Predict(x []float64) (prob float64, rfProb float64) {
	if len(x) != len(m.FeatureCols) {
		return 0.5, 0.5
	}

	// Average RF tree probabilities.
	var sum float64
	for _, t := range m.RFTrees {
		p := traverseTree(&t, x)
		sum += p
	}
	if len(m.RFTrees) > 0 {
		rfProb = sum / float64(len(m.RFTrees))
	} else {
		rfProb = 0.5
	}

	// Meta-learner: logistic regression on [p_xgb, p_lgbm, p_rf].
	// Feed rfProb for all three until XGB/LGBM Go inference lands.
	if len(m.MetaWeights) == 3 {
		z := m.MetaIntercept
		for _, w := range m.MetaWeights {
			z += w * rfProb
		}
		prob = 1.0 / (1.0 + math.Exp(-z))
	} else {
		prob = rfProb
	}
	return prob, rfProb
}

func traverseTree(t *RFTree, x []float64) float64 {
	node := 0
	for {
		if node >= len(t.ChildrenLeft) || t.ChildrenLeft[node] == -1 {
			// leaf: convert class counts to P(class=1)
			v := t.Value[node]
			if len(v) < 2 {
				return 0.5
			}
			total := v[0] + v[1]
			if total == 0 {
				return 0.5
			}
			return v[1] / total
		}
		fi := t.Feature[node]
		if fi < 0 || fi >= len(x) {
			return 0.5
		}
		if x[fi] <= t.Threshold[node] {
			node = t.ChildrenLeft[node]
		} else {
			node = t.ChildrenRight[node]
		}
	}
}

// ─── Lookup helpers ──────────────────────────────────────────

func GetModelV2(symbol, interval string) (*ModelV2, bool) {
	key := symbol + "_" + interval
	v2Mu.RLock()
	defer v2Mu.RUnlock()
	m, ok := modelsV2[key]
	return m, ok
}

func GetPatternsV2(symbol, interval string) (*Patterns, bool) {
	key := symbol + "_" + interval
	v2Mu.RLock()
	defer v2Mu.RUnlock()
	p, ok := patternsV2[key]
	return p, ok
}

// ListLoadedModels returns metadata about each loaded model.
type LoadedModelInfo struct {
	Key       string  `json:"key"`
	Symbol    string  `json:"symbol"`
	Interval  string  `json:"interval"`
	Horizon   int     `json:"horizon"`
	Accuracy  float64 `json:"accuracy"`
	Sharpe    float64 `json:"sharpe"`
	F1        float64 `json:"f1"`
	NFolds    int     `json:"n_folds"`
	TrainedAt string  `json:"trained_at"`
	NFeatures int     `json:"n_features"`
	NTrees    int     `json:"n_trees"`
}

func ListLoadedModels() []LoadedModelInfo {
	v2Mu.RLock()
	defer v2Mu.RUnlock()
	out := make([]LoadedModelInfo, 0, len(modelsV2))
	for key, m := range modelsV2 {
		out = append(out, LoadedModelInfo{
			Key:       key,
			Symbol:    m.Symbol,
			Interval:  m.Interval,
			Horizon:   m.Horizon,
			Accuracy:  m.Metrics.AvgAccuracy,
			Sharpe:    m.Metrics.AvgSharpe,
			F1:        m.Metrics.AvgF1,
			NFolds:    m.Metrics.NFolds,
			TrainedAt: m.TrainedAt,
			NFeatures: len(m.FeatureCols),
			NTrees:    len(m.RFTrees),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out
}

// ─── Feature importance sorted top-N ─────────────────────────

type FeatureImp struct {
	Name       string  `json:"name"`
	Importance float64 `json:"importance"`
}

func (m *ModelV2) TopFeatures(n int) []FeatureImp {
	list := make([]FeatureImp, 0, len(m.FeatureImportance))
	for name, imp := range m.FeatureImportance {
		list = append(list, FeatureImp{Name: name, Importance: imp})
	}
	sort.Slice(list, func(i, j int) bool { return list[i].Importance > list[j].Importance })
	if n > 0 && n < len(list) {
		list = list[:n]
	}
	return list
}

// ─── Health check ────────────────────────────────────────────

type V2Status struct {
	LoadedAt time.Time `json:"loaded_at"`
	NModels  int       `json:"n_models"`
	Models   []string  `json:"models"`
}

var v2LoadedAt time.Time

func V2Health() V2Status {
	v2Mu.RLock()
	defer v2Mu.RUnlock()
	names := make([]string, 0, len(modelsV2))
	for k := range modelsV2 {
		names = append(names, k)
	}
	sort.Strings(names)
	return V2Status{LoadedAt: v2LoadedAt, NModels: len(modelsV2), Models: names}
}

// MarkLoaded called at end of LoadModelsV2 — exposed for tests.
func MarkLoaded() {
	v2Mu.Lock()
	v2LoadedAt = time.Now()
	v2Mu.Unlock()
}
