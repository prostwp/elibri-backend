package ml

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// Patterns mirrors pattern_matcher.py PatternIndex serialization.
// Scaler stored as mean/scale arrays; samples pre-scaled.
// Linear k-NN scan (n typically 5-10k, k=10 → fast enough).
type Patterns struct {
	FeatureCols []string    `json:"feature_cols"`
	ScalerMean  []float64   `json:"scaler_mean"`
	ScalerScale []float64   `json:"scaler_scale"`
	Samples     [][]float64 `json:"samples"`   // N × F, already z-scored
	Outcomes    [][]float64 `json:"outcomes"`  // N × 3 (5/10/20 bar returns)
	Timestamps  []string    `json:"timestamps"`
	Closes      []float64   `json:"closes"`
}

type SimilarSituation struct {
	Date        string  `json:"date"`
	Distance    float64 `json:"distance"`
	Outcome5    float64 `json:"outcome_5"`
	Outcome10   float64 `json:"outcome_10"`
	Outcome20   float64 `json:"outcome_20"`
	Description string  `json:"description"`
}

func readPatternsFile(path string) (*Patterns, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var p Patterns
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, err
	}
	return &p, nil
}

// Query returns top-k most similar historical samples.
// rawFeatures are in original feature space — we z-score them with stored scaler.
func (p *Patterns) Query(rawFeatures []float64, k int) []SimilarSituation {
	// Strict validation to prevent index-out-of-bounds panic when
	// model and scaler arrays have mismatched lengths (corrupted JSON).
	if len(p.Samples) == 0 ||
		len(rawFeatures) != len(p.FeatureCols) ||
		len(p.ScalerMean) != len(rawFeatures) ||
		len(p.ScalerScale) != len(rawFeatures) {
		return nil
	}
	// Scale.
	x := make([]float64, len(rawFeatures))
	for i, v := range rawFeatures {
		scale := p.ScalerScale[i]
		if scale == 0 {
			scale = 1
		}
		x[i] = (v - p.ScalerMean[i]) / scale
	}

	// Compute distances (linear; fine for <50k samples).
	type hit struct {
		idx  int
		dist float64
	}
	hits := make([]hit, len(p.Samples))
	for i, s := range p.Samples {
		var d float64
		for j, v := range s {
			diff := v - x[j]
			d += diff * diff
		}
		hits[i] = hit{idx: i, dist: math.Sqrt(d)}
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].dist < hits[j].dist })

	if k > len(hits) {
		k = len(hits)
	}
	out := make([]SimilarSituation, 0, k)
	for i := 0; i < k; i++ {
		h := hits[i]
		out5 := p.Outcomes[h.idx][0]
		out10 := p.Outcomes[h.idx][1]
		out20 := p.Outcomes[h.idx][2]
		out = append(out, SimilarSituation{
			Date:        p.Timestamps[h.idx],
			Distance:    h.dist,
			Outcome5:    out5,
			Outcome10:   out10,
			Outcome20:   out20,
			Description: describeOutcome(h.dist, out10),
		})
	}
	return out
}

func describeOutcome(distance, outcome10 float64) string {
	var closeness string
	switch {
	case distance < 1:
		closeness = "очень похожая"
	case distance < 2:
		closeness = "схожая"
	default:
		closeness = "отдалённая"
	}
	sign := "+"
	pct := outcome10 * 100
	if pct < 0 {
		sign = ""
	}
	return fmt.Sprintf("%s ситуация → %s%.1f%% за 10 баров", closeness, sign, pct)
}

// AggregateOutcome averages the 5/10/20-bar returns across neighbours,
// weighted by inverse distance. Used for the "average outcome" summary in UI.
func AggregateOutcome(sims []SimilarSituation) (avg5, avg10, avg20 float64) {
	if len(sims) == 0 {
		return 0, 0, 0
	}
	var wsum, s5, s10, s20 float64
	for _, s := range sims {
		w := 1.0 / (1.0 + s.Distance)
		wsum += w
		s5 += s.Outcome5 * w
		s10 += s.Outcome10 * w
		s20 += s.Outcome20 * w
	}
	if wsum == 0 {
		return 0, 0, 0
	}
	return s5 / wsum, s10 / wsum, s20 / wsum
}
