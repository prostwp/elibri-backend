package ml

import (
	"math"
	"math/rand"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// DecisionStump — one weak learner in GBDT
type DecisionStump struct {
	FeatureIdx int     `json:"feature_idx"`
	Threshold  float64 `json:"threshold"`
	LeftValue  float64 `json:"left_value"`
	RightValue float64 `json:"right_value"`
}

// GBDTModel — Gradient Boosted Decision Trees
type GBDTModel struct {
	Trees        []DecisionStump `json:"trees"`
	LearningRate float64         `json:"learning_rate"`
	Intercept    float64         `json:"intercept"`
	Version      int             `json:"version"`
}

// Predict returns probability of price going up (0-1)
func (m *GBDTModel) Predict(f Features) float64 {
	feats := [6]float64{f.RSINorm, f.MACDNorm, f.VolRatio, f.ATRNorm, f.MomentumNorm, f.BBPosition}

	score := m.Intercept
	for _, tree := range m.Trees {
		val := feats[tree.FeatureIdx]
		if val <= tree.Threshold {
			score += m.LearningRate * tree.LeftValue
		} else {
			score += m.LearningRate * tree.RightValue
		}
	}

	// Sigmoid
	return 1.0 / (1.0 + math.Exp(-score))
}

// Train creates a GBDT model from labeled data
func Train(candles []types.OHLCVCandle, lookforward int, numTrees int, lr float64) *GBDTModel {
	if len(candles) < 50 {
		return defaultModel()
	}

	var samples []sample
	for i := 30; i < len(candles)-lookforward; i++ {
		f := ExtractFeatures(candles[:i+1])
		futurePrice := candles[i+lookforward].Close
		currentPrice := candles[i].Close
		atr := calcATR(candles[:i+1], 14)

		label := 0.0
		if futurePrice > currentPrice+atr*0.5 {
			label = 1.0
		}
		samples = append(samples, sample{features: f, label: label})
	}

	if len(samples) < 20 {
		return defaultModel()
	}

	// Simple GBDT training
	residuals := make([]float64, len(samples))
	avgLabel := 0.0
	for _, s := range samples {
		avgLabel += s.label
	}
	avgLabel /= float64(len(samples))
	intercept := math.Log(avgLabel / (1 - avgLabel + 1e-10))

	// Initialize residuals
	for i, s := range samples {
		pred := 1.0 / (1.0 + math.Exp(-intercept))
		residuals[i] = s.label - pred
	}

	trees := make([]DecisionStump, 0, numTrees)

	for t := 0; t < numTrees; t++ {
		// Find best split
		bestStump := findBestSplit(samples, residuals)
		trees = append(trees, bestStump)

		// Update residuals
		for i, s := range samples {
			feats := [6]float64{s.features.RSINorm, s.features.MACDNorm, s.features.VolRatio,
				s.features.ATRNorm, s.features.MomentumNorm, s.features.BBPosition}
			_ = feats[bestStump.FeatureIdx] // ensure used
			pred := intercept
			for _, tree := range trees {
				if feats[tree.FeatureIdx] <= tree.Threshold {
					pred += lr * tree.LeftValue
				} else {
					pred += lr * tree.RightValue
				}
			}
			prob := 1.0 / (1.0 + math.Exp(-pred))
			residuals[i] = s.label - prob
		}
	}

	return &GBDTModel{
		Trees:        trees,
		LearningRate: lr,
		Intercept:    intercept,
		Version:      1,
	}
}

type sample struct {
	features Features
	label    float64
}

func findBestSplit(samples []sample, residuals []float64) DecisionStump {
	bestGain := -math.MaxFloat64
	best := DecisionStump{}

	for fi := 0; fi < 6; fi++ {
		// Try 10 random thresholds
		for attempt := 0; attempt < 10; attempt++ {
			threshold := rand.Float64()

			var leftSum, rightSum float64
			var leftCount, rightCount int

			for i, s := range samples {
				feats := [6]float64{s.features.RSINorm, s.features.MACDNorm, s.features.VolRatio,
					s.features.ATRNorm, s.features.MomentumNorm, s.features.BBPosition}
				if feats[fi] <= threshold {
					leftSum += residuals[i]
					leftCount++
				} else {
					rightSum += residuals[i]
					rightCount++
				}
			}

			if leftCount == 0 || rightCount == 0 {
				continue
			}

			leftVal := leftSum / float64(leftCount)
			rightVal := rightSum / float64(rightCount)

			// Gain = sum of squared predictions
			gain := leftVal*leftVal*float64(leftCount) + rightVal*rightVal*float64(rightCount)

			if gain > bestGain {
				bestGain = gain
				best = DecisionStump{
					FeatureIdx: fi,
					Threshold:  threshold,
					LeftValue:  leftVal,
					RightValue: rightVal,
				}
			}
		}
	}

	return best
}

func defaultModel() *GBDTModel {
	return &GBDTModel{
		Trees: []DecisionStump{
			{FeatureIdx: 0, Threshold: 0.3, LeftValue: 0.5, RightValue: -0.3},   // RSI oversold = buy
			{FeatureIdx: 1, Threshold: 0.5, LeftValue: -0.3, RightValue: 0.4},   // MACD positive = buy
			{FeatureIdx: 4, Threshold: 0.5, LeftValue: -0.2, RightValue: 0.3},   // Momentum up = buy
			{FeatureIdx: 5, Threshold: 0.2, LeftValue: 0.5, RightValue: -0.1},   // BB near lower = buy
		},
		LearningRate: 0.1,
		Intercept:    0,
		Version:      0,
	}
}
