// Package narrative implements the Narrative Radar scoring layer:
// pure functions and type definitions that classify crypto narratives
// (e.g. "Restaking", "AI Tokens", "RWA") into a stage and a 0..100
// trend score. This package is the contract between the (future)
// ingestion worker and the (future) REST/UI layer — the on-disk schema
// in migration 009_narratives.sql is mirrored here field-for-field.
//
// This file holds types only. The math lives in scorer.go.
package narrative

import "time"

// Stage labels a narrative's position in its lifecycle. Values match the
// CHECK constraint on narrative_snapshots.stage in migration 009.
type Stage string

const (
	// StageEarly: low absolute volume but explosive growth. The "alpha"
	// bucket — high upside, high noise.
	StageEarly Stage = "early"
	// StageTrending: meaningful volume + accelerating growth. Most users
	// will care about this bucket.
	StageTrending Stage = "trending"
	// StageMainstream: high volume with growth flattening out. The narrative
	// is priced in.
	StageMainstream Stage = "mainstream"
	// StageDeclining: negative growth regardless of absolute volume — the
	// narrative is bleeding attention.
	StageDeclining Stage = "declining"
)

// SentimentLabel buckets a numeric sentiment score (-1..+1) into three
// human-readable labels. Values match the CHECK constraint on
// narrative_snapshots.sentiment_label.
type SentimentLabel string

const (
	// SentBull: aggregate sentiment > +0.2 (VADER convention).
	SentBull SentimentLabel = "bull"
	// SentNeutral: |sentiment| <= 0.2 (boundary inclusive).
	SentNeutral SentimentLabel = "neutral"
	// SentBear: aggregate sentiment < -0.2.
	SentBear SentimentLabel = "bear"
)

// Source identifies where a Mention came from. Kept as a string-typed
// constant set so the ingestion layer can add new providers without a
// schema migration — the SQL column is plain VARCHAR(32).
type Source string

const (
	// SourceReddit: r/CryptoCurrency hot, etc.
	SourceReddit Source = "reddit"
	// SourceCoinDesk: CoinDesk RSS feed.
	SourceCoinDesk Source = "coindesk"
	// SourceCoinTelegraph: CoinTelegraph RSS feed.
	SourceCoinTelegraph Source = "cointelegraph"
)

// Mention mirrors one row of narrative_mentions (migration 009). One row
// per (URL, narrative) pair — the same article matching multiple
// narratives is denormalised on purpose so count queries stay simple.
type Mention struct {
	Narrative string    `json:"narrative"`
	Source    Source    `json:"source"`
	URL       string    `json:"url"`
	Title     string    `json:"title"`
	PostedAt  time.Time `json:"posted_at"`
	// Sentiment in [-1, +1], from news.scoreSentiment.
	Sentiment float64  `json:"sentiment"`
	Tickers   []string `json:"tickers"`
}

// Snapshot mirrors one row of narrative_snapshots (migration 009). The
// worker writes one Snapshot per (narrative, tick); the read path
// returns the latest one per narrative.
type Snapshot struct {
	Narrative           string    `json:"narrative"`
	CapturedAt          time.Time `json:"captured_at"`
	MentionCount24h     int       `json:"mention_count_24h"`
	MentionCountPrev24h int       `json:"mention_count_prev_24h"`
	// GrowthPct is capped at ±999 by ComputeGrowthPct.
	GrowthPct float64 `json:"growth_pct"`
	// TrendScore in [0, 100], from ComputeTrendScore.
	TrendScore int   `json:"trend_score"`
	Stage      Stage `json:"stage"`
	// SentimentScore in [-1, +1], aggregated across mentions.
	SentimentScore float64        `json:"sentiment_score"`
	SentimentLabel SentimentLabel `json:"sentiment_label"`
	// Confidence in [0, 100] — data-quality score (volume, source diversity,
	// sentiment consistency). NOT a prediction-strength score.
	Confidence       int            `json:"confidence"`
	RelatedAssets    []string       `json:"related_assets"`
	SourcesBreakdown map[string]int `json:"sources_breakdown"`
	Reasons          []string       `json:"reasons"`
}
