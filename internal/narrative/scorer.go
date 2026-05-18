package narrative

import "math"

// Calibration constants. All boundaries here are first-pass guesses meant
// to be tuned against backtest data once we have a few weeks of mentions
// in narrative_snapshots. Each constant has a one-line rationale so the
// next person tuning them knows what they're trading off.
const (
	// growthCap is the absolute cap on ComputeGrowthPct so a brand-new
	// narrative (prev=0) doesn't return +Inf and break the SMALLINT/REAL
	// columns downstream. ±999% covers anything we'd want to render.
	growthCap = 999.0

	// growthInfinite is the value returned when prev=0 and curr>0:
	// treat "first appearance" as the saturated growth value. Same number
	// as growthCap on purpose so the UI can render one "🚀 new" badge.
	growthInfinite = 999.0

	// stageDecliningGrowthMax: anything below this is declining regardless
	// of volume. -20% chosen to ignore minor day-to-day noise.
	stageDecliningGrowthMax = -20.0

	// stageEarlyMaxMentions: cap on absolute volume for "early" — above
	// this the narrative is no longer a niche signal.
	stageEarlyMaxMentions = 50

	// stageEarlyMinGrowth: a narrative needs at least this much growth to
	// qualify as early (otherwise it's just a quiet topic, not a brewing one).
	stageEarlyMinGrowth = 200.0

	// stageMainstreamMinMentions: floor on absolute volume for mainstream.
	// 500 mentions/24h is roughly "covered by every major outlet".
	stageMainstreamMinMentions = 500

	// stageMainstreamGrowthMin / stageMainstreamGrowthMax: mainstream means
	// growth has plateaued. Outside this band → trending or declining wins.
	stageMainstreamGrowthMin = -20.0
	stageMainstreamGrowthMax = 50.0

	// stageTrendingMinGrowth: above this is "trending" regardless of volume.
	stageTrendingMinGrowth = 50.0

	// stageTrendingMinMentions: alternative trending threshold via raw volume.
	stageTrendingMinMentions = 100

	// trendGrowthSaturation: growth value at which the growth component of
	// trend score saturates. 200% chosen empirically — past 2x growth more
	// growth doesn't make a narrative meaningfully more interesting.
	trendGrowthSaturation = 200.0
	// trendGrowthMaxPoints: cap on growth contribution to trend score.
	// v2: lowered from 40 → 35 to make room for the new 15-pt importance
	// component (35 + 25 + 15 + 10 + 15 = 100).
	trendGrowthMaxPoints = 35.0

	// trendVolumeSaturation: mentions/24h at which the volume component
	// saturates. 1000 ~= top-tier news cycle.
	trendVolumeSaturation = 1000.0
	// trendVolumeMaxPoints: cap on volume contribution.
	// v2: lowered from 30 → 25 (see growth comment for the v2 rebalance).
	trendVolumeMaxPoints = 25.0

	// trendSentimentMaxPoints: bullish sentiment of +1.0 contributes this many
	// points. Negative sentiment contributes 0 (bearish narratives can still
	// be tradeable, but we don't reward them in a trend score).
	trendSentimentMaxPoints = 15.0

	// trendDiversityMaxPoints / trendDiversityPerSource: 3+ distinct sources
	// caps. Single-source signals are over-weighted otherwise.
	// v2: lowered from 15 → 10 (see growth comment for the v2 rebalance).
	// trendDiversityPerSource stays at 5 because we still want 1 source = 5,
	// 2 = 10 — the cap just clips at 10 instead of 15.
	trendDiversityPerSource = 5.0
	trendDiversityMaxPoints = 10.0

	// trendImportanceSaturation / trendImportanceMaxPoints: avg LLM importance
	// of 100 saturates the new 15-pt component. Below 100 the contribution
	// scales linearly: an avg importance of 50 gives 7.5 points.
	// 15 chosen so the importance component is the same weight as sentiment —
	// "this news matters" and "this news is bullish" deserve equal voice in
	// the trend score.
	trendImportanceSaturation = 100.0
	trendImportanceMaxPoints  = 15.0

	// sentimentBullThreshold / sentimentBearThreshold: VADER convention.
	// Anything in [-0.2, +0.2] is "neutral" — boundary inclusive of neutral.
	sentimentBullThreshold = 0.2
	sentimentBearThreshold = -0.2

	// confVolumeSaturation: mentions at which the volume confidence
	// component saturates. 100 mentions = enough sample to trust the count.
	confVolumeSaturation = 100.0
	// confVolumeMaxPoints: max points from raw volume.
	confVolumeMaxPoints = 30.0

	// confDiversityPerSource / confDiversityMaxPoints: 3 sources caps the
	// diversity component at 45. One-source narratives are easier to spoof.
	confDiversityPerSource = 15.0
	confDiversityMaxPoints = 45.0

	// confConsistencyMaxPoints: a sentiment std-dev of 0 means every
	// mention agrees on the tone — high data quality. std-dev >= 1.0
	// (the theoretical max for [-1, +1] data) zeroes this component.
	confConsistencyMaxPoints = 25.0
)

// ComputeGrowthPct returns the percentage change from prev to curr,
// capped at ±growthCap. Special cases:
//   - prev == 0 && curr > 0: returns growthInfinite (treat new narrative
//     as "infinite" growth, capped). This avoids division by zero AND
//     keeps the value bounded for storage in REAL.
//   - prev == 0 && curr == 0: returns 0 (no signal either way).
//
// The formula uses max(prev, 1) in the denominator only when prev > 0;
// the prev==0 case is handled separately above.
func ComputeGrowthPct(curr, prev int) float64 {
	if prev == 0 {
		if curr > 0 {
			return growthInfinite
		}
		return 0.0
	}
	pct := float64(curr-prev) / float64(prev) * 100.0
	if pct > growthCap {
		return growthCap
	}
	if pct < -growthCap {
		return -growthCap
	}
	return pct
}

// ClassifyStage assigns a Stage based on absolute 24h mention volume and
// 24h-vs-prior-24h growth percentage.
//
// Order matters: declining wins over volume thresholds (a high-volume
// narrative bleeding mentions is declining, not mainstream). Early
// requires BOTH low volume AND high growth — high-growth + high-volume
// is "trending", not "early".
func ClassifyStage(mentions24h int, growthPct float64) Stage {
	// Rule 1: declining — negative growth past the noise floor wins
	// regardless of how high the absolute volume is.
	if growthPct < stageDecliningGrowthMax {
		return StageDeclining
	}
	// Rule 2: early — low volume + explosive growth.
	if mentions24h < stageEarlyMaxMentions && growthPct > stageEarlyMinGrowth {
		return StageEarly
	}
	// Rule 3: mainstream — high volume + flat-ish growth band.
	if mentions24h >= stageMainstreamMinMentions &&
		growthPct >= stageMainstreamGrowthMin &&
		growthPct <= stageMainstreamGrowthMax {
		return StageMainstream
	}
	// Rule 4: trending — either high growth OR enough volume to matter.
	if growthPct > stageTrendingMinGrowth || mentions24h >= stageTrendingMinMentions {
		return StageTrending
	}
	// Fallback: low volume, low (but non-negative) growth. Conservatively
	// label as early — these are nascent topics that may or may not develop.
	return StageEarly
}

// ComputeTrendScore returns a 0..100 score combining FIVE weighted
// components (v2):
//
//	growth     35 pts
//	volume     25 pts
//	sentiment  15 pts
//	importance 15 pts  ← new in v2 (LLM-graded market impact)
//	diversity  10 pts
//	──────────────────
//	total     100 pts
//
// Negative growth and negative sentiment each contribute 0 — we don't want
// a "high" trend score for a bleeding, bearish narrative. Importance is
// always positive (the LLM scores [0, 100]) so it always contributes.
//
// avgImportance is the mean of importance_score across mentions in the
// 24h window, in [0, 100]. Pass 0 if no mentions are scored yet — the
// component contributes 0 points and the score behaves as v1 with a
// 5-point reduction in growth, 5 in volume, 5 in diversity (rebalanced
// to make room for the new component).
//
// Defensive clamps on inputs: NaN inputs are coerced to 0, mention/
// diversity counts to 0, sentiment to [0, 1] (well-formed sentiment
// already lives in [-1, +1] but a malformed DB row could deliver 5.0
// and silently blow the 15-point sentiment ceiling — final clamp
// would still cap at 100, but we want each component contract honest).
// avgImportance is clamped to [0, 100] for the same reason.
func ComputeTrendScore(mentions24h int, growthPct float64, sentimentScore float64, sourceDiversity int, avgImportance float64) int {
	if math.IsNaN(growthPct) {
		growthPct = 0
	}
	if math.IsNaN(sentimentScore) {
		sentimentScore = 0
	}
	if math.IsNaN(avgImportance) {
		avgImportance = 0
	}
	if mentions24h < 0 {
		mentions24h = 0
	}
	if sourceDiversity < 0 {
		sourceDiversity = 0
	}

	// Growth: clamp negative to 0, saturate at trendGrowthSaturation.
	g := growthPct
	if g < 0 {
		g = 0
	}
	growthPoints := math.Min(g/trendGrowthSaturation*trendGrowthMaxPoints, trendGrowthMaxPoints)

	// Volume: saturates at trendVolumeSaturation mentions/24h.
	volumePoints := math.Min(float64(mentions24h)/trendVolumeSaturation*trendVolumeMaxPoints, trendVolumeMaxPoints)

	// Sentiment: only positive sentiment counts; cap at +1.0 so a
	// malformed input can't silently overflow the 15-pt ceiling.
	s := sentimentScore
	if s < 0 {
		s = 0
	}
	if s > 1 {
		s = 1
	}
	sentimentPoints := s * trendSentimentMaxPoints

	// Diversity: linear up to the cap.
	diversityPoints := math.Min(float64(sourceDiversity)*trendDiversityPerSource, trendDiversityMaxPoints)

	// Importance: avg LLM score in [0, 100], clamped + linearly scaled to
	// the 15-pt ceiling. min(avg/100*15, 15).
	imp := avgImportance
	if imp < 0 {
		imp = 0
	}
	if imp > trendImportanceSaturation {
		imp = trendImportanceSaturation
	}
	importancePoints := math.Min(imp/trendImportanceSaturation*trendImportanceMaxPoints, trendImportanceMaxPoints)

	total := growthPoints + volumePoints + sentimentPoints + diversityPoints + importancePoints
	return clampInt(int(math.Round(total)), 0, 100)
}

// LabelSentiment buckets a numeric sentiment score into bull / neutral / bear
// using the VADER convention: thresholds at ±0.2, with the boundary
// belonging to neutral (i.e. exactly +0.2 or -0.2 is neutral, not bull/bear).
// NaN is treated as neutral.
func LabelSentiment(score float64) SentimentLabel {
	if math.IsNaN(score) {
		return SentNeutral
	}
	if score > sentimentBullThreshold {
		return SentBull
	}
	if score < sentimentBearThreshold {
		return SentBear
	}
	return SentNeutral
}

// ComputeConfidence returns a 0..100 data-quality score (NOT a prediction-
// strength score). Three components:
//   - volume points (max 30): how many mentions back the snapshot.
//   - diversity points (max 45): how many distinct sources contributed.
//     1 source = 15, 2 = 30, 3+ = 45 (clamped).
//   - consistency points (max 25): inverse of sentiment std-dev across
//     mentions. std-dev of 0 = all mentions agree = full points; std-dev
//     of >=1.0 (the theoretical max for [-1, +1] data) = 0 points.
func ComputeConfidence(mentionCount int, sourceDiversity int, sentimentStdDev float64) int {
	if mentionCount < 0 {
		mentionCount = 0
	}
	if sourceDiversity < 0 {
		sourceDiversity = 0
	}
	if math.IsNaN(sentimentStdDev) {
		sentimentStdDev = 1.0 // worst case → 0 consistency points
	}

	volumePoints := math.Min(float64(mentionCount)/confVolumeSaturation*confVolumeMaxPoints, confVolumeMaxPoints)

	diversityPoints := math.Min(float64(sourceDiversity)*confDiversityPerSource, confDiversityMaxPoints)

	// Clamp std-dev into [0, 1] before inverting — a caller passing a
	// nonsense value of 5.0 shouldn't drive consistency negative.
	sd := sentimentStdDev
	if sd < 0 {
		sd = 0
	}
	if sd > 1 {
		sd = 1
	}
	consistencyPoints := (1.0 - sd) * confConsistencyMaxPoints

	total := volumePoints + diversityPoints + consistencyPoints
	return clampInt(int(math.Round(total)), 0, 100)
}

// clampInt is a tiny helper kept private to the package — the stdlib
// generic min/max aren't used because the 0/100 bound here is semantic,
// not just numeric, and a named function reads more clearly at the
// call-sites above.
func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
