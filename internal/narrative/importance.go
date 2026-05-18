package narrative

// importance.go — interfaces and shared types for the LLM importance
// classifier. The two implementations live in importance_rule.go (deterministic
// keyword fallback) and importance_haiku.go (Anthropic Messages API). The
// CachedClassifier wrapper that persists results to narrative_mentions lives
// in importance_cached.go.
//
// VERSION: v2 of the Narrative Radar — replaces the four-component weighted
// trend score with a five-component one where the new component is "average
// LLM-graded importance across mentions in the 24h window". The 0..100
// importance score is bounded so it fits cleanly into a SMALLINT column.

import (
	"context"
	"errors"
	"fmt"
)

// Bounded category vocabulary. Anything the LLM returns outside this set
// falls through to CategoryOther in the parse path. The constants match the
// importance_category VARCHAR(32) column — kept short so a fixed-width column
// is comfortable.
const (
	CategoryMacro       = "macro"        // Fed, ECB, BoJ, rates, inflation, GDP
	CategoryGeopolitics = "geopolitics"  // wars, sanctions, presidents, elections
	CategoryRegulation  = "regulation"   // SEC, ETF approvals, lawsuits
	CategoryExchange    = "exchange"     // hacks, bankruptcies, listings
	CategoryPartnership = "partnership"  // collabs, integrations
	CategoryAltcoin     = "altcoin"      // small token releases
	CategoryOther       = "other"
)

// validCategories is the lookup set used by normalizeCategory. Kept as a
// package-private map so additions to the vocabulary go in this file (and
// not silently in callers).
var validCategories = map[string]bool{
	CategoryMacro:       true,
	CategoryGeopolitics: true,
	CategoryRegulation:  true,
	CategoryExchange:    true,
	CategoryPartnership: true,
	CategoryAltcoin:     true,
	CategoryOther:       true,
}

// Importance is the score-and-rationale tuple returned by every classifier.
// Score is clamped to [0, 100] by every implementation. Category is one of
// the package-level Category* constants (any unknown value normalised to
// CategoryOther). Why is a one-sentence rationale (LLM output) or a short
// rule name (rule-based fallback).
type Importance struct {
	Score    int    `json:"score"`
	Category string `json:"category"`
	Why      string `json:"why"`
}

// ImportanceClassifier scores a news item's market impact in [0, 100].
// Higher = more market-moving. Examples (from the system prompt):
//
//	"Trump declares national emergency"   ≈ 95
//	"Fed cuts rates 50bp"                 ≈ 85
//	"UK CPI release matches expectations" ≈ 35
//	"Small altcoin lists on KuCoin"       ≈ 15
//
// The url parameter is included in the signature because the CachedClassifier
// wrapper uses it as the cache key (deduping classifies across narratives —
// the same article matching multiple slugs gets the same score from one
// LLM call).
//
// Implementations MUST be safe under concurrent calls and MUST tolerate ctx
// cancellation. They SHOULD return an error rather than a zero Importance
// when the upstream call fails, so the caller can decide whether to fall
// back to a different implementation or skip the row.
type ImportanceClassifier interface {
	Classify(ctx context.Context, title, summary, url string) (Importance, error)
}

// errEmptyInput signals that every meaningful input field is blank. We treat
// this as "score 0, other category" — there's nothing for the LLM to grade
// and we don't want to burn an API call on an empty row.
var errEmptyInput = errors.New("narrative.classifier: empty title and summary")

// clampScore enforces the [0, 100] contract on Score. Used by every
// implementation as the last step before returning. The function takes int
// (not float) because both the rule-based and LLM-parsed scores arrive as
// ints — keeping the wire format integer-only avoids a class of "what does
// 73.6 round to" debates.
func clampScore(s int) int {
	if s < 0 {
		return 0
	}
	if s > 100 {
		return 100
	}
	return s
}

// normalizeCategory canonicalises a free-form category string into one of
// the package-level CategoryXxx constants. Unknown values fall through to
// CategoryOther. Case-sensitive on purpose — the LLM is instructed to reply
// with exactly the lowercase vocabulary, and a case-insensitive match would
// hide prompt-drift bugs.
func normalizeCategory(c string) string {
	if validCategories[c] {
		return c
	}
	return CategoryOther
}

// validateInputs returns errEmptyInput if both title and summary are blank
// after trimming. Callers should treat this as a soft failure (return Score=0
// without calling the upstream LLM).
func validateInputs(title, summary string) error {
	if isBlank(title) && isBlank(summary) {
		return errEmptyInput
	}
	return nil
}

// isBlank: every byte is whitespace or string is empty. Avoids depending on
// strings.TrimSpace + len comparison so a malformed UTF-8 input doesn't
// surprise us — we treat anything that's not ASCII whitespace as non-blank.
func isBlank(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			return false
		}
	}
	return true
}

// emptyImportance is the canonical zero result returned when validateInputs
// fails. Category=other so downstream queries don't have to special-case
// the empty/NULL branch.
func emptyImportance() Importance {
	return Importance{Score: 0, Category: CategoryOther, Why: "empty input"}
}

// wrapClassifierErr is a tiny helper to keep error messages consistent
// across implementations. Unused by the interface itself but kept here so
// new implementations follow the same pattern.
func wrapClassifierErr(impl, op string, err error) error {
	return fmt.Errorf("narrative.%s: %s: %w", impl, op, err)
}
