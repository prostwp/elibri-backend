package narrative

// idea.go — interfaces and shared types for the LLM-generated trade-idea
// text rendered in the Dashboard. Sister to importance.go: same shape
// (interface + Haiku impl + static fallback + cache), different prompt and
// different cache key.
//
// VERSION: v2.1 — adds a contextual one-paragraph idea per top narrative.
// Phase 3 will move the cache from in-memory to Postgres (per-snapshot row)
// so a backend restart doesn't burn the cached idea.

import (
	"context"
	"sync"
)

// TickerQuote is the minimal price snapshot the IdeaGenerator needs to
// reason about a related asset. Mirrors the api package's tickerQuote shape;
// kept here as a small DTO so the narrative package isn't reaching INTO the
// api package for its own input type. The api layer maps its richer struct
// down to this one before calling Generate.
type TickerQuote struct {
	// Symbol is the bare ticker, no exchange suffix ("JTO" not "JTOUSDT").
	Symbol string
	// Price is the last traded price in USD-quoted units.
	Price float64
	// ChangePct is the 24h percent change (e.g. +12.5 for "up 12.5%").
	ChangePct float64
}

// IdeaGenerator produces a one-paragraph trading-idea text for the top
// narrative, given the snapshot context + (optionally) the related-asset
// price snapshots. Implementations:
//
//   - HaikuIdeaGenerator: Anthropic Messages API call (production).
//   - StaticIdeaGenerator: returns a hardcoded watchlist line (no-key
//     environments / tests / Anthropic outage fallback in callers).
//
// Callers SHOULD treat any non-nil error as "use the empty-string fallback"
// — the frontend already renders a default "Add to watchlist…" line when
// the field is empty, so a Haiku outage degrades gracefully.
type IdeaGenerator interface {
	GenerateIdea(ctx context.Context, narrativeSlug string, snapshot Snapshot, prices []TickerQuote) (string, error)
}

// StaticIdeaGenerator is the no-key / no-LLM fallback. Returns a fixed
// English watchlist line — same string the frontend uses as its empty-state
// default, so plugging this in keeps the Dashboard visually stable when the
// LLM is unavailable. Safe for concurrent use (no state).
type StaticIdeaGenerator struct{}

// staticIdeaText is the one canonical fallback string. Kept in one place so
// frontend and backend defaults can't drift apart. If you change this, also
// update the JSX fallback in NarrativeDashboardNode.tsx.
const staticIdeaText = "Add to watchlist, wait for confirmation on volume"

// GenerateIdea implements IdeaGenerator. Always returns staticIdeaText, nil.
// The narrative + snapshot + prices arguments are ignored on purpose — the
// static fallback is intentionally context-free so callers can rely on
// "this never fails" semantics. ctx is honoured (returns ctx.Err() when
// cancelled) so a shutdown signal isn't masked by the static branch.
func (StaticIdeaGenerator) GenerateIdea(ctx context.Context, _ string, _ Snapshot, _ []TickerQuote) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	return staticIdeaText, nil
}

// Compile-time check: both implementations satisfy the interface. If the
// interface signature changes, the build fails here rather than at the
// call-site in the api package.
var _ IdeaGenerator = StaticIdeaGenerator{}
var _ IdeaGenerator = (*HaikuIdeaGenerator)(nil)

// IdeaCache is an in-memory store of GeneratedIdea results keyed by
// (narrative, captured_at_truncated_to_hour). The TTL is implicit in the
// key — a new captured_at hour produces a new key, so the previous hour's
// entry simply stops being looked up. We never explicitly evict; the
// hour-boundary churn keeps the map small (one entry per active narrative
// per hour, ~12 narratives × 24 hours = 288 worst case, far below any GC
// pressure).
//
// Concurrency: sync.Map is fine here — load-heavy workload (every list
// request hits the cache; only one writer per (narrative, hour) per request
// burst). The Store method exits early on duplicate writes for the same
// key, so two concurrent first-cache-misses for the same narrative+hour
// race at most once and the loser's idea is discarded.
type IdeaCache struct {
	m sync.Map // map[string]string — key = narrativeSlug+"@"+hour
}

// Get returns the cached idea for (narrative, hour). The hour is the
// captured_at truncated to the top of the hour (UTC). Returns ("", false)
// on miss.
func (c *IdeaCache) Get(narrativeSlug, hourKey string) (string, bool) {
	if c == nil {
		return "", false
	}
	v, ok := c.m.Load(narrativeSlug + "@" + hourKey)
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// Set stores the idea under (narrative, hour). Idempotent — overwriting an
// existing entry is harmless because the second compute should yield a
// nearly-identical result (same input context within the hour).
func (c *IdeaCache) Set(narrativeSlug, hourKey, idea string) {
	if c == nil {
		return
	}
	c.m.Store(narrativeSlug+"@"+hourKey, idea)
}
