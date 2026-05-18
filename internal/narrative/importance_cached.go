package narrative

// importance_cached.go — wrapper that persists classifier results to
// narrative_mentions and short-circuits a re-classify of an already-scored
// URL.
//
// Why URL-keyed caching:
//   - The same article can match multiple narratives (denormalised on
//     purpose in migration 009). Without caching, ingest fan-out would call
//     the LLM N times for one article — N times the cost, N times the
//     latency.
//   - The cache is the database itself: narrative_mentions rows for the
//     same URL share the importance_* columns. The first classify → write
//     for one URL persists the verdict; subsequent reads (whether by
//     another narrative match in the same tick or by a re-ingest of the
//     same article on a future tick) hit the cached row.
//
// Concurrency: the wrapper uses singleflight keyed by URL to coalesce
// concurrent in-flight classifies for the same URL into a single LLM call.
// Without this, two narratives matching the same article in the same tick
// would both miss the DB cache (no row written yet) and both burn an API
// call. The DB cache covers the cross-tick case (URL seen on Monday → row
// written → Tuesday's tick hits the cached row); singleflight covers the
// intra-tick case (two callers within the same millisecond).

import (
	"context"
	"errors"
	"fmt"

	"golang.org/x/sync/singleflight"
)

// importanceCacheStore is the subset of Store methods CachedClassifier
// needs. Defined as an interface so unit tests can stub the DB without
// spinning up testcontainers.
type importanceCacheStore interface {
	GetImportanceByURL(ctx context.Context, url string) (Importance, bool, error)
	UpdateImportance(ctx context.Context, url string, imp Importance) error
}

// CachedClassifier wraps an inner ImportanceClassifier with a URL-keyed
// look-aside cache backed by narrative_mentions.importance_*. If Store is
// nil, the wrapper degrades gracefully to a pass-through (still calls Inner)
// — that path matters for unit tests that don't want to plumb a Store stub.
//
// The sf field coalesces concurrent calls for the same URL into a single
// upstream classify. Lazy-initialised; zero value works.
type CachedClassifier struct {
	Inner ImportanceClassifier
	Store importanceCacheStore
	sf    singleflight.Group
}

// Classify returns the cached Importance if one exists for url; otherwise
// delegates to Inner and persists the result. Errors:
//   - Inner == nil → static error, no DB call.
//   - cache lookup error → bubbled up. We don't silently fall through to
//     Inner because a flaky cache that's silently bypassed would burn LLM
//     budget without anyone noticing.
//   - Inner.Classify error → bubbled up; no cache write.
//   - cache write error → logged via the returned error but Inner's result
//     is still returned to the caller. This is a deliberate "best-effort
//     persistence" trade-off — the radar should keep working even if the
//     PG pool is degraded.
func (c *CachedClassifier) Classify(ctx context.Context, title, summary, url string) (Importance, error) {
	if c.Inner == nil {
		return Importance{}, errors.New("narrative.CachedClassifier: Inner is nil")
	}

	// Empty URL = no cache key → fall through to Inner without a cache hit
	// or write. Worth 1 line of code to avoid pinning a wildcard cache row.
	if url == "" {
		return c.Inner.Classify(ctx, title, summary, url)
	}

	// 1. Cache lookup. Skipped when Store is unset (test pass-through path).
	if c.Store != nil {
		imp, hit, err := c.Store.GetImportanceByURL(ctx, url)
		if err != nil {
			return Importance{}, fmt.Errorf("narrative.CachedClassifier: cache lookup: %w", err)
		}
		if hit {
			return imp, nil
		}
	}

	// 2. Cache miss → delegate via singleflight. Concurrent callers for the
	//    same URL share one Inner.Classify invocation. The Do callback also
	//    persists, so all sharers see a fully-cached row after the first
	//    return (the loser of the race re-checks the DB on its next call).
	v, err, _ := c.sf.Do(url, func() (interface{}, error) {
		imp, ierr := c.Inner.Classify(ctx, title, summary, url)
		if ierr != nil {
			return nil, ierr
		}
		// Persist inside the singleflight closure so the cache is hot before
		// the next caller's Do unwraps. Skipped when Store is unset (test).
		if c.Store != nil {
			if werr := c.Store.UpdateImportance(ctx, url, imp); werr != nil {
				// Best-effort: return the result with a wrapped error so the
				// caller can log + continue without losing the LLM verdict.
				return imp, fmt.Errorf("narrative.CachedClassifier: cache write: %w", werr)
			}
		}
		return imp, nil
	})

	if v != nil {
		// On both success and "wrapped cache-write failure" the inner returns
		// the LLM verdict. A nil v happens only when Inner.Classify itself
		// failed before persist — err carries that.
		if imp, ok := v.(Importance); ok {
			return imp, err
		}
	}
	return Importance{}, err
}
