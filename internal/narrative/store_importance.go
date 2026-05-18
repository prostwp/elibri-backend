package narrative

// store_importance.go — DB methods backing the importance classifier.
// Kept in a separate file from store.go so the v1 / v2 surface is easy to
// audit independently. All methods on *Store; same nil-pool guards and
// pgx error handling as the v1 file.
//
// TODO(integration): test PendingImportance against a real Postgres. The
// fakeImportanceStore in importance_test.go doesn't model GROUP BY / ORDER BY
// semantics, so the SQL contract (oldest-first, dedup-by-URL, partial-index
// hit) is verified at runtime via the worker's Phase 2c logs rather than a
// unit test. Spinning up testcontainers-go is out of scope for this patch.

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

// pendingImportanceMaxLimit caps the upper bound of PendingImportance(limit).
// Pegged at 500 because (a) backfill is a per-tick operation and we don't want
// one tick to monopolise the Haiku quota, (b) at ~1.5s/Haiku call a 500-row
// batch would take 12.5min — well past our 5min classifyTimeout — so the
// cap also guards against a caller passing a wildly large value.
const pendingImportanceMaxLimit = 500

// PendingMention is the minimal shape needed to backfill — title and URL.
// Summary isn't stored in narrative_mentions, so backfill calls the
// classifier with summary="" — the classifier already tolerates that.
type PendingMention struct {
	URL   string
	Title string
}

// PendingImportance returns up to `limit` mentions whose importance_score is
// NULL, ordered by ingested_at ASC (oldest first — fairness). Deduped by URL
// since the same article can match multiple narratives → multiple rows with
// the same URL → we only want to classify it once.
//
// `limit` is clamped to (0, pendingImportanceMaxLimit] — values <= 0 return
// (nil, nil) without touching the DB; values above the cap are pinned to it.
//
// SQL:
//
//	SELECT url, MIN(title)
//	FROM narrative_mentions
//	WHERE importance_score IS NULL
//	GROUP BY url
//	ORDER BY MIN(ingested_at) ASC
//	LIMIT $1
//
// The query plan should hit idx_narrative_mentions_pending_importance (the
// partial index from migration 010) for the WHERE filter; the GROUP BY +
// MIN() aggregation runs over the small filtered set. EXPLAIN should report
// "Index Scan using idx_narrative_mentions_pending_importance" or a HashAgg
// node fed by it.
//
// MIN(title) is a deterministic tie-breaker when the same URL has multiple
// rows with different titles (rare — happens when an upstream source rewrites
// the headline between fetches). Either title is fine for classification;
// MIN gives reproducible output for testability.
func (s *Store) PendingImportance(ctx context.Context, limit int) ([]PendingMention, error) {
	if s == nil || s.Pool == nil {
		return nil, errors.New("narrative.Store: nil pool")
	}
	if limit <= 0 {
		return nil, nil
	}
	if limit > pendingImportanceMaxLimit {
		limit = pendingImportanceMaxLimit
	}

	rows, err := s.Pool.Query(ctx, `
		SELECT url, MIN(title) AS title
		FROM narrative_mentions
		WHERE importance_score IS NULL
		GROUP BY url
		ORDER BY MIN(ingested_at) ASC
		LIMIT $1
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("PendingImportance: %w", err)
	}
	defer rows.Close()

	out := make([]PendingMention, 0, limit)
	for rows.Next() {
		var p PendingMention
		if err := rows.Scan(&p.URL, &p.Title); err != nil {
			return nil, fmt.Errorf("PendingImportance: scan: %w", err)
		}
		out = append(out, p)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("PendingImportance: rows: %w", err)
	}
	return out, nil
}

// GetImportanceByURL looks up the cached importance for one URL. Returns
// (imp, true, nil) on a hit, (zero, false, nil) on a miss, (zero, false, err)
// on a real error.
//
// "Hit" means importance_score IS NOT NULL — the row exists in
// narrative_mentions but has been scored. A row that exists but is unscored
// (importance_score NULL) is a miss; the caller should re-classify.
//
// We pull the first row with this URL; the schema's narrative_mentions_uq
// constraint is on (narrative, url), not url alone, so multiple rows may
// share a URL across narratives. They all carry the same importance_*
// columns once UpdateImportance has run, so reading the first one is fine.
func (s *Store) GetImportanceByURL(ctx context.Context, url string) (Importance, bool, error) {
	if s == nil || s.Pool == nil {
		return Importance{}, false, errors.New("narrative.Store: nil pool")
	}
	if url == "" {
		return Importance{}, false, errors.New("narrative.Store.GetImportanceByURL: empty url")
	}

	var (
		score    *int16
		category *string
		why      *string
	)
	err := s.Pool.QueryRow(ctx, `
		SELECT importance_score, importance_category, importance_why
		FROM narrative_mentions
		WHERE url = $1
		  AND importance_score IS NOT NULL
		LIMIT 1
	`, url).Scan(&score, &category, &why)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return Importance{}, false, nil
		}
		return Importance{}, false, fmt.Errorf("GetImportanceByURL: %w", err)
	}

	imp := Importance{
		Score: 0,
		// CachedClassifier expects normalised category, but we trust the
		// write path (UpdateImportance always writes a normalised value).
		// Re-normalising on read is cheap defence against a future migration
		// that injects bad rows.
		Category: CategoryOther,
		Why:      "",
	}
	if score != nil {
		imp.Score = clampScore(int(*score))
	}
	if category != nil {
		imp.Category = normalizeCategory(*category)
	}
	if why != nil {
		imp.Why = *why
	}
	return imp, true, nil
}

// UpdateImportance writes a classification result to ALL rows in
// narrative_mentions sharing this URL. The same article may match multiple
// narratives — we want every row to carry the same score so the radar's
// "average importance per narrative" metric is consistent regardless of
// which narrative the row was inserted under first.
//
// Idempotent: re-running with the same Importance is a no-op (UPDATE on
// already-equal columns is a write but doesn't change semantics).
//
// Returns nil even when no rows match — that's a soft case (the URL was
// classified before any mention row existed, e.g. a test stub). Callers who
// care can wrap and check the affected-row count themselves.
func (s *Store) UpdateImportance(ctx context.Context, url string, imp Importance) error {
	if s == nil || s.Pool == nil {
		return errors.New("narrative.Store: nil pool")
	}
	if url == "" {
		return errors.New("narrative.Store.UpdateImportance: empty url")
	}

	// Defensive normalisation on the write path so a buggy classifier can't
	// inject e.g. score=200 or category="garbage".
	score := clampScore(imp.Score)
	cat := normalizeCategory(imp.Category)

	_, err := s.Pool.Exec(ctx, `
		UPDATE narrative_mentions
		SET importance_score        = $1,
		    importance_category     = $2,
		    importance_why          = $3,
		    importance_classified_at = $4
		WHERE url = $5
	`, int16(score), cat, imp.Why, time.Now(), url)
	if err != nil {
		return fmt.Errorf("UpdateImportance: %w", err)
	}
	return nil
}

// AvgImportance returns the mean importance_score across mentions for one
// narrative in the half-open window [since, until). Rows where
// importance_score IS NULL are excluded — they haven't been scored yet, so
// including them would skew the average toward 0 unfairly.
//
// Returns 0 when no mentions are scored in the window. Callers shouldn't
// distinguish "0 score" from "no data" — both legitimately translate to
// "this narrative gets 0 importance points in the trend score".
//
// pgx returns AVG(SMALLINT) as float64 via a NULL-able numeric scan path.
// We use COALESCE in SQL to fold the NULL case to 0 so the Go side sees a
// plain float64.
func (s *Store) AvgImportance(ctx context.Context, narrative string, since, until time.Time) (float64, error) {
	if s == nil || s.Pool == nil {
		return 0, errors.New("narrative.Store: nil pool")
	}
	var avg float64
	err := s.Pool.QueryRow(ctx, `
		SELECT COALESCE(AVG(importance_score)::float8, 0.0)
		FROM narrative_mentions
		WHERE narrative = $1
		  AND posted_at >= $2
		  AND posted_at < $3
		  AND importance_score IS NOT NULL
	`, narrative, since, until).Scan(&avg)
	if err != nil {
		return 0, fmt.Errorf("AvgImportance: %w", err)
	}
	return avg, nil
}
