package narrative

// store.go — DB access layer for narrative_mentions and narrative_snapshots
// (migration 009). Uses *pgxpool.Pool with parameterized queries throughout
// — no string concatenation. JSONB columns are encoded via encoding/json and
// passed as []byte with a ::jsonb cast in SQL.
//
// /* TODO(integration tests): the methods below are not unit-testable without
//    a live Postgres. The intended approach is testcontainers-go: spin up a
//    pg image per test package, run migration 009, then exercise the round-
//    trip (Insert → Count → Latest). Skipped here on purpose — adding the
//    testcontainers-go dependency is out of scope for the "stdlib + pgx
//    only" rule for this patch. The pure-function paths (classifier,
//    scorer) cover everything that doesn't require a DB. */

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Store wraps a pgx pool with the narrative-radar query surface. The Pool
// field is exported so the worker can hand it to other layers (e.g. a future
// metrics endpoint) without re-wrapping.
type Store struct {
	Pool *pgxpool.Pool
}

// NewStore is a small constructor that mostly exists for symmetry with other
// packages and for nil-checking before any query runs. Callers MAY also do
// `&Store{Pool: pool}` directly — both are valid.
func NewStore(pool *pgxpool.Pool) *Store {
	return &Store{Pool: pool}
}

// InsertMention upserts one row into narrative_mentions. The composite
// uniqueness constraint (narrative, url) means re-ingesting the same article
// for the same narrative is a no-op (ON CONFLICT DO NOTHING). The bool
// returns true if a NEW row was inserted, false if the constraint blocked it.
//
// We rely on `RETURNING id` + pgx.ErrNoRows to detect the no-op path — the
// scenario package uses the same pattern in InsertAlert, so this is the
// established convention in this codebase.
func (s *Store) InsertMention(ctx context.Context, m Mention) (bool, error) {
	if s == nil || s.Pool == nil {
		return false, errors.New("narrative.Store: nil pool")
	}
	// Empty tickers should hit PG as `{}`, not NULL — the column is NOT NULL
	// with default '{}'. pgx serialises a nil []string the same as []string{},
	// so this is just defensive on the Go side.
	tickers := m.Tickers
	if tickers == nil {
		tickers = []string{}
	}

	var id int64
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO narrative_mentions (
			narrative, source, url, title, posted_at, sentiment, tickers
		) VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT ON CONSTRAINT narrative_mentions_uq DO NOTHING
		RETURNING id
	`,
		m.Narrative, string(m.Source), m.URL, m.Title, m.PostedAt, m.Sentiment, tickers,
	).Scan(&id)
	if err != nil {
		// pgx.ErrNoRows on RETURNING after ON CONFLICT DO NOTHING means the
		// row was a duplicate and DO NOTHING short-circuited. That's not a
		// real error — translate to (false, nil) for the caller.
		if errors.Is(err, pgx.ErrNoRows) {
			return false, nil
		}
		return false, fmt.Errorf("InsertMention: %w", err)
	}
	return true, nil
}

// CountMentions returns the count of rows for one narrative whose
// posted_at is in the half-open interval [since, until). The half-open
// convention matches the worker's "now-24h .. now" and "now-48h .. now-24h"
// windows — a row exactly at the boundary belongs to the LATER window.
func (s *Store) CountMentions(ctx context.Context, narrative string, since, until time.Time) (int, error) {
	if s == nil || s.Pool == nil {
		return 0, errors.New("narrative.Store: nil pool")
	}
	var n int
	err := s.Pool.QueryRow(ctx, `
		SELECT COUNT(*)::int
		FROM narrative_mentions
		WHERE narrative = $1
		  AND posted_at >= $2
		  AND posted_at < $3
	`, narrative, since, until).Scan(&n)
	if err != nil {
		return 0, fmt.Errorf("CountMentions: %w", err)
	}
	return n, nil
}

// MentionsSince returns every mention for a narrative posted at or after
// `since`, sorted by posted_at DESC (newest first). Used by the worker to
// pull the rows it needs to compute mean sentiment + std-dev. We return the
// full Mention struct rather than just the sentiment so the same query can
// be reused by a debug endpoint later without an extra round-trip.
func (s *Store) MentionsSince(ctx context.Context, narrative string, since time.Time) ([]Mention, error) {
	if s == nil || s.Pool == nil {
		return nil, errors.New("narrative.Store: nil pool")
	}
	rows, err := s.Pool.Query(ctx, `
		SELECT narrative, source, url, title, posted_at, sentiment, tickers
		FROM narrative_mentions
		WHERE narrative = $1 AND posted_at >= $2
		ORDER BY posted_at DESC
	`, narrative, since)
	if err != nil {
		return nil, fmt.Errorf("MentionsSince: %w", err)
	}
	defer rows.Close()

	out := []Mention{}
	for rows.Next() {
		var m Mention
		var sourceStr string
		if err := rows.Scan(
			&m.Narrative, &sourceStr, &m.URL, &m.Title,
			&m.PostedAt, &m.Sentiment, &m.Tickers,
		); err != nil {
			return nil, fmt.Errorf("MentionsSince: scan: %w", err)
		}
		m.Source = Source(sourceStr)
		if m.Tickers == nil {
			m.Tickers = []string{} // contract: never return nil slice for tickers
		}
		out = append(out, m)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("MentionsSince: rows: %w", err)
	}
	return out, nil
}

// MentionsByNarrative returns up to `limit` recent click-target mentions for
// one narrative, posted at or after `since`, sorted by posted_at DESC.
// Designed for the public GET /api/v1/narratives/:slug/mentions endpoint —
// returns a slim MentionRef (title/url/source/posted_at/importance_score)
// rather than the full Mention so the on-wire payload stays small.
//
// Filter rationale:
//   - We use a WHERE on minImportance (with `IS NULL OR ... >= ...`) so rows
//     that haven't been classified yet (importance_score IS NULL) STILL show
//     up. This is a deliberate fallback for fresh ingests — a brand-new
//     mention shouldn't be invisible just because the importance worker
//     hasn't run yet. Callers passing minImportance<=0 disable the filter
//     entirely (return all rows in the window).
//
// Limit is clamped to [1, 100] by the caller; we don't re-clamp here.
func (s *Store) MentionsByNarrative(ctx context.Context, narrative string, since time.Time, limit int, minImportance int) ([]MentionRef, error) {
	if s == nil || s.Pool == nil {
		return nil, errors.New("narrative.Store: nil pool")
	}
	if limit <= 0 {
		return []MentionRef{}, nil
	}

	// Two paths: with vs. without importance filter. Keeping the SQL literal
	// (rather than building a string) so the planner sees identical query
	// shapes between calls — the indexes on (narrative, posted_at) get reused.
	var (
		rows pgx.Rows
		err  error
	)
	if minImportance > 0 {
		rows, err = s.Pool.Query(ctx, `
			SELECT title, url, source, posted_at, importance_score
			FROM narrative_mentions
			WHERE narrative = $1
			  AND posted_at >= $2
			  AND (importance_score IS NULL OR importance_score >= $3)
			ORDER BY posted_at DESC
			LIMIT $4
		`, narrative, since, minImportance, limit)
	} else {
		rows, err = s.Pool.Query(ctx, `
			SELECT title, url, source, posted_at, importance_score
			FROM narrative_mentions
			WHERE narrative = $1
			  AND posted_at >= $2
			ORDER BY posted_at DESC
			LIMIT $3
		`, narrative, since, limit)
	}
	if err != nil {
		return nil, fmt.Errorf("MentionsByNarrative: %w", err)
	}
	defer rows.Close()

	out := make([]MentionRef, 0, limit)
	for rows.Next() {
		var m MentionRef
		// importance_score is SMALLINT NULL — scan into *int16 so NULL stays
		// NULL, then promote to *int for the JSON contract.
		var imp *int16
		if err := rows.Scan(&m.Title, &m.URL, &m.Source, &m.PostedAt, &imp); err != nil {
			return nil, fmt.Errorf("MentionsByNarrative: scan: %w", err)
		}
		if imp != nil {
			v := int(*imp)
			m.ImportanceScore = &v
		}
		out = append(out, m)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("MentionsByNarrative: rows: %w", err)
	}
	return out, nil
}

// SourceBreakdown returns map[source]count for one narrative in [since, until).
// Used by the worker to (a) populate sources_breakdown JSONB and (b) compute
// source diversity (= len(map)). Returns an empty (non-nil) map when there
// are no rows.
func (s *Store) SourceBreakdown(ctx context.Context, narrative string, since, until time.Time) (map[string]int, error) {
	if s == nil || s.Pool == nil {
		return nil, errors.New("narrative.Store: nil pool")
	}
	rows, err := s.Pool.Query(ctx, `
		SELECT source, COUNT(*)::int
		FROM narrative_mentions
		WHERE narrative = $1
		  AND posted_at >= $2
		  AND posted_at < $3
		GROUP BY source
	`, narrative, since, until)
	if err != nil {
		return nil, fmt.Errorf("SourceBreakdown: %w", err)
	}
	defer rows.Close()

	out := make(map[string]int)
	for rows.Next() {
		var src string
		var cnt int
		if err := rows.Scan(&src, &cnt); err != nil {
			return nil, fmt.Errorf("SourceBreakdown: scan: %w", err)
		}
		out[src] = cnt
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("SourceBreakdown: rows: %w", err)
	}
	return out, nil
}

// TopTickers returns the N most-mentioned tickers for a narrative in the
// half-open window [since, until). Ranking is by row count DESC, ties
// broken alphabetically ASC so the output is deterministic.
//
// Implementation note: pgx doesn't ship a clean "expand a TEXT[] column"
// operator that's portable across pg versions for our needs, so we use
// UNNEST in a CTE. The query is read-only and short-circuits to an empty
// slice when no rows exist for the window.
func (s *Store) TopTickers(ctx context.Context, narrative string, since, until time.Time, n int) ([]string, error) {
	if s == nil || s.Pool == nil {
		return nil, errors.New("narrative.Store: nil pool")
	}
	if n <= 0 {
		// Fast-path: caller asked for zero or fewer — return empty slice.
		// Avoids a needless DB round trip and a "LIMIT 0" oddity.
		return []string{}, nil
	}
	rows, err := s.Pool.Query(ctx, `
		SELECT ticker, COUNT(*)::int AS c
		FROM (
			SELECT UNNEST(tickers) AS ticker
			FROM narrative_mentions
			WHERE narrative = $1
			  AND posted_at >= $2
			  AND posted_at < $3
		) t
		GROUP BY ticker
		ORDER BY c DESC, ticker ASC
		LIMIT $4
	`, narrative, since, until, n)
	if err != nil {
		return nil, fmt.Errorf("TopTickers: %w", err)
	}
	defer rows.Close()

	out := make([]string, 0, n)
	for rows.Next() {
		var t string
		var c int
		if err := rows.Scan(&t, &c); err != nil {
			return nil, fmt.Errorf("TopTickers: scan: %w", err)
		}
		out = append(out, t)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("TopTickers: rows: %w", err)
	}
	return out, nil
}

// UpsertSnapshot writes one snapshot row. The (narrative, captured_at)
// uniqueness constraint makes a retried tick a no-op via ON CONFLICT DO
// NOTHING — without that, two ticks colliding at the same captured_at would
// double the row and corrupt the growth_pct sparkline derived from history.
//
// Encodes sources_breakdown via encoding/json and passes the bytes with a
// ::jsonb cast in SQL.
func (s *Store) UpsertSnapshot(ctx context.Context, snap Snapshot) error {
	if s == nil || s.Pool == nil {
		return errors.New("narrative.Store: nil pool")
	}

	// JSONB encode. An empty/nil map encodes as `{}` which is exactly what
	// the column default expects.
	breakdown := snap.SourcesBreakdown
	if breakdown == nil {
		breakdown = map[string]int{}
	}
	// Stable JSON output (sorted keys) — encoding/json default. We sort
	// explicitly via a small helper so a nondeterministic map iteration
	// can't make snapshot bytes differ between identical states (helps
	// downstream diffing if anyone hashes this row).
	breakdownJSON, err := jsonEncodeStableMap(breakdown)
	if err != nil {
		return fmt.Errorf("UpsertSnapshot: jsonEncode breakdown: %w", err)
	}

	related := snap.RelatedAssets
	if related == nil {
		related = []string{}
	}
	reasons := snap.Reasons
	if reasons == nil {
		reasons = []string{}
	}

	// captured_at falls back to NOW() server-side if zero — keeps the
	// timestamp single-sourced. We pass it through pgx as a *time.Time so
	// nil → server default; a real value is honoured (lets the worker pin
	// captured_at to the tick start for consistent snapshots).
	var capturedAt any
	if snap.CapturedAt.IsZero() {
		capturedAt = nil
	} else {
		capturedAt = snap.CapturedAt
	}

	_, err = s.Pool.Exec(ctx, `
		INSERT INTO narrative_snapshots (
			narrative, captured_at, mention_count_24h, mention_count_prev_24h,
			growth_pct, trend_score, stage,
			sentiment_score, sentiment_label, confidence,
			related_assets, sources_breakdown, reasons
		) VALUES (
			$1, COALESCE($2, NOW()), $3, $4,
			$5, $6, $7,
			$8, $9, $10,
			$11, $12::jsonb, $13
		)
		ON CONFLICT ON CONSTRAINT narrative_snapshots_uq DO NOTHING
	`,
		snap.Narrative, capturedAt, snap.MentionCount24h, snap.MentionCountPrev24h,
		snap.GrowthPct, snap.TrendScore, string(snap.Stage),
		snap.SentimentScore, string(snap.SentimentLabel), snap.Confidence,
		related, breakdownJSON, reasons,
	)
	if err != nil {
		return fmt.Errorf("UpsertSnapshot: %w", err)
	}
	return nil
}

// LatestSnapshots returns the most recent snapshot per narrative, ordered by
// trend_score DESC (radar leaderboard). Uses DISTINCT ON, which is supported
// by Postgres and matches the index idx_narrative_snapshots_latest exactly.
//
// JSONB sources_breakdown is decoded into map[string]int. A bad JSONB blob
// (impossible given our UpsertSnapshot path, but a future migration could
// inject one) returns the row with an empty map rather than failing the
// whole query — that's a deliberate trade-off so the radar page can still
// render when one bad row sneaks in.
func (s *Store) LatestSnapshots(ctx context.Context) ([]Snapshot, error) {
	if s == nil || s.Pool == nil {
		return nil, errors.New("narrative.Store: nil pool")
	}
	rows, err := s.Pool.Query(ctx, `
		SELECT DISTINCT ON (narrative)
			narrative, captured_at, mention_count_24h, mention_count_prev_24h,
			growth_pct, trend_score, stage,
			sentiment_score, sentiment_label, confidence,
			related_assets, sources_breakdown, reasons
		FROM narrative_snapshots
		ORDER BY narrative, captured_at DESC
	`)
	if err != nil {
		return nil, fmt.Errorf("LatestSnapshots: %w", err)
	}
	defer rows.Close()

	out := []Snapshot{}
	for rows.Next() {
		var sn Snapshot
		var stage, sentLabel string
		var breakdownJSON []byte
		if err := rows.Scan(
			&sn.Narrative, &sn.CapturedAt,
			&sn.MentionCount24h, &sn.MentionCountPrev24h,
			&sn.GrowthPct, &sn.TrendScore, &stage,
			&sn.SentimentScore, &sentLabel, &sn.Confidence,
			&sn.RelatedAssets, &breakdownJSON, &sn.Reasons,
		); err != nil {
			return nil, fmt.Errorf("LatestSnapshots: scan: %w", err)
		}
		sn.Stage = Stage(stage)
		sn.SentimentLabel = SentimentLabel(sentLabel)
		// Decode breakdown; on parse error, leave map empty rather than
		// dropping the whole row. See docstring above for rationale.
		breakdown := map[string]int{}
		if len(breakdownJSON) > 0 {
			_ = json.Unmarshal(breakdownJSON, &breakdown)
		}
		sn.SourcesBreakdown = breakdown
		if sn.RelatedAssets == nil {
			sn.RelatedAssets = []string{}
		}
		if sn.Reasons == nil {
			sn.Reasons = []string{}
		}
		out = append(out, sn)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("LatestSnapshots: rows: %w", err)
	}

	// In-memory sort by trend_score DESC, ties broken by narrative ASC.
	// Doing it here (rather than in the SQL) keeps DISTINCT ON happy:
	// DISTINCT ON requires the leading ORDER BY column be the DISTINCT key.
	sort.Slice(out, func(i, j int) bool {
		if out[i].TrendScore != out[j].TrendScore {
			return out[i].TrendScore > out[j].TrendScore
		}
		return out[i].Narrative < out[j].Narrative
	})
	return out, nil
}

// jsonEncodeStableMap marshals map[string]int into byte-stable JSONB. Go's
// stdlib json.Marshal already sorts string-keyed maps, so we just delegate;
// this thin wrapper exists to (a) document the byte-stability requirement at
// the call-site in UpsertSnapshot, and (b) handle the empty case explicitly
// so an empty map encodes as `{}` rather than `null`.
func jsonEncodeStableMap(m map[string]int) ([]byte, error) {
	if len(m) == 0 {
		return []byte("{}"), nil
	}
	return json.Marshal(m)
}

// MentionForBackfill is the minimal shape needed for tickers re-extraction.
// Only id + title — backfill re-runs ExtractTickers on the saved title and
// updates the row in place. URL and other fields aren't needed because we
// key the UPDATE by primary id (not by url, which would over-touch siblings
// in a different narrative bucket).
type MentionForBackfill struct {
	ID    int64
	Title string
}

// PendingTickers returns up to `limit` mentions whose tickers array is empty
// (tickers = '{}'). Ordered by ingested_at ASC (oldest first — fairness, same
// convention as PendingImportance). Used by the worker's Phase 2d to backfill
// rows that pre-date the Path 3 (coin-name → ticker) addition in classifier.go.
//
// SQL note: array_length(tickers, 1) returns NULL when tickers='{}', so the
// IS NULL test is the canonical way to filter empty arrays. We rely on
// idx_narrative_mentions_recent (ingested_at DESC) for ordering, even though
// we want ASC — pgx flips the index scan direction at no cost. The empty-
// tickers subset should be small after the v2.1 fix lands so a sequential
// post-filter is fine even on the long tail.
//
// `limit` is clamped: <= 0 returns (nil, nil) without touching the DB.
//
// TODO(integration): test against real PG — this query plan is not exercised
// by any unit test. The fakeBackfillStore in the worker test path doesn't
// model array_length semantics, so we rely on runtime logs ("backfilled=N
// tickers") to confirm the SQL contract holds.
func (s *Store) PendingTickers(ctx context.Context, limit int) ([]MentionForBackfill, error) {
	if s == nil || s.Pool == nil {
		return nil, errors.New("narrative.Store: nil pool")
	}
	if limit <= 0 {
		return nil, nil
	}

	rows, err := s.Pool.Query(ctx, `
		SELECT id, title
		FROM narrative_mentions
		WHERE array_length(tickers, 1) IS NULL
		ORDER BY ingested_at ASC
		LIMIT $1
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("PendingTickers: %w", err)
	}
	defer rows.Close()

	out := make([]MentionForBackfill, 0, limit)
	for rows.Next() {
		var m MentionForBackfill
		if err := rows.Scan(&m.ID, &m.Title); err != nil {
			return nil, fmt.Errorf("PendingTickers: scan: %w", err)
		}
		out = append(out, m)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("PendingTickers: rows: %w", err)
	}
	return out, nil
}

// UpdateTickersByID updates a single row's tickers array. Targets by primary
// key (id) rather than by url so the update touches exactly one row even
// when the same article matched multiple narratives (each narrative has its
// own row with its own id, and we want each row to learn its own ticker
// extraction independently — the worker re-runs Path 3 per row).
//
// nil tickers is normalised to an empty slice so the column never sees NULL
// (matches the InsertMention contract — tickers is NOT NULL DEFAULT '{}').
//
// TODO(integration): test against real PG — see the package-level docstring
// in store.go for why we skip live-DB unit tests.
func (s *Store) UpdateTickersByID(ctx context.Context, id int64, tickers []string) error {
	if s == nil || s.Pool == nil {
		return errors.New("narrative.Store: nil pool")
	}
	if tickers == nil {
		tickers = []string{}
	}
	_, err := s.Pool.Exec(ctx, `
		UPDATE narrative_mentions
		SET tickers = $1
		WHERE id = $2
	`, tickers, id)
	if err != nil {
		return fmt.Errorf("UpdateTickersByID: %w", err)
	}
	return nil
}
