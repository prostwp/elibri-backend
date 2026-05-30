package whale

// store.go — DB access layer for whale_transfers and whale_snapshots
// (migration 013). Uses *pgxpool.Pool with parameterized queries throughout
// — no string concatenation. JSONB columns are encoded via encoding/json and
// passed as []byte with a ::jsonb cast in SQL. Mirrors narrative.Store idioms:
// nil-guards, ON CONFLICT DO NOTHING + RETURNING id + pgx.ErrNoRows→(false,nil),
// in-Go sort after DISTINCT ON.
//
// /* TODO(integration tests): the DB methods here are not unit-testable
//    without a live Postgres (testcontainers-go). The pure-function paths
//    (scorer) cover everything that doesn't require a DB. */

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Store wraps a pgx pool with the whale-flow query surface. The Pool field is
// exported so the worker can hand it to other layers without re-wrapping.
type Store struct {
	Pool *pgxpool.Pool
}

// NewStore is a small constructor that mostly exists for symmetry with the
// narrative package and for nil-checking before any query runs. Callers MAY
// also do `&Store{Pool: pool}` directly — both are valid.
func NewStore(pool *pgxpool.Pool) *Store {
	return &Store{Pool: pool}
}

// InsertTransfer upserts one row into whale_transfers. The composite
// uniqueness constraint (chain, tx_hash, to_addr) makes re-ingesting the same
// transfer a no-op (ON CONFLICT DO NOTHING). Returns true if a NEW row was
// inserted, false if the constraint blocked it.
//
// Uses `RETURNING id` + pgx.ErrNoRows to detect the no-op path — the same
// convention narrative.InsertMention uses.
func (s *Store) InsertTransfer(ctx context.Context, t WhaleTransfer) (bool, error) {
	if s == nil || s.Pool == nil {
		return false, errors.New("whale.Store: nil pool")
	}

	var id int64
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO whale_transfers (
			chain, asset, exchange, tx_hash, from_addr, to_addr,
			amount_native, amount_usd, direction, transferred_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		ON CONFLICT ON CONSTRAINT whale_transfers_uq DO NOTHING
		RETURNING id
	`,
		string(t.Chain), t.Asset, t.Exchange, t.TxHash, t.FromAddr, t.ToAddr,
		t.AmountNative, t.AmountUSD, string(t.Direction), t.Timestamp,
	).Scan(&id)
	if err != nil {
		// pgx.ErrNoRows on RETURNING after ON CONFLICT DO NOTHING means the
		// row was a duplicate — translate to (false, nil) for the caller.
		if errors.Is(err, pgx.ErrNoRows) {
			return false, nil
		}
		return false, fmt.Errorf("InsertTransfer: %w", err)
	}
	return true, nil
}

// RecentTransfers returns up to `limit` transfers, newest-first
// (ORDER BY transferred_at DESC). Powers the live feed + flow map. `limit`
// <= 0 returns an empty slice without touching the DB.
func (s *Store) RecentTransfers(ctx context.Context, limit int) ([]WhaleTransfer, error) {
	if s == nil || s.Pool == nil {
		return nil, errors.New("whale.Store: nil pool")
	}
	if limit <= 0 {
		return []WhaleTransfer{}, nil
	}
	rows, err := s.Pool.Query(ctx, `
		SELECT chain, tx_hash, transferred_at, from_addr, to_addr,
		       asset, amount_native, amount_usd, direction, exchange
		FROM whale_transfers
		ORDER BY transferred_at DESC
		LIMIT $1
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("RecentTransfers: %w", err)
	}
	defer rows.Close()

	out := make([]WhaleTransfer, 0, limit)
	for rows.Next() {
		var t WhaleTransfer
		var chainStr, dirStr string
		if err := rows.Scan(
			&chainStr, &t.TxHash, &t.Timestamp, &t.FromAddr, &t.ToAddr,
			&t.Asset, &t.AmountNative, &t.AmountUSD, &dirStr, &t.Exchange,
		); err != nil {
			return nil, fmt.Errorf("RecentTransfers: scan: %w", err)
		}
		t.Chain = Chain(chainStr)
		t.Direction = FlowDirection(dirStr)
		out = append(out, t)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("RecentTransfers: rows: %w", err)
	}
	return out, nil
}

// InflowOutflowSince aggregates transfers for one asset in the half-open
// window [since, until):
//   - inUSD:  sum of amount_usd where direction='inflow'
//   - outUSD: sum of amount_usd where direction='outflow'
//   - n:      total tx count (inflow + outflow; neutrals excluded from the
//     directional sums but they don't contribute to a side anyway)
//   - byExch: map[exchange]count over inflow+outflow rows with a non-empty
//     exchange label
//
// Half-open [since, until) matches the worker's 24h / prev-24h windows so a
// row exactly at the boundary belongs to the LATER window (no double-count).
// Neutral-direction rows (BTC live feed) are counted in `n` only if they're
// in the directional set — here we deliberately count only inflow/outflow for
// the directional math, but include ALL rows in `n` so tx_count_24h reflects
// total observed activity for the asset.
func (s *Store) InflowOutflowSince(ctx context.Context, asset string, since, until time.Time) (inUSD, outUSD float64, n int, byExch map[string]int, err error) {
	if s == nil || s.Pool == nil {
		return 0, 0, 0, nil, errors.New("whale.Store: nil pool")
	}
	byExch = map[string]int{}

	rows, qerr := s.Pool.Query(ctx, `
		SELECT direction, amount_usd, exchange
		FROM whale_transfers
		WHERE asset = $1
		  AND transferred_at >= $2
		  AND transferred_at < $3
	`, asset, since, until)
	if qerr != nil {
		return 0, 0, 0, byExch, fmt.Errorf("InflowOutflowSince: %w", qerr)
	}
	defer rows.Close()

	for rows.Next() {
		var dir, exch string
		var amtUSD float64
		if scanErr := rows.Scan(&dir, &amtUSD, &exch); scanErr != nil {
			return 0, 0, 0, byExch, fmt.Errorf("InflowOutflowSince: scan: %w", scanErr)
		}
		n++
		switch FlowDirection(dir) {
		case FlowInflow:
			inUSD += amtUSD
		case FlowOutflow:
			outUSD += amtUSD
		}
		// Count exchange attribution for inflow/outflow rows only — neutral
		// (BTC) rows carry no exchange label, so this naturally skips them.
		if exch != "" {
			byExch[exch]++
		}
	}
	if rowsErr := rows.Err(); rowsErr != nil {
		return 0, 0, 0, byExch, fmt.Errorf("InflowOutflowSince: rows: %w", rowsErr)
	}
	return inUSD, outUSD, n, byExch, nil
}

// UpsertSnapshot writes one snapshot row. The (asset, captured_at) uniqueness
// constraint makes a retried tick a no-op via ON CONFLICT DO NOTHING. Encodes
// exchange_breakdown via encoding/json and passes the bytes with a ::jsonb
// cast. captured_at falls back to NOW() server-side when zero, keeping the
// timestamp single-sourced (mirrors narrative.UpsertSnapshot).
func (s *Store) UpsertSnapshot(ctx context.Context, snap Snapshot) error {
	if s == nil || s.Pool == nil {
		return errors.New("whale.Store: nil pool")
	}

	breakdown := snap.ExchangeBreakdown
	if breakdown == nil {
		breakdown = map[string]int{}
	}
	breakdownJSON, err := jsonEncodeStableMap(breakdown)
	if err != nil {
		return fmt.Errorf("UpsertSnapshot: jsonEncode breakdown: %w", err)
	}

	reasons := snap.Reasons
	if reasons == nil {
		reasons = []string{}
	}

	source := snap.Source
	if source == "" {
		source = "etherscan" // matches column DEFAULT
	}

	// captured_at → NOW() server-side if zero; a real value is honoured (lets
	// the worker pin captured_at to the tick start for consistent snapshots).
	var capturedAt any
	if snap.CapturedAt.IsZero() {
		capturedAt = nil
	} else {
		capturedAt = snap.CapturedAt
	}

	_, err = s.Pool.Exec(ctx, `
		INSERT INTO whale_snapshots (
			asset, captured_at, net_flow_usd_24h, net_flow_prev_24h,
			flow_pct, direction, inflow_usd_24h, outflow_usd_24h,
			tx_count_24h, confidence, exchange_breakdown, reasons,
			is_new_spike, source, partial
		) VALUES (
			$1, COALESCE($2, NOW()), $3, $4,
			$5, $6, $7, $8,
			$9, $10, $11::jsonb, $12,
			$13, $14, $15
		)
		ON CONFLICT ON CONSTRAINT whale_snapshots_uq DO NOTHING
	`,
		snap.Asset, capturedAt, snap.NetFlowUSD24h, snap.NetFlowPrev24h,
		snap.FlowPct, string(snap.Direction), snap.InflowUSD24h, snap.OutflowUSD24h,
		snap.TxCount24h, snap.Confidence, breakdownJSON, reasons,
		snap.IsNewSpike, source, snap.Partial,
	)
	if err != nil {
		return fmt.Errorf("UpsertSnapshot: %w", err)
	}
	return nil
}

// LatestSnapshots returns the most recent snapshot per asset, sorted by
// abs(net_flow_usd_24h) DESC in Go (the "hottest flow" leaderboard — the HERO
// asset is the one with the biggest absolute net move). Uses DISTINCT ON
// (asset) which matches idx_whale_snapshots_latest exactly.
//
// A bad exchange_breakdown JSONB blob (impossible via UpsertSnapshot, but a
// future migration could inject one) returns the row with an empty map rather
// than failing the whole query — same trade-off as narrative.LatestSnapshots.
func (s *Store) LatestSnapshots(ctx context.Context) ([]Snapshot, error) {
	if s == nil || s.Pool == nil {
		return nil, errors.New("whale.Store: nil pool")
	}
	rows, err := s.Pool.Query(ctx, `
		SELECT DISTINCT ON (asset)
			asset, captured_at, net_flow_usd_24h, net_flow_prev_24h,
			flow_pct, direction, inflow_usd_24h, outflow_usd_24h,
			tx_count_24h, confidence, exchange_breakdown, reasons,
			is_new_spike, source, partial
		FROM whale_snapshots
		ORDER BY asset, captured_at DESC
	`)
	if err != nil {
		return nil, fmt.Errorf("LatestSnapshots: %w", err)
	}
	defer rows.Close()

	out := []Snapshot{}
	for rows.Next() {
		var sn Snapshot
		var dirStr string
		var breakdownJSON []byte
		if err := rows.Scan(
			&sn.Asset, &sn.CapturedAt, &sn.NetFlowUSD24h, &sn.NetFlowPrev24h,
			&sn.FlowPct, &dirStr, &sn.InflowUSD24h, &sn.OutflowUSD24h,
			&sn.TxCount24h, &sn.Confidence, &breakdownJSON, &sn.Reasons,
			&sn.IsNewSpike, &sn.Source, &sn.Partial,
		); err != nil {
			return nil, fmt.Errorf("LatestSnapshots: scan: %w", err)
		}
		sn.Direction = FlowDirection(dirStr)
		// Decode breakdown; on parse error leave map empty rather than
		// dropping the whole row.
		breakdown := map[string]int{}
		if len(breakdownJSON) > 0 {
			_ = json.Unmarshal(breakdownJSON, &breakdown)
		}
		sn.ExchangeBreakdown = breakdown
		if sn.Reasons == nil {
			sn.Reasons = []string{}
		}
		out = append(out, sn)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("LatestSnapshots: rows: %w", err)
	}

	// In-memory sort by |net_flow_usd_24h| DESC, ties broken by asset ASC.
	// Doing it here (not in SQL) keeps DISTINCT ON happy — DISTINCT ON
	// requires the leading ORDER BY column be the DISTINCT key (asset).
	sort.Slice(out, func(i, j int) bool {
		ai, aj := math.Abs(out[i].NetFlowUSD24h), math.Abs(out[j].NetFlowUSD24h)
		if ai != aj {
			return ai > aj
		}
		return out[i].Asset < out[j].Asset
	})
	return out, nil
}

// jsonEncodeStableMap marshals map[string]int into byte-stable JSONB. Go's
// stdlib json.Marshal already sorts string-keyed maps, so we delegate; this
// thin wrapper documents the byte-stability requirement at the call-site and
// handles the empty case explicitly so an empty map encodes as `{}` not `null`.
func jsonEncodeStableMap(m map[string]int) ([]byte, error) {
	if len(m) == 0 {
		return []byte("{}"), nil
	}
	return json.Marshal(m)
}
