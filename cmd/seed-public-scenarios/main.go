// Package main — seed-public-scenarios CLI.
//
// Helper utility for the Sprint 3 smoke test of the public catalog. Flips
// existing strategies (rows the developer already created via the regular
// editor) into is_public=TRUE so the /scenarios/featured + /scenarios/public
// endpoints have something to serve. Without this, a fresh dev DB has zero
// public rows and the catalog page renders empty.
//
// Why a separate binary instead of a SQL migration:
//   Migrations have to be deterministic — they run on every start, on every
//   environment, and they can't pick "the user's own strategies" because
//   a migration can't know which auth.user the operator wants to seed for.
//   This CLI is owned by the operator who runs it explicitly, so it can
//   take args + env and act on whatever rows happen to exist.
//
// Modes:
//   --count=N       (default 6): pick the N most recent strategies that
//                   have author_slug NOT NULL and flip is_public=TRUE +
//                   published_at=NOW(). Top 4 also get is_featured=TRUE.
//   --slug=foo      (mutually exclusive with --count): act on a single
//                   strategy identified by its author_slug.
//   --featured      paired with --slug: also set is_featured=TRUE.
//   --dsn=...       overrides DATABASE_URL.
//   --dry-run       prints the SQL it would issue without committing.
//
// Idempotency:
//   Re-running --count=6 doesn't duplicate rows — strategies are UPDATEd in
//   place. Already-public rows have their published_at refreshed so the
//   catalog ORDER BY (which uses published_at DESC) re-shuffles them to
//   the top, which is what an operator running the seed script expects.
//
// Examples:
//   go run ./cmd/seed-public-scenarios --count=6
//   go run ./cmd/seed-public-scenarios --slug=btc-momentum-pro --featured
//   go run ./cmd/seed-public-scenarios --dsn=postgresql://... --count=10
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// We accept either DATABASE_URL from env or a --dsn flag. The flag wins
// when both are set (matches the pg_dump / psql convention).
const defaultDSN = "postgresql://elibri:elibri@localhost:5432/elibri?sslmode=disable"

// featuredFromCount is the number of top rows that get is_featured=TRUE
// when running in --count mode. Hard-coded at 4 because the home-page
// hero strip on the v3 redesign shows 4 cards; raising/lowering this
// would change UI behaviour without touching the UI.
const featuredFromCount = 4

func main() {
	var (
		count    = flag.Int("count", 6, "number of recent strategies to flip public (ignored when --slug is given)")
		slug     = flag.String("slug", "", "author_slug of a single strategy to flip public; mutually exclusive with --count")
		featured = flag.Bool("featured", false, "in --slug mode: also set is_featured=TRUE")
		dsn      = flag.String("dsn", "", "Postgres DSN; defaults to $DATABASE_URL")
		dryRun   = flag.Bool("dry-run", false, "print intended changes without executing")
	)
	flag.Parse()

	// Resolve DSN: --dsn > $DATABASE_URL > built-in dev default. The dev
	// default keeps `go run ./cmd/seed-public-scenarios --count=6` zero-
	// config on Денис's workstation, mirroring the same fallback chain
	// in internal/config/config.go.
	conn := *dsn
	if conn == "" {
		conn = os.Getenv("DATABASE_URL")
	}
	if conn == "" {
		conn = defaultDSN
		log.Printf("DSN not provided; using built-in dev default %q", defaultDSN)
	}

	// Connect with a short timeout so the operator gets a clear error
	// instead of staring at a hung process when DATABASE_URL points to
	// a local container that's not running.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	pool, err := pgxpool.New(ctx, conn)
	if err != nil {
		log.Fatalf("connect: %v", err)
	}
	defer pool.Close()

	if err := pool.Ping(ctx); err != nil {
		log.Fatalf("ping: %v", err)
	}

	if *slug != "" {
		if err := seedSingle(ctx, pool, *slug, *featured, *dryRun); err != nil {
			log.Fatalf("seed slug %q: %v", *slug, err)
		}
		return
	}

	if *count <= 0 {
		log.Fatal("--count must be >= 1, or pass --slug instead")
	}
	if err := seedRecent(ctx, pool, *count, *dryRun); err != nil {
		log.Fatalf("seed recent (n=%d): %v", *count, err)
	}
}

// seedRecent flips the N most-recently-updated strategies (with author_slug
// non-null) into is_public=TRUE. The top featuredFromCount rows also get
// is_featured=TRUE. We rely on author_slug NOT NULL because that's the
// gate the catalog API uses (see scenarios_public_handlers.go: rows
// without an author_slug never appear, so flipping them public is wasted
// effort).
//
// The whole flip runs in a single transaction so a partial failure leaves
// the catalog in a consistent state — either all 6 rows are public or
// none are.
func seedRecent(ctx context.Context, pool *pgxpool.Pool, count int, dryRun bool) error {
	tx, err := pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	// pgx tolerates Rollback on a committed tx — it returns
	// ErrTxClosed which we ignore. Using a closure-captured tx
	// pointer would let us nil-out on commit, but the noisy error
	// here is harmless and the simpler defer is easier to read.
	defer func() { _ = tx.Rollback(ctx) }()

	// Pull the candidate IDs in updated_at DESC order. We do it in two
	// steps (SELECT then UPDATE) instead of a CTE because the UPDATE
	// also needs to know which IDs landed in the "featured" top slice,
	// and ORDER BY inside an UPDATE FROM is awkward to express without
	// a derived table.
	rows, err := tx.Query(ctx, `
		SELECT id, COALESCE(author_slug, ''), name
		FROM strategies
		WHERE author_slug IS NOT NULL
		ORDER BY updated_at DESC
		LIMIT $1
	`, count)
	if err != nil {
		return fmt.Errorf("select candidates: %w", err)
	}

	type candidate struct {
		ID   string
		Slug string
		Name string
	}
	var cands []candidate
	for rows.Next() {
		var c candidate
		if err := rows.Scan(&c.ID, &c.Slug, &c.Name); err != nil {
			rows.Close()
			return fmt.Errorf("scan candidate: %w", err)
		}
		cands = append(cands, c)
	}
	rows.Close()

	if len(cands) == 0 {
		log.Printf("no eligible strategies (need at least one row with author_slug NOT NULL); nothing to seed")
		return nil
	}
	log.Printf("found %d candidate strategies (requested %d):", len(cands), count)
	for i, c := range cands {
		marker := ""
		if i < featuredFromCount {
			marker = " [featured]"
		}
		log.Printf("  %d. %s (slug=%s, id=%s)%s", i+1, c.Name, c.Slug, c.ID, marker)
	}

	if dryRun {
		log.Printf("--dry-run: skipping UPDATE")
		return nil
	}

	// Split into "featured" (top N) and "rest" arrays so the two
	// UPDATEs differ only in the is_featured value. We pass each batch
	// of IDs as a single uuid[] via ANY($1) — keeps the placeholder
	// count at 1 and dodges the dynamic "$1, $2, $3" string-build.
	var featuredIDs, restIDs []string
	for i, c := range cands {
		if i < featuredFromCount {
			featuredIDs = append(featuredIDs, c.ID)
		} else {
			restIDs = append(restIDs, c.ID)
		}
	}

	if len(featuredIDs) > 0 {
		if err := flipPublic(ctx, tx, featuredIDs, true); err != nil {
			return fmt.Errorf("flip featured rows: %w", err)
		}
		log.Printf("flipped %d row(s) to public + featured", len(featuredIDs))
	}
	if len(restIDs) > 0 {
		if err := flipPublic(ctx, tx, restIDs, false); err != nil {
			return fmt.Errorf("flip non-featured rows: %w", err)
		}
		log.Printf("flipped %d row(s) to public (not featured)", len(restIDs))
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit: %w", err)
	}
	log.Printf("seed complete; verify with: SELECT id, name, is_public, is_featured, published_at FROM strategies WHERE is_public=TRUE ORDER BY published_at DESC")
	return nil
}

// seedSingle flips a single strategy by author_slug. Useful when the
// operator is curating the catalog manually ("publish this exact one,
// no others"). The --featured flag is opt-in: the default behaviour is
// to flip is_public only and leave is_featured at whatever it was.
func seedSingle(ctx context.Context, pool *pgxpool.Pool, slug string, featured, dryRun bool) error {
	// Sanity-check the slug exists before issuing UPDATE so the operator
	// gets a useful error ("no strategy with that slug") instead of a
	// silent zero-rows-affected.
	var (
		id        string
		name      string
		curPublic bool
	)
	err := pool.QueryRow(ctx, `
		SELECT id, name, COALESCE(is_public, false)
		FROM strategies
		WHERE author_slug = $1
	`, slug).Scan(&id, &name, &curPublic)
	if err != nil {
		return fmt.Errorf("lookup slug: %w", err)
	}
	log.Printf("found strategy %q (id=%s, currently is_public=%v)", name, id, curPublic)

	if dryRun {
		if featured {
			log.Printf("--dry-run: would set is_public=TRUE, is_featured=TRUE, published_at=NOW() on id=%s", id)
		} else {
			log.Printf("--dry-run: would set is_public=TRUE, published_at=NOW() on id=%s", id)
		}
		return nil
	}

	// Single-row UPDATE — no transaction needed; the statement itself is atomic.
	if featured {
		_, err = pool.Exec(ctx, `
			UPDATE strategies
			SET is_public = TRUE,
			    is_featured = TRUE,
			    published_at = NOW()
			WHERE id = $1
		`, id)
	} else {
		_, err = pool.Exec(ctx, `
			UPDATE strategies
			SET is_public = TRUE,
			    published_at = NOW()
			WHERE id = $1
		`, id)
	}
	if err != nil {
		return fmt.Errorf("update: %w", err)
	}
	log.Printf("flipped %q to public%s", slug, featuredSuffix(featured))
	return nil
}

func featuredSuffix(featured bool) string {
	if featured {
		return " + featured"
	}
	return ""
}

// flipPublic runs the bulk UPDATE for a set of IDs inside an open
// transaction. We pass IDs as a Postgres uuid[] (via the $1::uuid[] cast)
// and use ANY($1) so the placeholder count stays at 1 regardless of how
// many rows we're flipping. pgx's pgtype encoder converts the []string
// into the array literal automatically.
//
// The published_at = NOW() update is the bit that makes re-runs
// idempotent-but-fresh: the operator's smoke test sees the just-seeded
// rows at the top of /scenarios/public ordered by published_at DESC.
func flipPublic(ctx context.Context, tx pgx.Tx, ids []string, featured bool) error {
	if featured {
		_, err := tx.Exec(ctx, `
			UPDATE strategies
			SET is_public = TRUE,
			    is_featured = TRUE,
			    published_at = NOW()
			WHERE id = ANY($1::uuid[])
		`, ids)
		return err
	}
	_, err := tx.Exec(ctx, `
		UPDATE strategies
		SET is_public = TRUE,
		    is_featured = FALSE,
		    published_at = NOW()
		WHERE id = ANY($1::uuid[])
	`, ids)
	return err
}
