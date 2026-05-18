package narrative

// worker.go — periodic ingest+score+persist loop for Narrative Radar.
//
// Lifecycle:
//   1. Run() does ONE warm-start refreshAll synchronously — guarantees the
//      first request to /api/narratives finds at least one snapshot row per
//      narrative (even if it shows mention_count_24h=0 for empty narratives).
//   2. Then loops on RefreshInterval ticks until ctx is cancelled.
//   3. Each tick is independent: a failure in one fetcher (e.g. Reddit 403,
//      RSS feed timeout) is logged but doesn't abort the cycle. The next
//      tick retries from scratch.
//
// Why a fixed-interval ticker (rather than e.g. dynamic backoff): refresh
// frequency is the dominant input to the freshness of mention_count_24h. A
// fixed 10-minute cadence gives us 144 ticks/day → ~140 unique snapshots
// per narrative per day, which is more than enough resolution for the
// growth_pct sparkline without spamming the upstream APIs (each tick fans
// out to 3 fetchers in parallel).

import (
	"context"
	"fmt"
	"log"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/prostwp/elibri-backend/internal/news"
)

// Worker drives the periodic ingest+score+persist cycle.
type Worker struct {
	Store *Store

	// Classifier scores each newly-inserted mention's market impact in
	// [0, 100]. Optional: when nil, the worker skips the classifier phase
	// and the trend score's importance component contributes 0 points.
	// Production should always wire a Classifier (HaikuClassifier in normal
	// operation, RuleClassifier when ANTHROPIC_API_KEY is empty).
	Classifier ImportanceClassifier

	// RefreshInterval — how often to run refreshAll. Default 10 min if zero.
	// 10 min was chosen because (a) the upstream RSS feeds publish at most
	// every few minutes, so faster polling gives no fresher data, (b) Reddit
	// rate-limits aggressively, and (c) growth_pct is a 24h delta — sub-10-
	// minute resolution is noise.
	RefreshInterval time.Duration

	// Logger optional; if nil, uses log.Default(). Callers who want
	// structured logs (slog, zap) wrap their logger here.
	Logger *log.Logger

	// fetcherCtxTimeout — per-tick HTTP timeout for ALL upstream fetchers
	// combined. Each individual fetcher already enforces its own 10s
	// http.Client.Timeout; this wraps the whole fan-out so a hung goroutine
	// can't block the next tick. Set to 30s: 3× the per-fetcher timeout
	// gives slack for the slowest one to finish without runaway delay.
	fetcherCtxTimeout time.Duration
}

// defaultRefreshInterval is the cadence applied when RefreshInterval is zero.
// See struct field comment for the rationale on choosing 10 minutes.
const defaultRefreshInterval = 10 * time.Minute

// defaultFetcherCtxTimeout caps the wall-clock duration of one fan-out cycle.
// 30s = 3× each fetcher's 10s http.Client.Timeout — comfortably long for the
// slowest source while keeping the next tick on schedule.
const defaultFetcherCtxTimeout = 30 * time.Second

// classifyTimeout caps the entire Phase 2b LLM classification loop. On a
// burst of ~100 new mentions × ~1.5s/Haiku call = ~150s. We give it 5min so
// the next 10min tick still fires on schedule even on a slow Anthropic day.
// Items that don't classify in this window stay importance_score=NULL and
// get retried next tick (partial index in migration 010 covers the lookup).
const classifyTimeout = 5 * time.Minute

// backfillBatchSize caps how many NULL rows we classify per tick. With Haiku's
// ~1.5s/call and a 5min classifyTimeout, 50 is comfortable. Larger batches
// risk monopolising the Haiku quota when fresh news is the priority.
const backfillBatchSize = 50

// tickersBackfillBatchSize caps how many tickers='{}' rows we re-extract per
// tick. Bigger than backfillBatchSize because (a) no LLM cost, just one
// pure-function ExtractTickers call + one PG UPDATE per row, (b) the empty-
// tickers backlog is finite (only rows that pre-date the Path 3 fix) so
// chewing through it quickly is preferable to dragging it across many ticks.
const tickersBackfillBatchSize = 200

// newsHoursWindow is how many hours back each fetcher pulls. We use 24h to
// match the worker's 24h growth window — the deduper on (narrative, url)
// will silently drop re-ingested rows from previous ticks, so the marginal
// cost of "always fetch 24h" is one cheap PG INSERT-with-DO-NOTHING per
// already-seen item.
const newsHoursWindow = 24

// querySymbol is the "primary" symbol passed to FetchReddit for picking
// subreddits. Narrative Radar isn't symbol-scoped — we want news items
// across the whole crypto market — but FetchReddit was designed for the
// per-coin path and walks coinSubreddits[BTC] = [Bitcoin, CryptoCurrency].
// "Bitcoin" + "CryptoCurrency" is a defensible default catchment for
// generic narrative coverage; in Phase 2 we'll add a "narrative scan" mode
// to news/ that walks all major subs.
const querySymbol = "BTCUSDT"

// Run blocks until ctx is cancelled. Returns the ctx error on cancellation
// (typically context.Canceled or context.DeadlineExceeded) so callers can
// distinguish "shutdown requested" from "fatal worker bug".
//
// Deliberate choice: the warm-start refresh runs synchronously BEFORE the
// ticker starts. If it fails, we log and continue — the loop will retry on
// the first tick. Treating the warm-start as fatal would mean a flaky
// Reddit response on boot crashes the whole backend, which is too brittle.
func (w *Worker) Run(ctx context.Context) error {
	if w.Store == nil {
		return fmt.Errorf("Worker.Run: Store is nil")
	}
	interval := w.RefreshInterval
	if interval <= 0 {
		interval = defaultRefreshInterval
	}
	logger := w.logger()

	// Warm-start: run one cycle before the ticker so /api/narratives has
	// data on first request. Errors here are logged, not returned.
	if err := w.refreshAll(ctx); err != nil {
		logger.Printf("narrative_radar: warm-start failed (continuing): %v", err)
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := w.refreshAll(ctx); err != nil {
				logger.Printf("narrative_radar: refresh cycle error: %v", err)
			}
		}
	}
}

// refreshAll runs one full ingest+score+persist cycle. Errors from the inner
// pieces are aggregated into the log line at the end rather than short-
// circuiting — ingest is best-effort by design.
func (w *Worker) refreshAll(ctx context.Context) error {
	start := time.Now()
	logger := w.logger()

	// Per-cycle context with timeout caps the fetcher fan-out (see
	// fetcherCtxTimeout docstring). The store-write phase below uses the
	// outer ctx, since DB writes should be fast and we want them to bind
	// to shutdown rather than the fetcher cap.
	//
	// Resolved as a local — never mutate the Worker struct. If two Run
	// calls overlap (test harness, restart) we don't want a writer racing
	// the reader for the timeout field.
	fetcherTimeout := w.fetcherCtxTimeout
	if fetcherTimeout <= 0 {
		fetcherTimeout = defaultFetcherCtxTimeout
	}
	fetchCtx, fetchCancel := context.WithTimeout(ctx, fetcherTimeout)
	defer fetchCancel()

	// ── Phase 1: parallel fetch ──────────────────────────────────────
	items := fetchAllSources(fetchCtx, logger)

	// ── Phase 2: classify + persist mentions ─────────────────────────
	insertedNew := 0
	totalMatched := 0
	// Track URLs that produced at least one new insert this cycle. We
	// deduplicate by URL because the same article can match multiple
	// narratives (one URL → many rows) — we only want to call the LLM once.
	classifyURLs := make(map[string]news.Item, 16)
	for _, item := range items {
		slugs := MatchNarratives(item.Headline, item.Summary)
		if len(slugs) == 0 {
			continue
		}
		tickers := ExtractTickers(item.Headline + " " + item.Summary)
		// Source maps from news.Item.Source ("reddit","coindesk",…) directly
		// to the typed Source constants. The migration's CHECK constraint
		// gates anything else from reaching the table.
		src := Source(item.Source)
		anyNewThisItem := false
		for _, slug := range slugs {
			totalMatched++
			m := Mention{
				Narrative: slug,
				Source:    src,
				URL:       item.URL,
				Title:     item.Headline,
				PostedAt:  item.PublishedAt,
				Sentiment: item.Sentiment,
				Tickers:   tickers,
			}
			inserted, err := w.Store.InsertMention(ctx, m)
			if err != nil {
				// Log and continue: a single bad row (e.g. CHECK violation
				// on an unrecognised source) shouldn't drop the rest.
				logger.Printf("narrative_radar: InsertMention slug=%s url=%s: %v", slug, item.URL, err)
				continue
			}
			if inserted {
				insertedNew++
				anyNewThisItem = true
			}
		}
		if anyNewThisItem && item.URL != "" {
			classifyURLs[item.URL] = item
		}
	}

	// ── Phase 2b: classify newly-inserted mentions ───────────────────
	//
	// Bounded by classifyTimeout so a slow Anthropic upstream can't drag the
	// whole tick past the next refresh interval. With ~1.5s per Haiku call
	// and bursts of 100 new items per tick = 150s — well past the 10-min
	// refresh on the worst day. classifyTimeout=5min gives us at most ~200
	// items per tick at Haiku-typical latency, then the rest get a NULL
	// importance_score and are picked up next tick (the partial index in
	// migration 010 is exactly for this backfill case).
	//
	// CachedClassifier persists internally on its first miss-path UPDATE;
	// production always wraps with CachedClassifier (see cmd/server/main.go).
	// We do NOT call Store.UpdateImportance here — that was a redundant
	// second UPDATE per item under cache miss. Bare classifiers (no cache)
	// are only used in tests.
	classifyCtx, classifyCancel := context.WithTimeout(ctx, classifyTimeout)
	defer classifyCancel()
	classified := 0
	if w.Classifier != nil {
		for url, item := range classifyURLs {
			// Honour both shutdown ctx (outer) and classifyCtx deadline.
			if err := classifyCtx.Err(); err != nil {
				logger.Printf("narrative_radar: classifier loop stopped: %v (classified=%d, remaining=%d)",
					err, classified, len(classifyURLs)-classified)
				break
			}
			_, err := w.Classifier.Classify(classifyCtx, item.Headline, item.Summary, url)
			if err != nil {
				// Best-effort: one Anthropic outage must not halt ingest.
				// The mention is already persisted; importance_score stays
				// NULL and the next cycle picks it up via partial index.
				logger.Printf("narrative_radar: Classifier.Classify url=%s: %v", url, err)
				continue
			}
			classified++
		}
	}

	// ── Phase 2c: backfill historical mentions with importance_score = NULL ──
	//
	// New rows are scored in 2b above. But pre-Haiku rows (ingested before
	// ANTHROPIC_API_KEY was wired) live in narrative_mentions with NULL
	// importance_score forever — they don't get re-touched by Phase 2 because
	// InsertMention is a no-op on the duplicate (narrative, url) constraint
	// and never re-enters classifyURLs.
	//
	// This phase walks the partial index idx_narrative_mentions_pending_importance
	// (migration 010) for up to backfillBatchSize rows per tick and runs them
	// through the same classifier path. CachedClassifier persists internally
	// via the cache miss → UpdateImportance branch, so we don't double-write.
	//
	// Bounded by classifyTimeout (re-used from Phase 2b) so a slow Anthropic
	// upstream can't drag the tick past the next refresh interval. Items that
	// don't classify in the window stay NULL and get retried next tick.
	backfilled := 0
	if w.Classifier != nil {
		backfillCtx, backfillCancel := context.WithTimeout(ctx, classifyTimeout)
		pending, err := w.Store.PendingImportance(backfillCtx, backfillBatchSize)
		if err != nil {
			logger.Printf("narrative_radar: PendingImportance lookup failed: %v", err)
		} else {
			for _, p := range pending {
				if cerr := backfillCtx.Err(); cerr != nil {
					logger.Printf("narrative_radar: backfill stopped: %v (backfilled=%d, remaining=%d)",
						cerr, backfilled, len(pending)-backfilled)
					break
				}
				// Classifier persists internally via CachedClassifier. Don't double-write.
				// summary="" — narrative_mentions doesn't store summary; the
				// classifier already tolerates a blank summary.
				if _, cerr := w.Classifier.Classify(backfillCtx, p.Title, "", p.URL); cerr != nil {
					logger.Printf("narrative_radar: backfill Classify url=%s: %v", p.URL, cerr)
					continue
				}
				backfilled++
			}
		}
		backfillCancel()
		if backfilled > 0 {
			logger.Printf("narrative_radar: backfill classified %d historical mentions", backfilled)
		}
	}

	// ── Phase 2d: backfill tickers on rows that pre-date the Path 3 fix ──
	//
	// ExtractTickers grew Path 3 (coin-name → ticker mapping) after the v2
	// schema landed, so rows ingested before that fix have tickers='{}' even
	// though their titles like "Jito Labs launches…" should yield JTO. We walk
	// up to tickersBackfillBatchSize empty-tickers rows per tick and re-run
	// ExtractTickers against the saved title.
	//
	// Pure function — no LLM, no network. The batch is bounded by the outer
	// ctx (worker shutdown), not by classifyTimeout, because each iteration
	// is sub-millisecond. Rows where ExtractTickers returns empty are LEFT
	// AS-IS (no UPDATE) — re-writing tickers='{}' to tickers='{}' would be a
	// pointless write that bumps WAL traffic without changing state. They'll
	// re-appear in the next tick's PendingTickers query but that's cheap (one
	// indexed read) and harmless.
	backfilledTickers := 0
	pending, err := w.Store.PendingTickers(ctx, tickersBackfillBatchSize)
	if err != nil {
		logger.Printf("narrative_radar: PendingTickers lookup failed: %v", err)
	} else {
		for _, p := range pending {
			if cerr := ctx.Err(); cerr != nil {
				logger.Printf("narrative_radar: tickers backfill stopped: %v (backfilled=%d, remaining=%d)",
					cerr, backfilledTickers, len(pending)-backfilledTickers)
				break
			}
			tickers := ExtractTickers(p.Title) // re-runs Path 3 against the saved title
			if len(tickers) == 0 {
				// Nothing to update — leave as-is to avoid pointless writes.
				continue
			}
			if uerr := w.Store.UpdateTickersByID(ctx, p.ID, tickers); uerr != nil {
				logger.Printf("narrative_radar: UpdateTickersByID id=%d: %v", p.ID, uerr)
				continue
			}
			backfilledTickers++
		}
	}

	// ── Phase 3: score + snapshot per narrative ──────────────────────
	now := time.Now()
	since24h := now.Add(-24 * time.Hour)
	since48h := now.Add(-48 * time.Hour)

	snapsWritten := 0
	for _, slug := range AllNarrativeSlugs() {
		if err := w.refreshOneSnapshot(ctx, slug, now, since24h, since48h); err != nil {
			logger.Printf("narrative_radar: snapshot %s failed: %v", slug, err)
			continue
		}
		snapsWritten++
	}

	logger.Printf("narrative_radar: refreshed %d narratives, ingested %d new mentions (%d matched, %d items scanned, %d classified, %d backfilled, tickers_backfilled=%d), took %s",
		snapsWritten, insertedNew, totalMatched, len(items), classified, backfilled, backfilledTickers, time.Since(start).Round(time.Millisecond))
	return nil
}

// refreshOneSnapshot does the per-narrative score-and-persist work. Pulled
// out of refreshAll so a failure on one narrative doesn't abort the rest.
func (w *Worker) refreshOneSnapshot(ctx context.Context, slug string, now, since24h, since48h time.Time) error {
	// 24h count.
	count24, err := w.Store.CountMentions(ctx, slug, since24h, now)
	if err != nil {
		return fmt.Errorf("CountMentions 24h: %w", err)
	}
	// Prior-24h count: half-open [now-48h, now-24h). The CountMentions
	// contract enforces half-open, so a row exactly at now-24h belongs to
	// the LATER (current) window. No double-counting at the seam.
	countPrev, err := w.Store.CountMentions(ctx, slug, since48h, since24h)
	if err != nil {
		return fmt.Errorf("CountMentions prev-24h: %w", err)
	}

	// Pull 24h mentions for sentiment stats. We could reuse the count
	// query path, but pulling the rows once gives us mean + stdDev + the
	// row list (used elsewhere if/when this function grows).
	mentions, err := w.Store.MentionsSince(ctx, slug, since24h)
	if err != nil {
		return fmt.Errorf("MentionsSince: %w", err)
	}
	meanSent, stdDev := sentimentMeanStdDev(mentions)

	// Source breakdown for the 24h window.
	breakdown, err := w.Store.SourceBreakdown(ctx, slug, since24h, now)
	if err != nil {
		return fmt.Errorf("SourceBreakdown: %w", err)
	}
	sourceDiv := len(breakdown)

	// Top tickers: 5 is the radar UI's badge slot count.
	topTick, err := w.Store.TopTickers(ctx, slug, since24h, now, 5)
	if err != nil {
		return fmt.Errorf("TopTickers: %w", err)
	}

	// Average LLM-graded importance across mentions in the 24h window.
	// Returns 0 when no mentions are scored (e.g. classifier disabled or
	// every row still pending classification) — the trend score's
	// importance component contributes 0 in that case, which is correct.
	avgImp, err := w.Store.AvgImportance(ctx, slug, since24h, now)
	if err != nil {
		return fmt.Errorf("AvgImportance: %w", err)
	}

	// Pure scoring.
	growth := ComputeGrowthPct(count24, countPrev)
	stage := ClassifyStage(count24, growth)
	trendScore := ComputeTrendScore(count24, growth, meanSent, sourceDiv, avgImp)
	sentLabel := LabelSentiment(meanSent)
	confidence := ComputeConfidence(count24, sourceDiv, stdDev)

	snap := Snapshot{
		Narrative:           slug,
		CapturedAt:          now,
		MentionCount24h:     count24,
		MentionCountPrev24h: countPrev,
		GrowthPct:           growth,
		TrendScore:          trendScore,
		Stage:               stage,
		SentimentScore:      meanSent,
		SentimentLabel:      sentLabel,
		Confidence:          confidence,
		RelatedAssets:       topTick,
		SourcesBreakdown:    breakdown,
		Reasons:             buildReasons(count24, growth, sourceDiv, meanSent, sentLabel, stage),
	}
	return w.Store.UpsertSnapshot(ctx, snap)
}

// fetchAllSources fans out to the three news fetchers in parallel and returns
// the deduplicated, combined item list. Errors are logged but never returned —
// one source down ≠ failure (Reddit 403s from server IPs are routine; see
// internal/news/reddit.go for that story).
//
// Deduplication is by URL across ALL sources combined. Without it, an article
// syndicated by both CoinDesk and Cointelegraph would double-insert into
// narrative_mentions (the (narrative, url) constraint catches it at insert
// time, but pre-dedup saves the round trip).
func fetchAllSources(ctx context.Context, logger *log.Logger) []news.Item {
	var (
		wg                        sync.WaitGroup
		rdItems, cdItems, ctItems []news.Item
		rdErr, cdErr, ctErr       error
	)

	wg.Add(3)
	go func() {
		defer wg.Done()
		rdItems, rdErr = news.FetchReddit(ctx, querySymbol, newsHoursWindow)
	}()
	go func() {
		defer wg.Done()
		cdItems, cdErr = news.FetchRSS(ctx,
			"https://www.coindesk.com/arc/outboundfeeds/rss?outputType=xml",
			"coindesk", "crypto", newsHoursWindow)
	}()
	go func() {
		defer wg.Done()
		ctItems, ctErr = news.FetchRSS(ctx,
			"https://cointelegraph.com/rss",
			"cointelegraph", "crypto", newsHoursWindow)
	}()
	wg.Wait()

	if rdErr != nil {
		logger.Printf("narrative_radar: reddit fetch error (continuing): %v", rdErr)
	}
	if cdErr != nil {
		logger.Printf("narrative_radar: coindesk RSS fetch error (continuing): %v", cdErr)
	}
	if ctErr != nil {
		logger.Printf("narrative_radar: cointelegraph RSS fetch error (continuing): %v", ctErr)
	}

	all := make([]news.Item, 0, len(rdItems)+len(cdItems)+len(ctItems))
	all = append(all, rdItems...)
	all = append(all, cdItems...)
	all = append(all, ctItems...)

	// Dedup by URL. First-seen wins → Reddit's permalinks are kept; the RSS
	// duplicates are dropped. (Reddit's URL field is permalink-based so
	// collisions across sources are rare anyway, but this is defensive.)
	seen := make(map[string]bool, len(all))
	out := make([]news.Item, 0, len(all))
	for _, it := range all {
		if it.URL == "" {
			out = append(out, it) // empty URLs can't collide; keep them
			continue
		}
		if seen[it.URL] {
			continue
		}
		seen[it.URL] = true
		out = append(out, it)
	}
	return out
}

// sentimentMeanStdDev computes the mean and population standard deviation of
// the Sentiment field across mentions. Returns (0, 0) for an empty slice and
// (mean, 0) for a single-element slice. Population (not sample) std-dev is
// used because we treat the in-window mentions as the entire population, not
// a sample — std-dev=0 means "every mention agrees", which is the contract
// ComputeConfidence expects.
func sentimentMeanStdDev(mentions []Mention) (float64, float64) {
	if len(mentions) == 0 {
		return 0, 0
	}
	if len(mentions) == 1 {
		return mentions[0].Sentiment, 0
	}
	var sum float64
	for _, m := range mentions {
		sum += m.Sentiment
	}
	mean := sum / float64(len(mentions))
	var sqSum float64
	for _, m := range mentions {
		d := m.Sentiment - mean
		sqSum += d * d
	}
	stdDev := math.Sqrt(sqSum / float64(len(mentions)))
	return mean, stdDev
}

// buildReasons assembles short bullet phrases for the radar UI. Order is
// (volume → growth → diversity → sentiment) matching how a trader scans the
// row. Phrases are deliberately terse and use plain ASCII for the "↑/↓"
// indicators so a user's terminal/copy-paste doesn't mangle them.
func buildReasons(mentions24h int, growthPct float64, sourceDiv int, meanSent float64, sentLabel SentimentLabel, stage Stage) []string {
	out := make([]string, 0, 4)

	// Growth bullet — only if the change is meaningful (≥10% in either
	// direction) AND the prior period had data. Fresh narratives (the
	// growthInfinite path) get a dedicated phrase rather than an arrow.
	switch {
	case growthPct >= 999.0 && mentions24h > 0:
		out = append(out, "New narrative this cycle")
	case growthPct >= 10:
		out = append(out, fmt.Sprintf("Mentions up %.0f%% vs prior 24h", growthPct))
	case growthPct <= -10:
		// abs() in the message — "down 30%" reads better than "down -30%".
		out = append(out, fmt.Sprintf("Mentions down %.0f%% vs prior 24h", math.Abs(growthPct)))
	}

	// Source diversity — call out 1 source as a caveat, 2+ as positive.
	switch {
	case sourceDiv == 0:
		// Skip — silent when there's no data, the stage bullet says enough.
	case sourceDiv == 1:
		out = append(out, "Only 1 source reporting")
	default:
		out = append(out, fmt.Sprintf("%d sources reporting", sourceDiv))
	}

	// Sentiment — only call it out when not neutral (neutral is the default
	// expectation, no need to repeat it in the UI).
	if sentLabel != SentNeutral {
		direction := "bullish"
		if sentLabel == SentBear {
			direction = "bearish"
		}
		out = append(out, fmt.Sprintf("Sentiment leans %s (%+.2f)", direction, meanSent))
	}

	// Stage — always include, since this is the headline framing.
	out = append(out, fmt.Sprintf("Stage: %s", strings.ToUpper(string(stage))))
	return out
}

// logger returns the Worker's logger or log.Default() when nil.
func (w *Worker) logger() *log.Logger {
	if w.Logger != nil {
		return w.Logger
	}
	return log.Default()
}
