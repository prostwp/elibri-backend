package api

// narrative_handlers.go — Narrative Radar REST surface.
//
// AUTH: this route is registered on the same mux that auth.Middleware
// wraps in NewRouter, so JWT is enforced automatically (anything outside
// the publicPaths whitelist in internal/auth/jwt.go inherits Bearer-token
// authentication). No extra wiring needed here.
//
// v2.1 additions:
//   - related_assets_quotes: per-snapshot Binance 24h quotes for tickers in
//     RelatedAssets, fetched once per request (deduped across snapshots) and
//     cached for narrativeQuotesCacheTTL to avoid hammering Binance.
//   - generated_idea: Haiku-generated Russian paragraph attached to the TOP
//     narrative only. Cached by (narrativeSlug, captured_at hour) so within
//     the hour the same narrative reuses one Haiku call.

import (
	"context"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/prostwp/elibri-backend/internal/market"
	"github.com/prostwp/elibri-backend/internal/narrative"
	"github.com/prostwp/elibri-backend/pkg/types"
)

// narrativeIdeaGen is the package-level IdeaGenerator wired in from
// cmd/server/main.go via SetIdeaGenerator. nil = use the static fallback
// (zero generated_idea field), so unit tests don't have to plumb anything.
//
// Mirrors the SetMacroConfig / SetRunner pattern already used in this
// package — a single global wired once at boot, never mutated thereafter.
var narrativeIdeaGen narrative.IdeaGenerator

// SetIdeaGenerator wires the IdeaGenerator implementation that
// handleNarrativesList will use to enrich the top narrative. Called once
// from cmd/server/main.go. nil disables the LLM call entirely (response
// returns generated_idea=""), letting the frontend render its static
// fallback.
func SetIdeaGenerator(g narrative.IdeaGenerator) { narrativeIdeaGen = g }

// narrativeIdeaCache is the in-memory IdeaCache used by handleNarrativesList.
// Keyed by (narrativeSlug, captured_at_hour) so requests within the same
// hour reuse the LLM verdict. See narrative.IdeaCache for the eviction
// (or lack thereof) story.
var narrativeIdeaCache = &narrative.IdeaCache{}

// ── Binance price cache ──────────────────────────────────────────────────
//
// We fetch ALL related-asset prices in one Binance call per request, but
// successive list requests within narrativeQuotesCacheTTL share the result.
// Keyed by sorted-tickers-CSV so a request asking for {JTO,SOL,JUP} and a
// request asking for {SOL,JTO,JUP} hit the same cache row.

const (
	// narrativeQuotesCacheTTL caps how long we serve a Binance snapshot
	// before re-fetching. 30s is short enough that "current price" feels
	// live in the dashboard, long enough that a refresh-spam doesn't burn
	// our Binance quota.
	narrativeQuotesCacheTTL = 30 * time.Second

	// binanceFetchTimeout caps the wall-clock for the Binance call. 5s per
	// the spec — if Binance is down we want to return the snapshots without
	// prices rather than fail the whole endpoint.
	binanceFetchTimeout = 5 * time.Second

	// usdtSuffix is appended to bare tickers ("JTO" → "JTOUSDT") before the
	// Binance lookup. Pegged here as a constant so a future quote-asset
	// switch (e.g. USDC for some tokens) can be done in one place.
	usdtSuffix = "USDT"
)

// quotesCacheEntry holds one cached Binance result + the captured timestamp
// used for TTL math.
type quotesCacheEntry struct {
	quotes  map[string]types.StockQuote
	capturedAt time.Time
}

// quotesCache is keyed by sorted-tickers CSV. sync.Map keeps it lock-free
// on the hot read path — the misses are rare relative to hits, and the
// rare-miss code-path takes the write through Store.
var quotesCache sync.Map

// fetchCryptoQuotesFn is the indirection used for stubbing in tests. By
// default it points at market.FetchCryptoQuotes; tests overwrite the var
// to return canned data without touching the network. NOT exported — only
// the package's own test file pokes at it.
var fetchCryptoQuotesFn = market.FetchCryptoQuotes

// tickerQuote is the JSON-facing shape attached to each snapshot. Mirrored
// down to narrative.TickerQuote (drop the JSON tags, drop LastUpdated)
// when handed to the IdeaGenerator.
type tickerQuote struct {
	Symbol      string  `json:"symbol"`            // e.g. "JTO" — bare ticker, no exchange suffix
	Price       float64 `json:"price"`
	ChangePct   float64 `json:"change_pct_24h"`
	LastUpdated int64   `json:"last_updated_unix"` // unix seconds (Binance ticker timestamp)
}

// narrativeWithPrices wraps a Snapshot with the per-asset quotes and the
// optional LLM-generated idea. Embedded Snapshot ensures we don't drift
// from the canonical struct shape — every existing field still flows
// through unchanged, so the frontend's parsing of trend_score / stage /
// reasons / etc. keeps working.
type narrativeWithPrices struct {
	narrative.Snapshot                          // embedded — preserves all existing JSON fields
	RelatedAssetsQuotes []tickerQuote `json:"related_assets_quotes"`         // empty slice (never nil) when Binance failed or no tickers
	GeneratedIdea       string        `json:"generated_idea,omitempty"`      // populated only for the top narrative
}

// handleNarrativesList serves GET /api/v1/narratives.
//
// Returns the latest snapshot per narrative, sorted by trend_score DESC.
// Two optional query parameters:
//
//	limit      int 1..100  — slice the result to first N items.
//	min_trend  int 0..100  — drop items with trend_score < min_trend.
//
// Response shape (v2.1):
//
//	{
//	  "captured_at": "2026-05-05T18:00:00Z",
//	  "narratives": [
//	    {
//	      ...all v2 Snapshot fields...,
//	      "related_assets_quotes": [{"symbol":"JTO","price":2.5,"change_pct_24h":12.4,"last_updated_unix":1714912800}],
//	      "generated_idea": "..."           // top narrative only
//	    },
//	    ...
//	  ]
//	}
//
// When the store has no rows yet (worker hasn't run, or first deploy),
// returns 200 with an empty narratives slice and captured_at = "".
func handleNarrativesList(reader narrative.SnapshotReader) http.HandlerFunc {
	type response struct {
		CapturedAt string                `json:"captured_at"`
		Narratives []narrativeWithPrices `json:"narratives"`
	}

	return func(w http.ResponseWriter, r *http.Request) {
		snaps, err := reader.LatestSnapshots(r.Context())
		if err != nil {
			writeError(w, http.StatusInternalServerError, "failed to load narratives")
			return
		}

		// min_trend filter — drop low-score rows BEFORE limit, so a
		// limit=10/min_trend=80 query returns the top-10 above 80, not 10
		// rows then filtered to whatever's left.
		if v := r.URL.Query().Get("min_trend"); v != "" {
			if minScore, perr := strconv.Atoi(v); perr == nil && minScore > 0 {
				kept := make([]narrative.Snapshot, 0, len(snaps))
				for _, s := range snaps {
					if s.TrendScore >= minScore {
						kept = append(kept, s)
					}
				}
				snaps = kept
			}
		}

		// limit — cap to first N. 1..100 only; anything else (zero,
		// negative, non-integer) is ignored.
		if v := r.URL.Query().Get("limit"); v != "" {
			if n, perr := strconv.Atoi(v); perr == nil && n >= 1 && n <= 100 {
				if n < len(snaps) {
					snaps = snaps[:n]
				}
			}
		}

		// captured_at — pick the most recent across the (possibly filtered)
		// result. This is the value frontend uses for "last refreshed Xm
		// ago" display.
		var latest time.Time
		for _, s := range snaps {
			if !s.CapturedAt.IsZero() && s.CapturedAt.After(latest) {
				latest = s.CapturedAt
			}
		}
		capturedAt := ""
		if !latest.IsZero() {
			capturedAt = latest.UTC().Format(time.RFC3339)
		}

		// Pull Binance prices for ALL related-asset tickers across the
		// (filtered) snapshots, in one call, with caching. Failure here is
		// best-effort — empty quotes per snapshot is fine.
		quotes := loadQuotesForSnapshots(r.Context(), snaps)

		// Build the per-snapshot wrapper objects. Always emit a non-nil
		// quotes slice so the JSON renders `[]` not `null` (frontend depends
		// on `.length > 0` checks).
		wrapped := make([]narrativeWithPrices, 0, len(snaps))
		for _, s := range snaps {
			wrapped = append(wrapped, narrativeWithPrices{
				Snapshot:            s,
				RelatedAssetsQuotes: tickerQuotesFor(s.RelatedAssets, quotes),
			})
		}

		// Generate the idea for the TOP narrative only — the highest
		// trend_score row. The store already returns rows sorted DESC, and
		// we haven't re-sorted, so wrapped[0] is the top after filtering.
		// Skipped when:
		//   - the IdeaGenerator isn't wired (test mode without an
		//     explicit override)
		//   - the list is empty
		//   - the top narrative has zero mentions in the last 24h. With
		//     mention_count_24h=0 the Reasons string is essentially "no
		//     data" and the LLM will fabricate a paragraph about
		//     "fragmented dynamics" out of nothing. The frontend renders
		//     its static "Add to watchlist, wait for confirmation on
		//     volume" fallback when generated_idea is empty, which is the
		//     honest answer here.
		if len(wrapped) > 0 && narrativeIdeaGen != nil && wrapped[0].MentionCount24h > 0 {
			top := &wrapped[0]
			idea := generateIdeaCached(r.Context(), top.Snapshot, top.RelatedAssetsQuotes)
			top.GeneratedIdea = idea
		}

		writeJSON(w, response{
			CapturedAt: capturedAt,
			Narratives: wrapped,
		})
	}
}

// loadQuotesForSnapshots collects every unique ticker across the snapshots'
// RelatedAssets, fetches the Binance 24h quotes once, and returns a map
// keyed by bare ticker (NOT the USDT-suffixed Binance symbol). Failure is
// non-fatal — returns an empty map, callers attach an empty slice per
// snapshot.
func loadQuotesForSnapshots(ctx context.Context, snaps []narrative.Snapshot) map[string]types.StockQuote {
	tickers := uniqueTickers(snaps)
	if len(tickers) == 0 {
		return map[string]types.StockQuote{}
	}

	cacheKey := strings.Join(tickers, ",") // already sorted by uniqueTickers
	if cached, ok := loadCachedQuotes(cacheKey); ok {
		return cached
	}

	// Bound the Binance call. The fetch helper itself uses the package-level
	// httpClient (10s timeout), so we double-bound with a per-call ctx so a
	// hung connection still returns within binanceFetchTimeout.
	fetchCtx, cancel := context.WithTimeout(ctx, binanceFetchTimeout)
	defer cancel()

	// Build the Binance-shaped symbol list ("JTO" → "JTOUSDT"). The
	// fetcher returns its result keyed by the symbol it was called with,
	// so we need to translate back to bare tickers before returning.
	binanceSymbols := make([]string, 0, len(tickers))
	for _, t := range tickers {
		binanceSymbols = append(binanceSymbols, t+usdtSuffix)
	}

	// Run the (synchronous) Binance fetch in a goroutine so we can honour
	// the binanceFetchTimeout via select. FetchCryptoQuotes itself doesn't
	// take a ctx, so this is the only way to enforce the ceiling.
	type result struct {
		quotes map[string]types.StockQuote
		err    error
	}
	resultCh := make(chan result, 1)
	go func() {
		q, err := fetchCryptoQuotesFn(binanceSymbols)
		resultCh <- result{quotes: q, err: err}
	}()

	var raw map[string]types.StockQuote
	select {
	case <-fetchCtx.Done():
		// Binance is slow / down. Return an empty map; the frontend renders
		// the Dashboard without the price line. Don't cache the empty
		// result — a transient blip shouldn't poison the next 30s.
		return map[string]types.StockQuote{}
	case res := <-resultCh:
		if res.err != nil || res.quotes == nil {
			return map[string]types.StockQuote{}
		}
		raw = res.quotes
	}

	// Translate back to bare-ticker keys. Anything missing on Binance
	// (delisted, never listed) just doesn't appear in the result map.
	out := make(map[string]types.StockQuote, len(raw))
	for _, t := range tickers {
		if q, ok := raw[t+usdtSuffix]; ok {
			out[t] = q
		}
	}

	storeCachedQuotes(cacheKey, out)
	return out
}

// uniqueTickers returns the sorted, deduplicated set of related-asset
// tickers across all snapshots. Sorted on output so the cache key is
// stable across request orderings.
func uniqueTickers(snaps []narrative.Snapshot) []string {
	seen := make(map[string]bool, 8)
	for _, s := range snaps {
		for _, t := range s.RelatedAssets {
			if t == "" {
				continue
			}
			seen[t] = true
		}
	}
	if len(seen) == 0 {
		return nil
	}
	out := make([]string, 0, len(seen))
	for t := range seen {
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}

// tickerQuotesFor maps the raw price map down to the JSON-facing slice
// for one snapshot's related assets. Order matches the snapshot's
// RelatedAssets order so the frontend can render in the same order it
// already shows the badges. Tickers missing from the price map are
// silently dropped (no row in related_assets_quotes for them).
func tickerQuotesFor(assets []string, prices map[string]types.StockQuote) []tickerQuote {
	out := make([]tickerQuote, 0, len(assets))
	for _, sym := range assets {
		q, ok := prices[sym]
		if !ok {
			continue
		}
		out = append(out, tickerQuote{
			Symbol:      sym,
			Price:       q.Price,
			ChangePct:   q.ChangePercent,
			LastUpdated: q.Timestamp,
		})
	}
	return out
}

// loadCachedQuotes returns (quotes, true) when the cache holds a non-stale
// entry for cacheKey. Stale = older than narrativeQuotesCacheTTL.
func loadCachedQuotes(cacheKey string) (map[string]types.StockQuote, bool) {
	v, ok := quotesCache.Load(cacheKey)
	if !ok {
		return nil, false
	}
	entry, ok := v.(quotesCacheEntry)
	if !ok {
		return nil, false
	}
	if time.Since(entry.capturedAt) > narrativeQuotesCacheTTL {
		// Don't bother deleting — the next miss overwrites it anyway, and a
		// concurrent reader holding a stale value is fine (we already
		// returned ok=false here).
		return nil, false
	}
	return entry.quotes, true
}

// storeCachedQuotes writes the result under cacheKey with the current
// timestamp. Concurrent writers race; the loser's entry wins for the next
// reader, which is fine — both quotes are equally fresh by definition
// (they were both Binance reads within the request burst).
func storeCachedQuotes(cacheKey string, quotes map[string]types.StockQuote) {
	quotesCache.Store(cacheKey, quotesCacheEntry{
		quotes:     quotes,
		capturedAt: time.Now(),
	})
}

// generateIdeaCached returns the LLM-generated Russian paragraph for the
// top narrative, hitting the in-memory IdeaCache when one exists for the
// (narrativeSlug, captured_at_hour) key. On cache miss, calls the
// IdeaGenerator and persists the result. On any error, returns "" — the
// frontend renders its static fallback in that case.
func generateIdeaCached(ctx context.Context, snap narrative.Snapshot, quotes []tickerQuote) string {
	if narrativeIdeaGen == nil {
		return ""
	}
	hourKey := snap.CapturedAt.UTC().Truncate(time.Hour).Format(time.RFC3339)
	if cached, ok := narrativeIdeaCache.Get(snap.Narrative, hourKey); ok {
		return cached
	}

	// Map the API's tickerQuote down to narrative.TickerQuote. Done here
	// rather than at the IdeaGenerator's signature so the package boundary
	// stays clean (narrative doesn't know about the JSON-facing wrapper).
	narrPrices := make([]narrative.TickerQuote, 0, len(quotes))
	for _, q := range quotes {
		narrPrices = append(narrPrices, narrative.TickerQuote{
			Symbol:    q.Symbol,
			Price:     q.Price,
			ChangePct: q.ChangePct,
		})
	}

	idea, err := narrativeIdeaGen.GenerateIdea(ctx, snap.Narrative, snap, narrPrices)
	if err != nil {
		// Best-effort. Don't pollute the cache with the empty string — a
		// transient outage shouldn't lock in the static fallback for the
		// rest of the hour.
		return ""
	}
	narrativeIdeaCache.Set(snap.Narrative, hourKey, idea)
	return idea
}
