package api

// narrative_handlers_test.go — offline tests for handleNarrativesList.
//
// These tests rely on the SnapshotReader interface in internal/narrative
// so we can stub the DB layer entirely. Production wiring uses
// narrative.NewStore(store.Pool); tests use fakeReader.
//
// Coverage:
//   - empty store → 200 with captured_at="" and narratives=[].
//   - limit param → result is truncated to first N rows after ordering.
//   - min_trend filter → drops rows below the threshold.
//   - invalid params (limit=abc, limit=999, min_trend=-3) → silently
//     ignored, full result returned (matching handleFundamentalNews
//     tolerance for malformed query strings).
//   - upstream error → 500 with the writeError envelope.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/internal/narrative"
	"github.com/prostwp/elibri-backend/pkg/types"
)

// init isolates the package's test runs from the live Binance API. The
// production fetchCryptoQuotesFn talks to api.binance.com via httpClient
// (no per-call ctx), and a sandboxed CI without internet access would hang
// every test for the binanceFetchTimeout. The default stub returns empty
// quotes; tests that exercise the price-attach path overwrite it via
// stubFetchQuotes (which restores the previous value via a defer).
func init() {
	fetchCryptoQuotesFn = func(_ []string) (map[string]types.StockQuote, error) {
		return map[string]types.StockQuote{}, nil
	}
}

// fakeReader implements narrative.SnapshotReader. We expose ReturnErr so a
// test can simulate an upstream failure (e.g. transient PG error).
type fakeReader struct {
	snaps     []narrative.Snapshot
	returnErr error
}

func (f *fakeReader) LatestSnapshots(ctx context.Context) ([]narrative.Snapshot, error) {
	if f.returnErr != nil {
		return nil, f.returnErr
	}
	// Return a copy so the handler's in-place filtering (snaps[:0]) can't
	// mutate the test fixture across subtests.
	out := make([]narrative.Snapshot, len(f.snaps))
	copy(out, f.snaps)
	return out, nil
}

// Compile-time guard: fakeReader must satisfy the same interface the
// handler depends on, so the test always exercises the production code
// path and not a divergent shape.
var _ narrative.SnapshotReader = (*fakeReader)(nil)

// fixtureSnapshots returns 4 snapshots with deliberately-spaced trend
// scores so the limit and min_trend assertions are unambiguous. Already
// sorted by trend DESC because the production *Store does the same — we
// match the contract here so tests don't accidentally depend on the
// handler re-sorting.
//
// Every fixture row has MentionCount24h > 0 — the May 2026 Phase 1 fix
// in handleNarrativesList skips the LLM idea call when the top
// narrative has zero mentions, so existing "idea generated" assertions
// would otherwise misfire on the (zero-default) mention count.
// Tests that care about the zero-mentions branch use
// fixtureSnapshotsZeroMentions instead.
func fixtureSnapshots() []narrative.Snapshot {
	now := time.Date(2026, 5, 5, 18, 0, 0, 0, time.UTC)
	return []narrative.Snapshot{
		{
			Narrative: "restaking", CapturedAt: now,
			MentionCount24h: 42,
			TrendScore:      92, Stage: narrative.StageTrending,
			SentimentLabel: narrative.SentBull,
			RelatedAssets:  []string{"ETH", "EIGEN"},
			Reasons:        []string{"Stage: TRENDING"},
		},
		{
			Narrative: "ai-tokens", CapturedAt: now.Add(-2 * time.Minute),
			MentionCount24h: 27,
			TrendScore:      81, Stage: narrative.StageTrending,
			SentimentLabel: narrative.SentBull,
			RelatedAssets:  []string{"FET", "RNDR"},
			Reasons:        []string{"Stage: TRENDING"},
		},
		{
			Narrative: "rwa", CapturedAt: now.Add(-1 * time.Minute),
			MentionCount24h: 12,
			TrendScore:      55, Stage: narrative.StageMainstream,
			SentimentLabel: narrative.SentNeutral,
			RelatedAssets:  []string{"ONDO"},
			Reasons:        []string{"Stage: MAINSTREAM"},
		},
		{
			Narrative: "memecoins", CapturedAt: now.Add(-3 * time.Minute),
			MentionCount24h: 4,
			TrendScore:      18, Stage: narrative.StageDeclining,
			SentimentLabel: narrative.SentBear,
			RelatedAssets:  []string{"DOGE"},
			Reasons:        []string{"Stage: DECLINING"},
		},
	}
}

// fixtureSnapshotsZeroMentions returns the same 4-narrative fixture but
// with the TOP narrative ("restaking") carrying zero mentions in the
// last 24h. Used to drive the May 2026 Phase 1 skip-on-zero branch:
// when MentionCount24h == 0 the handler must NOT call the LLM, because
// otherwise Haiku will hallucinate an analysis of "fragmented dynamics"
// out of literally no data. Other rows keep nonzero counts so tests can
// verify the call-count is exactly zero (top is gated) rather than
// "at most one but maybe more".
func fixtureSnapshotsZeroMentions() []narrative.Snapshot {
	snaps := fixtureSnapshots()
	snaps[0].MentionCount24h = 0
	return snaps
}

// decodedTickerQuote mirrors the per-asset price block embedded in each
// snapshot in the v2.1 response. Kept local so the test asserts the wire
// shape without coupling to the handler's internal struct names.
type decodedTickerQuote struct {
	Symbol      string  `json:"symbol"`
	Price       float64 `json:"price"`
	ChangePct   float64 `json:"change_pct_24h"`
	LastUpdated int64   `json:"last_updated_unix"`
}

// decodedNarrative mirrors one item in the response.narratives slice. We
// embed Snapshot so the v2 fields decode unchanged and only the v2.1
// additions need explicit tags.
type decodedNarrative struct {
	narrative.Snapshot
	RelatedAssetsQuotes []decodedTickerQuote `json:"related_assets_quotes"`
	GeneratedIdea       string               `json:"generated_idea"`
}

// decoded matches the response shape emitted by handleNarrativesList.
// Defined here so the test can inspect both keys without depending on the
// (intentionally) anonymous response struct in the handler.
type decoded struct {
	CapturedAt string             `json:"captured_at"`
	Narratives []decodedNarrative `json:"narratives"`
}

func runRequest(t *testing.T, reader narrative.SnapshotReader, query string) (int, decoded) {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/narratives?"+query, nil)
	rec := httptest.NewRecorder()
	handleNarrativesList(reader)(rec, req)

	res := rec.Result()
	defer res.Body.Close()

	var body decoded
	if res.StatusCode == http.StatusOK {
		if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
			t.Fatalf("decode body: %v", err)
		}
	}
	return res.StatusCode, body
}

func TestNarrativesList_EmptyStore(t *testing.T) {
	reader := &fakeReader{snaps: nil}
	code, body := runRequest(t, reader, "")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if body.CapturedAt != "" {
		t.Errorf("captured_at = %q, want empty string", body.CapturedAt)
	}
	if body.Narratives == nil {
		t.Errorf("narratives is nil — must serialise as empty array, not null")
	}
	if len(body.Narratives) != 0 {
		t.Errorf("narratives len = %d, want 0", len(body.Narratives))
	}
}

func TestNarrativesList_NoFilters(t *testing.T) {
	reader := &fakeReader{snaps: fixtureSnapshots()}
	code, body := runRequest(t, reader, "")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if len(body.Narratives) != 4 {
		t.Errorf("narratives len = %d, want 4", len(body.Narratives))
	}
	if body.CapturedAt == "" {
		t.Errorf("captured_at unexpectedly empty with 4 rows present")
	}
	// Most-recent fixture row is "restaking" at exactly 18:00:00Z.
	if body.CapturedAt != "2026-05-05T18:00:00Z" {
		t.Errorf("captured_at = %q, want 2026-05-05T18:00:00Z", body.CapturedAt)
	}
}

func TestNarrativesList_LimitTwo(t *testing.T) {
	reader := &fakeReader{snaps: fixtureSnapshots()}
	code, body := runRequest(t, reader, "limit=2")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if len(body.Narratives) != 2 {
		t.Fatalf("narratives len = %d, want 2", len(body.Narratives))
	}
	// Top two should be restaking (92) and ai-tokens (81) by trend DESC.
	if body.Narratives[0].Narrative != "restaking" {
		t.Errorf("first = %q, want restaking", body.Narratives[0].Narrative)
	}
	if body.Narratives[1].Narrative != "ai-tokens" {
		t.Errorf("second = %q, want ai-tokens", body.Narratives[1].Narrative)
	}
}

func TestNarrativesList_MinTrendFilter(t *testing.T) {
	reader := &fakeReader{snaps: fixtureSnapshots()}
	code, body := runRequest(t, reader, "min_trend=80")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	// Only restaking (92) and ai-tokens (81) clear the 80 floor.
	if len(body.Narratives) != 2 {
		t.Fatalf("narratives len = %d, want 2 (>=80 only)", len(body.Narratives))
	}
	for _, s := range body.Narratives {
		if s.TrendScore < 80 {
			t.Errorf("snapshot %q has trend_score %d, should be filtered out",
				s.Narrative, s.TrendScore)
		}
	}
}

func TestNarrativesList_InvalidParams(t *testing.T) {
	// limit out of range, min_trend non-numeric — both must be silently
	// ignored, returning the full unfiltered list.
	cases := []string{"limit=abc", "limit=999", "limit=0", "min_trend=oops", "min_trend=-3"}
	for _, q := range cases {
		t.Run(q, func(t *testing.T) {
			reader := &fakeReader{snaps: fixtureSnapshots()}
			code, body := runRequest(t, reader, q)
			if code != http.StatusOK {
				t.Fatalf("status = %d, want 200", code)
			}
			if len(body.Narratives) != 4 {
				t.Errorf("narratives len = %d, want 4 (param ignored)", len(body.Narratives))
			}
		})
	}
}

func TestNarrativesList_LimitAndMinTrendCombined(t *testing.T) {
	// min_trend=50 keeps {restaking 92, ai-tokens 81, rwa 55} → limit=2 truncates
	// to first two. Order: trend DESC, so [restaking, ai-tokens].
	reader := &fakeReader{snaps: fixtureSnapshots()}
	code, body := runRequest(t, reader, "min_trend=50&limit=2")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if len(body.Narratives) != 2 {
		t.Fatalf("narratives len = %d, want 2", len(body.Narratives))
	}
	if body.Narratives[0].Narrative != "restaking" || body.Narratives[1].Narrative != "ai-tokens" {
		t.Errorf("got [%q, %q], want [restaking, ai-tokens]",
			body.Narratives[0].Narrative, body.Narratives[1].Narrative)
	}
}

func TestNarrativesList_UpstreamError(t *testing.T) {
	reader := &fakeReader{returnErr: errors.New("simulated PG outage")}
	req := httptest.NewRequest(http.MethodGet, "/api/v1/narratives", nil)
	rec := httptest.NewRecorder()
	handleNarrativesList(reader)(rec, req)
	res := rec.Result()
	defer res.Body.Close()

	if res.StatusCode != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", res.StatusCode)
	}
	var env map[string]string
	if err := json.NewDecoder(res.Body).Decode(&env); err != nil {
		t.Fatalf("decode error envelope: %v", err)
	}
	if env["error"] == "" {
		t.Errorf("error envelope missing 'error' key: %+v", env)
	}
}

// ─── v2.1: Binance prices + LLM idea ──────────────────────────────────────

// stubFetchQuotes swaps the package-level fetchCryptoQuotesFn for the
// duration of one test. Returns a restore function the caller defers.
// Also clears the in-memory quotes cache so a previous test's writes
// can't leak across (the handler caches by sorted-tickers CSV — different
// fixtures use overlapping tickers, so a leak would show as wrong prices).
func stubFetchQuotes(t *testing.T, fn func(symbols []string) (map[string]types.StockQuote, error)) func() {
	t.Helper()
	old := fetchCryptoQuotesFn
	fetchCryptoQuotesFn = fn
	// Clear cache by reseting the sync.Map. We can't .Range + Delete from
	// outside the package, but we can replace the entire value.
	quotesCache.Range(func(k, _ interface{}) bool {
		quotesCache.Delete(k)
		return true
	})
	return func() {
		fetchCryptoQuotesFn = old
		quotesCache.Range(func(k, _ interface{}) bool {
			quotesCache.Delete(k)
			return true
		})
	}
}

// TestNarrativesList_BinanceQuotesAttached: the handler asks Binance for
// the union of related-asset tickers, attaches matching quotes per
// snapshot, and returns the right symbol/price/change_pct payload. Tickers
// not present in the (stub) quotes map are silently dropped — emulates a
// delisted symbol or one that never had a USDT pair.
func TestNarrativesList_BinanceQuotesAttached(t *testing.T) {
	// Don't t.Parallel — we mutate the package-level fetchCryptoQuotesFn
	// and the shared quotesCache; concurrent tests would race the stub.

	var capturedSymbols []string
	restore := stubFetchQuotes(t, func(symbols []string) (map[string]types.StockQuote, error) {
		capturedSymbols = append([]string{}, symbols...)
		return map[string]types.StockQuote{
			"ETHUSDT":   {Symbol: "ETHUSDT", Price: 3400, ChangePercent: 2.5, Timestamp: 1000},
			"EIGENUSDT": {Symbol: "EIGENUSDT", Price: 1.20, ChangePercent: -3.1, Timestamp: 1000},
			"FETUSDT":   {Symbol: "FETUSDT", Price: 1.50, ChangePercent: 5.0, Timestamp: 1000},
			// Note: RNDR, ONDO, DOGE deliberately missing — should drop.
		}, nil
	})
	defer restore()

	reader := &fakeReader{snaps: fixtureSnapshots()}
	code, body := runRequest(t, reader, "")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}

	// We sent every unique related-asset ticker from the fixture, sorted,
	// suffixed with "USDT". Fixture covers: ETH, EIGEN (restaking),
	// FET, RNDR (ai-tokens), ONDO (rwa), DOGE (memecoins).
	wantSymbols := []string{"DOGEUSDT", "EIGENUSDT", "ETHUSDT", "FETUSDT", "ONDOUSDT", "RNDRUSDT"}
	if len(capturedSymbols) != len(wantSymbols) {
		t.Errorf("Binance fetch saw %v, want %v", capturedSymbols, wantSymbols)
	}
	// Order check — should be sorted alphabetically.
	for i, s := range wantSymbols {
		if i >= len(capturedSymbols) || capturedSymbols[i] != s {
			t.Errorf("Binance symbols[%d] = %q, want %q (full slice: %v)", i, capturedSymbols[i], s, capturedSymbols)
			break
		}
	}

	// Find the restaking snapshot — should have ETH + EIGEN attached.
	var restaking *decodedNarrative
	for i := range body.Narratives {
		if body.Narratives[i].Narrative == "restaking" {
			restaking = &body.Narratives[i]
			break
		}
	}
	if restaking == nil {
		t.Fatal("restaking snapshot missing from response")
	}
	if len(restaking.RelatedAssetsQuotes) != 2 {
		t.Fatalf("restaking has %d quotes, want 2 (ETH + EIGEN)", len(restaking.RelatedAssetsQuotes))
	}
	gotSymbols := []string{restaking.RelatedAssetsQuotes[0].Symbol, restaking.RelatedAssetsQuotes[1].Symbol}
	if gotSymbols[0] != "ETH" || gotSymbols[1] != "EIGEN" {
		t.Errorf("restaking quotes order = %v, want [ETH, EIGEN]", gotSymbols)
	}
	if restaking.RelatedAssetsQuotes[0].Price != 3400 {
		t.Errorf("ETH price = %v, want 3400", restaking.RelatedAssetsQuotes[0].Price)
	}
	if restaking.RelatedAssetsQuotes[0].ChangePct != 2.5 {
		t.Errorf("ETH change_pct = %v, want 2.5", restaking.RelatedAssetsQuotes[0].ChangePct)
	}

	// Find the rwa snapshot — ONDO is missing from the stub, so
	// related_assets_quotes should be empty (not nil — the JSON should be []).
	var rwa *decodedNarrative
	for i := range body.Narratives {
		if body.Narratives[i].Narrative == "rwa" {
			rwa = &body.Narratives[i]
			break
		}
	}
	if rwa == nil {
		t.Fatal("rwa snapshot missing from response")
	}
	if len(rwa.RelatedAssetsQuotes) != 0 {
		t.Errorf("rwa quotes len = %d, want 0 (ONDO missing in stub)", len(rwa.RelatedAssetsQuotes))
	}
}

// TestNarrativesList_BinanceFailureGraceful: when the Binance stub returns
// an error, every snapshot still ships with an EMPTY (not nil) quotes
// slice and the response is 200. The frontend's price line just won't
// render — Dashboard stays usable.
func TestNarrativesList_BinanceFailureGraceful(t *testing.T) {
	restore := stubFetchQuotes(t, func(_ []string) (map[string]types.StockQuote, error) {
		return nil, errors.New("simulated Binance outage")
	})
	defer restore()

	reader := &fakeReader{snaps: fixtureSnapshots()}
	code, body := runRequest(t, reader, "")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (Binance outage should not fail endpoint)", code)
	}
	for _, n := range body.Narratives {
		if n.RelatedAssetsQuotes == nil {
			t.Errorf("snapshot %q has nil quotes slice — must be empty []", n.Narrative)
		}
		if len(n.RelatedAssetsQuotes) != 0 {
			t.Errorf("snapshot %q has %d quotes, want 0 (Binance failed)", n.Narrative, len(n.RelatedAssetsQuotes))
		}
	}
}

// fakeIdeaGen is a stub IdeaGenerator counting calls + capturing the input
// for the assertion on "called only once for the top narrative".
type fakeIdeaGen struct {
	called      int
	lastNarr    string
	lastPrices  []narrative.TickerQuote
	returnIdea  string
	returnError error
}

func (f *fakeIdeaGen) GenerateIdea(_ context.Context, narr string, _ narrative.Snapshot, prices []narrative.TickerQuote) (string, error) {
	f.called++
	f.lastNarr = narr
	f.lastPrices = prices
	if f.returnError != nil {
		return "", f.returnError
	}
	return f.returnIdea, nil
}

// TestNarrativesList_GeneratedIdeaTopOnly: the LLM idea is attached to the
// TOP narrative only (highest trend_score), and the IdeaGenerator is
// called exactly once per request even when multiple snapshots are present.
func TestNarrativesList_GeneratedIdeaTopOnly(t *testing.T) {
	restore := stubFetchQuotes(t, func(_ []string) (map[string]types.StockQuote, error) {
		return map[string]types.StockQuote{}, nil
	})
	defer restore()

	gen := &fakeIdeaGen{returnIdea: "Тестовая идея."}
	oldGen := narrativeIdeaGen
	narrativeIdeaGen = gen
	defer func() { narrativeIdeaGen = oldGen }()
	// Wipe the idea cache so a previous run can't deflect this call.
	narrativeIdeaCache = &narrative.IdeaCache{}

	reader := &fakeReader{snaps: fixtureSnapshots()}
	code, body := runRequest(t, reader, "")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}

	if gen.called != 1 {
		t.Errorf("IdeaGenerator called %d times, want 1 (top narrative only)", gen.called)
	}
	if gen.lastNarr != "restaking" {
		t.Errorf("IdeaGenerator called for narrative %q, want 'restaking' (highest trend)", gen.lastNarr)
	}

	// Top narrative — restaking — has the idea text. Others are empty.
	for _, n := range body.Narratives {
		if n.Narrative == "restaking" {
			if n.GeneratedIdea != "Тестовая идея." {
				t.Errorf("restaking generated_idea = %q, want 'Тестовая идея.'", n.GeneratedIdea)
			}
		} else {
			if n.GeneratedIdea != "" {
				t.Errorf("non-top narrative %q has generated_idea = %q, want empty", n.Narrative, n.GeneratedIdea)
			}
		}
	}
}

// TestNarrativesList_GeneratedIdeaErrorFallback: when the IdeaGenerator
// returns an error, the response still ships (200) but with an EMPTY
// generated_idea so the frontend renders its static fallback.
func TestNarrativesList_GeneratedIdeaErrorFallback(t *testing.T) {
	restore := stubFetchQuotes(t, func(_ []string) (map[string]types.StockQuote, error) {
		return map[string]types.StockQuote{}, nil
	})
	defer restore()

	gen := &fakeIdeaGen{returnError: errors.New("Anthropic outage")}
	oldGen := narrativeIdeaGen
	narrativeIdeaGen = gen
	defer func() { narrativeIdeaGen = oldGen }()
	narrativeIdeaCache = &narrative.IdeaCache{}

	reader := &fakeReader{snaps: fixtureSnapshots()}
	code, body := runRequest(t, reader, "")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200 even on Idea error", code)
	}
	for _, n := range body.Narratives {
		if n.GeneratedIdea != "" {
			t.Errorf("snapshot %q has idea %q, want empty (LLM error → empty)", n.Narrative, n.GeneratedIdea)
		}
	}
}

// TestNarrativesList_GeneratedIdeaCaching: two requests within the same
// captured_at hour should share one IdeaGenerator call. Pinning the cache
// behaviour makes the per-hour cost predictable.
func TestNarrativesList_GeneratedIdeaCaching(t *testing.T) {
	restore := stubFetchQuotes(t, func(_ []string) (map[string]types.StockQuote, error) {
		return map[string]types.StockQuote{}, nil
	})
	defer restore()

	gen := &fakeIdeaGen{returnIdea: "Cached idea."}
	oldGen := narrativeIdeaGen
	narrativeIdeaGen = gen
	defer func() { narrativeIdeaGen = oldGen }()
	narrativeIdeaCache = &narrative.IdeaCache{}

	reader := &fakeReader{snaps: fixtureSnapshots()}
	for i := 0; i < 3; i++ {
		code, _ := runRequest(t, reader, "")
		if code != http.StatusOK {
			t.Fatalf("request #%d status = %d, want 200", i, code)
		}
	}
	if gen.called != 1 {
		t.Errorf("IdeaGenerator called %d times across 3 requests, want 1 (cache hit on 2nd+3rd)", gen.called)
	}
}

// TestNarrativesList_SkipsIdeaOnZeroMentions pins the May 2026 Phase 1
// guard. The bug it prevents: with mention_count_24h=0 the snapshot's
// Reasons are essentially "no data", so Haiku invents a paragraph about
// "fragmented dynamics" — a confident-sounding hallucination over a
// blank canvas. Skipping the LLM entirely lets the frontend fall back
// to its static "Add to watchlist, wait for confirmation on volume"
// line, which is the honest answer.
//
// Asserts:
//   - IdeaGenerator is NEVER called when the top narrative has zero
//     mentions (gen.called == 0)
//   - top narrative ships with generated_idea = "" so the frontend's
//     `idea || fallback` JSX picks the static fallback
//   - the rest of the response is otherwise unchanged (200, all rows
//     present, captured_at populated) — the skip is surgical
func TestNarrativesList_SkipsIdeaOnZeroMentions(t *testing.T) {
	restore := stubFetchQuotes(t, func(_ []string) (map[string]types.StockQuote, error) {
		return map[string]types.StockQuote{}, nil
	})
	defer restore()

	gen := &fakeIdeaGen{returnIdea: "should not be returned"}
	oldGen := narrativeIdeaGen
	narrativeIdeaGen = gen
	defer func() { narrativeIdeaGen = oldGen }()
	narrativeIdeaCache = &narrative.IdeaCache{}

	reader := &fakeReader{snaps: fixtureSnapshotsZeroMentions()}
	code, body := runRequest(t, reader, "")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}

	if gen.called != 0 {
		t.Errorf("IdeaGenerator called %d times, want 0 (top narrative has zero mentions; LLM must be skipped)", gen.called)
	}

	// Top narrative is "restaking" with mention_count_24h=0 in this
	// fixture — verify both the order and the empty idea field.
	if len(body.Narratives) == 0 {
		t.Fatal("response had no narratives")
	}
	top := body.Narratives[0]
	if top.Narrative != "restaking" {
		t.Errorf("top narrative = %q, want 'restaking'", top.Narrative)
	}
	if top.MentionCount24h != 0 {
		t.Errorf("top narrative mention_count_24h = %d, want 0 (test fixture broken)", top.MentionCount24h)
	}
	if top.GeneratedIdea != "" {
		t.Errorf("top generated_idea = %q, want empty string (LLM must be skipped on zero mentions)", top.GeneratedIdea)
	}

	// Sanity: the rest of the rows still ship without an idea.
	for _, n := range body.Narratives[1:] {
		if n.GeneratedIdea != "" {
			t.Errorf("non-top narrative %q has generated_idea = %q, want empty", n.Narrative, n.GeneratedIdea)
		}
	}

	// The handler should still emit captured_at — the skip only gates the
	// LLM call, not the rest of the response.
	if body.CapturedAt == "" {
		t.Errorf("captured_at unexpectedly empty; skip should be surgical")
	}
}

// _ keeps time imported even if no test directly references it via
// time.X — narrative.Snapshot's CapturedAt is still the load-bearing
// time-typed field.
var _ = time.Second
