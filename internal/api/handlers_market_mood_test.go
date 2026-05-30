package api

// handlers_market_mood_test.go — hermetic tests for handleMarketMoodRead.
//
// Fully offline: the Fear & Greed call is stubbed via fetchFearGreedFn and
// the LLM call via a fake MarketMoodReader set through api.SetMoodReader.
// The SnapshotReader is the shared fakeReader from narrative_handlers_test.go.
//
// Each test resets the package-level mood-read state (cache + reader) at the
// start and restores it via defer so cases don't leak state into each other.
//
// Coverage:
//   a. marketMoodReader == nil → {"read":""}, no panic.
//   b. fresh success → cached; second call serves the cached value even when
//      the reader would now return something different (TTL cache hit).
//   c. blacklist-empty ("",nil) → {"read":""} AND cached (second call does
//      not re-invoke the reader — asserted via the call counter).
//   d. reader error, no prior cache → {"read":""}.
//   e. reader error WITH prior cache → serves the stale value.
//   f. no input (F&G label empty + empty snapshots) → {"read":""} WITHOUT
//      invoking the reader (call counter == 0).

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/prostwp/elibri-backend/internal/narrative"
)

// fakeMoodReader implements narrative.MarketMoodReader with a configurable
// return value and a call counter so tests can assert whether the handler
// actually invoked the LLM path (vs. served a cached value or skipped on
// empty input).
type fakeMoodReader struct {
	read  string
	err   error
	calls int32 // atomic
}

func (f *fakeMoodReader) ReadMood(_ context.Context, _ narrative.MarketMoodInput) (string, error) {
	atomic.AddInt32(&f.calls, 1)
	return f.read, f.err
}

func (f *fakeMoodReader) callCount() int32 { return atomic.LoadInt32(&f.calls) }

// Compile-time guard: fakeMoodReader satisfies the production interface.
var _ narrative.MarketMoodReader = (*fakeMoodReader)(nil)

// stubFearGreed swaps the package-level fetchFearGreedFn for the duration of
// a test and restores it via the returned cleanup func. Mirrors stubFetchQuotes.
func stubFearGreed(t *testing.T, value int, label string, err error) func() {
	t.Helper()
	old := fetchFearGreedFn
	fetchFearGreedFn = func() (int, string, error) {
		return value, label, err
	}
	return func() { fetchFearGreedFn = old }
}

// resetMoodState clears the cache and the wired reader, returning a cleanup
// that restores both so the test doesn't pollute sibling cases (the package
// global marketMoodReader persists otherwise).
func resetMoodState(t *testing.T) func() {
	t.Helper()
	resetMoodReadCache()
	old := marketMoodReader
	marketMoodReader = nil
	return func() {
		marketMoodReader = old
		resetMoodReadCache()
	}
}

// moodFixture returns one neutral snapshot so the "has input" path is taken.
func moodFixture() []narrative.Snapshot {
	return []narrative.Snapshot{
		{Narrative: "restaking", Stage: narrative.StageTrending, SentimentLabel: narrative.SentBull, TrendScore: 88},
	}
}

// decodeMood drives the handler with a fresh request/recorder and decodes the
// JSON body into read+source. Fails the test on a non-200 status or bad JSON.
func decodeMood(t *testing.T, reader narrative.SnapshotReader) (read, source string) {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/market/mood-read", nil)
	rec := httptest.NewRecorder()

	handleMarketMoodRead(reader).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (handler must never 5xx)", rec.Code)
	}
	var body struct {
		Read   string `json:"read"`
		Source string `json:"source"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode body %q: %v", rec.Body.String(), err)
	}
	if body.Source != "alphavizor-ai" {
		t.Errorf("source = %q, want alphavizor-ai", body.Source)
	}
	return body.Read, body.Source
}

// (a) marketMoodReader == nil → {"read":""}, no panic.
func TestMoodRead_NilReader(t *testing.T) {
	defer resetMoodState(t)()
	defer stubFearGreed(t, 34, "Fear", nil)()
	// resetMoodState already set marketMoodReader = nil.

	read, _ := decodeMood(t, &fakeReader{snaps: moodFixture()})
	if read != "" {
		t.Errorf("read = %q, want empty when no reader wired", read)
	}
}

// (b) fresh success → cached; a second call with a DIFFERENT reader still
// returns the first (cached) value within TTL.
func TestMoodRead_FreshSuccessThenCacheHit(t *testing.T) {
	defer resetMoodState(t)()
	defer stubFearGreed(t, 34, "Fear", nil)()

	first := &fakeMoodReader{read: "Calm climate."}
	marketMoodReader = first

	read, _ := decodeMood(t, &fakeReader{snaps: moodFixture()})
	if read != "Calm climate." {
		t.Fatalf("first read = %q, want 'Calm climate.'", read)
	}
	if first.callCount() != 1 {
		t.Errorf("first reader calls = %d, want 1", first.callCount())
	}

	// Swap in a reader that would return something else. The cache must win.
	second := &fakeMoodReader{read: "DIFFERENT"}
	marketMoodReader = second

	read2, _ := decodeMood(t, &fakeReader{snaps: moodFixture()})
	if read2 != "Calm climate." {
		t.Errorf("second read = %q, want cached 'Calm climate.' (TTL hit)", read2)
	}
	if second.callCount() != 0 {
		t.Errorf("second reader calls = %d, want 0 (served from cache)", second.callCount())
	}
}

// (c) blacklist-empty: reader returns ("",nil) → {"read":""} AND it is
// cached, so a second call does not re-invoke the reader.
func TestMoodRead_BlacklistEmptyIsCached(t *testing.T) {
	defer resetMoodState(t)()
	defer stubFearGreed(t, 50, "Neutral", nil)()

	reader := &fakeMoodReader{read: "", err: nil} // blacklist trip
	marketMoodReader = reader

	read, _ := decodeMood(t, &fakeReader{snaps: moodFixture()})
	if read != "" {
		t.Fatalf("read = %q, want empty on blacklist trip", read)
	}
	if reader.callCount() != 1 {
		t.Fatalf("reader calls = %d after first request, want 1", reader.callCount())
	}

	// Second call: must serve the cached empty value WITHOUT re-invoking.
	read2, _ := decodeMood(t, &fakeReader{snaps: moodFixture()})
	if read2 != "" {
		t.Errorf("second read = %q, want cached empty", read2)
	}
	if reader.callCount() != 1 {
		t.Errorf("reader calls = %d after second request, want still 1 (empty was cached)", reader.callCount())
	}
}

// (d) reader error, no prior cache → {"read":""}.
func TestMoodRead_ErrorNoCache(t *testing.T) {
	defer resetMoodState(t)()
	defer stubFearGreed(t, 25, "Fear", nil)()

	reader := &fakeMoodReader{err: errors.New("upstream down")}
	marketMoodReader = reader

	read, _ := decodeMood(t, &fakeReader{snaps: moodFixture()})
	if read != "" {
		t.Errorf("read = %q, want empty on error with no prior cache", read)
	}
	if reader.callCount() != 1 {
		t.Errorf("reader calls = %d, want 1 (attempted before failing)", reader.callCount())
	}
}

// (e) reader error WITH prior cache: prime the cache via a success, then swap
// to an error-returning reader → the stale value is served.
func TestMoodRead_ErrorServesStale(t *testing.T) {
	defer resetMoodState(t)()
	defer stubFearGreed(t, 34, "Fear", nil)()

	// Prime cache with a success.
	good := &fakeMoodReader{read: "Stable mood."}
	marketMoodReader = good
	if read, _ := decodeMood(t, &fakeReader{snaps: moodFixture()}); read != "Stable mood." {
		t.Fatalf("priming read = %q, want 'Stable mood.'", read)
	}

	// Force the cache to look expired so the handler proceeds to the reader,
	// which now errors — exercising the stale-while-error branch.
	moodReadMu.Lock()
	moodReadCache.at = moodReadCache.at.Add(-2 * moodReadTTL)
	moodReadMu.Unlock()

	bad := &fakeMoodReader{err: errors.New("upstream down")}
	marketMoodReader = bad

	read, _ := decodeMood(t, &fakeReader{snaps: moodFixture()})
	if read != "Stable mood." {
		t.Errorf("read = %q, want stale 'Stable mood.' served on error", read)
	}
	if bad.callCount() != 1 {
		t.Errorf("error reader calls = %d, want 1 (attempted, then fell back to stale)", bad.callCount())
	}
}

// (f) no input: F&G returns an error (label ""), snapshots empty → {"read":""}
// WITHOUT invoking the reader (call counter == 0).
func TestMoodRead_SkipWhenNoInput(t *testing.T) {
	defer resetMoodState(t)()
	defer stubFearGreed(t, 0, "", errors.New("fg unavailable"))()

	reader := &fakeMoodReader{read: "should not be called"}
	marketMoodReader = reader

	read, _ := decodeMood(t, &fakeReader{snaps: nil})
	if read != "" {
		t.Errorf("read = %q, want empty when no F&G and no narratives", read)
	}
	if reader.callCount() != 0 {
		t.Errorf("reader calls = %d, want 0 (handler must skip LLM on empty input)", reader.callCount())
	}
}
