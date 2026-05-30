package market

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"sync/atomic"
	"testing"
	"time"
)

// roundTripFunc lets us stub *http.Client transport without a live server.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// newCountingClient returns a client whose every outbound request increments
// *calls and then succeeds with a minimal valid payload. Used as a trip-wire:
// a cache hit must make zero calls.
func newCountingClient(calls *int32) *http.Client {
	return &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		atomic.AddInt32(calls, 1)
		body := `{"data":{"dataList":[{"score":1,"name":"Extreme Fear"}]}}`
		return &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(bytes.NewBufferString(body)),
			Header:     make(http.Header),
		}, nil
	})}
}

// newFailingClient returns a client that increments *calls and fails every
// request, to exercise the stale-while-error fallback.
func newFailingClient(calls *int32) *http.Client {
	return &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		atomic.AddInt32(calls, 1)
		return nil, fmt.Errorf("simulated upstream failure")
	})}
}

// parseFearGreed: table-driven coverage of the decode rule. The load-bearing
// case is "picks the LAST dataList element" — getting it backwards would
// report a stale 14-day-old value as current.
func TestParseFearGreed(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		wantValue int
		wantLabel string
		wantErr   bool
	}{
		{
			name:      "picks last of multiple",
			body:      `{"data":{"dataList":[{"score":71,"name":"Greed"},{"score":50,"name":"Neutral"},{"score":33,"name":"Fear"}]}}`,
			wantValue: 33,
			wantLabel: "Fear",
		},
		{
			name:      "single element",
			body:      `{"data":{"dataList":[{"score":88,"name":"Extreme Greed"}]}}`,
			wantValue: 88,
			wantLabel: "Extreme Greed",
		},
		{
			name:      "ignores extra fields (btcPrice/timestamp)",
			body:      `{"data":{"dataList":[{"score":10,"name":"Extreme Fear","timestamp":"1780015990","btcPrice":"60000","btcVolume":"1"},{"score":42,"name":"Fear","timestamp":"1780102390","btcPrice":"61000","btcVolume":"2"}]}}`,
			wantValue: 42,
			wantLabel: "Fear",
		},
		{
			name:    "empty dataList errors",
			body:    `{"data":{"dataList":[]}}`,
			wantErr: true,
		},
		{
			name:    "malformed json errors",
			body:    `{not json`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			value, label, err := parseFearGreed([]byte(tt.body))
			if tt.wantErr {
				if err == nil {
					t.Fatalf("parseFearGreed(%s): want error, got value=%d label=%q", tt.body, value, label)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseFearGreed(%s): unexpected error: %v", tt.body, err)
			}
			if value != tt.wantValue {
				t.Errorf("value = %d, want %d", value, tt.wantValue)
			}
			if label != tt.wantLabel {
				t.Errorf("label = %q, want %q", label, tt.wantLabel)
			}
		})
	}
}

// FetchFearGreed must serve from cache within TTL without a second upstream
// hit. We can't easily count real CMC calls, so we pre-seed the package cache
// with a fresh entry and assert the fast path returns it. A sentinel httpClient
// that records any outbound call proves no network round-trip happens.
func TestFetchFearGreed_CacheHitWithinTTL(t *testing.T) {
	// Force the keyless path regardless of the developer's shell env, so the
	// counting-client call-count assertion stays deterministic.
	t.Setenv(cmcAPIKeyEnv, "")
	t.Cleanup(resetFearGreedCache)
	resetFearGreedCache()

	// Trip-wire: if FetchFearGreed touches the network on a cache hit, this
	// transport fires and we fail. Restored after the test.
	var calls int32
	origClient := httpClient
	httpClient = newCountingClient(&calls)
	t.Cleanup(func() { httpClient = origClient })

	// Seed a fresh cache entry (within TTL).
	fearGreedMu.Lock()
	fearGreedCache.value = 64
	fearGreedCache.label = "Greed"
	fearGreedCache.at = time.Now()
	fearGreedMu.Unlock()

	value, label, err := FetchFearGreed()
	if err != nil {
		t.Fatalf("FetchFearGreed: unexpected error: %v", err)
	}
	if value != 64 || label != "Greed" {
		t.Errorf("got value=%d label=%q, want 64/Greed", value, label)
	}
	if got := atomic.LoadInt32(&calls); got != 0 {
		t.Errorf("cache hit made %d upstream call(s), want 0", got)
	}
}

// When the cache entry is older than the TTL, the fast path must NOT serve it
// — the function falls through to a fetch (here a forced upstream error), and
// because a stale value still exists it is returned via stale-while-error.
func TestFetchFearGreed_StaleEntryFallsThroughThenStaleWhileError(t *testing.T) {
	// Force the keyless path so exactly one upstream attempt is made (a set key
	// would add a Pro attempt and break the call-count assertion below).
	t.Setenv(cmcAPIKeyEnv, "")
	t.Cleanup(resetFearGreedCache)
	resetFearGreedCache()

	var calls int32
	origClient := httpClient
	httpClient = newFailingClient(&calls)
	t.Cleanup(func() { httpClient = origClient })

	// Seed an EXPIRED cache entry (older than TTL).
	fearGreedMu.Lock()
	fearGreedCache.value = 20
	fearGreedCache.label = "Fear"
	fearGreedCache.at = time.Now().Add(-fearGreedTTL - time.Minute)
	fearGreedMu.Unlock()

	value, label, err := FetchFearGreed()
	if err != nil {
		t.Fatalf("stale-while-error: want cached fallback, got error: %v", err)
	}
	if value != 20 || label != "Fear" {
		t.Errorf("got value=%d label=%q, want stale 20/Fear", value, label)
	}
	// Proves the stale entry did NOT short-circuit the TTL check: a fetch was
	// attempted (and failed), then the stale value was served.
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Errorf("expected exactly 1 upstream attempt on expired cache, got %d", got)
	}
}

// jsonResp is a tiny helper for building stubbed *http.Response values.
func jsonResp(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Body:       io.NopCloser(bytes.NewBufferString(body)),
		Header:     make(http.Header),
	}
}

// newHostBranchingClient routes by request host so a single httpClient stub can
// drive BOTH the Pro endpoint (pro-api.coinmarketcap.com) and the keyless one
// (api.coinmarketcap.com). onPro, if non-nil, observes the Pro request (e.g. to
// assert auth headers) before its response is returned.
func newHostBranchingClient(onPro func(*http.Request), proResp, keylessResp *http.Response) *http.Client {
	return &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		switch r.URL.Host {
		case "pro-api.coinmarketcap.com":
			if onPro != nil {
				onPro(r)
			}
			return proResp, nil
		case "api.coinmarketcap.com":
			return keylessResp, nil
		default:
			return nil, fmt.Errorf("unexpected host %q", r.URL.Host)
		}
	})}
}

// parseFearGreedPro: table-driven coverage of the Pro decode rules — number
// rounding, the value-0 ("Extreme Fear") edge, HTTP-200-with-error_code, empty
// payload, and malformed JSON.
func TestParseFearGreedPro(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		wantValue int
		wantLabel string
		wantErr   bool
	}{
		{
			name:      "valid payload",
			body:      `{"data":{"value":33,"value_classification":"Fear"},"status":{"error_code":"0"}}`,
			wantValue: 33,
			wantLabel: "Fear",
		},
		{
			name:      "value zero is legitimate",
			body:      `{"data":{"value":0,"value_classification":"Extreme Fear"},"status":{"error_code":"0"}}`,
			wantValue: 0,
			wantLabel: "Extreme Fear",
		},
		{
			name:    "invalid key (error_code 1001)",
			body:    `{"status":{"error_code":"1001","error_message":"This API Key is invalid. "}}`,
			wantErr: true,
		},
		{
			name:    "empty classification errors",
			body:    `{"data":{"value":50,"value_classification":""},"status":{"error_code":"0"}}`,
			wantErr: true,
		},
		{
			name:    "malformed json errors",
			body:    `{not json`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			value, label, err := parseFearGreedPro([]byte(tt.body))
			if tt.wantErr {
				if err == nil {
					t.Fatalf("parseFearGreedPro(%s): want error, got value=%d label=%q", tt.body, value, label)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseFearGreedPro(%s): unexpected error: %v", tt.body, err)
			}
			if value != tt.wantValue {
				t.Errorf("value = %d, want %d", value, tt.wantValue)
			}
			if label != tt.wantLabel {
				t.Errorf("label = %q, want %q", label, tt.wantLabel)
			}
		})
	}
}

// When CMC_API_KEY is set, FetchFearGreed must use the Pro endpoint (carrying
// the key header) and NOT the keyless one. The keyless stub returns a different
// value so a mistaken path would be caught by the value assertion.
func TestFetchFearGreed_PrefersProWhenKeySet(t *testing.T) {
	t.Setenv(cmcAPIKeyEnv, "testkey")
	t.Cleanup(resetFearGreedCache)
	resetFearGreedCache()

	var sawKey string
	proBody := `{"data":{"value":77,"value_classification":"Greed"},"status":{"error_code":"0"}}`
	keylessBody := `{"data":{"dataList":[{"score":11,"name":"Extreme Fear"}]}}`

	origClient := httpClient
	httpClient = newHostBranchingClient(
		func(r *http.Request) { sawKey = r.Header.Get("X-CMC_PRO_API_KEY") },
		jsonResp(200, proBody),
		jsonResp(200, keylessBody),
	)
	t.Cleanup(func() { httpClient = origClient })

	value, label, err := FetchFearGreed()
	if err != nil {
		t.Fatalf("FetchFearGreed: unexpected error: %v", err)
	}
	if value != 77 || label != "Greed" {
		t.Errorf("got value=%d label=%q, want 77/Greed (Pro path)", value, label)
	}
	if sawKey != "testkey" {
		t.Errorf("Pro request carried X-CMC_PRO_API_KEY=%q, want %q", sawKey, "testkey")
	}
}

// When the Pro endpoint errors (here HTTP 401 invalid-key), FetchFearGreed must
// fall through to the keyless endpoint rather than blanking the gauge. The
// keyless stub's LAST dataList element (22/"Extreme Fear") is the expected
// result.
func TestFetchFearGreed_FallsBackToKeylessOnProError(t *testing.T) {
	t.Setenv(cmcAPIKeyEnv, "testkey")
	t.Cleanup(resetFearGreedCache)
	resetFearGreedCache()

	proBody := `{"status":{"error_code":"1001","error_message":"This API Key is invalid. "}}`
	keylessBody := `{"data":{"dataList":[{"score":71,"name":"Greed"},{"score":22,"name":"Extreme Fear"}]}}`

	origClient := httpClient
	httpClient = newHostBranchingClient(
		nil,
		jsonResp(401, proBody),
		jsonResp(200, keylessBody),
	)
	t.Cleanup(func() { httpClient = origClient })

	value, label, err := FetchFearGreed()
	if err != nil {
		t.Fatalf("FetchFearGreed: unexpected error: %v", err)
	}
	if value != 22 || label != "Extreme Fear" {
		t.Errorf("got value=%d label=%q, want 22/Extreme Fear (keyless fallback)", value, label)
	}
}

// With no key configured, FetchFearGreed must never touch the Pro endpoint and
// go straight to keyless. The onPro hook is a trip-wire: if a Pro request is
// ever attempted with the key unset, proHits increments and we fail. The Pro
// stub also returns a distinct value (99) so a wrong path trips the value check.
func TestFetchFearGreed_SkipsProWhenKeyUnset(t *testing.T) {
	t.Setenv(cmcAPIKeyEnv, "")
	t.Cleanup(resetFearGreedCache)
	resetFearGreedCache()

	var proHits int32
	proBody := `{"data":{"value":99,"value_classification":"Extreme Greed"},"status":{"error_code":"0"}}`
	keylessBody := `{"data":{"dataList":[{"score":55,"name":"Neutral"}]}}`

	origClient := httpClient
	httpClient = newHostBranchingClient(
		func(*http.Request) { atomic.AddInt32(&proHits, 1) },
		jsonResp(200, proBody),
		jsonResp(200, keylessBody),
	)
	t.Cleanup(func() { httpClient = origClient })

	value, label, err := FetchFearGreed()
	if err != nil {
		t.Fatalf("FetchFearGreed: unexpected error: %v", err)
	}
	if value != 55 || label != "Neutral" {
		t.Errorf("got value=%d label=%q, want 55/Neutral (keyless path)", value, label)
	}
	if got := atomic.LoadInt32(&proHits); got != 0 {
		t.Errorf("Pro endpoint hit %d time(s) with key unset, want 0", got)
	}
}
