package market

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

// feargreed.go — server-side proxy + cache for CoinMarketCap's Fear & Greed
// index.
//
// Why this lives in the backend at all: CMC's data-api blocks browser CORS
// (no access-control-allow-origin header, empty OPTIONS preflight), so the
// React frontend cannot fetch it client-side. Server-to-server has no CORS,
// so we proxy here and cache the result.
//
// Sources, in priority order:
//   1. Pro (PRIMARY, when CMC_API_KEY is set):
//      https://pro-api.coinmarketcap.com/v3/fear-and-greed/latest
//      Auth header X-CMC_PRO_API_KEY; returns the single latest value. 1 credit/call.
//   2. Keyless (FALLBACK, verified working):
//      https://api.coinmarketcap.com/data-api/v3/fear-greed/chart?start=<sec>&end=<sec>
//      Returns ~14 daily points for a 14-day window; the LAST element of
//      dataList is the most recent value.
// Both are CoinMarketCap, so the handler's source label ("coinmarketcap")
// stays accurate regardless of which path served the value.

const fearGreedBaseURL = "https://api.coinmarketcap.com/data-api/v3/fear-greed/chart"

// fearGreedProURL — official CMC Pro endpoint for the single latest value.
const fearGreedProURL = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/latest"

// cmcAPIKeyEnv — env var holding the CMC Pro API key. When empty we skip the
// Pro path entirely and use the keyless endpoint.
const cmcAPIKeyEnv = "CMC_API_KEY"

// fearGreedTTL — how long a fetched value is served before we refetch. 30 min
// matches the index's coarse daily granularity; refetching faster would just
// hammer CMC for an unchanged number.
const fearGreedTTL = 30 * time.Minute

// fearGreedCache — package-level cache guarded by a RWMutex. `at` zero value
// means "never fetched". On upstream error we serve a stale value if one
// exists (stale-while-error), so a transient CMC blip doesn't 502 the client.
var (
	fearGreedMu    sync.RWMutex
	fearGreedCache struct {
		value int
		label string
		at    time.Time
	}
)

// fearGreedResponse mirrors the subset of the CMC payload we consume.
type fearGreedResponse struct {
	Data struct {
		DataList []struct {
			Score int    `json:"score"`
			Name  string `json:"name"`
		} `json:"dataList"`
	} `json:"data"`
}

// parseFearGreed is a pure, network-free decoder so the parsing rule ("the
// LAST dataList element is the most recent") is unit-testable without a live
// HTTP call. Score is decoded as a JSON number; name is the label string.
func parseFearGreed(body []byte) (value int, label string, err error) {
	var fg fearGreedResponse
	if err := json.Unmarshal(body, &fg); err != nil {
		return 0, "", fmt.Errorf("fear-greed decode: %w", err)
	}
	list := fg.Data.DataList
	if len(list) == 0 {
		return 0, "", fmt.Errorf("fear-greed: empty dataList")
	}
	last := list[len(list)-1]
	return last.Score, last.Name, nil
}

// fearGreedProResponse mirrors the subset of the CMC Pro payload we consume.
// Note status.error_code is a STRING ("0", "1001", ...), not a number, and
// data.value is a JSON number we round to int.
type fearGreedProResponse struct {
	Data struct {
		Value          float64 `json:"value"`
		Classification string  `json:"value_classification"`
	} `json:"data"`
	Status struct {
		ErrorCode    string `json:"error_code"`
		ErrorMessage string `json:"error_message"`
	} `json:"status"`
}

// parseFearGreedPro is the pure, network-free decoder for the Pro endpoint,
// mirroring parseFearGreed so its rules are unit-testable without a live call.
// value is a JSON number (rounded to int); a value of 0 ("Extreme Fear") is a
// legitimate reading, not a sentinel.
func parseFearGreedPro(body []byte) (value int, label string, err error) {
	var fg fearGreedProResponse
	if err := json.Unmarshal(body, &fg); err != nil {
		return 0, "", fmt.Errorf("fear-greed pro decode: %w", err)
	}
	// An HTTP 200 can still carry an application error (non-zero error_code).
	if fg.Status.ErrorCode != "" && fg.Status.ErrorCode != "0" {
		return 0, "", fmt.Errorf("fear-greed pro status %s: %s",
			fg.Status.ErrorCode, fg.Status.ErrorMessage)
	}
	if fg.Data.Classification == "" {
		return 0, "", fmt.Errorf("fear-greed pro: empty payload")
	}
	// Round half-up; value is a small non-negative integer in practice.
	return int(fg.Data.Value + 0.5), fg.Data.Classification, nil
}

// FetchFearGreed returns the most recent Fear & Greed value (0-100) and its
// label (e.g. "Fear", "Neutral", "Greed"). It serves from an in-memory cache
// within fearGreedTTL; on a cache miss it fetches the last 14 days from CMC.
// On upstream error it falls back to any cached value (stale-while-error) and
// only returns the error when nothing is cached.
func FetchFearGreed() (value int, label string, err error) {
	// Fast path: serve fresh cache without touching the network.
	fearGreedMu.RLock()
	if !fearGreedCache.at.IsZero() && time.Since(fearGreedCache.at) < fearGreedTTL {
		v, l := fearGreedCache.value, fearGreedCache.label
		fearGreedMu.RUnlock()
		return v, l, nil
	}
	fearGreedMu.RUnlock()

	v, l, ferr := fetchFearGreedFresh()
	if ferr != nil {
		// Stale-while-error: a previously cached value beats a 502.
		fearGreedMu.RLock()
		defer fearGreedMu.RUnlock()
		if !fearGreedCache.at.IsZero() {
			return fearGreedCache.value, fearGreedCache.label, nil
		}
		return 0, "", ferr
	}

	fearGreedMu.Lock()
	fearGreedCache.value = v
	fearGreedCache.label = l
	fearGreedCache.at = time.Now()
	fearGreedMu.Unlock()
	return v, l, nil
}

// fetchFearGreedFresh encapsulates source selection for a cache-miss refetch:
// Pro first (when a key is configured), keyless as fallback. A Pro failure
// (Pro down, bad key, plan/credit limit) must NOT blank the gauge — we fall
// through to the keyless endpoint instead of surfacing the error.
func fetchFearGreedFresh() (int, string, error) {
	if key := os.Getenv(cmcAPIKeyEnv); key != "" {
		if v, l, err := fetchFearGreedPro(fearGreedProURL, key); err == nil {
			return v, l, nil
		}
		// fall through to keyless on any Pro error
	}

	now := time.Now()
	start := now.Add(-14 * 24 * time.Hour).Unix()
	end := now.Unix()
	url := fmt.Sprintf("%s?start=%s&end=%s", fearGreedBaseURL,
		strconv.FormatInt(start, 10), strconv.FormatInt(end, 10))
	return fetchFearGreedFrom(url)
}

// fetchFearGreedPro does the authenticated HTTP GET + parse against the Pro
// endpoint, reusing the package-shared httpClient (10s timeout, defined in
// moex.go). The API key travels in the X-CMC_PRO_API_KEY header.
func fetchFearGreedPro(url, apiKey string) (int, string, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return 0, "", fmt.Errorf("fear-greed pro request: %w", err)
	}
	req.Header.Set("X-CMC_PRO_API_KEY", apiKey)
	req.Header.Set("Accept", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return 0, "", fmt.Errorf("fear-greed pro fetch: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return 0, "", fmt.Errorf("fear-greed pro upstream status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, "", fmt.Errorf("fear-greed pro read: %w", err)
	}
	return parseFearGreedPro(body)
}

// fetchFearGreedFrom does the HTTP GET + parse against a given URL, reusing
// the package-shared httpClient (10s timeout, defined in moex.go). Split out
// from FetchFearGreed so the URL-building/caching logic stays separate from
// the wire call.
func fetchFearGreedFrom(url string) (int, string, error) {
	resp, err := httpClient.Get(url)
	if err != nil {
		return 0, "", fmt.Errorf("fear-greed fetch: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return 0, "", fmt.Errorf("fear-greed upstream status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, "", fmt.Errorf("fear-greed read: %w", err)
	}
	return parseFearGreed(body)
}

// resetFearGreedCache — test hook to clear the cache between cases. Not used
// in the hot path.
func resetFearGreedCache() {
	fearGreedMu.Lock()
	fearGreedCache.value = 0
	fearGreedCache.label = ""
	fearGreedCache.at = time.Time{}
	fearGreedMu.Unlock()
}
