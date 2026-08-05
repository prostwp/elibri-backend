package api

// funding_handlers_test.go — offline tests for handleFundingLiquidations.
//
// Relies on the funding.LiquidationReader interface so the WS/store layer is
// stubbed entirely. Production wiring uses api.SetFundingStore(funding.NewStore());
// tests set the package global to a fakeFundingReader (and reset it after).
//
// Coverage:
//   - nil store (never wired) → 200 with captured_at="" and feed=[]/zones=[].
//   - empty store → 200 with captured_at="" and []/[].
//   - populated → captured_at set, feed/zones forwarded.
//   - limit param → forwarded to Recent, clamped to [1,200]; invalid → default 40.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/internal/funding"
)

// fakeFundingReader implements funding.LiquidationReader. capturedLimit records
// the limit the handler forwarded to Recent so the clamping tests can assert it.
type fakeFundingReader struct {
	feed          []funding.Liquidation
	zones         []funding.Zone
	capturedLimit int
}

func (f *fakeFundingReader) Recent(limit int) []funding.Liquidation {
	f.capturedLimit = limit
	out := make([]funding.Liquidation, len(f.feed))
	copy(out, f.feed)
	return out
}

func (f *fakeFundingReader) Zones() []funding.Zone {
	out := make([]funding.Zone, len(f.zones))
	copy(out, f.zones)
	return out
}

var _ funding.LiquidationReader = (*fakeFundingReader)(nil)

// withFundingStore sets the package global for the duration of the test and
// restores it afterward (the global is process-wide; leaking it would poison
// sibling tests).
func withFundingStore(t *testing.T, r funding.LiquidationReader) {
	t.Helper()
	prev := fundingStore
	fundingStore = r
	t.Cleanup(func() { fundingStore = prev })
}

func runFundingRequest(t *testing.T, query string) (int, funding.Response) {
	t.Helper()
	url := "/api/v1/funding/liquidations"
	if query != "" {
		url += "?" + query
	}
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	handleFundingLiquidations(rec, req)

	res := rec.Result()
	defer res.Body.Close()

	var body funding.Response
	if res.StatusCode == http.StatusOK {
		if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
			t.Fatalf("decode body: %v", err)
		}
	}
	return res.StatusCode, body
}

func TestFundingLiquidations_NilStore(t *testing.T) {
	withFundingStore(t, nil) // explicitly unset
	code, body := runFundingRequest(t, "")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if body.CapturedAt != "" {
		t.Errorf("captured_at = %q, want empty string", body.CapturedAt)
	}
	if body.Feed == nil || body.Zones == nil {
		t.Errorf("feed/zones must serialise as [] not null (feed=%v zones=%v)", body.Feed, body.Zones)
	}
	if len(body.Feed) != 0 || len(body.Zones) != 0 {
		t.Errorf("feed=%d zones=%d, want 0/0", len(body.Feed), len(body.Zones))
	}
}

func TestFundingLiquidations_EmptyStore(t *testing.T) {
	withFundingStore(t, &fakeFundingReader{})
	code, body := runFundingRequest(t, "")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	// A wired store means the worker is live, so even a quiet (empty) window
	// stamps captured_at = now() — only the nil-store path returns "". Verify
	// it's a present, well-formed RFC3339 timestamp.
	if body.CapturedAt == "" {
		t.Errorf("captured_at = %q, want a non-empty RFC3339 timestamp for a wired store", body.CapturedAt)
	} else if _, err := time.Parse(time.RFC3339, body.CapturedAt); err != nil {
		t.Errorf("captured_at = %q, not valid RFC3339: %v", body.CapturedAt, err)
	}
	if len(body.Feed) != 0 || len(body.Zones) != 0 {
		t.Errorf("feed=%d zones=%d, want 0/0", len(body.Feed), len(body.Zones))
	}
}

func TestFundingLiquidations_Populated(t *testing.T) {
	reader := &fakeFundingReader{
		feed: []funding.Liquidation{
			{Symbol: "SOLUSDT", Side: funding.SideLongLiq, Qty: 0.014, Price: 9910, USDValue: 138.74, TS: "2026-05-30T00:48:51Z"},
		},
		zones: []funding.Zone{
			{Symbol: "SOLUSDT", PriceBand: "9900-9950", TotalUSD: 412000, Count: 37, Side: funding.SideLongLiq},
		},
	}
	withFundingStore(t, reader)
	code, body := runFundingRequest(t, "")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if len(body.Feed) != 1 {
		t.Errorf("feed len = %d, want 1", len(body.Feed))
	}
	if len(body.Zones) != 1 {
		t.Errorf("zones len = %d, want 1", len(body.Zones))
	}
	// captured_at must be set (non-empty) when the window has data.
	if body.CapturedAt == "" {
		t.Errorf("captured_at empty, want a UTC timestamp when feed/zones non-empty")
	}
	// Default limit forwarded.
	if reader.capturedLimit != fundingFeedDefaultLimit {
		t.Errorf("forwarded limit = %d, want default %d", reader.capturedLimit, fundingFeedDefaultLimit)
	}
}

func TestFundingLiquidations_LimitClamping(t *testing.T) {
	cases := []struct {
		query     string
		wantLimit int
	}{
		{"limit=10", 10},   // valid
		{"limit=200", 200}, // boundary max
		{"limit=1", 1},     // boundary min
		{"limit=abc", fundingFeedDefaultLimit}, // non-numeric → default
		{"limit=0", fundingFeedDefaultLimit},   // below min → default
		{"limit=201", fundingFeedDefaultLimit}, // above max → default
		{"limit=-5", fundingFeedDefaultLimit},  // negative → default
		{"", fundingFeedDefaultLimit},          // absent → default
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.query, func(t *testing.T) {
			reader := &fakeFundingReader{}
			withFundingStore(t, reader)
			code, _ := runFundingRequest(t, tc.query)
			if code != http.StatusOK {
				t.Fatalf("status = %d, want 200", code)
			}
			if reader.capturedLimit != tc.wantLimit {
				t.Errorf("query %q → forwarded limit %d, want %d", tc.query, reader.capturedLimit, tc.wantLimit)
			}
		})
	}
}
