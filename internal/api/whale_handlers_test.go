package api

// whale_handlers_test.go — offline tests for handleWhaleFlowList.
//
// Relies on the whale.SnapshotReader interface so the DB layer is stubbed
// entirely. Production wiring uses whale.NewStore(store.Pool); tests use
// fakeWhaleReader.
//
// Coverage:
//   - empty store → 200 with captured_at="" and flows=[]/transfers=[].
//   - limit param → forwarded to RecentTransfers, clamped to [1,100].
//   - invalid limit (abc, 0, 999) → silently falls back to default 50.
//   - upstream snapshot error → 500 envelope.
//   - upstream transfers error → 500 envelope.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/internal/whale"
)

// fakeWhaleReader implements whale.SnapshotReader. capturedLimit records the
// limit the handler forwarded to RecentTransfers so the clamping tests can
// assert it.
type fakeWhaleReader struct {
	snaps         []whale.Snapshot
	transfers     []whale.WhaleTransfer
	snapsErr      error
	transfersErr  error
	capturedLimit int
}

func (f *fakeWhaleReader) LatestSnapshots(_ context.Context) ([]whale.Snapshot, error) {
	if f.snapsErr != nil {
		return nil, f.snapsErr
	}
	out := make([]whale.Snapshot, len(f.snaps))
	copy(out, f.snaps)
	return out, nil
}

func (f *fakeWhaleReader) RecentTransfers(_ context.Context, limit int) ([]whale.WhaleTransfer, error) {
	f.capturedLimit = limit
	if f.transfersErr != nil {
		return nil, f.transfersErr
	}
	out := make([]whale.WhaleTransfer, len(f.transfers))
	copy(out, f.transfers)
	return out, nil
}

// Compile-time guard: the fake must satisfy the same interface the handler
// depends on, so tests exercise the production code path.
var _ whale.SnapshotReader = (*fakeWhaleReader)(nil)

// whaleResponse mirrors the handler's anonymous response struct so the test
// can inspect all three keys.
type whaleResponse struct {
	CapturedAt string                `json:"captured_at"`
	Flows      []whale.Snapshot      `json:"flows"`
	Transfers  []whale.WhaleTransfer `json:"transfers"`
}

func runWhaleRequest(t *testing.T, reader whale.SnapshotReader, query string) (int, whaleResponse) {
	t.Helper()
	url := "/api/v1/whale-flow"
	if query != "" {
		url += "?" + query
	}
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	handleWhaleFlowList(reader)(rec, req)

	res := rec.Result()
	defer res.Body.Close()

	var body whaleResponse
	if res.StatusCode == http.StatusOK {
		if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
			t.Fatalf("decode body: %v", err)
		}
	}
	return res.StatusCode, body
}

func whaleFixtureSnapshots() []whale.Snapshot {
	now := time.Date(2026, 5, 30, 0, 49, 0, 0, time.UTC)
	return []whale.Snapshot{
		{
			Asset: "ETH", CapturedAt: now,
			NetFlowUSD24h: -1_850_000, Direction: whale.FlowOutflow,
			InflowUSD24h: 4_200_000, OutflowUSD24h: 6_050_000,
			TxCount24h: 87, Confidence: 64,
			ExchangeBreakdown: map[string]int{"Binance": 12, "Coinbase": 5},
			Reasons:           []string{"Coins leaving exchanges outpaced deposits over 24h."},
			Source:            "etherscan", Partial: true,
		},
		{
			Asset: "BTC", CapturedAt: now.Add(-1 * time.Minute),
			NetFlowUSD24h: 0, Direction: whale.FlowNeutral,
			TxCount24h: 20, Confidence: 0,
			ExchangeBreakdown: map[string]int{},
			Reasons:           []string{},
			Source:            "mempool", Partial: true,
		},
	}
}

func whaleFixtureTransfers() []whale.WhaleTransfer {
	now := time.Date(2026, 5, 30, 0, 47, 11, 0, time.UTC)
	return []whale.WhaleTransfer{
		{
			Chain: whale.ChainETH, TxHash: "0xabc", Timestamp: now,
			Asset: "USDT", AmountNative: 5_000_000, AmountUSD: 5_000_000,
			Direction: whale.FlowInflow, Exchange: "Binance",
		},
		{
			Chain: whale.ChainBTC, TxHash: "0xdef", Timestamp: now.Add(-30 * time.Second),
			Asset: "BTC", AmountNative: 250, AmountUSD: 0,
			Direction: whale.FlowNeutral, Exchange: "",
		},
	}
}

func TestWhaleFlowList_EmptyStore(t *testing.T) {
	reader := &fakeWhaleReader{}
	code, body := runWhaleRequest(t, reader, "")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if body.CapturedAt != "" {
		t.Errorf("captured_at = %q, want empty string", body.CapturedAt)
	}
	if body.Flows == nil {
		t.Errorf("flows is nil — must serialise as empty array, not null")
	}
	if body.Transfers == nil {
		t.Errorf("transfers is nil — must serialise as empty array, not null")
	}
	if len(body.Flows) != 0 || len(body.Transfers) != 0 {
		t.Errorf("flows=%d transfers=%d, want 0/0", len(body.Flows), len(body.Transfers))
	}
}

func TestWhaleFlowList_PopulatedAndCapturedAt(t *testing.T) {
	reader := &fakeWhaleReader{
		snaps:     whaleFixtureSnapshots(),
		transfers: whaleFixtureTransfers(),
	}
	code, body := runWhaleRequest(t, reader, "")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if len(body.Flows) != 2 {
		t.Errorf("flows len = %d, want 2", len(body.Flows))
	}
	if len(body.Transfers) != 2 {
		t.Errorf("transfers len = %d, want 2", len(body.Transfers))
	}
	// captured_at = newest snapshot's captured_at (ETH at 00:49:00Z).
	if body.CapturedAt != "2026-05-30T00:49:00Z" {
		t.Errorf("captured_at = %q, want 2026-05-30T00:49:00Z", body.CapturedAt)
	}
	// Default limit should have been forwarded.
	if reader.capturedLimit != whaleTransfersDefaultLimit {
		t.Errorf("forwarded limit = %d, want default %d", reader.capturedLimit, whaleTransfersDefaultLimit)
	}
}

func TestWhaleFlowList_LimitClamping(t *testing.T) {
	cases := []struct {
		query     string
		wantLimit int
	}{
		{"limit=10", 10},   // valid → forwarded as-is
		{"limit=100", 100}, // boundary max → forwarded
		{"limit=1", 1},     // boundary min → forwarded
		{"limit=abc", whaleTransfersDefaultLimit}, // non-numeric → default
		{"limit=0", whaleTransfersDefaultLimit},   // below min → default
		{"limit=999", whaleTransfersDefaultLimit}, // above max → default
		{"limit=-5", whaleTransfersDefaultLimit},  // negative → default
		{"", whaleTransfersDefaultLimit},          // absent → default
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.query, func(t *testing.T) {
			reader := &fakeWhaleReader{
				snaps:     whaleFixtureSnapshots(),
				transfers: whaleFixtureTransfers(),
			}
			code, _ := runWhaleRequest(t, reader, tc.query)
			if code != http.StatusOK {
				t.Fatalf("status = %d, want 200", code)
			}
			if reader.capturedLimit != tc.wantLimit {
				t.Errorf("query %q → forwarded limit %d, want %d", tc.query, reader.capturedLimit, tc.wantLimit)
			}
		})
	}
}

func TestWhaleFlowList_SnapshotsError(t *testing.T) {
	reader := &fakeWhaleReader{snapsErr: errors.New("simulated PG outage")}
	req := httptest.NewRequest(http.MethodGet, "/api/v1/whale-flow", nil)
	rec := httptest.NewRecorder()
	handleWhaleFlowList(reader)(rec, req)
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

func TestWhaleFlowList_TransfersError(t *testing.T) {
	// Snapshots succeed but transfers fail — still a 500 (the feed is part of
	// the contract; a half response would mislead the UI).
	reader := &fakeWhaleReader{
		snaps:        whaleFixtureSnapshots(),
		transfersErr: errors.New("simulated PG outage on transfers"),
	}
	req := httptest.NewRequest(http.MethodGet, "/api/v1/whale-flow", nil)
	rec := httptest.NewRecorder()
	handleWhaleFlowList(reader)(rec, req)
	res := rec.Result()
	defer res.Body.Close()

	if res.StatusCode != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", res.StatusCode)
	}
}

var _ = time.Second
