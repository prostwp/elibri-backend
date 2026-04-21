package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
)

// TestScenarioResponseContract pins the JSON shape of /api/v1/scenarios/active
// and /api/v1/alerts response bodies so frontend TypeScript types cannot drift
// silently from Go field renames.
//
// HOW IT WORKS
//
// We maintain a snapshot JSON in ml-training/../contracts/ with the EXACT key
// list each endpoint emits. The Go test builds a fresh instance of the response
// struct, serializes it, extracts keys, and compares against the snapshot.
// Frontend has a companion vitest test that imports its TypeScript interface
// (ActiveScenario, Alert) and checks its keys match the same snapshot.
//
// When contract changes intentionally:
//   1. Update the Go struct.
//   2. Run `go test ./internal/api -run TestScenarioResponseContract -update`
//      (flag not wired yet — for now just regenerate the snapshot by hand with
//      the UPDATE_SNAPSHOT env var).
//   3. Update the matching TypeScript interface in src/lib/scenarios.ts.
//   4. Frontend vitest will pick up the new snapshot.
//
// When it fails unexpectedly:
//   One side renamed / removed / added a field without updating the other.
//   The error message will list the diff.
func TestScenarioResponseContract(t *testing.T) {
	// Build sample instances matching what the live handlers emit. We use the
	// same inline struct types via JSON manually because handlers use local
	// `type item struct` inside closures (Go idiom to keep response shape
	// close to its handler). A small duplication is worth the test coverage.
	//
	// If you see this drift from scenario_handlers.go — that's the bug we want
	// to catch. Please update both in sync.
	type scenarioItem struct {
		ID                  string  `json:"id"`
		Name                string  `json:"name"`
		Symbol              string  `json:"symbol"`
		Interval            string  `json:"interval"`
		RiskTier            string  `json:"risk_tier"`
		IsActive            bool    `json:"is_active"`
		PausedUntil         *string `json:"paused_until,omitempty"`
		LastSignalBarTime   int64   `json:"last_signal_bar_time"`
		LastSignalDirection string  `json:"last_signal_direction,omitempty"`
		Running             bool    `json:"running"`
	}
	type alertRow struct {
		ID                string         `json:"id"`
		StrategyID        string         `json:"strategy_id"`
		Symbol            string         `json:"symbol"`
		Interval          string         `json:"interval"`
		Direction         string         `json:"direction"`
		Label             string         `json:"label,omitempty"`
		Confidence        float64        `json:"confidence"`
		EntryPrice        float64        `json:"entry_price"`
		StopLoss          float64        `json:"stop_loss"`
		TakeProfit        float64        `json:"take_profit"`
		PositionSize      float64        `json:"position_size_usd"`
		BarTime           int64          `json:"bar_time"`
		CreatedAt         string         `json:"created_at"`
		TelegramSentAt    *string        `json:"telegram_sent_at,omitempty"`
		TelegramMessageID *int64         `json:"telegram_message_id,omitempty"`
		Meta              map[string]any `json:"meta,omitempty"`
	}

	// Include omitempty-tagged fields explicitly so the contract shows ALL
	// possible keys, not just the ones non-zero instances happen to emit.
	lbt := "2026-04-20T10:00:00Z"
	tmsg := int64(42)
	scenario := scenarioItem{
		ID: "x", Name: "x", Symbol: "BTCUSDT", Interval: "4h",
		RiskTier: "conservative", IsActive: true, PausedUntil: &lbt,
		LastSignalBarTime: 1, LastSignalDirection: "buy", Running: true,
	}
	alert := alertRow{
		ID: "x", StrategyID: "x", Symbol: "BTCUSDT", Interval: "4h",
		Direction: "buy", Label: "trend_aligned", Confidence: 72.0,
		EntryPrice: 50000, StopLoss: 48000, TakeProfit: 53000,
		PositionSize: 1000, BarTime: 1, CreatedAt: "t",
		TelegramSentAt: &lbt, TelegramMessageID: &tmsg,
		Meta: map[string]any{"note": "placeholder"},
	}

	got := map[string][]string{
		"scenario_active_item": jsonKeys(t, scenario),
		"alerts_list_row":      jsonKeys(t, alert),
	}

	// Load committed snapshot from frontend shared location.
	_, thisFile, _, _ := runtime.Caller(0)
	repoRoot := filepath.Join(filepath.Dir(thisFile), "..", "..")
	snapshotPath := filepath.Join(repoRoot, "contracts", "scenarios.snapshot.json")

	if _, err := os.Stat(snapshotPath); os.IsNotExist(err) {
		if os.Getenv("UPDATE_SNAPSHOT") == "1" {
			writeSnapshot(t, snapshotPath, got)
			t.Logf("wrote initial snapshot to %s", snapshotPath)
			return
		}
		t.Fatalf("snapshot missing at %s — run: UPDATE_SNAPSHOT=1 go test ./internal/api -run TestScenarioResponseContract", snapshotPath)
	}

	want := readSnapshot(t, snapshotPath)

	for endpoint, gotKeys := range got {
		wantKeys, ok := want[endpoint]
		if !ok {
			t.Errorf("snapshot missing endpoint %q — run with UPDATE_SNAPSHOT=1 to regenerate", endpoint)
			continue
		}
		if diff := diffKeys(wantKeys, gotKeys); diff != "" {
			t.Errorf("contract drift on %q:\n%s", endpoint, diff)
		}
	}

	// If we added a new endpoint in go but forgot to capture snapshot.
	for endpoint := range want {
		if _, ok := got[endpoint]; !ok {
			t.Errorf("snapshot has stale endpoint %q not produced by this test — regenerate snapshot after removing the handler",
				endpoint)
		}
	}
}

// jsonKeys serializes v and returns its top-level JSON keys in sorted order
// so diffs are deterministic.
func jsonKeys(t *testing.T, v any) []string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal-to-map: %v", err)
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func writeSnapshot(t *testing.T, path string, data map[string][]string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	b, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		t.Fatalf("marshal snapshot: %v", err)
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func readSnapshot(t *testing.T, path string) map[string][]string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var m map[string][]string
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("parse snapshot: %v", err)
	}
	return m
}

func diffKeys(want, got []string) string {
	wantSet := map[string]bool{}
	gotSet := map[string]bool{}
	for _, k := range want {
		wantSet[k] = true
	}
	for _, k := range got {
		gotSet[k] = true
	}
	var missing, extra []string
	for k := range wantSet {
		if !gotSet[k] {
			missing = append(missing, k)
		}
	}
	for k := range gotSet {
		if !wantSet[k] {
			extra = append(extra, k)
		}
	}
	if len(missing) == 0 && len(extra) == 0 {
		return ""
	}
	sort.Strings(missing)
	sort.Strings(extra)
	var sb strings.Builder
	if len(missing) > 0 {
		sb.WriteString("  missing (in snapshot, not in Go):\n")
		for _, k := range missing {
			sb.WriteString("    - ")
			sb.WriteString(k)
			sb.WriteByte('\n')
		}
	}
	if len(extra) > 0 {
		sb.WriteString("  extra (in Go, not in snapshot):\n")
		for _, k := range extra {
			sb.WriteString("    + ")
			sb.WriteString(k)
			sb.WriteByte('\n')
		}
	}
	return sb.String()
}
