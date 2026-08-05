package api

// prefs_handlers_test.go — offline coverage for the pure watchlist + notif
// pref helpers (normalizeWatchlist, isValidFrequency, defaultNotificationPrefs,
// mergeNotificationPrefs). No Postgres needed.

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestNormalizeWatchlist(t *testing.T) {
	tests := []struct {
		name string
		in   []string
		want []string
	}{
		{"empty", nil, []string{}},
		{"trim+upper", []string{" btcusdt ", "ethusdt"}, []string{"BTCUSDT", "ETHUSDT"}},
		{"dedup preserves first order", []string{"BTC", "eth", "btc", "ETH"}, []string{"BTC", "ETH"}},
		{"drop empties", []string{"", "   ", "BTC"}, []string{"BTC"}},
		{"drop oversized (>20 chars)", []string{strings.Repeat("A", 21), "OK"}, []string{"OK"}},
		{"keep exactly 20 chars", []string{strings.Repeat("A", 20)}, []string{strings.Repeat("A", 20)}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := normalizeWatchlist(tc.in)
			if len(got) != len(tc.want) {
				t.Fatalf("len = %d (%v), want %d (%v)", len(got), got, len(tc.want), tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("idx %d = %q, want %q (full: %v)", i, got[i], tc.want[i], got)
				}
			}
		})
	}
}

func TestNormalizeWatchlist_CapsAt50(t *testing.T) {
	in := make([]string, 0, 80)
	for i := 0; i < 80; i++ {
		// distinct symbols so dedup doesn't shrink the input
		in = append(in, "SYM"+strings.Repeat("X", i%5)+itoa(i))
	}
	got := normalizeWatchlist(in)
	if len(got) > maxWatchlistSymbols {
		t.Fatalf("normalizeWatchlist returned %d symbols, want <= %d", len(got), maxWatchlistSymbols)
	}
	if len(got) != maxWatchlistSymbols {
		t.Fatalf("expected exactly %d after cap, got %d", maxWatchlistSymbols, len(got))
	}
}

func TestIsValidFrequency(t *testing.T) {
	valid := []string{"realtime", "hourly", "daily"}
	for _, v := range valid {
		if !isValidFrequency(v) {
			t.Errorf("isValidFrequency(%q) = false, want true", v)
		}
	}
	invalid := []string{"", "REALTIME", "weekly", "instant", "Daily"}
	for _, v := range invalid {
		if isValidFrequency(v) {
			t.Errorf("isValidFrequency(%q) = true, want false", v)
		}
	}
}

func TestDefaultNotificationPrefs_Shape(t *testing.T) {
	d := defaultNotificationPrefs()

	ch, ok := d["channels"].(map[string]any)
	if !ok {
		t.Fatalf("channels missing/wrong type: %#v", d["channels"])
	}
	if ch["email"] != true || ch["dashboard"] != true {
		t.Errorf("channels email/dashboard should default true: %#v", ch)
	}
	if ch["telegram"] != false {
		t.Errorf("channels.telegram should default false: %#v", ch)
	}
	if d["daily_digest"] != false {
		t.Errorf("daily_digest default = %v, want false", d["daily_digest"])
	}
	if d["critical_alerts"] != true {
		t.Errorf("critical_alerts default = %v, want true", d["critical_alerts"])
	}
	if d["frequency"] != "realtime" {
		t.Errorf("frequency default = %v, want realtime", d["frequency"])
	}
	ps, ok := d["per_scenario"].(map[string]any)
	if !ok || len(ps) != 0 {
		t.Errorf("per_scenario default should be empty object: %#v", d["per_scenario"])
	}
}

func TestMergeNotificationPrefs_PreservesUnknownKeys(t *testing.T) {
	// Forward-compat: a key sub-chat F adds and that lives in base must
	// survive a PUT that doesn't mention it.
	base := defaultNotificationPrefs()
	base["future_key_from_subchat_F"] = "keepme"

	incoming := map[string]any{"daily_digest": true}
	merged := mergeNotificationPrefs(base, incoming)

	if merged["future_key_from_subchat_F"] != "keepme" {
		t.Errorf("merge clobbered unknown key: %#v", merged["future_key_from_subchat_F"])
	}
	if merged["daily_digest"] != true {
		t.Errorf("merge did not apply daily_digest=true: %#v", merged["daily_digest"])
	}
	// scalar that wasn't touched stays default
	if merged["critical_alerts"] != true {
		t.Errorf("untouched critical_alerts changed: %#v", merged["critical_alerts"])
	}
}

func TestMergeNotificationPrefs_ChannelsMergePerKey(t *testing.T) {
	// A PUT of just {channels:{email:false}} must NOT wipe telegram/dashboard.
	base := defaultNotificationPrefs()
	incoming := map[string]any{
		"channels": map[string]any{"email": false},
	}
	merged := mergeNotificationPrefs(base, incoming)

	ch, ok := merged["channels"].(map[string]any)
	if !ok {
		t.Fatalf("channels wrong type after merge: %#v", merged["channels"])
	}
	if ch["email"] != false {
		t.Errorf("channels.email = %v, want false (the PUT value)", ch["email"])
	}
	if ch["dashboard"] != true {
		t.Errorf("channels.dashboard = %v, want true (preserved)", ch["dashboard"])
	}
	if _, present := ch["telegram"]; !present {
		t.Errorf("channels.telegram was dropped by merge: %#v", ch)
	}
}

func TestMergeNotificationPrefs_PerScenarioMerge(t *testing.T) {
	base := defaultNotificationPrefs()
	base["per_scenario"] = map[string]any{"s1": true}
	incoming := map[string]any{
		"per_scenario": map[string]any{"s2": false},
	}
	merged := mergeNotificationPrefs(base, incoming)
	ps, ok := merged["per_scenario"].(map[string]any)
	if !ok {
		t.Fatalf("per_scenario wrong type: %#v", merged["per_scenario"])
	}
	if ps["s1"] != true {
		t.Errorf("per_scenario.s1 lost in merge: %#v", ps)
	}
	if ps["s2"] != false {
		t.Errorf("per_scenario.s2 not applied: %#v", ps)
	}
}

func TestMergeNotificationPrefs_FrequencyOverwrites(t *testing.T) {
	base := defaultNotificationPrefs()
	incoming := map[string]any{"frequency": "hourly"}
	merged := mergeNotificationPrefs(base, incoming)
	if merged["frequency"] != "hourly" {
		t.Errorf("frequency = %v, want hourly", merged["frequency"])
	}
}

func TestValidateObjectFields(t *testing.T) {
	// Each case is what a JSON body would decode into via encoding/json.
	cases := []struct {
		name    string
		in      map[string]any
		wantBad string // "" means "valid"
		wantOK  bool
	}{
		{"empty body", map[string]any{}, "", true},
		{"absent object keys (partial PUT)", map[string]any{"daily_digest": true}, "", true},
		{"valid channels object", map[string]any{"channels": map[string]any{"email": false}}, "", true},
		{"valid per_scenario object", map[string]any{"per_scenario": map[string]any{"s1": true}}, "", true},
		{"both valid objects", map[string]any{"channels": map[string]any{}, "per_scenario": map[string]any{}}, "", true},
		// The defect this fix closes: null decodes to a nil interface, NOT a map.
		{"channels null", map[string]any{"channels": nil}, "channels", false},
		{"per_scenario null", map[string]any{"per_scenario": nil}, "per_scenario", false},
		// Scalars / arrays must also be rejected.
		{"channels scalar string", map[string]any{"channels": "nope"}, "channels", false},
		{"channels number", map[string]any{"channels": float64(1)}, "channels", false},
		{"channels bool", map[string]any{"channels": true}, "channels", false},
		{"per_scenario array", map[string]any{"per_scenario": []any{"x"}}, "per_scenario", false},
		// channels valid, per_scenario bad → reports per_scenario (channels passed).
		{"per_scenario bad after good channels", map[string]any{"channels": map[string]any{}, "per_scenario": nil}, "per_scenario", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			badKey, ok := validateObjectFields(tc.in)
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v (badKey=%q)", ok, tc.wantOK, badKey)
			}
			if badKey != tc.wantBad {
				t.Fatalf("badKey = %q, want %q", badKey, tc.wantBad)
			}
		})
	}
}

// TestNotificationPrefs_NullChannelsDecodesToNil pins the std-lib assumption
// the fix relies on: a JSON `null` for channels unmarshals to a nil interface
// (not a map), so validateObjectFields catches it. This is the real decode
// path a PUT {"channels":null} takes.
func TestNotificationPrefs_NullChannelsDecodesToNil(t *testing.T) {
	incoming := map[string]any{}
	if err := json.Unmarshal([]byte(`{"channels": null, "daily_digest": true}`), &incoming); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if v, present := incoming["channels"]; !present || v != nil {
		t.Fatalf("channels = %#v (present=%v), want nil interface", v, present)
	}
	if badKey, ok := validateObjectFields(incoming); ok || badKey != "channels" {
		t.Fatalf("validateObjectFields = (%q,%v), want (channels,false)", badKey, ok)
	}
}

// TestMergeNotificationPrefs_ValidPartialStillMergesAfterGuard is the "valid
// partial still merges correctly" assertion the review asked for: a real
// channels object passes the guard and still merges per-key against stored
// prefs (telegram/dashboard preserved).
func TestMergeNotificationPrefs_ValidPartialStillMergesAfterGuard(t *testing.T) {
	incoming := map[string]any{"channels": map[string]any{"email": false}}
	if badKey, ok := validateObjectFields(incoming); !ok {
		t.Fatalf("valid partial rejected on key %q", badKey)
	}
	merged := mergeNotificationPrefs(defaultNotificationPrefs(), incoming)
	ch, ok := merged["channels"].(map[string]any)
	if !ok {
		t.Fatalf("channels wrong type: %#v", merged["channels"])
	}
	if ch["email"] != false {
		t.Errorf("channels.email = %v, want false", ch["email"])
	}
	if ch["dashboard"] != true || ch["telegram"] != false {
		t.Errorf("untouched channels not preserved: %#v", ch)
	}
}
