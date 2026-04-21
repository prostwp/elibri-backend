package macrocal

import (
	"testing"
	"time"
)

func TestIsBlackout(t *testing.T) {
	t.Parallel()

	// Install a fake event cache.
	fomcAt := time.Date(2026, 5, 1, 18, 0, 0, 0, time.UTC)
	cpiAt := time.Date(2026, 5, 2, 12, 30, 0, 0, time.UTC)
	mediumUS := time.Date(2026, 5, 1, 14, 0, 0, 0, time.UTC)
	euEvent := time.Date(2026, 5, 1, 12, 0, 0, 0, time.UTC)

	cacheMu.Lock()
	cacheData = []Event{
		{Country: "US", Event: "FOMC Statement", Impact: "high", Time: fomcAt},
		{Country: "US", Event: "CPI YoY", Impact: "high", Time: cpiAt},
		{Country: "US", Event: "Retail Sales", Impact: "medium", Time: mediumUS},
		{Country: "EU", Event: "ECB Rate Decision", Impact: "high", Time: euEvent},
		{Country: "", Event: "Fed Chair Powell speaks", Impact: "high", Time: fomcAt.Add(2 * time.Hour)},
	}
	cacheMu.Unlock()

	cfg := DefaultConfig() // blackout 30m before / 15m after, impact=high

	cases := []struct {
		name    string
		at      time.Time
		blocked bool
		event   string
	}{
		{
			name:    "45m before FOMC — not yet blacked out",
			at:      fomcAt.Add(-45 * time.Minute),
			blocked: false,
		},
		{
			name:    "29m before FOMC — blocked",
			at:      fomcAt.Add(-29 * time.Minute),
			blocked: true,
			event:   "FOMC Statement",
		},
		{
			name:    "at FOMC — blocked",
			at:      fomcAt,
			blocked: true,
			event:   "FOMC Statement",
		},
		{
			name:    "14m after FOMC — still blocked",
			at:      fomcAt.Add(14 * time.Minute),
			blocked: true,
			event:   "FOMC Statement",
		},
		{
			name:    "16m after FOMC — cleared",
			at:      fomcAt.Add(16 * time.Minute),
			blocked: false,
		},
		{
			name:    "medium-impact US — should NOT block (below threshold)",
			at:      mediumUS,
			blocked: false,
		},
		{
			name:    "EU event — should NOT block (we filter US only)",
			at:      euEvent,
			blocked: false,
		},
		{
			name:    "Powell speech caught by fedByName even with empty country",
			at:      fomcAt.Add(2 * time.Hour),
			blocked: true,
			event:   "Fed Chair Powell speaks",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := IsBlackout(tc.at, cfg)
			if got.Blocked != tc.blocked {
				t.Errorf("blocked: got %v want %v (event=%q)", got.Blocked, tc.blocked, got.Event)
			}
			if tc.event != "" && got.Event != tc.event {
				t.Errorf("event: got %q want %q", got.Event, tc.event)
			}
		})
	}
}

func TestDisabledConfigNeverBlocks(t *testing.T) {
	cacheMu.Lock()
	cacheData = []Event{
		{Country: "US", Event: "FOMC", Impact: "high", Time: time.Now()},
	}
	cacheMu.Unlock()

	cfg := DefaultConfig()
	cfg.Enabled = false

	if got := IsBlackout(time.Now(), cfg); got.Blocked {
		t.Error("disabled config must never return blocked")
	}
}

func TestImpactThreshold(t *testing.T) {
	now := time.Now().UTC().Add(5 * time.Minute)
	cacheMu.Lock()
	cacheData = []Event{
		{Country: "US", Event: "Low event", Impact: "low", Time: now},
	}
	cacheMu.Unlock()

	// With default high filter, low event shouldn't block.
	if IsBlackout(now, DefaultConfig()).Blocked {
		t.Error("low impact must not block under high filter")
	}

	// With low filter, it should block.
	cfg := DefaultConfig()
	cfg.MinImpact = "low"
	if !IsBlackout(now, cfg).Blocked {
		t.Error("low impact must block under low filter")
	}
}

func TestUpcomingEventsSorted(t *testing.T) {
	now := time.Now().UTC()
	cacheMu.Lock()
	cacheData = []Event{
		{Country: "US", Event: "Event B", Impact: "high", Time: now.Add(5 * time.Hour)},
		{Country: "US", Event: "Event A", Impact: "high", Time: now.Add(2 * time.Hour)},
		{Country: "US", Event: "Event C", Impact: "high", Time: now.Add(48 * time.Hour)},
		{Country: "US", Event: "Past", Impact: "high", Time: now.Add(-2 * time.Hour)},
	}
	cacheMu.Unlock()

	got := UpcomingEvents(24, DefaultConfig())
	if len(got) != 2 {
		t.Fatalf("expected 2 events in 24h window, got %d", len(got))
	}
	if got[0].Event != "Event A" || got[1].Event != "Event B" {
		t.Errorf("not sorted: %v", got)
	}
}
