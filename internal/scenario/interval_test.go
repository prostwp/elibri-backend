package scenario

import (
	"testing"
	"time"
)

// TestTFToPoll locks in the poll-interval mapping so an accidental swap
// (e.g. 5m → 5min poll) would fail. Lower-bound on poll interval also
// matters for Binance rate limits — 1m poll on a 5m bar = 5× unnecessary
// requests. Upper-bound matters for freshness (1d bar should be polled
// at least once per hour so a late-closing bar isn't missed by up to 24h).
func TestTFToPoll(t *testing.T) {
	t.Parallel()

	cases := []struct {
		interval string
		want     time.Duration
	}{
		{"5m", 30 * time.Second},
		{"15m", 60 * time.Second},
		{"1h", 5 * time.Minute},
		{"4h", 15 * time.Minute},
		{"1d", 1 * time.Hour},

		// Unknown intervals fall back to 5 min — NOT a panic.
		{"unknown", 5 * time.Minute},
		{"", 5 * time.Minute},
		{"30m", 5 * time.Minute},
		{"1w", 5 * time.Minute},
	}
	for _, tc := range cases {
		t.Run(tc.interval, func(t *testing.T) {
			if got := TFToPoll(tc.interval); got != tc.want {
				t.Errorf("TFToPoll(%q) = %v, want %v", tc.interval, got, tc.want)
			}
		})
	}
}
