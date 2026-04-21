package macrocal

import (
	"strings"
	"time"
)

// Default blackout windows (override via env in config.go wiring).
// Picked based on typical vol-expansion profile around HIGH-impact USD
// releases: signal is garbage ~30 min before (positioning, stop runs)
// and ~15 min after (initial reaction fades, second wave starts).
var (
	DefaultBlackoutBefore = 30 * time.Minute
	DefaultBlackoutAfter  = 15 * time.Minute
	DefaultImpactFilter   = "high" // "low" | "medium" | "high" — minimum impact to trigger blackout
)

// Config tunables — can be overridden from main.go via env.
type Config struct {
	Enabled         bool
	BlackoutBefore  time.Duration
	BlackoutAfter   time.Duration
	MinImpact       string // "high" | "medium" | "low"
}

// DefaultConfig is used when the caller doesn't override anything.
// Enabled=true by default — paper-trading should respect macro releases.
func DefaultConfig() Config {
	return Config{
		Enabled:        true,
		BlackoutBefore: DefaultBlackoutBefore,
		BlackoutAfter:  DefaultBlackoutAfter,
		MinImpact:      DefaultImpactFilter,
	}
}

// GateResult explains why a blackout decision was made so the UI / logs
// can surface something meaningful instead of just true/false.
type GateResult struct {
	Blocked   bool
	Event     string    // the specific release that triggered the block
	EventTime time.Time // its scheduled time (UTC)
	Minutes   int       // signed minutes — negative = before, positive = after
}

// IsBlackout answers "is `at` within a blackout window around a HIGH-impact
// USD event?". Crypto prices are dominated by USD macro regardless of the
// instrument (BTC, ETH, etc.) — so we filter by country=US AND impact=high
// rather than per-symbol.
//
// Returns Blocked=false when the package wasn't Init'd (no API key).
// Iteration cost: O(N) on ~50-200 cached events; called at most once per
// scenario tick (every 30s on 5m TF, less on higher TFs) → negligible.
func IsBlackout(at time.Time, cfg Config) GateResult {
	if !cfg.Enabled {
		return GateResult{}
	}
	if cfg.BlackoutBefore <= 0 {
		cfg.BlackoutBefore = DefaultBlackoutBefore
	}
	if cfg.BlackoutAfter <= 0 {
		cfg.BlackoutAfter = DefaultBlackoutAfter
	}
	if cfg.MinImpact == "" {
		cfg.MinImpact = DefaultImpactFilter
	}

	events := Snapshot()
	atUTC := at.UTC()

	for _, e := range events {
		if !isRelevant(e, cfg.MinImpact) {
			continue
		}
		diff := atUTC.Sub(e.Time)
		// Blackout window: [event - before, event + after].
		//   diff < 0  → before the event (diff == -25min → 25 before)
		//   diff > 0  → after the event
		// We block when -before <= diff <= +after.
		if diff >= -cfg.BlackoutBefore && diff <= cfg.BlackoutAfter {
			minutes := int(diff.Minutes())
			return GateResult{
				Blocked:   true,
				Event:     e.Event,
				EventTime: e.Time,
				Minutes:   minutes,
			}
		}
	}
	return GateResult{}
}

// isRelevant applies the crypto-trading lens: US country + minimum impact
// threshold. We also whitelist Fed-speaker events which Finnhub sometimes
// tags with variable country codes.
func isRelevant(e Event, minImpact string) bool {
	if !matchCountry(e.Country) {
		// Fed events sometimes have empty country — catch them by name.
		if !fedByName(e.Event) {
			return false
		}
	}
	return impactAtLeast(e.Impact, minImpact)
}

// matchCountry treats US as the single authoritative macro source for
// crypto blackouts. ECB/BoJ/China are HIGH-impact too but the market
// impact on BTC/ETH is ~3-5× weaker; adding them would over-suppress
// signals. Future: per-symbol allowlist.
func matchCountry(country string) bool {
	return strings.EqualFold(country, "US")
}

// fedByName catches Fed speakers + FOMC where Finnhub sometimes leaves
// Country empty or tags with the speaker's home city. Uses strong
// Fed-specific keywords only — generic terms like "rate decision"
// or "rate statement" are excluded because ECB, BoE, BoJ, SNB all
// have their own rate decisions and we don't want to block on those
// (crypto reacts much weaker to non-USD macro).
func fedByName(name string) bool {
	lower := strings.ToLower(name)
	for _, kw := range []string{
		"fomc", "federal reserve", "fed chair", "fed speaker",
		"powell", "waller", "bowman", "kashkari", "bostic",
	} {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

// impactAtLeast is a total order high > medium > low. Missing impact is
// treated as "low" (don't block on it).
func impactAtLeast(actual, threshold string) bool {
	rank := map[string]int{"low": 0, "medium": 1, "high": 2, "": 0}
	a, okA := rank[strings.ToLower(actual)]
	t, okT := rank[strings.ToLower(threshold)]
	if !okA || !okT {
		return false
	}
	return a >= t
}

// UpcomingEvents returns events within the next N hours that match the
// config's relevance filter. Used by the /api/v1/macrocal endpoint and
// Toolbar chip. Sorted ascending by time.
func UpcomingEvents(hours int, cfg Config) []Event {
	if cfg.MinImpact == "" {
		cfg.MinImpact = DefaultImpactFilter
	}
	cutoff := time.Now().UTC().Add(time.Duration(hours) * time.Hour)
	out := []Event{}
	for _, e := range Snapshot() {
		if !isRelevant(e, cfg.MinImpact) {
			continue
		}
		if e.Time.Before(time.Now().UTC()) {
			continue
		}
		if e.Time.After(cutoff) {
			continue
		}
		out = append(out, e)
	}
	// Events come from Finnhub already time-sorted, but be defensive.
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j].Time.Before(out[j-1].Time); j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}
