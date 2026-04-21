// Package macrocal provides a lightweight economic-calendar client +
// blackout gate for the scenario runner. The goal is to avoid firing
// paper-trading signals around high-impact macro releases (FOMC, CPI,
// NFP, PCE, PPI, GDP, unemployment, Fed speakers) where volatility
// spikes and the ML model's edge collapses.
//
// Source: Finnhub /calendar/economic (free tier supports it). Polled
// once per hour via a background goroutine — we keep a 72h forward
// window in memory to answer IsBlackout() in O(N) without a network
// round-trip.
//
// Wiring: main.go starts the poller; internal/scenario/evaluate.go
// calls IsBlackout() after VolGate and before HC emission.
package macrocal

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

const finnhubBase = "https://finnhub.io/api/v1"

// Event is a single economic release (or Fed speaker, policy decision).
// Time is UTC. Empty Impact = unknown → treated as low severity.
type Event struct {
	Country  string    `json:"country"`
	Event    string    `json:"event"`
	Impact   string    `json:"impact"` // "low" | "medium" | "high"
	Time     time.Time `json:"time"`
	Estimate float64   `json:"estimate,omitempty"`
	Actual   float64   `json:"actual,omitempty"`
	Prev     float64   `json:"prev,omitempty"`
}

// Cache is the package-wide in-memory store for upcoming events.
// Goroutine-safe via RWMutex. Consumers read via Snapshot().
var (
	cacheMu    sync.RWMutex
	cacheData  []Event
	cacheFetch time.Time
	apiKey     string
)

// Init wires the Finnhub API key and starts a background poller. Safe to
// call with empty apiKey — the package degrades gracefully (IsBlackout
// returns false, UpcomingEvents returns empty).
func Init(ctx context.Context, finnhubKey string) {
	apiKey = strings.TrimSpace(finnhubKey)
	if apiKey == "" {
		log.Printf("[macrocal] disabled: FINNHUB_API_KEY not set")
		return
	}

	// Initial fetch synchronously so the first tick has real data.
	if err := refresh(ctx); err != nil {
		log.Printf("[macrocal] initial fetch failed: %v (will retry in 1h)", err)
	}

	go func() {
		t := time.NewTicker(1 * time.Hour)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				if err := refresh(ctx); err != nil {
					log.Printf("[macrocal] refresh failed: %v", err)
				}
			}
		}
	}()
}

// finnhubRaw matches Finnhub's economicCalendar array element.
type finnhubRaw struct {
	Actual   *float64 `json:"actual"`
	Country  string   `json:"country"`
	Estimate *float64 `json:"estimate"`
	Event    string   `json:"event"`
	Impact   string   `json:"impact"`
	Prev     *float64 `json:"prev"`
	Time     string   `json:"time"` // "2026-04-25 12:30:00" UTC
	Unit     string   `json:"unit"`
}

type finnhubResponse struct {
	EconomicCalendar []finnhubRaw `json:"economicCalendar"`
}

// refresh fetches the upcoming 72h window from Finnhub and replaces cache.
// Keeps events with Time >= now-1h so IsBlackout can still pick up the
// "right after release" blackout window on events that just fired.
func refresh(ctx context.Context) error {
	from := time.Now().UTC().Add(-1 * time.Hour).Format("2006-01-02")
	to := time.Now().UTC().Add(72 * time.Hour).Format("2006-01-02")
	url := fmt.Sprintf("%s/calendar/economic?from=%s&to=%s&token=%s",
		finnhubBase, from, to, apiKey)

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	client := &http.Client{Timeout: 10 * time.Second}
	res, err := client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("finnhub status %d", res.StatusCode)
	}

	var payload finnhubResponse
	if err := json.NewDecoder(res.Body).Decode(&payload); err != nil {
		return err
	}

	parsed := make([]Event, 0, len(payload.EconomicCalendar))
	for _, r := range payload.EconomicCalendar {
		// Finnhub time is UTC "YYYY-MM-DD HH:MM:SS".
		ts, err := time.ParseInLocation("2006-01-02 15:04:05", r.Time, time.UTC)
		if err != nil {
			continue
		}
		e := Event{
			Country: strings.ToUpper(r.Country),
			Event:   strings.TrimSpace(r.Event),
			Impact:  strings.ToLower(r.Impact),
			Time:    ts,
		}
		if r.Estimate != nil {
			e.Estimate = *r.Estimate
		}
		if r.Actual != nil {
			e.Actual = *r.Actual
		}
		if r.Prev != nil {
			e.Prev = *r.Prev
		}
		parsed = append(parsed, e)
	}

	cacheMu.Lock()
	cacheData = parsed
	cacheFetch = time.Now()
	cacheMu.Unlock()

	log.Printf("[macrocal] refreshed: %d events over next 72h", len(parsed))
	return nil
}

// Snapshot returns a copy of the current event cache. Safe to iterate.
func Snapshot() []Event {
	cacheMu.RLock()
	defer cacheMu.RUnlock()
	out := make([]Event, len(cacheData))
	copy(out, cacheData)
	return out
}

// LastFetch returns when the cache was last refreshed (zero value = never).
func LastFetch() time.Time {
	cacheMu.RLock()
	defer cacheMu.RUnlock()
	return cacheFetch
}
