package ml

import (
	"sync"
	"time"
)

// GetDailyDirectionCached returns the cached 1d trend direction for symbol.
//
// Moved from internal/api/handlers_ml_v2.go during PHASE 1 fixes so that
// both the HTTP multi-predict handler AND the scenario runner share the
// same cache — previously the runner called resolveDailyDir directly,
// hammering a 1d PredictV2 every tick (findings #9 from code review).
//
// compute is invoked only on miss or after TTL expiry. Returning ""
// from compute leaves the cache empty so the next call retries.
//
// TTL = 15 min. Daily bars roll once per UTC day, so 15 min is coarse
// but cheap; we trade at most one stale tick at bar boundary.
//
// The cache is keyed by symbol alone (not by (symbol, tier)) because
// 1d direction is tier-agnostic.

type dailyDirEntry struct {
	direction string
	expiresAt time.Time
}

var (
	dailyDirCache   sync.Map // string → dailyDirEntry
	dailyDirCacheMu sync.Mutex
	dailyDirTTL     = 15 * time.Minute
)

func GetDailyDirectionCached(symbol string, compute func() string) string {
	if v, ok := dailyDirCache.Load(symbol); ok {
		if e := v.(dailyDirEntry); time.Now().Before(e.expiresAt) {
			return e.direction
		}
	}
	// Serialize computation per-symbol to avoid stampede on cold start.
	dailyDirCacheMu.Lock()
	defer dailyDirCacheMu.Unlock()
	if v, ok := dailyDirCache.Load(symbol); ok {
		if e := v.(dailyDirEntry); time.Now().Before(e.expiresAt) {
			return e.direction
		}
	}
	dir := compute()
	if dir != "" {
		dailyDirCache.Store(symbol, dailyDirEntry{
			direction: dir,
			expiresAt: time.Now().Add(dailyDirTTL),
		})
	}
	return dir
}

// InvalidateDailyDirCache — test/admin helper. Clears all cached entries.
func InvalidateDailyDirCache() {
	dailyDirCache.Range(func(k, _ interface{}) bool {
		dailyDirCache.Delete(k)
		return true
	})
}
