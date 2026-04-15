package crypto

import (
	"log"
	"sync"
	"time"

	"github.com/prostwp/elibri-backend/internal/market"
	"github.com/prostwp/elibri-backend/pkg/types"
)

var (
	knownSymbols = make(map[string]bool)
	knownMu      sync.RWMutex
	initialized  bool
)

// CheckNewListings compares current Binance pairs against known set
func CheckNewListings() ([]types.NewListing, error) {
	pairs, err := market.FetchAllUSDTPairs()
	if err != nil {
		return nil, err
	}

	knownMu.Lock()
	defer knownMu.Unlock()

	// First run — just populate known set
	if !initialized {
		for _, p := range pairs {
			knownSymbols[p] = true
		}
		initialized = true
		log.Printf("Listing hunter initialized with %d known USDT pairs", len(knownSymbols))
		return nil, nil
	}

	// Find new symbols
	var newListings []types.NewListing
	now := time.Now()

	for _, p := range pairs {
		if !knownSymbols[p] {
			knownSymbols[p] = true

			// Extract base asset (remove USDT suffix)
			base := p
			if len(p) > 4 {
				base = p[:len(p)-4]
			}

			newListings = append(newListings, types.NewListing{
				Symbol:     p,
				BaseAsset:  base,
				ListedAt:   now,
				DetectedAt: now,
			})

			log.Printf("🚀 NEW LISTING DETECTED: %s", p)
		}
	}

	return newListings, nil
}

// GetKnownCount returns number of tracked pairs
func GetKnownCount() int {
	knownMu.RLock()
	defer knownMu.RUnlock()
	return len(knownSymbols)
}
