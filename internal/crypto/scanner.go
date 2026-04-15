package crypto

import (
	"log"
	"math"
	"sort"
	"time"

	"github.com/prostwp/elibri-backend/internal/market"
	"github.com/prostwp/elibri-backend/pkg/types"
)

// ScanConfig defines scanner parameters
type ScanConfig struct {
	MinQuoteVolume float64 // minimum 24h quote volume in USDT
	RSIOversold    float64 // RSI below this = buy signal
	VolumeSpikeX   float64 // volume > X times average = spike
	DipPercent     float64 // price drop % from recent high
}

var DefaultScanConfig = ScanConfig{
	MinQuoteVolume: 1_000_000,
	RSIOversold:    30,
	VolumeSpikeX:   3.0,
	DipPercent:     15,
}

// ScanResult with scoring
type ScanResult = types.CryptoScanResult

// RunScan scans all USDT pairs for opportunities
func RunScan(cfg ScanConfig) ([]ScanResult, error) {
	start := time.Now()

	// Fetch all tickers
	quotes, err := market.FetchCryptoQuotes(nil) // all pairs
	if err != nil {
		return nil, err
	}

	var results []ScanResult

	for symbol, q := range quotes {
		// Filter: only USDT pairs with sufficient volume
		if q.MarketCap < cfg.MinQuoteVolume { // MarketCap stores quoteVolume for crypto
			continue
		}

		score := 0
		reasons := []string{}

		// 1. Volume spike detection
		// Using change as proxy — high volume usually accompanies high volatility
		absChange := math.Abs(q.ChangePercent)
		if absChange > 5 {
			score += 30
			reasons = append(reasons, "High volatility")
		}

		// 2. Price dip from high (potential reversal)
		if q.High > 0 {
			dipFromHigh := ((q.High - q.Price) / q.High) * 100
			if dipFromHigh > cfg.DipPercent {
				score += 40
				reasons = append(reasons, "Price dip from 24h high")
			} else if dipFromHigh > cfg.DipPercent/2 {
				score += 20
				reasons = append(reasons, "Moderate pullback")
			}
		}

		// 3. Negative change (contrarian — buying the dip)
		if q.ChangePercent < -5 {
			score += 30
			reasons = append(reasons, "Significant 24h drop — potential reversal")
		} else if q.ChangePercent < -2 {
			score += 15
			reasons = append(reasons, "Minor dip")
		}

		// 4. High volume (interest)
		if q.MarketCap > 10_000_000 {
			score += 10
			reasons = append(reasons, "High trading volume")
		}

		if score < 20 {
			continue
		}

		signal := "neutral"
		if score >= 60 {
			signal = "buy"
		} else if score >= 40 {
			signal = "buy"
		}

		reason := ""
		if len(reasons) > 0 {
			reason = reasons[0]
			if len(reasons) > 1 {
				reason += " + " + reasons[1]
			}
		}

		results = append(results, ScanResult{
			Symbol:      symbol,
			Signal:      signal,
			Score:       score,
			Reason:      reason,
			Volume24h:   q.MarketCap,
			PriceChange: q.ChangePercent,
			RSI:         0, // TODO: calculate from candles
			ScanType:    "dip_scanner",
		})
	}

	// Sort by score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Keep top 20
	if len(results) > 20 {
		results = results[:20]
	}

	log.Printf("Crypto scan complete: %d results in %v (scanned %d pairs)",
		len(results), time.Since(start).Round(time.Millisecond), len(quotes))

	return results, nil
}
