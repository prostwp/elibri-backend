// Package scenario runs user-defined trading strategies live: polls market
// data on a TF-appropriate cadence, evaluates the graph, emits alerts.
package scenario

import "time"

// TFToPoll maps a candle interval to how often the runner should re-evaluate
// the scenario. Roughly 1/10th of the bar duration — fast enough to catch a
// fresh HC signal within seconds of bar close, slow enough not to hammer
// Binance's 1200 req/min IP limit across many scenarios.
func TFToPoll(tf string) time.Duration {
	switch tf {
	case "5m":
		return 30 * time.Second
	case "15m":
		return 1 * time.Minute
	case "1h":
		return 5 * time.Minute
	case "4h":
		return 15 * time.Minute
	case "1d":
		return 1 * time.Hour
	default:
		return 5 * time.Minute
	}
}

// BarDuration returns how much wall time a single candle covers.
// Used for dedup keys and "fresh bar" detection.
func BarDuration(tf string) time.Duration {
	switch tf {
	case "5m":
		return 5 * time.Minute
	case "15m":
		return 15 * time.Minute
	case "1h":
		return 1 * time.Hour
	case "4h":
		return 4 * time.Hour
	case "1d":
		return 24 * time.Hour
	default:
		return 1 * time.Hour
	}
}
