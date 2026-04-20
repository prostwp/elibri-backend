// Package telegram wraps the Bot API used for live trade alerts.
package telegram

import (
	"fmt"
	"strings"

	"github.com/prostwp/elibri-backend/internal/scenario"
)

// FormatAlert renders an Alert as the Markdown message sent to the user.
// Pure function — tested independently of the bot transport.
//
// Example:
//   🟢 BTCUSDT 4h — LONG (Trend-Aligned)
//   Entry:  74 836.00
//   Stop:   74 405.00   (-1.5×ATR)
//   Target: 75 555.00   (+2.5×ATR)
//   Size:   $24.00      0.25% risk · Balanced
//   Conf:   72%
func FormatAlert(a *scenario.Alert) string {
	var icon, word string
	if a.Direction == "buy" {
		icon = "🟢"
		word = "LONG"
	} else {
		icon = "🔴"
		word = "SHORT"
	}

	labelPretty := prettyLabel(a.Label)

	var b strings.Builder
	fmt.Fprintf(&b, "%s *%s %s* — *%s*", icon, a.Symbol, a.Interval, word)
	if labelPretty != "" {
		fmt.Fprintf(&b, " (%s)", labelPretty)
	}
	b.WriteString("\n")
	fmt.Fprintf(&b, "Entry:  `%s`\n", formatPrice(a.EntryPrice))
	fmt.Fprintf(&b, "Stop:   `%s`\n", formatPrice(a.StopLoss))
	fmt.Fprintf(&b, "Target: `%s`\n", formatPrice(a.TakeProfit))
	if a.PositionSizeUSD > 0 {
		fmt.Fprintf(&b, "Size:   `$%.2f`\n", a.PositionSizeUSD)
	}
	if a.Confidence > 0 {
		fmt.Fprintf(&b, "Conf:   `%.0f%%`\n", a.Confidence)
	}
	return b.String()
}

func prettyLabel(label string) string {
	switch label {
	case "trend_aligned":
		return "Trend-Aligned"
	case "mean_reversion":
		return "Mean Reversion"
	case "random":
		return "Counter"
	default:
		return ""
	}
}

func formatPrice(p float64) string {
	if p >= 1000 {
		return fmt.Sprintf("%.2f", p)
	}
	if p >= 1 {
		return fmt.Sprintf("%.4f", p)
	}
	return fmt.Sprintf("%.6f", p)
}
