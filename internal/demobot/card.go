package demobot

import (
	"html"
	"math"
	"strconv"
	"strings"
	"time"
)

// Semaphore emojis — the only three the card contract allows.
const (
	emojiBull    = "🟢"
	emojiBear    = "🔴"
	emojiNeutral = "⚪"
)

// ── machine-readable status ──────────────────────────────────────────────────

// cardStatus is the machine-readable state behind a card, threaded from the
// builders to the HTTP envelope's ok/reason pair (docs/demobot-http.md) so
// templates can branch on WHY a reading is absent instead of sniffing verdict
// strings. The zero value is statusOK — only degraded paths set it.
type cardStatus int

const (
	statusOK                  cardStatus = iota // real reading produced
	statusSourceOffline                         // data source unreachable (or the headline read's source is)
	statusInsufficientHistory                   // source alive, too few closed bars for the indicator set
	statusBelowThreshold                        // narrative radar warming up: thin mention base / no snapshots
	statusNoData                                // upstream alive but nothing to read: macro unknown in the open window, no flow snapshot
	statusMarketClosed                          // macro unknown because the tradfin market is closed
)

// reason maps a status to its JSON enum value; "" for statusOK (the envelope
// serves null instead).
func (s cardStatus) reason() string {
	switch s {
	case statusSourceOffline:
		return "source_offline"
	case statusInsufficientHistory:
		return "insufficient_history"
	case statusBelowThreshold:
		return "below_threshold"
	case statusNoData:
		return "no_data"
	case statusMarketClosed:
		return "market_closed"
	default:
		return ""
	}
}

// ── machine-readable levels ──────────────────────────────────────────────────
//
// Card.Levels is the numeric companion to the facts strings for agents whose
// reading IS a set of price levels: raw computed float64s at full precision,
// never display-rounded. The HTTP envelope serves it verbatim as "levels";
// Telegram rendering ignores it. nil for agents without levels.

// PullbackZone is the EMA20-EMA50 band served ONLY in confirmed-trend states
// (B3). From = EMA20 (the shallow edge, nearer price), To = EMA50 (the deep,
// trend-defining edge) — raw floats, unordered by size on purpose: in an
// uptrend From > To, in a downtrend From < To.
//
// Why this band is the pullback target: in a confirmed trend price extends
// away from its short averages between impulses and mean-reverts toward them;
// EMA20 is the first dynamic edge a routine pullback touches, and EMA50 is the
// same average the state machine reads — a close beyond it starts breaking
// the structure. The honest "pullback with the trend intact" region is
// therefore bounded by the two. Outside confirmed states there is no trend to
// pull back within, so the field is absent.
type PullbackZone struct {
	From float64 `json:"from"` // EMA20
	To   float64 `json:"to"`   // EMA50
}

// TrendLevels — the price at which the trend structure this card reads is
// broken (see invalidationFor), plus the pullback band in confirmed states.
// InvalidationSide ("below" | "above") makes the break direction machine-
// readable: downtrends invalidate ABOVE the EMA cluster, everything else
// below it (review fix 8) — additive field, old consumers ignore it.
type TrendLevels struct {
	Invalidation     float64       `json:"invalidation"`
	InvalidationSide string        `json:"invalidation_side"`
	PullbackZone     *PullbackZone `json:"pullback_zone,omitempty"`
}

// SRPoint is one clustered level at raw precision (SRLevel.Raw — the cluster
// mean, not the display-rounded integer) with its touch count and the B4
// metrics. Formulas live on SRLevel / the indicator functions; the wire adds:
//   - strength: touches + 0.5 per above-median-volume touch (== touches on
//     volume-less FX series)
//   - weakening: ≥7 touches with the last 3 touches' mean volume below the
//     first 3's
//   - breaks / holds: frequency counts of level tests over the window
//     (test = close within 0.25×ATR; break = close beyond by >0.25×ATR
//     within 3 bars) — frequencies, never probabilities
//   - last_touch: RFC3339 UTC of the newest touch's bar time
type SRPoint struct {
	Level     float64 `json:"level"`
	Touches   int     `json:"touches"`
	Strength  float64 `json:"strength"`
	Weakening bool    `json:"weakening"`
	Breaks    int     `json:"breaks"`
	Holds     int     `json:"holds"`
	LastTouch string  `json:"last_touch"`
}

// SRLevels — strength-sorted supports/resistances. Both slices are always
// non-nil so an empty side serializes as [] ("we looked, nothing clustered"),
// never null.
type SRLevels struct {
	Supports    []SRPoint `json:"supports"`
	Resistances []SRPoint `json:"resistances"`
}

// VolLevels — ATR(14) now over its 30-bar average, unrounded.
type VolLevels struct {
	ExpansionRatio float64 `json:"expansion_ratio"`
}

// AssetResult is one asset's machine-readable outcome inside a multi-asset
// momentum card (review fix 3): a mixed scan is no longer distinguishable
// from a full one only by reading fact strings. OK mirrors the per-asset
// read; Reason is nil when OK, else "insufficient_history" |
// "source_offline". Served as the envelope's "results" (momentum only).
type AssetResult struct {
	Asset  string  `json:"asset"`
	OK     bool    `json:"ok"`
	Reason *string `json:"reason,omitempty"`
}

// Card is one agent's reply. RenderHTML produces the exact Telegram
// HTML-parse-mode body (golden-tested in card_test.go).
type Card struct {
	Emoji      string // one of emojiBull / emojiBear / emojiNeutral
	Agent      string // full agent name, e.g. "Momentum Agent"
	ShortName  string // digest one-liner name, e.g. "Momentum"
	Asset      string // "" when the agent is not asset-specific
	Verdict    string // bold verdict line
	Short      string // short verdict for digest one-liners
	Facts      []string
	Confidence *int // nil → no confidence bar (API gave none)
	DataTime   time.Time
	Command    string // bot command key, used for Refresh/How-it-works callbacks
	HowItWorks string // ≤200 chars, shown via answerCallbackQuery alert
	Deviation  int    // 0..100 deviation-from-neutral used by the priority rule
	Offline    bool   // true when the data source was unreachable
	SourceNote string // extra footer note, e.g. "data: Yahoo Finance"
	// Status is the machine state behind the HTTP envelope's ok/reason pair.
	// Zero value = real reading; degraded builders set the matching enum.
	Status cardStatus
	// Levels carries one of TrendLevels / SRLevels / VolLevels (or nil) —
	// served verbatim as the envelope's "levels" object, ignored by Telegram.
	Levels any
	// Results is the per-asset machine outcome of a multi-asset momentum card
	// (nil elsewhere) — served as the envelope's "results", ignored by
	// Telegram (the facts carry the human form).
	Results []AssetResult
	// AIHTML is a pre-rendered AI block ("<b>AI idea:</b> <i>…</i>") appended
	// after the facts and confidence bar. Builders MUST esc() every dynamic
	// value when composing it — RenderHTML writes it verbatim.
	AIHTML string
}

func esc(s string) string { return html.EscapeString(s) }

// RenderHTML renders the full card:
//
//	{emoji} <b>{Agent}</b> · {Asset}
//	<b>{Verdict}</b>
//	• fact …
//	Confidence: ■■■□□ 60%          (only when Confidence != nil)
//	<b>AI …:</b> <i>…</i>          (only when AIHTML != "")
//
//	<i>Analytics, not financial advice · AlphaVizor · {UTC time}</i>
func (c Card) RenderHTML() string {
	footer := "\n<i>Analytics, not financial advice · AlphaVizor · " +
		c.DataTime.UTC().Format("2006-01-02 15:04") + " UTC"
	if c.SourceNote != "" {
		footer += " · " + esc(c.SourceNote)
	}
	return c.renderBody() + footer + "</i>"
}

// renderBody is the card without the footer — /digest embeds the winner's
// body and closes the whole message with a single footer.
func (c Card) renderBody() string {
	var b strings.Builder
	b.WriteString(c.Emoji)
	b.WriteString(" <b>")
	b.WriteString(esc(c.Agent))
	b.WriteString("</b>")
	if c.Asset != "" {
		b.WriteString(" · ")
		b.WriteString(esc(c.Asset))
	}
	b.WriteString("\n<b>")
	b.WriteString(esc(c.Verdict))
	b.WriteString("</b>\n")
	for _, f := range c.Facts {
		b.WriteString("• ")
		b.WriteString(esc(f))
		b.WriteString("\n")
	}
	if c.Confidence != nil {
		v := clampInt(*c.Confidence, 0, 100)
		b.WriteString("Confidence: ")
		b.WriteString(confidenceBar(v))
		b.WriteString(" ")
		b.WriteString(strconv.Itoa(v))
		b.WriteString("%\n")
	}
	if c.AIHTML != "" {
		b.WriteString(c.AIHTML)
		b.WriteString("\n")
	}
	return b.String()
}

// OneLiner renders the compact digest form: "🟢 <b>Momentum</b> BTC: bullish".
func (c Card) OneLiner() string {
	var b strings.Builder
	b.WriteString(c.Emoji)
	b.WriteString(" <b>")
	b.WriteString(esc(c.ShortName))
	b.WriteString("</b>")
	if c.Asset != "" {
		b.WriteString(" ")
		b.WriteString(esc(c.Asset))
	}
	b.WriteString(": ")
	b.WriteString(esc(c.Short))
	return b.String()
}

// confidenceBar maps 0-100 to five blocks; filled = round(v/20),
// rounding half away from zero (Go math.Round), clamped to 0..5.
func confidenceBar(v int) string {
	filled := int(math.Round(float64(clampInt(v, 0, 100)) / 20.0))
	return strings.Repeat("■", filled) + strings.Repeat("□", 5-filled)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// effectiveStatus is the machine status the HTTP layer serves: the builder's
// explicit Status, with one defensive collapse — a card flagged Offline that
// carries no specific status reads as source_offline, so a future degraded
// path that only sets the flag can never serve ok=true.
func (c Card) effectiveStatus() cardStatus {
	if c.Status == statusOK && c.Offline {
		return statusSourceOffline
	}
	return c.Status
}

// offlineCard is the honest degraded state — never fake data.
func offlineCard(agent, shortName, asset, command, how string) Card {
	return Card{
		Emoji:      emojiNeutral,
		Agent:      agent,
		ShortName:  shortName,
		Asset:      asset,
		Verdict:    "Data source offline right now. The team is on it.",
		Short:      "offline",
		DataTime:   time.Now().UTC(),
		Command:    command,
		HowItWorks: how,
		Deviation:  0,
		Offline:    true,
		Status:     statusSourceOffline,
	}
}
