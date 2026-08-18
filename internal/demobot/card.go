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

// OneLiner renders the compact digest form: "🟢 <b>Momentum</b> BTC: buy".
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
	}
}
