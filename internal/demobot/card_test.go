package demobot

import (
	"strings"
	"testing"
	"time"
)

func intPtr(v int) *int { return &v }

var goldenTime = time.Date(2026, 8, 18, 6, 0, 0, 0, time.UTC)

// ── Golden: full card with confidence bar ────────────────────────────────────

func TestRenderHTMLFullCard(t *testing.T) {
	c := Card{
		Emoji:      "🟢",
		Agent:      "Momentum Agent",
		Asset:      "BTC",
		Verdict:    "BULLISH — bulls in control",
		Facts:      []string{"RSI(14): 62.4", "MACD hist: +120.50"},
		Confidence: intPtr(60),
		DataTime:   goldenTime,
	}
	want := "🟢 <b>Momentum Agent</b> · BTC\n" +
		"<b>BULLISH — bulls in control</b>\n" +
		"• RSI(14): 62.4\n" +
		"• MACD hist: +120.50\n" +
		"Confidence: ■■■□□ 60%\n" +
		"\n<i>Analytics, not financial advice · AlphaVizor · 2026-08-18 06:00 UTC</i>"
	if got := c.RenderHTML(); got != want {
		t.Fatalf("full card mismatch:\ngot:\n%s\nwant:\n%s", got, want)
	}
}

// ── Golden: offline card, no confidence, no facts ────────────────────────────

func TestRenderHTMLOfflineCard(t *testing.T) {
	c := Card{
		Emoji:    "⚪",
		Agent:    "Whale Flow Agent",
		Asset:    "BTC",
		Verdict:  "Data source offline right now. The team is on it.",
		DataTime: goldenTime,
	}
	want := "⚪ <b>Whale Flow Agent</b> · BTC\n" +
		"<b>Data source offline right now. The team is on it.</b>\n" +
		"\n<i>Analytics, not financial advice · AlphaVizor · 2026-08-18 06:00 UTC</i>"
	if got := c.RenderHTML(); got != want {
		t.Fatalf("offline card mismatch:\ngot:\n%s\nwant:\n%s", got, want)
	}
}

// ── HTML escaping of dynamic content ─────────────────────────────────────────

func TestRenderHTMLEscapesDynamicContent(t *testing.T) {
	c := Card{
		Emoji:    "⚪",
		Agent:    "A<b>gent",
		Asset:    "B&C",
		Verdict:  `x < y & "z"`,
		Facts:    []string{"<script>alert(1)</script>"},
		DataTime: goldenTime,
	}
	got := c.RenderHTML()
	for _, bad := range []string{"<script>", "A<b>gent", `"z"`} {
		if strings.Contains(got, bad) {
			t.Errorf("unescaped %q leaked into: %s", bad, got)
		}
	}
	for _, want := range []string{"&lt;script&gt;", "x &lt; y &amp; &#34;z&#34;", "B&amp;C"} {
		if !strings.Contains(got, want) {
			t.Errorf("expected escaped %q in: %s", want, got)
		}
	}
}

// ── Golden: AI block renders after facts/confidence, before the footer ───────

func TestRenderHTMLWithAIBlock(t *testing.T) {
	c := Card{
		Emoji:      "🟢",
		Agent:      "Narrative Radar",
		Verdict:    "Top narrative: ai-agents — trending, trend score 84/100",
		Facts:      []string{"1. ai-agents — trending · score 84 · 412 mentions/24h"},
		Confidence: intPtr(71),
		AIHTML:     "<b>AI idea:</b> <i>Mentions tripled in 24h.</i>",
		DataTime:   goldenTime,
	}
	want := "🟢 <b>Narrative Radar</b>\n" +
		"<b>Top narrative: ai-agents — trending, trend score 84/100</b>\n" +
		"• 1. ai-agents — trending · score 84 · 412 mentions/24h\n" +
		"Confidence: ■■■■□ 71%\n" +
		"<b>AI idea:</b> <i>Mentions tripled in 24h.</i>\n" +
		"\n<i>Analytics, not financial advice · AlphaVizor · 2026-08-18 06:00 UTC</i>"
	if got := c.RenderHTML(); got != want {
		t.Fatalf("AI-block card mismatch:\ngot:\n%s\nwant:\n%s", got, want)
	}
}

// ── Confidence bar mapping 0-100 → 5 blocks ──────────────────────────────────

func TestConfidenceBar(t *testing.T) {
	cases := map[int]string{
		0:   "□□□□□",
		10:  "■□□□□", // 0.5 rounds away from zero → 1
		45:  "■■□□□", // 2.25 → 2
		50:  "■■■□□", // 2.5 → 3
		60:  "■■■□□",
		89:  "■■■■□", // 4.45 → 4
		90:  "■■■■■", // 4.5 → 5
		100: "■■■■■",
	}
	for in, want := range cases {
		if got := confidenceBar(in); got != want {
			t.Errorf("confidenceBar(%d): got %q, want %q", in, got, want)
		}
	}
}

// ── One-liner rendering for /digest ──────────────────────────────────────────

func TestOneLiner(t *testing.T) {
	withAsset := Card{Emoji: "🟢", ShortName: "Momentum", Asset: "BTC", Short: "bullish"}
	if got, want := withAsset.OneLiner(), "🟢 <b>Momentum</b> BTC: bullish"; got != want {
		t.Errorf("one-liner with asset: got %q, want %q", got, want)
	}
	noAsset := Card{Emoji: "🔴", ShortName: "Funding", Short: "shorts crowded"}
	if got, want := noAsset.OneLiner(), "🔴 <b>Funding</b>: shorts crowded"; got != want {
		t.Errorf("one-liner without asset: got %q, want %q", got, want)
	}
	escaped := Card{Emoji: "⚪", ShortName: "S<R", Short: "a & b"}
	if got, want := escaped.OneLiner(), "⚪ <b>S&lt;R</b>: a &amp; b"; got != want {
		t.Errorf("one-liner escaping: got %q, want %q", got, want)
	}
}
