package narrative

// moodread.go — backend client for the AlphaVizor AI market-mood read engine.
// Mirrors idea_haiku.go's transport pattern (same retry policy, same header
// set, same post-response blacklist) but operates at the MARKET level: it
// describes the OVERALL crypto market climate, not a single narrative.
//
// Input comes from two sources assembled by the api layer:
//   - CoinMarketCap Fear & Greed index (value + label)
//   - Top-N narrative snapshots from narrative_snapshots (stage + sentiment)
//
// Output: a 2-3 sentence neutral, observational English paragraph. The same
// bannedIdeaPattern blacklist from idea_haiku.go is the authoritative gate —
// on a match we return ("", nil) so the caller treats it as "no read this
// cycle" and the frontend hides the block gracefully.
//
// Branding: the model identifier is intentionally not surfaced. The public
// name in the API response is "alphavizor-ai" only.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// moodReadMaxTokens caps the reply. 2-3 short observational sentences about
// the overall market climate fit comfortably under 200 tokens.
const moodReadMaxTokens = 200

// moodReadTemperature is tuned low so the model stays close to the
// observational neutral tone instead of drifting into trade-directive copy.
const moodReadTemperature = 0.35

// moodReadTimeout is the per-request HTTP timeout for the mood endpoint.
const moodReadTimeout = 15 * time.Second

// moodReadSystemPrompt is the contract the model is held to. The hard rule
// is: describe the OVERALL MARKET MOOD (fear/greed climate, the general tone
// across themes, what kinds of narratives are drawing attention, the breadth
// of sentiment), NEVER the price action, NEVER trade directives.
//
// Structure mirrors haikuIdeaSystemPrompt exactly: role anchor → ALLOWED list
// → FORBIDDEN list → imperative anchor → English-only directive (twice).
// The "FORBIDDEN" section and both "English" mentions are load-bearing — the
// test suite (moodread_test.go) asserts they are present verbatim.
const moodReadSystemPrompt = `You write neutral analytical commentary about the OVERALL CRYPTO MARKET MOOD. ` +
	`Output 2-3 short sentences in English. ` +
	`ALLOWED: describe the overall fear or greed climate, the general tone across themes, ` +
	`what kinds of narratives are drawing attention, the breadth of sentiment, ` +
	`the stage of attention across the market (early / trending / mainstream / declining), ` +
	`and whether sentiment appears broad-based or concentrated in a few areas. ` +
	`Voice: observational, not advisory — like a research note, not a trade idea. ` +
	`Describe what the mood IS, not whether conditions are good or bad for acting — ` +
	`avoid action-nudging adjectives such as "favorable", "constructive", "supportive", or "ripe". ` +
	`FORBIDDEN — if you write any of these you fail the task: ` +
	`specific dollar prices, price targets, support levels, resistance levels, breakout calls, ` +
	`"buy", "sell", "entry", "stop loss", "take profit", "position", "long", "short", ` +
	`"fade", "accumulation", "watch for", "momentum", and any technical-analysis terminology. ` +
	`NEVER use the phrase "traders should" or any other imperative direction to the reader. ` +
	`Speak about market mood and climate, NOT price action. ` +
	`Reply in English ONLY — no Russian, no other languages. ` +
	`Reply with the paragraph text only — no JSON, no preamble.`

// ─── Input types ──────────────────────────────────────────────────────────

// MoodNarrative is the minimal per-narrative context passed to ReadMood.
// TrendScore is accepted for completeness but intentionally not rendered in
// the user message — numeric scores would anchor the model on specifics and
// risk DoD #11 language violations (same reasoning as idea_haiku.go's
// deliberate omission of prices).
type MoodNarrative struct {
	Name           string
	Stage          string
	SentimentLabel string
	TrendScore     int // accepted but not rendered; see buildMoodUserMessage
}

// MarketMoodInput bundles all context the mood-read prompt needs. All fields
// are optional — empty means "no data available for this dimension"; the
// caller checks before calling the LLM (api handler skips the call when both
// F&G label and narratives are absent).
type MarketMoodInput struct {
	FearGreedValue  int    // 0-100; meaningful only when FearGreedLabel != ""
	FearGreedLabel  string // e.g. "Fear", "Neutral", "Extreme Greed"
	TopNarratives   []MoodNarrative
}

// ─── Interface + implementations ──────────────────────────────────────────

// MarketMoodReader produces a short neutral English paragraph describing the
// current overall crypto market mood. Callers SHOULD treat any non-nil error
// as "use the empty-string fallback" — the frontend hides the block when the
// field is empty.
type MarketMoodReader interface {
	ReadMood(ctx context.Context, in MarketMoodInput) (string, error)
}

// StaticMoodReader is the no-key / no-LLM fallback. Returns ("", nil) so the
// caller (and the frontend) treats it as "no read this cycle" — the frontend
// hides the block rather than rendering a stale or made-up paragraph. Safe
// for concurrent use (no state).
type StaticMoodReader struct{}

// ReadMood implements MarketMoodReader. Always returns ("", nil) after
// honouring ctx cancellation. The empty string signals "nothing to show" to
// the caller; it is NOT an error condition.
func (StaticMoodReader) ReadMood(ctx context.Context, _ MarketMoodInput) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	return "", nil
}

// Compile-time interface checks. If either implementation diverges from the
// interface signature, the build fails here rather than at the call-site.
var _ MarketMoodReader = StaticMoodReader{}
var _ MarketMoodReader = (*HaikuMoodReader)(nil)

// ─── HaikuMoodReader ──────────────────────────────────────────────────────

// HaikuMoodReader is the production transport for the AlphaVizor AI market-
// mood read engine. Mirrors HaikuIdeaGenerator's shape exactly: same HTTP
// client injection, same APIBase override for tests, same retry policy.
// Safe for concurrent use (every ReadMood call is a fresh HTTP request).
type HaikuMoodReader struct {
	// APIKey is the Anthropic API key. Empty → ReadMood returns an error
	// immediately without making a network call.
	APIKey string
	// Model defaults to DefaultHaikuModel when empty.
	Model string
	// HTTP is injectable for tests. Production wires &http.Client{Timeout: 15s}.
	HTTP *http.Client
	// APIBase overrides the endpoint host. Empty = production. Used by tests
	// pointing at httptest.NewServer.
	APIBase string
}

// ReadMood implements MarketMoodReader. Returns the LLM's reply text on
// success. Errors:
//   - empty APIKey → static error, no API call.
//   - HTTP transport error → wrapped error.
//   - 4xx → wrapped error, no retry.
//   - 5xx (after one retry) → wrapped error.
//   - empty response body → wrapped error.
//   - ctx cancellation → ctx.Err() bubbled up.
//
// On blacklist match: returns ("", nil) — not an error.
// On any non-nil error the caller falls back to the stale cache or empty read.
func (h *HaikuMoodReader) ReadMood(ctx context.Context, in MarketMoodInput) (string, error) {
	if h.APIKey == "" {
		return "", errors.New("narrative.HaikuMoodReader: APIKey is empty")
	}
	httpClient := h.HTTP
	if httpClient == nil {
		httpClient = &http.Client{Timeout: moodReadTimeout}
	}
	model := h.Model
	if model == "" {
		model = DefaultHaikuModel
	}
	endpoint := h.APIBase
	if endpoint == "" {
		endpoint = anthropicEndpoint
	}

	userMsg := buildMoodUserMessage(in)

	body, err := json.Marshal(anthropicRequest{
		Model:       model,
		MaxTokens:   moodReadMaxTokens,
		System:      moodReadSystemPrompt,
		Temperature: moodReadTemperature,
		Messages: []anthropicMessage{{
			Role:    "user",
			Content: userMsg,
		}},
	})
	if err != nil {
		return "", wrapClassifierErr("HaikuMoodReader", "marshal request", err)
	}

	// First attempt + at most one retry on 5xx — same policy as HaikuIdeaGenerator.
	text, err := h.doOnce(ctx, httpClient, endpoint, body)
	if err != nil {
		var retryable *retryableError
		if !errors.As(err, &retryable) {
			return "", err
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(haiku5xxBackoff):
		}
		text, err = h.doOnce(ctx, httpClient, endpoint, body)
		if err != nil {
			return "", err
		}
	}

	// Post-response blacklist — same authoritative gate as idea_haiku.go.
	// bannedIdeaPattern is defined in idea_haiku.go; reuse it directly (same
	// package) rather than redefining — ensures one canonical gate for both
	// idea and mood paths.
	if bannedIdeaPattern.MatchString(text) {
		log.Printf("narrative/moodread: response rejected by blacklist len=%d", len(text))
		return "", nil
	}
	return cleanIdeaText(text), nil
}

// doOnce performs ONE HTTP round-trip for HaikuMoodReader. Returns a
// *retryableError for 5xx so the caller can retry, or a wrapped non-retryable
// error otherwise. On success returns the trimmed reply text.
// Mirrors HaikuIdeaGenerator.doOnce exactly (same headers, same 64KB cap,
// same status-code branches). Separate method to avoid coupling the two
// workloads' tuning surfaces.
func (h *HaikuMoodReader) doOnce(ctx context.Context, client *http.Client, endpoint string, body []byte) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return "", wrapClassifierErr("HaikuMoodReader", "build request", err)
	}
	req.Header.Set("x-api-key", h.APIKey)
	req.Header.Set("anthropic-version", anthropicVersion)
	req.Header.Set("content-type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", wrapClassifierErr("HaikuMoodReader", "http do", err)
	}
	defer resp.Body.Close()

	// 64KB cap — generous for a 200-token reply.
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	if err != nil {
		return "", wrapClassifierErr("HaikuMoodReader", "read body", err)
	}

	if resp.StatusCode >= 500 && resp.StatusCode < 600 {
		log.Printf("narrative.HaikuMoodReader: 5xx status=%d body_len=%d (will retry)", resp.StatusCode, len(respBody))
		return "", &retryableError{status: resp.StatusCode}
	}
	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("narrative.HaikuMoodReader: upstream 4xx status=%d", resp.StatusCode)
	}

	var env anthropicResponse
	if err := json.Unmarshal(respBody, &env); err != nil {
		return "", wrapClassifierErr("HaikuMoodReader", "unmarshal envelope", err)
	}
	if len(env.Content) == 0 || env.Content[0].Text == "" {
		return "", errors.New("narrative.HaikuMoodReader: empty content in response")
	}

	return cleanIdeaText(env.Content[0].Text), nil
}

// fearGreedBands is the canonical CoinMarketCap classification set. CMC is the
// only intended source and its vocabulary is fixed, so any label outside this
// set is treated as untrusted and replaced (see normalizeFearGreedLabel) before
// it reaches the LLM prompt.
var fearGreedBands = map[string]bool{
	"Extreme Fear":  true,
	"Fear":          true,
	"Neutral":       true,
	"Greed":         true,
	"Extreme Greed": true,
}

// normalizeFearGreedLabel returns label unchanged when it is a known CMC band;
// otherwise it derives a band from the 0-100 value. This is the prompt-input
// hardening for the one genuinely external string in the mood prompt — it stops
// a malformed or injected classification string from being interpolated into
// the model input verbatim. Callers invoke it only when label != "" (an empty
// label means "no F&G data" and is handled by the caller).
func normalizeFearGreedLabel(value int, label string) string {
	if fearGreedBands[label] {
		return label
	}
	switch {
	case value <= 24:
		return "Extreme Fear"
	case value <= 44:
		return "Fear"
	case value <= 55:
		return "Neutral"
	case value <= 75:
		return "Greed"
	default:
		return "Extreme Greed"
	}
}

// buildMoodUserMessage assembles the user-side prompt from the MarketMoodInput.
// Rendering rules:
//   - F&G line only when FearGreedLabel != "" (value alone is meaningless).
//   - F&G label is clamped to the canonical CMC band set (untrusted strings
//     are replaced by a value-derived band).
//   - Narratives line only when len(TopNarratives) > 0.
//   - NO numeric trend scores, NO prices, NO $ anywhere.
//   - NO Cyrillic anywhere.
//   - Always ends with the English-only + subject-anchor tail line.
func buildMoodUserMessage(in MarketMoodInput) string {
	var sb strings.Builder

	if in.FearGreedLabel != "" {
		label := normalizeFearGreedLabel(in.FearGreedValue, in.FearGreedLabel)
		fmt.Fprintf(&sb, "Market Fear & Greed: %d/100 (%s)\n", in.FearGreedValue, label)
	}

	if len(in.TopNarratives) > 0 {
		parts := make([]string, 0, len(in.TopNarratives))
		for _, n := range in.TopNarratives {
			parts = append(parts, fmt.Sprintf("%s (%s, %s)", n.Name, n.Stage, n.SentimentLabel))
		}
		fmt.Fprintf(&sb, "Active themes: %s\n", strings.Join(parts, "; "))
	}

	// Tail anchor — locks the model to English and reminds it of the subject.
	// Third EN anchor (after the two in the system prompt) for the same reason
	// as idea_haiku.go: the closing instruction wins most reliably.
	sb.WriteString("Reply in English. Describe the overall market mood, not price action.\n")
	return sb.String()
}
