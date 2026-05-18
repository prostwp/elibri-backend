package narrative

// idea_haiku.go — Anthropic Messages API client for the IdeaGenerator.
// Mirrors importance_haiku.go's transport pattern (same retry policy, same
// header set) but ships a different system+user prompt and parses the
// response as RAW TEXT (not JSON). Anthropic returns content[0].text
// directly, so for a free-form English paragraph there's no inner unwrap to
// do — the LLM's reply is the answer.
//
// Phase 1 (May 2026): system prompt was switched from Russian to English.
// The site is EN-only (per UI English-Only rule), and the user-side prompt
// still ships some Russian context (snapshot.Reasons can be either lang),
// so the system prompt now repeats the "Reply in English ONLY" instruction
// twice — once in the role description, once at the end — to override the
// LLM's tendency to mirror the user message's language.
//
// Why a separate file from importance_haiku.go: same model can run very
// different workloads, and we want one place to tune the importance prompt
// vs one place to tune the idea prompt. Sharing a transport helper would
// save 30 LOC at the cost of coupling two unrelated tuning surfaces.

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

// haikuIdeaMaxTokens caps the English paragraph reply. 2-3 short sentences
// at ~150-200 chars each fit comfortably under 250 tokens; 400 gives slack
// for the rare LLM run that wants to be more verbose.
const haikuIdeaMaxTokens = 400

// haikuIdeaSystemPrompt is the contract the LLM is held to — English text,
// 2-3 sentences, three required ingredients (lead vs lag, entry signal,
// risk note). The "Reply ONLY with the paragraph text" line is critical:
// without it the LLM sometimes replies with "Sure! Here's the analysis: …"
// which the frontend would then render as if it were the idea.
//
// The "Reply in English ONLY" line is what stops Haiku from mirroring the
// language of the user message (snapshot.Reasons can carry Russian editorial
// strings from the worker). Repeating "English" in both the role line and
// the final imperative beats a single mention — Haiku obeys the last
// instruction more reliably than the first.
const haikuIdeaSystemPrompt = `You are a crypto trading copilot. Given a narrative + price context, ` +
	`write ONE short paragraph in ENGLISH (2-3 sentences) covering: ` +
	`1) which related token leads / which lags ` +
	`2) one concrete entry signal a trader should watch (e.g. "JTO above $2.40") ` +
	`3) brief risk note. ` +
	`Be specific. Don't recommend buying outright. Use trader vocabulary ` +
	`(volume, breakout, resistance, support, fade). ` +
	`Reply in English ONLY — no Russian, no other languages. ` +
	`Reply with the paragraph text only — no JSON, no preamble.`

// haikuIdeaTimeout is the per-request HTTP timeout for the Haiku idea
// endpoint. Longer than importance (10s) because the response is a 2-3
// sentence English paragraph (~200-300 tokens decoded) vs the importance
// schema's ~80-char JSON.
const haikuIdeaTimeout = 15 * time.Second

// HaikuIdeaGenerator calls Anthropic's Messages API to generate the
// Russian-paragraph trade idea. Safe for concurrent use (every Generate
// call is a fresh HTTP request).
type HaikuIdeaGenerator struct {
	APIKey string
	// Model defaults to DefaultHaikuModel when empty.
	Model string
	// HTTP is injectable for tests. Production wires &http.Client{Timeout: 15s}.
	HTTP *http.Client
	// APIBase overrides the endpoint host. Empty = production. Used by tests
	// pointing at httptest.NewServer.
	APIBase string
}

// GenerateIdea implements IdeaGenerator. Returns the LLM's reply text on
// success. Errors:
//   - empty APIKey → static error, no API call.
//   - HTTP transport error → wrapped error (caller should fall back to "").
//   - 4xx → wrapped error, no retry.
//   - 5xx (after one retry) → wrapped error.
//   - empty response body → wrapped error.
//   - ctx cancellation → ctx.Err() bubbled up.
//
// On any error the caller renders the empty-state default in the UI, so a
// transient Anthropic outage shows the static "Add to watchlist…" line
// rather than crashing the dashboard.
func (h *HaikuIdeaGenerator) GenerateIdea(ctx context.Context, narrativeSlug string, snapshot Snapshot, prices []TickerQuote) (string, error) {
	if h.APIKey == "" {
		return "", errors.New("narrative.HaikuIdeaGenerator: APIKey is empty")
	}
	httpClient := h.HTTP
	if httpClient == nil {
		// Defensive default — production should always inject. The 15s here
		// matches haikuIdeaTimeout but lives on the client, not per-request.
		httpClient = &http.Client{Timeout: haikuIdeaTimeout}
	}
	model := h.Model
	if model == "" {
		model = DefaultHaikuModel
	}
	endpoint := h.APIBase
	if endpoint == "" {
		endpoint = anthropicEndpoint
	}

	userMsg := buildIdeaUserMessage(narrativeSlug, snapshot, prices)

	// Reuse the anthropicRequest shape from importance_haiku.go — same wire
	// format, same headers, just different prompt + max_tokens.
	body, err := json.Marshal(anthropicRequest{
		Model:     model,
		MaxTokens: haikuIdeaMaxTokens,
		System:    haikuIdeaSystemPrompt,
		Messages: []anthropicMessage{{
			Role:    "user",
			Content: userMsg,
		}},
	})
	if err != nil {
		return "", wrapClassifierErr("HaikuIdeaGenerator", "marshal request", err)
	}

	// First attempt + at most one retry on 5xx — same policy as the
	// importance classifier, kept for symmetry so a transient blip resolves
	// the same way for both prompts.
	text, err := h.doOnce(ctx, httpClient, endpoint, body)
	if err == nil {
		return text, nil
	}
	var retryable *retryableError
	if !errors.As(err, &retryable) {
		return "", err
	}
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case <-time.After(haiku5xxBackoff):
	}
	return h.doOnce(ctx, httpClient, endpoint, body)
}

// doOnce performs ONE HTTP round-trip. Returns a *retryableError for 5xx so
// the caller can retry, or a wrapped non-retryable error otherwise. On
// success returns the trimmed reply text.
func (h *HaikuIdeaGenerator) doOnce(ctx context.Context, client *http.Client, endpoint string, body []byte) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return "", wrapClassifierErr("HaikuIdeaGenerator", "build request", err)
	}
	req.Header.Set("x-api-key", h.APIKey)
	req.Header.Set("anthropic-version", anthropicVersion)
	req.Header.Set("content-type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", wrapClassifierErr("HaikuIdeaGenerator", "http do", err)
	}
	defer resp.Body.Close()

	// 64KB cap — generous for a 400-token reply. A larger payload signals
	// either a bug or an upstream behaviour change; failing fast beats OOM.
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	if err != nil {
		return "", wrapClassifierErr("HaikuIdeaGenerator", "read body", err)
	}

	if resp.StatusCode >= 500 && resp.StatusCode < 600 {
		log.Printf("narrative.HaikuIdeaGenerator: 5xx status=%d body_len=%d (will retry)", resp.StatusCode, len(respBody))
		return "", &retryableError{status: resp.StatusCode}
	}
	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("narrative.HaikuIdeaGenerator: upstream 4xx status=%d", resp.StatusCode)
	}

	// Parse the outer envelope. Same shape as the importance classifier;
	// content[0].text holds the LLM's reply.
	var env anthropicResponse
	if err := json.Unmarshal(respBody, &env); err != nil {
		return "", wrapClassifierErr("HaikuIdeaGenerator", "unmarshal envelope", err)
	}
	if len(env.Content) == 0 || env.Content[0].Text == "" {
		return "", errors.New("narrative.HaikuIdeaGenerator: empty content in response")
	}

	// Trim whitespace and a stray surrounding markdown fence (Haiku
	// sometimes wraps even free-form replies in ``` despite the system
	// prompt). The fence-stripper from parseLLMJSON would discard everything
	// after the first ``` so we apply a lighter touch here.
	return cleanIdeaText(env.Content[0].Text), nil
}

// buildIdeaUserMessage assembles the user-side prompt from the narrative
// slug, the latest snapshot, and the price snapshots. Format pinned in
// tests to keep the prompt regression-safe — if you change the layout,
// expect TestHaikuIdeaGenerator_PromptShape to flag it.
//
// All labels are English-only (Phase 1, May 2026). Prior versions used
// "Нарратив:" as the lead label, which steered Haiku into mirroring the
// user's language back as Russian even with an English system prompt;
// switching the anchor to "Narrative:" lets the system prompt's
// "English ONLY" instruction win uncontested.
func buildIdeaUserMessage(narrativeSlug string, snapshot Snapshot, prices []TickerQuote) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "Narrative: %s\n", narrativeSlug)
	fmt.Fprintf(&sb, "Stage: %s, Trend: %d/100, Sentiment: %s\n",
		string(snapshot.Stage), snapshot.TrendScore, string(snapshot.SentimentLabel))
	if len(prices) > 0 {
		assetsLine := formatPricesLine(prices)
		fmt.Fprintf(&sb, "Related assets: %s\n", assetsLine)
	} else {
		// Fall back to plain ticker list when prices are unavailable so the
		// LLM still has the candidate symbol set to reason about.
		if len(snapshot.RelatedAssets) > 0 {
			fmt.Fprintf(&sb, "Related assets: %s\n", strings.Join(snapshot.RelatedAssets, ", "))
		}
	}
	// Reasons from the snapshot give the LLM the per-narrative editorial
	// framing the worker already wrote (e.g. "Mentions up 35% vs prior 24h",
	// "Stage: TRENDING"). Each line becomes one bullet in the prompt.
	if len(snapshot.Reasons) > 0 {
		fmt.Fprintf(&sb, "Importance reasons: %s\n", strings.Join(snapshot.Reasons, "; "))
	}
	// Third EN anchor at the user-message tail — if a future change adds a
	// Russian Mention.Title or Reason, the closing instruction here keeps
	// Haiku locked to English even when system-prompt anchors weaken.
	sb.WriteString("Reply in English.\n")
	return sb.String()
}

// formatPricesLine renders a comma-separated "JTO $2.50 +12%, SOL $180 +3%"
// list. Always 1-decimal % and 2-decimal price — the LLM is sensitive to
// inconsistent formatting (e.g. some "$1" and some "$1.00" makes it think
// the cheaper one is more interesting).
func formatPricesLine(prices []TickerQuote) string {
	parts := make([]string, 0, len(prices))
	for _, p := range prices {
		sign := "+"
		if p.ChangePct < 0 {
			sign = ""
		}
		parts = append(parts, fmt.Sprintf("%s $%.2f %s%.1f%%", p.Symbol, p.Price, sign, p.ChangePct))
	}
	return strings.Join(parts, ", ")
}

// cleanIdeaText trims whitespace and an optional surrounding markdown code
// fence. Unlike parseLLMJSON we DON'T discard everything outside the fence
// — the goal is to strip stray formatting, not extract a JSON island.
func cleanIdeaText(raw string) string {
	s := strings.TrimSpace(raw)
	// Strip a leading ``` (with optional language tag) and a trailing ```.
	if strings.HasPrefix(s, "```") {
		nl := strings.Index(s, "\n")
		if nl > 0 {
			s = s[nl+1:]
		} else {
			s = strings.TrimPrefix(s, "```")
		}
		if i := strings.LastIndex(s, "```"); i >= 0 {
			s = s[:i]
		}
		s = strings.TrimSpace(s)
	}
	return s
}
