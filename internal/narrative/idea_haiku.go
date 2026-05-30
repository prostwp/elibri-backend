package narrative

// idea_haiku.go — backend client for the AlphaVizor AI commentary engine.
// Mirrors importance_haiku.go's transport pattern (same retry policy, same
// header set) but ships a different system+user prompt and parses the
// response as RAW TEXT (not JSON). Upstream returns content[0].text
// directly, so for a free-form English paragraph there's no inner unwrap to
// do — the model's reply is the answer.
//
// Phase 1 (May 2026): system prompt switched from Russian to English (EN-
// only UI rule).
// Phase 2 (May 2026): output is now neutral NARRATIVE commentary — no
// prices, no support/resistance, no trade directives. The prompt is
// explicit about banned vocabulary, AND we additionally pass the
// returned text through bannedIdeaPattern (a regex blacklist) so a
// model that disobeys the prompt still can't ship a banned phrase to
// production. If the blacklist trips we return "" — the frontend hides
// the block.
//
// Branding: the model itself is intentionally NOT named in any output
// path that touches the UI ("AlphaVizor AI" is the public name; the
// upstream vendor/model identifier never crosses the API boundary).
//
// Why a separate file from importance_haiku.go: same upstream can run very
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
	"regexp"
	"strings"
	"time"
)

// haikuIdeaMaxTokens caps the English paragraph reply. 2-4 short sentences
// of neutral narrative commentary fit comfortably under 200 tokens.
// Tightened from 400 → 200 to discourage the model from drifting into
// per-token speculation when a short observation is enough.
const haikuIdeaMaxTokens = 200

// haikuIdeaTemperature is tuned down from the API default (1.0) so the
// model sticks closer to the prompt's neutral observational tone. Lower
// values also reduce the chance of hallucinated specifics (prices,
// targets) that the bannedIdeaPattern would otherwise filter out.
const haikuIdeaTemperature = 0.35

// haikuIdeaSystemPrompt is the contract the model is held to. The hard
// rule is: describe the NARRATIVE (theme, traction, attention stage,
// reporting sources), NEVER the price action. The "FORBIDDEN" list is
// deliberately long and explicit — every word in it has historically
// leaked into output and tripped the DoD #11 legal review.
//
// The "Reply in English ONLY" line is what stops the upstream model from
// mirroring the language of the user message (snapshot.Reasons can carry
// Russian editorial strings from the worker). Repeating "English" in both
// the role line and the final imperative beats a single mention — the
// upstream obeys the last instruction more reliably than the first.
//
// Even with this prompt, the post-response bannedIdeaPattern is the
// authoritative gate — if the model regresses, that regex kicks in and
// returns "" so the frontend hides the block.
const haikuIdeaSystemPrompt = `You write neutral analytical commentary about a crypto NARRATIVE. ` +
	`Output 2-4 short sentences in ENGLISH. ` +
	`ALLOWED: describe the theme, why it is gaining traction, what assets cluster around it, ` +
	`what stage of attention it is in (early / trending / mainstream / declining), ` +
	`and what kinds of sources are reporting on it. ` +
	`FORBIDDEN — if you write any of these you fail the task: ` +
	`specific dollar prices, price targets, support levels, resistance levels, breakout calls, ` +
	`"buy", "sell", "entry", "stop loss", "take profit", "position", "long", "short", ` +
	`"fade", "accumulation", "watch for", "momentum", and any technical-analysis terminology. ` +
	`NEVER use the phrase "traders should" or any other imperative direction to the reader. ` +
	`Speak about the NARRATIVE, not the price action. ` +
	`Voice: observational, not advisory — like a research note, not a trade idea. ` +
	`Reply in English ONLY — no Russian, no other languages. ` +
	`Reply with the paragraph text only — no JSON, no preamble.`

// haikuIdeaTimeout is the per-request HTTP timeout for the idea endpoint.
// 15s is generous for a 200-token reply but matches the upstream tail
// latency budget.
const haikuIdeaTimeout = 15 * time.Second

// bannedIdeaPattern is the post-response blacklist. Any match → return ""
// from GenerateIdea so the caller skips caching and the frontend renders
// the neutral static fallback. This is the BACKSTOP for the prompt
// instructions; if the model regresses despite the explicit FORBIDDEN
// list, this regex still keeps banned vocabulary off production.
//
// Patterns covered (all case-insensitive):
//   - dollar-and-cents prices ("$2.40", "$ 0.05")
//   - explicit support/resistance/breakout/target/entry/position vocabulary
//   - "stop loss" / "take profit" / "stop-loss" (hyphen optional)
//   - directional verbs (buy/sell/long/short/fade)
//   - editorial imperatives ("watch for", "traders should")
//   - "accumulation" / "momentum" (technical-analysis red flags per
//     DoD #11)
//
// Word-boundary anchors (\b) prevent false positives like the substring
// "long" inside "belong" or "short" inside "shortlist".
var bannedIdeaPattern = regexp.MustCompile(`(?i)` +
	`\$\s?[0-9]+\.[0-9]+` +
	`|\bsupport\b` +
	`|\bresistance\b` +
	`|\bbreakout\b` +
	`|\bposition\b` +
	`|\bentry\b` +
	`|\bstop[ -]?loss\b` +
	`|\btake[ -]?profit\b` +
	`|\btarget\b` +
	`|\bwatch for\b` +
	`|\bfade\b` +
	`|\baccumulation\b` +
	`|\bbuy\b` +
	`|\bsell\b` +
	`|\blong\b` +
	`|\bshort\b` +
	`|\btraders should\b` +
	`|\bmomentum\b`)

// HaikuIdeaGenerator is the backend transport for the AlphaVizor AI
// narrative-commentary engine. (The struct name retains the historic
// "Haiku" identifier purely for internal continuity — the public output
// is branded AlphaVizor AI and the upstream vendor/model identifier is
// never surfaced to the UI.) Safe for concurrent use (every Generate
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
	// format, same headers, just a different prompt + max_tokens + a lower
	// temperature (idea commentary should be tight & observational, not
	// creative).
	body, err := json.Marshal(anthropicRequest{
		Model:       model,
		MaxTokens:   haikuIdeaMaxTokens,
		System:      haikuIdeaSystemPrompt,
		Temperature: haikuIdeaTemperature,
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

	// Post-response blacklist. Even if the prompt was honoured perfectly,
	// the regex is the AUTHORITATIVE gate for legal compliance (DoD #11
	// bans specific prices, support/resistance, trade-direction
	// vocabulary). On a match: return ("", nil) so the caller treats it
	// as "no idea this hour" — the in-memory cache won't store an empty
	// string (handler logic in narrative_handlers.go already short-
	// circuits on err != nil; we deliberately return nil here so the
	// empty string flows up cleanly), and the frontend falls back to its
	// neutral placeholder.
	if bannedIdeaPattern.MatchString(text) {
		log.Printf("narrative/idea: response rejected by blacklist for slug=%s len=%d", narrativeSlug, len(text))
		return "", nil
	}
	return text, nil
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
// slug + snapshot context. The `prices` arg is accepted for interface
// symmetry but DELIBERATELY UNUSED — feeding the upstream specific
// dollar values used to encourage price-specific commentary (the exact
// DoD #11 violation we are fixing). We now pass ONLY narrative-level
// context (stage, sentiment label, mention reasons) so the model has
// no per-token price to anchor on.
//
// All labels are English-only (Phase 1, May 2026). Prior versions used
// "Нарратив:" as the lead label, which steered the upstream into mirroring
// the user's language back as Russian even with an English system prompt;
// switching the anchor to "Narrative:" lets the system prompt's
// "English ONLY" instruction win uncontested.
func buildIdeaUserMessage(narrativeSlug string, snapshot Snapshot, _ []TickerQuote) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "Narrative: %s\n", narrativeSlug)
	fmt.Fprintf(&sb, "Stage: %s, Trend: %d/100, Sentiment: %s\n",
		string(snapshot.Stage), snapshot.TrendScore, string(snapshot.SentimentLabel))
	// Related assets are passed as a bare ticker list — NO prices. The
	// model is told to discuss what theme these assets cluster around,
	// not to anchor recommendations to specific levels.
	if len(snapshot.RelatedAssets) > 0 {
		fmt.Fprintf(&sb, "Related assets: %s\n", strings.Join(snapshot.RelatedAssets, ", "))
	}
	// Reasons from the snapshot give the model the per-narrative editorial
	// framing the worker already wrote (e.g. "Mentions up 35% vs prior 24h",
	// "Stage: TRENDING"). Each line becomes one bullet in the prompt.
	if len(snapshot.Reasons) > 0 {
		fmt.Fprintf(&sb, "Context: %s\n", strings.Join(snapshot.Reasons, "; "))
	}
	// Third EN anchor at the user-message tail — if a future change adds a
	// Russian Mention.Title or Reason, the closing instruction here keeps
	// the upstream locked to English even when system-prompt anchors weaken.
	sb.WriteString("Reply in English. Describe the narrative, not the price action.\n")
	return sb.String()
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
