package narrative

// importance_haiku.go — Anthropic Messages API client for the importance
// classifier. Calls Claude Haiku and parses the JSON response into an
// Importance.
//
// Wire format reference (anthropic-version: 2023-06-01):
//
//	POST https://api.anthropic.com/v1/messages
//	{
//	  "model": "claude-haiku-4-5-20251001",
//	  "max_tokens": 200,
//	  "system": "...",
//	  "messages": [{"role": "user", "content": "Title: ...\nSummary: ..."}]
//	}
//
//	Response (truncated, only the load-bearing fields):
//	{
//	  "content": [{"type": "text", "text": "{\"score\":85,\"category\":\"macro\",...}"}],
//	  ...
//	}
//
// Quality gates (per the task):
//   - HTTP timeout 10s (set on the http.Client; injectable for tests).
//   - Retry once on 5xx with 500ms backoff. Don't retry on 4xx — auth/quota
//     failures won't get better in 500ms.
//   - Clamp score to [0, 100]. Unknown category → CategoryOther.
//   - Don't log API key. Don't log request body in plain text.

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

// DefaultHaikuModel is the Claude model used by HaikuClassifier when
// HaikuClassifier.Model is empty. Pinned to claude-haiku-4-5-20251001.
const DefaultHaikuModel = "claude-haiku-4-5-20251001"

// anthropicEndpoint is overridable in tests via the unexported APIBase
// field on HaikuClassifier (see below). Production code never sets it.
const anthropicEndpoint = "https://api.anthropic.com/v1/messages"

// anthropicVersion is the API version header. Pinned per task spec.
const anthropicVersion = "2023-06-01"

// haikuMaxTokens caps the LLM response. Our schema is small (~80 chars
// JSON) so 200 is generous.
const haikuMaxTokens = 200

// haikuSystemPrompt is intentionally short and concrete. The few-shot
// examples in-line cover all four score bands so a temperature-driven drift
// can't push the median far from where we want it.
const haikuSystemPrompt = `You score crypto/markets news for market impact 0-100. ` +
	`Examples: 'Fed cuts rates 50bp' = 85, 'Trump declares national emergency' = 95, ` +
	`'UK CPI release matches expectations' = 35, 'Small altcoin lists on KuCoin' = 15. ` +
	`Reply ONLY with valid JSON: {"score": 0-100, "category": "macro|geopolitics|regulation|exchange|partnership|altcoin|other", "why": "one short sentence"}`

// haiku5xxBackoff is the single retry delay for 5xx upstream errors. 500ms
// is long enough that a transient blip resolves and short enough that one
// retry doesn't dominate the worker tick.
const haiku5xxBackoff = 500 * time.Millisecond

// HaikuClassifier calls Anthropic's Messages API to score a news item.
// Safe for concurrent use (every Classify call is a fresh HTTP request).
type HaikuClassifier struct {
	APIKey string
	// Model defaults to DefaultHaikuModel when empty. Exposed so tests can
	// pin a specific model without rebuilding the client.
	Model string
	// HTTP is injectable for tests. Production wires &http.Client{Timeout: 10s}.
	HTTP *http.Client
	// APIBase overrides the endpoint host. Empty = production. Used by tests
	// pointing at httptest.NewServer.
	APIBase string
}

// anthropicRequest mirrors only the fields we actually send. Temperature
// is omitempty so importance_haiku.go (which leaves it at zero / API
// default of 1.0) doesn't broadcast a 0.0 that would make the model
// greedy. idea_haiku.go sets it explicitly to haikuIdeaTemperature.
type anthropicRequest struct {
	Model       string             `json:"model"`
	MaxTokens   int                `json:"max_tokens"`
	System      string             `json:"system"`
	Messages    []anthropicMessage `json:"messages"`
	Temperature float64            `json:"temperature,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// anthropicResponse mirrors only the load-bearing parts of the Messages
// response. The "type" field is ignored; we only care about content[0].text.
type anthropicResponse struct {
	Content []anthropicContentBlock `json:"content"`
}

type anthropicContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// llmJSON is the shape we expect inside content[0].text. The struct is
// permissive on the score field — we accept either int or float (some LLM
// runs reply 85.0) and clamp on parse.
type llmJSON struct {
	Score    float64 `json:"score"`
	Category string  `json:"category"`
	Why      string  `json:"why"`
}

// Classify implements ImportanceClassifier. The url parameter is unused —
// caching is the CachedClassifier's job, not the LLM caller's.
//
// Errors:
//   - validateInputs failure → returns emptyImportance(), nil (no API call).
//   - HTTP transport error → wrapped error, score 0 zero-value Importance.
//   - 4xx → wrapped error, no retry.
//   - 5xx (after one retry) → wrapped error.
//   - JSON parse failure (response or inner text) → wrapped error.
//   - ctx cancellation → ctx.Err() bubbled up.
func (h *HaikuClassifier) Classify(ctx context.Context, title, summary, url string) (Importance, error) {
	if err := validateInputs(title, summary); err != nil {
		return emptyImportance(), nil
	}
	if h.APIKey == "" {
		return Importance{}, errors.New("narrative.HaikuClassifier: APIKey is empty")
	}
	httpClient := h.HTTP
	if httpClient == nil {
		// Defensive default — tests should always inject. Production wires
		// a 10s-timeout client at construction.
		httpClient = &http.Client{Timeout: 10 * time.Second}
	}
	model := h.Model
	if model == "" {
		model = DefaultHaikuModel
	}
	endpoint := h.APIBase
	if endpoint == "" {
		endpoint = anthropicEndpoint
	}

	body, err := json.Marshal(anthropicRequest{
		Model:     model,
		MaxTokens: haikuMaxTokens,
		System:    haikuSystemPrompt,
		Messages: []anthropicMessage{{
			Role:    "user",
			Content: fmt.Sprintf("Title: %s\nSummary: %s", title, summary),
		}},
	})
	if err != nil {
		return Importance{}, wrapClassifierErr("HaikuClassifier", "marshal request", err)
	}

	// First attempt + at most one retry on 5xx.
	imp, err := h.doOnce(ctx, httpClient, endpoint, body)
	if err == nil {
		return imp, nil
	}
	var retryable *retryableError
	if !errors.As(err, &retryable) {
		return Importance{}, err
	}

	// Honour ctx during the backoff sleep — a cancelled worker should not
	// burn 500ms before exiting.
	select {
	case <-ctx.Done():
		return Importance{}, ctx.Err()
	case <-time.After(haiku5xxBackoff):
	}

	imp, err = h.doOnce(ctx, httpClient, endpoint, body)
	return imp, err
}

// parseLLMJSON unwraps the LLM's reply text into an llmJSON. It tolerates
// three formats: raw JSON ({"score":...}), JSON inside a ```json fenced
// block, and JSON inside a bare ``` fence. Anything else is treated as the
// raw form and the unmarshal will surface the real error.
func parseLLMJSON(raw string) (llmJSON, error) {
	s := strings.TrimSpace(raw)

	// Strip a leading ``` (with optional language tag) and a trailing ```.
	if strings.HasPrefix(s, "```") {
		// Find the end of the opening fence line.
		nl := strings.Index(s, "\n")
		if nl > 0 {
			s = s[nl+1:]
		} else {
			s = strings.TrimPrefix(s, "```")
		}
		// Strip the trailing closing fence (anything from the last ``` on).
		if i := strings.LastIndex(s, "```"); i >= 0 {
			s = s[:i]
		}
		s = strings.TrimSpace(s)
	}

	var inner llmJSON
	if err := json.Unmarshal([]byte(s), &inner); err != nil {
		return llmJSON{}, err
	}
	return inner, nil
}

// retryableError marks 5xx responses so the outer Classify can decide to
// retry once. Keeps the retry policy in one place.
type retryableError struct {
	status int
}

func (e *retryableError) Error() string {
	return fmt.Sprintf("upstream 5xx (status %d)", e.status)
}

// doOnce performs ONE HTTP round-trip. Returns a *retryableError for 5xx so
// the caller can retry, or a wrapped non-retryable error for everything
// else. On success, returns the parsed Importance.
func (h *HaikuClassifier) doOnce(ctx context.Context, client *http.Client, endpoint string, body []byte) (Importance, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return Importance{}, wrapClassifierErr("HaikuClassifier", "build request", err)
	}
	req.Header.Set("x-api-key", h.APIKey)
	req.Header.Set("anthropic-version", anthropicVersion)
	req.Header.Set("content-type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return Importance{}, wrapClassifierErr("HaikuClassifier", "http do", err)
	}
	defer resp.Body.Close()

	// Read up to a sane cap. 64KB is generous for a 200-token reply.
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	if err != nil {
		return Importance{}, wrapClassifierErr("HaikuClassifier", "read body", err)
	}

	if resp.StatusCode >= 500 && resp.StatusCode < 600 {
		// Don't log the full response body — it can include partial prompt
		// echoes. Length + status is enough to debug.
		log.Printf("narrative.HaikuClassifier: 5xx status=%d body_len=%d (will retry)", resp.StatusCode, len(respBody))
		return Importance{}, &retryableError{status: resp.StatusCode}
	}
	if resp.StatusCode >= 400 {
		// 4xx is auth/quota — don't retry. Don't log body (may contain
		// sensitive auth-error text).
		return Importance{}, fmt.Errorf("narrative.HaikuClassifier: upstream 4xx status=%d", resp.StatusCode)
	}

	// Decode the outer envelope.
	var env anthropicResponse
	if err := json.Unmarshal(respBody, &env); err != nil {
		return Importance{}, wrapClassifierErr("HaikuClassifier", "unmarshal envelope", err)
	}
	if len(env.Content) == 0 || env.Content[0].Text == "" {
		return Importance{}, errors.New("narrative.HaikuClassifier: empty content in response")
	}

	// Decode the inner JSON the LLM returned. Haiku occasionally wraps the
	// response in a markdown code fence (```json ... ``` or just ``` ... ```)
	// even though the system prompt says "Reply ONLY with valid JSON". Strip
	// the fence before unmarshaling.
	inner, err := parseLLMJSON(env.Content[0].Text)
	if err != nil {
		return Importance{}, wrapClassifierErr("HaikuClassifier", "unmarshal inner json", err)
	}

	return Importance{
		Score:    clampScore(int(inner.Score)),
		Category: normalizeCategory(inner.Category),
		Why:      inner.Why,
	}, nil
}
