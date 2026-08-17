package demobot

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// TGClient is a minimal stdlib-only Telegram Bot API client (net/http +
// encoding/json). The existing internal/telegram package is a domain-specific
// alerts bot coupled to go-telegram/bot and pgx — not a reusable generic
// client — so the demo bot carries its own instead of adding coupling or deps.
type TGClient struct {
	base  string // https://api.telegram.org/bot<token>
	token string // kept ONLY to redact itself out of error messages
	hc    *http.Client
	pace  *pacer
}

func NewTGClient(token string) *TGClient {
	return newTGClientWithBase(token, "https://api.telegram.org/bot"+token)
}

// newTGClientWithBase exists for tests (httptest servers).
func newTGClientWithBase(token, base string) *TGClient {
	return &TGClient{
		base:  base,
		token: token,
		// No global timeout: getUpdates long-polls for up to pollTimeout
		// seconds. Per-call context deadlines bound everything instead.
		hc:   &http.Client{},
		pace: newPacer(50*time.Millisecond, time.Second),
	}
}

// ── Outbound send pacer ──────────────────────────────────────────────────────
//
// A 10-person group hitting /digest bursts sends; Telegram allows ~30 msg/s
// globally and ~1 msg/s per chat. The pacer reserves send slots: ≥globalGap
// between any two sends, ≥chatGap between sends to the same chat.

type pacer struct {
	mu        sync.Mutex
	globalGap time.Duration
	chatGap   time.Duration
	global    time.Time
	perChat   map[int64]time.Time
}

func newPacer(globalGap, chatGap time.Duration) *pacer {
	return &pacer{globalGap: globalGap, chatGap: chatGap, perChat: map[int64]time.Time{}}
}

// wait reserves the next send slot for chatID and sleeps until it.
func (p *pacer) wait(ctx context.Context, chatID int64) {
	p.mu.Lock()
	earliest := time.Now()
	if t := p.global.Add(p.globalGap); t.After(earliest) {
		earliest = t
	}
	if last, ok := p.perChat[chatID]; ok {
		if t := last.Add(p.chatGap); t.After(earliest) {
			earliest = t
		}
	}
	p.global = earliest
	p.perChat[chatID] = earliest
	p.mu.Unlock()

	if d := time.Until(earliest); d > 0 {
		select {
		case <-time.After(d):
		case <-ctx.Done():
		}
	}
}

// ── Wire types (only the fields the bot reads) ───────────────────────────────

type Update struct {
	UpdateID      int64          `json:"update_id"`
	Message       *Message       `json:"message"`
	CallbackQuery *CallbackQuery `json:"callback_query"`
}

type Message struct {
	MessageID int64  `json:"message_id"`
	Chat      Chat   `json:"chat"`
	Text      string `json:"text"`
}

type Chat struct {
	ID int64 `json:"id"`
}

type CallbackQuery struct {
	ID      string   `json:"id"`
	Data    string   `json:"data"`
	Message *Message `json:"message"`
}

type InlineKeyboardButton struct {
	Text         string `json:"text"`
	CallbackData string `json:"callback_data"`
}

type InlineKeyboardMarkup struct {
	InlineKeyboard [][]InlineKeyboardButton `json:"inline_keyboard"`
}

// ── Errors ───────────────────────────────────────────────────────────────────

// ErrNotModified — Telegram rejects edits that change nothing; callers treat
// it as "already up to date", not a failure.
var ErrNotModified = errors.New("message is not modified")

// apiError is a Telegram API-level failure (ok=false envelope).
type apiError struct {
	Method      string
	Code        int
	Description string
}

func (e *apiError) Error() string {
	return fmt.Sprintf("telegram %s: %s (code %d)", e.Method, e.Description, e.Code)
}

// isFatalAuth: 401 = bad token, 409 = another poller owns getUpdates. Both
// mean this process can never work — retrying forever just looks alive while
// the bot is dead (adversarial review item 2).
func isFatalAuth(err error) bool {
	var ae *apiError
	return errors.As(err, &ae) && (ae.Code == 401 || ae.Code == 409)
}

// rateLimitedError carries retry_after from a 429 envelope.
type rateLimitedError struct {
	Method     string
	RetryAfter int
}

func (e *rateLimitedError) Error() string {
	return fmt.Sprintf("telegram %s: rate limited, retry_after=%ds", e.Method, e.RetryAfter)
}

// redact strips the bot token from error text before it can reach a log.
// Transport errors (*url.Error) embed the full request URL — token included.
// Typed sentinels pass through untouched (their text never contains the URL).
func (c *TGClient) redact(err error) error {
	if err == nil || errors.Is(err, ErrNotModified) {
		return err
	}
	var ae *apiError
	var rl *rateLimitedError
	if errors.As(err, &ae) || errors.As(err, &rl) {
		return err
	}
	if c.token == "" || !strings.Contains(err.Error(), c.token) {
		return err
	}
	return errors.New(strings.ReplaceAll(err.Error(), c.token, "<token>"))
}

// ── API calls ────────────────────────────────────────────────────────────────

type apiResponse struct {
	OK          bool            `json:"ok"`
	Result      json.RawMessage `json:"result"`
	ErrorCode   int             `json:"error_code"`
	Description string          `json:"description"`
	Parameters  *struct {
		RetryAfter int `json:"retry_after"`
	} `json:"parameters"`
}

const maxRetryAfter = 10 // seconds — cap what a 429 can make us sleep

// call marshals params, performs the request, and on a 429 sleeps
// retry_after (capped) and retries exactly once.
func (c *TGClient) call(ctx context.Context, method string, params any, result any) error {
	body, err := json.Marshal(params)
	if err != nil {
		return err
	}
	err = c.doOnce(ctx, method, body, result)
	var rl *rateLimitedError
	if errors.As(err, &rl) {
		waitSec := rl.RetryAfter
		if waitSec > maxRetryAfter {
			waitSec = maxRetryAfter
		}
		if waitSec < 0 {
			waitSec = 0
		}
		select {
		case <-time.After(time.Duration(waitSec) * time.Second):
		case <-ctx.Done():
			return ctx.Err()
		}
		err = c.doOnce(ctx, method, body, result)
	}
	return err
}

func (c *TGClient) doOnce(ctx context.Context, method string, body []byte, result any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/"+method, bytes.NewReader(body))
	if err != nil {
		return c.redact(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.hc.Do(req)
	if err != nil {
		return c.redact(err)
	}
	defer resp.Body.Close()

	// Telegram sends the JSON envelope on error statuses too (that is where
	// error_code lives), so decode regardless of the HTTP status.
	limited := io.LimitReader(resp.Body, 2<<20)
	var api apiResponse
	if err := json.NewDecoder(limited).Decode(&api); err != nil {
		_, _ = io.Copy(io.Discard, limited)
		return c.redact(fmt.Errorf("telegram %s: HTTP %d: decode: %w", method, resp.StatusCode, err))
	}
	_, _ = io.Copy(io.Discard, limited) // drain for connection reuse

	if !api.OK {
		if strings.Contains(api.Description, "message is not modified") {
			return ErrNotModified
		}
		if api.ErrorCode == 429 {
			retryAfter := 1
			if api.Parameters != nil {
				retryAfter = api.Parameters.RetryAfter
			}
			return &rateLimitedError{Method: method, RetryAfter: retryAfter}
		}
		return &apiError{Method: method, Code: api.ErrorCode, Description: api.Description}
	}
	if result != nil {
		return json.Unmarshal(api.Result, result)
	}
	return nil
}

// getUpdatesRaw is the shared long-poll/probe primitive. limit<=0 omits the
// limit parameter.
func (c *TGClient) getUpdatesRaw(ctx context.Context, offset int64, limit, timeoutSec int) ([]Update, error) {
	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSec+10)*time.Second)
	defer cancel()
	params := map[string]any{
		"offset":          offset,
		"timeout":         timeoutSec,
		"allowed_updates": []string{"message", "callback_query"},
	}
	if limit > 0 {
		params["limit"] = limit
	}
	var updates []Update
	if err := c.call(ctx, "getUpdates", params, &updates); err != nil {
		return nil, err
	}
	return updates, nil
}

// GetUpdates long-polls for up to timeoutSec seconds.
func (c *TGClient) GetUpdates(ctx context.Context, offset int64, timeoutSec int) ([]Update, error) {
	return c.getUpdatesRaw(ctx, offset, 0, timeoutSec)
}

// LastUpdateID probes the newest pending update (offset=-1, limit=1,
// timeout=0) so a restarted bot can fast-forward past the backlog instead of
// replaying it (adversarial review item 1). Returns 0 when the queue is empty.
func (c *TGClient) LastUpdateID(ctx context.Context) (int64, error) {
	updates, err := c.getUpdatesRaw(ctx, -1, 1, 0)
	if err != nil {
		return 0, err
	}
	if len(updates) == 0 {
		return 0, nil
	}
	return updates[len(updates)-1].UpdateID, nil
}

// SendMessage sends an HTML-parse-mode message, optionally with an inline
// keyboard. Returns the new message id. Paced.
func (c *TGClient) SendMessage(ctx context.Context, chatID int64, htmlText string, kb *InlineKeyboardMarkup) (int64, error) {
	c.pace.wait(ctx, chatID)
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	params := map[string]any{
		"chat_id":                  chatID,
		"text":                     htmlText,
		"parse_mode":               "HTML",
		"disable_web_page_preview": true,
	}
	if kb != nil {
		params["reply_markup"] = kb
	}
	var msg Message
	if err := c.call(ctx, "sendMessage", params, &msg); err != nil {
		return 0, err
	}
	return msg.MessageID, nil
}

// EditMessageText rewrites a previously sent card in place (Refresh button).
// Paced like a send — edits count against the same flood limits.
func (c *TGClient) EditMessageText(ctx context.Context, chatID, messageID int64, htmlText string, kb *InlineKeyboardMarkup) error {
	c.pace.wait(ctx, chatID)
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	params := map[string]any{
		"chat_id":                  chatID,
		"message_id":               messageID,
		"text":                     htmlText,
		"parse_mode":               "HTML",
		"disable_web_page_preview": true,
	}
	if kb != nil {
		params["reply_markup"] = kb
	}
	return c.call(ctx, "editMessageText", params, nil)
}

// AnswerCallbackQuery acknowledges a button press; with alert=true Telegram
// shows a modal (used for "How it works", capped at 200 chars by Telegram).
func (c *TGClient) AnswerCallbackQuery(ctx context.Context, id, text string, alert bool) error {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	params := map[string]any{
		"callback_query_id": id,
		"show_alert":        alert,
	}
	if text != "" {
		params["text"] = text
	}
	return c.call(ctx, "answerCallbackQuery", params, nil)
}

// BotCommand is one native "/" menu entry.
type BotCommand struct {
	Command     string `json:"command"`
	Description string `json:"description"`
}

// SetMyCommands fills Telegram's native "/" menu button. Called once at
// startup; failures are non-fatal (the bot works without the menu).
func (c *TGClient) SetMyCommands(ctx context.Context, commands []BotCommand) error {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	return c.call(ctx, "setMyCommands", map[string]any{"commands": commands}, nil)
}

// GetMe validates the token at startup and returns the bot username.
func (c *TGClient) GetMe(ctx context.Context) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	var me struct {
		Username string `json:"username"`
	}
	if err := c.call(ctx, "getMe", nil, &me); err != nil {
		return "", err
	}
	return me.Username, nil
}
