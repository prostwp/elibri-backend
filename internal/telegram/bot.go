package telegram

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log"
	"strings"
	"time"

	tg "github.com/go-telegram/bot"
	"github.com/go-telegram/bot/models"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Bot wraps go-telegram/bot with our domain handlers (/start, /link).
// Nil-safe: if the token is empty, NewBot returns (nil, nil) so the rest
// of the system stays happy without Telegram delivery.
type Bot struct {
	client    *tg.Bot
	pool      *pgxpool.Pool
	username  string // for deep-links like https://t.me/{username}?start=...
}

// NewBot constructs a Bot. Returns (nil, nil) when token is empty — callers
// should check for nil.
func NewBot(token, username string, pool *pgxpool.Pool) (*Bot, error) {
	if token == "" {
		return nil, nil
	}
	b := &Bot{pool: pool, username: username}
	opts := []tg.Option{
		tg.WithDefaultHandler(b.dispatch),
	}
	client, err := tg.New(token, opts...)
	if err != nil {
		return nil, fmt.Errorf("telegram bot init: %w", err)
	}
	b.client = client
	return b, nil
}

// Start begins long-polling. Call from main.go as `go bot.Start(ctx)`.
// Exits gracefully when ctx is canceled.
func (b *Bot) Start(ctx context.Context) {
	if b == nil || b.client == nil {
		return
	}
	log.Printf("[tg bot] long-poll started (@%s)", b.username)
	b.client.Start(ctx)
}

// SendMarkdown delivers one message. Returns the Telegram message_id.
func (b *Bot) SendMarkdown(ctx context.Context, chatID int64, text string) (int64, error) {
	if b == nil || b.client == nil {
		return 0, fmt.Errorf("bot not configured")
	}
	msg, err := b.client.SendMessage(ctx, &tg.SendMessageParams{
		ChatID:    chatID,
		Text:      text,
		ParseMode: models.ParseModeMarkdown,
	})
	if err != nil {
		return 0, err
	}
	return int64(msg.ID), nil
}

// dispatch is the top-level router for all incoming bot updates.
// Only /start and /link <code> need custom handling; everything else
// gets a short usage reply.
func (b *Bot) dispatch(ctx context.Context, c *tg.Bot, u *models.Update) {
	if u.Message == nil {
		return
	}
	text := strings.TrimSpace(u.Message.Text)
	chatID := u.Message.Chat.ID

	switch {
	case strings.HasPrefix(text, "/start"):
		b.replyStart(ctx, chatID)
	case strings.HasPrefix(text, "/link"):
		b.replyLink(ctx, chatID, text)
	case strings.HasPrefix(text, "/unlink"):
		b.replyUnlink(ctx, chatID)
	default:
		b.reply(ctx, chatID, "Usage:\n`/link <code>` — link your Elibri account\n`/unlink` — disconnect\n\nGet a code from the Elibri web UI → Profile → Telegram.")
	}
}

func (b *Bot) replyStart(ctx context.Context, chatID int64) {
	b.reply(ctx, chatID,
		"*Elibri FX — trade signals bot* 🤖\n\n"+
			"To receive live signals:\n"+
			"1. Open the Elibri web UI → Profile → Telegram\n"+
			"2. Copy the 6-digit linking code\n"+
			"3. Send here: `/link 123456`\n\n"+
			"Your chat will then receive LONG/SHORT alerts with entry, SL, TP, and position size.",
	)
}

func (b *Bot) replyLink(ctx context.Context, chatID int64, text string) {
	parts := strings.Fields(text)
	if len(parts) < 2 {
		b.reply(ctx, chatID, "Usage: `/link <code>` where `<code>` is from the web UI.")
		return
	}
	code := strings.ToUpper(strings.TrimSpace(parts[1]))

	// Look up non-expired code, consume it, bind chat_id to user.
	var userID string
	err := b.pool.QueryRow(ctx, `
		SELECT user_id FROM telegram_link_codes
		WHERE code = $1 AND expires_at > NOW()
	`, code).Scan(&userID)
	if err != nil {
		b.reply(ctx, chatID, "⚠️ Code invalid or expired. Generate a new one in the web UI.")
		return
	}

	// Enforce 1:1 — if another user already holds this chat_id, refuse.
	var existing string
	_ = b.pool.QueryRow(ctx, `SELECT id FROM users WHERE telegram_chat_id = $1`, chatID).Scan(&existing)
	if existing != "" && existing != userID {
		b.reply(ctx, chatID, "⚠️ This Telegram chat is already linked to another Elibri account. Use `/unlink` there first.")
		return
	}

	// Bind and burn the code.
	_, err = b.pool.Exec(ctx, `
		UPDATE users SET telegram_chat_id = $2, telegram_linked_at = NOW() WHERE id = $1
	`, userID, chatID)
	if err != nil {
		log.Printf("[tg bot] link bind error: %v", err)
		b.reply(ctx, chatID, "⚠️ Internal error. Try again in a moment.")
		return
	}
	_, _ = b.pool.Exec(ctx, `DELETE FROM telegram_link_codes WHERE code = $1`, code)

	b.reply(ctx, chatID, "✅ Linked. You'll receive live trade alerts here when scenarios trigger HC signals.")
}

func (b *Bot) replyUnlink(ctx context.Context, chatID int64) {
	res, err := b.pool.Exec(ctx, `
		UPDATE users SET telegram_chat_id = NULL, telegram_linked_at = NULL
		WHERE telegram_chat_id = $1
	`, chatID)
	if err != nil {
		log.Printf("[tg bot] unlink error: %v", err)
		b.reply(ctx, chatID, "⚠️ Internal error.")
		return
	}
	if res.RowsAffected() == 0 {
		b.reply(ctx, chatID, "No Elibri account was linked to this chat.")
		return
	}
	b.reply(ctx, chatID, "🔓 Unlinked. Re-link with `/link <code>` anytime.")
}

func (b *Bot) reply(ctx context.Context, chatID int64, text string) {
	_, _ = b.client.SendMessage(ctx, &tg.SendMessageParams{
		ChatID:    chatID,
		Text:      text,
		ParseMode: models.ParseModeMarkdown,
	})
}

// IssueLinkCode creates a 6-char one-time code valid 10 min. Called by the
// /api/v1/telegram/link HTTP handler. Returns (code, expiresAt).
func IssueLinkCode(ctx context.Context, pool *pgxpool.Pool, userID string) (string, time.Time, error) {
	// 3 random bytes → 6 hex chars, uppercase.
	buf := make([]byte, 3)
	if _, err := rand.Read(buf); err != nil {
		return "", time.Time{}, err
	}
	code := strings.ToUpper(hex.EncodeToString(buf))
	expiresAt := time.Now().Add(10 * time.Minute)

	_, err := pool.Exec(ctx, `
		INSERT INTO telegram_link_codes (code, user_id, expires_at)
		VALUES ($1, $2, $3)
	`, code, userID, expiresAt)
	if err != nil {
		return "", time.Time{}, err
	}
	return code, expiresAt, nil
}
