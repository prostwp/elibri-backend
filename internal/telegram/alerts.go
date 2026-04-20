package telegram

import (
	"context"
	"log"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/prostwp/elibri-backend/internal/scenario"
)

// AlertQueue accepts alerts from scenario loops and delivers them via the
// Telegram bot. Implements scenario.AlertSink.
//
// Buffered channel with drop-on-overflow + per-chat rate limiting (3/sec
// per chat, Telegram API limit is 30/sec globally).
//
// If the Bot is nil (token not configured), alerts are logged and marked
// sent-skipped. Stays idempotent against InsertAlert dedup.
type AlertQueue struct {
	bot   *Bot
	pool  *pgxpool.Pool
	ch    chan *scenario.Alert
	done  chan struct{}
	once  sync.Once
}

const (
	queueSize    = 256
	perChatDelay = 350 * time.Millisecond
)

// NewAlertQueue constructs a queue. bot may be nil (Telegram disabled —
// alerts still log through DB, runner stays functional).
func NewAlertQueue(bot *Bot, pool *pgxpool.Pool) *AlertQueue {
	return &AlertQueue{
		bot:  bot,
		pool: pool,
		ch:   make(chan *scenario.Alert, queueSize),
		done: make(chan struct{}),
	}
}

// Push enqueues an alert non-blocking. Drops with a warning on overflow
// (better than back-pressuring the runner tick).
func (q *AlertQueue) Push(ctx context.Context, a *scenario.Alert) {
	select {
	case q.ch <- a:
	default:
		log.Printf("[tg alerts] queue full (%d), dropped alert %s", queueSize, a.ID)
	}
}

// Run drains the queue in one goroutine. Call from main.go under parent ctx.
// Exits when ctx is done or Shutdown is called.
func (q *AlertQueue) Run(ctx context.Context) {
	log.Printf("[tg alerts] running (bot_enabled=%t)", q.bot != nil)
	for {
		select {
		case <-ctx.Done():
			q.once.Do(func() { close(q.done) })
			return
		case a := <-q.ch:
			q.deliver(ctx, a)
		}
	}
}

// deliver sends one alert. Errors are logged; the alert row stays in DB so
// the user can still see it via /api/v1/alerts history.
func (q *AlertQueue) deliver(ctx context.Context, a *scenario.Alert) {
	// Resolve chat_id lazily — user may have linked TG after scenario started.
	chatID, linked, err := scenario.GetUserTelegramChatID(ctx, q.pool, a.UserID)
	if err != nil {
		log.Printf("[tg alerts] chat_id lookup error for user %s: %v", a.UserID, err)
		return
	}
	if !linked {
		// Not a failure — alert stays in DB, user sees it in web UI.
		return
	}

	if q.bot == nil {
		log.Printf("[tg alerts] bot disabled; alert %s stays in DB only", a.ID)
		return
	}

	text := FormatAlert(a)
	msgID, err := q.bot.SendMarkdown(ctx, chatID, text)
	if err != nil {
		log.Printf("[tg alerts] send failed for chat %d: %v", chatID, err)
		return
	}

	if err := scenario.MarkAlertSent(ctx, q.pool, a.ID, msgID); err != nil {
		log.Printf("[tg alerts] mark-sent DB error for alert %s: %v", a.ID, err)
	}

	// Gentle per-chat throttle.
	time.Sleep(perChatDelay)
}
