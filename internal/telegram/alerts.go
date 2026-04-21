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
	bot       *Bot
	pool      *pgxpool.Pool
	ch        chan *scenario.Alert
	done      chan struct{}
	once      sync.Once
	maxPerDay int // AlertsMaxPerDayPerUser from config; 0 disables the quota
}

const (
	queueSize    = 256
	perChatDelay = 350 * time.Millisecond
)

// NewAlertQueue constructs a queue. bot may be nil (Telegram disabled —
// alerts still log through DB, runner stays functional). maxPerDay is the
// per-user daily cap enforced in deliver(); 0 disables the check.
func NewAlertQueue(bot *Bot, pool *pgxpool.Pool, maxPerDay int) *AlertQueue {
	return &AlertQueue{
		bot:       bot,
		pool:      pool,
		ch:        make(chan *scenario.Alert, queueSize),
		done:      make(chan struct{}),
		maxPerDay: maxPerDay,
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
// Exits when ctx is done; before exit, drains any remaining alerts with a
// short-lived background context so in-flight Telegram sends + DB marks
// complete even after SIGINT (PHASE 1 fix #2: shutdown drain).
func (q *AlertQueue) Run(ctx context.Context) {
	log.Printf("[tg alerts] running (bot_enabled=%t)", q.bot != nil)
	defer q.once.Do(func() { close(q.done) })
	for {
		select {
		case <-ctx.Done():
			q.drainRemaining()
			return
		case a := <-q.ch:
			q.deliver(ctx, a)
		}
	}
}

// drainRemaining processes any buffered alerts before Run returns. Uses its
// own background context with a short timeout so Postgres + Telegram API
// calls can actually complete after the parent ctx is cancelled. Safe to
// call with an empty queue — returns immediately.
func (q *AlertQueue) drainRemaining() {
	drainCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	drained := 0
	for {
		select {
		case a := <-q.ch:
			q.deliver(drainCtx, a)
			drained++
			if drainCtx.Err() != nil {
				log.Printf("[tg alerts] drain deadline hit; %d delivered, %d dropped", drained, len(q.ch))
				return
			}
		default:
			log.Printf("[tg alerts] shutdown drain complete — %d alerts delivered", drained)
			return
		}
	}
}

// Done returns a channel that is closed when Run exits. main.go should
// <-alertQ.Done() after server.Shutdown to guarantee no lost alerts.
func (q *AlertQueue) Done() <-chan struct{} {
	return q.done
}

// deliver sends one alert. Errors are logged; the alert row stays in DB so
// the user can still see it via /api/v1/alerts history.
//
// PHASE 2L fix — per-delivery timeout + ctx-aware throttle so a single hung
// Telegram TCP connection can't eat the entire shutdown drain budget, and
// SIGINT during throttle sleep doesn't add unkillable 350ms per queued alert.
// Previously a slow Telegram response burned the 10s drain ctx for everyone.
func (q *AlertQueue) deliver(parentCtx context.Context, a *scenario.Alert) {
	// Cap one delivery at 5s so Run() and drain can make forward progress
	// even if Telegram servers are slow or unreachable. The DB row stays
	// intact; user sees the alert in web UI.
	ctx, cancel := context.WithTimeout(parentCtx, 5*time.Second)
	defer cancel()

	// Enforce daily alert quota per user. Config default 100 — adjust via env.
	// Previously AlertsMaxPerDayPerUser was loaded and CountAlertsLast24h
	// was defined, but nothing tied them together. Compounded by no rate
	// limit on scenario creation, one runaway scenario could flood Telegram.
	if q.maxPerDay > 0 {
		if n, err := scenario.CountAlertsLast24h(ctx, q.pool, a.UserID); err == nil && n > q.maxPerDay {
			log.Printf("[tg alerts] daily quota %d exceeded for user %s (count=%d); alert %s kept in DB only",
				q.maxPerDay, a.UserID, n, a.ID)
			return
		}
	}

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

	// Gentle per-chat throttle; bail immediately if the parent context
	// (SIGINT, drain deadline) fires so we don't strand the Run goroutine.
	select {
	case <-parentCtx.Done():
	case <-time.After(perChatDelay):
	}
}
