package scenario

import (
	"context"
	"log"
	"strings"

	"github.com/prostwp/elibri-backend/internal/store"
	"github.com/redis/go-redis/v9"
)

// PublishToStream fans the alert out to a Redis Stream for downstream consumers
// (Telegram bots, future webhook subscribers, analytics pipelines).
//
// Why streams instead of pub/sub:
//   - Persistent: if a bot is offline, messages buffer until it reconnects
//   - Acknowledged: consumers explicitly XACK, otherwise redelivery on idle
//   - Multi-consumer ready: per-author bots subscribe to different streams
//
// Stream key shapes (V4 author routing):
//   - Author-backed signals: signals:author:<slug>     e.g. "signals:author:gold_news"
//   - Personal/legacy signals: signals:<base>:<interval>  e.g. "signals:btc:4h"
//
// The bot menu in Telegram derives subscription topology from author_slug,
// so each author gets a dedicated stream keyspace. Personal user-owned
// strategies (no author_slug) keep the legacy by-symbol stream so the
// existing Telegram channel pipeline keeps working unchanged.
//
// Failure to publish is intentionally non-fatal: alerts are already durable
// in postgres `alerts` table. A bot that missed a publish can be backfilled
// by replaying from postgres or by re-XADDing the row manually.
//
// Schema MUST stay in sync with telegram-bot/src/formatters/default.py:
// changing or removing a key here = breaking the bot's formatter.
func PublishToStream(ctx context.Context, a *Alert) {
	if store.Redis == nil {
		// Redis was not configured / failed to connect at startup — alerts
		// still go to postgres + Go-side Telegram fanout. Streams just no-op.
		return
	}

	var streamKey string
	if a.AuthorSlug != "" {
		streamKey = "signals:author:" + a.AuthorSlug
	} else {
		streamKey = "signals:" + extractSymbolBase(a.Symbol) + ":" + a.Interval
	}

	values := map[string]interface{}{
		"alert_id":          a.ID,
		"symbol":            a.Symbol,
		"interval":          a.Interval,
		"direction":         a.Direction,
		"label":             a.Label,
		"confidence":        a.Confidence,
		"entry":             a.EntryPrice,
		"stop_loss":         a.StopLoss,
		"take_profit":       a.TakeProfit,
		"position_size_usd": a.PositionSizeUSD,
		"bar_time":          a.BarTime,
		"strategy_id":       a.StrategyID,
		"user_id":           a.UserID,
		"author_slug":       a.AuthorSlug,
		"is_premium":        a.IsPremium,
	}

	// MaxLen with approximate trim (~) keeps the stream bounded so a misbehaving
	// producer can't blow up Redis memory. 10k entries × ~200 bytes = ~2MB hard
	// ceiling per stream — easily enough for any realistic backlog and replay.
	args := &redis.XAddArgs{
		Stream: streamKey,
		Values: values,
		MaxLen: 10000,
		Approx: true,
	}

	id, err := store.Redis.XAdd(ctx, args).Result()
	if err != nil {
		log.Printf("[stream] XADD %s failed for alert %s: %v", streamKey, a.ID, err)
		return
	}
	log.Printf("[stream] XADD %s id=%s alert=%s %s %s", streamKey, id, a.ID, a.Direction, a.Symbol)
}

// extractSymbolBase strips the quote currency from a Binance-style symbol.
// BTCUSDT → btc, ETHUSDT → eth, XAUUSD → xau, EURUSD → eur.
// Lower-cased to keep stream keys stable regardless of input casing.
func extractSymbolBase(symbol string) string {
	s := strings.ToLower(symbol)
	for _, suffix := range []string{"usdt", "usdc", "usd"} {
		if strings.HasSuffix(s, suffix) {
			return strings.TrimSuffix(s, suffix)
		}
	}
	return s
}
