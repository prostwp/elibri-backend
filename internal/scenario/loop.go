package scenario

import (
	"context"
	"encoding/json"
	"log"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// AlertSink is what the loop calls when Evaluate returns Ready=true.
// Runner provides a real implementation backed by the Telegram queue;
// tests can pass a stub.
type AlertSink interface {
	Push(ctx context.Context, a *Alert)
}

// scenarioLoop is one goroutine per active scenario. It wakes on TFToPoll,
// calls Evaluate, writes alerts through the sink, updates dedup state.
type scenarioLoop struct {
	scenario ActiveScenario
	pool     *pgxpool.Pool
	sink     AlertSink
	cancel   context.CancelFunc
	wg       *sync.WaitGroup // Runner's group — Add/Done bracket the goroutine
}

// Start launches the loop in the background. Registers with the Runner's
// WaitGroup so Runner.Shutdown() can block until the goroutine's final
// in-flight tick completes (Patch 2L).
func (l *scenarioLoop) Start(parent context.Context) {
	ctx, cancel := context.WithCancel(parent)
	l.cancel = cancel

	if l.wg != nil {
		l.wg.Add(1)
	}
	go func() {
		defer func() {
			if l.wg != nil {
				l.wg.Done()
			}
		}()
		defer log.Printf("[scenario %s] loop exited (%s %s)", l.scenario.ID, l.scenario.Symbol, l.scenario.Interval)
		poll := TFToPoll(l.scenario.Interval)
		ticker := time.NewTicker(poll)
		defer ticker.Stop()

		log.Printf("[scenario %s] started: %s %s tier=%s poll=%s",
			l.scenario.ID, l.scenario.Symbol, l.scenario.Interval, l.scenario.RiskTier, poll)

		// Immediate first tick so fresh scenarios don't wait a full period.
		l.tick(ctx)

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				l.tick(ctx)
			}
		}
	}()
}

func (l *scenarioLoop) Stop() {
	if l.cancel != nil {
		l.cancel()
	}
}

// tick does one evaluation pass. Isolated so panics in a single iteration
// don't kill the loop (recover handled by caller goroutine wrapper).
func (l *scenarioLoop) tick(ctx context.Context) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("[scenario %s] tick panic: %v", l.scenario.ID, r)
		}
	}()

	// Equity: runner is stateless on user balance; use a fixed $10k for
	// alert sizing until we wire it to user account state / Binance balance.
	// Frontend RiskManagerNode overrides this for UI display.
	const defaultEquityUSD = 10000.0

	ev := Evaluate(ctx, &l.scenario, defaultEquityUSD)
	if !ev.Ready {
		if ev.BlockReason != "" && ev.BlockReason != "same_bar" {
			log.Printf("[scenario %s] skip: %s (bar=%d)", l.scenario.ID, ev.BlockReason, ev.BarTime)
		}
		return
	}

	// Build alert + mark dedup.
	meta, _ := json.Marshal(map[string]any{
		"hc_pass":        ev.HCPass,
		"model_version":  ev.ModelVersion,
		"label_reason":   ev.LabelReason,
		"confidence":     ev.Confidence,
	})
	a := &Alert{
		UserID:          l.scenario.UserID,
		StrategyID:      l.scenario.ID,
		AuthorSlug:      l.scenario.AuthorSlug,
		IsPremium:       l.scenario.IsPremium,
		Symbol:          l.scenario.Symbol,
		Interval:        l.scenario.Interval,
		Direction:       ev.Direction,
		Label:           ev.Label,
		Confidence:      ev.Confidence,
		EntryPrice:      ev.EntryPrice,
		StopLoss:        ev.StopLoss,
		TakeProfit:      ev.TakeProfit,
		PositionSizeUSD: ev.PositionSizeUSD,
		BarTime:         ev.BarTime,
		CreatedAt:       time.Now(),
		Meta:            meta,
	}
	inserted, err := InsertAlert(ctx, l.pool, a)
	if err != nil {
		log.Printf("[scenario %s] alert insert error: %v", l.scenario.ID, err)
		return
	}
	if !inserted {
		// Dedup kicked in — a previous tick already fired this (bar, direction).
		return
	}

	// Real-time fanout via Redis Streams. Non-fatal: alerts are durable in
	// postgres regardless of Redis state. See streams.go for the contract.
	PublishToStream(ctx, a)

	// Update last_signal_* in strategies so next tick sees the same bar as dup.
	if err := MarkLastSignal(ctx, l.pool, l.scenario.ID, ev.BarTime, ev.Direction); err != nil {
		log.Printf("[scenario %s] mark last_signal error: %v", l.scenario.ID, err)
	}
	// Update in-memory copy too.
	l.scenario.LastSignalBarTime = ev.BarTime
	l.scenario.LastSignalDirection = ev.Direction

	// Enqueue for Telegram.
	if l.sink != nil {
		l.sink.Push(ctx, a)
	}
	log.Printf("[scenario %s] SIGNAL %s %s @ %.2f → TG enqueued",
		l.scenario.ID, ev.Direction, l.scenario.Symbol, ev.EntryPrice)
}
