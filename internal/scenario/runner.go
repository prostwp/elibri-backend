package scenario

import (
	"context"
	"log"
	"sync"

	"github.com/jackc/pgx/v5/pgxpool"
)

// Runner owns the pool of per-scenario goroutines. One process-wide instance;
// main.go creates it, wires it to HTTP handlers via SetRunner, and calls
// StartAllActive on boot.
type Runner struct {
	pool  *pgxpool.Pool
	sink  AlertSink
	mu    sync.Mutex
	loops map[string]*scenarioLoop
	ctx   context.Context
}

// NewRunner constructs a Runner bound to a DB pool and alert sink.
// ctx is the parent lifecycle (usually main.go's signal-cancellable ctx).
func NewRunner(ctx context.Context, pool *pgxpool.Pool, sink AlertSink) *Runner {
	return &Runner{
		pool:  pool,
		sink:  sink,
		loops: map[string]*scenarioLoop{},
		ctx:   ctx,
	}
}

// StartAllActive hydrates loops from DB on boot — so a server restart
// doesn't silently abandon user scenarios.
func (r *Runner) StartAllActive(ctx context.Context) error {
	actives, err := ListActiveScenarios(ctx, r.pool)
	if err != nil {
		return err
	}
	log.Printf("[runner] hydrating %d active scenarios", len(actives))
	for _, s := range actives {
		r.startLoop(s)
	}
	return nil
}

// Start runs one scenario by its ID (called by HTTP handler after toggling
// is_active=true). Idempotent — double-start is a no-op.
func (r *Runner) Start(ctx context.Context, scenarioID string) error {
	r.mu.Lock()
	if _, exists := r.loops[scenarioID]; exists {
		r.mu.Unlock()
		return nil
	}
	r.mu.Unlock()

	// Re-fetch from DB to get fresh state (nodes_json, tier, interval).
	// Single-row query — cheaper than subscribing to changes.
	actives, err := ListActiveScenarios(ctx, r.pool)
	if err != nil {
		return err
	}
	for _, s := range actives {
		if s.ID == scenarioID {
			r.startLoop(s)
			return nil
		}
	}
	return nil // scenario not active (maybe toggled off concurrently) — silent noop
}

// Stop halts a single scenario's loop. Safe on missing IDs.
func (r *Runner) Stop(scenarioID string) {
	r.mu.Lock()
	loop, ok := r.loops[scenarioID]
	if ok {
		delete(r.loops, scenarioID)
	}
	r.mu.Unlock()

	if loop != nil {
		loop.Stop()
	}
}

// Shutdown cancels every loop. Called from main.go on SIGINT/SIGTERM.
func (r *Runner) Shutdown() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for id, loop := range r.loops {
		loop.Stop()
		delete(r.loops, id)
	}
	log.Printf("[runner] shutdown: all loops stopped")
}

// ActiveIDs returns the current running set — for /api/v1/scenarios/active.
func (r *Runner) ActiveIDs() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, 0, len(r.loops))
	for id := range r.loops {
		out = append(out, id)
	}
	return out
}

// IsRunning answers a single-ID check (for /scenarios/active per-row status).
func (r *Runner) IsRunning(scenarioID string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	_, ok := r.loops[scenarioID]
	return ok
}

// startLoop is the internal common path for hydrate + start.
func (r *Runner) startLoop(s ActiveScenario) {
	r.mu.Lock()
	if _, exists := r.loops[s.ID]; exists {
		r.mu.Unlock()
		return
	}
	loop := &scenarioLoop{
		scenario: s,
		pool:     r.pool,
		sink:     r.sink,
	}
	r.loops[s.ID] = loop
	r.mu.Unlock()

	loop.Start(r.ctx)
}
