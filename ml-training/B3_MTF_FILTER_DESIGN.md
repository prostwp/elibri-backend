# Session B.3 d6 — 1h as MTF Entry-Confirmation Filter

**Rationale:** Session B.3 d1-d5 показали что 1h model — **trend-follower short-biased**
с catastrophic drawdown в non-bear регimes (Bull 2024 H1: Cons -20 Sharpe, Bal -12, Agg -10).
Direct trading 1h сигналов не viable. Но 1h retains predictive value as an MTF voter when
used to confirm / deny signals from higher TF (4h).

## Current production scenario (Patch 3)

```
User creates strategy → scenario runner polls (4h → every 15min)
  → for each interval independently:
      ML predict (v2) → HC threshold → tier filters → alert
  → Telegram push
```

1h runs as independent tracker. B.2 decision deploys 4h paper + potential live.
1h scenario would also be deployed — this proposal REMOVES that.

## Proposed — 1h as 4h confirmator

```
User creates strategy → scenario runner polls (4h → every 15min)
  → ML predict 4h (v2) → HC threshold → tier filters
  → NEW: check 1h co-direction vote:
       - Compute 1h proba (lookup cached or predict on-demand)
       - If 1h direction == 4h direction: PASS, boost confidence +10%
       - If 1h direction opposite: BLOCK, reason="1h disagrees"
       - If 1h neutral (proba inside [1-hc_1h, hc_1h]): PASS, no boost
  → alert
```

**Expected impact (from backtest data):**
- 4h Conservative canonical: 26 trades, Sharpe +0.76
- 4h canonical where 1h co-direction: estimated ~15-18 trades remaining (1h predicts
  short most of the time same as 4h, so agreement rate high). Slight Sharpe
  improvement from culling disagreement (where 1h says long → often spike zones).
- 4h Conservative + 1h confirm: projected Sharpe +1.0-1.3 (modest bump).

**Measurement:** after implementation, rerun backtest_v2 with new 1h-confirm
gate; Sharpe should go up OR stay equal. Never should go down — if it does,
roll back.

## Implementation plan

### Backend (Go)

1. `internal/ml/predict_v2.go` — add `PredictV2WithConfirmator(ctx, interval, confirmInterval)` wrapper that runs two predictions and returns combined.
2. `internal/scenario/evaluate.go` — after current 4h inference, call `PredictV2WithConfirmator("4h", "1h")` when strategy opts in.
3. `internal/ml/classify.go` — optional new label reason `"1h_disagrees"` for observability.
4. Feature flag on strategy: `mtf_confirm_interval` column (default NULL = disabled).

### Frontend

1. `RiskManagerNode.tsx` add checkbox "Require 1h confirmation for 4h signals" (default OFF for existing users, ON for new strategies).
2. UI hint: "Adds 1h voter that must agree with 4h direction. Expected: fewer alerts, higher reliability."

### Database

Migration: `ALTER TABLE strategies ADD COLUMN mtf_confirm_interval VARCHAR(8);`

### Tests

- Go unit test: `TestPredictV2WithConfirmator_AgreeBoosts` / `TestPredictV2WithConfirmator_DisagreeBlocks`.
- Parity test unaffected (no feature changes).

## What this does NOT solve

- 1h on its own as direct source — dropped, not viable.
- Bull-market weakness of 4h — 1h confirmator won't help in regime где ОБА шортят bull.
- Need MTF Fusion Architecture (Session C) для full resolution.

## What this DOES solve

- 1h model is kept in production **without losing money** (it stops generating bad direct
  trades, starts acting as a veto vote for 4h).
- Users who had 1h scenarios now have them rerouted to 4h-confirm role automatically via
  migration + backfill.
- Investor narrative preserved: "We trained 1h model but discovered its direct edge
  collapses outside bear markets; we reassigned it to confirm 4h decisions, giving a
  multi-timeframe consensus check."

## Decision deferred to Session C

Session C (MTF Fusion Architecture) was queued as post-B-completion work. It will
formalize the full 1d → 4h → 1h → 15m → 5m ladder. This doc is the 1h slice of that
design, ready to land early if needed.

## Not in this patch

- No retrain needed.
- No production cutover of 1h models (they stay in `latest.json` as feature extractors).
- 15m/5m unchanged — B.4/B.5 to decide their roles.
