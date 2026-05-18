# vast.ai pipeline — Stage 2 macro + AutoGluon + multi-agent fusion

## What's new (2026-04-26)

This adds an AutoGluon-based, regime-aware, multi-agent training pipeline
**without removing** the existing XGB+LGBM+RF stack. The old `train.py` and
`models/latest.json` keep working; the new files write to parallel pointers
(`models/latest_autogluon.json`, `models/fusion_*/`) so we can compare.

New files:

| File | Role |
|---|---|
| `fetch_macro_features.py` | Real Binance/alternative.me/CoinGecko fetchers (was scaffold/mocks). Pulls funding, OI, F&G, BTC.D into `data/macro/*.parquet`. |
| `regime_classifier.py` | Bull/chop/bear classifier on 200d EMA slope ÷ ATR. Buckets into windows + picks 5 test slots covering all regimes. |
| `feature_engine.py` (extended) | New `attach_macro_features()` adds 8 macro columns. `FEATURE_NAMES_WITH_MACRO` = 53 features. Old `FEATURE_NAMES` (45) preserved for Go-parity. |
| `train_autogluon.py` | Drop-in for `train.train_one_3class`. AutoGluon `TabularPredictor`, `multi:softprob`, holdout eval, leaderboard. |
| `wfcv_regime_aware.py` | 5-fold WFCV with regime stratification. Ground truth: regime tags from BTC 1d slope. Reports per-regime Sharpe + direction balance. |
| `multi_agent_fusion.py` | Long-specialist + Short-specialist + Judge (average / veto / learned). Solves the 100%-short-bias collapse from Patch 4. |

## Bootstrap on vast.ai

```bash
ssh -i ~/.ssh/id_ed25519_vast -p <port> root@<host>
cd /root/elibri-backend/ml-training

# 1. Pull updated repo (pre-flight: confirm new files arrived)
git pull
ls -la fetch_macro_features.py regime_classifier.py train_autogluon.py \
       wfcv_regime_aware.py multi_agent_fusion.py

# 2. Install AutoGluon (one-time, ~5 minutes — ~1.5 GB)
pip install -r requirements.txt

# 3. Refresh macro data on the GPU box (fetcher uses HTTP; 1-2 minutes)
python fetch_macro_features.py --also-eth --start 2018-01-01

# 4. Verify regime classifier picks reasonable windows
python regime_classifier.py --parquet data/BTCUSDT_1d.parquet
```

## Recommended run order (Scenario A: Crypto ML / PREMIUM 4h BTC)

Each step takes a tmux pane so disconnect-resilient. Total budget ≈ 4-5 GPU-hr.

### Step 1 — Single-shot AutoGluon baseline (≈30 min)

```bash
tmux new -s ag_baseline
python train_autogluon.py \
    --symbol BTCUSDT --interval 4h \
    --with-macro \
    --presets medium_quality --time-limit 1500 \
    --eval-metric log_loss
# Output: models/ag_BTCUSDT_4h_macro/, logs/autogluon_BTCUSDT_4h_macro.json
```

If holdout `f1_macro` ≥ 0.40 and `prec_long_55 ≥ 50%`, proceed to Step 2.
Otherwise, inspect `leaderboard` in the JSON and re-run with `presets=good_quality`.

### Step 2 — Walk-forward CV (≈60 min)

```bash
tmux new -s wfcv
python wfcv_regime_aware.py \
    --symbol BTCUSDT --interval 4h \
    --with-macro \
    --presets medium_quality --time-limit 600
# Output: logs/wfcv_regime_BTCUSDT_4h_macro.json
```

Key fields in the report:
- `summary.sharpe_mean`, `sharpe_std`, `sharpe_min`
- `summary.long_share_mean` — should be 25-50% (anything < 10% = direction bias)
- `summary.verdict` — auto-classified GREEN / CONDITIONAL / KNOWN_LIMITATION / DROP

### Step 3 — Multi-agent fusion (≈45 min × 2 specialists + judge fit)

```bash
tmux new -s fusion
python multi_agent_fusion.py \
    --symbol BTCUSDT --interval 4h \
    --with-macro \
    --presets medium_quality --time-limit 1200 \
    --judge learned
# Output: models/fusion_BTCUSDT_4h_macro/
```

Compare `fusion_meta.json/holdout_metrics` against Step 1 `autogluon_*` log:

| Metric | Step 1 (single 3-class) | Step 3 (fusion) |
|---|---|---|
| accuracy | ≥0.50 ⇒ baseline | should be ≥ Step 1 |
| f1_macro | ≥0.40 ⇒ baseline | should be ≥ Step 1 |
| log_loss | lower is better | should be ≤ Step 1 |
| `long_share@0.55` | 25-50% target | should not be < 10% |
| `prec_long_55` | ≥50% | should be ≥ Step 1 |

### Step 4 — Cross-asset validation (optional, after BTC GREEN)

If BTC 4h fusion passes Step 3, repeat Step 3 for ETH 4h:

```bash
python multi_agent_fusion.py \
    --symbol ETHUSDT --interval 4h --with-macro \
    --presets medium_quality --time-limit 1200 --judge learned
```

If both BTC and ETH pass with similar metrics, add to `models/latest_autogluon.json`
and proceed to paper-trade arming (see `B2_PAPER_TRADE_SETUP.sh`).

## Failure modes seen in earlier patches (now mitigated)

| Failure | Old cause | New mitigation |
|---|---|---|
| 100% short bias | 3-class softmax on 58/39/3 imbalanced target | Multi-agent: long-specialist & short-specialist train independently against their own balanced binary tasks |
| WFCV avg Sharpe -1.07 (Patch 4 Stage 1) | Per-fold retrain hides regime shifts; 24mo expanding folds were bull-heavy | `wfcv_regime_aware.py` picks regimes deliberately; per-window, per-regime metrics |
| Mock macro data (Patch 2H) | `fetch_macro_features.py` was scaffold with `np.random` | Real Binance + alternative.me + CoinGecko fetchers; manifest tracks coverage gaps |
| Threshold/precision inversion (4h Patch 2H) | Asymmetric tb barriers + binary target trained on bull-heavy | 3-class target with proper hold class + AutoGluon's calibrated probabilities |

## Honest limits we ship with

- **OI history**: Binance free API caps at 30 days. Pre-coverage bars get
  neutral fill (oi_norm=1.0). Forward-going inference gets fresh OI; backtest
  effectively trains "OI absent before X". Coinglass/Glassnode unlock full
  history for $30-100/mo (currently NOT subscribed).
- **BTC dominance**: only current snapshot from CoinGecko free. Acts as a
  constant in training. Stage 3 work.
- **Funding rate**: Binance Futures launched 2019-09-10. Pre-2019-09 → 0.0
  (correct: no perpetuals existed).

These are documented in `data/macro/manifest.json` after every fetcher run.

## Local smoke test (no AG, no GPU)

For sanity-checking on the dev machine before pushing to vast.ai:

```bash
source .venv/bin/activate
python regime_classifier.py        # writes regime_manifest_BTCUSDT_1d.json
python -c "
from train_autogluon import build_dataset
feat, _, meta = build_dataset('BTCUSDT', '4h', years=2.0, with_macro=True)
print('rows:', len(feat), 'features:', meta['n_features'])
print('class dist:', meta['class_distribution'])
"
```

`train_autogluon.py` and `multi_agent_fusion.py` will fail at the AG import
locally — that's expected. They're production scripts; only `regime_classifier`
+ `fetch_macro_features` + `feature_engine` smoke-test cleanly without AG.
