# Elibri FX — ML Pipeline Report

**Generated:** (to be filled by `run_after_training.sh`)
**Model version:** ensemble_v2
**Training scope:** top-5 USDT pairs × 3 timeframes × up to 8 years history

---

## 1. Methodology

### Training data
- **Source:** Binance Spot public API (no key required for klines)
- **Pairs:** BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, BNBUSDT
- **Timeframes:** 1h, 4h, 1d
- **Period:** 2018-04 → 2026-04 (8 years; SOL from 2020 since launch)
- **Market regimes covered:**
  - 2018 Crypto winter (-80%)
  - 2020 COVID crash (-50% in 2 days)
  - 2021 Bull to ATH
  - May 2022 Terra/LUNA collapse
  - Nov 2022 FTX blowup
  - 2023 SVB banking crisis
  - 2024 BTC ETF + halving
  - Russia-Ukraine war (Feb 2022 →)

### Features (31 total)
Bit-compatible Python ↔ Go implementation. Any formula change = version bump.

| Category | Features |
|----------|----------|
| Momentum | RSI 7/14/21, MACD hist, MACD signal, BB position, Stoch K |
| Trend | EMA cross 20/50, EMA cross 50/200, ADX 14, price vs EMA 50/200 |
| Volatility | ATR/close, BB width, volatility regime (ATR percentile 100d) |
| Volume | vol ratio 5/20, taker-buy ratio |
| Price action | return 1/5/20, higher highs 10, lower lows 10 |
| Candlestick | doji, engulfing, hammer |
| Cross-asset | BTC correlation 30, BTC beta 30 |
| Lagged | RSI 14 lag-4, return 5 lag-4, vol ratio 20 lag-4 |

### Model architecture
Stacked ensemble with 3 base learners + 1 meta:
- **XGBoost** (200 trees, depth 5, subsample 0.8)
- **LightGBM** (200 trees, depth 6, subsample 0.8)
- **RandomForest** (200 trees, depth 10)
- **Meta:** Logistic Regression on OOF base probabilities (3-fold stacking)

### Validation
- **Walk-forward** cross-validation: train 24 months → test 3 months, expanding
- Produces ~24 folds per model (quarterly windows)
- Separate bull/bear/range metrics via BTC 50d-EMA trend filter

### Key metric: High-Confidence Precision
Raw accuracy in crypto direction prediction is typically 51-55%. This is
roughly coin-flip territory, making Raw-accuracy models NOT profitable.

**What actually works:** filter to bars where the model is VERY confident,
trade only those. Metric: `hc_precision` = precision of signals with
`probability > 0.65` OR `probability < 0.35`. Smaller subset (5-15% of bars),
but much higher precision (target: 65-80%).

---

## 2. Trained Models

(Auto-populated after training finishes)

<!-- MODELS_TABLE -->

---

## 3. Backtest Results

OOS simulation: train on first 70%, test on last 30%. Trades triggered on
HC signals, exit on SL (1.5×ATR) or TP (2.5×ATR). Position sizing: 5% equity
at risk per trade.

<!-- BACKTEST_TABLE -->

### Per-Regime Breakdown

<!-- REGIME_TABLE -->

---

## 4. Paper Trading (last 90 days)

Virtual portfolio: $2,000 per (symbol, timeframe) = $20k total capital across
5 pairs × 2 timeframes.

<!-- PAPER_TRADES_TABLE -->

---

## 5. Deployment

### Files
- `models/{sym}_{iv}_v{timestamp}.json` — trained ensemble (RF trees + meta weights)
- `models/{sym}_{iv}_patterns.json` — k-NN pattern index (BallTree)
- `models/latest.json` — pointer: key → filename mapping
- `logs/best_thresholds.json` — per-model HC threshold (from `analyze_thresholds.py`)

### Go backend loads at startup
```go
ml.LoadModelsV2()      // reads models/latest.json
ml.LoadThresholds()    // reads logs/best_thresholds.json
```

### API
- `POST /api/v1/ml/predict` — ensemble inference + pattern matching
- `GET /api/v1/ml/models` — list loaded models + health
- `POST /api/v1/ml/reload` — hot-reload after retraining
- `POST /api/v1/ml/train` — trigger background training via Python script

---

## 6. Honest Limitations

1. **Accuracy ceiling:** 51-55% raw on crypto is industry-standard.
   Top hedge funds sit at 52-53%. 95%+ would indicate overfitting or leakage.

2. **HC filter reduces opportunity:** 65-80% precision on 5-15% of bars means
   ~1-3 trades per week per pair. Not a scalping strategy.

3. **Regime fragility:** Models trained through 2022 bear may underperform
   in novel regimes (e.g., 2024 ETF spot flows introduced new dynamics).

4. **No slippage/fees in backtest.** Realistic fills will be 0.05-0.1% worse.
   Binance spot taker fee: 0.1%.

5. **Fallback layer:** When V2 model missing for a symbol/TF, `PredictV2`
   falls back to the 6-feature V1 GBDT. Always check `model_version` field.

---

## 7. Next Steps (post-MVP)

- **Expand to top-20 pairs** — 40 more models. Needs ~6 more hours compute.
- **Calibration:** Isotonic regression over base probs to spread meta output.
- **Feature engineering:** Order-book imbalance (needs Binance L2 feed),
  funding rate (Binance futures), social sentiment scores.
- **Deep learning:** LSTM/Transformer on 48-bar sequences for pattern capture.
- **Real execution:** Bind `internal/crypto/execution.go` to Binance testnet
  API with user-supplied keys; run paper trades weekly to collect real fill
  statistics.
