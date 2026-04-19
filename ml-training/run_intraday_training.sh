#!/bin/bash
# Waits for data_fetcher to finish, then trains 5m/15m/1h models.
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

# Wait for fetch PID.
if [ -f /tmp/fetch_pid ]; then
  FETCH_PID=$(cat /tmp/fetch_pid)
  echo "Waiting for fetch (PID=$FETCH_PID)…"
  while kill -0 "$FETCH_PID" 2>/dev/null; do
    sleep 20
  done
  echo "Fetch completed at $(date -u +%H:%M:%S)"
fi

# Verify parquet files exist for all expected combinations.
echo ""
echo "━━━ Cached parquets ━━━"
for pair in BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT; do
  for iv in 5m 15m 1h; do
    if [ -f "data/${pair}_${iv}.parquet" ]; then
      size=$(stat -f%z "data/${pair}_${iv}.parquet" 2>/dev/null || echo "0")
      echo "  ${pair}_${iv}.parquet  $(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B")"
    else
      echo "  ${pair}_${iv}.parquet  MISSING"
    fi
  done
done

# Training: 1h first (fastest), then 15m, then 5m.
# Different --years per TF to balance training time vs regime coverage.
echo ""
echo "━━━ Training 1h × 5 pairs × 8 years ━━━"
python -u train.py --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT --intervals 1h --years 8 2>&1 \
  | grep --line-buffered -E "━━━|saved|→ avg|FAILED|candles:" | tee -a logs/train_intraday_1h.log

echo ""
echo "━━━ Training 15m × 5 pairs × 4 years ━━━"
python -u train.py --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT --intervals 15m --years 4 2>&1 \
  | grep --line-buffered -E "━━━|saved|→ avg|FAILED|candles:" | tee -a logs/train_intraday_15m.log

echo ""
echo "━━━ Training 5m × 5 pairs × 2 years ━━━"
python -u train.py --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT --intervals 5m --years 2 2>&1 \
  | grep --line-buffered -E "━━━|saved|→ avg|FAILED|candles:" | tee -a logs/train_intraday_5m.log

echo ""
echo "━━━ analyze_thresholds ━━━"
python -u analyze_thresholds.py --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT --intervals 5m 15m 1h 2>&1 \
  | grep --line-buffered -vE "warnings.warn|NotOpenSSL|UserWarning|RuntimeWarning|overflow|divide by zero|invalid value|raw_prediction|ret = a @" | tee logs/analyze_intraday.log

echo ""
echo "━━━ INTRADAY PIPELINE DONE ══━━ at $(date -u +%H:%M:%S)"
