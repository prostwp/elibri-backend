#!/bin/bash
# Waits for train.py to finish, then runs full analysis pipeline:
#   1. analyze_thresholds.py  → best_thresholds.json
#   2. backtest.py            → per-strategy equity curves
#   3. paper_trade.py         → 90-day virtual portfolio
#   4. Reload Go backend models (HTTP POST to /ml/reload)
#   5. Append summary to logs/pipeline_summary.md
set -euo pipefail

cd "$(dirname "$0")"

TRAIN_PID_FILE="/tmp/train_pid"
if [ ! -f "$TRAIN_PID_FILE" ]; then
  echo "No /tmp/train_pid found — is training running?" >&2
  exit 1
fi

TRAIN_PID=$(cat "$TRAIN_PID_FILE")
echo "Waiting for train.py (PID=$TRAIN_PID)…"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
done
echo "Training ended at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source .venv/bin/activate

# 1. Analyze thresholds.
echo ""
echo "━━━ analyze_thresholds.py ━━━"
python -u analyze_thresholds.py --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT \
  --intervals 4h 1d 2>&1 | tee logs/analyze.log

# 2. Backtest (OOS walk-forward already inside each strategy).
echo ""
echo "━━━ backtest.py ━━━"
python -u backtest.py --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT \
  --intervals 4h 1d --years 8 2>&1 | tee logs/backtest.log

# 3. Paper trading (last 90 days).
echo ""
echo "━━━ paper_trade.py ━━━"
python -u paper_trade.py --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT \
  --intervals 4h 1d --days 90 2>&1 | tee logs/paper_trade.log

# 4. Reload backend models (requires backend running on :8080).
echo ""
echo "━━━ reload backend ━━━"
TOKEN=$(curl -s -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"mlv3@test.com","password":"mlv3test123"}' \
  | python -c "import sys,json; print(json.load(sys.stdin).get('token',''))" 2>/dev/null)
if [ -n "$TOKEN" ]; then
  curl -s -X POST -H "Authorization: Bearer $TOKEN" http://localhost:8080/api/v1/ml/reload
  echo ""
else
  echo "backend not running or login failed — manual reload required"
fi

# 5. Compose summary.
cat > logs/pipeline_summary.md <<EOF
# ML Pipeline Summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)

## Training
$(ls models/*_v*.json 2>/dev/null | wc -l | xargs echo "- Models trained:")

## Threshold Analysis
\`\`\`
$(tail -20 logs/analyze.log 2>/dev/null | grep -v "warnings.warn" | tail -15)
\`\`\`

## Backtest Results
\`\`\`
$(tail -25 logs/backtest.log 2>/dev/null | grep -v "warnings.warn" | tail -20)
\`\`\`

## Paper Trading (90 days)
\`\`\`
$(tail -15 logs/paper_trade.log 2>/dev/null | grep -v "warnings.warn" | tail -10)
\`\`\`
EOF

echo ""
echo "═══ pipeline done. See logs/pipeline_summary.md ═══"
cat logs/pipeline_summary.md
