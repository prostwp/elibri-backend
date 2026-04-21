#!/bin/bash
# Patch 2H-extended Phase 1 launcher — retrains BTC 1h/4h/1d with wider barriers.
# Waits for refetch to finish first (checks for 1d parquet presence).
set -euo pipefail
cd /root/elibri-backend/ml-training

# Preflight: wait for 8y parquets to exist (refetch may still be running).
needed=(5m 15m 1h 4h 1d)
for tf in "${needed[@]}"; do
  p="data/BTCUSDT_${tf}.parquet"
  while [ ! -f "$p" ] || [ $(stat -c%s "$p") -lt 40000 ]; do
    echo "waiting for $p..."
    sleep 20
  done
  echo "OK: $p ready ($(stat -c%s "$p") bytes)"
done

# Check horizon of each parquet.
python3 -c "
import pandas as pd
for tf in ['5m','15m','1h','4h','1d']:
    df = pd.read_parquet(f'data/BTCUSDT_{tf}.parquet')
    print(f'{tf} rows={len(df)} range={df.open_time.min()}..{df.open_time.max()}')
"

# Now retrain. TF_CONFIG already patched: 1h tb=2.5/1.5, 4h tb=3.0/2.0 h=12, 1d h=3.
python3 -u train.py --symbols BTCUSDT --intervals 1h 4h 1d --target-mode tb_atr 2>&1
echo "PHASE1_DONE"
