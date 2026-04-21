# Patch 2H-extended — vast.ai Playbook

**Prepared:** 2026-04-22
**Budget:** ≤2 GPU-hours total on RTX 4060 Ti (~ what Patch 2H used).
**Target:** fix BTC 1h/4h Conservative divergence diagnosed in DEVLOG Patch 2H-extended section.

## Preflight (30 sec, mac)

```bash
ssh -i ~/.ssh/id_ed25519_vast -p 26287 root@67.70.78.39 "nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; cd /workspace/elibri-backend && git log --oneline -3"
```

Confirm GPU warm, repo at `74b488b` (`Patch 2H preparation`). If stale — `git pull origin main` first.

## Phase 1 — Retrain 1h/4h/1d with wider triple-barrier (45-60 min GPU)

Edit `train.py` `TF_CONFIG` locally on vast.ai before kickoff. Do NOT commit this change yet (experimental):

```python
TF_CONFIG = {
    "5m":  {"horizon": 12, "tb_upper": 2.5, "tb_lower": 1.5, "n_est": 200, "xgb_depth": 5},
    "15m": {"horizon": 16, "tb_upper": 2.5, "tb_lower": 1.5, "n_est": 200, "xgb_depth": 5},
    # Patch 2H-extended H1: widen barriers on 1h to match scalp TFs
    "1h":  {"horizon": 24, "tb_upper": 2.5, "tb_lower": 1.5, "n_est": 200, "xgb_depth": 5},
    # Patch 2H-extended H2: 4h needs even wider + shorter horizon for conviction-only setups
    "4h":  {"horizon": 12, "tb_upper": 3.0, "tb_lower": 2.0, "n_est": 200, "xgb_depth": 5},
    # Patch 2H-extended H3: 1d — shorter horizon, wider barrier
    "1d":  {"horizon": 3,  "tb_upper": 2.0, "tb_lower": 1.5, "n_est": 150, "xgb_depth": 3},
}
```

**H5-required co-edit** (found during H5 audit 2026-04-22): `analyze_thresholds.py`
lines 204-208 pick `tb_upper`/`tb_lower` from CLI argparse defaults (1.5/1.0) for
any TF other than 5m/15m. After we widen 1h/4h/1d barriers in TF_CONFIG, the
threshold sweep in Phase 2 will compute thresholds for a model trained with
different barriers than the candidates → silent mismatch. Fix before Phase 2:

```python
# analyze_thresholds.py main() loop around line 204 — read from TF_CONFIG:
if args.target_mode == "tb_atr":
    tf_cfg_local = TF_CONFIG.get(iv, {"tb_upper": args.tb_upper, "tb_lower": args.tb_lower})
    upper = tf_cfg_local.get("tb_upper", args.tb_upper)
    lower = tf_cfg_local.get("tb_lower", args.tb_lower)
else:
    upper, lower = args.tb_upper, args.tb_lower
```

Remove the `--scalp-upper`/`--scalp-lower` hardcoded scalp-only branch. TF_CONFIG
becomes the single source of truth across train.py / analyze_thresholds.py /
backtest_v2.py.

Then retrain ONLY 1h/4h/1d on BTC:

```bash
cd /workspace/elibri-backend/ml-training
python train.py \
    --symbols BTCUSDT \
    --intervals 1h 4h 1d \
    --target-mode tb_atr 2>&1 | tee logs/patch2h_ext_train.log
```

Expected wall time: ~15-20 min on RTX 4060 Ti.

**Checkpoint:** after train finishes, inspect the 3 new `models/BTCUSDT_{1h,4h,1d}_v<ts>.json` files and their training logs. Verify:
- `n_samples_after_drop` on 1h: should drop ~20-30% more than before (wider barriers = more timeouts → label=-1 → dropped).
- Metadata `tb_upper` / `tb_lower` match the new TF_CONFIG.

DO NOT flip `latest.json` pointer yet — these are candidate models.

## Phase 2 — Regenerate thresholds for new models (10 min GPU)

```bash
python analyze_thresholds.py \
    --symbols BTCUSDT \
    --intervals 1h 4h 1d \
    --target-mode tb_atr \
    --policy-min-precision 0.60 \
    --out logs/patch2h_ext_thresholds.json 2>&1 | tee logs/patch2h_ext_threshold.log
```

**Primary success criterion** (H1+H2 confirmation): the proba_max on new 1h model should drop **from 0.72 → 0.55-0.62 range** AND hc_table should show **monotonic precision increase** with threshold (like 5m does). If precision still decreases with threshold → hypothesis failed, do not proceed with Phase 3.

## Phase 3 — Honest backtest on candidate models (30 min GPU)

Point backtest at the candidate `models/` dir explicitly (don't touch production `latest.json`):

```bash
# Temporarily point at candidates via a local latest.candidate.json
python -c "
import json, pathlib
d = pathlib.Path('models')
cands = {}
for tf in ['1h', '4h', '1d']:
    files = sorted(d.glob(f'BTCUSDT_{tf}_v*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    cands[f'BTCUSDT_{tf}'] = files[0].name
pathlib.Path('models/latest.candidate.json').write_text(json.dumps(cands, indent=2))
print(cands)
"

python backtest_v2.py \
    --symbols BTCUSDT \
    --intervals 1h 4h 1d \
    --tiers conservative balanced aggressive \
    --best-thresholds logs/patch2h_ext_thresholds.json \
    --models-pointer models/latest.candidate.json \
    --out logs/backtest_v2_ext 2>&1 | tee logs/patch2h_ext_backtest.log
```

NOTE: `backtest_v2.py` may not currently have `--models-pointer` — if it doesn't, the cleanest path is to `cp` the candidate JSON over the production one in a scratch `models_ext/` folder and run with `MODELS_DIR=models_ext` env var, depending on how train.py loads them. Worst case: temporarily swap `latest.json` → run backtest → swap back. Coder agent should verify the code supports this before kicking off GPU time.

## Success criteria (must meet ALL)

- **1h Conservative:** Sharpe ≥ +1, WR ≥ 45%, ≥5 trades OOS, long-short ratio no worse than 10/90 (was 0/100 on prod).
- **4h Conservative:** Sharpe ≥ 0 (break-even acceptable given low trade count), WR ≥ 40%, max DD ≤ 5%.
- **1d Conservative:** direction accuracy on OOS still ≥ 0.55 (existing role: trend filter).
- **5m Conservative:** UNTOUCHED. Still Sharpe ~+9.4. Confirm by re-running backtest on prod model pointer — should reproduce within ±10%.

If any criterion fails → roll back TF_CONFIG edits, document in DEVLOG why hypothesis was wrong, do NOT push candidate models to prod.

## If successful — production cutover

1. Copy candidate models: `cp models/BTCUSDT_{1h,4h,1d}_v*.json models/` (already there).
2. Update `models/latest.json` with new 1h/4h/1d version pointers.
3. `scp` `models/` and `logs/best_thresholds.json` to mac.
4. Rebuild Go backend (`cd /Users/admin/NodeElibiri/elibri-backend && go build ./cmd/server && ./server`).
5. Verify `curl -s localhost:8080/health` shows 25 models loaded.
6. Commit TF_CONFIG change in `ml-training/train.py` with message like `fix(ml): Patch 2H-extended — widen tb barriers on 1h/4h/1d`.
7. Update DEVLOG with new honest metrics table.

## What NOT to do

- Do NOT touch 5m/15m TF_CONFIG in this patch.
- Do NOT retrain production models until backtest success criteria are met.
- Do NOT extend to ETH/SOL/BNB/XRP — that's separate sprint after BTC is verified.
- Do NOT spend more than 2 GPU-hours total on this — if hypotheses fail within budget, stop and redesign.
