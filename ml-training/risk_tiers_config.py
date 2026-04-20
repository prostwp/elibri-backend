"""
risk_tiers_config.py — Python mirror of Go risk tier gating config (Patch 2C).

Three tiers (conservative / balanced / aggressive) that the Go runtime applies
BEFORE a signal is shown to the user. This file is the single source-of-truth
for the backtest so numbers match what end users will actually see.

Keep in lockstep with:
  - elibri-go-backend/internal/signals/risk_tiers.go (TBD)
  - elibri-backend/ml-training/backtest_v2.py

Knobs per tier:
  min_vol_pct[tf] : require atr_norm_14 (= ATR/price) >= this at entry. Below
                    it, the market is too thin for the stop distance to make
                    risk-reward sense. Intraday floors are lower because 5m ATR
                    is naturally smaller than 4h ATR.
  max_trades_per_day : hard cap on signals taken per calendar day (entry-side).
  allowed_labels  : trend_aligned (MTF agrees and ADX>20), mean_reversion
                    (flat MTF + extreme RSI/BB), or random (no label → drop).
  risk_per_trade_pct : fraction of equity risked on SL distance (Turtle sizing).
  sl_atr_mult / tp_atr_mult : multiples of ATR(14) for stop/target.

Patch 2E: removed the `min_confidence` knob. HC threshold (per-TF from
best_thresholds.json) is the single confidence gate — this tier field was
duplicating it and producing zero trades on Conservative/Balanced.
"""
from __future__ import annotations


TIERS: dict = {
    "conservative": {
        "min_vol_pct": {
            "5m":  0.008,
            "15m": 0.010,
            "1h":  0.015,
            "4h":  0.020,
            "1d":  0.025,
        },
        "max_trades_per_day": 3,
        "allowed_labels": ["trend_aligned"],
        "risk_per_trade_pct": 0.0025,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.5,
    },
    "balanced": {
        "min_vol_pct": {
            "5m":  0.005,
            "15m": 0.007,
            "1h":  0.010,
            "4h":  0.015,
            "1d":  0.020,
        },
        "max_trades_per_day": 7,
        "allowed_labels": ["trend_aligned", "mean_reversion"],
        "risk_per_trade_pct": 0.005,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.5,
    },
    "aggressive": {
        "min_vol_pct": {
            "5m":  0.0025,
            "15m": 0.004,
            "1h":  0.006,
            "4h":  0.010,
            "1d":  0.015,
        },
        "max_trades_per_day": 20,
        "allowed_labels": ["trend_aligned", "mean_reversion"],
        "risk_per_trade_pct": 0.01,
        "sl_atr_mult": 1.2,
        "tp_atr_mult": 2.0,
    },
}


# Reject reason codes — stay short and stable so the Go side can match 1:1.
# Patch 2E: removed "confidence" — HC threshold is the single confidence gate.
REJECT_REASONS = (
    "hc_threshold",        # proba did not clear HC band
    "vol_floor",           # atr_norm_14 below min_vol_pct for this TF
    "label_not_allowed",   # label (e.g. "random") is not in allowed_labels
    "rate_limit",          # already hit max_trades_per_day
)


def tier_names() -> list[str]:
    return list(TIERS.keys())


def get_tier(name: str) -> dict:
    if name not in TIERS:
        raise ValueError(f"unknown risk tier: {name!r}; available: {tier_names()}")
    return TIERS[name]
