"""
train.py — trains an XGB+LGBM+RF ensemble with walk-forward validation.

Usage:
  python train.py                          # all top-20 × (1h, 4h, 1d)
  python train.py --symbols BTCUSDT        # one pair
  python train.py --intervals 4h           # one timeframe
  python train.py --quick                  # 1 pair, 1 tf, reduced estimators

Output per (symbol, interval):
  models/{symbol}_{interval}_v{timestamp}.json    # ensemble weights + metadata
  models/{symbol}_{interval}_patterns.json        # pattern matcher index
  models/latest.json                               # pointer to best model per pair
  logs/{symbol}_{interval}_metrics.json           # walk-forward metrics
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from feature_engine import (
    FEATURE_NAMES, build_features,
    make_target, make_target_triple_barrier, make_target_triple_class,
    TRIPLE_CLASS_HOLD, TRIPLE_CLASS_LONG, TRIPLE_CLASS_SHORT, TRIPLE_CLASS_UNLABELED,
    _atr,
)
from data_fetcher import fetch_or_cache, TOP_PAIRS
from pattern_matcher import PatternIndex, save_index


ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


# Horizon in bars per interval — matches TradingStyle node ranges.
# Lower TFs predict shorter horizons (high-freq); higher TFs look further out.
# Kept for backward compatibility — new code should use TF_CONFIG below.
HORIZON_MAP = {
    "5m":  12,   # ~1 hour ahead (scalping)
    "15m": 16,   # ~4 hours ahead (day trading early)
    "1h":  24,   # ~1 day ahead (day trading / short swing)
    "4h":  18,   # ~3 days ahead (swing short)
    "1d":  10,   # ~10 days ahead (swing / position)
}


# Per-interval training overrides.
# 1d has only ~2400 train rows vs ~840k on 5m → needs shallower trees, fewer
# estimators to avoid overfit. Also longer horizons benefit from wider barriers.
TF_CONFIG = {
    "5m":  {"horizon": 12, "tb_upper": 2.5, "tb_lower": 1.5, "n_est": 200, "xgb_depth": 5},
    "15m": {"horizon": 16, "tb_upper": 2.5, "tb_lower": 1.5, "n_est": 200, "xgb_depth": 5},
    "1h":  {"horizon": 24, "tb_upper": 1.5, "tb_lower": 1.0, "n_est": 200, "xgb_depth": 5},
    "4h":  {"horizon": 18, "tb_upper": 1.5, "tb_lower": 1.0, "n_est": 200, "xgb_depth": 5},
    "1d":  {"horizon":  5, "tb_upper": 2.5, "tb_lower": 1.8, "n_est": 100, "xgb_depth": 3},
}


@dataclass
class FoldMetrics:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    sharpe: float
    # High-confidence stats: only signals where prob > 0.80 or < 0.20.
    # This filters the "loud" bars — much fewer trades, but higher precision.
    # Mirrors how quant funds actually trade: skip noise, hit conviction.
    hc_precision: float  # precision on filtered subset
    hc_count: int        # how many bars got a signal
    hc_win_rate: float   # P(correct direction | signal)


def compute_high_confidence_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                     threshold: float = 0.80) -> tuple[float, int, float]:
    """
    Subset: bars where |prob - 0.5| > (threshold - 0.5), i.e. model is very confident.
    Returns (precision, n_signals, win_rate).
    If no signals, returns (0.0, 0, 0.0).
    """
    high_conf_mask = (y_pred_proba > threshold) | (y_pred_proba < 1 - threshold)
    n = int(high_conf_mask.sum())
    if n == 0:
        return 0.0, 0, 0.0

    # Predictions on filtered subset.
    y_hc_true = y_true[high_conf_mask]
    y_hc_pred = (y_pred_proba[high_conf_mask] > 0.5).astype(int)

    # Precision = correct predictions / all predictions on filtered subset.
    correct = (y_hc_pred == y_hc_true).sum()
    precision = float(correct) / float(n)

    # Win rate: same as precision for binary classification with directional trading.
    win_rate = precision
    return precision, n, win_rate


def compute_hc_table(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Compute HC metrics at multiple thresholds. Reports both raw and quantile-based."""
    out = {}
    for t in (0.55, 0.60, 0.65, 0.70, 0.80):
        p, n, _ = compute_high_confidence_metrics(y_true, y_pred_proba, t)
        out[f"thr_{t:.2f}"] = {"precision": p, "n_signals": n, "fraction": n / max(len(y_true), 1)}
    # Quantile-based: top-10% most extreme predictions (always has signals).
    deviations = np.abs(y_pred_proba - 0.5)
    if len(deviations) > 10:
        cutoff = np.quantile(deviations, 0.90)
        mask = deviations >= cutoff
        n = int(mask.sum())
        if n > 0:
            y_hc_pred = (y_pred_proba[mask] > 0.5).astype(int)
            correct = (y_hc_pred == y_true[mask]).sum()
            out["top_10pct"] = {"precision": float(correct) / float(n), "n_signals": n, "fraction": n / len(y_true)}
    return out


def compute_sharpe(y_true: np.ndarray, y_pred_proba: np.ndarray,
                   returns: np.ndarray, interval: str = "1d") -> float:
    """
    Trading Sharpe: position = sign(pred - 0.5) × (|pred - 0.5| > 0.1 ? 1 : 0).
    Returns × position, annualized using TF-specific bars-per-year factor.
    Previously used sqrt(252) for all TFs which massively under-annualized 4h/1h
    and over-annualized 1d+. Now matches actual TF.
    """
    position = np.where(y_pred_proba > 0.6, 1.0, np.where(y_pred_proba < 0.4, -1.0, 0.0))
    strategy_ret = position * returns
    if strategy_ret.std() < 1e-9:
        return 0.0
    # Bars per calendar year per interval.
    bars_per_year = {
        "5m":  365 * 24 * 12,   # 105,120
        "15m": 365 * 24 * 4,    # 35,040
        "1h":  365 * 24,        # 8,760
        "4h":  365 * 6,         # 2,190
        "1d":  252,             # trading days
    }.get(interval, 252)
    return float(np.sqrt(bars_per_year) * strategy_ret.mean() / strategy_ret.std())


def walk_forward_split(dates: np.ndarray, train_months: int = 12, test_months: int = 1):
    """Yield (train_idx, test_idx) tuples via expanding window."""
    s = pd.to_datetime(dates)
    start = s.min()
    end = s.max()
    test_start = start + pd.DateOffset(months=train_months)
    fold = 0
    while test_start + pd.DateOffset(months=test_months) <= end:
        test_end = test_start + pd.DateOffset(months=test_months)
        train_mask = s < test_start
        test_mask = (s >= test_start) & (s < test_end)
        if train_mask.sum() > 100 and test_mask.sum() > 10:
            yield fold, np.where(train_mask)[0], np.where(test_mask)[0], \
                str(start.date()), str(test_start.date()), str(test_start.date()), str(test_end.date())
        fold += 1
        test_start = test_end


def train_ensemble(X_train: np.ndarray, y_train: np.ndarray, quick: bool = False,
                   tf_overrides: dict | None = None):
    """
    Train XGB + LGBM + RF + F1-weighted average meta.

    Previous implementation used a 3-way temporal "OOF" split that fed LogReg
    lookahead data: part B trained on [:split1]∪[split2:n] and predicted the
    middle — the model saw FUTURE data when predicting past segments. This
    caused the meta-learner to collapse toward 0.5 because the base probs
    were artificially confident on training data.

    Fix: proper forward-only OOF via TimeSeriesSplit, then weight base models
    by their individual OOF F1 (not a LogReg meta that L2-regularizes to mean).
    This preserves probability dynamic range so HC signals actually trigger.

    CPU: n_jobs capped at 4 to stay cool on passively-cooled MacBook Air
    (n_jobs=-1 spawns 10+ workers which triggers thermal throttling after
    ~5 min of sustained compute, dropping effective speed by 50%).
    Override via ML_N_JOBS env var for desktops with active cooling.

    tf_overrides: optional dict with per-TF hyperparams (n_est, xgb_depth).
    Used by train_one to scale model complexity for data-scarce intervals (1d).
    """
    import os
    from sklearn.model_selection import TimeSeriesSplit

    n_est = (tf_overrides or {}).get("n_est") or (50 if quick else 200)
    xgb_depth = (tf_overrides or {}).get("xgb_depth", 5)
    n_jobs = int(os.getenv("ML_N_JOBS", "4"))

    xgb = XGBClassifier(
        n_estimators=n_est, max_depth=xgb_depth, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", n_jobs=n_jobs, random_state=42,
        tree_method="hist", verbosity=0,
    )
    lgbm = LGBMClassifier(
        n_estimators=n_est, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=n_jobs, random_state=42, verbose=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=n_est, max_depth=10,
        n_jobs=n_jobs, random_state=42,
    )

    # Proper forward-only time-series OOF.
    tss = TimeSeriesSplit(n_splits=3)
    oof_xgb = np.zeros(len(X_train))
    oof_lgbm = np.zeros(len(X_train))
    oof_rf = np.zeros(len(X_train))
    mask_oof = np.zeros(len(X_train), dtype=bool)

    for tr_idx, te_idx in tss.split(X_train):
        xgb.fit(X_train[tr_idx], y_train[tr_idx])
        lgbm.fit(X_train[tr_idx], y_train[tr_idx])
        rf.fit(X_train[tr_idx], y_train[tr_idx])
        oof_xgb[te_idx] = xgb.predict_proba(X_train[te_idx])[:, 1]
        oof_lgbm[te_idx] = lgbm.predict_proba(X_train[te_idx])[:, 1]
        oof_rf[te_idx] = rf.predict_proba(X_train[te_idx])[:, 1]
        mask_oof[te_idx] = True

    # Weights = per-model F1 on OOF subset (masked to skip first fold).
    from sklearn.metrics import f1_score as _f1
    if mask_oof.sum() > 10:
        f1_xgb = _f1(y_train[mask_oof], (oof_xgb[mask_oof] > 0.5).astype(int), zero_division=0)
        f1_lgbm = _f1(y_train[mask_oof], (oof_lgbm[mask_oof] > 0.5).astype(int), zero_division=0)
        f1_rf = _f1(y_train[mask_oof], (oof_rf[mask_oof] > 0.5).astype(int), zero_division=0)
        total = max(f1_xgb + f1_lgbm + f1_rf, 1e-6)
        weights = np.array([f1_xgb / total, f1_lgbm / total, f1_rf / total])
    else:
        weights = np.array([1 / 3, 1 / 3, 1 / 3])

    # Final base models: train on full data.
    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # "Meta" is now a tiny LogReg wrapper that stores these weights as
    # coefficients so the Go inference path stays unchanged (predicts via
    # meta.coef_ × base_probs + intercept).
    meta = LogisticRegression(max_iter=1000, C=100.0)  # near-zero regularization
    # Seed it with our computed weights using a lightweight fit.
    # Input: synthetic points near corners of prob space to anchor weights.
    X_seed = np.array([
        [0.1, 0.1, 0.1],   # all-bearish → label 0
        [0.9, 0.9, 0.9],   # all-bullish → label 1
        [0.2, 0.3, 0.2],
        [0.8, 0.7, 0.8],
    ])
    y_seed = np.array([0, 1, 0, 1])
    meta.fit(X_seed, y_seed)
    # Overwrite with f1-proportional weights (keeps Go deserialization happy).
    # Scale by 4 so probs don't flatten through sigmoid.
    meta.coef_ = (weights.reshape(1, -1) * 4.0).astype(np.float64)
    meta.intercept_ = np.array([-2.0], dtype=np.float64)

    return xgb, lgbm, rf, meta


def ensemble_predict(xgb, lgbm, rf, meta, X):
    p = np.stack([
        xgb.predict_proba(X)[:, 1],
        lgbm.predict_proba(X)[:, 1],
        rf.predict_proba(X)[:, 1],
    ], axis=1)
    return meta.predict_proba(p)[:, 1], p


def _xgb_device_kwargs():
    """Patch 4 night work: detect CUDA for XGBoost 2.0+ device='cuda'.

    Falls back to CPU (empty dict) on any error. Force CPU via XGB_DEVICE=cpu.
    """
    import os
    if os.getenv("XGB_DEVICE", "").lower() == "cpu":
        return {}
    try:
        import xgboost as xgb
        ver_parts = xgb.__version__.split(".")
        if int(ver_parts[0]) < 2:
            return {}
        from xgboost import XGBClassifier
        probe = XGBClassifier(n_estimators=3, device="cuda", tree_method="hist",
                               objective="multi:softprob", num_class=3, verbosity=0)
        Xp = np.random.rand(60, 8).astype(np.float32)
        yp = np.random.randint(0, 3, 60)
        probe.fit(Xp, yp)
        return {"device": "cuda"}
    except Exception:
        return {}


def _lgbm_device_kwargs():
    """Probe LightGBM GPU. Fall back to CPU on failure."""
    import os
    if os.getenv("LGBM_DEVICE", "").lower() == "cpu":
        return {}
    try:
        from lightgbm import LGBMClassifier
        probe = LGBMClassifier(n_estimators=3, device="gpu", gpu_device_id=0,
                                objective="multiclass", num_class=3, verbose=-1)
        Xp = np.random.rand(60, 8).astype(np.float32)
        yp = np.random.randint(0, 3, 60)
        probe.fit(Xp, yp)
        return {"device": "gpu", "gpu_device_id": 0}
    except Exception:
        return {}


_XGB_DEVICE_KW = None
_LGBM_DEVICE_KW = None


def _get_xgb_device():
    global _XGB_DEVICE_KW
    if _XGB_DEVICE_KW is None:
        _XGB_DEVICE_KW = _xgb_device_kwargs()
        print(f"  [gpu_probe] xgb device kwargs: {_XGB_DEVICE_KW or 'CPU'}")
    return _XGB_DEVICE_KW


def _get_lgbm_device():
    global _LGBM_DEVICE_KW
    if _LGBM_DEVICE_KW is None:
        _LGBM_DEVICE_KW = _lgbm_device_kwargs()
        print(f"  [gpu_probe] lgbm device kwargs: {_LGBM_DEVICE_KW or 'CPU'}")
    return _LGBM_DEVICE_KW


def train_ensemble_3class(X_train: np.ndarray, y_train: np.ndarray, quick: bool = False,
                          tf_overrides: dict | None = None):
    """
    Patch 4 Stage 1 — 3-class ensemble (hold/long/short).

    Parallel implementation of `train_ensemble` for XGBoost `multi:softprob`,
    LightGBM `multiclass`, RF multi-class. Meta is a dict of per-class weights
    instead of a LogReg wrapper because there's no clean way to wrap a 3x3
    weight matrix through sklearn's LogReg interface without breaking Go
    deserialization.

    GPU: probes CUDA/GPU once per process (cached). XGB 2.0+ uses device='cuda',
    LightGBM uses device='gpu'. Falls back to CPU silently. Force CPU via
    XGB_DEVICE=cpu or LGBM_DEVICE=cpu env vars.

    Returns:
        xgb, lgbm, rf: fitted 3-class classifiers
        meta: dict with 'class_weights': shape (3 bases, 3 classes) float array
              and 'class_order': [hold, long, short] encoded as [0, 1, 2]

    y_train values must be in {0, 1, 2}. Use make_target_triple_class() labels
    and drop -1 sentinel via `target >= 0` mask before calling.
    """
    import os
    from sklearn.model_selection import TimeSeriesSplit

    n_est = (tf_overrides or {}).get("n_est") or (50 if quick else 200)
    xgb_depth = (tf_overrides or {}).get("xgb_depth", 5)
    n_jobs = int(os.getenv("ML_N_JOBS", "4"))

    xgb_kw = _get_xgb_device()
    lgbm_kw = _get_lgbm_device()

    xgb = XGBClassifier(
        n_estimators=n_est, max_depth=xgb_depth, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", n_jobs=n_jobs, random_state=42,
        tree_method="hist", verbosity=0,
        **xgb_kw,
    )
    lgbm = LGBMClassifier(
        n_estimators=n_est, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multiclass", num_class=3,
        n_jobs=n_jobs, random_state=42, verbose=-1,
        **lgbm_kw,
    )
    rf = RandomForestClassifier(
        n_estimators=n_est, max_depth=10,
        n_jobs=n_jobs, random_state=42,
    )

    # Walk-forward OOF F1 per class per base — used to weight each base's
    # contribution to each class independently. E.g. if XGB is great at
    # predicting 'long' but bad at 'short', it gets high weight on the long
    # column and low weight on the short column.
    tss = TimeSeriesSplit(n_splits=3)
    n = len(X_train)
    oof_xgb = np.full((n, 3), 1 / 3.0)
    oof_lgbm = np.full((n, 3), 1 / 3.0)
    oof_rf = np.full((n, 3), 1 / 3.0)
    mask_oof = np.zeros(n, dtype=bool)

    for tr_idx, te_idx in tss.split(X_train):
        # Skip folds where either split lacks all 3 classes (rare but breaks xgb).
        tr_classes = np.unique(y_train[tr_idx])
        if len(tr_classes) < 3:
            continue
        xgb.fit(X_train[tr_idx], y_train[tr_idx])
        lgbm.fit(X_train[tr_idx], y_train[tr_idx])
        rf.fit(X_train[tr_idx], y_train[tr_idx])
        oof_xgb[te_idx] = xgb.predict_proba(X_train[te_idx])
        oof_lgbm[te_idx] = lgbm.predict_proba(X_train[te_idx])
        oof_rf[te_idx] = rf.predict_proba(X_train[te_idx])
        mask_oof[te_idx] = True

    # Per-class F1 for each base: argmax prediction vs true label.
    from sklearn.metrics import f1_score as _f1
    class_weights = np.ones((3, 3)) / 3.0  # default equal if OOF empty
    if mask_oof.sum() > 30:
        y_oof = y_train[mask_oof]
        preds = {
            "xgb": np.argmax(oof_xgb[mask_oof], axis=1),
            "lgbm": np.argmax(oof_lgbm[mask_oof], axis=1),
            "rf": np.argmax(oof_rf[mask_oof], axis=1),
        }
        f1_per_class = {}  # base -> [f1_hold, f1_long, f1_short]
        for base, pred in preds.items():
            f1s = _f1(y_oof, pred, labels=[0, 1, 2], average=None, zero_division=0)
            f1_per_class[base] = f1s
        # Normalize columns (per class) so weights sum to 1 across bases.
        raw = np.stack([f1_per_class["xgb"], f1_per_class["lgbm"], f1_per_class["rf"]])
        # raw shape: (3 bases, 3 classes). Add epsilon to avoid div-by-0.
        col_sums = raw.sum(axis=0) + 1e-6
        class_weights = raw / col_sums  # shape (3 bases, 3 classes)

    # Final models on full data.
    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    meta = {
        "kind": "soft_voting_3class",
        "class_weights": class_weights.tolist(),  # (bases, classes)
        "class_order": [0, 1, 2],  # TRIPLE_CLASS_HOLD, LONG, SHORT
    }
    return xgb, lgbm, rf, meta


def ensemble_predict_3class(xgb, lgbm, rf, meta, X):
    """
    Returns (proba, base_probs) where:
        proba: shape (n, 3) — weighted ensemble probability per class
        base_probs: shape (n, 3 bases, 3 classes) — for inspection/debug
    """
    p_xgb = xgb.predict_proba(X)    # (n, 3)
    p_lgbm = lgbm.predict_proba(X)
    p_rf = rf.predict_proba(X)
    base_probs = np.stack([p_xgb, p_lgbm, p_rf], axis=1)  # (n, 3 bases, 3 classes)
    w = np.asarray(meta["class_weights"])  # (3 bases, 3 classes)
    # Weighted sum over bases per class: out[i, c] = Σ_b base_probs[i, b, c] × w[b, c]
    weighted = base_probs * w[np.newaxis, :, :]  # broadcast (1, 3, 3)
    proba = weighted.sum(axis=1)  # (n, 3)
    # Renormalize so rows sum to 1 (weights don't automatically sum to 1 across bases
    # since f1-weighted, plus rounding noise).
    row_sums = proba.sum(axis=1, keepdims=True) + 1e-12
    proba = proba / row_sums
    return proba, base_probs


def serialize_model(xgb, lgbm, rf, meta, feature_cols, symbol, interval, horizon,
                    metrics: List[FoldMetrics], feature_importance: dict) -> dict:
    """Export model as JSON-serializable dict for Go consumption."""
    # Base model outputs are needed for Go inference. We export predictions
    # on a held-out calibration set + meta weights, not tree internals.
    # This is the "stacked probability" export format.
    return {
        "version": "ensemble_v2",
        "symbol": symbol,
        "interval": interval,
        "horizon": horizon,
        "feature_cols": feature_cols,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "meta_weights": meta.coef_[0].tolist(),
        "meta_intercept": float(meta.intercept_[0]),
        "xgb_model": xgb.get_booster().save_raw(raw_format="ubj").hex(),
        "lgbm_model": lgbm.booster_.model_to_string(),
        "rf_trees": _serialize_rf(rf),
        "metrics": _summarize_metrics(metrics),
        "feature_importance": feature_importance,
    }


def _summarize_metrics(metrics: List[FoldMetrics]) -> dict:
    """Aggregate fold metrics + compute high-confidence stats + regime-aware metrics."""
    if not metrics:
        return {
            "folds": [],
            "avg_accuracy": 0.0, "avg_sharpe": 0.0, "avg_f1": 0.0,
            "avg_precision": 0.0, "avg_recall": 0.0,
            "hc_precision": 0.0, "hc_signals_total": 0, "hc_signal_rate": 0.0,
            "n_folds": 0,
        }
    total_hc = sum(m.hc_count for m in metrics)
    total_test = sum(m.n_test for m in metrics)
    hc_precision_weighted = (
        sum(m.hc_precision * m.hc_count for m in metrics) / total_hc
        if total_hc > 0 else 0.0
    )
    return {
        "folds": [asdict(m) for m in metrics],
        "avg_accuracy": float(np.mean([m.accuracy for m in metrics])),
        "avg_sharpe": float(np.mean([m.sharpe for m in metrics])),
        "avg_f1": float(np.mean([m.f1 for m in metrics])),
        "avg_precision": float(np.mean([m.precision for m in metrics])),
        "avg_recall": float(np.mean([m.recall for m in metrics])),
        "hc_precision": hc_precision_weighted,
        "hc_signals_total": total_hc,
        "hc_signal_rate": total_hc / max(total_test, 1),
        "n_folds": len(metrics),
        "n_test_total": total_test,
    }


def _serialize_rf(rf: RandomForestClassifier) -> list:
    """Export each tree as a minimal structure for Go inference."""
    trees = []
    for est in rf.estimators_:
        t = est.tree_
        trees.append({
            "children_left": t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "feature": t.feature.tolist(),
            "threshold": t.threshold.tolist(),
            "value": [v[0].tolist() for v in t.value],  # class probabilities
        })
    return trees


# ═══════════════════════════════════════════════════════════
# Patch 4 Stage 1 — 3-class (hold/long/short) training path
# ═══════════════════════════════════════════════════════════


@dataclass
class FoldMetrics3Class:
    """Per-fold metrics for 3-class training. Compared to FoldMetrics (binary):
      - precision/recall/f1 are macro-averaged over 3 classes
      - HC stats are split: hc_long_* for predict-long signals,
        hc_short_* for predict-short signals. 'hold' class doesn't
        produce trades.
    """
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    accuracy: float
    f1_macro: float
    sharpe: float
    # Per-direction HC metrics.
    hc_long_count: int
    hc_long_precision: float
    hc_short_count: int
    hc_short_precision: float
    # Class distribution in test set (sanity check).
    class_dist_hold: float
    class_dist_long: float
    class_dist_short: float


def compute_hc_3class(
    y_true: np.ndarray,
    proba_3class: np.ndarray,
    hc_long: float = 0.60,
    hc_short: float = 0.60,
) -> tuple[int, float, int, float]:
    """
    Patch 4 Stage 1 HC metrics for 3-class proba.

    Inputs:
        y_true: shape (n,) with values in {0=hold, 1=long, 2=short}
        proba_3class: shape (n, 3), columns = [p_hold, p_long, p_short]
        hc_long, hc_short: thresholds. A bar is a "long signal" if
            proba_3class[:, 1] > hc_long (regardless of p_hold/p_short).
            Same for short with proba_3class[:, 2] > hc_short.

    Returns: (n_long_signals, precision_long, n_short_signals, precision_short)
        where precision_long = fraction of long signals where y_true == long.
    """
    p_long = proba_3class[:, 1]
    p_short = proba_3class[:, 2]

    long_mask = p_long > hc_long
    short_mask = p_short > hc_short

    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())

    if n_long > 0:
        prec_long = float((y_true[long_mask] == 1).sum()) / float(n_long)
    else:
        prec_long = 0.0

    if n_short > 0:
        prec_short = float((y_true[short_mask] == 2).sum()) / float(n_short)
    else:
        prec_short = 0.0

    return n_long, prec_long, n_short, prec_short


def compute_sharpe_3class(
    proba_3class: np.ndarray,
    returns: np.ndarray,
    interval: str,
    hc_long: float = 0.60,
    hc_short: float = 0.60,
) -> float:
    """
    3-class Sharpe: position = +1 on long signal, -1 on short signal, 0 elsewhere.

    Cleaner than binary version — we're no longer sign-flipping based on
    p(long) > 0.5 which was the symptom of 100% short bias when p(long)
    distribution shifted below 0.5 globally. Now long and short have
    explicit, independent entry criteria.
    """
    p_long = proba_3class[:, 1]
    p_short = proba_3class[:, 2]
    pos = np.where(p_long > hc_long, 1.0, np.where(p_short > hc_short, -1.0, 0.0))
    strategy_ret = pos * returns
    if strategy_ret.std() < 1e-9:
        return 0.0
    bars_per_year = {
        "5m": 365 * 24 * 12, "15m": 365 * 24 * 4, "1h": 365 * 24,
        "4h": 365 * 6, "1d": 252,
    }.get(interval, 252)
    return float(np.sqrt(bars_per_year) * strategy_ret.mean() / strategy_ret.std())


def serialize_model_3class(xgb, lgbm, rf, meta, feature_cols, symbol, interval, horizon,
                            metrics: List[FoldMetrics3Class],
                            feature_importance: dict) -> dict:
    """
    Patch 4 Stage 1 serialization format. Distinct 'version' key so Go
    inference can branch on it.

    meta is a dict {kind, class_weights, class_order} from train_ensemble_3class.
    """
    return {
        "version": "ensemble_v2_3class",
        "symbol": symbol,
        "interval": interval,
        "horizon": horizon,
        "feature_cols": feature_cols,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,  # dict with class_weights (3 bases × 3 classes) + class_order
        "xgb_model": xgb.get_booster().save_raw(raw_format="ubj").hex(),
        "lgbm_model": lgbm.booster_.model_to_string(),
        "rf_trees": _serialize_rf(rf),
        "metrics": _summarize_metrics_3class(metrics),
        "feature_importance": feature_importance,
        "class_encoding": {
            "hold": TRIPLE_CLASS_HOLD,
            "long": TRIPLE_CLASS_LONG,
            "short": TRIPLE_CLASS_SHORT,
        },
    }


def _summarize_metrics_3class(metrics: List[FoldMetrics3Class]) -> dict:
    if not metrics:
        return {
            "folds": [], "n_folds": 0,
            "avg_accuracy": 0.0, "avg_f1_macro": 0.0, "avg_sharpe": 0.0,
            "hc_long_precision": 0.0, "hc_long_total": 0,
            "hc_short_precision": 0.0, "hc_short_total": 0,
        }
    total_long = sum(m.hc_long_count for m in metrics)
    total_short = sum(m.hc_short_count for m in metrics)
    prec_long_w = (
        sum(m.hc_long_precision * m.hc_long_count for m in metrics) / total_long
        if total_long > 0 else 0.0
    )
    prec_short_w = (
        sum(m.hc_short_precision * m.hc_short_count for m in metrics) / total_short
        if total_short > 0 else 0.0
    )
    return {
        "folds": [asdict(m) for m in metrics],
        "n_folds": len(metrics),
        "avg_accuracy": float(np.mean([m.accuracy for m in metrics])),
        "avg_f1_macro": float(np.mean([m.f1_macro for m in metrics])),
        "avg_sharpe": float(np.mean([m.sharpe for m in metrics])),
        "hc_long_precision": prec_long_w,
        "hc_long_total": total_long,
        "hc_short_precision": prec_short_w,
        "hc_short_total": total_short,
    }


def train_one_3class(symbol: str, interval: str, years: float, quick: bool,
                     btc_close: np.ndarray | None,
                     tb_upper: float = -1.0, tb_lower: float = -1.0):
    """
    Patch 4 Stage 1 — train one symbol/interval with the 3-class target.

    Sibling of `train_one` kept separate so the binary path stays untouched
    (backtest_v2, Patch 2H artefacts all rely on the binary ensemble shape).
    """
    t0 = time.time()
    tf_cfg = TF_CONFIG.get(interval, {
        "horizon": HORIZON_MAP.get(interval, 10),
        "tb_upper": 1.5, "tb_lower": 1.0,
        "n_est": None, "xgb_depth": 5,
    })
    horizon = tf_cfg["horizon"]
    eff_tb_upper = tf_cfg["tb_upper"] if tb_upper < 0 else tb_upper
    eff_tb_lower = tf_cfg["tb_lower"] if tb_lower < 0 else tb_lower

    print(f"\n━━━ {symbol} {interval} (horizon={horizon}, target=tb_3class, "
          f"upper={eff_tb_upper}, lower={eff_tb_lower}) ━━━")

    df = fetch_or_cache(symbol, interval, years=years)
    if len(df) < 500:
        print(f"  skip: only {len(df)} candles")
        return None
    print(f"  candles: {len(df):,} ({df['open_time'].min()} → {df['open_time'].max()})")

    feat = build_features(df, btc_close=btc_close)
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    close_arr = df["close"].to_numpy()
    atr_arr = _atr(high_arr, low_arr, close_arr, period=14)
    target = make_target_triple_class(
        high_arr, low_arr, close_arr, atr_arr, horizon=horizon,
        upper_mult=eff_tb_upper, lower_mult=eff_tb_lower,
    )

    mask = target >= 0
    feat = feat.loc[mask].reset_index(drop=True)
    target = target[mask]
    hold_rate = float((target == TRIPLE_CLASS_HOLD).mean())
    long_rate = float((target == TRIPLE_CLASS_LONG).mean())
    short_rate = float((target == TRIPLE_CLASS_SHORT).mean())
    print(f"  labeled: {len(feat):,} rows — "
          f"hold={hold_rate:.1%}, long={long_rate:.1%}, short={short_rate:.1%}")

    if long_rate < 0.05 or short_rate < 0.05:
        print(f"  WARN: class imbalance — model may collapse to majority class")

    X_all = feat[FEATURE_NAMES].to_numpy()
    dates = feat["open_time"].to_numpy()
    closes = feat["close"].to_numpy()

    # Walk-forward CV windows identical to binary path.
    if quick:
        train_months, test_months = 4, 1
    elif interval in ("5m", "15m"):
        train_months, test_months = 6, 1
    else:
        train_months, test_months = 24, 3

    metrics_3c: List[FoldMetrics3Class] = []
    from sklearn.metrics import f1_score

    for fold, train_idx, test_idx, t_start, t_end, v_start, v_end in walk_forward_split(
        dates, train_months=train_months, test_months=test_months
    ):
        if quick and fold >= 3:
            break
        X_tr, y_tr = X_all[train_idx], target[train_idx]
        X_te, y_te = X_all[test_idx], target[test_idx]
        if len(np.unique(y_tr)) < 3:
            print(f"  fold {fold:2d}: skip — train set missing a class")
            continue

        close_te = closes[test_idx]
        returns_te = np.diff(close_te, prepend=close_te[0]) / (np.roll(close_te, 1) + 1e-12)
        returns_te[0] = 0

        xgb, lgbm, rf, meta = train_ensemble_3class(
            X_tr, y_tr, quick=quick,
            tf_overrides={"n_est": tf_cfg["n_est"], "xgb_depth": tf_cfg["xgb_depth"]},
        )
        proba_3c, _ = ensemble_predict_3class(xgb, lgbm, rf, meta, X_te)
        y_pred = np.argmax(proba_3c, axis=1)

        acc = float(accuracy_score(y_te, y_pred))
        f1_m = float(f1_score(y_te, y_pred, labels=[0, 1, 2], average="macro", zero_division=0))
        sharpe = compute_sharpe_3class(proba_3c, returns_te, interval=interval)
        n_l, prec_l, n_s, prec_s = compute_hc_3class(y_te, proba_3c, hc_long=0.60, hc_short=0.60)

        m = FoldMetrics3Class(
            fold=fold,
            train_start=t_start, train_end=t_end,
            test_start=v_start, test_end=v_end,
            n_train=len(train_idx), n_test=len(test_idx),
            accuracy=acc, f1_macro=f1_m, sharpe=sharpe,
            hc_long_count=n_l, hc_long_precision=prec_l,
            hc_short_count=n_s, hc_short_precision=prec_s,
            class_dist_hold=float((y_te == 0).mean()),
            class_dist_long=float((y_te == 1).mean()),
            class_dist_short=float((y_te == 2).mean()),
        )
        metrics_3c.append(m)
        print(f"  fold {fold:2d}: n_test={m.n_test:4d} acc={acc:.3f} f1_m={f1_m:.3f} "
              f"sharpe={sharpe:+.2f} HC_long={n_l}@{prec_l:.1%} HC_short={n_s}@{prec_s:.1%}")

    if not metrics_3c:
        print("  no folds completed (not enough data)")
        return None

    avg_acc = np.mean([m.accuracy for m in metrics_3c])
    avg_f1m = np.mean([m.f1_macro for m in metrics_3c])
    avg_sharpe = np.mean([m.sharpe for m in metrics_3c])
    t_long = sum(m.hc_long_count for m in metrics_3c)
    t_short = sum(m.hc_short_count for m in metrics_3c)
    hc_long_wavg = (
        sum(m.hc_long_precision * m.hc_long_count for m in metrics_3c) / t_long
        if t_long > 0 else 0.0
    )
    hc_short_wavg = (
        sum(m.hc_short_precision * m.hc_short_count for m in metrics_3c) / t_short
        if t_short > 0 else 0.0
    )
    print(f"  → avg acc={avg_acc:.3f} f1_macro={avg_f1m:.3f} sharpe={avg_sharpe:+.2f}")
    print(f"    HC_long {t_long} signals at {hc_long_wavg:.1%} precision")
    print(f"    HC_short {t_short} signals at {hc_short_wavg:.1%} precision")

    # Final model on full data.
    print("  training final 3-class model on full data...")
    xgb, lgbm, rf, meta = train_ensemble_3class(
        X_all, target, quick=quick,
        tf_overrides={"n_est": tf_cfg["n_est"], "xgb_depth": tf_cfg["xgb_depth"]},
    )

    imp_xgb = xgb.feature_importances_
    imp_lgbm = lgbm.feature_importances_ / max(lgbm.feature_importances_.sum(), 1)
    imp_xgb = imp_xgb / max(imp_xgb.sum(), 1)
    avg_imp = (imp_xgb + imp_lgbm) / 2
    fi = {FEATURE_NAMES[i]: float(avg_imp[i]) for i in range(len(FEATURE_NAMES))}

    ts = int(time.time())
    model_payload = serialize_model_3class(
        xgb, lgbm, rf, meta, FEATURE_NAMES,
        symbol, interval, horizon, metrics_3c, fi,
    )
    model_path = MODELS_DIR / f"{symbol}_{interval}_v{ts}_3class.json"
    with open(model_path, "w") as f:
        json.dump(model_payload, f)
    print(f"  saved: {model_path.name} ({model_path.stat().st_size / 1024:.1f} KB)")

    # Metrics log.
    log = {
        "symbol": symbol, "interval": interval, "horizon": horizon,
        "target_mode": "tb_3class",
        "tb_upper": eff_tb_upper, "tb_lower": eff_tb_lower,
        "duration_sec": time.time() - t0,
        "class_distribution": {"hold": hold_rate, "long": long_rate, "short": short_rate},
        "avg_accuracy": float(avg_acc),
        "avg_f1_macro": float(avg_f1m),
        "avg_sharpe": float(avg_sharpe),
        "hc_long_precision": float(hc_long_wavg),
        "hc_long_total": int(t_long),
        "hc_short_precision": float(hc_short_wavg),
        "hc_short_total": int(t_short),
        "n_folds": len(metrics_3c),
        "folds": [asdict(m) for m in metrics_3c],
    }
    with open(LOGS_DIR / f"{symbol}_{interval}_metrics_3class.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"  ⏱  {time.time() - t0:.1f}s")
    return model_path


def train_one(symbol: str, interval: str, years: float, quick: bool, btc_close: np.ndarray | None,
              target_mode: str = "binary", tb_upper: float = -1.0, tb_lower: float = -1.0):
    t0 = time.time()
    # Pull per-TF config; fall back to HORIZON_MAP for unknown intervals.
    tf_cfg = TF_CONFIG.get(interval, {
        "horizon": HORIZON_MAP.get(interval, 10),
        "tb_upper": 1.5, "tb_lower": 1.0,
        "n_est": None, "xgb_depth": 5,
    })
    horizon = tf_cfg["horizon"]

    # Sentinel -1 → user didn't pass explicit barriers; pull TF defaults.
    # Lets Patch 2 tune 1d with wider barriers (2.5/1.8) while 1h stays at 1.5/1.0.
    eff_tb_upper = tf_cfg["tb_upper"] if tb_upper < 0 else tb_upper
    eff_tb_lower = tf_cfg["tb_lower"] if tb_lower < 0 else tb_lower

    print(f"\n━━━ {symbol} {interval} (horizon={horizon}, target={target_mode}) ━━━")

    # 1. Fetch data.
    df = fetch_or_cache(symbol, interval, years=years)
    if len(df) < 500:
        print(f"  skip: only {len(df)} candles")
        return None
    print(f"  candles: {len(df):,} ({df['open_time'].min()} → {df['open_time'].max()})")

    # 2. Build features.
    feat = build_features(df, btc_close=btc_close)
    close_arr = df["close"].to_numpy()
    if target_mode == "tb_atr":
        high_arr = df["high"].to_numpy()
        low_arr = df["low"].to_numpy()
        atr_arr = _atr(high_arr, low_arr, close_arr, period=14)
        target = make_target_triple_barrier(
            high_arr, low_arr, close_arr, atr_arr, horizon=horizon,
            upper_mult=eff_tb_upper, lower_mult=eff_tb_lower,
        )
        n_timeout = int((target == -1).sum())
        print(f"  tb_atr: upper={eff_tb_upper} lower={eff_tb_lower} "
              f"timeout/noise dropped = {n_timeout:,}/{len(target):,} "
              f"({n_timeout/max(len(target),1):.1%})")
    else:
        target = make_target(close_arr, horizon=horizon, threshold=0.0)

    # Drop rows with sentinel target (-1). For tb_atr this also drops chop bars.
    mask = target >= 0
    feat = feat.loc[mask].reset_index(drop=True)
    target = target[mask]
    print(f"  labeled: {len(feat):,} rows, positive rate: {target.mean():.3f}")

    X_all = feat[FEATURE_NAMES].to_numpy()
    dates = feat["open_time"].to_numpy()
    closes = feat["close"].to_numpy()

    # 3. Walk-forward CV.
    # TF-specific training window: high-freq data needs shorter train_months
    # or we get 0 folds (e.g. 5m with 2 years and train_months=24 → data
    # ends before first fold can start). Quick mode always uses 4mo/1mo.
    if quick:
        train_months, test_months = 4, 1
    elif interval in ("5m", "15m"):
        train_months, test_months = 6, 1   # 5m/15m: short windows = more folds
    else:
        train_months, test_months = 24, 3  # 1h/4h/1d: quarterly windows
    metrics: List[FoldMetrics] = []
    all_test_proba = []
    all_test_y = []

    for fold, train_idx, test_idx, t_start, t_end, v_start, v_end in walk_forward_split(
        dates, train_months=train_months, test_months=test_months
    ):
        if quick and fold >= 3:
            break
        X_tr, y_tr = X_all[train_idx], target[train_idx]
        X_te, y_te = X_all[test_idx], target[test_idx]
        close_te = closes[test_idx]
        returns_te = np.diff(close_te, prepend=close_te[0]) / (np.roll(close_te, 1) + 1e-12)
        returns_te[0] = 0

        xgb, lgbm, rf, meta = train_ensemble(
            X_tr, y_tr, quick=quick,
            tf_overrides={"n_est": tf_cfg["n_est"], "xgb_depth": tf_cfg["xgb_depth"]},
        )
        y_proba, _ = ensemble_predict(xgb, lgbm, rf, meta, X_te)
        y_pred = (y_proba > 0.5).astype(int)

        hc_prec, hc_n, hc_win = compute_high_confidence_metrics(y_te, y_proba, 0.80)
        m = FoldMetrics(
            fold=fold,
            train_start=t_start, train_end=t_end,
            test_start=v_start, test_end=v_end,
            n_train=len(train_idx), n_test=len(test_idx),
            accuracy=float(accuracy_score(y_te, y_pred)),
            precision=float(precision_score(y_te, y_pred, zero_division=0)),
            recall=float(recall_score(y_te, y_pred, zero_division=0)),
            f1=float(f1_score(y_te, y_pred, zero_division=0)),
            sharpe=compute_sharpe(y_te, y_proba, returns_te, interval=interval),
            hc_precision=hc_prec,
            hc_count=hc_n,
            hc_win_rate=hc_win,
        )
        metrics.append(m)
        all_test_proba.append(y_proba)
        all_test_y.append(y_te)
        print(f"  fold {fold:2d}: n_test={m.n_test:4d} acc={m.accuracy:.3f} f1={m.f1:.3f} "
              f"sharpe={m.sharpe:+.2f} HC:{m.hc_count:3d}trades,prec={m.hc_precision:.1%}")

    if not metrics:
        print("  no folds completed (not enough data)")
        return None

    avg_acc = np.mean([m.accuracy for m in metrics])
    avg_f1 = np.mean([m.f1 for m in metrics])
    avg_sharpe = np.mean([m.sharpe for m in metrics])
    # Aggregate high-confidence precision: weighted by signal count (not simple avg).
    total_hc = sum(m.hc_count for m in metrics)
    total_test = sum(m.n_test for m in metrics)
    if total_hc > 0:
        avg_hc_prec = sum(m.hc_precision * m.hc_count for m in metrics) / total_hc
    else:
        avg_hc_prec = 0.0
    print(f"  → avg acc={avg_acc:.3f} f1={avg_f1:.3f} sharpe={avg_sharpe:+.2f} "
          f"HC_precision={avg_hc_prec:.1%} on {total_hc} signals ({total_hc/max(total_test,1):.1%} of bars)")

    # 4. Final training on all data for deployment model.
    print("  training final model on full data...")
    xgb, lgbm, rf, meta = train_ensemble(
        X_all, target, quick=quick,
        tf_overrides={"n_est": tf_cfg["n_est"], "xgb_depth": tf_cfg["xgb_depth"]},
    )

    # Feature importance = avg of XGB + LGBM gain importances.
    imp_xgb = xgb.feature_importances_
    imp_lgbm = lgbm.feature_importances_ / max(lgbm.feature_importances_.sum(), 1)
    imp_xgb = imp_xgb / max(imp_xgb.sum(), 1)
    avg_imp = (imp_xgb + imp_lgbm) / 2
    fi = {FEATURE_NAMES[i]: float(avg_imp[i]) for i in range(len(FEATURE_NAMES))}

    # 5. Serialize.
    ts = int(time.time())
    model_payload = serialize_model(xgb, lgbm, rf, meta, FEATURE_NAMES, symbol, interval, horizon, metrics, fi)
    model_path = MODELS_DIR / f"{symbol}_{interval}_v{ts}.json"
    with open(model_path, "w") as f:
        json.dump(model_payload, f)
    print(f"  saved: {model_path.name} ({model_path.stat().st_size / 1024:.1f} KB)")

    # 6. Pattern matcher index.
    pattern = PatternIndex(FEATURE_NAMES)
    pattern.fit(
        feat.iloc[: -max(5, horizon)],   # drop last rows without full outcomes
        closes[: -max(5, horizon)],
        [pd.Timestamp(ts).isoformat() for ts in dates[: -max(5, horizon)]],
    )
    pattern_path = MODELS_DIR / f"{symbol}_{interval}_patterns.json"
    save_index(pattern, pattern_path)
    print(f"  pattern index: {pattern_path.name} ({pattern_path.stat().st_size / 1024:.1f} KB)")

    # 7. Logs.
    log = {
        "symbol": symbol, "interval": interval, "horizon": horizon,
        "duration_sec": time.time() - t0,
        "avg_accuracy": avg_acc, "avg_f1": avg_f1, "avg_sharpe": avg_sharpe,
        "n_folds": len(metrics),
        "folds": [asdict(m) for m in metrics],
    }
    with open(LOGS_DIR / f"{symbol}_{interval}_metrics.json", "w") as f:
        json.dump(log, f, indent=2)

    # Update latest pointer.
    latest_path = MODELS_DIR / "latest.json"
    if latest_path.exists():
        with open(latest_path) as f:
            latest = json.load(f)
    else:
        latest = {}
    key = f"{symbol}_{interval}"
    latest[key] = {
        "model": model_path.name,
        "patterns": pattern_path.name,
        "horizon": horizon,
        "accuracy": avg_acc,
        "sharpe": avg_sharpe,
    }
    with open(latest_path, "w") as f:
        json.dump(latest, f, indent=2)

    print(f"  ⏱  {time.time() - t0:.1f}s")
    return model_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None, help="Defaults to top-20")
    ap.add_argument("--intervals", nargs="*", default=["4h"], choices=["5m", "15m", "1h", "4h", "1d"])
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--quick", action="store_true", help="50 estimators, 3 folds, for iteration")
    ap.add_argument("--target-mode", choices=["binary", "tb_atr", "tb_3class"], default="binary",
                    help="binary = up/down @ horizon; tb_atr = triple-barrier ATR (drops chop); "
                         "tb_3class = Patch 4 3-class {hold, long, short} for multi:softprob")
    ap.add_argument("--tb-upper", type=float, default=-1.0,
                    help="Upper barrier in ATR multiples (-1 = use TF_CONFIG default)")
    ap.add_argument("--tb-lower", type=float, default=-1.0,
                    help="Lower barrier in ATR multiples (-1 = use TF_CONFIG default)")
    args = ap.parse_args()

    symbols = args.symbols or TOP_PAIRS

    # Pre-load BTC close for cross-asset features.
    btc_close_by_iv = {}
    for iv in args.intervals:
        btc = fetch_or_cache("BTCUSDT", iv, years=args.years)
        btc_close_by_iv[iv] = btc["close"].to_numpy()

    t0 = time.time()
    trained: List[str] = []
    for sym in symbols:
        for iv in args.intervals:
            try:
                btc_close = btc_close_by_iv.get(iv) if sym != "BTCUSDT" else None
                if args.target_mode == "tb_3class":
                    mp = train_one_3class(sym, iv, args.years, args.quick, btc_close,
                                           tb_upper=args.tb_upper, tb_lower=args.tb_lower)
                else:
                    mp = train_one(sym, iv, args.years, args.quick, btc_close,
                                   target_mode=args.target_mode,
                                   tb_upper=args.tb_upper, tb_lower=args.tb_lower)
                if mp:
                    trained.append(mp.name)
            except Exception as e:
                print(f"  {sym} {iv} FAILED: {e}")

    print(f"\n═══ done: {len(trained)} models in {time.time() - t0:.1f}s ═══")
    for t in trained:
        print(f"  {t}")


if __name__ == "__main__":
    main()
