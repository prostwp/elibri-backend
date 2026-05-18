"""train_autogluon.py — AutoGluon TabularPredictor for the 3-class
{hold, long, short} crypto target.

Why AutoGluon (vs the existing XGB+LGBM+RF stack in train.py):
    - AutoGluon trains and stacks ~10 model families in one run (XGBoost,
      LightGBM, CatBoost, KNN, ExtraTrees, RandomForest, XT, Linear, NN_TORCH,
      WeightedEnsemble), each with auto-tuned hyperparameters. Patch 4 Stage 1
      already proved that just XGB+LGBM+RF doesn't extract enough signal —
      WFCV avg Sharpe -1.07 across 5 folds.
    - Probabilistic 3-class output is first-class (problem_type="multiclass").
    - Ensemble weighting is automatic and validation-based, removing the
      hand-rolled F1-weight code that drifts as features change.

This module is a *drop-in alternative* to train.train_one_3class — same input
(symbol, interval, feature DataFrame) and same output kind (a per-(symbol,
interval) JSON pointer in models/latest_autogluon.json plus an AutoGluon
predictor directory at models/ag_{symbol}_{interval}/).

Run on vast.ai:
    python train_autogluon.py --symbol BTCUSDT --interval 4h --with-macro \
        --presets medium_quality --time-limit 1800

`time-limit` is in seconds; AutoGluon will train as many models as fit in that
budget. medium_quality + 30 min on a single T4 typically fits ~6-8 models.

The trained predictor is portable: load with
    from autogluon.tabular import TabularPredictor
    pred = TabularPredictor.load("models/ag_BTCUSDT_4h/")
    pred.predict_proba(new_features)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# AutoGluon import is intentionally LAZY so this file imports on machines
# without it (regime_classifier, feature_engine etc. don't need AG). The
# real call site in build_predictor will fail if AG isn't installed.

from feature_engine import (
    FEATURE_NAMES, FEATURE_NAMES_WITH_MACRO,
    build_features, attach_macro_features,
    make_target_triple_class,
    compute_sample_uniqueness,
    TRIPLE_CLASS_HOLD, TRIPLE_CLASS_LONG, TRIPLE_CLASS_SHORT, TRIPLE_CLASS_UNLABELED,
    _atr,
)
from data_fetcher import fetch_or_cache


ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None and numpy scalars with
    Python natives. Round-8 reviewer find: `json.dump` emits literal `NaN`
    for `float('nan')` which is invalid JSON per RFC 8259 — strict parsers
    (Go encoding/json, jq) reject the file. Default fallback `default=str`
    doesn't intercept floats (they're "serializable", just non-strict).
    """
    import math
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if hasattr(obj, "item"):  # numpy scalars
        try:
            v = obj.item()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        except (ValueError, AttributeError):
            return str(obj)
    return obj


def atomic_write_json(path: Path, payload: dict) -> None:
    """Crash-safe JSON write: tmp+rename so SIGINT mid-write doesn't truncate
    the destination. NaN/Inf sanitized to null pre-write for RFC 8259
    compliance — strict downstream parsers (Go, jq) reject literal `NaN`.
    """
    sanitized = _sanitize_for_json(payload)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(sanitized, f, indent=2, default=str, allow_nan=False)
    tmp.replace(path)

TARGET_COL = "target_3class"

# Per-TF horizons + barriers.
#
# 2026-04-27 SYMMETRIC FIX: barriers are now upper==lower per TF. The old
# legacy values 1.5/1.0 (long needs +1.5×ATR, short only -1.0×ATR) baked a
# structural prior into the labels — even on a noisy random walk over an
# 18-bar horizon, the closer barrier wins ~58% of the time. That's the
# real source of the 27% long / 63% short bias seen in WFCV (Sharpe -14.49
# pre-bug-fix) — model was just matching the asymmetric base rate of labels.
#
# Symmetric barriers force the target definition itself to be direction-
# neutral: long-wins and short-wins should now be ~equal in count, with
# the residual being timeout-hold. Any direction skew the model learns
# will then be a TRUE signal from the features, not a label artifact.
TF_CONFIG = {
    "5m":  {"horizon": 12, "tb_upper": 2.0, "tb_lower": 2.0},
    "15m": {"horizon": 16, "tb_upper": 2.0, "tb_lower": 2.0},
    "1h":  {"horizon": 24, "tb_upper": 1.5, "tb_lower": 1.5},
    "4h":  {"horizon": 18, "tb_upper": 1.5, "tb_lower": 1.5},
    "1d":  {"horizon":  5, "tb_upper": 2.2, "tb_lower": 2.2},
}

# AutoGluon preset → expected wall-clock per ~50k rows on a single T4. Picked
# medium_quality as default because best_quality blows past the 6 GPU-hour
# session budget on its own.
PRESET_NOTES = {
    "medium_quality":     "~10-20 min / 50k rows — first try, 6-8 models in ensemble",
    "good_quality":       "~30-60 min / 50k rows — same models, longer HPO",
    "high_quality":       "~1-2 hr / 50k rows — adds NN and CatBoost variants",
    "best_quality":       "~3-6 hr / 50k rows — full bag stacking, often only 1-3% better",
    "experimental_quality":"unbounded — multiple stack levels, mostly research",
}


# ─── Build labelled feature frame ────────────────────────────────────────────

def build_dataset(
    symbol: str, interval: str, years: float, with_macro: bool,
    btc_close: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, str, dict]:
    """Returns (frame, target_col_name, metadata).

    Frame contains all FEATURE_NAMES_WITH_MACRO (or FEATURE_NAMES if
    with_macro=False), plus columns: target_3class, open_time, close.

    Rows with target == TRIPLE_CLASS_UNLABELED (-1) are dropped.
    """
    cfg = TF_CONFIG.get(interval)
    if cfg is None:
        raise ValueError(f"Unknown interval {interval}; expected one of {list(TF_CONFIG)}")

    df = fetch_or_cache(symbol, interval, years=years)
    print(f"  candles: {len(df):,} ({df['open_time'].min()} → {df['open_time'].max()})")

    feat = build_features(df, btc_close=btc_close)
    if with_macro:
        # strict=True: missing macro parquet raises instead of silent zero-fill.
        # Production training MUST fail loudly when claimed-macro is actually
        # zero macro — this prevents the "with_macro=True in log, but real
        # signal is identical to no-macro" regression seen in earlier sessions.
        feat = attach_macro_features(feat, symbol, strict=True)
        feature_cols = FEATURE_NAMES_WITH_MACRO
    else:
        feature_cols = FEATURE_NAMES

    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    close_arr = df["close"].to_numpy()
    atr_arr = _atr(high_arr, low_arr, close_arr, period=14)
    target = make_target_triple_class(
        high_arr, low_arr, close_arr, atr_arr,
        horizon=cfg["horizon"],
        upper_mult=cfg["tb_upper"], lower_mult=cfg["tb_lower"],
    )

    feat[TARGET_COL] = target
    feat = feat[feat[TARGET_COL] != TRIPLE_CLASS_UNLABELED].reset_index(drop=True)

    # Sample weights for overlapping labels (AFML Ch.4). Adjacent bars share
    # 17/18 of their future window with horizon=18 — without weights, AG
    # double-counts shared information and validation looks better than truth.
    feat["_sample_weight"] = compute_sample_uniqueness(len(feat), cfg["horizon"])

    dist = {
        "hold": float((feat[TARGET_COL] == TRIPLE_CLASS_HOLD).mean()),
        "long": float((feat[TARGET_COL] == TRIPLE_CLASS_LONG).mean()),
        "short": float((feat[TARGET_COL] == TRIPLE_CLASS_SHORT).mean()),
    }
    print(f"  labeled: {len(feat):,} rows — "
          f"hold={dist['hold']:.1%} long={dist['long']:.1%} short={dist['short']:.1%}")

    meta = {
        "symbol": symbol,
        "interval": interval,
        "with_macro": with_macro,
        "horizon": cfg["horizon"],
        "tb_upper": cfg["tb_upper"],
        "tb_lower": cfg["tb_lower"],
        "n_rows": int(len(feat)),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "class_distribution": dist,
    }
    return feat, TARGET_COL, meta


# ─── AutoGluon train wrapper ─────────────────────────────────────────────────

def assert_gpu_available(num_gpus_required: int = 1) -> dict:
    """Hard-fail if CUDA isn't visible. Returns a debug summary dict.

    Background: previous sessions silently fell back to CPU when XGBoost's
    `device='cuda'` probe raised. AutoGluon doesn't fail loudly either — it
    just trains the GPU-preferring models on CPU. Result: a 6-hour run that
    delivered the same numbers as a 20-minute CPU baseline.

    Call this BEFORE TabularPredictor.fit() so we abort cheaply when the
    box is misconfigured (vast.ai sometimes hands out instances where the
    container's CUDA libs don't match the host driver).
    """
    import subprocess
    info = {"num_gpus_required": num_gpus_required}

    # nvidia-smi is the source of truth — it polls the actual driver, not
    # whatever cached env vars torch/xgb might think they see.
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            raise RuntimeError(f"nvidia-smi rc={out.returncode}: {out.stderr.strip()}")
        gpus = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        info["nvidia_smi"] = gpus
        if len(gpus) < num_gpus_required:
            raise RuntimeError(
                f"Need {num_gpus_required} GPU(s); nvidia-smi reports {len(gpus)}: {gpus}"
            )
    except FileNotFoundError:
        raise RuntimeError(
            "nvidia-smi not found — this script must run on a CUDA host. "
            "If you're testing locally, pass --no-require-gpu (CPU smoke test only)."
        )

    # Cross-check with PyTorch (NN_TORCH model in AG uses this).
    try:
        import torch
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_device_count"] = int(torch.cuda.device_count())
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() == False")
    except ImportError:
        info["torch_cuda_available"] = "torch not installed"

    # Cross-check XGBoost — in AG 1.1.1 the XGB sub-model uses device='cuda'.
    try:
        import xgboost as xgb
        from xgboost import XGBClassifier
        probe = XGBClassifier(n_estimators=3, device="cuda", tree_method="hist",
                              objective="binary:logistic", verbosity=0)
        Xp = np.random.rand(60, 8).astype(np.float32)
        yp = np.random.randint(0, 2, 60)
        probe.fit(Xp, yp)
        info["xgb_cuda_probe"] = "OK"
    except Exception as e:
        raise RuntimeError(f"XGBoost CUDA probe failed: {e}")

    print(f"  [gpu_check] PASS — {info['nvidia_smi']}")
    return info


# AutoGluon hyperparameters with explicit num_gpus=1 on every model that
# supports GPU. Without this, AG 1.1.1 trains GBM/XGB/CAT/NN_TORCH on CPU
# even when CUDA is available — a silent regression we hit before.
GPU_HYPERPARAMETERS = {
    # Tree boosters — must request GPU explicitly. AG hands these to the
    # underlying lib's GPU code path.
    "GBM": [
        # Default LightGBM with GPU. AG 1.1.1 only honors num_gpus when the
        # underlying box has CUDA libs (otherwise it warns + falls back).
        {"ag_args_fit": {"num_gpus": 1}},
        # Light variant for runtime diversity.
        {"extra_trees": True, "ag_args": {"name_suffix": "XT"},
         "ag_args_fit": {"num_gpus": 1}},
    ],
    "XGB": [{"ag_args_fit": {"num_gpus": 1}}],
    "CAT": [{"ag_args_fit": {"num_gpus": 1}}],
    "NN_TORCH": [{"ag_args_fit": {"num_gpus": 1}}],
    # CPU-only models still useful for diversity in stacked ensemble — AG
    # parallelizes them on the box's CPU cores.
    "RF":  [{}],
    "XT":  [{}],
    "KNN": [{}],
}


def train_predictor(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: Path,
    presets: str = "medium_quality",
    time_limit: int = 1800,
    eval_metric: str = "log_loss",
    num_gpus: int = 1,
    require_gpu: bool = True,
    tuning_frac: float = 0.15,
    horizon: int = 18,
    seed: int = 42,
) -> "TabularPredictor":  # noqa: F821 — runtime import
    """Train a multi-class TabularPredictor and return it.

    Round 4 fixes:
    - Replaced explicit `hyperparameters=GPU_HYPERPARAMETERS` (which
      OVERRIDES the preset's tuned hyperparameters dict) with global
      `ag_args_fit={"num_gpus": num_gpus}`. This propagates to all
      GPU-supporting sub-models AND lets AG's preset defaults (HPO sweeps,
      tuned LR/depth) actually run.
    - Added chronological `tuning_data` split + `num_bag_folds=0` to defeat
      AG's default random k-fold bagging, which leaks future into past via
      WeightedEnsemble_L2 stacker (López de Prado AFML standard violation).
      Tuning_data is the LAST `tuning_frac` of train_df chronologically.
    """
    from autogluon.tabular import TabularPredictor  # lazy import

    if require_gpu:
        assert_gpu_available(num_gpus_required=num_gpus)

    # Reproducibility: seed numpy + python random. Most AG sub-models
    # (XGB/LGBM/RF/CAT) honor global numpy seed for bootstrap sampling.
    # Without this, identical data gives different leaderboards across runs.
    import random
    random.seed(seed)
    np.random.seed(seed)

    cols = feature_cols + [TARGET_COL]
    if "_sample_weight" in train_df.columns:
        cols = cols + ["_sample_weight"]
    full_train = train_df[cols].copy().reset_index(drop=True)

    # Chronological split with INNER purge gap: last `tuning_frac` of train_df
    # is held out as AG's internal validation set. We also drop `horizon`
    # rows BETWEEN train_data tail and tuning_data head — same AFML rule as
    # the outer split, applied recursively. Without it, the last bars of
    # train_data have labels computed from prices INSIDE tuning_data → AG
    # validation log_loss is inflated → WeightedEnsemble picks based on
    # leaked scores. (Round-5 analyst find.)
    n = len(full_train)
    n_tune = max(int(n * tuning_frac), 50)
    train_end_inner = max(0, n - n_tune - horizon)
    if train_end_inner < 100:
        raise ValueError(
            f"Inner purge gap left only {train_end_inner} train rows "
            f"(n={n}, n_tune={n_tune}, horizon={horizon}). Need >=100. "
            f"Pass larger --years, smaller tuning_frac, or shorter horizon."
        )
    train_data = full_train.iloc[:train_end_inner]
    tuning_data = full_train.iloc[n - n_tune:]

    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    print(f"  AutoGluon presets={presets} time_limit={time_limit}s "
          f"num_gpus={num_gpus} ({PRESET_NOTES.get(presets, '?')})")
    print(f"    chronological inner split: train={len(train_data):,} + "
          f"tuning={len(tuning_data):,}")

    sample_weight_kw = {}
    if "_sample_weight" in full_train.columns:
        # weight_evaluation=False: AG 1.5.0 bug — calibrate_model() in
        # post-fit calls score with weights=None which collides with
        # weight_evaluation=True (raises ValueError mid-fit). We still pass
        # sample_weight= so each sub-model trains weighted; we just don't
        # weight the validation score. AFML uniqueness still applied at
        # training time, just not at calibration scoring.
        sample_weight_kw = {"sample_weight": "_sample_weight",
                            "weight_evaluation": False}
        print(f"    AFML sample weights: mean={full_train['_sample_weight'].mean():.3f}, "
              f"min={full_train['_sample_weight'].min():.3f}")

    pred = TabularPredictor(
        label=TARGET_COL,
        path=str(output_dir),
        problem_type="multiclass",
        eval_metric=eval_metric,
        verbosity=2,
        **sample_weight_kw,
    ).fit(
        train_data=train_data,
        tuning_data=tuning_data,
        presets=presets,
        time_limit=time_limit,
        # GPU pass-through to all GPU-capable sub-models — preset defaults
        # otherwise pick HP for tree boosters, NN, etc.
        ag_args_fit={"num_gpus": num_gpus} if num_gpus > 0 else None,
        # Disable random k-fold bagging — we provide chronological tuning_data.
        # AG accepts num_bag_folds=0 to mean "no bagging, single fits".
        num_bag_folds=0,
        num_stack_levels=0,
    )
    return pred


def evaluate_predictor(pred, holdout_df: pd.DataFrame, feature_cols: List[str]) -> dict:
    """Compute holdout metrics: accuracy, per-class F1, log_loss, leaderboard.

    Robust column-ordering: AutoGluon's predict_proba columns are ordered by
    `predictor.class_labels` and may be int / np.int64 / string across
    versions. We canonicalise via class_labels lookup so log_loss and HC
    indexing don't get scrambled when AG returns columns in unexpected order.
    """
    # Feature-set drift guard (Round-7 architecture I7): warn loudly if AG
    # dropped features during training so we notice when "with_macro" runs
    # actually used fewer features than declared.
    try:
        trained_features = set(pred.feature_metadata_in.get_features())
        dropped = set(feature_cols) - trained_features
        if dropped:
            print(f"  [feature_drift_warn] {len(dropped)} requested features absent "
                  f"from trained predictor: {sorted(dropped)[:5]}"
                  f"{'...' if len(dropped) > 5 else ''}")
    except Exception:
        pass  # older AG versions lack feature_metadata_in

    cols = feature_cols + [TARGET_COL]
    holdout = holdout_df[cols].copy()
    # AG leaderboard contains np.float32/np.int64 — json.dump can't serialize
    # those directly. Cast to plain Python types before downstream JSON dump.
    leaderboard = pred.leaderboard(holdout, silent=True).astype(object)
    # Replace np types with native Python equivalents row-by-row (cheap on
    # ~10 rows × 8 cols).
    def _native(v):
        if hasattr(v, "item"):
            try:
                return v.item()
            except (ValueError, AttributeError):
                pass
        if isinstance(v, float) and (v != v):  # NaN
            return None
        return v
    leaderboard = leaderboard.map(_native) if hasattr(leaderboard, "map") else leaderboard.applymap(_native)
    proba = pred.predict_proba(holdout[feature_cols])
    pred_labels = pred.predict(holdout[feature_cols]).to_numpy()
    y_true = holdout[TARGET_COL].to_numpy()

    from sklearn.metrics import accuracy_score, f1_score, log_loss
    acc = float(accuracy_score(y_true, pred_labels))
    f1m = float(f1_score(y_true, pred_labels, labels=[0, 1, 2], average="macro",
                         zero_division=0))

    # Canonical class label → column index map. Coerce to int for AG version
    # robustness (np.int64 in 1.6+ would break `.index()`).
    class_labels = [int(x) for x in pred.class_labels]
    n_holdout = len(holdout)

    def _col_for(class_id: int) -> np.ndarray:
        if class_id in class_labels:
            return proba.iloc[:, class_labels.index(class_id)].to_numpy()
        # See wfcv_regime_aware.run_fold() for rationale: silent-zero hides
        # AG class-drop bugs. Raise if class exists in y_true.
        n_in_test = int((y_true == class_id).sum())
        if n_in_test > 0:
            raise RuntimeError(
                f"Class {class_id} present in y_true ({n_in_test} bars) but "
                f"missing from predictor.class_labels={class_labels}. Silent "
                f"zero would corrupt log_loss/accuracy."
            )
        return np.zeros(n_holdout)

    # log_loss with correctly-ordered columns. We assemble [hold, long, short]
    # so labels=[0,1,2] matches column order regardless of how AG stored them.
    proba_canonical = np.stack([
        _col_for(TRIPLE_CLASS_HOLD),
        _col_for(TRIPLE_CLASS_LONG),
        _col_for(TRIPLE_CLASS_SHORT),
    ], axis=1)
    try:
        ll = float(log_loss(y_true, proba_canonical, labels=[0, 1, 2]))
    except Exception:
        ll = float("nan")

    p_long = _col_for(TRIPLE_CLASS_LONG)
    p_short = _col_for(TRIPLE_CLASS_SHORT)
    hc = {}
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70):
        long_mask = p_long > thr
        short_mask = p_short > thr
        n_l, n_s = int(long_mask.sum()), int(short_mask.sum())
        prec_l = float((y_true[long_mask] == TRIPLE_CLASS_LONG).mean()) if n_l else 0.0
        prec_s = float((y_true[short_mask] == TRIPLE_CLASS_SHORT).mean()) if n_s else 0.0
        hc[f"thr_{thr:.2f}"] = {
            "n_long": n_l, "prec_long": prec_l,
            "n_short": n_s, "prec_short": prec_s,
        }

    return {
        "n_holdout": int(len(holdout)),
        "accuracy": acc,
        "f1_macro": f1m,
        "log_loss": ll,
        "hc": hc,
        "leaderboard": leaderboard.to_dict(orient="records"),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

_SYMBOL_RE = __import__("re").compile(r"^[A-Z0-9]{2,12}$")


def _validate_symbol(s: str) -> str:
    """CLI validator: prevents path-traversal via --symbol "../etc"."""
    if not _SYMBOL_RE.fullmatch(s):
        raise __import__("argparse").ArgumentTypeError(
            f"--symbol must match {_SYMBOL_RE.pattern}; got {s!r}"
        )
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT", type=_validate_symbol)
    ap.add_argument("--interval", default="4h", choices=list(TF_CONFIG))
    ap.add_argument("--years", type=float, default=8.0,
                    help="History window. 8 ≈ full Binance spot coverage.")
    ap.add_argument("--with-macro", action="store_true",
                    help="Include Stage 2 macro features (funding/OI/F&G)")
    ap.add_argument("--presets", default="medium_quality",
                    choices=list(PRESET_NOTES))
    ap.add_argument("--time-limit", type=int, default=1800,
                    help="AutoGluon training budget (seconds)")
    ap.add_argument("--holdout-frac", type=float, default=0.15,
                    help="Last N%% of data reserved for evaluation (chronological)")
    ap.add_argument("--eval-metric", default="log_loss",
                    choices=["log_loss", "accuracy", "balanced_accuracy", "f1_macro"])
    ap.add_argument("--quick", action="store_true",
                    help="time_limit=300, holdout 0.10 — for local smoke tests "
                         "(do NOT use for vast.ai runs)")
    ap.add_argument("--num-gpus", type=int, default=1,
                    help="GPUs to allocate per model (1 default, 0 = CPU only). "
                         "Set 0 only for local smoke tests on machines without CUDA.")
    ap.add_argument("--no-require-gpu", action="store_true",
                    help="Skip the nvidia-smi GPU sanity check before fit. "
                         "Use ONLY for local CPU smoke tests; production runs "
                         "must keep the check on so we fail loudly if the box "
                         "has no CUDA.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Reproducibility seed for numpy/random (passed to "
                         "AG sub-models that honor global rng).")
    args = ap.parse_args()

    t0 = time.time()
    print(f"━━━ AutoGluon train: {args.symbol} {args.interval} "
          f"(macro={args.with_macro}) ━━━")
    if args.quick:
        args.time_limit = 300
        args.holdout_frac = 0.10
        print("  QUICK MODE: time_limit=300s, holdout_frac=0.10 (smoke test only)")

    # Load BTC close for cross-asset features when training non-BTC.
    btc_close = None
    if args.symbol != "BTCUSDT":
        btc_df = fetch_or_cache("BTCUSDT", args.interval, years=args.years)
        btc_close = btc_df["close"].to_numpy()

    feat, _, meta = build_dataset(args.symbol, args.interval, args.years,
                                   args.with_macro, btc_close=btc_close)
    feature_cols = meta["feature_cols"]

    # Chronological train/holdout split with PURGE GAP.
    #
    # Target = close[t+horizon]. Without a gap, the last `horizon` rows of
    # `train_df` have labels computed from close prices INSIDE `holdout_df` —
    # so reported holdout accuracy/log_loss are inflated by label leakage.
    # Round-3 review (López de Prado, AFML Ch.7 + Bailey "Pseudo-mathematics
    # of finance") flagged this as FATAL. Same fix as wfcv_regime_aware.run_fold
    # and multi_agent_fusion.main: drop the last `horizon` rows of train.
    horizon = TF_CONFIG[args.interval]["horizon"]
    n = len(feat)
    cut = int(n * (1 - args.holdout_frac))
    train_end = max(0, cut - horizon)
    train_df = feat.iloc[:train_end]
    holdout_df = feat.iloc[cut:]
    print(f"  split: train={len(train_df):,} ({train_df['open_time'].min()} → "
          f"{train_df['open_time'].max()}), "
          f"holdout={len(holdout_df):,} ({holdout_df['open_time'].min()} → "
          f"{holdout_df['open_time'].max()})  "
          f"[purge gap = {horizon} bars]")

    macro_tag = "_macro" if args.with_macro else ""
    output_dir = MODELS_DIR / f"ag_{args.symbol}_{args.interval}{macro_tag}"
    pred = train_predictor(
        train_df, feature_cols, output_dir,
        presets=args.presets, time_limit=args.time_limit,
        eval_metric=args.eval_metric,
        num_gpus=args.num_gpus,
        require_gpu=not args.no_require_gpu,
        horizon=meta["horizon"],
        seed=args.seed,
    )
    eval_results = evaluate_predictor(pred, holdout_df, feature_cols)

    elapsed = time.time() - t0
    print(f"\n━━━ Holdout eval ({eval_results['n_holdout']:,} bars) ━━━")
    print(f"  accuracy = {eval_results['accuracy']:.4f}")
    print(f"  f1_macro = {eval_results['f1_macro']:.4f}")
    print(f"  log_loss = {eval_results['log_loss']:.4f}")
    print(f"  HC table:")
    for thr, stats in eval_results["hc"].items():
        print(f"    {thr}: long={stats['n_long']:4d}@{stats['prec_long']:.1%} "
              f"short={stats['n_short']:4d}@{stats['prec_short']:.1%}")
    print(f"  leaderboard top 5:")
    for row in eval_results["leaderboard"][:5]:
        print(f"    {row.get('model','?'):30s} score_val={row.get('score_val',0):.4f} "
              f"score_test={row.get('score_test',0):.4f}")

    # Persist run summary.
    summary = {
        "version": "autogluon_v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": elapsed,
        "symbol": args.symbol,
        "interval": args.interval,
        "presets": args.presets,
        "time_limit": args.time_limit,
        "eval_metric": args.eval_metric,
        "with_macro": args.with_macro,
        "horizon": meta["horizon"],
        "tb_upper": meta["tb_upper"],
        "tb_lower": meta["tb_lower"],
        "n_features": meta["n_features"],
        "feature_cols": feature_cols,
        "class_distribution": meta["class_distribution"],
        "predictor_path": str(output_dir.relative_to(ROOT)),
        "holdout": eval_results,
    }
    summary_path = LOGS_DIR / f"autogluon_{args.symbol}_{args.interval}{macro_tag}.json"
    atomic_write_json(summary_path, summary)
    print(f"\n  predictor → {output_dir.relative_to(ROOT)}")
    print(f"  summary   → {summary_path.relative_to(ROOT)}")
    print(f"  ⏱  {elapsed:.1f}s")

    # Update latest_autogluon pointer with flock + atomic rename. Prevents
    # concurrent training runs (manual + tmux background) from clobbering
    # each other's pointer entries. Round-6 security finding.
    import fcntl
    latest_path = MODELS_DIR / "latest_autogluon.json"
    lock_path = latest_path.with_suffix(".json.lock")
    tmp_path = latest_path.with_suffix(".json.tmp")
    with open(lock_path, "a+") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            if latest_path.exists():
                with open(latest_path) as f:
                    latest = json.load(f)
            else:
                latest = {}
            key = f"{args.symbol}_{args.interval}{macro_tag}"
            latest[key] = {
                "predictor_path": summary["predictor_path"],
                "summary_log": str(summary_path.relative_to(ROOT)),
                "trained_at": summary["trained_at"],
                "accuracy": eval_results["accuracy"],
                "f1_macro": eval_results["f1_macro"],
                "log_loss": eval_results["log_loss"],
                "with_macro": args.with_macro,
            }
            with open(tmp_path, "w") as f:
                json.dump(latest, f, indent=2, default=str)
            tmp_path.replace(latest_path)  # atomic on POSIX
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)
    print(f"  pointer   -> {latest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
