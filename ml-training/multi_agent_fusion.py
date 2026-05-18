"""multi_agent_fusion.py — long-specialist + short-specialist + judge ensemble
with probabilistic output (P_long, P_short, P_hold).

Variant 1 from project_v3_layered_fusion.md §"Multi-agent / Mixture of Experts":

    Long-specialist  → binary classifier P(long wins | bar features)
    Short-specialist → binary classifier P(short wins | bar features)
    Judge agent      → combines specialist outputs into final 3-class proba

Why this beats a single 3-class model:
    1. **Targeted decision boundary.** The long-specialist trains only on
       (long-wins / not-long-wins) — it doesn't have to also distinguish hold
       from short. Sharper feature importance per class.
    2. **No 100%-direction collapse.** The collapse seen in Patch 4 Stage 1
       (model predicts short on every bar) happens when one class dominates
       3-class softmax. With two binary specialists, "I'm not confident on
       long" and "I'm not confident on short" produce a hold signal naturally.
    3. **Calibration per side.** Long and short probabilities are calibrated
       independently against their own positive base rates.

Judge composition strategies (selectable via --judge):
    average  : P_long = LS_proba; P_short = SS_proba; P_hold = 1 - max(P_long, P_short)
               After max-floor at 0, normalized to sum=1.
    veto     : P_long iff LS > t and SS < 1-t. Else hold. (Strict — fewer trades.)
    learned  : Train a tiny meta classifier on a small calibration set whose
               input is [LS_proba, SS_proba] and target is the original 3-class
               label. Output is real 3-class probabilities. Default.

Three-class probabilistic output is what `analyzers.py` will display in the
Telegram bot — see project_v3_layered_fusion.md §"Probabilistic output".

Usage:
    python multi_agent_fusion.py --symbol BTCUSDT --interval 4h --with-macro \
        --judge learned --presets medium_quality --time-limit 1200

This script is a sibling of train_autogluon.py — it produces a directory at
models/fusion_{symbol}_{interval}/ containing:
    long_specialist/      AutoGluon predictor (binary)
    short_specialist/     AutoGluon predictor (binary)
    judge.json            judge config + (if learned) sklearn LogReg coefs
    fusion_meta.json      provenance + holdout metrics

To predict at inference time:
    from multi_agent_fusion import FusionPredictor
    fp = FusionPredictor.load("models/fusion_BTCUSDT_4h/")
    p_hold, p_long, p_short = fp.predict_proba(features_row)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from feature_engine import (
    FEATURE_NAMES, FEATURE_NAMES_WITH_MACRO,
    TRIPLE_CLASS_HOLD, TRIPLE_CLASS_LONG, TRIPLE_CLASS_SHORT,
)
from train_autogluon import (
    TF_CONFIG, TARGET_COL, build_dataset, PRESET_NOTES, _validate_symbol,
    atomic_write_json,
)


ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"


# ─── Specialist label engineering ────────────────────────────────────────────

LS_LABEL = "is_long"   # 1 if target == long, else 0
SS_LABEL = "is_short"  # 1 if target == short, else 0


def attach_specialist_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Adds LS_LABEL and SS_LABEL columns alongside TARGET_COL.

    The DataFrame is returned with new columns; original is not mutated.
    """
    out = df.copy()
    t = out[TARGET_COL]
    out[LS_LABEL] = (t == TRIPLE_CLASS_LONG).astype(int)
    out[SS_LABEL] = (t == TRIPLE_CLASS_SHORT).astype(int)
    return out


# ─── Specialist trainer (binary AutoGluon) ───────────────────────────────────

def train_specialist(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    output_dir: Path,
    presets: str,
    time_limit: int,
    num_gpus: int = 1,
    require_gpu: bool = True,
    horizon: int = 18,
):
    """Train a binary AutoGluon predictor for one direction.

    GPU enforcement mirrors train_autogluon.train_predictor: assert_gpu_available
    is called once when require_gpu=True, then GPU_HYPERPARAMETERS is passed
    to fit() so each sub-model that supports CUDA actually uses it.
    """
    from autogluon.tabular import TabularPredictor
    from train_autogluon import assert_gpu_available

    if require_gpu:
        assert_gpu_available(num_gpus_required=num_gpus)

    # Carry _sample_weight if present (AFML Ch.4 uniqueness — Round-7 ADR
    # consistency fix; specialists were silently training without it).
    cols = feature_cols + [label_col]
    if "_sample_weight" in train_df.columns:
        cols = cols + ["_sample_weight"]
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    full = train_df[cols].copy().reset_index(drop=True)
    n = len(full)
    n_tune = max(int(n * 0.15), 50)
    inner_train_end = max(0, n - n_tune - horizon)
    if inner_train_end < 100:
        raise ValueError(
            f"Inner purge gap left only {inner_train_end} train rows for "
            f"{label_col} (n={n}, n_tune={n_tune}, horizon={horizon}). Need >=100."
        )
    inner_train = full.iloc[:inner_train_end]
    inner_tune = full.iloc[n - n_tune:]

    sample_weight_kw = {}
    if "_sample_weight" in inner_train.columns:
        # weight_evaluation=False — AG 1.5.0 calibrate_model bug (see
        # train_autogluon.train_predictor for full explanation).
        sample_weight_kw = {"sample_weight": "_sample_weight",
                            "weight_evaluation": False}

    return TabularPredictor(
        label=label_col,
        path=str(output_dir),
        problem_type="binary",
        eval_metric="log_loss",
        verbosity=2,
        **sample_weight_kw,
    ).fit(
        train_data=inner_train,
        tuning_data=inner_tune,
        presets=presets,
        time_limit=time_limit,
        ag_args_fit={"num_gpus": num_gpus} if num_gpus > 0 else None,
        num_bag_folds=0,
        num_stack_levels=0,
    )


# ─── Judges ──────────────────────────────────────────────────────────────────

def judge_average(p_long_arr: np.ndarray, p_short_arr: np.ndarray) -> np.ndarray:
    """Naive judge: P_hold = 1 - max(P_long, P_short), then normalize."""
    n = len(p_long_arr)
    p_hold = 1.0 - np.maximum(p_long_arr, p_short_arr)
    p_hold = np.clip(p_hold, 0.0, 1.0)
    raw = np.stack([p_hold, p_long_arr, p_short_arr], axis=1)
    return raw / raw.sum(axis=1, keepdims=True)


def judge_veto(p_long_arr: np.ndarray, p_short_arr: np.ndarray,
               threshold: float = 0.55) -> np.ndarray:
    """Veto judge: long requires LS > thr AND SS < 1-thr (and mirror for short).

    Avoids "agree to disagree" bars where both specialists are bullish-leaning.
    Most bars resolve to hold, hence this is the conservative judge.
    """
    n = len(p_long_arr)
    long_mask = (p_long_arr > threshold) & (p_short_arr < 1.0 - threshold)
    short_mask = (p_short_arr > threshold) & (p_long_arr < 1.0 - threshold)
    out = np.zeros((n, 3), dtype=float)
    out[long_mask, 1] = p_long_arr[long_mask]
    out[short_mask, 2] = p_short_arr[short_mask]
    out[~long_mask & ~short_mask, 0] = 1.0
    # For long/short bars: distribute remaining mass to hold for honesty.
    rem = 1.0 - out.sum(axis=1)
    out[:, 0] += rem.clip(0)
    return out / out.sum(axis=1, keepdims=True)


def fit_learned_judge(p_long_train: np.ndarray, p_short_train: np.ndarray,
                      y_3class_train: np.ndarray) -> dict:
    """Multinomial LogReg on (LS_proba, SS_proba) → 3-class label.

    Returns a dict with serialized coefficients (loadable on inference side
    without sklearn dependency at runtime if we choose).
    """
    from sklearn.linear_model import LogisticRegression
    X = np.stack([p_long_train, p_short_train], axis=1)
    # multinomial + lbfgs to get true 3-class proba (default OvR loses calibration).
    lr = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=500, C=1.0,
    )
    lr.fit(X, y_3class_train)
    return {
        "kind": "learned_logreg",
        "classes": lr.classes_.tolist(),
        "coef": lr.coef_.tolist(),
        "intercept": lr.intercept_.tolist(),
    }


def apply_learned_judge(judge: dict, p_long_arr: np.ndarray, p_short_arr: np.ndarray) -> np.ndarray:
    """Score a learned judge without re-instantiating sklearn (pure numpy).

    SAFETY (2026-04-27): if the calibration set was missing a class (e.g.
    judge_df happens to be a chop-only window with zero long-wins), then
    `judge['classes']` returns only the present labels — say `[0, 2]`. In
    that case `out[:, 1]` (P_long) stays at zero forever and silently
    suppresses every long signal at inference. We fall back to
    `judge_average` when fewer than 3 classes are available.
    """
    classes = judge["classes"]
    if sorted(classes) != [0, 1, 2]:
        # Degenerate judge — log loudly and use the naive judge instead so
        # at least one direction class doesn't lose its proba mass.
        print(f"  WARN: learned judge has classes={classes}, missing some of "
              f"[hold=0, long=1, short=2]. Falling back to judge_average.")
        return judge_average(p_long_arr, p_short_arr)

    coef = np.asarray(judge["coef"])         # shape (n_classes, 2)
    intercept = np.asarray(judge["intercept"])  # shape (n_classes,)
    X = np.stack([p_long_arr, p_short_arr], axis=1)
    logits = X @ coef.T + intercept
    # Softmax per row.
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    proba = exp / exp.sum(axis=1, keepdims=True)
    # Reorder columns into [hold=0, long=1, short=2] for downstream consistency.
    out = np.zeros((len(X), 3), dtype=float)
    for i, c in enumerate(classes):
        out[:, c] = proba[:, i]
    return out


# ─── Inference-time wrapper ──────────────────────────────────────────────────

@dataclass
class FusionMeta:
    symbol: str
    interval: str
    with_macro: bool
    feature_cols: List[str]
    judge: str
    judge_config: dict
    long_specialist_path: str
    short_specialist_path: str
    trained_at: str
    holdout_metrics: dict


class FusionPredictor:
    """Loads a saved fusion bundle and predicts 3-class proba on new features."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path / "fusion_meta.json") as f:
            self.meta = FusionMeta(**json.load(f))
        from autogluon.tabular import TabularPredictor
        self.long_specialist = TabularPredictor.load(self.meta.long_specialist_path)
        self.short_specialist = TabularPredictor.load(self.meta.short_specialist_path)

    @classmethod
    def load(cls, path: str | Path) -> "FusionPredictor":
        return cls(Path(path))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Returns shape (n, 3) with columns [P_hold, P_long, P_short]."""
        cols = self.meta.feature_cols
        x = features[cols].copy()
        ls_proba = self.long_specialist.predict_proba(x).iloc[:, 1].to_numpy()
        ss_proba = self.short_specialist.predict_proba(x).iloc[:, 1].to_numpy()
        if self.meta.judge == "average":
            return judge_average(ls_proba, ss_proba)
        if self.meta.judge == "veto":
            thr = self.meta.judge_config.get("threshold", 0.55)
            return judge_veto(ls_proba, ss_proba, threshold=thr)
        if self.meta.judge == "learned":
            return apply_learned_judge(self.meta.judge_config, ls_proba, ss_proba)
        raise ValueError(f"unknown judge: {self.meta.judge}")


# ─── Holdout evaluation ──────────────────────────────────────────────────────

def evaluate_fusion(
    p_arr: np.ndarray, y_3class: np.ndarray,
) -> dict:
    """Compute fusion-level metrics: accuracy, F1, HC table, direction balance.

    p_arr: (n, 3) with cols [hold, long, short].
    """
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    pred_labels = np.argmax(p_arr, axis=1)
    acc = float(accuracy_score(y_3class, pred_labels))
    f1m = float(f1_score(y_3class, pred_labels, labels=[0, 1, 2],
                         average="macro", zero_division=0))
    try:
        ll = float(log_loss(y_3class, p_arr, labels=[0, 1, 2]))
    except Exception:
        ll = float("nan")

    p_long = p_arr[:, 1]
    p_short = p_arr[:, 2]
    hc = {}
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70):
        long_mask = p_long > thr
        short_mask = p_short > thr
        n_l, n_s = int(long_mask.sum()), int(short_mask.sum())
        prec_l = float((y_3class[long_mask] == TRIPLE_CLASS_LONG).mean()) if n_l else 0.0
        prec_s = float((y_3class[short_mask] == TRIPLE_CLASS_SHORT).mean()) if n_s else 0.0
        hc[f"thr_{thr:.2f}"] = {
            "n_long": n_l, "prec_long": prec_l,
            "n_short": n_s, "prec_short": prec_s,
        }
    long_share = float((p_long > 0.55).mean())
    short_share = float((p_short > 0.55).mean())
    return {
        "accuracy": acc,
        "f1_macro": f1m,
        "log_loss": ll,
        "hc": hc,
        "long_share": long_share,
        "short_share": short_share,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT", type=_validate_symbol)
    ap.add_argument("--interval", default="4h", choices=list(TF_CONFIG))
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--with-macro", action="store_true")
    ap.add_argument("--presets", default="medium_quality", choices=list(PRESET_NOTES))
    ap.add_argument("--time-limit", type=int, default=1200,
                    help="Per-specialist AutoGluon budget (seconds). Total run = "
                         "2 × time_limit + judge fit + holdout eval.")
    ap.add_argument("--judge", default="learned",
                    choices=("average", "veto", "learned"))
    ap.add_argument("--veto-threshold", type=float, default=0.55,
                    help="Threshold for veto judge (ignored otherwise)")
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--judge-frac", type=float, default=0.10,
                    help="Fraction of train data reserved for fitting the learned "
                         "judge — kept disjoint from specialist train to avoid "
                         "double-dipping on the same rows.")
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--no-require-gpu", action="store_true",
                    help="Skip the GPU sanity check (CPU smoke test only).")
    args = ap.parse_args()

    t0 = time.time()
    print(f"━━━ Multi-agent fusion: {args.symbol} {args.interval} "
          f"(macro={args.with_macro}, judge={args.judge}) ━━━")

    btc_close = None
    if args.symbol != "BTCUSDT":
        from data_fetcher import fetch_or_cache
        btc_df = fetch_or_cache("BTCUSDT", args.interval, years=args.years)
        btc_close = btc_df["close"].to_numpy()

    feat, _, meta = build_dataset(args.symbol, args.interval, args.years,
                                   args.with_macro, btc_close=btc_close)
    feature_cols = meta["feature_cols"]
    feat = attach_specialist_labels(feat)

    # Three-way chronological split with PURGE GAPS.
    #
    # Target = close[t+horizon] (e.g. 18 bars on 4h). The last `horizon` rows
    # of any left-of-boundary slice have labels that come from bars sitting
    # INSIDE the next slice. Without purging, specialists train on labels
    # informed by judge_df bars; judge_calib is informed by holdout bars.
    # Standard fix: drop `horizon` rows on each side of every boundary.
    horizon = TF_CONFIG[args.interval]["horizon"]
    n = len(feat)
    n_holdout = int(n * args.holdout_frac)
    n_judge = int(n * args.judge_frac)
    cut_holdout = n - n_holdout
    cut_judge = cut_holdout - n_judge
    # Train ends `horizon` rows before judge starts; judge ends `horizon`
    # rows before holdout starts.
    train_end = max(0, cut_judge - horizon)
    judge_end = max(train_end + 1, cut_holdout - horizon)
    train_df = feat.iloc[:train_end]
    judge_df = feat.iloc[cut_judge:judge_end]
    holdout_df = feat.iloc[cut_holdout:]
    print(f"  splits: specialist_train={len(train_df):,}  "
          f"judge_calib={len(judge_df):,}  holdout={len(holdout_df):,}  "
          f"(purge gap = {horizon} bars on each side)")

    # Train specialists in series. AutoGluon already parallelizes inside its
    # own training loop on a multi-core box, so launching both in subprocesses
    # would just halve each one's effective CPU.
    macro_tag = "_macro" if args.with_macro else ""
    fusion_dir = MODELS_DIR / f"fusion_{args.symbol}_{args.interval}{macro_tag}"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    ls_dir = fusion_dir / "long_specialist"
    ss_dir = fusion_dir / "short_specialist"

    print(f"\n[1/3] Long specialist → {ls_dir.relative_to(ROOT)}")
    ls = train_specialist(train_df, feature_cols, LS_LABEL, ls_dir,
                           args.presets, args.time_limit,
                           num_gpus=args.num_gpus, require_gpu=not args.no_require_gpu,
                           horizon=horizon)
    print(f"\n[2/3] Short specialist → {ss_dir.relative_to(ROOT)}")
    ss = train_specialist(train_df, feature_cols, SS_LABEL, ss_dir,
                           args.presets, args.time_limit,
                           num_gpus=args.num_gpus, require_gpu=not args.no_require_gpu,
                           horizon=horizon)

    # Score judge data with both specialists.
    ls_calib = ls.predict_proba(judge_df[feature_cols]).iloc[:, 1].to_numpy()
    ss_calib = ss.predict_proba(judge_df[feature_cols]).iloc[:, 1].to_numpy()
    y_calib = judge_df[TARGET_COL].to_numpy()

    # Fit judge.
    if args.judge == "learned":
        print(f"\n[3/3] Fitting learned judge on {len(judge_df):,} bars")
        judge_cfg = fit_learned_judge(ls_calib, ss_calib, y_calib)
    elif args.judge == "veto":
        judge_cfg = {"kind": "veto", "threshold": args.veto_threshold}
        print(f"\n[3/3] Veto judge with threshold {args.veto_threshold}")
    else:
        judge_cfg = {"kind": "average"}
        print(f"\n[3/3] Average judge")

    # Holdout evaluation.
    ls_hold = ls.predict_proba(holdout_df[feature_cols]).iloc[:, 1].to_numpy()
    ss_hold = ss.predict_proba(holdout_df[feature_cols]).iloc[:, 1].to_numpy()
    if args.judge == "average":
        proba_hold = judge_average(ls_hold, ss_hold)
    elif args.judge == "veto":
        proba_hold = judge_veto(ls_hold, ss_hold, threshold=args.veto_threshold)
    else:
        proba_hold = apply_learned_judge(judge_cfg, ls_hold, ss_hold)

    metrics = evaluate_fusion(proba_hold, holdout_df[TARGET_COL].to_numpy())
    print(f"\n━━━ Fusion holdout ({len(holdout_df):,} bars) ━━━")
    print(f"  accuracy = {metrics['accuracy']:.4f}")
    print(f"  f1_macro = {metrics['f1_macro']:.4f}")
    print(f"  log_loss = {metrics['log_loss']:.4f}")
    print(f"  HC table:")
    for thr, st in metrics["hc"].items():
        print(f"    {thr}: long={st['n_long']:4d}@{st['prec_long']:.1%}  "
              f"short={st['n_short']:4d}@{st['prec_short']:.1%}")
    print(f"  Long  share@0.55: {metrics['long_share']:.1%}  "
          f"(target ~33%; short  bias = bad)")
    print(f"  Short share@0.55: {metrics['short_share']:.1%}")

    # Persist meta.
    meta_obj = FusionMeta(
        symbol=args.symbol,
        interval=args.interval,
        with_macro=args.with_macro,
        feature_cols=feature_cols,
        judge=args.judge,
        judge_config=judge_cfg,
        long_specialist_path=str(ls_dir),
        short_specialist_path=str(ss_dir),
        trained_at=datetime.now(timezone.utc).isoformat(),
        holdout_metrics=metrics,
    )
    atomic_write_json(fusion_dir / "fusion_meta.json", asdict(meta_obj))

    print(f"\n  bundle    → {fusion_dir.relative_to(ROOT)}")
    print(f"  ⏱ total   {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
