"""One-shot script: build the BTC 4h pattern-match index for the bot.

Reads BTC 4h OHLCV from ml-training/data/BTCUSDT_4h.parquet, computes a
small feature vector per bar (RSI/MACD/ATR/level proximity/regime-score
lite — same primitives as analyzers.py), and writes a HNSW index +
forward-returns array to telegram-bot/data/pattern_indexes/BTCUSDT_4h/.

Run from telegram-bot/:
    ml-training/.venv/bin/python scripts/build_btc_4h_pattern_index.py

(uses ml-training venv because that one already has pandas+pyarrow;
the bot venv stays slim — runtime only needs numpy + hnswlib).

Re-run whenever:
- new history is fetched (covers more bars)
- feature definition changes (then rebuild and bump dim)
- horizon (forward-bar count) changes
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure we can import bot's pattern_match module.
HERE = Path(__file__).resolve().parent
BOT_ROOT = HERE.parent
sys.path.insert(0, str(BOT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pattern_match import PatternIndex  # noqa: E402

# ───────────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────────

PARQUET = Path("/Users/admin/NodeElibiri/elibri-backend/ml-training/data/BTCUSDT_4h.parquet")
OUTPUT_DIR = BOT_ROOT / "data" / "pattern_indexes" / "BTCUSDT_4h"
HORIZON_BARS = 12          # 4h × 12 = 48h fwd window (matches analyzer's TP horizon ish)
WARMUP_BARS = 250          # need EMA-200 + slope window
SYMBOL = "BTCUSDT"
INTERVAL = "4h"


# ───────────────────────────────────────────────────────────────────────────
# Feature compute (lite — 7 dims)
# ───────────────────────────────────────────────────────────────────────────
# Keep dim small for HNSW build speed and feature-stability across versions.
# Each feature is causal (.shift(1) where it would otherwise leak current bar).

def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """7 causal features used both at index-build time and at query time
    (analyzer must compute the SAME 7 numbers for the live bar).
    """
    close = df["close"]
    rsi = _rsi(close, 14)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    atr = _atr(df, 14)
    ema200 = _ema(close, 200)
    # EMA-200 slope normalised by ATR — the regime proxy.
    ema200_slope = (ema200 - ema200.shift(50)) / (50 * atr.replace(0, np.nan))
    ema200_slope = ema200_slope.fillna(0.0).clip(-5, 5)
    # Distance to 50-bar high/low (level proximity).
    high_50 = df["high"].rolling(50).max().shift(1)
    low_50 = df["low"].rolling(50).min().shift(1)
    dist_to_high = ((high_50 - close) / atr).replace([np.inf, -np.inf], 0).fillna(0.0)
    dist_to_low = ((close - low_50) / atr).replace([np.inf, -np.inf], 0).fillna(0.0)

    feats = pd.DataFrame({
        "rsi": rsi.fillna(50.0),
        "macd_diff": (macd_line - macd_signal).fillna(0.0),
        "atr_norm": (atr / close).fillna(0.0),
        "regime_score": ema200_slope,
        "dist_to_high_atr": dist_to_high.clip(-20, 20),
        "dist_to_low_atr": dist_to_low.clip(-20, 20),
        "ret_5": close.pct_change(5).fillna(0.0).clip(-0.5, 0.5),
    })
    return feats


def normalize(features: np.ndarray, fit_mu: np.ndarray | None = None,
              fit_sigma: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize. Returns (normalized, mu, sigma) so query-time can
    reuse the same parameters."""
    if fit_mu is None:
        mu = features.mean(axis=0)
        sigma = features.std(axis=0)
        sigma[sigma < 1e-9] = 1.0
    else:
        mu, sigma = fit_mu, fit_sigma
    return ((features - mu) / sigma).astype(np.float32), mu, sigma


# ───────────────────────────────────────────────────────────────────────────
# Regime classifier (mirrors src/idea_engine.infer_regime semantics, applied
# bar-by-bar so each historical bar gets a regime label)
# ───────────────────────────────────────────────────────────────────────────

def classify_regime_per_bar(close: pd.Series) -> np.ndarray:
    ema = _ema(close, 200)
    atr_proxy = close.diff().abs().rolling(14).mean().replace(0, np.nan)
    slope = (ema - ema.shift(50)) / (50 * atr_proxy)
    out = np.full(len(close), "warm", dtype="<U4")
    out[(slope >= 0.04).values] = "bull"
    out[(slope <= -0.04).values] = "bear"
    chop_mask = ((slope > -0.04) & (slope < 0.04)).values
    out[chop_mask] = "chop"
    return out


# ───────────────────────────────────────────────────────────────────────────
# Main build
# ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading {PARQUET}")
    df = pd.read_parquet(PARQUET).sort_values("open_time").reset_index(drop=True)
    print(f"  {len(df):,} bars: {df['open_time'].min()} → {df['open_time'].max()}")

    print("Computing features...")
    feats = build_features(df)
    regimes = classify_regime_per_bar(df["close"])

    print(f"Computing {HORIZON_BARS}-bar forward returns...")
    fwd = (df["close"].shift(-HORIZON_BARS) / df["close"] - 1).astype(np.float32)

    # Drop warmup + tail (no forward return for last HORIZON bars).
    valid = (df.index >= WARMUP_BARS) & (df.index < len(df) - HORIZON_BARS)
    valid_arr = valid.to_numpy() if hasattr(valid, "to_numpy") else np.asarray(valid)
    feats_v = feats[valid].to_numpy(dtype=np.float32)
    fwd_v = fwd[valid].to_numpy(dtype=np.float32)
    reg_v = regimes[valid_arr]
    print(f"  valid bars: {len(feats_v):,}")

    print("Normalizing features...")
    feats_norm, mu, sigma = normalize(feats_v)

    print("Building HNSW index...")
    idx = PatternIndex(symbol=SYMBOL, interval=INTERVAL, dim=feats_norm.shape[1])
    idx.build(feats_norm, fwd_v, reg_v)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    idx.save(OUTPUT_DIR)
    # Also save mu/sigma so the analyzer can normalise the live bar identically.
    np.save(OUTPUT_DIR / "mu.npy", mu)
    np.save(OUTPUT_DIR / "sigma.npy", sigma)

    print(f"\n✓ Index saved to {OUTPUT_DIR}")
    print(f"  dim={feats_norm.shape[1]}, n_items={len(feats_norm)}, horizon={HORIZON_BARS}")
    print(f"  regime split: {dict(zip(*np.unique(reg_v, return_counts=True)))}")
    print(f"  fwd return mean={fwd_v.mean()*100:+.3f}%, std={fwd_v.std()*100:.3f}%")
    print(f"  feature columns: {list(feats.columns)}")


if __name__ == "__main__":
    main()
