"""
feature_engine.py — computes 30+ features for ML training.

CRITICAL: formulas must be bit-compatible with Go implementation in
internal/ml/features_v2.go. Any change here requires matching change there.

Feature order is fixed — see FEATURE_NAMES. Go reads features as positional
float64 array in this exact order when loading the model.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ─── Canonical feature order ─────────────────────────────────
# Change = version bump in both sides. Go parses in this order.
FEATURE_NAMES: List[str] = [
    # Momentum / oscillators
    "rsi_7",
    "rsi_14",
    "rsi_21",
    "macd_hist",
    "macd_signal",
    "bb_position",
    "stoch_k_14",
    # Trend
    "ema_cross_20_50",
    "ema_cross_50_200",
    "adx_14",
    "price_vs_ema_50",
    "price_vs_ema_200",
    # Volatility
    "atr_norm_14",
    "bb_width",
    "vol_regime",
    # Volume
    "vol_ratio_5",
    "vol_ratio_20",
    "taker_buy_ratio",
    # Price action (last 5 bars)
    "return_1",
    "return_5",
    "return_20",
    "higher_highs_10",
    "lower_lows_10",
    # Candlestick patterns
    "doji_last",
    "engulfing_last",
    "hammer_last",
    # Cross-asset (placeholder — computed from BTC separately)
    "btc_corr_30",
    "btc_beta_30",
    # Lagged
    "rsi_14_lag_4",
    "return_5_lag_4",
    "vol_ratio_20_lag_4",
]


# ─── Primitive indicators (numpy-only for speed) ─────────────

def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI using EMA-style smoothing. 0-100 range. NaN for warmup."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)

    if len(close) < period + 1:
        return np.full_like(close, 50.0)

    avg_gain[period] = gain[1 : period + 1].mean()
    avg_loss[period] = loss[1 : period + 1].mean()
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - 100 / (1 + rs)
    # Fill warmup with neutral 50.
    np.nan_to_num(rsi, nan=50.0, copy=False)
    return rsi


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Standard EMA with α=2/(period+1). Seeded with first value."""
    if len(values) == 0:
        return values
    k = 2.0 / (period + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def _macd(close: np.ndarray):
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


def _bollinger(close: np.ndarray, period: int = 20, num_std: float = 2.0):
    s = pd.Series(close)
    ma = s.rolling(period, min_periods=1).mean().to_numpy()
    sd = s.rolling(period, min_periods=1).std(ddof=0).to_numpy()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    width = (upper - lower) / (ma + 1e-12)
    width = np.nan_to_num(width, nan=0.0)
    pos = (close - lower) / (upper - lower + 1e-12)
    pos = np.clip(np.nan_to_num(pos, nan=0.5), 0, 1)
    return upper, lower, ma, width, pos


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's ATR — matches Go wilderATR byte-equal (Patch 2N+2 parity fix).

    The canonical Wilder formula (New Concepts in Technical Trading Systems,
    1978) skips TR[0] because it requires prev_close which doesn't exist on
    the first bar. Previously this Python impl faked prev_close[0] = close[0]
    and included TR[0] in the initial mean, which made atr[i] drift from Go's
    (Go skips TR[0] correctly) by ~6e-6 on the last bar of a 60-bar series.

    The drift was surfaced by the Patch 2N parity test. Fixing Python to match
    Go means the next retrain on vast.ai will produce models trained on the
    canonical ATR. Not retraining would leave production still on the old
    formula; the 2H-extended retrain cycle is exactly where we cut over.
    """
    n = len(close)
    atr = np.zeros_like(close)
    if n < period + 1:
        return atr
    # TR from bar 1 onward — needs prev_close[i-1], undefined at i=0.
    tr = np.zeros_like(close)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    # Bootstrap from mean of the first `period` TRs (indices 1..period).
    atr[period] = tr[1 : period + 1].mean()
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _stoch(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    s_high = pd.Series(high).rolling(period, min_periods=1).max().to_numpy()
    s_low = pd.Series(low).rolling(period, min_periods=1).min().to_numpy()
    k = (close - s_low) / (s_high - s_low + 1e-12) * 100
    return np.clip(np.nan_to_num(k, nan=50.0), 0, 100)


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Simplified ADX: trend strength 0-100. Closest classic formula."""
    up = high - np.roll(high, 1)
    down = np.roll(low, 1) - low
    up[0] = 0
    down[0] = 0
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = np.maximum.reduce([high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))])
    tr[0] = high[0] - low[0]

    tr_s = pd.Series(tr).rolling(period, min_periods=1).sum().to_numpy() + 1e-12
    plus_di = 100 * pd.Series(plus_dm).rolling(period, min_periods=1).sum().to_numpy() / tr_s
    minus_di = 100 * pd.Series(minus_dm).rolling(period, min_periods=1).sum().to_numpy() / tr_s
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
    adx = pd.Series(dx).rolling(period, min_periods=1).mean().to_numpy()
    return np.nan_to_num(adx, nan=0.0)


# ─── Candlestick patterns ────────────────────────────────────

def _doji(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    body = np.abs(close - open_)
    full = high - low + 1e-12
    return (body / full < 0.1).astype(np.float32)


def _engulfing(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_body = np.roll(close - open_, 1)
    cur_body = close - open_
    bullish = (prev_body < 0) & (cur_body > 0) & (np.abs(cur_body) > np.abs(prev_body))
    bearish = (prev_body > 0) & (cur_body < 0) & (np.abs(cur_body) > np.abs(prev_body))
    result = np.zeros_like(close, dtype=np.float32)
    result[bullish] = 1.0
    result[bearish] = -1.0
    result[0] = 0.0
    return result


def _hammer(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    body = np.abs(close - open_)
    lower_wick = np.minimum(open_, close) - low
    upper_wick = high - np.maximum(open_, close)
    is_hammer = (lower_wick > 2 * body) & (upper_wick < body)
    return is_hammer.astype(np.float32)


# ─── Main builder ────────────────────────────────────────────

def build_features(df: pd.DataFrame, btc_close: np.ndarray | None = None) -> pd.DataFrame:
    """
    df: columns open, high, low, close, volume, taker_buy_base, open_time.
    btc_close: aligned BTCUSDT close series for cross-asset features. If None
    and symbol == BTC → self-correlation (expected 1).
    Returns DataFrame with FEATURE_NAMES columns + 'close' + 'open_time'.
    """
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    open_ = df["open"].to_numpy()
    vol = df["volume"].to_numpy()
    taker_buy = df["taker_buy_base"].to_numpy() if "taker_buy_base" in df else np.zeros_like(vol)

    # Indicators
    rsi7 = _rsi(close, 7)
    rsi14 = _rsi(close, 14)
    rsi21 = _rsi(close, 21)
    macd_line, macd_signal, macd_hist = _macd(close)
    _, _, ma20, bb_width, bb_pos = _bollinger(close, 20)
    atr14 = _atr(high, low, close, 14)
    stoch_k = _stoch(high, low, close, 14)

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    adx14 = _adx(high, low, close, 14)

    # Volume stats
    vol_s = pd.Series(vol)
    vol5 = vol / (vol_s.rolling(5, min_periods=1).mean().to_numpy() + 1e-12)
    vol20 = vol / (vol_s.rolling(20, min_periods=1).mean().to_numpy() + 1e-12)
    taker_ratio = taker_buy / (vol + 1e-12)

    # Returns. Clip to ±100% to prevent outlier altcoin pumps from overflowing
    # downstream LogReg / matmul operations (observed 50x spikes in SOL/SHIB).
    ret1 = np.diff(close, prepend=close[0]) / (np.roll(close, 1) + 1e-12)
    ret1[0] = 0
    close_s = pd.Series(close)
    ret5 = close_s.pct_change(5, fill_method=None).fillna(0).to_numpy()
    ret20 = close_s.pct_change(20, fill_method=None).fillna(0).to_numpy()
    ret1 = np.clip(ret1, -1.0, 1.0)
    ret5 = np.clip(ret5, -1.0, 1.0)
    ret20 = np.clip(ret20, -1.0, 1.0)

    # Higher highs / lower lows in last 10 bars
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    hh10 = (high_s.rolling(10, min_periods=1).max() == high).astype(np.float32).to_numpy()
    ll10 = (low_s.rolling(10, min_periods=1).min() == low).astype(np.float32).to_numpy()

    # Candlestick patterns
    doji = _doji(open_, high, low, close)
    engulf = _engulfing(open_, close)
    hammer = _hammer(open_, high, low, close)

    # Cross-asset
    if btc_close is None:
        btc_corr = np.ones_like(close)
        btc_beta = np.ones_like(close)
    else:
        min_len = min(len(close), len(btc_close))
        c_s = pd.Series(close[-min_len:])
        b_s = pd.Series(btc_close[-min_len:])
        r1 = c_s.pct_change(fill_method=None)
        r2 = b_s.pct_change(fill_method=None)
        corr30 = r1.rolling(30, min_periods=10).corr(r2).fillna(0).to_numpy()
        var30 = r2.rolling(30, min_periods=10).var().to_numpy() + 1e-12
        cov30 = r1.rolling(30, min_periods=10).cov(r2).fillna(0).to_numpy()
        beta30 = np.clip(cov30 / var30, -3, 3)
        # Pad back to full length
        btc_corr = np.concatenate([np.zeros(len(close) - min_len), corr30])
        btc_beta = np.concatenate([np.ones(len(close) - min_len), beta30])

    # Volatility regime: ATR/price ratio percentile (0-1). Shift by 1 so bar t
    # doesn't rank itself (that would be label leakage via its own ATR value).
    atr_norm = atr14 / (close + 1e-12)
    # Patch 2N+2 parity fix: vol_regime now matches Go rollingPercentile byte-equal.
    # Changes vs the previous pandas-based impl:
    #   1. No shift(1) — use current-included window, same as Go idx.
    #   2. `strictly-less-than` ratio, NOT pandas .rank(pct=True). Pandas
    #      uses average-of-ties which gave ~0.2 divergence from Go's count/n.
    #   3. Warmup: < 20 bars → 0.5 (matches Go's `if idx < 20`).
    # Same O(n²) bound as Go (called once per bar at feature-build time; for
    # the production path that hits Go we only care about the last bar, which
    # is O(window) = O(100)).
    vol_regime = np.full_like(close, 0.5, dtype=float)
    for idx in range(20, len(close)):
        start = max(0, idx - 100 + 1)
        cur = atr_norm[idx]
        window = atr_norm[start : idx + 1]
        less = int((window < cur).sum())
        cnt = len(window)
        vol_regime[idx] = less / cnt if cnt > 0 else 0.5

    # Lagged (4 bars back)
    rsi14_lag4 = np.roll(rsi14, 4)
    rsi14_lag4[:4] = rsi14[:4]
    ret5_lag4 = np.roll(ret5, 4)
    ret5_lag4[:4] = 0
    vol20_lag4 = np.roll(vol20, 4)
    vol20_lag4[:4] = 1.0

    # Assemble in canonical order (FEATURE_NAMES).
    feat = pd.DataFrame({
        "rsi_7": rsi7,
        "rsi_14": rsi14,
        "rsi_21": rsi21,
        "macd_hist": np.tanh(macd_hist / (close + 1e-12) * 100),  # normalized
        "macd_signal": np.tanh(macd_signal / (close + 1e-12) * 100),
        "bb_position": bb_pos,
        "stoch_k_14": stoch_k / 100.0,
        "ema_cross_20_50": np.sign(ema20 - ema50),
        "ema_cross_50_200": np.sign(ema50 - ema200),
        "adx_14": adx14 / 100.0,
        "price_vs_ema_50": (close - ema50) / (ema50 + 1e-12),
        "price_vs_ema_200": (close - ema200) / (ema200 + 1e-12),
        "atr_norm_14": atr_norm,
        "bb_width": bb_width,
        "vol_regime": vol_regime,
        "vol_ratio_5": np.clip(vol5, 0, 20),
        "vol_ratio_20": np.clip(vol20, 0, 20),
        "taker_buy_ratio": taker_ratio,
        "return_1": ret1,
        "return_5": ret5,
        "return_20": ret20,
        "higher_highs_10": hh10,
        "lower_lows_10": ll10,
        "doji_last": doji,
        "engulfing_last": engulf,
        "hammer_last": hammer,
        "btc_corr_30": btc_corr,
        "btc_beta_30": btc_beta,
        "rsi_14_lag_4": rsi14_lag4,
        "return_5_lag_4": ret5_lag4,
        "vol_ratio_20_lag_4": np.clip(vol20_lag4, 0, 20),
    })

    # Ensure exact column order matches FEATURE_NAMES.
    feat = feat[FEATURE_NAMES]

    # Drop warmup NaN/inf rows so model never sees them.
    feat = feat.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    out = feat.copy()
    out["close"] = close
    out["open_time"] = df["open_time"].values
    return out


def make_target(close: np.ndarray, horizon: int, threshold: float = 0.0) -> np.ndarray:
    """
    Binary target: 1 if close[t+horizon] > close[t] * (1 + threshold), else 0.
    Last `horizon` rows get label=-1 (unlabeled, drop before training).
    """
    future = np.roll(close, -horizon)
    ret = (future - close) / (close + 1e-12)
    label = np.where(ret > threshold, 1, 0).astype(np.int8)
    label[-horizon:] = -1  # sentinel
    return label


def make_target_triple_barrier(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    upper_mult: float = 1.5,
    lower_mult: float = 1.0,
) -> np.ndarray:
    """
    Triple-barrier labels (López de Prado):
        1  — price hit +upper_mult*ATR before -lower_mult*ATR (long wins)
        0  — first hit -lower_mult*ATR before +upper_mult*ATR (long loses)
       -1  — timeout with no barrier touch (noise — drop before training)

    Removes the ~50% of bars that live in chop, raising class separation
    and letting HC-filtering extract clean conviction signals.
    """
    n = len(close)
    labels = np.full(n, -1, dtype=np.int8)
    for t in range(n):
        end = t + horizon
        if end >= n:
            continue
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        up = close[t] + upper_mult * a
        dn = close[t] - lower_mult * a
        for j in range(t + 1, end + 1):
            if high[j] >= up:
                labels[t] = 1
                break
            if low[j] <= dn:
                labels[t] = 0
                break
    return labels
