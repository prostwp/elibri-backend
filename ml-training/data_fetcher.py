"""
data_fetcher.py — downloads historical klines from Binance and caches to parquet.
Public endpoint, no API key required. Paginated with 1000-candle chunks.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

BINANCE_URL = "https://api.binance.com/api/v3/klines"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Interval → milliseconds (for window stepping).
INTERVAL_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}

# Top-20 USDT pairs by volume (24 Apr 2026 snapshot, stable over years).
TOP_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "DOTUSDT",
    "LINKUSDT", "MATICUSDT", "SHIBUSDT", "LTCUSDT", "BCHUSDT",
    "UNIUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT", "SUIUSDT",
]


_SYMBOL_RE = __import__("re").compile(r"^[A-Z0-9]{2,12}$")
_INTERVAL_RE = __import__("re").compile(r"^[0-9]+[smhdw]$")


def _cache_path(symbol: str, interval: str) -> Path:
    # Defense-in-depth path-traversal guard: even when callers do their own
    # validation, _cache_path is the single chokepoint that converts
    # user-controlled strings to filesystem paths.
    if not _SYMBOL_RE.fullmatch(symbol):
        raise ValueError(f"invalid symbol {symbol!r}; must match {_SYMBOL_RE.pattern}")
    if not _INTERVAL_RE.fullmatch(interval):
        raise ValueError(f"invalid interval {interval!r}; must match {_INTERVAL_RE.pattern}")
    return DATA_DIR / f"{symbol}_{interval}.parquet"


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Pull klines with pagination. Returns canonical OHLCV DataFrame.

    Key behaviour: when a batch is empty (common for pre-listing start dates),
    we advance `cur` by the window size and keep trying rather than bailing out.
    Only give up after 5 consecutive empty batches (e.g. SOL queried from 2018
    has ~28 months of pre-listing empty windows before data starts 2020-08).
    """
    step = INTERVAL_MS[interval] * 1000  # 1000 candles per request
    rows: list[list] = []
    cur = start_ms
    consecutive_empty = 0
    while cur < end_ms:
        nxt = min(cur + step, end_ms)
        r = requests.get(
            BINANCE_URL,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cur,
                "endTime": nxt,
                "limit": 1000,
            },
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            consecutive_empty += 1
            # Walk forward by the window size; skip this gap.
            cur = nxt
            if consecutive_empty >= 20:
                # 20 empty windows = 2+ years of nothing → truly no data.
                break
            time.sleep(0.15)
            continue
        consecutive_empty = 0
        rows.extend(batch)
        last_ts = batch[-1][0]
        cur = last_ts + INTERVAL_MS[interval]
        time.sleep(0.15)  # respect public rate limit ~6k/min

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "num_trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df[["open_time", "open", "high", "low", "close", "volume", "taker_buy_base", "num_trades"]]


def _validate_ohlcv(df: pd.DataFrame, label: str) -> None:
    """Defensive checks on a freshly-loaded OHLCV DataFrame.

    Catches the corruption modes flagged in Round-3 security review:
    duplicate or out-of-order timestamps, NaN/zero prices, negative volumes,
    high < low, future-dated rows (clock skew).
    Raises with a specific message on any violation; does not "fix" silently.
    """
    if len(df) == 0:
        return  # empty handled by caller
    if df["open_time"].duplicated().any():
        n = int(df["open_time"].duplicated().sum())
        raise ValueError(f"{label}: {n} duplicate open_time rows")
    if not df["open_time"].is_monotonic_increasing:
        raise ValueError(f"{label}: open_time not monotonically increasing")
    if not (df[["open", "high", "low", "close"]] > 0).all().all():
        raise ValueError(f"{label}: non-positive OHLC values")
    if not df["high"].ge(df["low"]).all():
        raise ValueError(f"{label}: high < low on some bars")
    if (df["volume"] < 0).any():
        n = int((df["volume"] < 0).sum())
        raise ValueError(f"{label}: {n} negative volume rows")
    now_utc = pd.Timestamp.now(tz="UTC")
    if (df["open_time"] > now_utc + pd.Timedelta(minutes=5)).any():
        raise ValueError(f"{label}: future-dated rows (clock skew or feed bug)")


def fetch_or_cache(symbol: str, interval: str, years: float = 2.0, force: bool = False) -> pd.DataFrame:
    p = _cache_path(symbol, interval)
    if p.exists() and not force:
        df = pd.read_parquet(p)
        _validate_ohlcv(df, f"cached {p.name}")
        return df

    end = datetime.now(tz=timezone.utc)
    start = end.timestamp() - years * 365 * 24 * 3600
    df = fetch_klines(symbol, interval, int(start * 1000), int(end.timestamp() * 1000))
    if len(df) > 0:
        _validate_ohlcv(df, f"freshly-fetched {symbol}_{interval}")
        # Atomic write: tmp file + rename so a crash mid-write doesn't leave
        # a half-baked parquet that subsequent reads silently use.
        tmp = p.with_suffix(p.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)  # atomic on POSIX
    return df


def fetch_all(pairs: Iterable[str] = None, intervals: Iterable[str] = ("1h", "4h", "1d"), years: float = 2.0):
    pairs = list(pairs) if pairs else TOP_PAIRS
    for sym in pairs:
        for iv in intervals:
            cached = _cache_path(sym, iv).exists()
            try:
                df = fetch_or_cache(sym, iv, years=years)
                print(f"  {sym:<10} {iv:<4} {'cached' if cached else 'fetched':<8} rows={len(df):>6}")
            except Exception as e:
                print(f"  {sym:<10} {iv:<4} FAILED: {e}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=None, help="Symbols (default top-20)")
    ap.add_argument("--intervals", nargs="*", default=["1h", "4h", "1d"])
    ap.add_argument("--years", type=float, default=2.0)
    args = ap.parse_args()
    fetch_all(args.pairs, args.intervals, years=args.years)
