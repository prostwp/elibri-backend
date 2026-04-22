"""fetch_macro_features.py — Stage 2 macro features fetcher (scaffold).

Downloads 4 macro signals for BTC with 8-year history:
  1. Binance Futures Funding Rate (per 8h, historical via /fapi/v1/fundingRate)
  2. Binance Futures Open Interest (/futures/data/openInterestHist, daily)
  3. BTC Dominance (CoinGecko /global endpoint, requires aggregation)
  4. Fear & Greed Index (alternative.me /fng/?limit=0)

Saves to data/macro/*.parquet aligned to 15m/1h/4h bar grids via
forward-fill (macro signals update at coarser cadence than OHLCV bars).

Usage:
  python3 fetch_macro_features.py --backfill-days 2920  # 8 years
  python3 fetch_macro_features.py --refresh              # last 7 days only

Output layout:
  data/macro/funding_rate_BTCUSDT.parquet  [bar_open_time, funding_rate, mark_price]
  data/macro/open_interest_BTCUSDT.parquet [bar_open_time, open_interest_usd]
  data/macro/btc_dominance.parquet          [bar_open_time, dominance_pct]
  data/macro/fear_greed.parquet             [bar_open_time, fng_value, fng_class]

This is a SCAFFOLD. Runs but saves mock data today. Stage 2 implementation
(TBD 2026-04-23) fills actual HTTP fetchers.

TODO for Stage 2:
  [ ] Binance Futures funding rate endpoint (limit=1000, pagination loop)
  [ ] Binance OI endpoint (daily resolution)
  [ ] CoinGecko global endpoint + cache (rate limit 10-50/min free tier)
  [ ] alternative.me F&G (single call, no auth)
  [ ] Merge to feature_engine.build_features as features #46-#53
  [ ] Update FEATURE_NAMES to 53 entries
  [ ] Go parity features_v2.go (if re-activating Go backend for Stage 2)
"""
import argparse
import json
import pathlib
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
DATA_DIR = ROOT / "data" / "macro"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def mock_funding_rate(start: datetime, end: datetime) -> pd.DataFrame:
    """Placeholder: 8h-spaced rows with mean-reverting synthetic funding."""
    ts = pd.date_range(start, end, freq="8H", tz="UTC")
    # Random walk around 0.0001 (1 bps), clip to [-0.003, 0.003].
    np.random.seed(42)
    base = np.cumsum(np.random.normal(0, 0.0001, len(ts))) * 0.1 + 0.0001
    base = np.clip(base, -0.003, 0.003)
    return pd.DataFrame({"bar_open_time": ts, "funding_rate": base})


def mock_open_interest(start: datetime, end: datetime) -> pd.DataFrame:
    """Placeholder: daily OI values, 1-50bn USD range with trend."""
    ts = pd.date_range(start, end, freq="1D", tz="UTC")
    np.random.seed(43)
    # Growth from 1bn to 30bn with noise.
    growth = np.linspace(1e9, 3e10, len(ts)) + np.random.normal(0, 5e8, len(ts))
    return pd.DataFrame({"bar_open_time": ts, "open_interest_usd": np.clip(growth, 5e8, 5e10)})


def mock_dominance(start: datetime, end: datetime) -> pd.DataFrame:
    """Placeholder: daily BTC dominance, 40-70% range."""
    ts = pd.date_range(start, end, freq="1D", tz="UTC")
    np.random.seed(44)
    dom = 50 + np.cumsum(np.random.normal(0, 0.3, len(ts))) * 0.5
    return pd.DataFrame({"bar_open_time": ts, "dominance_pct": np.clip(dom, 35, 72)})


def mock_fear_greed(start: datetime, end: datetime) -> pd.DataFrame:
    """Placeholder: daily F&G 0-100."""
    ts = pd.date_range(start, end, freq="1D", tz="UTC")
    np.random.seed(45)
    fng = 50 + np.cumsum(np.random.normal(0, 5, len(ts)))
    fng = np.clip(fng, 0, 100)

    def classify(v):
        if v < 20:
            return "extreme_fear"
        if v < 40:
            return "fear"
        if v < 60:
            return "neutral"
        if v < 80:
            return "greed"
        return "extreme_greed"

    return pd.DataFrame({
        "bar_open_time": ts,
        "fng_value": fng,
        "fng_class": [classify(v) for v in fng],
    })


def align_to_bar_grid(df_macro: pd.DataFrame, bar_ts: pd.DatetimeIndex, col: str) -> np.ndarray:
    """Forward-fill macro signal at cadence C to finer bar grid.

    Given macro df with rows at some coarser cadence (8h / 1d), and a target
    bar_ts grid (15m / 1h / 4h), returns array of length len(bar_ts) where
    each bar uses the most recent macro value AT OR BEFORE its open_time.
    """
    merged = pd.merge_asof(
        pd.DataFrame({"bar_open_time": bar_ts}),
        df_macro[["bar_open_time", col]].sort_values("bar_open_time"),
        on="bar_open_time", direction="backward",
    )
    return merged[col].fillna(method="ffill").fillna(0.0).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill-days", type=int, default=2920,
                    help="Days of history to fetch (default 2920 = 8 years)")
    ap.add_argument("--refresh", action="store_true",
                    help="Only fetch last 7 days (append to existing parquet)")
    ap.add_argument("--mock", action="store_true",
                    help="Use synthetic data (for Stage 2 scaffold testing)")
    args = ap.parse_args()

    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if args.refresh:
        start = end - timedelta(days=7)
    else:
        start = end - timedelta(days=args.backfill_days)

    print(f"Fetch window: {start.date()} → {end.date()}")
    print(f"Output dir: {DATA_DIR}")

    if not args.mock:
        print("WARN: real fetchers not implemented (Stage 2 TBD 2026-04-23). "
              "Falling back to --mock. Use --mock explicitly to suppress this warning.")

    print("\n1/4 funding_rate...")
    fr = mock_funding_rate(start, end)
    fr.to_parquet(DATA_DIR / "funding_rate_BTCUSDT.parquet")
    print(f"  {len(fr)} rows, range {fr['funding_rate'].min():.4f} → {fr['funding_rate'].max():.4f}")

    print("\n2/4 open_interest...")
    oi = mock_open_interest(start, end)
    oi.to_parquet(DATA_DIR / "open_interest_BTCUSDT.parquet")
    print(f"  {len(oi)} rows, range ${oi['open_interest_usd'].min()/1e9:.1f}B → ${oi['open_interest_usd'].max()/1e9:.1f}B")

    print("\n3/4 btc_dominance...")
    dom = mock_dominance(start, end)
    dom.to_parquet(DATA_DIR / "btc_dominance.parquet")
    print(f"  {len(dom)} rows, range {dom['dominance_pct'].min():.1f}% → {dom['dominance_pct'].max():.1f}%")

    print("\n4/4 fear_greed...")
    fng = mock_fear_greed(start, end)
    fng.to_parquet(DATA_DIR / "fear_greed.parquet")
    print(f"  {len(fng)} rows, range {fng['fng_value'].min():.0f} → {fng['fng_value'].max():.0f}")

    print("\nDone. Scaffold data saved. Stage 2 will replace mock_* with real fetchers.")


if __name__ == "__main__":
    main()
