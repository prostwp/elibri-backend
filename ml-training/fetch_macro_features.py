"""fetch_macro_features.py — Stage 2 macro features (REAL data, no mocks).

Replaces the prior scaffold (which saved np.random.normal data with seed=42).
All fetchers hit live public endpoints. No API keys required.

Sources & coverage (verified 2026-04-26):
  funding_rate    Binance Futures /fapi/v1/fundingRate
                  BTC: 2019-09-10 → now (8h cadence). ETH: 2019-11-27 → now.
                  Pre-2019 bars → 0.0 (neutral, no perpetual futures existed).

  open_interest   Binance Futures /futures/data/openInterestHist
                  ~30-DAY HARD LIMIT on free API. Older startTime returns
                  HTTP error code -1130 "parameter 'startTime' is invalid".
                  Pre-coverage bars → 0.0. Marked as KNOWN LIMITATION.
                  Forward-going paper trade gets real OI; backtest training
                  effectively trains without OI on >30d-old bars (model learns
                  to ignore the column on those rows).

  fear_greed      alternative.me /fng/?limit=0
                  2018-02-01 → now, daily. ~3000 records, no pagination.

  btc_dominance   CoinGecko /global (current snapshot only on free tier).
                  History requires paid CMC/CoinGecko Pro. We capture the
                  current point and write a single-row parquet. Marked as
                  KNOWN LIMITATION; feature_engine treats absence as 0.

Output layout (all parquet):
  data/macro/funding_rate_BTCUSDT.parquet  [bar_open_time, funding_rate, mark_price]
  data/macro/funding_rate_ETHUSDT.parquet  (same schema for cross-asset)
  data/macro/open_interest_BTCUSDT.parquet [bar_open_time, oi_contracts, oi_value_usd]
  data/macro/fear_greed.parquet            [bar_open_time, fng_value, fng_class_int]
  data/macro/btc_dominance.parquet         [bar_open_time, dominance_pct]  (single row)

Usage:
  python fetch_macro_features.py                # full backfill
  python fetch_macro_features.py --refresh      # last 7 days only (incremental)
  python fetch_macro_features.py --symbol BTCUSDT --only funding,fng

After running, use feature_engine.attach_macro_features(...) to merge into
the FEATURE_NAMES vector (added in same patch).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).parent
DATA_DIR = ROOT / "data" / "macro"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_FAPI = "https://fapi.binance.com"
ALTME_FNG = "https://api.alternative.me/fng/"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"

# Hard limit on Binance OI history (verified 2026-04-26: 35d startTime is rejected).
OI_MAX_DAYS = 30

# Polite delays to avoid rate limits (Binance: 2400 req/min weight; we use ~2 req/s
# to stay well under, plus retries on transient connection resets).
SLEEP_BETWEEN_PAGES = 0.5
MAX_RETRIES = 5


def _request_with_retry(url: str, params: dict, timeout: int = 20):
    """Wrapper around requests.get with exponential backoff for transient
    network errors (Connection reset by peer, ReadTimeout, 5xx). Raises after
    MAX_RETRIES failed attempts.

    Background: macOS Python on LibreSSL hits intermittent ConnectionResetError
    against Binance during long paginated runs. Bare `requests.get` blows up
    after ~50 successful pages.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (418, 429):
                # Binance rate-limit / IP ban warning: back off hard.
                wait = 30 * (attempt + 1)
                print(f"    rate-limited ({r.status_code}); sleeping {wait}s")
                time.sleep(wait)
                continue
            return r
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last_exc = e
            wait = 2 ** attempt  # 1, 2, 4, 8, 16
            print(f"    transient {type(e).__name__} (attempt {attempt + 1}/{MAX_RETRIES}); "
                  f"retry in {wait}s")
            time.sleep(wait)
    raise last_exc if last_exc else RuntimeError("Request failed without exception")


# ─── Funding rate (Binance Futures) ──────────────────────────────────────────

def fetch_funding_rate(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch full funding-rate history paginated by startTime.

    Binance returns max 1000 rows per call ordered chronologically. We page
    forward using the last fundingTime + 1ms as next startTime until we cover
    [start_ms, end_ms].

    Returns DataFrame columns:
        bar_open_time : datetime64[ns, UTC]
        funding_rate  : float (e.g. 0.0001 = 1bp per 8h)
        mark_price    : float (NaN for very old rows where Binance returns "")
    """
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    rows: List[dict] = []
    cursor = start_ms
    pages = 0
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        r = _request_with_retry(url, params, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last_time = batch[-1]["fundingTime"]
        if last_time <= cursor:
            break
        cursor = last_time + 1
        pages += 1
        time.sleep(SLEEP_BETWEEN_PAGES)
        if pages > 200:
            # Sanity guard: 200 pages × 1000 rows × 8h = 27 years coverage,
            # well past the universe of perpetual swap history.
            break

    if not rows:
        return pd.DataFrame(columns=["bar_open_time", "funding_rate", "mark_price"])

    df = pd.DataFrame(rows)
    df["bar_open_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    # Pre-2020 records have mark_price="" because Binance didn't store it;
    # convert to NaN so downstream can drop or impute.
    df["mark_price"] = pd.to_numeric(df["markPrice"], errors="coerce")
    df = df[["bar_open_time", "funding_rate", "mark_price"]].drop_duplicates(
        subset=["bar_open_time"]
    ).sort_values("bar_open_time").reset_index(drop=True)
    return df


# ─── Open Interest (Binance Futures, ~30d limit) ─────────────────────────────

def fetch_open_interest(
    symbol: str, period: str, start_ms: int, end_ms: int
) -> pd.DataFrame:
    """Fetch OI history, capped at 30 days backward from now.

    The free Binance Futures API rejects startTime older than ~30 days with
    error -1130 ("parameter 'startTime' is invalid"). We clamp `start_ms` to
    `now - 30d` and document this as a known limitation.

    period choices: "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d".
    """
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    floor = now_ms - OI_MAX_DAYS * 86_400_000 + 60_000  # +60s safety margin
    effective_start = max(start_ms, floor)
    if effective_start >= end_ms:
        return pd.DataFrame(columns=["bar_open_time", "oi_contracts", "oi_value_usd"])

    url = f"{BINANCE_FAPI}/futures/data/openInterestHist"
    rows: List[dict] = []
    cursor = effective_start
    pages = 0
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 500,
        }
        r = _request_with_retry(url, params, timeout=20)
        if r.status_code != 200:
            # Most likely cause: startTime drifted past the 30d window during
            # pagination. Log the body before stopping so silent truncation
            # of OI history doesn't go unnoticed (Round-6 security finding).
            print(f"    OI fetch stopped at status {r.status_code}: "
                  f"{r.text[:200]!r} — partial history kept, expect "
                  f"limited OI coverage downstream.")
            break
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last_time = batch[-1]["timestamp"]
        if last_time <= cursor:
            break
        cursor = last_time + 1
        pages += 1
        time.sleep(SLEEP_BETWEEN_PAGES)
        if pages > 30:
            break

    if not rows:
        return pd.DataFrame(columns=["bar_open_time", "oi_contracts", "oi_value_usd"])

    df = pd.DataFrame(rows)
    df["bar_open_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["oi_contracts"] = df["sumOpenInterest"].astype(float)
    df["oi_value_usd"] = df["sumOpenInterestValue"].astype(float)
    return df[["bar_open_time", "oi_contracts", "oi_value_usd"]].drop_duplicates(
        subset=["bar_open_time"]
    ).sort_values("bar_open_time").reset_index(drop=True)


# ─── Fear & Greed Index (alternative.me) ─────────────────────────────────────

_FNG_CLASS_TO_INT = {
    "Extreme Fear": 0,
    "Fear": 1,
    "Neutral": 2,
    "Greed": 3,
    "Extreme Greed": 4,
}


def fetch_fear_greed() -> pd.DataFrame:
    """Single call returns full daily history back to 2018-02-01."""
    r = requests.get(ALTME_FNG, params={"limit": 0}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame(columns=["bar_open_time", "fng_value", "fng_class_int"])
    df = pd.DataFrame(rows)
    # Default `timestamp` is unix seconds (string). With date_format=us it'd be
    # "MM-DD-YYYY"; we prefer epoch.
    # to_numeric/coerce defends against rare null entries from alternative.me
    # (Round-6 security finding). NaN rows get dropped before parquet write.
    df["bar_open_time"] = pd.to_datetime(
        pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True
    )
    df["fng_value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["bar_open_time", "fng_value"])
    df["fng_value"] = df["fng_value"].astype(int)
    df["fng_class_int"] = df["value_classification"].map(_FNG_CLASS_TO_INT).fillna(2).astype(int)
    return df[["bar_open_time", "fng_value", "fng_class_int"]].sort_values(
        "bar_open_time"
    ).reset_index(drop=True)


# ─── BTC Dominance (CoinGecko, current snapshot only) ────────────────────────

def fetch_btc_dominance_snapshot() -> pd.DataFrame:
    """One row, current dominance %. No history on free tier.

    Production caveat: this gives the live model a known-fresh dominance value
    but in training it acts as a constant. Effective contribution to the model
    is therefore zero unless we backfill. Tracked as Stage 3 work (paid API).
    """
    r = requests.get(COINGECKO_GLOBAL, timeout=15)
    r.raise_for_status()
    data = r.json().get("data", {})
    btc_d = data.get("market_cap_percentage", {}).get("btc")
    if btc_d is None:
        return pd.DataFrame(columns=["bar_open_time", "dominance_pct"])
    return pd.DataFrame({
        "bar_open_time": [pd.Timestamp.now(tz="UTC").floor("h")],
        "dominance_pct": [float(btc_d)],
    })


# ─── Alignment helper (used by feature_engine) ───────────────────────────────

def align_to_bars(
    macro_df: pd.DataFrame,
    bar_open_times: pd.DatetimeIndex,
    value_col: str,
    pre_coverage_value: float = 0.0,
) -> pd.Series:
    """Forward-fill macro[value_col] onto a finer bar grid via merge_asof.

    Each bar in `bar_open_times` receives the most recent macro value at or
    before its timestamp. Bars predating the macro coverage receive
    `pre_coverage_value` (default 0.0 = neutral).

    `bar_open_times` MUST be sorted ascending and timezone-aware UTC.
    """
    if macro_df.empty:
        return pd.Series([pre_coverage_value] * len(bar_open_times), index=bar_open_times)
    left = pd.DataFrame({"bar_open_time": bar_open_times})
    right = macro_df[["bar_open_time", value_col]].sort_values("bar_open_time")
    merged = pd.merge_asof(left, right, on="bar_open_time", direction="backward")
    return merged[value_col].fillna(pre_coverage_value).reset_index(drop=True)


# ─── CLI / orchestration ─────────────────────────────────────────────────────

def _save(df: pd.DataFrame, path: pathlib.Path, label: str) -> None:
    if df.empty:
        print(f"  {label}: NO ROWS — skipping write")
        return
    # Atomic write — SIGINT mid-write would otherwise leave a corrupt parquet
    # which the existence-check skip logic would silently reuse on next run
    # (Round-8 reviewer find).
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    rng = f"{df['bar_open_time'].min()} -> {df['bar_open_time'].max()}"
    print(f"  {label}: {len(df):,} rows, {rng}")
    print(f"     -> {path.relative_to(ROOT)}")


_SYMBOL_RE = __import__("re").compile(r"^[A-Z0-9]{2,12}$")


def _validate_symbol(s: str) -> str:
    if not _SYMBOL_RE.fullmatch(s):
        raise argparse.ArgumentTypeError(
            f"--symbol must match {_SYMBOL_RE.pattern}; got {s!r}"
        )
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT", type=_validate_symbol,
                    help="Primary symbol for funding/OI (default BTCUSDT)")
    ap.add_argument("--also-eth", action="store_true",
                    help="Additionally fetch ETHUSDT funding/OI (cross-asset)")
    ap.add_argument("--refresh", action="store_true",
                    help="Last 7 days only (incremental update)")
    ap.add_argument("--start", default="2018-01-01",
                    help="Backfill start date YYYY-MM-DD (default 2018-01-01)")
    ap.add_argument("--only", default="all",
                    help="Comma list: funding,oi,fng,dominance (default all)")
    args = ap.parse_args()

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if args.refresh:
        start = end - pd.Timedelta(days=7)
    else:
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    targets = {t.strip() for t in args.only.split(",")} if args.only != "all" else {
        "funding", "oi", "fng", "dominance"
    }

    print(f"Fetch window: {start.isoformat()} → {end.isoformat()}")
    print(f"Output dir:   {DATA_DIR}")
    print(f"Targets:      {sorted(targets)}")
    print()

    summary = {"start": start.isoformat(), "end": end.isoformat(),
               "fetched_at": datetime.now(timezone.utc).isoformat(),
               "results": {}}

    def _existing(path: pathlib.Path) -> Optional[pd.DataFrame]:
        """Idempotent skip: if --refresh isn't set and the file already exists
        with rows, reuse it for the manifest/summary instead of re-fetching."""
        if args.refresh:
            return None
        if path.exists():
            df_old = pd.read_parquet(path)
            if not df_old.empty:
                print(f"  (skip; {path.name} already has {len(df_old):,} rows. "
                      f"Use --refresh to redownload)")
                return df_old
        return None

    def _fetch_or_reuse(path: pathlib.Path, fetcher, label: str) -> pd.DataFrame:
        cached = _existing(path)
        if cached is not None:
            return cached
        df = fetcher()
        _save(df, path, label)
        return df

    if "funding" in targets:
        print(f"[1] funding_rate {args.symbol}")
        df_fr = _fetch_or_reuse(
            DATA_DIR / f"funding_rate_{args.symbol}.parquet",
            lambda: fetch_funding_rate(args.symbol, start_ms, end_ms),
            f"funding_rate_{args.symbol}",
        )
        summary["results"][f"funding_rate_{args.symbol}"] = {
            "rows": len(df_fr),
            "first": str(df_fr["bar_open_time"].min()) if len(df_fr) else None,
            "last": str(df_fr["bar_open_time"].max()) if len(df_fr) else None,
        }
        if args.also_eth:
            print(f"[1b] funding_rate ETHUSDT")
            df_eth = _fetch_or_reuse(
                DATA_DIR / "funding_rate_ETHUSDT.parquet",
                lambda: fetch_funding_rate("ETHUSDT", start_ms, end_ms),
                "funding_rate_ETHUSDT",
            )
            summary["results"]["funding_rate_ETHUSDT"] = {
                "rows": len(df_eth),
                "first": str(df_eth["bar_open_time"].min()) if len(df_eth) else None,
                "last": str(df_eth["bar_open_time"].max()) if len(df_eth) else None,
            }

    if "oi" in targets:
        print(f"\n[2] open_interest {args.symbol} (period=4h, last ≤30d)")
        df_oi = _fetch_or_reuse(
            DATA_DIR / f"open_interest_{args.symbol}.parquet",
            lambda: fetch_open_interest(args.symbol, "4h", start_ms, end_ms),
            f"open_interest_{args.symbol}",
        )
        summary["results"][f"open_interest_{args.symbol}"] = {
            "rows": len(df_oi),
            "limitation": f"Binance free API caps OI history to ~{OI_MAX_DAYS} days. "
                          f"Pre-coverage bars get 0 in alignment.",
            "first": str(df_oi["bar_open_time"].min()) if len(df_oi) else None,
            "last": str(df_oi["bar_open_time"].max()) if len(df_oi) else None,
        }
        if args.also_eth:
            print(f"\n[2b] open_interest ETHUSDT (period=4h, last ≤30d)")
            df_eth_oi = fetch_open_interest("ETHUSDT", "4h", start_ms, end_ms)
            _save(df_eth_oi, DATA_DIR / "open_interest_ETHUSDT.parquet",
                  "open_interest_ETHUSDT")
            summary["results"]["open_interest_ETHUSDT"] = {
                "rows": len(df_eth_oi),
                "limitation": f"Binance free API caps OI history to ~{OI_MAX_DAYS} days.",
                "first": str(df_eth_oi["bar_open_time"].min()) if len(df_eth_oi) else None,
                "last": str(df_eth_oi["bar_open_time"].max()) if len(df_eth_oi) else None,
            }

    if "fng" in targets:
        print(f"\n[3] fear_greed (alternative.me)")
        df_fng = fetch_fear_greed()
        _save(df_fng, DATA_DIR / "fear_greed.parquet", "fear_greed")
        summary["results"]["fear_greed"] = {
            "rows": len(df_fng),
            "first": str(df_fng["bar_open_time"].min()) if len(df_fng) else None,
            "last": str(df_fng["bar_open_time"].max()) if len(df_fng) else None,
        }

    if "dominance" in targets:
        print(f"\n[4] btc_dominance (CoinGecko snapshot)")
        df_dom = fetch_btc_dominance_snapshot()
        _save(df_dom, DATA_DIR / "btc_dominance.parquet", "btc_dominance")
        summary["results"]["btc_dominance"] = {
            "rows": len(df_dom),
            "limitation": "CoinGecko free tier only exposes current snapshot. "
                          "Historical BTC.D needs paid CMC/CG-Pro. Treated as "
                          "constant in training; effectively zero contribution.",
            "current_value": float(df_dom["dominance_pct"].iloc[0]) if len(df_dom) else None,
        }

    # Manifest with provenance — required by the 'no mocks' policy so future
    # readers can audit what was fetched and when.
    manifest_path = DATA_DIR / "manifest.json"
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    with open(tmp_manifest, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    tmp_manifest.replace(manifest_path)
    print(f"\nManifest → {manifest_path.relative_to(ROOT)}")
    print("\nDone. All data is real. No mocks. Coverage limitations documented in manifest.")


if __name__ == "__main__":
    main()
