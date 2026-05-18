"""HNSW-based pattern matcher — finds historical bars whose features are
nearest to the current bar, reports forward-return distribution.

This is the Phase-6 enrichment over rule-based `idea_engine`. It does NOT
predict direction (the rule-based engine still picks long/short and writes
the trade idea); it only adds historical CONTEXT to the post:

    "87 похожих setup-ов в 2021-2023, win rate 62%, средний +1.8% за 12h."

Built artifacts live on disk (one per symbol×interval) and are loaded
lazily by the bot. Build script: scripts/build_btc_4h_pattern_index.py
(re-run when adding new history; bot reads stale snapshots fine).

Per `feedback_ml_pipeline_safety.md`:
- Tests in tests/test_pattern_match.py written FIRST.
- < 500 LOC; pure numpy + hnswlib (no pandas at runtime).
- Defensive: if index file missing, returns "no data" summary, doesn't crash.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import hnswlib
import numpy as np

Direction = Literal["long", "short"]


# ─── Index data structure ────────────────────────────────────────────────────

class PatternIndex:
    """HNSW + companion arrays.

    Files layout under `path` (a directory):
      hnsw.bin       — binary HNSW index (hnswlib native format)
      features.npy   — float32, shape (n, dim) — the bar feature vectors
      forward.npy    — float32, shape (n,)   — forward return per bar
      regimes.npy    — '<U4', shape (n,)     — regime label per bar
      meta.json      — {symbol, interval, dim, n_items, horizon_bars, built_at}
    """

    def __init__(self, symbol: str, interval: str, dim: int):
        self.symbol = symbol
        self.interval = interval
        self.dim = dim
        self.n_items = 0
        self._hnsw: hnswlib.Index | None = None
        self.features: np.ndarray | None = None
        self.forward_returns: np.ndarray | None = None
        self.regimes: np.ndarray | None = None

    def build(self, features: np.ndarray, fwd: np.ndarray, regimes: np.ndarray,
              ef_construction: int = 200, M: int = 16) -> None:
        # Bug #9 fix: chained comparison `a != b != c` short-circuits and
        # may return False even with mismatched lengths. Set membership is
        # the unambiguous test.
        if len({features.shape[0], fwd.shape[0], regimes.shape[0]}) > 1:
            raise ValueError(
                f"features/fwd/regimes length mismatch: "
                f"{features.shape[0]}/{fwd.shape[0]}/{regimes.shape[0]}"
            )
        n, dim = features.shape
        self.n_items = n
        self.features = features.astype(np.float32, copy=False)
        self.forward_returns = fwd.astype(np.float32, copy=False)
        self.regimes = regimes.astype("<U4", copy=False)

        index = hnswlib.Index(space="l2", dim=dim)
        index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
        index.add_items(self.features, np.arange(n))
        index.set_ef(min(200, n))
        self._hnsw = index

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if self._hnsw is None or self.features is None:
            raise RuntimeError("nothing to save — call build() first")
        self._hnsw.save_index(str(p / "hnsw.bin"))
        np.save(p / "features.npy", self.features)
        np.save(p / "forward.npy", self.forward_returns)
        np.save(p / "regimes.npy", self.regimes)
        with open(p / "meta.json", "w") as f:
            json.dump({
                "symbol": self.symbol, "interval": self.interval,
                "dim": self.dim, "n_items": self.n_items,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PatternIndex":
        p = Path(path)
        with open(p / "meta.json") as f:
            meta = json.load(f)
        idx = cls(symbol=meta["symbol"], interval=meta["interval"], dim=meta["dim"])
        idx.n_items = int(meta["n_items"])
        idx.features = np.load(p / "features.npy")
        idx.forward_returns = np.load(p / "forward.npy")
        idx.regimes = np.load(p / "regimes.npy")
        index = hnswlib.Index(space="l2", dim=idx.dim)
        index.load_index(str(p / "hnsw.bin"), max_elements=idx.n_items)
        index.set_ef(min(200, idx.n_items))
        idx._hnsw = index
        return idx

    def query(self, vec: np.ndarray, top_k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Returns (indices, distances) sorted by ascending L2 distance."""
        if self._hnsw is None or self.n_items == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        k = min(top_k, self.n_items)
        labels, dists = self._hnsw.knn_query(vec.astype(np.float32), k=k)
        return labels[0], dists[0]


# ─── Aggregation ─────────────────────────────────────────────────────────────

@dataclass
class PatternSummary:
    n_matches: int
    win_rate: float                    # 0..1
    mean_return: float                 # decimal (e.g. 0.018 = +1.8%)
    median_return: float
    max_drawdown: float                # min forward return (most negative)
    regime_counts: dict[str, int] = field(default_factory=dict)


def aggregate(fwd_returns: np.ndarray, regimes: np.ndarray,
              direction: Direction) -> PatternSummary:
    """Compute summary stats over the K nearest historical setups.

    Win for `long` = forward return > 0; win for `short` = forward return < 0.
    """
    n = len(fwd_returns)
    if n == 0:
        return PatternSummary(0, 0.0, 0.0, 0.0, 0.0, {})

    if direction == "long":
        wins = (fwd_returns > 0).sum()
        # For long: drawdown = worst forward return (negative)
        dd = float(fwd_returns.min())
    else:
        wins = (fwd_returns < 0).sum()
        # For short: profit signs flip — "drawdown" is most positive (worst case)
        dd = float(fwd_returns.max())

    regime_counts: dict[str, int] = {}
    for r in regimes.tolist():
        regime_counts[str(r)] = regime_counts.get(str(r), 0) + 1

    return PatternSummary(
        n_matches=int(n),
        win_rate=float(wins / n),
        mean_return=float(fwd_returns.mean()),
        median_return=float(np.median(fwd_returns)),
        max_drawdown=dd,
        regime_counts=regime_counts,
    )


# ─── Text rendering ──────────────────────────────────────────────────────────

def format_summary_text(s: PatternSummary) -> str:
    """One-line context for the bot post.

    Examples:
      87 похожих setup-ов: win rate 62%, ср. движение +1.8% за 12 баров.
        Распределение по режимам: bull 50, chop 25, bear 12.
      Недостаточно данных в историческом индексе.
    """
    if s.n_matches == 0:
        return "Недостаточно данных в историческом индексе — pattern-match пропущен."
    win_pct = round(s.win_rate * 100)
    mean_pct = s.mean_return * 100
    median_pct = s.median_return * 100
    dd_pct = s.max_drawdown * 100
    lines = [
        f"<b>{s.n_matches} похожих setup-ов</b> в истории: "
        f"win rate <b>{win_pct}%</b>, ср. движение <b>{mean_pct:+.2f}%</b> "
        f"(медиана {median_pct:+.2f}%, худшее {dd_pct:+.2f}%)."
    ]
    if s.regime_counts:
        regs_sorted = sorted(s.regime_counts.items(), key=lambda kv: -kv[1])
        regs_str = ", ".join(f"{r} {n}" for r, n in regs_sorted[:3])
        lines.append(f"Распределение по режимам: {regs_str}.")
    return " ".join(lines)


# ─── Lazy singleton loader for runtime use ──────────────────────────────────

_INDEX_CACHE: dict[str, PatternIndex | None] = {}
_NORMALIZER_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_INDEX_MTIMES: dict[str, float] = {}


def compute_query_features(
    rsi: float, macd_line: float, macd_signal: float,
    atr: float, price: float,
    support_50: float, resistance_50: float,
    recent_closes: list[float],
    regime_score: float = 0.0,
) -> np.ndarray:
    """Build the 7-D feature vector mirroring `build_features` in the build script.

    Column order (MUST match build script):
        [rsi, macd_diff, atr_norm, regime_score, dist_to_high_atr,
         dist_to_low_atr, ret_5]

    Bug #2 fix (review 2026-04-30): `regime_score` is now a real parameter
    fed by `infer_regime()`'s second return value (raw EMA-200 slope). The
    earlier hardcoded 0.0 caused HNSW kNN bias after z-normalization.

    Bug #7 fix: final np.nan_to_num scrub guards against any upstream NaN
    (corrupt parquet, partial Binance kline, RSI=NaN edge) — NaN in the
    query vector silently corrupts hnswlib distances.
    """
    macd_diff = macd_line - macd_signal
    atr_norm = atr / price if price > 0 else 0.0
    if atr > 0:
        dist_high = (resistance_50 - price) / atr
        dist_low = (price - support_50) / atr
    else:
        dist_high = 0.0
        dist_low = 0.0
    dist_high = max(-20.0, min(20.0, dist_high))
    dist_low = max(-20.0, min(20.0, dist_low))
    if len(recent_closes) >= 6 and recent_closes[-6] > 0:
        ret_5 = recent_closes[-1] / recent_closes[-6] - 1
        ret_5 = max(-0.5, min(0.5, ret_5))
    else:
        ret_5 = 0.0
    arr = np.array(
        [rsi, macd_diff, atr_norm, regime_score, dist_high, dist_low, ret_5],
        dtype=np.float32,
    )
    # Bug #7 final scrub — NaN/Inf to safe defaults.
    return np.nan_to_num(arr, nan=0.0, posinf=20.0, neginf=-20.0)


def enrich_with_pattern(
    symbol: str, interval: str,
    query_vec: np.ndarray, direction: Direction,
    top_k: int = 100,
    base_dir: str | Path = "data/pattern_indexes",
) -> str | None:
    """Main API for analyzers: returns formatted text or None.

    None means: index not built yet OR query failed. Caller falls back to
    the rule-based-only post (no pattern-match line).
    """
    idx = get_index(symbol, interval, base_dir=base_dir)
    if idx is None or idx.n_items == 0 or idx.forward_returns is None or idx.regimes is None:
        return None
    # Apply same normalizer used at build time.
    key = f"{symbol}_{interval}"
    if key not in _NORMALIZER_CACHE:
        p = Path(base_dir) / key
        try:
            mu = np.load(p / "mu.npy")
            sigma = np.load(p / "sigma.npy")
            _NORMALIZER_CACHE[key] = (mu, sigma)
        except Exception:
            return None
    mu, sigma = _NORMALIZER_CACHE[key]
    sigma_safe = np.where(sigma < 1e-9, 1.0, sigma)
    q_norm = ((query_vec - mu) / sigma_safe).astype(np.float32)

    indices, _dists = idx.query(q_norm, top_k=top_k)
    if len(indices) == 0:
        return None
    fwd = idx.forward_returns[indices]
    regs = idx.regimes[indices]
    summary = aggregate(fwd, regs, direction=direction)
    return format_summary_text(summary)


def get_index(symbol: str, interval: str,
              base_dir: str | Path = "data/pattern_indexes") -> PatternIndex | None:
    """Load and cache the index for (symbol, interval).

    Bug #8 fix: invalidate cache when the on-disk meta.json mtime changes,
    so a fresh `build_*_pattern_index.py` run during bot uptime is picked up
    automatically without process restart.

    Returns None if the index hasn't been built yet — analyzer falls back
    to the base post (rule-based idea + analysis, no historical context).
    """
    key = f"{symbol}_{interval}"
    p = Path(base_dir) / key
    meta_path = p / "meta.json"
    if not meta_path.exists():
        _INDEX_CACHE[key] = None
        return None
    try:
        mtime = meta_path.stat().st_mtime
    except OSError:
        return _INDEX_CACHE.get(key)

    # Cache hit only when mtime matches stored version.
    if key in _INDEX_CACHE and _INDEX_MTIMES.get(key) == mtime:
        return _INDEX_CACHE[key]

    # Cache miss or stale — reload and refresh both caches.
    try:
        _INDEX_CACHE[key] = PatternIndex.load(p)
        _NORMALIZER_CACHE.pop(key, None)  # force normalizer reload too
        _INDEX_MTIMES[key] = mtime
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "pattern index load failed for %s: %s", key, e
        )
        _INDEX_CACHE[key] = None
    return _INDEX_CACHE[key]
