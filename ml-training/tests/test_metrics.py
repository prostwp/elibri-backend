"""Regression tests against the 45 bugs found in Scenario A (BTC 4h).

Run before launching any training:
    pytest tests/ -q

Each test maps to a bug class. If a future patch reintroduces the bug, the
test fails — so we don't ship the regression.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Bug class 1: Sharpe / metric math ───────────────────────────────────────

def test_sharpe_fees_only_on_transition():
    """Position [1,1,1,1] held 4 bars — fee charged on entry only, not 4×."""
    from wfcv_regime_aware import sharpe_after_fees, MIN_TRADES_FOR_SHARPE
    n = max(MIN_TRADES_FOR_SHARPE * 2, 100)
    proba = np.zeros((n, 3))
    proba[:, 1] = 0.99  # always long
    rets = np.full(n, 0.001)
    sh_zero_fee = sharpe_after_fees(proba, rets, "4h", fee=0.0, slippage=0.0)
    sh_with_fee = sharpe_after_fees(proba, rets, "4h", fee=0.001, slippage=0.0003)
    drag = sh_zero_fee - sh_with_fee
    assert drag < 0.5, f"Sharpe drag {drag:.3f} too large — fees on every bar?"


def test_sharpe_flip_charges_two_fees():
    """Position 1→-1 must charge |1-(-1)|=2 fees, not 1."""
    from wfcv_regime_aware import _positions_from_proba
    proba = np.array([[0.0, 0.99, 0.0], [0.0, 0.0, 0.99]])
    pos = _positions_from_proba(proba)
    pos_prev = np.concatenate([[0.0], pos[:-1]])
    pos_change = np.abs(pos - pos_prev)
    assert pos_change[0] == 1.0  # entry long
    assert pos_change[1] == 2.0  # flip 1 → -1 (close + open)


def test_sharpe_min_trades_guard():
    """<MIN_TRADES_FOR_SHARPE non-zero positions ⇒ NaN (no single-trade explosion)."""
    from wfcv_regime_aware import sharpe_after_fees, MIN_TRADES_FOR_SHARPE
    proba = np.zeros((1000, 3))
    proba[:, 0] = 0.99
    proba[5, 1] = 0.99  # ONE long signal
    rets = np.random.normal(0, 0.01, 1000)
    sh = sharpe_after_fees(proba, rets, "4h", hc_long=0.55, hc_short=0.55)
    assert math.isnan(sh), f"Expected NaN for <{MIN_TRADES_FOR_SHARPE} trades, got {sh}"


def test_bars_per_year_crypto_is_365():
    """1d crypto annualization = 365, NOT 252 (equity calendar)."""
    from wfcv_regime_aware import BARS_PER_YEAR
    assert BARS_PER_YEAR["1d"] == 365


# ── Bug class 2: Direction balance ──────────────────────────────────────────

def test_direction_share_sums_to_one():
    """long_share + short_share + hold_share == 1.0 exactly (mirrors pos)."""
    from wfcv_regime_aware import compute_direction_share
    proba = np.array([
        [0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.5, 0.3, 0.2],
        [0.1, 0.6, 0.3], [0.2, 0.4, 0.4],
    ])
    ds = compute_direction_share(proba)
    total = ds["long_share"] + ds["short_share"] + ds["hold_share"]
    assert abs(total - 1.0) < 1e-9, f"shares sum to {total}, not 1.0"


# ── Bug class 3: Sample weights ─────────────────────────────────────────────

def test_sample_uniqueness_normalized_mean_one():
    """Weights mean = 1.0 exactly; edges higher than middle."""
    from feature_engine import compute_sample_uniqueness
    w = compute_sample_uniqueness(1000, 18)
    assert abs(w.mean() - 1.0) < 1e-5
    assert w[0] > w[500]  # edge > middle
    assert (w > 0).all()
    assert not np.isnan(w).any()


# ── Bug class 4: Symmetric barriers ─────────────────────────────────────────

def test_tf_config_symmetric():
    """All TFs must have tb_upper == tb_lower (no asymmetric label prior)."""
    from train_autogluon import TF_CONFIG
    for tf, cfg in TF_CONFIG.items():
        assert cfg["tb_upper"] == cfg["tb_lower"], (
            f"{tf} barriers asymmetric: {cfg['tb_upper']}/{cfg['tb_lower']}")


def test_legacy_train_py_tf_config_synced():
    """train.py legacy TF_CONFIG must also be symmetric (architecture coherence)."""
    from train import TF_CONFIG as LEGACY
    for tf, cfg in LEGACY.items():
        assert cfg["tb_upper"] == cfg["tb_lower"], (
            f"legacy train.py {tf} asymmetric — sync to train_autogluon.py")


# ── Bug class 5: JSON NaN serialization ─────────────────────────────────────

def test_atomic_write_json_handles_nan():
    """NaN/Inf must serialize to null (RFC 8259 compliance)."""
    import json, tempfile
    from train_autogluon import atomic_write_json
    payload = {
        "sharpe_nan": float("nan"),
        "sharpe_inf": float("inf"),
        "good": 1.5,
        "nested": {"list": [1.0, float("nan"), 2.0]},
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.json"
        atomic_write_json(p, payload)
        # Strict json.loads — fails on literal NaN
        parsed = json.loads(p.read_text())
        assert parsed["sharpe_nan"] is None
        assert parsed["sharpe_inf"] is None
        assert parsed["good"] == 1.5
        assert parsed["nested"]["list"][1] is None


# ── Bug class 6: Path traversal ─────────────────────────────────────────────

def test_symbol_validation_rejects_traversal():
    """--symbol "../etc" must raise."""
    import argparse
    from train_autogluon import _validate_symbol
    with pytest.raises(argparse.ArgumentTypeError):
        _validate_symbol("../etc")
    with pytest.raises(argparse.ArgumentTypeError):
        _validate_symbol("BTC/USDT")  # slash
    assert _validate_symbol("BTCUSDT") == "BTCUSDT"


def test_cache_path_blocks_invalid_symbol():
    """Defense-in-depth: _cache_path raises on invalid symbol even without CLI."""
    from data_fetcher import _cache_path
    with pytest.raises(ValueError):
        _cache_path("../etc", "4h")


# ── Bug class 7: Macro strict mode ──────────────────────────────────────────

def test_attach_macro_strict_raises_on_missing():
    """strict=True with missing macro parquets → FileNotFoundError, not silent zero."""
    import tempfile
    from feature_engine import attach_macro_features
    feat = pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC"),
        "close": np.arange(10, dtype=float),
    })
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(FileNotFoundError):
            attach_macro_features(feat, "BTCUSDT", macro_dir=td, strict=True)


# ── Bug class 8: Class label canonicalization ───────────────────────────────

def test_macro_feature_names_count():
    """MACRO_FEATURE_NAMES must have exactly 9 entries (was 8 before oi_in_coverage)."""
    from feature_engine import MACRO_FEATURE_NAMES, FEATURE_NAMES_WITH_MACRO
    assert len(MACRO_FEATURE_NAMES) == 9
    assert "oi_in_coverage" in MACRO_FEATURE_NAMES
    assert len(FEATURE_NAMES_WITH_MACRO) == 45 + 9


# ── Bug class 9: Judge fallback ─────────────────────────────────────────────

def test_judge_fallback_on_missing_class():
    """If calibration set lacks class 1 (long), judge falls back to average,
    NOT silent zero P_long."""
    from multi_agent_fusion import apply_learned_judge, judge_average
    bad = {"classes": [0, 2], "coef": [[1, 0], [0, 1]], "intercept": [0, 0]}
    p_long = np.array([0.7, 0.3])
    p_short = np.array([0.2, 0.6])
    out = apply_learned_judge(bad, p_long, p_short)
    expected = judge_average(p_long, p_short)
    np.testing.assert_allclose(out, expected, atol=1e-9)
    assert out[:, 1].sum() > 0  # P_long not silently zeroed


# ── Bug class 10: OHLCV validation ──────────────────────────────────────────

def test_ohlcv_validation_catches_corruption():
    """_validate_ohlcv raises on duplicate timestamps, negative volume,
    non-positive prices, future dates."""
    from data_fetcher import _validate_ohlcv

    base = pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC"),
        "open":   [100, 101, 102, 103, 104],
        "high":   [101, 102, 103, 104, 105],
        "low":    [ 99, 100, 101, 102, 103],
        "close":  [100.5, 101.5, 102.5, 103.5, 104.5],
        "volume": [10, 20, 30, 40, 50],
    })
    _validate_ohlcv(base, "ok")  # no raise

    dup = base.copy()
    dup.loc[1, "open_time"] = base.loc[0, "open_time"]
    with pytest.raises(ValueError, match="duplicate"):
        _validate_ohlcv(dup, "dup")

    negv = base.copy()
    negv.loc[2, "volume"] = -5
    with pytest.raises(ValueError, match="negative volume"):
        _validate_ohlcv(negv, "negv")

    badp = base.copy()
    badp.loc[2, "close"] = 0
    with pytest.raises(ValueError, match="non-positive"):
        _validate_ohlcv(badp, "badp")


# ── Bug class 11: Position-return alignment ─────────────────────────────────

def test_position_return_alignment_monotone_long_positive_sharpe():
    """Always-long on monotone-up price MUST give POSITIVE Sharpe.
    Old code (pos[i]*ret[i] where ret was past return) gave negative."""
    from wfcv_regime_aware import sharpe_after_fees, MIN_TRADES_FOR_SHARPE
    n = max(MIN_TRADES_FOR_SHARPE * 2, 100)
    proba = np.zeros((n, 3))
    proba[:, 1] = 0.99  # always long
    # ret_next: up every bar except last (no next-bar after the last)
    rets = np.full(n, 0.001)
    rets[-1] = 0.0
    sh = sharpe_after_fees(proba, rets, "4h", fee=0.0, slippage=0.0)
    assert sh > 0, f"Always-long on monotone-up gave Sharpe={sh}, expected positive"
