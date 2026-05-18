"""Tests for signal_worker — written BEFORE implementation (TDD).

Per `feedback_ml_pipeline_safety.md`. Skips DB integration; mocks Db.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signal_worker import bar_time_ms, RefreshSummary  # noqa: E402


# ─── bar_time_ms — must floor to start of current bar (millis) ──────────────

def test_bar_time_ms_floor_5m():
    """5m bar floors to 5-minute boundary."""
    # 1714521600000 ms = 2024-05-01 00:00:00 UTC
    # +7 minutes → 1714522020000; should floor back to 1714521900000 (00:05:00)
    base = 1714521600000
    assert bar_time_ms("5m", base + 7 * 60 * 1000) == base + 5 * 60 * 1000


def test_bar_time_ms_floor_4h():
    """4h bar floors to 4-hour boundary (00:00, 04:00, 08:00, ...)."""
    # 1714521600000 = 2024-05-01 00:00:00 UTC (clean 4h boundary)
    base = 1714521600000
    # +1 minute → still in same 4h bar → floors back to base
    assert bar_time_ms("4h", base + 60_000) == base
    # +5h → next 4h bar (04:00 = base + 4*3600*1000)
    assert bar_time_ms("4h", base + 5 * 3600 * 1000) == base + 4 * 3600 * 1000


def test_bar_time_ms_floor_1d():
    """1d bar floors to UTC midnight."""
    base = 1714521600000  # 2024-05-01 00:00:00
    assert bar_time_ms("1d", base + 12 * 3600 * 1000) == base
    assert bar_time_ms("1d", base + 25 * 3600 * 1000) == base + 86400 * 1000


def test_bar_time_unknown_interval_raises():
    """Unknown interval → ValueError, not silent zero."""
    with pytest.raises(ValueError):
        bar_time_ms("99x", 0)


# ─── RefreshSummary stats container ─────────────────────────────────────────

def test_refresh_summary_counts():
    """Worker reports per-cycle stats: success/failed/skipped."""
    s = RefreshSummary()
    s.success += 3
    s.failed += 1
    s.skipped += 2
    assert s.total == 6
    assert s.success == 3
    assert s.failed == 1
    assert s.skipped == 2


# ─── Direction normalisation for UNIQUE constraint ──────────────────────────

def test_direction_normalised_for_dedup():
    """Worker maps Setup.direction → 'BUY' / 'SELL' / 'HOLD' to match
    alerts.direction VARCHAR(10) used by historical seed data."""
    from signal_worker import setup_to_alert_direction
    assert setup_to_alert_direction("long") == "BUY"
    assert setup_to_alert_direction("short") == "SELL"
    # For non-TA authors (Astro/Currency/Index/Oil), direction is just informational.
    assert setup_to_alert_direction("hold") == "HOLD"
    assert setup_to_alert_direction("info") == "INFO"


# ─── Cold-start fallback ────────────────────────────────────────────────────

def test_cold_start_text_when_alert_missing():
    """When no alert exists for an author, render_for_author returns a
    waiting-message instead of a crash or live recompute."""
    from signal_worker import COLD_START_MESSAGE
    assert "сигнал" in COLD_START_MESSAGE.lower() or "ждём" in COLD_START_MESSAGE.lower()
    assert len(COLD_START_MESSAGE) > 30
