"""
b1_1d_breakdown.py — direction/label/regime breakdown of Session B.1 1d backtest trades.

Reads logs/b1_backtest_1d/BTCUSDT_{tier}_1d.json for all 3 tiers and reports:
  - long vs short split + per-direction WR
  - label breakdown (trend_aligned / mean_reversion / random)
  - exit_reason split (tp / sl / timeout)
  - rejection reasons (hc_threshold / vol_floor / label_not_allowed / rate_limit)
  - month-by-month Sharpe to spot regime-specific weakness
"""
import json
import math
import pathlib
from collections import defaultdict

LOGS = pathlib.Path("/root/elibri-backend/ml-training/logs/b1_backtest_1d")


def per_month_sharpe(trades):
    """Group trades by YYYY-MM and compute per-bucket Sharpe-like ratio."""
    buckets = defaultdict(list)
    for t in trades:
        d = t.get("entry_date", "")
        key = d[:7] if len(d) >= 7 else "?"
        buckets[key].append(t.get("pnl_pct", 0))
    rows = []
    for key in sorted(buckets.keys()):
        rets = buckets[key]
        if len(rets) < 2:
            rows.append((key, len(rets), sum(rets) * 100, 0.0))
            continue
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        sd = math.sqrt(max(var, 1e-12))
        # annualized-ish factor for monthly buckets
        annual_factor = math.sqrt(12 / max(1, (sum(t.get("bars_held", 1) for t in trades) / len(trades))))
        sharpe = mean / sd * annual_factor if sd > 0 else 0.0
        rows.append((key, len(rets), sum(rets) * 100, sharpe))
    return rows


def summarize(tier):
    path = LOGS / f"BTCUSDT_{tier}_1d.json"
    if not path.exists():
        print(f"  NO FILE: {path}")
        return
    j = json.loads(path.read_text())
    trades = j["trades"]
    metrics = j["metrics"]

    print()
    print(f"=== {tier.upper()} TIER — {len(trades)} trades ===")

    # Direction + per-direction WR.
    longs = [t for t in trades if t.get("direction") == 1]
    shorts = [t for t in trades if t.get("direction") == -1]
    lw = sum(1 for t in longs if t.get("pnl_dollars", 0) > 0)
    sw = sum(1 for t in shorts if t.get("pnl_dollars", 0) > 0)
    print(f"  direction: longs={len(longs)} (WR {lw/max(1,len(longs)):.1%}) "
          f"shorts={len(shorts)} (WR {sw/max(1,len(shorts)):.1%})")
    balance = len(shorts) / max(1, len(trades))
    verdict_dir = "BALANCED" if 0.30 <= balance <= 0.70 else f"SHORT-BIASED {balance:.0%}"
    print(f"  direction balance: {verdict_dir}")

    # Labels.
    lbls = defaultdict(int)
    lbl_wr = defaultdict(lambda: [0, 0])
    for t in trades:
        k = t.get("label", "?")
        lbls[k] += 1
        lbl_wr[k][0] += 1
        if t.get("pnl_dollars", 0) > 0:
            lbl_wr[k][1] += 1
    for lbl, n in sorted(lbls.items()):
        total, wins = lbl_wr[lbl][0], lbl_wr[lbl][1]
        print(f"  label={lbl}: {n} trades, WR={wins/max(1,total):.1%}")

    # Exit reasons.
    er = defaultdict(int)
    for t in trades:
        er[t.get("exit_reason", "?")] += 1
    print(f"  exit_reason: {dict(er)}")

    # Rejections.
    print(f"  rejected: {metrics.get('n_rejected_by_reason', {})}")

    # Test window.
    if trades:
        print(f"  test window: {trades[0]['entry_date'][:10]} .. {trades[-1]['entry_date'][:10]}")

    # Per-month Sharpe.
    print(f"  per-month (regime hint):")
    rows = per_month_sharpe(trades)
    for key, n, ret_pct, sh in rows:
        print(f"    {key}: n={n:2d} ret%={ret_pct:+.2f} sharpe-ish={sh:+.2f}")

    # Headline.
    print(f"  HEADLINE: WR={metrics['win_rate']:.1%} "
          f"Sharpe={metrics['sharpe_after_fees']:+.2f} "
          f"ret={metrics['total_return_pct']:+.2f}% "
          f"DD={metrics['max_drawdown_pct']:.2f}%")


def main():
    print("SESSION B.1 — 1d honest backtest breakdown")
    print("Data: OOS 25% holdout (7.5 months of 30 months total post-warmup)")
    print()
    for tier in ("conservative", "balanced", "aggressive"):
        summarize(tier)

    print()
    print("=== CROSS-TIER KEY DIAGNOSTICS ===")
    # Compare how many short wins vs long wins across tiers.
    for tier in ("conservative", "balanced", "aggressive"):
        path = LOGS / f"BTCUSDT_{tier}_1d.json"
        if not path.exists():
            continue
        j = json.loads(path.read_text())
        trades = j["trades"]
        longs = [t for t in trades if t.get("direction") == 1]
        shorts = [t for t in trades if t.get("direction") == -1]
        lw = sum(1 for t in longs if t.get("pnl_dollars", 0) > 0)
        sw = sum(1 for t in shorts if t.get("pnl_dollars", 0) > 0)
        print(f"  {tier:<13} longs {lw}/{len(longs)}={lw/max(1,len(longs)):.0%}  "
              f"shorts {sw}/{len(shorts)}={sw/max(1,len(shorts)):.0%}")


if __name__ == "__main__":
    main()
