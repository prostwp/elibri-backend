"""
iter_analyze.py — Patch 4 iteration analysis helper.

Reads a backtest_v2 per-tier JSON (produced when honest backtest is run on a
candidate 3-class model) and prints the updated success criteria set:
  - trades_per_day (target: >= 1.0 on Aggressive for 15m)
  - longs/shorts balance (target: 30-70% short)
  - WR (target: >= 55% on direction-traded bars)
  - gross vs net return, fees/gross ratio (target: fees <= 30% gross)
  - Sharpe (target: >= 1.0 on 2/3 tiers)

Usage:
  python3 iter_analyze.py <tier-json-path> [test_window_days]
"""
import json
import pathlib
import sys
from collections import defaultdict


def analyze(path: pathlib.Path, test_days: float | None = None):
    j = json.loads(path.read_text())
    trades = j["trades"]
    m = j["metrics"]
    tier = j["tier"]
    interval = j["interval"]
    symbol = j["symbol"]

    n = len(trades)
    longs = [t for t in trades if t["direction"] == 1]
    shorts = [t for t in trades if t["direction"] == -1]
    lw = sum(1 for t in longs if t["pnl_dollars"] > 0)
    sw = sum(1 for t in shorts if t["pnl_dollars"] > 0)
    total_wins = sum(1 for t in trades if t["pnl_dollars"] > 0)

    # Days covered.
    if test_days is None and trades:
        from datetime import datetime
        start = datetime.fromisoformat(trades[0]["entry_date"][:19])
        end = datetime.fromisoformat(trades[-1]["entry_date"][:19])
        test_days = (end - start).days or 1

    tpd = n / max(1, test_days) if test_days else 0.0

    # Fee impact.
    gross = sum(t["raw_pnl_pct"] for t in trades) * 100
    net = sum(t["pnl_pct"] for t in trades) * 100
    fees = m["fees_paid_total"]
    initial_eq = m["initial_equity"]
    fees_pct_equity = fees / initial_eq * 100
    if abs(gross) > 1e-6:
        net_gross_ratio = net / gross
    else:
        net_gross_ratio = 0.0

    print(f"\n=== {symbol} {interval} {tier.upper()} — iteration metrics ===")
    print(f"  test window:          {test_days} days")
    print(f"  trades total:         {n}")
    print(f"  trades/day:           {tpd:.2f}    (target >= 1.0 for Aggressive)")
    print(f"  longs/shorts:         {len(longs)}/{len(shorts)}  "
          f"(short% = {len(shorts)/max(1,n)*100:.0f}%; target 30-70%)")
    print(f"  per-dir WR: long={lw}/{len(longs)}={lw/max(1,len(longs)):.1%}  "
          f"short={sw}/{len(shorts)}={sw/max(1,len(shorts)):.1%}")
    print(f"  overall WR:           {total_wins}/{n} = {total_wins/max(1,n):.1%}  (target >= 55%)")
    print(f"  gross return:         {gross:+.2f}%")
    print(f"  net return:           {net:+.2f}%")
    print(f"  fees paid:            ${fees:.0f} ({fees_pct_equity:+.2f}% of ${initial_eq:.0f})")
    print(f"  net/gross ratio:      {net_gross_ratio:.2f}  (target >= 0.40)")
    print(f"  Sharpe after fees:    {m['sharpe_after_fees']:+.2f}  (target >= 1.0)")
    print(f"  max DD:               {m['max_drawdown_pct']:.2f}%")

    # Label breakdown (trend_aligned/mean_rev/random).
    labels = defaultdict(lambda: [0, 0])
    for t in trades:
        k = t.get("label", "?")
        labels[k][0] += 1
        if t["pnl_dollars"] > 0:
            labels[k][1] += 1
    if labels:
        print(f"  label breakdown:")
        for lbl, (total, wins) in sorted(labels.items()):
            print(f"    {lbl:<16} {total} trades, WR {wins/max(1,total):.1%}")

    # Rejections.
    rej = m.get("n_rejected_by_reason", {})
    if rej:
        print(f"  rejections: {dict(rej)}")

    # Pass/fail verdict per criterion.
    checks = {
        "trades/day >= 1.0":       tpd >= 1.0,
        "short% in 30-70":         30 <= len(shorts) / max(1, n) * 100 <= 70,
        "WR >= 55%":                total_wins / max(1, n) >= 0.55,
        "net/gross >= 0.40":       net_gross_ratio >= 0.40,
        "Sharpe >= 1.0":           m["sharpe_after_fees"] >= 1.0,
    }
    print(f"\n  Verdict per criterion:")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL':<5} {k}")
    n_pass = sum(1 for v in checks.values() if v)
    print(f"\n  TOTAL: {n_pass}/{len(checks)} criteria green")


def main():
    if len(sys.argv) < 2:
        print("usage: iter_analyze.py <path-to-tier-json>")
        sys.exit(1)
    path = pathlib.Path(sys.argv[1])
    test_days = float(sys.argv[2]) if len(sys.argv) > 2 else None
    analyze(path, test_days)


if __name__ == "__main__":
    main()
