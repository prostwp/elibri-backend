"""
b4_d2_direction.py — B.4 d2: direction balance + label/exit breakdown for 15m
honest backtest from Phase 3.
"""
import json
import pathlib
from collections import defaultdict

LOGS = pathlib.Path("/root/elibri-backend/ml-training/logs/backtest_v2")


def summarize(tier):
    p = LOGS / f"BTCUSDT_{tier}_15m.json"
    if not p.exists():
        print(f"  MISSING: {p}")
        return None
    j = json.loads(p.read_text())
    trades = j["trades"]
    m = j["metrics"]

    longs = [t for t in trades if t["direction"] == 1]
    shorts = [t for t in trades if t["direction"] == -1]
    lw = sum(1 for t in longs if t["pnl_dollars"] > 0)
    sw = sum(1 for t in shorts if t["pnl_dollars"] > 0)

    print(f"\n=== {tier.upper()} 15m — {len(trades)} trades ===")
    print(f"  WR={m['win_rate']:.1%} Sharpe={m['sharpe_after_fees']:+.2f} "
          f"ret={m['total_return_pct']:+.2f}% DD={m['max_drawdown_pct']:.2f}% "
          f"fees=${m['fees_paid_total']:.0f}")
    bal = len(shorts) / max(1, len(trades))
    print(f"  direction: longs={len(longs)} (WR {lw/max(1,len(longs)):.1%}) "
          f"shorts={len(shorts)} (WR {sw/max(1,len(shorts)):.1%}) "
          f"→ short%={bal*100:.0f}%")

    labels = defaultdict(lambda: [0, 0])
    for t in trades:
        k = t["label"]
        labels[k][0] += 1
        if t["pnl_dollars"] > 0:
            labels[k][1] += 1
    for lbl, (total, wins) in sorted(labels.items()):
        print(f"  label={lbl:<16}: {total} trades, WR={wins/max(1,total):.1%}")

    er = defaultdict(int)
    for t in trades:
        er[t["exit_reason"]] += 1
    print(f"  exit_reason: {dict(er)}")
    print(f"  rejected: {m.get('n_rejected_by_reason', {})}")

    if trades:
        print(f"  test window: {trades[0]['entry_date'][:10]} .. {trades[-1]['entry_date'][:10]}")

    return {"tier": tier, "n": len(trades), "short_pct": bal,
            "wr": m["win_rate"], "sharpe": m["sharpe_after_fees"]}


def main():
    print("B.4 d2 — 15m direction balance (Phase 3 honest backtest)")
    rows = []
    for t in ("conservative", "balanced", "aggressive"):
        r = summarize(t)
        if r:
            rows.append(r)

    print("\n" + "=" * 68)
    print("CROSS-TIER 15m summary")
    print(f"{'tier':<14} {'n':<5} {'short%':<8} {'WR':<8} {'Sharpe':<10}")
    for r in rows:
        print(f"{r['tier']:<14} {r['n']:<5} {r['short_pct']*100:5.0f}%   "
              f"{r['wr']*100:5.1f}%   {r['sharpe']:+7.2f}")


if __name__ == "__main__":
    main()
