"""
b2_4h_breakdown.py — B.2 deliverables 2, 3, 6, 7:
  d2: direction balance + per-label WR per tier
  d3: per-month regime mapping from trade dates
  d6: fees/slippage sensitivity (gross vs net PnL)
  d7: MTF gate — how many signals killed by label_not_allowed

Reads logs/backtest_v2/BTCUSDT_{tier}_4h.json from Phase 3.
"""
import json
import math
import pathlib
from collections import defaultdict

LOGS = pathlib.Path("/root/elibri-backend/ml-training/logs/backtest_v2")


def regime_label(date_str: str) -> str:
    if not date_str or len(date_str) < 7:
        return "?"
    year = int(date_str[:4])
    month = int(date_str[5:7])
    if year == 2024:
        return "bull 2024 H1" if month <= 6 else "chop 2024 H2"
    if year == 2025:
        return "bull 2025"
    if year >= 2026:
        return "bull 2026"
    if year == 2023:
        return "chop 2023"
    if year == 2022:
        return "BEAR 2022"
    return f"{year}"


def summarize(tier):
    path = LOGS / f"BTCUSDT_{tier}_4h.json"
    if not path.exists():
        print(f"  MISSING: {path}")
        return None
    j = json.loads(path.read_text())
    trades = j["trades"]
    metrics = j["metrics"]

    print(f"\n{'=' * 90}\n=== {tier.upper()} TIER — {len(trades)} trades ===\n{'=' * 90}")
    print(f"  headline: WR={metrics['win_rate']:.1%} "
          f"Sharpe={metrics['sharpe_after_fees']:+.2f} "
          f"ret={metrics['total_return_pct']:+.2f}% "
          f"DD={metrics['max_drawdown_pct']:.2f}% "
          f"fees=${metrics['fees_paid_total']:.0f}")

    # --- d2 direction balance ---
    longs = [t for t in trades if t.get("direction") == 1]
    shorts = [t for t in trades if t.get("direction") == -1]
    lw = sum(1 for t in longs if t.get("pnl_dollars", 0) > 0)
    sw = sum(1 for t in shorts if t.get("pnl_dollars", 0) > 0)
    balance_pct = len(shorts) / max(1, len(trades))
    balance_label = "BALANCED" if 0.30 <= balance_pct <= 0.70 else f"BIASED {balance_pct:.0%}"
    print(f"\n  [d2] direction: longs={len(longs)} (WR {lw/max(1,len(longs)):.1%}) "
          f"shorts={len(shorts)} (WR {sw/max(1,len(shorts)):.1%}) → {balance_label}")

    # --- Per-label ---
    lbl_count = defaultdict(lambda: [0, 0])  # [total, wins]
    for t in trades:
        k = t.get("label", "?")
        lbl_count[k][0] += 1
        if t.get("pnl_dollars", 0) > 0:
            lbl_count[k][1] += 1
    print("  per-label:")
    for lbl, (total, wins) in sorted(lbl_count.items()):
        print(f"    {lbl}: {total} trades, WR={wins/max(1,total):.1%}")

    # --- d3 per-regime aggregation ---
    regime_trades = defaultdict(list)
    for t in trades:
        d = t.get("entry_date", "")
        regime_trades[regime_label(d)].append(t)

    print(f"\n  [d3] per-regime breakdown:")
    print(f"    {'regime':<20} {'n':<4} {'long/short':<12} {'WR':<7} {'avg pnl%':<10} {'total ret%':<11}")
    for reg in sorted(regime_trades.keys()):
        tt = regime_trades[reg]
        n = len(tt)
        lo = sum(1 for t in tt if t.get("direction") == 1)
        sh = n - lo
        w = sum(1 for t in tt if t.get("pnl_dollars", 0) > 0)
        avg_pnl = sum(t.get("pnl_pct", 0) for t in tt) / max(1, n) * 100
        ret = sum(t.get("pnl_pct", 0) for t in tt) * 100
        print(f"    {reg:<20} {n:<4} {lo}/{sh:<10} {w/n*100:5.1f}% "
              f"{avg_pnl:+8.3f}%  {ret:+9.2f}%")

    # --- d6 fees/slippage gross vs net ---
    gross_pnl = sum(t.get("raw_pnl_pct", 0) for t in trades)
    net_pnl = sum(t.get("pnl_pct", 0) for t in trades)
    total_fees_pct = sum(t.get("fees_paid", 0) for t in trades) / metrics.get("initial_equity", 10000) * 100
    print(f"\n  [d6] fees/slippage impact:")
    print(f"    gross raw_pnl% sum: {gross_pnl*100:+.2f}%")
    print(f"    net pnl% sum:        {net_pnl*100:+.2f}%")
    print(f"    fees paid ($):       ${metrics['fees_paid_total']:.0f} "
          f"({total_fees_pct:+.2f}% of initial equity)")
    # Stress test: what if fees×1.5?
    if metrics["initial_equity"] > 0:
        avg_per_trade_pnl = net_pnl / max(1, len(trades)) * 100  # percent of equity per trade
        avg_per_trade_fee = metrics["fees_paid_total"] / max(1, len(trades)) / metrics["initial_equity"] * 100
        stressed_pnl = sum(
            t.get("pnl_pct", 0) - 0.5 * (t.get("fees_paid", 0) / metrics["initial_equity"])
            for t in trades
        ) * 100
        print(f"    stress (fees ×1.5): net pnl would become: {stressed_pnl:+.2f}%")

    # --- d7 MTF gate impact ---
    rej = metrics.get("n_rejected_by_reason", {})
    print(f"\n  [d7] MTF & tier gates (rejections):")
    for reason, n in sorted(rej.items(), key=lambda x: -x[1]):
        print(f"    {reason:<22}: {n}")
    total_rej = sum(rej.values())
    label_rej = rej.get("label_not_allowed", 0)
    vol_rej = rej.get("vol_floor", 0)
    hc_rej = rej.get("hc_threshold", 0)
    rate_rej = rej.get("rate_limit", 0)
    total_candidates = total_rej + len(trades)
    if total_candidates > 0:
        print(f"    total candidates: {total_candidates} "
              f"→ {len(trades)} passed ({len(trades)/total_candidates*100:.2f}%)")
        print(f"    MTF label_not_allowed fraction: "
              f"{label_rej/total_candidates*100:.2f}% "
              f"({label_rej}/{total_candidates})")

    return {
        "tier": tier,
        "n_trades": len(trades),
        "balance_pct": balance_pct,
        "wr": metrics["win_rate"],
        "sharpe": metrics["sharpe_after_fees"],
        "dd": metrics["max_drawdown_pct"],
        "regime_trades": {k: len(v) for k, v in regime_trades.items()},
    }


def main():
    print("Session B.2 — BTC 4h deep dive: deliverables 2, 3, 6, 7\n")

    results = []
    for tier in ("conservative", "balanced", "aggressive"):
        r = summarize(tier)
        if r:
            results.append(r)

    # Summary table
    print("\n" + "=" * 90)
    print("CROSS-TIER SUMMARY — BTC 4h")
    print("=" * 90)
    print(f"{'tier':<14} {'trades':<7} {'short%':<8} {'WR':<7} {'Sharpe':<9} {'DD%':<7}")
    for r in results:
        print(f"{r['tier']:<14} {r['n_trades']:<7} "
              f"{r['balance_pct']*100:5.1f}%  "
              f"{r['wr']*100:5.1f}%  "
              f"{r['sharpe']:+7.2f}  {r['dd']:5.2f}%")


if __name__ == "__main__":
    main()
