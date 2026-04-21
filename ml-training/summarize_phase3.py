"""
summarize_phase3.py — Patch 2H-extended Phase 3 honest backtest results.

Prints per-TF / per-tier breakdown + rejection reasons. Run after backtest_v2
finishes to get a decision table.
"""
import json
import pathlib

ROOT = pathlib.Path("/root/elibri-backend/ml-training/logs/backtest_v2")

BASELINE_2H = {
    ("conservative", "5m"):  {"sharpe": 9.43, "wr": 0.556, "n": 18},
    ("conservative", "15m"): {"sharpe": -2.36, "wr": 0.429, "n": 28},
    ("conservative", "1h"):  {"sharpe": -10.27, "wr": 0.211, "n": 38},
    ("conservative", "4h"):  {"sharpe": -1.09, "wr": 0.389, "n": 54},
}


def main():
    print()
    print("=" * 110)
    print("PATCH 2H-EXTENDED PHASE 3 — Honest backtest on new candidate models")
    print("=" * 110)
    for tier in ("conservative", "balanced", "aggressive"):
        print()
        print(f"--- TIER: {tier.upper()} ---")
        print(
            f"{'TF':<4} {'trades':<7} {'WR':<7} {'return':<10} {'Sharpe':<9} "
            f"{'DD':<7} {'hc_eff':<7} {'vol_rej':<9} {'label_rej':<10} {'rate_rej':<9}"
        )
        for tf in ("5m", "15m", "1h", "4h"):
            p = ROOT / f"BTCUSDT_{tier}_{tf}.json"
            if not p.exists():
                continue
            j = json.loads(p.read_text())
            m = j["metrics"]
            r = m.get("n_rejected_by_reason", {})
            print(
                f"{tf:<4} {m['n_trades']:<7} {m['win_rate']*100:5.1f}% "
                f"{m['total_return_pct']:+8.2f}% {m['sharpe_after_fees']:+7.2f}  "
                f"{m['max_drawdown_pct']:5.2f}% {m['hc_threshold_effective']:.3f}   "
                f"{r.get('vol_floor', 0):<9} {r.get('label_not_allowed', 0):<10} "
                f"{r.get('rate_limit', 0):<9}"
            )

    print()
    print("=" * 110)
    print("CONSERVATIVE — comparison vs Patch 2H honest baseline")
    print("=" * 110)
    print(f"{'TF':<4} {'2H sharpe':<12} {'2H WR':<8} {'2H n':<6} {'NEW sharpe':<12} {'NEW WR':<8} {'NEW n':<6} {'Δ sharpe':<10} {'verdict'}")
    print("-" * 110)
    for tf in ("5m", "15m", "1h", "4h"):
        base = BASELINE_2H.get(("conservative", tf))
        p = ROOT / f"BTCUSDT_conservative_{tf}.json"
        if not p.exists() or base is None:
            continue
        j = json.loads(p.read_text())
        m = j["metrics"]
        delta = m["sharpe_after_fees"] - base["sharpe"]
        if delta > 0 and m["sharpe_after_fees"] > 0:
            verdict = "BETTER"
        elif delta < 0 and m["sharpe_after_fees"] < 0:
            verdict = "WORSE"
        elif m["sharpe_after_fees"] > 0:
            verdict = "OK"
        else:
            verdict = "FAIL"
        print(
            f"{tf:<4} {base['sharpe']:+11.2f} {base['wr']*100:5.1f}%   {base['n']:<6} "
            f"{m['sharpe_after_fees']:+11.2f} {m['win_rate']*100:5.1f}%   {m['n_trades']:<6} "
            f"{delta:+8.2f}  {verdict}"
        )

    print()
    print("DECISION:")
    print("  - If most TFs improved or flipped positive: cutover (copy new models to prod).")
    print("  - If 5m regressed and only 4h is positive: DO NOT CUTOVER production 5m model.")
    print("    Instead: keep old 5m model, deploy new 4h only (1h/1d remain MTF-filter role).")
    print("  - If all regressed: rollback TF_CONFIG, investigate feature parity or threshold logic.")


if __name__ == "__main__":
    main()
