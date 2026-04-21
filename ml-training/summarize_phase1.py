"""
summarize_phase1.py — Patch 2H-extended Phase 1 results parser.

Reads logs/BTCUSDT_{tf}_metrics.json and prints:
  - Per-fold headline metrics
  - Aggregated OOS HC precision + signal count
  - Side-by-side vs Patch 2H baseline (from memory)
"""
import json
import pathlib
import sys

LOGS = pathlib.Path("/root/elibri-backend/ml-training/logs")

# Baselines from Patch 2H honest metrics (memory/session_2026-04-22_honest_metrics).
BASELINES = {
    "5m":  {"sharpe": 9.43, "wr": 0.556, "trades": 18,  "note": "was working"},
    "15m": {"sharpe": -2.36, "wr": 0.429, "trades": 28,  "note": "negative"},
    "1h":  {"sharpe": -10.27, "wr": 0.211, "trades": 38,  "note": "catastrophe"},
    "4h":  {"sharpe": -1.09, "wr": 0.389, "trades": 54,  "note": "negative"},
    "1d":  {"sharpe": 4.10, "wr": 0.500, "trades": None, "note": "from Patch 2B"},
}

def agg_hc(folds):
    """Aggregate HC precision across folds weighted by trade count."""
    total_trades = 0
    total_correct = 0.0
    total_wins = 0
    total_test = 0
    for f in folds:
        n = f.get("hc_trades", 0)
        if n > 0:
            p = f.get("hc_precision", 0)
            total_trades += n
            total_correct += p * n
            wr = f.get("hc_win_rate", p)
            total_wins += wr * n
        total_test += f.get("n_test", 0)
    if total_trades == 0:
        return 0.0, 0.0, 0, 0, 0.0
    prec = total_correct / total_trades
    wr = total_wins / total_trades
    frac = total_trades / max(1, total_test)
    return prec, wr, total_trades, total_test, frac


def main():
    rows = []
    for tf in ("1h", "4h", "1d"):
        p = LOGS / f"BTCUSDT_{tf}_metrics.json"
        if not p.exists():
            print(f"SKIP {tf}: no metrics file")
            continue
        m = json.loads(p.read_text())
        prec, wr, trades, n_test, frac = agg_hc(m["folds"])
        rows.append({
            "tf": tf,
            "acc": m["avg_accuracy"],
            "f1": m["avg_f1"],
            "sharpe": m["avg_sharpe"],
            "hc_prec": prec,
            "hc_wr": wr,
            "hc_trades": trades,
            "n_test": n_test,
            "frac": frac,
            "folds": m["n_folds"],
            "horizon": m.get("horizon"),
            "dur": m.get("duration_sec", 0),
        })

    print()
    print("=" * 90)
    print("PATCH 2H-EXTENDED PHASE 1 RESULTS — BTC 1h/4h/1d on 8y data, tb widened")
    print("=" * 90)
    print()
    print(f"{'TF':<4} {'horizon':<8} {'acc':<6} {'f1':<6} {'sharpe':<8} {'HC prec':<8} {'HC wr':<7} {'HC #':<6} {'HC %':<6} {'folds':<6} {'dur':<5}")
    print("-" * 90)
    for r in rows:
        print(f"{r['tf']:<4} {str(r['horizon']):<8} {r['acc']:.3f}  {r['f1']:.3f}  {r['sharpe']:+6.2f}  "
              f"{r['hc_prec']:.1%}    {r['hc_wr']:.1%}   {r['hc_trades']:<6} {r['frac']:.2%}  {r['folds']:<6} {r['dur']:.0f}s")
    print()
    print("COMPARISON vs Patch 2H baseline (OOS 2y backtest_v2 Conservative tier):")
    print(f"{'TF':<4} {'prior sharpe':<14} {'prior WR':<10} {'prior trades':<14} {'new sharpe':<12} {'new HC WR':<12} {'delta':<10}")
    print("-" * 90)
    for r in rows:
        tf = r["tf"]
        base = BASELINES.get(tf, {})
        bs = base.get("sharpe", 0)
        bw = base.get("wr", 0)
        bt = base.get("trades") or "?"
        delta_sharpe = r["sharpe"] - bs
        marker = "UP  " if delta_sharpe > 0 else "DN  " if delta_sharpe < 0 else "EQ  "
        print(f"{tf:<4} {bs:+13.2f}  {bw:.1%}       {str(bt):<14} {r['sharpe']:+11.2f} {r['hc_wr']:.1%}         {marker}{delta_sharpe:+.2f}")

    print()
    print("MISSION criteria: WR >= 60%, Sharpe >= 1.5, OOS >= 2y, >= 18 trades")
    print()
    print(f"{'TF':<4} {'WR check':<14} {'Sharpe check':<16} {'trades check':<14} {'VERDICT'}")
    print("-" * 70)
    for r in rows:
        wr_ok = r["hc_wr"] >= 0.60
        sh_ok = r["sharpe"] >= 1.5
        tr_ok = r["hc_trades"] >= 18
        verdict = "PASS" if (wr_ok and sh_ok and tr_ok) else "FAIL"
        print(f"{r['tf']:<4} {'OK ' if wr_ok else 'FAIL':<4} {r['hc_wr']:.1%}   "
              f"{'OK ' if sh_ok else 'FAIL':<4} {r['sharpe']:+6.2f}    "
              f"{'OK ' if tr_ok else 'FAIL':<4} {r['hc_trades']:<3}     {verdict}")

    print()
    print("NOTE: Phase 1 shows CV walk-forward HC precision on default threshold 0.80.")
    print("Phase 2 regenerates best_thresholds.json — tighter-grid sweep might move HC")
    print("precision up/down depending on proba_max distribution. Phase 3 backtest_v2")
    print("with fees + slippage + MTF gate gives the HONEST number to compare against")
    print("baseline, not these raw CV numbers.")


if __name__ == "__main__":
    main()
