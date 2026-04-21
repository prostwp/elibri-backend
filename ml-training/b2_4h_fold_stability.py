"""
b2_4h_fold_stability.py — Session B.2 deliverable 1: fold-by-fold Sharpe stability for BTCUSDT 4h.

Reads logs/BTCUSDT_4h_metrics.json (produced by Phase 1 train.py retrain with
TF_CONFIG 4h: horizon=12, tb 3.0/2.0). Reports each walk-forward fold's
Sharpe + regime mapping (based on test_end date).
"""
import json
import pathlib
from collections import Counter

LOGS = pathlib.Path("/root/elibri-backend/ml-training/logs")


def regime_label(end_date_str: str) -> str:
    """Heuristic BTC regime mapping by test window end date."""
    if not end_date_str or len(end_date_str) < 7:
        return "?"
    year = int(end_date_str[:4])
    month = int(end_date_str[5:7])
    if year == 2019:
        return "chop/post-bear"
    if year == 2020:
        return "COVID shock + recovery"
    if year == 2021:
        return "bull 2021"
    if year == 2022:
        return "BEAR 2022 (LUNA, FTX)"
    if year == 2023:
        return "chop 2023"
    if year == 2024:
        return "bull 2024 H1" if month <= 6 else "chop 2024 H2"
    if year == 2025:
        return "bull 2025"
    if year >= 2026:
        return "bull 2026"
    return "?"


def main():
    m = json.loads((LOGS / "BTCUSDT_4h_metrics.json").read_text())
    folds = m["folds"]

    print("=" * 100)
    print(f"B.2 d1 — BTCUSDT 4h fold-by-fold Sharpe stability")
    print(f"horizon={m.get('horizon')} n_folds={m.get('n_folds')} avg_sharpe={m.get('avg_sharpe'):+.2f}")
    print("=" * 100)
    print()
    print(f"{'fold':<4} {'n_test':<7} {'acc':<6} {'f1':<6} {'sharpe':<9} "
          f"{'hc_n':<5} {'hc_prec':<8} {'test_end':<12} {'regime':<28}")
    print("-" * 100)

    negs = []
    regime_sharpes = {}
    for f in folds:
        s = f.get("sharpe", 0)
        end = str(f.get("test_end", ""))[:10]
        reg = regime_label(end)
        print(
            f"{f.get('fold', 0):<4} {f.get('n_test', 0):<7} "
            f"{f.get('accuracy', 0):.3f}  {f.get('f1', 0):.3f}  "
            f"{s:+8.2f}  "
            f"{f.get('hc_trades', 0):<5} {f.get('hc_precision', 0):.3f}   "
            f"{end:<12} {reg:<28}"
        )
        if s < 0:
            negs.append((f.get("fold"), s, end, reg))
        regime_sharpes.setdefault(reg, []).append(s)

    print()
    print("--- Negatives ---")
    if not negs:
        print("  NONE. All folds positive.")
    else:
        for fold_id, s, end, reg in negs:
            print(f"  fold {fold_id}: Sharpe={s:+.2f} at {end} ({reg})")

    print()
    print("--- Per-regime avg Sharpe ---")
    for reg in sorted(regime_sharpes.keys()):
        vals = regime_sharpes[reg]
        avg = sum(vals) / len(vals)
        n_neg = sum(1 for v in vals if v < 0)
        flag = " FAIL" if n_neg > 0 else ""
        print(f"  {reg:<28} folds={len(vals):2d} avg_sharpe={avg:+6.2f} neg_count={n_neg}{flag}")

    print()
    print("--- Verdict prep ---")
    print(f"  total folds:    {len(folds)}")
    print(f"  positive folds: {len(folds) - len(negs)}")
    print(f"  negative folds: {len(negs)}")
    if len(negs) == 0:
        print("  CHECK 1 RESULT: PASS (all folds positive Sharpe)")
    elif len(negs) <= 2:
        print(f"  CHECK 1 RESULT: CAVEAT — {len(negs)} folds negative, investigate regimes")
    else:
        print(f"  CHECK 1 RESULT: FAIL — {len(negs)} folds negative (more than 2)")


if __name__ == "__main__":
    main()
