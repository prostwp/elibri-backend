"""
b4_15m_deepdive.py — Session B.4 deliverables 1, 3, 4 for BTCUSDT 15m.

d1: fold-by-fold Sharpe stability (reads logs/BTCUSDT_15m_metrics.json —
    produced during Phase 1 retrain of 5m/15m/1h/4h/1d, TF_CONFIG horizon=16
    tb 2.5/1.5 — but wait, 15m was NOT retrained in Patch 2H-extended
    Phase 1 (only 1h/4h/1d). 15m metrics may still be from OLD training
    cycle. Need to check.)
d3: threshold monotonicity from best_thresholds.json.
d4: feature importance from models/BTCUSDT_15m_v*.json.
"""
import json
import pathlib


ROOT = pathlib.Path("/root/elibri-backend/ml-training")
LOGS = ROOT / "logs"
MODELS = ROOT / "models"


def section(title):
    print()
    print("=" * 84)
    print(title)
    print("=" * 84)


def regime_label(end_date: str) -> str:
    if not end_date or len(end_date) < 7:
        return "?"
    year = int(end_date[:4])
    month = int(end_date[5:7])
    if year == 2019:
        return "chop 2019"
    if year == 2020:
        return "COVID 2020"
    if year == 2021:
        return "bull 2021"
    if year == 2022:
        return "BEAR 2022"
    if year == 2023:
        return "chop 2023"
    if year == 2024:
        return "bull 2024 H1" if month <= 6 else "chop 2024 H2"
    if year == 2025:
        return "bull 2025"
    if year >= 2026:
        return "bull 2026"
    return f"{year}"


def d1_fold_breakdown():
    section("B.4 d1 — 15m fold-by-fold Sharpe stability")
    p = LOGS / "BTCUSDT_15m_metrics.json"
    if not p.exists():
        print(f"  NOT FOUND: {p}")
        return
    m = json.loads(p.read_text())
    folds = m["folds"]
    print(f"avg Sharpe={m['avg_sharpe']:+.2f}  n_folds={m['n_folds']}  horizon={m['horizon']}")
    print(f"training timestamp: {m.get('trained_at', '?')}")
    print()
    print(f"{'fold':<4} {'n_test':<7} {'acc':<6} {'sharpe':<9} "
          f"{'hc_n':<5} {'hc_prec':<8} {'test_end':<12} {'regime':<20}")
    print("-" * 84)

    regimes = {}
    negs = []
    sharpes_sorted = sorted([f.get("sharpe", 0) for f in folds], reverse=True)
    for f in folds:
        s = f.get("sharpe", 0)
        end = str(f.get("test_end", ""))[:10]
        reg = regime_label(end)
        regimes.setdefault(reg, []).append(s)
        if s < 0:
            negs.append((f.get("fold"), s, end, reg))
        print(f"{f.get('fold', 0):<4} {f.get('n_test', 0):<7} "
              f"{f.get('accuracy', 0):.3f}  {s:+8.2f}  "
              f"{f.get('hc_trades', 0):<5} {f.get('hc_precision', 0):.3f}   "
              f"{end:<12} {reg:<20}")

    print()
    print(f"negatives: {len(negs)}/{len(folds)}")
    for fid, s, e, r in negs:
        print(f"  fold {fid} sharpe={s:+.2f} ({e}, {r})")
    print()
    print("per-regime avg Sharpe:")
    for reg in sorted(regimes.keys()):
        vals = regimes[reg]
        avg = sum(vals) / len(vals)
        n_neg = sum(1 for v in vals if v < 0)
        print(f"  {reg:<20} folds={len(vals):<2} avg={avg:+6.2f} neg={n_neg}")

    if len(folds) >= 5:
        top5 = sharpes_sorted[:5]
        tail_avg = (sum(sharpes_sorted[5:]) / max(1, len(sharpes_sorted) - 5))
        print(f"\n  top-5 Sharpes: {[f'{x:+.2f}' for x in top5]}")
        print(f"  tail avg (excluding top-5): {tail_avg:+.2f}")


def d3_threshold_monotonicity():
    section("B.4 d3 — 15m threshold monotonicity (Phase 2 sweep)")
    raw = json.loads((LOGS / "best_thresholds.json").read_text())
    row = None
    for r in raw.get("results", []):
        if r["symbol"] == "BTCUSDT" and r["interval"] == "15m":
            row = r
            break
    if row is None:
        print("  NO 15m row in best_thresholds.json")
        print("  (Phase 2 was run for 1h/4h/1d only; 15m uses FALLBACK_HC=0.73)")
        return

    print(f"  proba_min={row['proba_min']:.3f} max={row['proba_max']:.3f} "
          f"mean={row['proba_mean']:.3f} std={row['proba_std']:.3f}")
    print()
    print(f"  {'threshold':<10} {'precision':<10} {'n_signals':<10} {'fraction':<10}")
    print("  " + "-" * 44)
    rows = []
    for k in sorted(row["hc_table"].keys()):
        s = row["hc_table"][k]
        rows.append(s)
        print(f"  {s['threshold']:<10.3f} {s['precision']:<10.3f} "
              f"{s['n_signals']:<10} {s['fraction']:<10.4f}")

    dense = [r for r in rows if r["n_signals"] >= 30]
    last_prec = 0.0
    drops = 0
    for r in dense:
        if r["precision"] < last_prec - 0.02:
            drops += 1
        last_prec = max(last_prec, r["precision"])
    print()
    if drops == 0:
        print(f"  PASS: monotonic on dense ({len(dense)} rows with n>=30)")
    else:
        print(f"  CAVEAT: {drops} rows drop >2pp")

    best = row.get("best", {})
    print(f"  Phase 2 chose: thr={best.get('threshold')} "
          f"precision={best.get('precision'):.3f} "
          f"n_signals={best.get('n_signals')} policy={best.get('policy')}")


def d4_feature_importance():
    section("B.4 d4 — 15m feature importance")
    # Find latest 15m model.
    candidates = sorted(MODELS.glob("BTCUSDT_15m_v*.json"))
    if not candidates:
        print("  NO 15m model files in models/")
        return
    model_path = candidates[-1]  # most recent
    print(f"  using: {model_path.name}")
    model = json.loads(model_path.read_text())

    fi = model.get("feature_importance")
    if not fi:
        print(f"  no feature_importance key. Keys: {list(model.keys())[:20]}")
        return
    if not isinstance(fi, dict):
        print(f"  feature_importance type={type(fi)}, bailing")
        return

    ranked = sorted(fi.items(), key=lambda kv: -kv[1])
    print()
    print(f"  top-10 features:")
    for name, val in ranked[:10]:
        print(f"    {name:<22} {val:.4f}")

    print()
    print("  lag_4 features:")
    for name in ("rsi_14_lag_4", "return_5_lag_4", "vol_ratio_20_lag_4"):
        val = fi.get(name, 0)
        rank_names = [r[0] for r in ranked]
        rank = rank_names.index(name) + 1 if name in rank_names else -1
        print(f"    {name:<22} importance={val:.4f} rank={rank}/{len(fi)}")

    model_ver = model.get("version", "?")
    trained_at = model.get("trained_at", "?")
    print(f"\n  model version: {model_ver}, trained: {trained_at}")
    print(f"  horizon: {model.get('horizon', '?')}, interval: {model.get('interval', '?')}")


def main():
    d1_fold_breakdown()
    d3_threshold_monotonicity()
    d4_feature_importance()
    print()
    print("=" * 84)
    print("B.4 d1/d3/d4 done. d2 needs Phase 3 backtest trades breakdown.")
    print("=" * 84)


if __name__ == "__main__":
    main()
