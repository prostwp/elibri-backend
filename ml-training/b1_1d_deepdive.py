"""
b1_1d_deepdive.py — Session B.1 1d deep dive diagnostics.

Runs 5 checks from the per-TF checklist:
  1. Fold-by-fold stability — all folds positive Sharpe?
  2. Direction balance on HC trades — not 100% short?
  3. Threshold calibration monotonicity
  4. Sample size / SE sanity
  5. Regime split prep — identify which folds map to bull/chop/bear periods

Emits a verdict at the end. Does NOT retrain — operates on existing Phase 1
artifacts (logs/BTCUSDT_1d_metrics.json + best_thresholds.json).
"""
import json
import math
import pathlib

LOGS = pathlib.Path("/root/elibri-backend/ml-training/logs")


def fmt(val, width=8, digits=3, signed=False):
    if val is None:
        return "?".rjust(width)
    fmt_str = f"%+.{digits}f" if signed else f"%.{digits}f"
    return (fmt_str % val).rjust(width)


def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def check_1_fold_stability():
    section("CHECK 1 — Fold-by-fold Sharpe stability (1d CV, horizon=3, tb 2.0/1.5)")
    m = json.loads((LOGS / "BTCUSDT_1d_metrics.json").read_text())
    folds = m["folds"]
    print(f"{'fold':<4} {'n_test':<7} {'acc':<6} {'f1':<6} {'sharpe':<8} "
          f"{'HC_n':<5} {'HC_prec':<8} {'t_end':<12} {'v_end':<12}")
    print("-" * 80)
    negatives = []
    for i, f in enumerate(folds):
        s = f.get("sharpe", 0)
        print(
            f"{f.get('fold', i):<4} {f.get('n_test', 0):<7} "
            f"{fmt(f.get('accuracy'), 6)} {fmt(f.get('f1'), 6)} "
            f"{fmt(s, 8, 2, signed=True)} "
            f"{f.get('hc_trades', 0):<5} {fmt(f.get('hc_precision'), 8)} "
            f"{str(f.get('train_end', ''))[:10]}  {str(f.get('test_end', ''))[:10]}"
        )
        if s < 0:
            negatives.append((f.get('fold', i), s, f.get('test_end')))

    print()
    print(f"Total folds: {len(folds)}")
    print(f"Avg Sharpe: {m['avg_sharpe']:+.2f}")
    if negatives:
        print(f"WARN: {len(negatives)} folds with negative Sharpe:")
        for fold_id, s, end in negatives:
            print(f"  fold {fold_id}: Sharpe={s:+.2f} (test window ends {end})")
    else:
        print("PASS: all folds have non-negative Sharpe")
    return len(folds), len(negatives), m


def check_2_direction_balance(m):
    section("CHECK 2 — Direction balance & HC signal count per fold")
    # Our train.py stores hc_trades and hc_precision per fold but NOT direction.
    # Direction lives only in backtest_v2 trades array — and 1d backtest was
    # never run. So we can only report HC trade counts here and flag the gap.
    folds = m["folds"]
    total_hc = sum(f.get("hc_trades", 0) for f in folds)
    per_fold = [f.get("hc_trades", 0) for f in folds]
    print(f"HC trades per fold: {per_fold}")
    print(f"Total HC trades across {len(folds)} folds: {total_hc}")
    print()
    print("GAP: backtest_v2 was run with `--intervals 5m 15m 1h 4h`, so no 1d")
    print("JSON exists with actual trade direction breakdown. To fully satisfy")
    print("check 2 we'd need: python3 backtest_v2.py --symbol BTCUSDT --intervals 1d")
    print("Flagged for later in this deep dive.")
    return total_hc


def check_3_threshold_monotonicity():
    section("CHECK 3 — Threshold calibration monotonicity (1d)")
    raw = json.loads((LOGS / "best_thresholds.json").read_text())
    row_1d = None
    for r in raw.get("results", []):
        if r.get("interval") == "1d" and r.get("symbol") == "BTCUSDT":
            row_1d = r
            break
    if row_1d is None:
        print("NO 1d row in best_thresholds.json — Phase 2 may not have covered it.")
        return False
    print(f"proba_min={row_1d['proba_min']:.3f} "
          f"proba_max={row_1d['proba_max']:.3f} "
          f"proba_mean={row_1d['proba_mean']:.3f} "
          f"proba_std={row_1d['proba_std']:.3f}")
    print()
    print(f"{'threshold':<10} {'precision':<10} {'n_signals':<10} {'fraction':<10}")
    print("-" * 48)
    rows = []
    for key, stats in sorted(row_1d["hc_table"].items()):
        rows.append(stats)
        print(f"{stats['threshold']:<10.3f} {stats['precision']:<10.3f} "
              f"{stats['n_signals']:<10} {stats['fraction']:<10.4f}")

    # Monotonicity test: precision should be non-decreasing as threshold grows
    # (within samples where n_signals >= 30 — noise territory otherwise).
    dense = [r for r in rows if r["n_signals"] >= 30]
    non_mono_breaks = 0
    last_prec = 0
    for r in dense:
        if r["precision"] < last_prec - 0.02:  # allow 2pp dip
            non_mono_breaks += 1
        last_prec = max(last_prec, r["precision"])
    print()
    if non_mono_breaks == 0:
        print(f"PASS: precision monotonic on dense rows (n>=30), "
              f"no drops >2pp across {len(dense)} points")
    else:
        print(f"WARN: precision non-monotonic — {non_mono_breaks} rows with "
              f"precision drops >2pp vs running max")
    return non_mono_breaks == 0


def check_4_sample_size(total_hc, n_folds):
    section("CHECK 4 — Sample size / statistical power")
    # Standard error of proportion (WR): SE = sqrt(p*(1-p)/N)
    # Standard error of Sharpe (for n trades): SE_sharpe ≈ sqrt((1 + sharpe^2/2)/n)
    # Use most-recent full production setup: 58 OOS signals from best_thresholds
    # holdout, but that's 25% holdout only. Train.py folds cover the full series.
    total = total_hc
    wr_center = 0.69  # 1d HC precision from Phase 2
    se_wr = math.sqrt(wr_center * (1 - wr_center) / max(1, total))
    print(f"Total HC trades across all folds: {total}")
    print(f"Assumed WR center: {wr_center:.1%}")
    print(f"SE(WR) = sqrt(p*(1-p)/n) = {se_wr:.3f} = {se_wr*100:.1f}pp")
    print(f"95% CI on WR: [{max(0, wr_center - 1.96*se_wr):.1%}, "
          f"{min(1, wr_center + 1.96*se_wr):.1%}]")
    print()
    # What does a mission criterion of WR >= 60% require?
    # We need CI lower bound >= 0.60.
    # lower_bound = p - 1.96 * sqrt(p*(1-p)/n) >= 0.60
    # 1.96 * sqrt(p*(1-p)/n) <= p - 0.60
    # n >= (1.96 / (p - 0.60))^2 * p * (1-p)
    if wr_center > 0.60:
        needed = ((1.96 / (wr_center - 0.60)) ** 2) * wr_center * (1 - wr_center)
        print(f"To prove WR > 60% with 95% confidence at observed {wr_center:.0%}, "
              f"need ~{needed:.0f} trades.")
        if total >= needed:
            print(f"PASS: {total} >= {needed:.0f}")
        else:
            print(f"FAIL: {total} < {needed:.0f} — sample too small for 95% CI "
                  f">= 60%. Need {needed - total:.0f} more trades.")
    else:
        print(f"WR center {wr_center:.1%} is BELOW 60% target — sample size moot.")


def check_5_regime_split_prep(m):
    section("CHECK 5 — Regime split — identify which folds cover bull vs chop vs bear")
    # Rough regime labels based on BTC price history:
    #   2018-2019: bear/chop (post-2017 bubble → 2018 crash → 2019 recovery)
    #   2020-Q1-Q3: chop/V-recovery (COVID crash + rebound)
    #   2020-Q4-2021-Q4: bull (40k → 69k)
    #   2022: bear (69k → 15k)
    #   2023: chop/slow recovery
    #   2024-Q1-Q2: bull (45k → 73k)
    #   2024-Q3-Q4: chop
    #   2025-Q1-2026-Q2: bull (70k → 120k+ in the data we have)
    print(f"{'fold':<4} {'test_end':<12} {'sharpe':<9} {'regime label (heuristic)':<40}")
    print("-" * 80)
    for f in m["folds"]:
        end = str(f.get("test_end", ""))[:10]
        s = f.get("sharpe", 0)
        regime = "?"
        if end:
            year = int(end[:4])
            month = int(end[5:7]) if len(end) >= 7 else 0
            if year == 2019:
                regime = "post-bear/chop 2019"
            elif year == 2020:
                regime = "COVID shock + recovery 2020"
            elif year == 2021:
                regime = "bull 2021"
            elif year == 2022:
                regime = "BEAR 2022 (LUNA, FTX)"
            elif year == 2023:
                regime = "chop/recovery 2023"
            elif year == 2024:
                regime = "bull 2024" if month <= 6 else "chop/consolidation 2024 H2"
            elif year == 2025:
                regime = "bull 2025"
            elif year >= 2026:
                regime = "bull 2026 (data end)"
        print(f"{f.get('fold'):<4} {end:<12} {fmt(s, 9, 2, signed=True)}  {regime:<40}")

    print()
    print("NOTE: fold-level regime labels are heuristic. Full check requires")
    print("backtest_v2 with per-period trade slicing, which Session B will do")
    print("once honest 1d backtest exists.")


def main():
    n_folds, n_neg, m = check_1_fold_stability()
    total_hc = check_2_direction_balance(m)
    thresh_pass = check_3_threshold_monotonicity()
    check_4_sample_size(total_hc, n_folds)
    check_5_regime_split_prep(m)

    section("B.1 verdict PREP (incomplete — need 1d backtest_v2)")
    print(f"  Fold stability:     {n_neg}/{n_folds} folds negative → "
          f"{'PASS' if n_neg == 0 else 'FAIL'}")
    print(f"  Threshold mono:     {'PASS' if thresh_pass else 'FAIL'}")
    print(f"  Direction balance:  UNKNOWN (need 1d backtest_v2)")
    print(f"  Honest WR/Sharpe:   UNKNOWN (need 1d backtest_v2)")
    print(f"  Regime split:       UNKNOWN (need per-period trade slicing)")
    print()
    print("Next GPU step: run backtest_v2 with --intervals 1d and --tier all")
    print("(~3 min on RTX 4060 Ti after Phase 4 is killed)")


if __name__ == "__main__":
    main()
