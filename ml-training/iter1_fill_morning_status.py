"""iter1_fill_morning_status.py — fills MORNING_STATUS.md placeholder tables
with actual numbers from the night run's JSON artifacts.

Reads:
  logs/iter_backtest/BTCUSDT_15m_iter_3class.json (all thr backtests)
  logs/iter1b_backtest/BTCUSDT_15m_iter1b_thr*.json (MTF variant)
  logs/iter1_expB_metrics.json  (sample-weighted)
  logs/iter1_expC_metrics.json  (recent-only)
  logs/iter1_expA_metrics.json  (flipped)

Writes: MORNING_STATUS.md with ⟨placeholder⟩ entries replaced.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).parent
LOGS = ROOT / "logs"
MORNING = ROOT.parent.parent / "MORNING_STATUS.md"


def fmt_pct(v):
    return f"{v*100:.0f}%" if v is not None else "n/a"


def fmt_sharpe(v):
    return f"{v:+.2f}" if v is not None else "n/a"


def fmt_num(v):
    return str(v) if v is not None else "n/a"


def read_iter1_backtests():
    """Scan logs/iter_backtest/ for per-thr JSONs."""
    results = {}
    d = LOGS / "iter_backtest"
    if not d.exists():
        return results
    for p in sorted(d.glob("*iter_3class*.json")):
        j = json.loads(p.read_text())
        thr = j.get("long_thr", 0)
        results[f"thr_{int(thr*100):02d}"] = j
    return results


def read_iter1b():
    results = {}
    d = LOGS / "iter1b_backtest"
    if not d.exists():
        return results
    for p in sorted(d.glob("*iter1b*.json")):
        j = json.loads(p.read_text())
        thr = j.get("long_thr", 0)
        results[f"thr_{int(thr*100):02d}"] = j
    return results


def read_exp(label):
    p = LOGS / f"iter1_{label}_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def render_iter1_table(results):
    if not results:
        return "| (no iter1 backtest results) | | | | | | | | |"
    rows = []
    for thr_key in sorted(results.keys()):
        r = results[thr_key]
        n = r.get("n_trades", 0)
        longs = r.get("longs", 0)
        shorts = r.get("shorts", 0)
        short_pct = (shorts / n * 100) if n else 0
        wr_long = (sum(1 for t in r.get("trades", []) if t.get("direction") == 1 and t.get("pnl_dollars", 0) > 0)
                   / max(1, longs) * 100) if r.get("trades") else 0
        wr_short = (sum(1 for t in r.get("trades", []) if t.get("direction") == -1 and t.get("pnl_dollars", 0) > 0)
                    / max(1, shorts) * 100) if r.get("trades") else 0
        wr = r.get("wr", 0) * 100
        sharpe = r.get("sharpe", 0)
        ng = r.get("net_gross", 0)
        thr = r.get("long_thr", 0)
        rows.append(
            f"| {thr:.2f} | {n} | {longs}/{shorts} | {short_pct:.0f}% "
            f"| {wr_long:.0f}% | {wr_short:.0f}% | {wr:.0f}% "
            f"| {sharpe:+.2f} | {ng:.2f} |"
        )
    return "\n".join(rows)


def render_iter1b_table(results):
    if not results:
        return "| (no iter1b results) | | | | | |"
    rows = []
    for thr_key in sorted(results.keys()):
        r = results[thr_key]
        n = r.get("n_trades", 0)
        longs = r.get("longs", 0)
        shorts = r.get("shorts", 0)
        short_pct = (shorts / n * 100) if n else 0
        wr = r.get("wr", 0) * 100
        sharpe = r.get("sharpe", 0)
        thr = r.get("long_thr", 0)
        rows.append(
            f"| {thr:.2f} | {n} | {longs}/{shorts} | {short_pct:.0f}% "
            f"| {wr:.0f}% | {sharpe:+.2f} |"
        )
    return "\n".join(rows)


def render_exp_row(label, exp):
    """Pick best threshold from an experiment's results_by_thr."""
    if not exp:
        return f"| {label} | — | — | — | — | — |"
    best_key = None
    best_sharpe = -999
    for thr_key, r in exp.get("results_by_thr", {}).items():
        if r.get("sharpe", -999) > best_sharpe:
            best_sharpe = r.get("sharpe", -999)
            best_key = thr_key
    if best_key is None:
        return f"| {label} | — | — | — | — | — |"
    r = exp["results_by_thr"][best_key]
    n = r.get("n_trades", 0)
    longs = r.get("longs", 0)
    shorts = r.get("shorts", 0)
    wr = r.get("wr_overall", 0) * 100
    sharpe = r.get("sharpe", 0)
    ng = r.get("net_gross", 0)
    return f"| {label} ({best_key}) | {n} | {longs}/{shorts} | {wr:.0f}% | {sharpe:+.2f} | {ng:.2f} |"


def main():
    iter1_results = read_iter1_backtests()
    iter1b_results = read_iter1b()
    expA = read_exp("expA")
    expB = read_exp("expB")
    expC = read_exp("expC")

    out = []
    out.append("# Ночной автономный прогон — статус к утру\n")
    out.append("**Обновлён автоматически** скриптом iter1_fill_morning_status.py.\n")
    out.append("Создан агентом в ночь 2026-04-24 для презентации в 11:00.\n\n")
    out.append("---\n\n")

    out.append("## TL;DR\n\n")
    out.append("Итерация 1 Патч 4 завершилась. Эксперименты с регим-сплитом выполнены.\n\n")

    # Best result across all.
    best_overall = None
    best_label = "none"
    best_sharpe = -999
    for label, exp in [("expA", expA), ("expB", expB), ("expC", expC)]:
        if not exp:
            continue
        for thr_key, r in exp.get("results_by_thr", {}).items():
            sh = r.get("sharpe", -999)
            if sh > best_sharpe:
                best_sharpe = sh
                best_overall = r
                best_label = f"{label}/{thr_key}"

    if best_overall:
        out.append(f"**Лучший результат из всех экспериментов:** {best_label}\n")
        out.append(f"- Sharpe: {best_overall.get('sharpe', 0):+.2f}\n")
        out.append(f"- WR: {best_overall.get('wr_overall', 0)*100:.0f}%\n")
        out.append(f"- Longs/Shorts: {best_overall.get('longs', 0)}/{best_overall.get('shorts', 0)}\n")
        out.append(f"- Net/Gross: {best_overall.get('net_gross', 0):.2f}\n\n")
    else:
        out.append("**Лучший результат:** (эксперименты не отработали)\n\n")

    out.append("---\n\n")
    out.append("## Таблица 1 — Iter 1 без MTF (baseline трёхклассовой модели)\n\n")
    out.append("| Threshold | Trades | L/S | Short% | WR long | WR short | WR overall | Sharpe | Net/Gross |\n")
    out.append("|---|---|---|---|---|---|---|---|---|\n")
    out.append(render_iter1_table(iter1_results) + "\n\n")

    out.append("## Таблица 2 — Iter 1b с MTF (1d trend + 5m momentum)\n\n")
    out.append("| Threshold | Trades | L/S | Short% | WR | Sharpe |\n")
    out.append("|---|---|---|---|---|---|\n")
    out.append(render_iter1b_table(iter1b_results) + "\n\n")

    out.append("## Таблица 3 — Регим-сплит эксперименты\n\n")
    out.append("| Эксперимент | Trades | L/S | WR | Sharpe | Net/Gross |\n")
    out.append("|---|---|---|---|---|---|\n")
    out.append(render_exp_row("Baseline (full 8y WF CV)", None) + "\n")  # filled manually
    out.append(render_exp_row("A — Flipped split", expA) + "\n")
    out.append(render_exp_row("B — Sample weights", expB) + "\n")
    out.append(render_exp_row("C — Recent-only", expC) + "\n")
    out.append("\n---\n\n")

    out.append("## Файлы к утру\n\n")
    out.append("- `elibri-backend/ml-training/feature_engine.py` — 3-класс target + regime_score\n")
    out.append("- `elibri-backend/ml-training/train.py` — train_ensemble_3class + GPU autodetect\n")
    out.append("- `elibri-backend/ml-training/iter_backtest_3class.py` — dual-threshold backtest\n")
    out.append("- `elibri-backend/ml-training/iter1b_backtest_mtf.py` — MTF-aware backtest\n")
    out.append("- `elibri-backend/ml-training/iter1_exp_b_sample_weights.py` — эксперименты A/B/C\n")
    out.append("- `elibri-backend/ml-training/gpu_probe.py` — GPU diagnostic\n")
    out.append("- `elibri-backend/ml-training/iter1_summary.py` — таблицы сводные\n")
    out.append("- `elibri-backend/ml-training/iter1_fill_morning_status.py` — автозаполнение этого файла\n\n")

    p_out = pathlib.Path("/Users/admin/NodeElibiri/MORNING_STATUS.md")
    # Don't overwrite from the run machine — emit to stdout + alternate file.
    # On vast.ai this path won't exist, so write to ml-training/logs/.
    alt_out = LOGS / "MORNING_STATUS_auto.md"
    text = "".join(out)
    alt_out.write_text(text)
    print(f"Wrote {alt_out} ({len(text)} chars)")
    print()
    print("=" * 72)
    print(text)


if __name__ == "__main__":
    main()
