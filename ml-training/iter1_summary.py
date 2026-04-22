"""iter1_summary.py — Patch 4 Iter 1 summary tables generator.

Reads:
  logs/iter_backtest/BTCUSDT_15m_iter_3class.json (per threshold)
  logs/iter1b_backtest/BTCUSDT_15m_iter1b_thr*.json (with MTF)

Emits markdown table ready to paste into DEVLOG or MORNING_STATUS.
Usage:
  python3 iter1_summary.py
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent
LOGS = ROOT / "logs"


def load_tier_json(pattern):
    """Load all JSONs matching glob, return list of dicts."""
    results = []
    for p in sorted(LOGS.glob(pattern)):
        try:
            j = json.loads(p.read_text())
            j["_path"] = str(p)
            results.append(j)
        except Exception as e:
            print(f"  skip {p}: {e}", file=sys.stderr)
    return results


def fmt_row(r, include_mtf_fields=False):
    n = r.get("n_trades", 0)
    longs = r.get("longs", 0)
    shorts = r.get("shorts", 0)
    short_pct = shorts / max(1, n) * 100
    wr = r.get("wr", 0) * 100
    sharpe = r.get("sharpe", 0)
    ng = r.get("net_gross", 0)
    net = r.get("net_pct", 0)
    tpd = r.get("trades_per_day", 0)
    long_thr = r.get("long_thr", 0)
    base = (
        f"| {long_thr:.2f} | {n} | {longs}/{shorts} | {short_pct:.0f}% | "
        f"{wr:.1f}% | {sharpe:+.2f} | {net:+.2f}% | {ng:.2f} | {tpd:.2f} |"
    )
    return base


def main():
    print("# Patch 4 Iter 1 — summary tables\n")

    # Iter 1 (no MTF).
    iter1_results = load_tier_json("iter_backtest/BTCUSDT_15m_iter_3class.json")
    # Also check differently named files just in case.
    if not iter1_results:
        iter1_results = load_tier_json("iter_backtest/*iter_3class*.json")

    print("## Table 1 — Iter 1 without MTF filters\n")
    print("| Threshold | Trades | L/S | Short% | WR | Sharpe | Net% | Net/Gross | Trades/day |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in iter1_results:
        print(fmt_row(r))
    if not iter1_results:
        print("| (no iter_3class results found) | | | | | | | | |")

    print()

    # Iter 1b MTF.
    iter1b_results = load_tier_json("iter1b_backtest/BTCUSDT_15m_iter1b_thr*.json")
    print("## Table 2 — Iter 1b with MTF filters (1d trend + 5m momentum)\n")
    print("| Threshold | Trades | L/S | Short% | WR | Sharpe | Net% | Net/Gross | Trades/day |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in iter1b_results:
        print(fmt_row(r))
    if not iter1b_results:
        print("| (no iter1b results found) | | | | | | | | |")

    print()

    # Compare to baseline.
    print("## Comparison to B.4 baseline (before Patch 4)\n")
    print("| Метрика | B.4 baseline (Conservative 15m) | Best Iter 1 | Best Iter 1b |")
    print("|---|---|---|---|")
    best_iter1 = max(iter1_results, key=lambda r: r.get("sharpe", -999), default=None)
    best_iter1b = max(iter1b_results, key=lambda r: r.get("sharpe", -999), default=None)
    b4_cons = {"sharpe": -5.38, "wr": 0.40, "longs": 0, "shorts": 25}

    def brief(r):
        if not r:
            return "n/a"
        sharpe = r.get("sharpe", 0)
        wr = r.get("wr", 0) * 100
        longs = r.get("longs", 0)
        shorts = r.get("shorts", 0)
        thr = r.get("long_thr", 0)
        return f"thr={thr:.2f}: Sharpe {sharpe:+.2f}, WR {wr:.0f}%, L/S {longs}/{shorts}"

    print(f"| Sharpe | {b4_cons['sharpe']:+.2f} | "
          f"{best_iter1.get('sharpe', 0):+.2f} | "
          f"{best_iter1b.get('sharpe', 0):+.2f} |" if best_iter1 and best_iter1b
          else f"| Sharpe | {b4_cons['sharpe']:+.2f} | n/a | n/a |")
    print(f"| WR | {b4_cons['wr']*100:.0f}% | "
          f"{best_iter1.get('wr', 0)*100:.0f}% | "
          f"{best_iter1b.get('wr', 0)*100:.0f}% |" if best_iter1 and best_iter1b
          else f"| WR | {b4_cons['wr']*100:.0f}% | n/a | n/a |")
    print(f"| Longs/Shorts | 0/25 (100% short) | "
          f"{best_iter1.get('longs', 0)}/{best_iter1.get('shorts', 0)} | "
          f"{best_iter1b.get('longs', 0)}/{best_iter1b.get('shorts', 0)} |" if best_iter1 and best_iter1b
          else f"| Longs/Shorts | 0/25 | n/a | n/a |")

    print()
    print("## Brief verdict per iteration\n")
    print(f"- Best Iter 1 (no MTF): {brief(best_iter1)}")
    print(f"- Best Iter 1b (MTF):  {brief(best_iter1b)}")


if __name__ == "__main__":
    main()
