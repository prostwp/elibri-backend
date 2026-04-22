"""Dump pretty output from a WFCV JSON."""
import json, sys, pathlib

p = pathlib.Path(__file__).parent / "logs" / "iter1_wfcv_4h_sym.json"
if len(sys.argv) > 1:
    p = pathlib.Path(sys.argv[1])
d = json.load(open(p))
print(f"=== {d['label']} {d['symbol']} {d['interval']} tb={d['tb_upper']}/{d['tb_lower']} h={d['horizon']} weighted={d['weighted']} ===")
print(f"{d['folds_used']}/{d['n_folds']} folds, OOS {d['oos_months']}mo each")
print()
print("Per-fold:")
for f in d["per_fold"]:
    ts = f["test_start"][:10]
    te = f["test_end"][:10]
    cd = f["class_dist"]
    print(f"  fold {f['fold']}: test {ts} -> {te}  [train dist: H{cd['hold']*100:.0f}% L{cd['long']*100:.0f}% S{cd['short']*100:.0f}%]")
    for k, r in f["results_by_thr"].items():
        n = r.get("n_trades", 0)
        if n:
            L = r.get("longs", 0); S = r.get("shorts", 0)
            wr = r.get("wr", 0) * 100
            sh = r.get("sharpe", 0); np_ = r.get("net_pct", 0)
            print(f"    {k}: {n}t ({L}L/{S}S) WR={wr:.1f}% S={sh:+.2f} net={np_:+.1f}%")
print("\nAggregate:")
for k, v in d["aggregate"].items():
    if v["folds"] > 0:
        print(f"  {k}: trades={v['trades']}, avg_WR={v['wr_sum']/v['folds']*100:.1f}%, avg_Sharpe={v['sharpe_sum']/v['folds']:+.2f}, total_net={v['net_sum']:+.1f}%, avg_net/fold={v['net_sum']/v['folds']:+.1f}%")
