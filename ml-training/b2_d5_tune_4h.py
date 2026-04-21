"""
b2_d5_tune_4h.py — Session B.2 d5 threshold-tuning experiment.

Rewrites logs/b1_backtest_1d/best_thresholds_tuned_4h.json with 4h threshold
changed from 0.750 to 0.725 (per d4 monotonicity analysis: peak precision at
0.725, not 0.750). Then runs backtest_v2 only on 4h with this tuned config.

Goal: does +48% more trade candidates improve tier-level Sharpe or degrade it?
"""
import json
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path("/root/elibri-backend/ml-training")
BT_DIR = ROOT / "logs"
SRC = BT_DIR / "best_thresholds.json"

# Step 1 — load canonical, patch 4h only.
data = json.loads(SRC.read_text())
for r in data["results"]:
    if r["interval"] == "4h" and r["symbol"] == "BTCUSDT":
        # Old: 0.750 / precision 0.687 / n=441. New: 0.725 / precision 0.696 / n=652.
        old = r["best"]["threshold"]
        new = 0.725
        key = f"thr_{new:.3f}"
        tuned = r["hc_table"][key]
        r["best"] = {
            "key": key,
            "policy": "b2_d5_tuned_peak_precision",
            "sig_per_day": tuned["fraction"] * 6.0,  # 6 bars/day on 4h
            **tuned,
        }
        print(f"B.2 d5: 4h threshold {old} -> {new} (precision "
              f"{tuned['precision']:.3f}, n_signals {tuned['n_signals']})")
        break
else:
    raise SystemExit("no BTCUSDT 4h row in best_thresholds.json")

# Step 2 — save tuned copy + swap.
tuned_path = BT_DIR / "best_thresholds_b2_tuned.json"
tuned_path.write_text(json.dumps(data, indent=2))
print(f"wrote {tuned_path}")

# Step 3 — swap in tuned, keep backup.
backup = BT_DIR / "best_thresholds_canonical.json"
if not backup.exists():
    shutil.copy(SRC, backup)
    print(f"backed up canonical to {backup}")
shutil.copy(tuned_path, SRC)
print(f"installed tuned thresholds -> {SRC}")

# Step 4 — run backtest_v2 on 4h only.
out_dir = BT_DIR / "b2_backtest_4h_tuned"
out_dir.mkdir(exist_ok=True)
cmd = [
    "python3", "-u", "backtest_v2.py",
    "--symbol", "BTCUSDT",
    "--intervals", "4h",
    "--tier", "all",
    "--use-best-thresholds",
    "--out-dir", str(out_dir),
]
print(f"running: {' '.join(cmd)}")
log = BT_DIR / "b2_4h_tuned_backtest.log"
with log.open("w") as f:
    proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
print(f"backtest exit={proc.returncode}, log={log}")

# Step 5 — always restore canonical thresholds so other B.* analyses stay pure.
shutil.copy(backup, SRC)
print(f"restored canonical thresholds from {backup}")
