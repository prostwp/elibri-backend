"""gpu_probe.py — check XGBoost/LightGBM GPU availability for multi-class training.

Tests:
  1. XGBoost version + CUDA multi-class fit (device='cuda', tree_method='hist').
  2. LightGBM version + GPU multi-class fit (device='gpu').
  3. Compare wall time vs CPU.

Runs on small synthetic dataset (200 rows × 10 features, 3 classes).
"""
import sys
import time

import numpy as np


def probe_xgb_cuda():
    print("\n=== XGBoost CUDA probe ===")
    import xgboost as xgb
    print(f"  version: {xgb.__version__}")
    from xgboost import XGBClassifier

    np.random.seed(0)
    X = np.random.rand(5000, 32).astype(np.float32)
    y = np.random.randint(0, 3, 5000)

    # CPU reference.
    t0 = time.time()
    cpu = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", n_jobs=4, random_state=42,
        tree_method="hist", verbosity=0,
    )
    cpu.fit(X, y)
    cpu_time = time.time() - t0
    print(f"  CPU fit: {cpu_time:.2f}s")

    # CUDA attempt.
    try:
        t0 = time.time()
        gpu = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", n_jobs=4, random_state=42,
            tree_method="hist", device="cuda", verbosity=0,
        )
        gpu.fit(X, y)
        gpu_time = time.time() - t0
        p = gpu.predict_proba(X[:5])
        print(f"  CUDA fit: {gpu_time:.2f}s  speedup: {cpu_time/gpu_time:.1f}x")
        print(f"  CUDA proba shape: {p.shape}")
        return True, gpu_time, cpu_time
    except Exception as e:
        print(f"  CUDA fit FAILED: {repr(e)}")
        return False, None, cpu_time


def probe_lgbm_gpu():
    print("\n=== LightGBM GPU probe ===")
    import lightgbm as lgb
    print(f"  version: {lgb.__version__}")
    from lightgbm import LGBMClassifier

    np.random.seed(0)
    X = np.random.rand(5000, 32).astype(np.float32)
    y = np.random.randint(0, 3, 5000)

    # CPU reference.
    t0 = time.time()
    cpu = LGBMClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multiclass", num_class=3,
        n_jobs=4, random_state=42, verbose=-1,
    )
    cpu.fit(X, y)
    cpu_time = time.time() - t0
    print(f"  CPU fit: {cpu_time:.2f}s")

    # GPU attempt.
    try:
        t0 = time.time()
        gpu = LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multiclass", num_class=3,
            device="gpu", gpu_device_id=0,
            n_jobs=4, random_state=42, verbose=-1,
        )
        gpu.fit(X, y)
        gpu_time = time.time() - t0
        print(f"  GPU fit: {gpu_time:.2f}s  speedup: {cpu_time/gpu_time:.1f}x")
        return True, gpu_time, cpu_time
    except Exception as e:
        print(f"  GPU fit FAILED: {repr(e)}")
        return False, None, cpu_time


def main():
    xgb_ok, xgb_gpu_t, xgb_cpu_t = probe_xgb_cuda()
    lgbm_ok, lgbm_gpu_t, lgbm_cpu_t = probe_lgbm_gpu()

    print("\n=== Recommendations ===")
    if xgb_ok:
        sp = xgb_cpu_t / xgb_gpu_t
        print(f"  XGBoost: use device='cuda', tree_method='hist' (~{sp:.1f}x vs CPU)")
    else:
        print(f"  XGBoost: stay on CPU (GPU not working — see error above)")
    if lgbm_ok:
        sp = lgbm_cpu_t / lgbm_gpu_t
        print(f"  LightGBM: use device='gpu' (~{sp:.1f}x vs CPU)")
    else:
        print(f"  LightGBM: stay on CPU (GPU not working — see error above)")


if __name__ == "__main__":
    main()
