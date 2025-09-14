# compute_ci.py (replace your file with this)
import sys
import os
import math
import glob
import json
import pandas as pd
import numpy as np

# try import scipy.t for accurate t-quantiles; fallback allowed
try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

def mean_ci_from_array(arr, alpha=0.05):
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    if n == 0:
        return None, None, (None, None), n
    m = float(arr.mean())
    if n <= 1:
        # can't estimate variance meaningfully with n==1; half-width = 0
        return m, 0.0, (m, m), n
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n)
    if HAVE_SCIPY:
        t = stats.t.ppf(1 - alpha/2, n - 1)
        half = float(t * se)
    else:
        # fallback: normal z (approx) and warn
        z = 1.959963984540054  # 97.5% quantile
        half = float(z * se)
    return m, half, (m - half, m + half), n

def analyze_csv(path):
    if not os.path.exists(path):
        print(f"[SKIP] File not found: {path}")
        return
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Could not read CSV '{path}': {e}")
        return

    print("\n" + "-"*60)
    print(f"Analyzing: {path}")

    for col in ["avg_reward", "success_rate_pct"]:
        if col not in df.columns:
            print(f"  [WARN] Column '{col}' not found in {path}. Available columns: {list(df.columns)}")
            continue
        series = df[col].dropna().astype(float)
        m, half, (lo, hi), n = mean_ci_from_array(series.values)
        if n == 0:
            print(f"  {col}: no data (n=0)")
        else:
            if n < 2:
                print(f"  {col}: n={n} (warning: n < 2; CI not meaningful). mean={m:.4f}, half-width={half:.4f}, CI=[{lo:.4f}, {hi:.4f}]")
            else:
                method = "t-dist" if HAVE_SCIPY else "z-approx (no scipy)"
                print(f"  {col}: n={n}, mean={m:.4f}, 95% CI = [{lo:.4f}, {hi:.4f}] (half-width={half:.4f}, method={method})")
            # show raw values for quick sanity check
            print(f"    raw values: {series.values.tolist()}")
    print("-"*60 + "\n")

def main():
    # If user gave one or more paths, use them. Otherwise auto-discover aggregated.csv under results_by_type/*
    args = sys.argv[1:]
    targets = []
    if args:
        for a in args:
            # expand globs
            for p in glob.glob(a):
                targets.append(p)
    else:
        # auto-discover
        base = "results_by_type"
        if not os.path.isdir(base):
            print(f"[ERROR] Auto-discovery failed: '{base}' not found. Provide path(s) to aggregated.csv on the command line.")
            print("Usage: python compute_ci.py results_by_type/allnodes/aggregated.csv")
            return
        candidates = glob.glob(os.path.join(base, "*", "aggregated.csv"))
        if not candidates:
            print(f"[ERROR] No aggregated.csv files found under '{base}/*/aggregated.csv'. Provide a path explicitly.")
            return
        targets = sorted(candidates)

    if not targets:
        print("[ERROR] No CSV targets found. Exiting.")
        return

    for t in targets:
        analyze_csv(t)

if __name__ == "__main__":
    main()
