#!/usr/bin/env python3
import os, json, csv, glob, sys

# ---------- Robust root discovery ----------
# Default: results_by_type sibling to the 'tools' directory (i.e. repo/results_by_type)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results_by_type"))

# Allow override via CLI: python tools/merge.py /path/to/results_by_type
ROOT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT

print(f"[merge_metrics] Using results root: {ROOT}")

def collect_type(type_dir):
    """Walk type_dir and collect any json files that look like metrics outputs."""
    rows = []
    # Walk recursively
    for root, dirs, files in os.walk(type_dir):
        for fn in files:
            # match common patterns ending with _metrics.json or containing "metrics" in name
            if fn.endswith("_metrics.json") or "metrics" in fn.lower():
                p = os.path.join(root, fn)
                try:
                    with open(p, "r") as f:
                        data = json.load(f)
                    rows.append(data)
                except Exception as e:
                    print("Failed to read", p, e)
    return rows


def write_csv(type_dir, rows):
    if not rows:
        print("No rows for", type_dir)
        return
    csv_path = os.path.join(type_dir, "aggregated.csv")
    fieldnames = ["artifact", "type", "seed", "run_version", "morph", "task", "episodes", "avg_reward", "std_reward", "success_rate_pct", "episode_returns"]
    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # ensure episode_returns are stringified
            r2 = {k: r.get(k, "") for k in fieldnames}
            r2["episode_returns"] = json.dumps(r.get("episode_returns", []))
            writer.writerow(r2)
    print("Wrote", csv_path)

def main():
    if not os.path.isdir(ROOT):
        print("No results root:", ROOT)
        return
    for type_name in sorted(os.listdir(ROOT)):
        type_dir = os.path.join(ROOT, type_name)
        if not os.path.isdir(type_dir):
            continue
        rows = collect_type(type_dir)
        write_csv(type_dir, rows)

if __name__ == "__main__":
    main()
