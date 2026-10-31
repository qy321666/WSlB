"""把 10-seed 的 tab:leak 产物与旧版并排打印,供改正文时逐条引用。

用法: python tools/analyze_leak_s10.py [新产物] [旧产物]
默认: results_paper/tab_leak.json      (20 runs: 10 seeds x 2 reps)
      results_paper/tab_leak_n6legacy.json (6 runs: 3 seeds x 2 reps, 旧稿用的)
"""
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_paper", "tab_leak.json")
OLD = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results_paper", "tab_leak_n6legacy.json")

NAMES = ("support_only", "+fp_aug", "+surrogate", "+true_aug")
LOADS = ("0", "1", "2", "3")


def load(p):
    return json.load(io.open(p, encoding="utf-8"))["by_support_load"]


def main():
    new, old = load(NEW), load(OLD) if os.path.exists(OLD) else {}
    print(f"新产物 {os.path.basename(NEW)}  n_runs={new['0']['rows']['support_only']['n_runs']}")
    print(f"旧产物 {os.path.basename(OLD)}  n_runs={old['0']['rows']['support_only']['n_runs'] if old else '-'}\n")

    print("同录制 / 跨录制 (mean±SD over runs)")
    for L in LOADS:
        print(f"  load{L}")
        for n in NAMES:
            r = new[L]["rows"][n]
            o = old.get(L, {}).get("rows", {}).get(n)
            tail = ""
            if o:
                tail = f"   old same {o['same_mean']:.3f} cross {o['cross_mean']:.3f}"
            print(f"    {n:12s} same {r['same_mean']:.3f}±{r['same_std']:.3f}  "
                  f"cross {r['cross_mean']:.3f}  opt {r['optimism']:+.3f}{tail}")

    print("\n跨录制对比(每行一个支撑域,单位 macro-F1)")
    for a, b in (("+fp_aug", "support_only"), ("+surrogate", "support_only"),
                 ("+true_aug", "support_only"), ("+fp_aug", "+surrogate")):
        vals = [new[L]["rows"][a]["cross_mean"] - new[L]["rows"][b]["cross_mean"] for L in LOADS]
        print(f"  {a:12s} - {b:12s}: " + "  ".join(f"load{L} {v:+.3f}" for L, v in zip(LOADS, vals))
              + f"   mean {np.mean(vals):+.3f}")

    print("\n四支撑域均值")
    for n in NAMES:
        sm = np.mean([new[L]["rows"][n]["same_mean"] for L in LOADS])
        cm = np.mean([new[L]["rows"][n]["cross_mean"] for L in LOADS])
        om = np.mean([new[L]["rows"][n]["optimism"] for L in LOADS])
        print(f"  {n:12s} same {sm:.3f}  cross {cm:.3f}  optimism {om:+.3f}")

    ops = [new[L]["rows"][n]["optimism"] for L in LOADS for n in NAMES]
    print(f"\n全部 16 个带标签单元格的乐观量: min {min(ops):+.3f}  max {max(ops):+.3f}")


if __name__ == "__main__":
    main()
