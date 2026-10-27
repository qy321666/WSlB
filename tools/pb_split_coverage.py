"""十 seed 各抽到哪些留出轴承:数一数覆盖了多少只。

三层分解每个 seed 都重新抽一次"每类一颗留出轴承"(balanced_split),但那只
记录在内存里,没进 pb_levels.json。于是"十 seed 到底覆盖了多少只试件"这个问题
只能靠重算 —— 这里只重放抽样这一步(不训练、不读信号),产物是
results_pb/pb_split_coverage.json,供正文那句"覆盖 15/21"引用。

前提:list_bearings 的文件顺序稳定,且种子相同时 RandomState 序列一致 ——
与被引用的 pb_levels*.json 是同一个 seed、同一个函数,故可重放。

用法: python tools/pb_split_coverage.py [n_seed]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pb_utils import CLASSES_LOC, class_of_location, damage_table, list_bearings  # noqa: E402
from probe_pb_levels import ROOT, RES, balanced_split  # noqa: E402


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tbl = damage_table(ROOT)
    lab = lambda b: class_of_location(b, tbl)  # noqa: E731
    codes = [b for b in list_bearings(ROOT) if lab(b)]

    per_seed, seen = {}, set()
    for seed in range(n_seed):
        _, te = balanced_split(lab, codes, np.random.RandomState(seed))
        per_seed[seed] = sorted(te)
        seen |= set(te)

    by_class = {c: len([b for b in codes if lab(b) == c]) for c in CLASSES_LOC}
    out = {"n_seed": n_seed, "bearings_total": len(codes), "bearings_by_class": by_class,
           "per_seed": per_seed, "distinct_held_out": sorted(seen),
           "distinct_by_class": {c: sorted(b for b in seen if lab(b) == c)
                                 for c in CLASSES_LOC},
           "n_distinct_held_out": len(seen)}
    print(f"  {n_seed} seed:每 seed 留出 {len(per_seed[0])} 只(每类一颗);"
          f"一共抽到 {len(seen)} / {len(codes)} 只不重复的轴承")
    for c in CLASSES_LOC:
        sub = [b for b in sorted(seen) if lab(b) == c]
        print(f"    {c:<10} {len(sub):>2} / {by_class[c]}")
    os.makedirs(RES, exist_ok=True)
    fn = os.path.join(RES, "pb_split_coverage.json")
    json.dump(out, open(fn, "w"), indent=1)
    print(f"  -> {fn}")


if __name__ == "__main__":
    main()
