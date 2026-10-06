"""把散落在 results*/ 下的所有 CSV 汇总成论文用的表。

之所以需要一个脚本:2026-10-05 中途改过方法命名(fingerprint 的"无物理/有物理"两个变体
互换过名字),而且结果分布在多个目录里。手工抄数极易出错,这里显式写出映射规则,
**所有论文表格都必须由本脚本生成**。

命名沿革(重要):
  results/fp/ 里的 `fp_rand`  = 现在的 `fp_dann`      (指纹 + 白噪声,无物理)
  results/fp/ 里的 `fp_dann`  = 现在的 `fp_phys_dann` (指纹 + 物理冲击串)

⚠️ 注意:本脚本会把**所有** results*/ 下的历史运行混在一起统计,因此 n 会大于单一批次的
种子数(例如 src_pure 的 n=11 来自多个目录)。混合多个批次的均值**不可直接写进论文**。

**论文 Table I 的唯一数据源是 `results/final/`** —— 那是 7 个方法 × 5 个种子、一次跑完、
命名统一的一批。引用该表时请直接用 `results/final/` 下的 CSV,不要用本脚本的混合汇总。

用法: python tools/collect_results.py [--csv out.csv]
"""
import argparse
import csv
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = {
    "fp_rand": "fp_dann",          # 旧名 -> 新名(仅限 results/fp/)
    "fp_dann": "fp_phys_dann",
}
ORDER = ["src_pure", "source_only", "dann", "rand_dann", "dt_dann",
         "fp_phys_dann", "fp_dann"]


def load_all(roots=()):
    """返回 {method: [(src, tgt, k, seed, acc, f1), ...]},已做命名归一。

    **危险默认**已在 2026-10-06 收敛:原实现把 `results*` 下**一切** CSV 无条件汇总,
    于是不同时期、不同配置(甚至不同 k/方向)的运行被混进同一个均值。这正是
    `paper/tables_source.csv` 里 n=11..22、且与 tab:main 对不上的原因 —— 该文件已
    移出 paper/ 并更名标注。以后要汇总请显式传 roots,例如
    `load_all(("results/final",))`。
    """
    if not roots:
        raise SystemExit("collect_results 现在要求显式指定结果目录,例如 "
                         "`load_all(('results/final',))` —— 见 load_all 的 docstring。")
    rec = {}
    for root in roots:
        pat = os.path.join(HERE, root, "**", "*.csv")
        for path in glob.glob(pat, recursive=True):
            if os.path.basename(path) == "all.csv":
                continue
            try:
                rows = list(csv.DictReader(open(path)))
            except Exception:
                continue
            if not rows or "method" not in rows[0]:
                continue
            for r in rows:
                m = r["method"]
                if os.sep + "fp" + os.sep in path:      # 只有 results/fp/ 用旧命名
                    m = CANON.get(m, m)
                rec.setdefault(m, []).append(
                    (r["src_load"], r["tgt_load"], int(r["k"]), int(r["seed"]),
                     float(r["acc"]), float(r["macro_f1"])))
    return rec


def table(rec, src, tgt, k, name):
    print(f"\n== {name}   (S{src}->S{tgt}, k={k}) ==")
    print(f"{'method':<16}{'n':>4}{'acc':>17}{'macro_f1':>18}")
    print("-" * 55)
    rows = []
    for m in ORDER:
        v = [x for x in rec.get(m, []) if x[0] == src and x[1] == tgt and x[2] == k]
        if not v:
            continue
        a = np.array([[x[4], x[5]] for x in v])
        print(f"{m:<16}{len(v):>4}{a[:,0].mean():>10.4f}±{a[:,0].std():<6.4f}"
              f"{a[:,1].mean():>11.4f}±{a[:,1].std():<6.4f}")
        rows.append((m, len(v), a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="同时写出成 CSV")
    ap.add_argument("--roots", default="results/final",
                    help="要汇总的结果目录,逗号分隔。**务必只给同一批运行**的目录,"
                         "默认 results/final 即 tab:main 的那一批")
    args = ap.parse_args()
    rec = load_all(tuple(args.roots.split(",")))

    tables = []
    tables.append(table(rec, "0", "3", 5, "主表(标准协议)"))
    for s, t in ((1, 3), (3, 0), (3, 1), (3, 2)):
        tables.append(table(rec, str(s), str(t), 5, f"复现:load3 方向 S{s}->S{t}"))
    for k in (1, 3, 10, 20):
        tables.append(table(rec, "0", "3", k, f"k 扫描(k={k})"))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["table", "method", "n", "acc_mean", "acc_std", "f1_mean", "f1_std"])
            for name, rows in zip(["S0->S3 k5"] + [f"S{s}->S{t} k5" for s, t in
                                                   ((1, 3), (3, 0), (3, 1), (3, 2))]
                                  + [f"S0->S3 k{k}" for k in (1, 3, 10, 20)], tables):
                for r in rows:
                    w.writerow([name] + list(r))
        print(f"\n[save] {args.csv}")


if __name__ == "__main__":
    main()
