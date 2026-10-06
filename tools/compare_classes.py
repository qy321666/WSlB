"""逐类对照:真实 vs 合成的谱形差异。

检验孪生模型是否复现了**类别之间**的差异(而不只是整体谱形)。
若各类真实谱在某个维度上有明确排序(如缺陷越大某频带越强),而合成没有,说明
孪生学不到真实的类别结构 —— 这正是"合成域训练在真实域只有随机水平"的原因。

用法: python tools/compare_classes.py [负载] [每类窗数]
"""
import os
import sys

import numpy as np
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import CLASS_SETS, build_cwru_domain  # noqa: E402
from dt_synth import CWRU_FR, synth_dataset  # noqa: E402

SR, W = 12000, 1024
BANDS = ((0, 200), (200, 800), (800, 2000), (2000, 2800), (2800, 3300),
         (3300, 3800), (3800, 4500), (4500, 6000))


def band_shares(X):
    f, P = welch(X, fs=SR, nperseg=W, axis=-1)
    P = P.mean(axis=0)
    P = P / P.sum()
    return np.array([P[(f >= lo) & (f < hi)].sum() for lo, hi in BANDS])


def main():
    load = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    classes = CLASS_SETS[10]

    hdr = "".join(f"{f'{lo}-{hi}':>10}" for lo, hi in BANDS)
    print(f"各频段能量占比(%) —— load{load}\n")
    print(f"{'类别':<8}{'来源':<6}{hdr}")
    print("-" * (14 + 10 * len(BANDS)))

    real_shares, syn_shares = {}, {}
    for cls in classes:
        Xr, _ = build_cwru_domain("data/cwru", load, (cls,), window=W)
        idx = np.linspace(0, len(Xr) - 1, min(n, len(Xr))).astype(int)
        sr_ = band_shares(Xr[idx])
        Xs, _ = synth_dataset((cls,), n, W, SR, CWRU_FR[load], seed=0)
        ss_ = band_shares(Xs)
        real_shares[cls], syn_shares[cls] = sr_, ss_
        print(f"{cls:<8}{'真实':<6}" + "".join(f"{100 * v:>10.2f}" for v in sr_))
        print(f"{'':<8}{'合成':<6}" + "".join(f"{100 * v:>10.2f}" for v in ss_))
        print()

    # 类间差异:同一频带上,各真实类别取值的离散程度(越大=该频带有判别力)
    print("类间差异(各频带上 10 个类别取值的标准差 ×1000):")
    print(f"{'':<22}" + "".join(f"{f'{lo}-{hi}':>10}" for lo, hi in BANDS))
    for name, d in (("真实", real_shares), ("合成", syn_shares)):
        M = np.stack([d[c] for c in classes])
        print(f"  {name:<20}" + "".join(f"{1000 * v:>10.2f}" for v in M.std(axis=0)))

    # 尺寸是否有单调趋势(以 IR 为例)
    print("\n尺寸趋势检验(IR 类在 2800-3300Hz 的能量占比,应随缺陷增大单调变化):")
    for name, d in (("真实", real_shares), ("合成", syn_shares)):
        vals = [d[f"IR{s}"][4] for s in ("007", "014", "021")]
        print(f"  {name}: IR007={100*vals[0]:.2f}%  IR014={100*vals[1]:.2f}%  "
              f"IR021={100*vals[2]:.2f}%   单调={vals[0]<vals[1]<vals[2] or vals[0]>vals[1]>vals[2]}")


if __name__ == "__main__":
    main()
