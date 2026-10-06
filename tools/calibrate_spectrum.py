"""从真实 CWRU **源域**数据标定仿真的谱形参数(只用于把仿真调到像真实设备,不看类别可分性)。

思路:
  - 正常轴承不含故障冲击,它的谱 ≈ 设备背景噪声 + 传递路径的着色 -> 用作**噪声模型**;
  - 故障类的谱减去正常类的谱 -> 冲击成分所在频带 -> 用作**共振带模型**。

输出:共振带范围、衰减常数、噪声着色曲线、低频谐波水平。供 dt_synth 使用。

用法: python tools/calibrate_spectrum.py [负载]
"""
import os
import sys

import numpy as np
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import CLASS_SETS, build_cwru_domain, load_de_channel, segment, zscore_per_window  # noqa: E402

SR, W = 12000, 1024


def avg_psd(X):
    f, P = welch(X, fs=SR, nperseg=W, axis=-1)
    return f, P.mean(axis=0)


def main():
    load = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    classes = CLASS_SETS[10]

    Xs = {}
    for cls in classes:
        X, _ = build_cwru_domain("data/cwru", load, (cls,), window=W)
        Xs[cls] = X

    f, P_norm = avg_psd(Xs["NORMAL"])
    P_norm = P_norm / P_norm.sum()

    faulty = [c for c in classes if c != "NORMAL"]
    P_fault = np.mean([avg_psd(Xs[c])[1] / avg_psd(Xs[c])[1].sum() for c in faulty], axis=0)

    print(f"load{load} —— 各频段归一化能量占比(%)\n")
    bands = ((0, 200), (200, 800), (800, 1500), (1500, 2500), (2500, 3000),
             (3000, 3500), (3500, 4000), (4000, 4500), (4500, 5000), (5000, 6000))
    print(f"{'频段':<16}{'正常(噪声模型)':>16}{'故障类平均':>14}")
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        print(f"  {lo:>5}-{hi:<6} Hz{100 * P_norm[m].sum():>15.2f}{100 * P_fault[m].sum():>14.2f}")

    # 共振带:故障谱相对正常谱增益最大的位置
    ratio = P_fault / (P_norm + 1e-18)
    m = (f >= 500) & (f <= 5800)
    peak_f = f[m][np.argmax(ratio[m])]
    print(f"\n故障/正常 能量比最高的频率: {peak_f:.0f} Hz  (比值 {ratio[m].max():.1f})")

    # 共振带宽度:比值 > 峰值一半的范围
    half = ratio[m] > ratio[m].max() / 2
    print(f"共振带半高宽范围: {f[m][half].min():.0f} - {f[m][half].max():.0f} Hz")

    # 噪声模型:正常类 PSD 的累积分布,给出主要能量的频率范围
    c = np.cumsum(P_norm)
    lo95 = f[np.searchsorted(c, 0.05)]
    hi95 = f[np.searchsorted(c, 0.95)]
    print(f"\n正常类 5%-95% 能量集中在: {lo95:.0f} - {hi95:.0f} Hz")
    print(f"正常类 90% 能量上限频率: {f[np.searchsorted(c, 0.90)]:.0f} Hz")

    # 输出供 dt_synth 用的噪声着色曲线(50 个频点,归一化幅值)
    npts = 25
    edges = np.linspace(0, SR / 2, npts + 1)
    shape = []
    for i in range(npts):
        mm = (f >= edges[i]) & (f < edges[i + 1])
        shape.append(float(np.sqrt(P_norm[mm].mean())) if mm.any() else 0.0)
    print(f"\n噪声着色曲线({npts} 段, 0-6000Hz, 已归一化到最大 1):")
    s = np.array(shape)
    s = s / (s.max() + 1e-18)
    print("  " + ", ".join(f"{v:.3f}" for v in s))


if __name__ == "__main__":
    main()
