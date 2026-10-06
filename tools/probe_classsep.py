"""机制诊断:各信号源到底复现了"类间差异"没有?

把每一类的平均幅度谱当作该类的"指纹",计算**类间余弦距离矩阵**并取其均值。
- 真实数据里类间差异大(那是录制指纹,见 接手文档 4.6 节);
- 若某个信号源生成的各类谱几乎一样,则它的类间平均距离会趋近 0 —— 该信号源
  在真实域上就不可能被分类。

对照:真实目标域 / 原物理孪生 / 目标指纹着色白噪声 / 目标指纹 + 冲击串。

用法: python tools/probe_classsep.py [tgt_load] [n_per_class]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import CLASS_SETS, build_cwru_domain, split_target_fewshot  # noqa: E402
from dt_synth import CWRU_FR, synth_dataset  # noqa: E402
from twin_calib import estimate_fingerprints, synth_target_like  # noqa: E402

WINDOW = 1024


def class_mean_spectra(X, y, n_class):
    """逐类平均幅度谱,每行归一化到单位 L2 范数。"""
    win = np.hanning(X.shape[-1])
    A = np.stack([np.abs(np.fft.rfft(X[y == c] * win, axis=-1)).mean(axis=0)
                  for c in range(n_class)])
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def sep(A):
    """类间平均余弦距离(1 - 余弦相似度),以及最小的那一对。"""
    S = A @ A.T
    n = len(A)
    iu = np.triu_indices(n, 1)
    cos = S[iu]
    j = int(np.argmin(cos))
    return float(1 - cos.mean()), float(1 - cos.min()), (iu[0][j], iu[1][j])


def main():
    tgt_load = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    n_per_class = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    classes = CLASS_SETS[10]
    n_class = len(classes)

    X_tgt, y_tgt = build_cwru_domain("data/cwru", tgt_load, classes, window=WINDOW)
    X_src, y_src = build_cwru_domain("data/cwru", 0, classes, window=WINDOW)
    sup_idx, _, _ = split_target_fewshot(X_tgt, y_tgt, 5, 0)
    A_k = estimate_fingerprints(X_tgt[sup_idx], y_tgt[sup_idx], n_class, window=WINDOW)
    A_env = estimate_fingerprints(X_tgt[sup_idx], y_tgt[sup_idx], n_class, window=WINDOW,
                                  envelope=True)

    y_gen = np.repeat(np.arange(n_class), n_per_class)
    sources = {
        "真实目标域 load%d" % tgt_load: (X_tgt, y_tgt),
        "真实源域 load0": (X_src, y_src),
        "原物理孪生(源转速)": synth_dataset(classes, n_per_class, WINDOW, 12000,
                                            CWRU_FR[0], seed=0),
        "指纹+白噪声": synth_target_like(classes, n_per_class, WINDOW, 12000,
                                         CWRU_FR[tgt_load], A_k, seed=0, physics=False),
        "指纹+目标转速冲击串": synth_target_like(classes, n_per_class, WINDOW, 12000,
                                                 CWRU_FR[tgt_load], A_k, seed=0, physics=True),
        "去梳齿包络+冲击串": synth_target_like(classes, n_per_class, WINDOW, 12000,
                                               CWRU_FR[tgt_load], A_env, seed=0, physics=True),
    }

    print(f"{'信号源':<26}{'类间平均余弦距离':>18}{'最相近的一对':>16}{'该对距离':>12}")
    print("-" * 74)
    for name, (X, y) in sources.items():
        A = class_mean_spectra(X, y, n_class)
        mean_d, min_d, (i, j) = sep(A)
        print(f"{name:<26}{mean_d:>18.4f}{f'{classes[i]}~{classes[j]}':>16}{min_d:>12.4f}")
    print("""
读法(两个独立的量,别混为一谈):
  * 类间距离  = 该信号源的各类**是否可分**(信息量);
  * 域可分性  = 该信号源**是否像真实信号**(见 tools/probe_fp.py 的判别器列)。
两者可以背离:实测"指纹+目标转速冲击串"的类间距离(0.346)高于"指纹+白噪声"(0.315),
却**更不像真实信号**(判别器 0.973 vs 0.925)、迁移更差(0.655 vs 0.875)。
说明物理冲击串并没有破坏类别信息,而是把样本分布推离了真实域 —— 它给信号强加了
**确定性的相位结构**(冲击串在时域上近乎周期),而真实信号更接近"幅度谱给定、相位随机"
的代理信号。""")


if __name__ == "__main__":
    main()
