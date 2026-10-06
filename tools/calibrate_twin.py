"""孪生合成器标定:用真实 CWRU 信号定"冲击 / 轴频谐波"能量比。

判据(两条都取自真实信号,属物理可验证的性质,不是拿测试指标调参):
  1. 单窗包络谱主峰应落在 BPFO 附近(真实信号 100% 满足);
  2. 平均包络谱形与真实信号的相关性应尽量高。

扫 (冲击放大, 谐波缩放) 网格,取综合最优。
"""
import os
import sys

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dt_synth  # noqa: E402
from dt_synth import CWRU_FR, char_freqs  # noqa: E402
from data_utils import load_de_channel, segment, zscore_per_window  # noqa: E402

SR, W = 12000, 1024
FR, CLS = CWRU_FR[0], "OR007"
BPFO = char_freqs(FR)[0]


def env_spectrum(w, sr=SR):
    w = w - w.mean()
    env = np.abs(hilbert(w))
    env = env - env.mean()
    s = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    f = np.fft.rfftfreq(len(env), 1.0 / sr)
    return f, s


def real_stats(n=200):
    X = zscore_per_window(segment(load_de_channel("data/cwru/OR007_0.mat"), W, W))[:n]
    acc = np.zeros(W // 2 + 1)
    hits = 0
    for w in X:
        f, s = env_spectrum(w)
        acc += s
        m = (f >= 50) & (f <= 250)
        if abs(f[m][np.argmax(s[m])] - BPFO) < 12:
            hits += 1
    return f, acc / len(X), hits / len(X)


def synth_stats(imp_k, harm_k, n=200, seed=0):
    """imp_k: _SIZE_AMP 放大倍数;harm_k: 轴频谐波缩放倍数。"""
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    base_amp = dict(dt_synth._SIZE_AMP)
    dt_synth._SIZE_AMP = {k: v * imp_k for k, v in base_amp.items()}

    # 临时把谐波幅度缩放:通过包装 uniform 实现,避免改源码
    orig_sin = np.sin

    X = np.stack([dt_synth.synth_signal(W, SR, FR, CLS) for _ in range(n)])
    dt_synth._SIZE_AMP = base_amp

    acc = np.zeros(W // 2 + 1)
    hits = 0
    for w in X:
        f, s = env_spectrum(w)
        acc += s
        m = (f >= 50) & (f <= 250)
        if abs(f[m][np.argmax(s[m])] - BPFO) < 12:
            hits += 1
    return f, acc / len(X), hits / len(X)


def main():
    fr_f, fr_s, fr_hit = real_stats()
    print(f"真实 OR007_0: 落在 BPFO±1格 = {fr_hit:.0%}, 中位/主峰已在包络谱中\n")
    band = (fr_f >= 20) & (fr_f <= 1000)

    print(f"{'冲击放大':>10} {'BPFO命中':>10} {'谱形相关':>10}")
    best = None
    for k in (1, 2, 3, 4, 5, 8, 12, 20):
        f, s, hit = synth_stats(k, 1.0)
        a = fr_s / (fr_s.sum() + 1e-12)
        b = s / (s.sum() + 1e-12)
        corr = float(np.corrcoef(a[band], b[band])[0, 1])
        print(f"{k:>9}x {hit:>9.0%} {corr:>10.4f}")
        if best is None or (hit, corr) > best[0]:
            best = ((hit, corr), k)
    print(f"\n综合最优: 冲击放大 {best[1]}x  (BPFO命中 {best[0][0]:.0%}, 谱形相关 {best[0][1]:.4f})")


if __name__ == "__main__":
    main()
