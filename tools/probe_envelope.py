"""诊断:孪生信号与真实 CWRU 信号的包络谱是否对得上?

孪生合成器最基本的物理主张是"外圈故障的冲击按 BPFO 周期出现"。
这里对 真实 OR007 与 合成 OR007 分别做包络谱,看主峰是否都落在理论 BPFO 上,
以及两者的谱形差多少 —— 这是 sim-real gap 最直观的一个切面。

用法: python tools/probe_envelope.py
"""
import os
import sys

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import load_de_channel, segment, zscore_per_window  # noqa: E402
from dt_synth import CWRU_FR, char_freqs, synth_signal  # noqa: E402

SR = 12000
WINDOW = 1024


def envelope_spectrum(w, sr=SR):
    """包络谱:去均值 -> Hilbert 包络 -> 去包络均值 -> 幅值谱。"""
    w = w - w.mean()
    env = np.abs(hilbert(w))
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    freqs = np.fft.rfftfreq(len(env), 1.0 / sr)
    return freqs, spec


def peak_in_band(freqs, spec, lo=50.0, hi=250.0):
    m = (freqs >= lo) & (freqs <= hi)
    i = np.argmax(spec[m])
    return freqs[m][i], spec[m][i]


def main():
    fr = CWRU_FR[0]
    bpfo, bpfi, bsf, ftf = char_freqs(fr)
    print(f"理论特征频率(load0, fr={fr:.2f}Hz): BPFO={bpfo:.1f}  BPFI={bpfi:.1f}  BSF={bsf:.1f}  FTF={ftf:.1f} Hz\n")

    # 真实 OR007 信号:取多个窗口平均包络谱(降低单窗随机性)
    real = load_de_channel("data/cwru/OR007_0.mat")
    wins = zscore_per_window(segment(real, WINDOW, WINDOW))
    wins = wins[:200]
    acc = np.zeros(WINDOW // 2 + 1)
    for w in wins:
        f, s = envelope_spectrum(w)
        acc += s
    acc /= len(wins)
    fp, _ = peak_in_band(f, acc)
    print(f"真实 OR007_0 : 200 窗平均包络谱主峰 = {fp:6.1f} Hz   (BPFO={bpfo:.1f})")

    # 合成 OR007
    syn = np.stack([synth_signal(WINDOW, SR, fr, "OR007") for _ in range(200)])
    accs = np.zeros(WINDOW // 2 + 1)
    for w in syn:
        f2, s2 = envelope_spectrum(w)
        accs += s2
    accs /= len(syn)
    fp2, _ = peak_in_band(f2, accs)
    print(f"合成 OR007    : 200 窗平均包络谱主峰 = {fp2:6.1f} Hz   (BPFO={bpfo:.1f})")

    # 谱形相关性(归一化后)
    a = acc / (acc.sum() + 1e-12)
    b = accs / (accs.sum() + 1e-12)
    band = (f >= 20) & (f <= 1000)
    corr = np.corrcoef(a[band], b[band])[0, 1]
    print(f"\n包络谱形相关系数(20-1000Hz, 归一化后): {corr:.4f}   (1.0=完全一致)")

    # 低频段(0-500Hz)能量分布对比 —— 特征频率谐波序列是否对得上
    print("\n前 6 个谱峰位置对比(20-500Hz):")
    def top_peaks(freqs, spec, n=6, lo=20, hi=500):
        m = (freqs >= lo) & (freqs <= hi)
        ff, ss = freqs[m], spec[m]
        order = np.argsort(ss)[::-1]
        picked, used = [], np.zeros(len(ff), bool)
        for i in order:
            if used[max(0, i - 8):i + 9].any():
                continue
            used[max(0, i - 8):i + 9] = True
            picked.append(ff[i])
            if len(picked) >= n:
                break
        return sorted(picked)

    pr = top_peaks(f, acc)
    ps = top_peaks(f2, accs)
    print(f"  真实: {['%.0f' % x for x in pr]}")
    print(f"  合成: {['%.0f' % x for x in ps]}")


if __name__ == "__main__":
    main()
