"""诊断:孪生信号包络峰为何不在 BPFO?冲击强度是不是那个旋钮?

真实 OR007 的单窗包络峰 100% 落在 BPFO 附近;合成信号只有 10%。
这里扫描"冲击幅度放大倍数"与"轴频谐波强度",看包络峰能否回到 BPFO。
"""
import sys
import os

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dt_synth  # noqa: E402
from dt_synth import CWRU_FR, char_freqs  # noqa: E402
from data_utils import load_de_channel, segment, zscore_per_window  # noqa: E402

SR, W = 12000, 1024
FR = CWRU_FR[0]
BPFO = char_freqs(FR)[0]


def env_peak(w, sr=SR):
    w = w - w.mean()
    env = np.abs(hilbert(w))
    env = env - env.mean()
    s = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    f = np.fft.rfftfreq(len(env), 1.0 / sr)
    m = (f >= 50) & (f <= 250)
    return f[m][np.argmax(s[m])]


def measure(cls="OR007", n=60):
    peaks = [env_peak(dt_synth.synth_signal(W, SR, FR, cls)) for _ in range(n)]
    peaks = np.array(peaks)
    return float(np.mean(np.abs(peaks - BPFO) < 12)), float(np.median(peaks))


def main():
    print(f"BPFO = {BPFO:.1f} Hz,频率分辨率 = {SR / W:.1f} Hz/格\n")

    peaks = np.array([env_peak(w) for w in
                      zscore_per_window(segment(load_de_channel("data/cwru/OR007_0.mat"), W, W))[:60]])
    print(f"{'真实 OR007_0':<18} 落在BPFO±1格={np.mean(np.abs(peaks - BPFO) < 12):>6.0%}  中位峰={np.median(peaks):>6.1f} Hz\n")

    base_amp = dict(dt_synth._SIZE_AMP)
    print(f"{'冲击放大':>10} {'落在BPFO±1格':>14} {'中位峰':>12}")
    for k in (1, 3, 5, 10, 20, 40):
        dt_synth._SIZE_AMP = {kk: vv * k for kk, vv in base_amp.items()}
        r, med = measure()
        print(f"{k:>9}x {r:>13.0%} {med:>10.1f} Hz")
    dt_synth._SIZE_AMP = base_amp


if __name__ == "__main__":
    main()
