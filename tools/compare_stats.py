"""对照真实与合成信号的统计特征,定位孪生到底差在哪。

对同一类别(默认 OR007 / load0)各取 N 个 z-score 后的窗,逐项对比:
  1. 功率谱密度(分频段能量占比)—— 低频谐波 / 共振带 / 高频噪声底
  2. 谱峭度(Spectral Kurtosis)—— 冲击性成分落在哪个频带
  3. 时域统计量 —— 峭度、波峰因数(越像冲击越大)
  4. 自相关 —— 周期性结构
  5. 包络谱 —— 特征频率是否对上

用法: python tools/compare_stats.py [类别] [负载] [窗数]
      python tools/compare_stats.py NORMAL 0 200
"""
import os
import sys

import numpy as np
from scipy.signal import hilbert, welch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import load_de_channel, segment, zscore_per_window, parse_class  # noqa: E402
from dt_synth import CWRU_FR, char_freqs, synth_signal  # noqa: E402

SR, W = 12000, 1024


def load_real(cls, load, n):
    typ, size = parse_class(cls)
    if typ == "NORMAL":
        path = f"data/cwru/Normal_{load}.mat"
    else:
        path = f"data/cwru/{typ}{size}_{load}.mat"
    sig = load_de_channel(path)
    wins = zscore_per_window(segment(sig, W, W))
    idx = np.linspace(0, len(wins) - 1, min(n, len(wins))).astype(int)
    return wins[idx]


def load_synth(cls, load, n, seed=0):
    np.random.seed(seed)
    fr = CWRU_FR[load]
    return np.stack([synth_signal(W, SR, fr, cls) for _ in range(n)])


def psd_bands(X, bands=((20, 200), (200, 800), (800, 2000), (2000, 3000),
                        (3000, 4500), (4500, 6000))):
    """各频段能量占比(对 0-6000Hz 总能量归一)。"""
    f, P = welch(X, fs=SR, nperseg=W, axis=-1)
    tot = P.sum(axis=-1)
    out = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        out.append((P[:, m].sum(axis=-1) / tot).mean())
    return out


def spectral_kurtosis(X, nseg=8):
    """简化的谱峭度:分频带做包络峭度,找冲击性最强的频带。"""
    f, P = welch(X, fs=SR, nperseg=W, axis=-1)
    # 用时域带通后的峭度近似:按倍频程划分,看哪一段峭度最高
    out = {}
    for lo, hi in ((500, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5500)):
        idx = (f >= lo) & (f < hi)
        band_e = P[:, idx].sum(axis=-1)
        band_e = band_e / (band_e.mean() + 1e-12)
        out[f"{lo}-{hi}"] = float(np.mean(band_e ** 2) - 1)  # 能量峭度
    return out


def time_stats(X):
    kurt = np.mean((X - X.mean(axis=-1, keepdims=True)) ** 4, axis=-1) / \
           (X.std(axis=-1) ** 4 + 1e-12)
    crest = np.abs(X).max(axis=-1) / (X.std(axis=-1) + 1e-12)
    return {"峭度": float(kurt.mean()), "波峰因数": float(crest.mean())}


def envelope_peaks(X, cls, load):
    typ, _ = parse_class(cls)
    f = np.fft.rfftfreq(W, 1.0 / SR)
    acc = np.zeros(len(f))
    for w in X:
        e = np.abs(hilbert(w - w.mean()))
        e = e - e.mean()
        acc += np.abs(np.fft.rfft(e * np.hanning(W)))
    acc /= len(X)
    m = (f >= 50) & (f <= 400)
    order = np.argsort(acc[m])[::-1]
    ff = f[m]
    picks, used = [], np.zeros(len(ff), bool)
    for i in order:
        if used[max(0, i - 8):i + 9].any():
            continue
        used[max(0, i - 8):i + 9] = True
        picks.append(ff[i])
        if len(picks) >= 4:
            break
    return sorted(picks)


def autocorr_peak(X):
    """自相关第一非零峰的位置(采样点)—— 反映周期性。"""
    out = []
    for w in X[:100]:
        w = w - w.mean()
        ac = np.correlate(w, w, "full")[len(w) - 1:]
        ac /= (ac[0] + 1e-12)
        seg = ac[20:400]
        out.append(int(np.argmax(seg)) + 20)
    return float(np.median(out))


def main():
    cls = sys.argv[1] if len(sys.argv) > 1 else "OR007"
    load = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    R = load_real(cls, load, n)
    S = load_synth(cls, load, n)
    print(f"类别 {cls} / load{load} / 各 {len(R)} 窗\n")

    print(f"{'指标':<26}{'真实':>12}{'合成':>12}")
    print("-" * 52)

    bands = ((20, 200), (200, 800), (800, 2000), (2000, 3000), (3000, 4500), (4500, 6000))
    pr, ps = psd_bands(R), psd_bands(S)
    print("[功率谱能量占比]")
    for (lo, hi), a, b in zip(bands, pr, ps):
        print(f"  {lo:>5}-{hi:<5} Hz{'':<8}{a:>11.3f}{b:>12.3f}")

    print("\n[时域统计]")
    tr, ts = time_stats(R), time_stats(S)
    for k in tr:
        print(f"  {k:<24}{tr[k]:>11.2f}{ts[k]:>12.2f}")

    print("\n[自相关首峰位置(采样点,越大=周期越长)]")
    print(f"  {'lag':<24}{autocorr_peak(R):>11.0f}{autocorr_peak(S):>12.0f}")

    print("\n[包络谱前几个峰 (Hz)]")
    print(f"  真实: {['%.0f' % x for x in envelope_peaks(R, cls, load)]}")
    print(f"  合成: {['%.0f' % x for x in envelope_peaks(S, cls, load)]}")
    bpfo, bpfi, bsf, ftf = char_freqs(CWRU_FR[load])
    print(f"  理论: BPFO={bpfo:.0f} BPFI={bpfi:.0f} BSF={bsf:.0f} FTF={ftf:.0f}")

    print("\n[谱峭度(各频带能量峭度,越大越像冲击)]")
    kr, ks = spectral_kurtosis(R), spectral_kurtosis(S)
    for k in kr:
        print(f"  {k:<24}{kr[k]:>11.2f}{ks[k]:>12.2f}")

    # 共振带定位:哪一段能量峭度最高
    def top_band(d):
        return max(d, key=d.get)
    print(f"\n  冲击性最强频带: 真实={top_band(kr)}   合成={top_band(ks)}")


if __name__ == "__main__":
    main()
