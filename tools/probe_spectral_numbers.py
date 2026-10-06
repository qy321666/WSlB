"""复现论文正文里**每一个**谱量标量,供审稿与自查。

之所以单独立一个脚本:这些数字分散在 §Diagnosis II、§Method、Table I 里,且用了
**两种不同的谱统计量**,肉眼看不出来 —— 若不写清楚,读者会以为是同一个量。

  (A) 原始类均值幅度谱   mean_f |FFT(x)|,不平滑。用于跨基准对照表(Table I):
      CWRU 与 Paderborn 用**完全相同**的算法,所以两者可比。
  (B) 平滑谱包络 A_c(f)   即 `twin_calib.estimate_fingerprints`(宽度 5 的滑动平均 +
      单位化),也是生成合成样本时真正用的那个量。用于 §Diagnosis II 的跨负载稳定性
      与 §Method 的估计精度。**它比 (A) 平滑,所以数值系统性偏高**。

用法: python tools/probe_spectral_numbers.py
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import CLASS_SETS, build_cwru_domain, split_target_fewshot  # noqa: E402
from twin_calib import estimate_fingerprints  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW = 1024


def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_data():
    classes = CLASS_SETS[10]
    X, y = {}, {}
    for l in (0, 1, 2, 3):
        X[l], y[l] = build_cwru_domain(os.path.join(HERE, "data", "cwru"), l, classes,
                                       window=WINDOW)
    return classes, X, y


def raw_class_mean(Xl, yl, nc):
    """(A) 原始类均值幅度谱,不平滑、不单位化(仅在算余弦时归一化)。"""
    S = np.abs(np.fft.rfft(Xl * np.hanning(WINDOW), axis=-1))
    return np.stack([S[yl == c].mean(0) for c in range(nc)])


def main():
    classes, X, y = load_data()
    nc = len(classes)
    print(f"CWRU 10 类,窗长 {WINDOW},步长 512\n")

    # ---- (A) 原始类均值谱:Table I 用的量 ----
    R = {l: raw_class_mean(X[l], y[l], nc) for l in (0, 1, 2, 3)}
    print("[A] 原始类均值幅度谱(Table I 的度量)")
    for l in (0, 1, 2, 3):
        An = R[l] / np.linalg.norm(R[l], axis=1, keepdims=True)
        C = An @ An.T
        iu = np.triu_indices(nc, 1)
        print(f"    load{l} 类间平均余弦 {C[iu].mean():.3f}  最小 {C[iu].min():.3f}")
    for s, t in ((0, 1), (1, 2), (0, 3)):
        c = [cos(R[s][i], R[t][i]) for i in range(nc)]
        print(f"    同类 load{s}->load{t}: {np.mean(c):.3f} ± {np.std(c):.3f}")

    # ---- (B) 平滑谱包络:§Diagnosis II 与 §Method 用的量 ----
    A = {l: estimate_fingerprints(X[l], y[l], nc, window=WINDOW, smooth=5)
         for l in (0, 1, 2, 3)}
    print("\n[B] 平滑谱包络 A_c(f)(§Diagnosis II / §Method 的度量)")
    for s in (0, 1, 2, 3):
        c = [np.mean([cos(A[s][i], A[t][i]) for i in range(nc)])
             for t in (0, 1, 2, 3) if t != s]
        print(f"    load{s} 对另三个负载: {[round(x,3) for x in c]}  均值 {np.mean(c):.3f}")
    allp = [np.mean([cos(A[s][i], A[t][i]) for i in range(nc)])
            for s, t in itertools.combinations((0, 1, 2, 3), 2)]
    print(f"    全部 6 对均值 {np.mean(allp):.3f}")

    print("\n    §Method:用 k=5 支撑估的包络 vs 用全量估的包络")
    for l in (0, 3):
        sup, _, _ = split_target_fewshot(X[l], y[l], 5, 0)
        A5 = estimate_fingerprints(X[l][sup], y[l][sup], nc, window=WINDOW, smooth=5)
        c = [cos(A5[i], A[l][i]) for i in range(nc)]
        print(f"      load{l}: 均值 {np.mean(c):.3f}  最差 {np.min(c):.3f}")

    print('\n    §Method:源域包络 → 目标域包络(源域指纹预测不了目标域)')
    for s, t in ((0, 1), (0, 3), (1, 2), (1, 3)):
        c = [cos(A[s][i], A[t][i]) for i in range(nc)]
        print(f"      load{s} -> load{t}: 均值 {np.mean(c):.3f}  最差 {np.min(c):.3f}")

    c = band_energy(classes, X, y, nc)
    print(f"\n[C] 类均值谱在 2800--3300 Hz 的能量占比(%)")
    for tag in ("真实 load0,九个故障类", "物理孪生, 九个故障类"):
        print(f"    {tag} 最小 {c[tag]['min_cls']} {c[tag]['min']:.0f}%  "
              f"最大 {c[tag]['max_cls']} {c[tag]['max']:.0f}%  均值 {c[tag]['mean']:.0f}%")
    ni = classes.index("NORMAL")
    print(f"    (NORMAL 两类恒为 {c['real_normal']:.0f}% / {c['twin_normal']:.0f}%"
          f" —— 按构造是宽带噪声,不计入上面的范围)")

    d = pairwise_db(classes, X[0], y[0], nc)
    print("\n[D] 类均值谱的两两 dB 间距(九个故障类)")
    for tag, v in d.items():
        print(f"    {tag:<10} 两两间距 最小 {v['pair_min']:.1f} 中位 {v['pair_med']:.1f} dB"
              f" | 逐频点跨类展宽 中位 {v['span_med']:.1f} dB")


def band_energy(classes, X, y, nc, band=(2800.0, 3300.0), sr=12000.0):
    """[C] §Diagnosis II 引用的"2800--3300 Hz 频带能量占比"。

    逐类取**类均值功率谱**在该频带的能量占比,返回机器可校对的字典
    (供 tools/verify_paper_numbers.py 逐条比对正文)。NORMAL 按构造是宽带噪声,
    占比恒为 0%,所以正文说的"九类"是**排除 NORMAL 的九个故障类**。
    """
    from dt_synth import CWRU_FR, synth_dataset  # noqa: E402
    f = np.fft.rfftfreq(WINDOW, 1.0 / sr)
    m = (f >= band[0]) & (f <= band[1])

    def frac(Xl, yl):
        S = np.abs(np.fft.rfft(Xl * np.hanning(WINDOW), axis=-1)) ** 2
        e = np.stack([S[yl == c].mean(0) for c in range(nc)])
        return 100.0 * e[:, m].sum(1) / e.sum(1)

    r = frac(X[0], y[0])
    Xt, yt = synth_dataset(classes, 120, WINDOW, sr, CWRU_FR[0], seed=0)
    t = frac(Xt, yt)
    fault = [i for i in range(nc) if classes[i] != "NORMAL"]
    ni = classes.index("NORMAL")

    out = {"real_normal": float(r[ni]), "twin_normal": float(t[ni]), "band_hz": band}
    for tag, v in (("真实 load0,九个故障类", r), ("物理孪生, 九个故障类", t)):
        s = sorted(((classes[i], float(v[i])) for i in fault), key=lambda z: z[1])
        out[tag] = {"min": s[0][1], "min_cls": s[0][0],
                    "max": s[-1][1], "max_cls": s[-1][0],
                    "mean": float(np.mean([z[1] for z in s]))}
    return out


def pairwise_db(classes, Xr, yr, nc, fault=None, Xt=None, yt=None):
    """[D] fig_spectra(b) 题注里引的两两 dB 间距(只取九个故障类),返回字典。

    (A)/(B) 用的是余弦,这里是 **dB 差** —— 配图上人眼读的是 dB 轴,用量纲一致的
    数更站得住。逐类先按自身最大值归一到 0 dB,再取两两在各频点的平均绝对差。
    """
    import itertools  # noqa: E402

    if fault is None:
        fault = [i for i in range(nc) if classes[i] != "NORMAL"]
    if Xt is None:
        from dt_synth import CWRU_FR, synth_dataset  # noqa: E402
        Xt, yt = synth_dataset(classes, 120, WINDOW, 12000.0, CWRU_FR[0], seed=0)
        yt = yt.astype(np.int64)

    def db(Xl, yl):
        S = np.abs(np.fft.rfft(Xl * np.hanning(WINDOW), axis=-1))
        C = np.stack([S[yl == c].mean(0) for c in range(nc)])
        return 20 * np.log10(C / C.max(1, keepdims=True) + 1e-9)

    out = {}
    for tag, C in (("真实 load0", db(Xr, yr)), ("物理孪生", db(Xt, yt))):
        v = np.array([np.abs(C[i] - C[j]).mean()
                      for i, j in itertools.combinations(fault, 2)])
        w = C[fault].max(0) - C[fault].min(0)
        out[tag] = {"pair_min": float(v.min()), "pair_med": float(np.median(v)),
                    "span_med": float(np.median(w))}
    return out


if __name__ == "__main__":
    main()
