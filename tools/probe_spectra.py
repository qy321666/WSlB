"""跨基准对照:类别在**平均幅度谱**里到底留没留下痕迹?

这是全文最关键的一对数字,必须在两个基准上用**完全相同**的度量算出来:

  CWRU      : 每个"类别 × 负载"只有一颗轴承、一次录制 —— 类均值谱彼此差很多(余弦≈0.63),
              所以分类器很容易分开。但"类"与"录制"在这里是同一件事。
  Paderborn : 每类有多颗**彼此独立**的轴承。把同类多颗轴承平均起来之后,
              类均值谱彼此几乎重合(余弦≈0.99)。

结论(两句话):
  * 平均幅度谱里承载的是"哪台台架、哪次录制",不是"哪种故障";
  * CWRU 上看起来很强的类间谱差异,在把多颗独立轴承平均掉之后就不见了 ——
    它们是**个体/录制**的差异,表现为类差异,只因为 CWRU 一个类恰好只有一颗轴承。

用法: python tools/probe_spectra.py [max_files]    (默认 4 个 .mat/轴承)
输出: results_pb/spectra_bench.json
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.signal import butter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import CLASS_SETS, build_cwru_domain  # noqa: E402
from pb_utils import (CLASSES_LOC, OP_CONDITIONS, build_pb_domain,  # noqa: E402
                      class_of_location, damage_table, list_bearings)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results_pb")
CW = 1024          # CWRU 窗长
PBW = 4096         # Paderborn 窗长(见 probe_paderborn.py 的说明:64 kHz 下 1024 点太短)


def unit(A):
    return A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)


def mean_spec(X, y, n_class, window):
    win = np.hanning(window)
    return np.stack([np.abs(np.fft.rfft(X[y == c] * win, axis=-1)).mean(axis=0)
                     for c in range(n_class)])


def offdiag_cos(A):
    C = unit(A) @ unit(A).T
    iu = np.triu_indices(len(A), 1)
    return C[iu]


def cwru_part(loads=(0, 1, 2, 3)):
    classes = CLASS_SETS[10]
    nc = len(classes)
    As = {}
    for l in loads:
        X, y = build_cwru_domain(os.path.join(HERE, "data", "cwru"), l, classes, window=CW)
        As[l] = mean_spec(X, y, nc, CW)
    between = {l: offdiag_cos(A) for l, A in As.items()}
    across = {}
    for s, t in ((0, 1), (1, 2), (0, 3)):
        across[f"S{s}->S{t}"] = (unit(As[s]) * unit(As[t])).sum(axis=1)
    return {
        "n_class": nc, "n_specimen_per_class": 1, "window": CW,
        "between_class_cos": {str(l): float(v.mean()) for l, v in between.items()},
        "between_class_cos_std": {str(l): float(v.std()) for l, v in between.items()},
        "same_class_across_load_cos": {k: [float(v.mean()), float(v.std())]
                                       for k, v in across.items()},
    }


def paderborn_part(root, max_files=4, hp=0.0):
    """hp>0 时先高通,把统治全场的台架线谱去掉再比 —— 否则"类均值谱相同"可能只是
    被 100 Hz 单线压平的结果,而不是类本身没有谱差异。"""
    tbl = damage_table(root)
    lab = lambda b: class_of_location(b, tbl)
    codes = [b for b in list_bearings(root) if lab(b)]
    cond = OP_CONDITIONS[3]
    sos = butter(4, hp, "hp", fs=64000, output="sos") if hp else None
    X, y, meta = build_pb_domain(root, cond, CLASSES_LOC, window=PBW, stride=PBW // 2,
                                 max_files=max_files, label_fn=lab, sos=sos)
    br = meta["bearing"]
    S = {}
    win = np.hanning(PBW)
    for b in codes:
        A = np.abs(np.fft.rfft(X[br == b] * win, axis=-1)).mean(axis=0)
        S[b] = A / np.linalg.norm(A)
    ks = sorted(S)
    wi, bw = [], []
    for i, b1 in enumerate(ks):
        for b2 in ks[i + 1:]:
            (wi if lab(b1) == lab(b2) else bw).append(float(S[b1] @ S[b2]))
    n_spec = {c: sum(1 for b in ks if lab(b) == c) for c in CLASSES_LOC}
    return {
        "n_specimen_per_class": n_spec, "window": PBW, "condition": cond,
        "high_pass_hz": hp,
        "within_class_across_bearing_cos": [float(np.mean(wi)), float(np.std(wi)), len(wi)],
        "between_class_across_bearing_cos": [float(np.mean(bw)), float(np.std(bw)), len(bw)],
        "n_bearings": len(ks),
    }


def main():
    max_files = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    root = os.path.join(HERE, "data", "paderborn")
    out = {"cwru": cwru_part(),
           "paderborn": paderborn_part(root, max_files),
           "paderborn_hp2000": paderborn_part(root, max_files, hp=2000)}
    c = out["cwru"]
    print("=== CWRU(每类 1 颗轴承 / 1 次录制,窗长 1024)")
    for l, v in c["between_class_cos"].items():
        print(f"    load{l}: 类间平均余弦 {v:.3f} ± {c['between_class_cos_std'][l]:.3f}")
    for k, v in c["same_class_across_load_cos"].items():
        print(f"    {k} 同类跨负载: {v[0]:.3f} ± {v[1]:.3f}")

    for key, title in (("paderborn", "原始信号"),
                       ("paderborn_hp2000", "高通 2 kHz(先去掉台架线谱)")):
        p = out[key]
        w = p["within_class_across_bearing_cos"]
        b = p["between_class_across_bearing_cos"]
        print(f"\n=== Paderborn · {title}  ({p['n_bearings']} 颗独立轴承,"
              f"{p['n_specimen_per_class']},窗长 {p['window']})")
        print(f"    同类 · 不同轴承 : {w[0]:.3f} ± {w[1]:.3f}  (n={w[2]} 对)")
        print(f"    异类 · 不同轴承 : {b[0]:.3f} ± {b[1]:.3f}  (n={b[2]} 对)")
        print(f"    → 判别裕度 {w[0]-b[0]:+.3f}")

    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "spectra_bench.json"), "w"), indent=1)
    print(f"\n[save] {os.path.join(RES, 'spectra_bench.json')}")


if __name__ == "__main__":
    main()
