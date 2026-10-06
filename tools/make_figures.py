"""生成论文配图(矢量 PDF)。所有图都由数据算出,不是手工画的概念图。

    fig_spectra.pdf : 2x2。(a) CWRU load0 十类类均值谱 —— 类间明显可分(每类一颗轴承)
                      (b) 同一统计量下的物理孪生 —— 九个故障类坍缩成一根带子
                      (c) Paderborn 原始三类         —— 被台架 100 Hz 线谱压平
                      (d) Paderborn 2 kHz 高通后     —— 去掉线谱仍然重叠(对照)

    fig_leakage.pdf : (a) 同录制 vs 换录制的 macro-F1(4 支撑域 x 4 配置)
                      (b) 乐观偏差 Δ = same - cross,即"标称成绩虚高多少"

用法: python tools/make_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter, sosfilt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import CLASS_SETS, build_cwru_domain  # noqa: E402
from dt_synth import CWRU_FR, synth_dataset  # noqa: E402
from pb_utils import (OP_CONDITIONS, CLASSES_LOC, build_pb_domain,  # noqa: E402
                      damage_table, class_of_location)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "paper", "figs")
CW = 1024      # CWRU 窗长(与正文一致)
FS_CW = 12000.0
PW = 4096      # Paderborn 窗长:64 kHz 下 1024 点装不下一个 BPFO 周期
FS_PB = 64000.0

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.framealpha": 0.85,
    "legend.edgecolor": "none",
})


def class_mean_spectrum(X, y, nc, window):
    """(A) 原始类均值幅度谱 —— 与正文 Table I 完全相同的量(不平滑)。"""
    S = np.abs(np.fft.rfft(X * np.hanning(window), axis=-1))
    return np.stack([S[y == c].mean(0) for c in range(nc)])


def fig_spectra():
    classes = CLASS_SETS[10]
    X0, y0 = build_cwru_domain(os.path.join(HERE, "data", "cwru"), 0, classes,
                               window=CW)
    Sc = class_mean_spectrum(X0, y0, len(classes), CW)
    f_cw = np.fft.rfftfreq(CW, 1.0 / FS_CW)

    # 物理孪生:同一统计量、同一频率轴、同类名(载荷取源域 load0,与 Table I 一致)
    Xs, ys = synth_dataset(classes, 120, CW, FS_CW, CWRU_FR[0], seed=0)
    Ss = class_mean_spectrum(Xs, ys.astype(np.int64), len(classes), CW)

    root = os.path.join(HERE, "data", "paderborn")
    cond = OP_CONDITIONS[3]
    tbl = damage_table(root)
    lab = lambda b: class_of_location(b, tbl)  # noqa: E731

    Xr, yr, _ = build_pb_domain(root, cond, classes=CLASSES_LOC, window=PW,
                                label_fn=lab)
    Sr = class_mean_spectrum(Xr, yr, 3, PW)

    sos = butter(4, 2000.0, btype="highpass", fs=FS_PB, output="sos")
    Xh, yh, _ = build_pb_domain(root, cond, classes=CLASSES_LOC, window=PW,
                                label_fn=lab, sos=sos)
    Sh = class_mean_spectrum(Xh, yh, 3, PW)
    f_pb = np.fft.rfftfreq(PW, 1.0 / FS_PB)

    fig, ax = plt.subplots(2, 2, figsize=(7.16, 4.5))
    a, b, c, d = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

    cmap = plt.get_cmap("tab10")
    for i, cl in enumerate(classes):
        a.plot(f_cw / 1e3, 20 * np.log10(Sc[i] / Sc[i].max() + 1e-6), lw=0.7,
               color=cmap(i % 10), label=cl)
        b.plot(f_cw / 1e3, 20 * np.log10(Ss[i] / Ss[i].max() + 1e-6), lw=0.7,
               color=cmap(i % 10), label=cl)
    a.set_title("(a) CWRU, load 0: real\n10 classes, one specimen each",
                fontsize=8)
    b.set_title("(b) Same statistic: physics twin\n9 fault classes $+$ normal",
                fontsize=8)
    for p_ in (a, b):
        p_.legend(fontsize=5.0, ncol=2, loc="lower left", handlelength=1.0,
                  columnspacing=0.7, labelspacing=0.15, borderpad=0.2)
        # 标注 2800--3300 Hz:正文引用的频带能量占比就是在这个带上算的
        p_.axvspan(2.8, 3.3, color="0.85", zorder=0, lw=0)

    pb_lab = ["healthy", "inner ring", "outer ring"]
    pb_col = ["#1b7f3b", "#c1272d", "#1f4e9c"]
    for i in range(3):
        c.plot(f_pb / 1e3, 20 * np.log10(Sr[i] / Sr[i].max() + 1e-6), lw=0.9,
               color=pb_col[i], label=pb_lab[i])
        d.plot(f_pb / 1e3, 20 * np.log10(Sh[i] / Sh[i].max() + 1e-6), lw=0.9,
               color=pb_col[i], label=pb_lab[i])
    c.set_title("(c) Paderborn, as recorded\n3 classes, many specimens",
                fontsize=8)
    d.set_title("(d) Paderborn, 2 kHz high-pass\n(rig-tone control)", fontsize=8)
    for p_ in (c, d):
        p_.legend(fontsize=5.5, loc="lower left", handlelength=1.0,
                  labelspacing=0.2, borderpad=0.2)

    for p_ in (a, b, c, d):
        p_.set_ylim(-75, 6)  # 同一下限,四栏可比:原始 Paderborn 中位谱在 -64 dB
        p_.grid(alpha=0.18, lw=0.4)
        p_.set_xlabel("Frequency (kHz)")
    for p_ in (a, c):
        p_.set_ylabel("Magnitude (dB, panel max $=0$)")
    fig.tight_layout(pad=0.35)
    p = os.path.join(OUT, "fig_spectra.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# fig_leakage 的数据源:tab:leak 的落盘产物,由 tools/probe_leakage.py all 产出。
# 从前这里是一份**手抄的常量**,与 tab:leak 是同一批数字的第二处呈现;改成读文件后
# 只剩一个来源,不会再出现"表改了图没改"。缺文件时直接报错,不静默退回旧值。
LEAK_JSON = os.path.join(HERE, "results_paper", "tab_leak.json")
LEAK_CFG = {"support only": "support_only", "fingerprint aug.": "+fp_aug",
            "phase surrogate": "+surrogate", "true samples": "+true_aug"}


def load_leak():
    if not os.path.exists(LEAK_JSON):
        raise SystemExit(f"缺少 {LEAK_JSON};请先跑 python tools/probe_leakage.py all")
    d = json.load(open(LEAK_JSON, encoding="utf-8"))["by_support_load"]
    return {f"load {L}": {pub: (d[L]["rows"][inner]["same_mean"],
                                d[L]["rows"][inner]["cross_mean"])
                          for pub, inner in LEAK_CFG.items()}
            for L in ("0", "1", "2", "3")}


def fig_leakage():
    LEAK = load_leak()
    loads = list(LEAK)
    cfgs = ["support only", "fingerprint aug.", "phase surrogate", "true samples"]
    nL, nC = len(loads), len(cfgs)
    w = 0.19
    x = np.arange(nL)
    col = ["#444444", "#1f4e9c", "#c1272d", "#1b7f3b"]

    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.25))

    for j, c in enumerate(cfgs):
        d = [LEAK[l][c][1] - LEAK[l][c][0] for l in loads]
        ax[0].bar(x + (j - 1.5) * w, d, w, color=col[j], label=c)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(loads)
    ax[0].set_ylabel(r"Optimism $\Delta$ (macro-F1)")
    ax[0].set_title("(a) Same-recording minus cross-recording", fontsize=8)
    ax[0].legend(fontsize=5.2, ncol=2, loc="lower left", handlelength=1.0,
                 columnspacing=0.7, labelspacing=0.2)

    w2 = 0.36
    for j, c in enumerate(cfgs):
        s = np.mean([LEAK[l][c][0] for l in loads])
        k = np.mean([LEAK[l][c][1] for l in loads])
        ax[1].bar(j - w2 / 2, k, w2, color=col[j], alpha=0.45,
                  edgecolor=col[j], lw=0.8)
        ax[1].bar(j + w2 / 2, s, w2, color=col[j])
        ax[1].text(j - w2 / 2, k + 0.012, f"{k:.3f}", ha="center", fontsize=5.2)
        ax[1].text(j + w2 / 2, s + 0.012, f"{s:.3f}", ha="center", fontsize=5.2)
    ax[1].set_xticks(range(nC))
    ax[1].set_xticklabels([c.replace(" ", "\n") for c in cfgs], fontsize=6)
    ax[1].set_ylim(0.6, 1.09)
    ax[1].set_ylabel("macro-F1")
    ax[1].set_title("(b) Mean over support loads\n(hollow = cross, solid = same)",
                    fontsize=8)

    fig.tight_layout(pad=0.3)
    p = os.path.join(OUT, "fig_leakage.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_spectra()
    fig_leakage()
