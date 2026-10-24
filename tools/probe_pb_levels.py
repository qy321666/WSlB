"""Paderborn 三层泄漏分解:把"同录制乐观量"拆成两层。

论文原来的对照只有两端:

    同录制(同文件留出 30% 窗)      0.885
    轴承不相交(换一整颗轴承)        0.611     落差 +0.274

问题是这两端之间**同时变了两件事**:换了录制会话,也换了试件(个体轴承)。
CWRU 上分不开(每类只有一颗轴承),Limitations (ii) 已如实承认。但 Paderborn
每个轴承在同一工况下有 **20 次独立录制**,于是可以插入中间一档:

    L1  同录制     : 文件 1..4,留出末 30% 窗        —— 与训练同一次测量
    L2  跨录制     : 文件 5..8,同一批轴承、同一工况  —— 只换了录制会话
    L3  跨试件     : 留出轴承的全部文件              —— 录制和试件一起换

三次评测用的是**同一个训练集**(训练轴承的文件 1..4 前 70% 窗),所以
L1→L2 的落差只来自"换录制",L2→L3 只来自"换试件"。这正是论文核心机制
想要的那个分解:会话泄漏 vs 个体指纹。

设计上刻意复现 tab:pb_leak 的两个端点(L1/L3 的训练集与测试集与那一致),
所以这两个数应当就是表 V 的两个端点 —— 顺便当一次自洽性检查。
现正文表 V 的端点是 L1 = 0.938、L3 = 0.597(见 results_pb/pb_levels.json 的 mean,
5 seed)。**注:本注释原先记的 0.885 / 0.611 是更早一轮的值**,已被现设置取代,勿再引用 ——
`tools/verify_paper_numbers.py` 是按现处的 mean 核对的。

台架线稳健性检验(2026-10-24 补):Paderborn 的 100 Hz 谱线是 1500 rpm 下的
4 倍轴频,是试验台的电机线,与故障无关。正文表 IV 的高通对照只改了**谱统计**,
没有重训分类器;这里在同一套 L1/L2/L3 分解上**重训**,看乐观量是否由这条线携带。

    --filter=none      原设置(写 pb_levels.json,勿覆盖的表 V 产物)
    --filter=notch     去掉 100 Hz 的 1..20 次谐波,每次 ±2 个频点
    --filter=highpass  去掉 2 kHz 以下(与表 IV 的对照同一截止)

用法: python tools/probe_pb_levels.py [epochs] [seeds] [--filter=notch]
                                      [--f0=100] [--bins=2] [--harm=20]
                                      [--cutoff=2000]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pb_utils import (CLASSES_LOC, OP_CONDITIONS, SR, build_pb_domain,  # noqa: E402
                      class_of_location, damage_table, list_bearings)
from train import evaluate  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(HERE, "data", "paderborn")
RES = os.path.join(HERE, "results_pb")

WINDOW = 4096     # 64 kHz 下 1024 点装不下一个 BPFO 周期,见 probe_paderborn.py
STRIDE = 2048
N_FIT = 4         # 训练用文件:序号 1..4  —— 与 tab:pb_leak 一致
N_HELD = 4        # 跨录制测试文件:序号 5..8
TRAIN_BEARINGS = 5


def spectral_keep(n, fs, kind, f0=100.0, n_bins=2, n_harm=20, cutoff=2000.0):
    """返回长度 n//2+1 的布尔掩码,True 表示该频点保留。

    notch   : 去掉 f0 的 1..n_harm 次谐波,每次 ±n_bins 个频点
              (100 Hz 的谱线落在 93.75 Hz 的栅栏里,两侧各有泄漏,故要按频点宽度去)
    highpass: 去掉 cutoff 以下的全部频点 —— 与表 IV 的 2 kHz 高通对照完全一致
    """
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    keep = np.ones(len(freqs), dtype=bool)
    if kind == "highpass":
        keep &= freqs >= cutoff
    elif kind == "notch":
        df = fs / n
        for k in range(1, n_harm + 1):
            fk = k * f0
            if fk > fs / 2:
                break
            keep &= np.abs(freqs - fk) > n_bins * df
    return keep


def apply_filter(X, kind, fs, **kw):
    """逐窗做频域掩码,再重新 z-score。返回 (滤波后的窗, 被去掉的谱能量占比)。"""
    if kind == "none":
        return X, 0.0
    keep = spectral_keep(X.shape[1], fs, kind, **kw)
    Z = np.fft.rfft(X.astype(np.float64), axis=1)
    e_all = float(np.sum(np.abs(Z) ** 2))
    Z = Z * keep[None, :]
    Y = np.fft.irfft(Z, n=X.shape[1], axis=1)
    Y = (Y - Y.mean(axis=1, keepdims=True)) / (Y.std(axis=1, keepdims=True) + 1e-12)
    removed = 1.0 - float(np.sum(np.abs(Z) ** 2)) / max(e_all, 1e-12)
    return Y.astype(np.float32), removed


def load_all(root, cond, max_files=N_FIT + N_HELD):
    tbl = damage_table(root)
    lab = lambda b: class_of_location(b, tbl)  # noqa: E731
    X, y, meta = build_pb_domain(root, cond, CLASSES_LOC, window=WINDOW,
                                 stride=STRIDE, max_files=max_files, label_fn=lab)
    # meta['file'] 形如 N15_M07_F10_K005_7.mat,取出序号用于分层
    num = np.array([int(os.path.basename(f).rsplit("_", 1)[1].split(".")[0])
                    for f in meta["file"]])
    return X, y, meta["bearing"], num, lab


def balanced_split(lab, codes, rng, n_per_class=TRAIN_BEARINGS):
    """每类 n 颗训练轴承 + 1 颗留出轴承(与 probe_paderborn.balanced_split 同构)。"""
    tr, te = [], []
    for c in CLASSES_LOC:
        bs = sorted([b for b in codes if lab(b) == c])
        if len(bs) < 2:
            continue
        rng.shuffle(bs)
        te.append(bs[0])
        tr.extend(bs[1:1 + n_per_class])
    return tr, te


def parse_args(argv):
    """位置参数 [epochs] [seeds] + 可选 --filter/--f0/--bins/--harm/--cutoff。"""
    pos = [a for a in argv[1:] if not a.startswith("--")]
    opt = {"filter": "none", "f0": 100.0, "bins": 2, "harm": 20, "cutoff": 2000.0}
    for a in argv[1:]:
        if "=" in a and a.startswith("--"):
            k, v = a[2:].split("=", 1)
            opt[k] = v if k == "filter" else float(v)
    epochs = int(pos[0]) if len(pos) > 0 else 200
    n_seed = int(pos[1]) if len(pos) > 1 else 5
    return epochs, n_seed, opt


def out_name(kind):
    """滤波条件各自写一个文件;none 仍写 pb_levels.json(表 V 的现成产物)。"""
    return {"none": "pb_levels.json", "notch": "pb_levels_notch.json",
            "highpass": "pb_levels_highpass.json"}[kind]


def main():
    epochs, n_seed, opt = parse_args(sys.argv)
    from probe_paderborn import train_clf  # 复用同一训练循环

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond = OP_CONDITIONS[3]
    X, y, br, num, lab = load_all(ROOT, cond)
    X, removed = apply_filter(X, opt["filter"], SR, f0=opt["f0"],
                              n_bins=int(opt["bins"]), n_harm=int(opt["harm"]),
                              cutoff=opt["cutoff"])
    codes = [b for b in list_bearings(ROOT) if lab(b)]
    print(f"[三层泄漏分解] {cond}  {len(X)} 窗(窗长 {WINDOW},步长 {STRIDE})  "
          f"{epochs} epoch  {n_seed} seed  device={device.type}")
    print(f"  滤波        : {opt['filter']}  (去掉谱能量 {removed * 100:.1f}%;"
          f" f0={opt['f0']} Hz, ±{int(opt['bins'])} 频点 ×{int(opt['harm'])} 次谐波,"
          f" highpass 截止 {opt['cutoff']} Hz)")
    print(f"  L1 同录制   : 文件 1..{N_FIT} 的末 30% 窗")
    print(f"  L2 跨录制   : 文件 {N_FIT + 1}..{N_FIT + N_HELD},同一批训练轴承、同一工况")
    print(f"  L3 跨试件   : 留出轴承的全部文件")
    print(f"  三次评测共用同一个训练集(训练轴承的文件 1..{N_FIT} 前 70% 窗)\n")

    hdr = f"  {'seed':<6}{'L1 同录制':>12}{'L2 跨录制':>12}{'L3 跨试件':>12}" \
          f"{'L1-L2':>9}{'L2-L3':>9}{'训练集':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rec = {"L1": [], "L2": [], "L3": [], "fit": []}
    for seed in range(n_seed):
        rng = np.random.RandomState(seed)
        tr_c, te_c = balanced_split(lab, codes, rng)
        tr_b = np.isin(br, tr_c)
        # 训练集:训练轴承 x 文件 1..4,前 70% 窗(按原始的窗序,与 tab:pb_leak 一致)
        fit_pool = np.where(tr_b & (num <= N_FIT))[0]
        perm = rng.permutation(len(fit_pool))
        n_te = int(0.3 * len(fit_pool))
        l1 = fit_pool[perm[:n_te]]            # 同录制留出窗
        fit = fit_pool[perm[n_te:]]           # 训练
        l2 = np.where(tr_b & (num > N_FIT))[0]                 # 跨录制、同一批轴承
        l3 = np.where(np.isin(br, te_c) & (num <= N_FIT))[0]   # 跨试件

        bb, cl = train_clf(X[fit], y[fit], device, len(CLASSES_LOC), epochs, seed)
        m = {k: evaluate(bb, cl, X[i], y[i], device)["macro_f1"]
             for k, i in (("L1", l1), ("L2", l2), ("L3", l3))}
        m["fit"] = evaluate(bb, cl, X[fit], y[fit], device)["macro_f1"]
        for k in rec:
            rec[k].append(float(m[k]))
        print(f"  {seed:<6}{m['L1']:>12.4f}{m['L2']:>12.4f}{m['L3']:>12.4f}"
              f"{m['L1'] - m['L2']:>+9.4f}{m['L2'] - m['L3']:>+9.4f}{m['fit']:>10.4f}")

    mu = {k: float(np.mean(v)) for k, v in rec.items()}
    sd = {k: float(np.std(v)) for k, v in rec.items()}
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'均值':<6}{mu['L1']:>12.4f}{mu['L2']:>12.4f}{mu['L3']:>12.4f}"
          f"{mu['L1'] - mu['L2']:>+9.4f}{mu['L2'] - mu['L3']:>+9.4f}{mu['fit']:>10.4f}")
    print(f"  {'标准差':<6}{sd['L1']:>12.4f}{sd['L2']:>12.4f}{sd['L3']:>12.4f}")
    print(f"\n  总乐观量 L1-L3 = {mu['L1'] - mu['L3']:+.4f}  "
          f"(会话泄漏 {mu['L1'] - mu['L2']:+.4f} + 试件差异 {mu['L2'] - mu['L3']:+.4f})")

    os.makedirs(RES, exist_ok=True)
    fn = os.path.join(RES, out_name(opt["filter"]))
    json.dump({"cond": cond, "epochs": epochs, "seeds": n_seed,
               "per_seed": rec, "mean": mu, "std": sd,
               "n_fit_files": N_FIT, "n_held_files": N_HELD,
               "filter": opt["filter"], "filter_spec": dict(opt),
               "spectral_energy_removed": removed},
              open(fn, "w"), indent=1)
    print(f"  -> {fn}")


if __name__ == "__main__":
    main()
