"""Paderborn(KAt)对照实验:类间差异到底是不是"物理的"?

CWRU 每个"类别×负载"只有**一颗轴承、一次录制**,所以"类间差异是录制指纹"这个结论在
CWRU 上**结构上无法证伪** —— 你没法把同一类的不同轴承放在一起比。Paderborn 每类含
**多颗彼此独立制造、独立损伤的轴承**,于是能问三个 CWRU 问不了的问题:

  A. 跨轴承的类内一致性 —— 同类不同轴承的谱像不像?
      (像 → 类信息由损伤机制决定,是物理的;不像 → 类信息绑在个体上)
      附带 A1:同一颗轴承、不同工况(CWRU 的 0.957 就是这个量)。
  B. 轴承不相交评估 —— 在 A 轴承上训练、B 轴承上测试,还准不准?
      对照随机按窗切分(= CWRU 的常规做法),落差就是泄漏乐观量。
  C. 工况迁移 —— 工况 A 训练、工况 B 测试。
  D. 孪生迁移 —— 类别按**损伤位置**(IR/OR)定义,而这正是物理孪生唯一会建模的东西
      (BPFI/BPFO 冲击串)。于是:在孪生"本该管用"的基准上,它管用吗?

类别默认用 CLASSES_LOC =(HEALTHY, IR, OR),位置取自每颗轴承自带的事实表 PDF。

用法:
    python tools/probe_paderborn.py a          # 只跑 A(秒级)
    python tools/probe_paderborn.py abc        # A+B+C
    python tools/probe_paderborn.py d          # 孪生迁移(要训练)
    python tools/probe_paderborn.py all
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pb_utils import (CLASSES_LOC, OP_CONDITIONS, SHAFT_HZ, build_pb_domain,  # noqa: E402
                      char_freqs_pb, class_of_location, damage_table, list_bearings)
from models import build_model  # noqa: E402
from train import evaluate, make_loader, set_seed  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "paderborn")
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results_pb")

# 窗长取 4096 而非 CWRU 的 1024:采样率 64 kHz 下 1024 点只有 16 ms,
# 而 N15 工况 BPFO≈76 Hz、周期 13 ms —— 一个窗里连一次完整冲击周期都装不下,
# 实测此时**所有**轴承的平均谱都几乎相同(余弦 0.99,连健康 vs 损伤都一样)。
# 4096 点 = 64 ms,BPFO 约 5 个周期,物理才装得下。
WINDOW = 4096
STRIDE = 2048


def loc_labeler(root):
    tbl = damage_table(root)
    return (lambda b: class_of_location(b, tbl)), tbl


def spectra_by_bearing(X, bearing, window=1024):
    """{轴承码: 单位化平均幅度谱}。"""
    win = np.hanning(X.shape[-1])
    out = {}
    for b in sorted(set(bearing)):
        A = np.abs(np.fft.rfft(X[bearing == b] * win, axis=-1)).mean(axis=0)
        out[b] = A / (np.linalg.norm(A) + 1e-12)
    return out


def _pairs(S, lab, same_class, same_bearing):
    out = []
    ks = list(S)
    for i, b1 in enumerate(ks):
        for b2 in ks[i + 1:]:
            if (lab(b1) == lab(b2)) != same_class:
                continue
            if (b1 == b2) != same_bearing:
                continue
            out.append(float(np.dot(S[b1], S[b2])))
    return out


def part_a(root, max_files=4, window=WINDOW, stride=STRIDE):
    """跨轴承的类内/类间谱一致性。

    注意:这里只用**原始**信号,而原始信号被台架线谱统治(1500 rpm 下 100 Hz 占
    90% 以上能量),所以裕度必然接近 0 —— 单看这个数会被质疑"是被单线压平的"。
    带高通对照的完整版本在 `tools/probe_spectra.py`(它同时给出 CWRU 的对应量)。
    本函数保留下来,是为了把原始数值与 `damage_table` 一起落盘备查。

    这里**刻意不算**"同一轴承跨工况"的余弦:工况一变,那条统治性线谱就从
    100 Hz 移到 60 Hz,余弦会掉到 0.098 —— 那是线谱搬家的结果,不是轴承信息,
    报出来只会误导(不能去掉线谱再算,否则又变成另一个问题)。
    """
    lab, tbl = loc_labeler(root)
    cond = OP_CONDITIONS[3]
    print(f"[A] 跨轴承类内一致性   工况 {cond}  窗长 {window}")
    X, y, meta = build_pb_domain(root, cond, CLASSES_LOC, window=window, stride=stride,
                                 max_files=max_files, label_fn=lab)
    S = spectra_by_bearing(X, meta["bearing"])
    n_b = {c: sum(1 for b in S if lab(b) == c) for c in CLASSES_LOC}
    print(f"    轴承数 {n_b},窗数 {len(X)}")
    w = _pairs(S, lab, True, False)
    b = _pairs(S, lab, False, False)
    print(f"    同类 · 不同轴承 : 余弦 {np.mean(w):.3f} ± {np.std(w):.3f}  (n={len(w)} 对)")
    print(f"    异类 · 不同轴承 : 余弦 {np.mean(b):.3f} ± {np.std(b):.3f}  (n={len(b)} 对)")
    print(f"    → 判别裕度 {np.mean(w)-np.mean(b):+.3f}")
    print("    (完整对照,含高通去线谱与 CWRU 对应量:python tools/probe_spectra.py)")
    os.makedirs(RES, exist_ok=True)
    json.dump({"cond": cond, "within": w, "between": b,
               "n_bearings": n_b, "damage_table": tbl},
              open(os.path.join(RES, "a_consistency.json"), "w"), indent=1,
              ensure_ascii=False)


def train_clf(X, y, device, n_class, epochs, seed):
    # set_seed 而非只 torch.manual_seed:它同时打开 cuDNN 确定性,
    # 否则同一 seed 两次运行会有 ~0.004 的 macro-F1 漂移(见 train.set_seed 注释)
    set_seed(seed)
    bb, cl, _ = build_model(n_class, device)
    opt = torch.optim.Adam(list(bb.parameters()) + list(cl.parameters()), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        bb.train(); cl.train()
        for xb, yb in make_loader(X, y, 64, shuffle=True):
            opt.zero_grad()
            ce(cl(bb(xb.unsqueeze(1).to(device))), yb.to(device)).backward()
            opt.step()
    return bb, cl


def balanced_split(lab, codes, rng, n_train_per_class=5):
    """每类取 n_train_per_class 颗训练轴承 + 1 颗**留出**轴承(三类均衡)。

    之所以要均衡:本仓库现有 6 颗健康、6 颗内圈、9 颗外圈,直接全用会让外圈类
    样本量是别人的 1.5 倍。按轴承抽样还能顺带把"用哪颗轴承"变成随机因子。
    """
    tr, te = [], []
    for c in CLASSES_LOC:
        bs = sorted([b for b in codes if lab(b) == c])
        if len(bs) < 2:
            continue
        rng.shuffle(bs)
        te.append(bs[0])
        tr.extend(bs[1:1 + n_train_per_class])
    return tr, te


def part_bc(root, max_files=4, epochs=200, n_seed=5, stride=STRIDE, window=WINDOW):
    """B. 配对比较:同一个模型,两个测试集。

    训练轴承  : 每类 5 颗(与留出轴承不相交)
    测试集 1  : 训练轴承里**留出的 30% 窗** —— 与训练同录制,即 CWRU 的常规做法
    测试集 2  : 每类 1 颗**从未见过**的轴承 —— 轴承不相交
    落差 = 测试集1 - 测试集2,即"同录制泄漏"贡献的乐观量。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lab, _ = loc_labeler(root)
    n_class = len(CLASSES_LOC)
    codes = [b for b in list_bearings(root) if lab(b)]
    cond = OP_CONDITIONS[3]
    X, y, meta = build_pb_domain(root, cond, CLASSES_LOC, window=window, stride=stride,
                                 max_files=max_files, label_fn=lab)
    br = meta["bearing"]
    print(f"\n[B] 同录制 vs 轴承不相交   工况 {cond}  窗长 {window}  "
          f"({len(X)} 窗, {epochs} epoch, {n_seed} seed)")
    print(f"    {'seed':<6}{'同录制':>10}{'轴承不相交':>13}{'乐观量':>11}{'训练集':>10}")
    same, disj = [], []
    for seed in range(n_seed):
        rng = np.random.RandomState(seed)
        tr_c, te_c = balanced_split(lab, codes, rng)
        tr_idx = np.where(np.isin(br, tr_c))[0]
        te_idx = np.where(np.isin(br, te_c))[0]
        perm = rng.permutation(len(tr_idx))
        n_te = int(0.3 * len(tr_idx))
        te_same, tr_fit = tr_idx[perm[:n_te]], tr_idx[perm[n_te:]]
        bb, cl = train_clf(X[tr_fit], y[tr_fit], device, n_class, epochs, seed)
        m_tr = evaluate(bb, cl, X[tr_fit], y[tr_fit], device)["macro_f1"]
        m_same = evaluate(bb, cl, X[te_same], y[te_same], device)["macro_f1"]
        m_disj = evaluate(bb, cl, X[te_idx], y[te_idx], device)["macro_f1"]
        same.append(m_same); disj.append(m_disj)
        print(f"    {seed:<6}{m_same:>10.4f}{m_disj:>13.4f}{m_same-m_disj:>+11.4f}{m_tr:>10.4f}")
    print(f"    {'均值':<6}{np.mean(same):>10.4f}{np.mean(disj):>13.4f}"
          f"{np.mean(same)-np.mean(disj):>+11.4f}")
    print(f"    (标准差 同录制 {np.std(same):.4f} / 不相交 {np.std(disj):.4f})")

    print(f"\n[C] 工况迁移(训练 {cond},三类均衡的 5+5+5 颗轴承)")
    rng = np.random.RandomState(0)
    tr_c, _ = balanced_split(lab, codes, rng)
    tr_idx = np.where(np.isin(br, tr_c))[0]
    bb, cl = train_clf(X[tr_idx], y[tr_idx], device, n_class, epochs, 0)
    for cnd in OP_CONDITIONS[:3]:
        Xb, yb, mb = build_pb_domain(root, cnd, CLASSES_LOC, window=window, stride=stride,
                                     max_files=max_files, label_fn=lab)
        m = evaluate(bb, cl, Xb, yb, device)
        print(f"    -> {cnd:<14} acc={m['acc']:.4f}  macro_f1={m['macro_f1']:.4f}")
    os.makedirs(RES, exist_ok=True)
    json.dump({"same_recording": list(map(float, same)),
               "bearing_disjoint": list(map(float, disj)), "epochs": epochs},
              open(os.path.join(RES, "b_leakage.json"), "w"), indent=1)


# ---------------------------------------------------------------- D: 孪生迁移

def pb_excitation(cls, window, sr, fr, rng, slip=0.015):
    """IR/OR -> 对应特征频率的冲击串;HEALTHY -> 白噪声(无冲击)。"""
    if cls == "HEALTHY":
        return rng.randn(window)
    bpfo, bpfi = char_freqs_pb(fr)[:2]
    f = bpfi if cls == "IR" else bpfo
    period = 1.0 / f
    imp = np.zeros(window)
    t = rng.uniform(0, period)
    while t < window / sr:
        i0 = int(round(t * sr))
        if 0 <= i0 < window:
            imp[i0] = 1.0
        t += period + rng.normal(0, slip * period)
    return imp


def pb_synth(classes, n_per_class, window, sr, fr, rng, fingerprints=None, physics=True):
    """合成样本。fingerprints=None 表示平坦谱(=纯物理孪生,不标定目标域)。"""
    X, y = [], []
    flat = np.ones(window // 2 + 1)
    for cid, cls in enumerate(classes):
        a = flat if fingerprints is None else fingerprints[cid]
        for _ in range(n_per_class):
            exc = pb_excitation(cls, window, sr, fr, rng) if physics else rng.randn(window)
            w = np.fft.irfft(np.fft.rfft(exc) * a, window)
            X.append((w - w.mean()) / (w.std() + 1e-6))
            y.append(cid)
    return np.stack(X).astype("float32"), np.array(y, dtype="int64")


def estimate_fp(X_sup, y_sup, n_class, window, smooth=5):
    win = np.hanning(window)
    A = np.zeros((n_class, window // 2 + 1))
    for c in range(n_class):
        Xc = X_sup[y_sup == c]
        S = np.abs(np.fft.rfft(Xc * win, axis=-1)).mean(axis=0) if len(Xc) else np.ones(window // 2 + 1)
        A[c] = np.convolve(S, np.ones(smooth) / smooth, mode="same")
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def part_d(root, max_files=4, k=5, n_syn=200, epochs=200, n_seed=5, stride=STRIDE,
           window=WINDOW, sr=64000):
    """孪生迁移:类别 = IR/OR/HEALTHY,正是孪生会建模的运动学。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lab, _ = loc_labeler(root)
    n_class = len(CLASSES_LOC)
    codes = [b for b in list_bearings(root) if lab(b)]
    cond = OP_CONDITIONS[3]
    fr = SHAFT_HZ[cond[:3]]
    X, y, meta = build_pb_domain(root, cond, CLASSES_LOC, window=window, stride=stride,
                                 max_files=max_files, label_fn=lab)
    br = meta["bearing"]
    print(f"\n[D] 孪生迁移  工况 {cond} (fr={fr:.1f}Hz, BPFO={char_freqs_pb(fr)[0]:.1f} "
          f"BPFI={char_freqs_pb(fr)[1]:.1f} Hz), k={k}, {epochs} epoch")
    names = ["真实支撑集(仅 k 个/类)", "纯孪生(无标定)", "指纹标定(无物理)", "指纹标定+物理"]
    rows = {nm: [] for nm in names}
    for seed in range(n_seed):
        rng = np.random.RandomState(1000 + seed)
        tr_c, te_c = balanced_split(lab, codes, rng)
        tr_idx = np.where(np.isin(br, tr_c))[0]
        te_idx = np.where(np.isin(br, te_c))[0]
        rng2 = np.random.RandomState(2000 + seed)
        sup = np.concatenate([
            rng2.choice(tr_idx[y[tr_idx] == c], k, replace=np.sum(y[tr_idx] == c) < k)
            for c in range(n_class)])
        Xs, ys = X[sup], y[sup]
        fp = estimate_fp(Xs, ys, n_class, window)

        sets = {
            names[0]: (Xs, ys),
            names[1]: pb_synth(CLASSES_LOC, n_syn, window, sr, fr, rng,
                               fingerprints=None, physics=True),
            names[2]: pb_synth(CLASSES_LOC, n_syn, window, sr, fr, rng,
                               fingerprints=fp, physics=False),
            names[3]: pb_synth(CLASSES_LOC, n_syn, window, sr, fr, rng,
                               fingerprints=fp, physics=True),
        }
        for nm, (Xt, yt) in sets.items():
            bb, cl = train_clf(Xt, yt, device, n_class, epochs, seed)
            rows[nm].append(evaluate(bb, cl, X[te_idx], y[te_idx], device)["macro_f1"])
    print(f"    {'训练集':<24}{'宏 F1(轴承不相交测试)':>22}")
    print("    " + "-" * 46)
    for nm in names:
        v = np.array(rows[nm])
        print(f"    {nm:<24}{v.mean():>14.4f} ± {v.std():.4f}")
    os.makedirs(RES, exist_ok=True)
    json.dump({nm: list(map(float, rows[nm])) for nm in names},
              open(os.path.join(RES, f"d_twin_k{k}.json"), "w"), indent=1)


def main():
    todo = (sys.argv[1] if len(sys.argv) > 1 else "a").lower()
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(RES, exist_ok=True)
    if todo in ("a", "all"):
        part_a(ROOT, max_files)
    if "b" in todo or todo == "all":
        part_bc(ROOT, max_files)
    if "d" in todo or todo == "all":
        part_d(ROOT, max_files)


if __name__ == "__main__":
    main()
