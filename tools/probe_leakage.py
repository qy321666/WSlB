"""方法学控制:同录制指纹泄漏到底贡献了多少?

**问题** CWRU 每个"类别 × 负载"只有**一次录制**。窗口切分(窗长 1024、步长 512)后又把
同一次录制的窗随机分成支撑集与测试集,于是两者共享同一个"录制指纹"。
结果是:任何用到目标域标签的方法都会虚高 —— 实测 k=1(10 个样本)时 support_only 已有 0.893。

**做法** 把评估拆成两层:

  同录制(标准协议)  : 在 load L 的支撑集/混合域上训练 -> 测 **load L** 自己的留出测试窗
  跨录制(无泄漏)    : 同上训练 -> 测 **其它负载** 的全部窗(不同录制、指纹余弦仅 ~0.92)

训练配置:
  support_only : 只用 k 个/类的真实支撑样本
  +fp_aug      : 支撑样本 + 目标指纹(类别平均谱)着色生成的 400/类合成样本
  +surrogate   : 支撑样本 + 相位随机化代理(保留每条支撑样本自身的幅度谱)400/类
  +true_aug    : 支撑样本 + 400/类**真实**目标样本(取自无标签池)—— 上界
  source_only  : 源域有监督(不用任何目标标签)—— 零标签参照
  dann         : 源域 + 目标无标签池的对抗域适应(不用任何目标标签)

用法: python tools/probe_leakage.py [tgt_load|all] [epochs] [n_seed] [k]
输出: results_paper/tab_leak.json(供 tools/verify_paper_numbers.py 逐条核对表 III)。
      tab:leak 的每一行 = 一次以该 load 为支撑域的运行,所以取 `all` 会跑 4 次。
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import CLASS_SETS, build_cwru_domain, split_target_fewshot  # noqa: E402
from dt_synth import CWRU_FR  # noqa: E402
from models import build_model, grad_reverse  # noqa: E402
from train import evaluate, make_loader  # noqa: E402
from twin_calib import estimate_fingerprints, synth_target_like  # noqa: E402

WINDOW, N_AUG = 1024, 400
SRC_LOAD = 0
LABELED_CFGS = ("support_only", "+fp_aug", "+surrogate", "+true_aug")
FREELESS_CFGS = ("source_only", "dann")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(HERE, "results_paper", "tab_leak.json")


def phase_surrogate(X_sup, y_sup, n_class, n_per_class, seed):
    """相位随机化代理:对每个支撑样本做傅里叶相位随机化(保留其幅度谱)。"""
    rng = np.random.RandomState(seed)
    X, y = [], []
    for c in range(n_class):
        idx = np.where(y_sup == c)[0]
        if len(idx) == 0:
            continue
        for _ in range(n_per_class):
            w = X_sup[rng.choice(idx)].astype(np.float64)
            W = np.fft.rfft(w)
            ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(W)))
            ph[0] = 1.0
            s = np.fft.irfft(np.abs(W) * ph, len(w))
            X.append((s - s.mean()) / (s.std() + 1e-6))
            y.append(c)
    return np.stack(X).astype("float32"), np.array(y, dtype="int64")


def train_supervised(X_tr, y_tr, device, n_class, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    bb, cl, _ = build_model(n_class, device)
    opt = torch.optim.Adam(list(bb.parameters()) + list(cl.parameters()), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        bb.train(); cl.train()
        for xb, yb in make_loader(X_tr, y_tr, 64, shuffle=True):
            opt.zero_grad()
            loss = ce(cl(bb(xb.unsqueeze(1).to(device))), yb.to(device))
            loss.backward(); opt.step()
    return bb, cl


def train_dann(X_src, y_src, X_pool, device, n_class, epochs, seed, adversarial=True):
    """极简 DANN:源域有监督 + 源(标注) vs 目标无标签池的对抗域适应。

    目标池的标签在这里**只是 DataLoader 的形状占位**,训练中从不使用。
    """
    torch.manual_seed(seed); np.random.seed(seed)
    bb, cl, disc = build_model(n_class, device)
    opt = torch.optim.Adam(list(bb.parameters()) + list(cl.parameters())
                           + list(disc.parameters()), lr=1e-3)
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    y_dummy = np.zeros(len(X_pool), dtype="int64")
    src_loader = make_loader(X_src, y_src, 64)
    pool_loader = make_loader(X_pool, y_dummy, 64)
    n_b = max(len(src_loader), 1)
    for ep in range(1, epochs + 1):
        bb.train(); cl.train(); disc.train()
        p = ep / epochs
        alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
        si, pi = iter(src_loader), iter(pool_loader)
        for _ in range(n_b):
            try:
                xs, ys = next(si)
            except StopIteration:
                si = iter(src_loader); xs, ys = next(si)
            try:
                xp, _ = next(pi)
            except StopIteration:
                pi = iter(pool_loader); xp, _ = next(pi)
            opt.zero_grad()
            zs = bb(xs.unsqueeze(1).to(device)); zp = bb(xp.unsqueeze(1).to(device))
            loss = ce(cl(zs), ys.to(device))
            if adversarial:
                Z = torch.cat([zs, zp])
                L = torch.cat([torch.ones(len(zs)), torch.zeros(len(zp))]).to(device)
                loss = loss + bce(disc(grad_reverse(Z, alpha)).squeeze(-1), L)
            loss.backward(); opt.step()
    return bb, cl


def run(tgt_load=3, epochs=60, n_seed=3, k=5, n_repeat=1):
    """跑一个支撑域(即 tab:leak 的一行),返回可直接核对的字典。

    `same` = 该 load 自己的留出测试窗;`cross` = 另外三个 load 的全部窗。
    `cross_to[L]` 是"跨录制、且测试集恰为 load L"那一列 —— tab:leak 里只有
    支撑域 != 3 的行才印它(支撑域就是 3 时它与 same 是同一批数据,故留空)。

    **n_repeat**:同配置整跑 n_repeat 遍。+fp_aug 那一行是出了名的高方差
    (实测 3 seed 的 std 到 0.047,且均值两次运行能差 0.05),单次 3 seed 的格子
    不足以支撑 tab:leak 这张论文的关键表。报告的数字取 n_repeat x n_seed 次运行。
    """
    classes = CLASS_SETS[10]
    n_class = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    others = [l for l in (0, 1, 2, 3) if l != tgt_load]

    doms = {l: build_cwru_domain("data/cwru", l, classes, window=WINDOW) for l in (0, 1, 2, 3)}
    X_t, y_t = doms[tgt_load]
    X_s, y_s = doms[SRC_LOAD]
    print(f"目标 load{tgt_load}(fr={CWRU_FR[tgt_load]:.2f}Hz),源 load{SRC_LOAD};"
          f"k={k},{epochs} epoch,{n_seed} seed x {n_repeat} 遍")
    print(f"同录制测试 = load{tgt_load} 留出测试窗;跨录制测试 = load{others} 全部窗\n")

    y_gen = np.repeat(np.arange(n_class), N_AUG)
    cols = ["same"] + others
    res = {c: {key: [] for key in cols} for c in LABELED_CFGS + FREELESS_CFGS}

    for rep in range(n_repeat):
        for seed in range(n_seed):
            sup_idx, pool_idx, test_idx = split_target_fewshot(X_t, y_t, k, seed)
            X_sup, y_sup = X_t[sup_idx], y_t[sup_idx]
            X_pool = X_t[pool_idx]
            # 注意:fp_aug / surrogate 的生成只用到支撑集,与测试集无交集
            A = estimate_fingerprints(X_sup, y_sup, n_class, window=WINDOW)
            X_fp = synth_target_like(classes, N_AUG, WINDOW, 12000, CWRU_FR[tgt_load],
                                     A, seed=seed, physics=False)[0]
            X_sg = phase_surrogate(X_sup, y_sup, n_class, N_AUG, seed)[0]
            rng = np.random.RandomState(seed)
            i_pool = np.concatenate([
                rng.choice(pool_idx[y_t[pool_idx] == c], min(N_AUG, (y_t[pool_idx] == c).sum()),
                           replace=False) for c in range(n_class)])

            trains = {
                "support_only": lambda: train_supervised(X_sup, y_sup, device, n_class, epochs, seed),
                "+fp_aug": lambda: train_supervised(
                    np.concatenate([X_sup, X_fp]), np.concatenate([y_sup, y_gen]),
                    device, n_class, epochs, seed),
                "+surrogate": lambda: train_supervised(
                    np.concatenate([X_sup, X_sg]), np.concatenate([y_sup, y_gen]),
                    device, n_class, epochs, seed),
                "+true_aug": lambda: train_supervised(
                    np.concatenate([X_sup, X_t[i_pool]]), np.concatenate([y_sup, y_t[i_pool]]),
                    device, n_class, epochs, seed),
                "source_only": lambda: train_dann(X_s, y_s, X_pool, device, n_class,
                                                  epochs, seed, adversarial=False),
                "dann": lambda: train_dann(X_s, y_s, X_pool, device, n_class, epochs, seed),
            }
            for name, fn in trains.items():
                bb, cl = fn()
                for l in [tgt_load] + others:
                    key = "same" if l == tgt_load else l
                    res[name][key].append(
                        evaluate(bb, cl, doms[l][0], doms[l][1], device)["macro_f1"])
        if n_repeat > 1:
            print(f"    第{rep+1}遍完成(support_only 同录制 {np.mean(res['support_only']['same'][-n_seed:]):.4f}, "
                  f"+fp_aug {np.mean(res['+fp_aug']['same'][-n_seed:]):.4f})", flush=True)

    head = f"{'训练配置':<14}" + "".join(
        f"{('测 load%d 同录制' % tgt_load) if c == 'same' else ('测 load%d 跨录制' % c):>17}"
        for c in cols) + f"{'跨录制均值':>13}{'落差':>10}"
    print(head)
    print("-" * len(head))
    out = {"config": {"tgt_load": tgt_load, "src_load": SRC_LOAD, "k": k,
                      "epochs": epochs, "n_seed": n_seed, "n_repeat": n_repeat,
                      "n_aug_per_class": N_AUG},
           "rows": {}}
    for name in LABELED_CFGS + FREELESS_CFGS:
        same = float(np.mean(res[name]["same"]))
        cross = float(np.mean([np.mean(res[name][c]) for c in others]))
        print(f"{name:<14}" + "".join(
            f"{np.mean(res[name][c]):>10.4f}±{np.std(res[name][c]):<6.4f}" for c in cols)
            + f"{cross:>13.4f}{same - cross:>+10.4f}")
        out["rows"][name] = {
            "same_mean": same, "same_std": float(np.std(res[name]["same"])),
            "cross_mean": cross, "optimism": same - cross,
            "n_runs": len(res[name]["same"]),
            "cross_to": {str(l): {"mean": float(np.mean(res[name][l])),
                                  "std": float(np.std(res[name][l]))} for l in others},
            "same_per_run": [float(v) for v in res[name]["same"]]}
    print("\n读法:落差 = 同录制成绩 - 跨录制成绩,即该配置吃到的指纹泄漏幅度。"
          "source_only / dann 不用任何目标标签,是天然无泄漏的参照。")
    return out


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else "3"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    n_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    n_repeat = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    loads = (0, 1, 2, 3) if a == "all" else (int(a),)

    res = json.load(open(OUT_JSON, encoding="utf-8")) if (
        a != "all" and os.path.exists(OUT_JSON)) else {"by_support_load": {}}
    # 产物自己写清来源:每格的 n_runs = n_seed x n_repeat,provenance 记下是哪条命令跑出来的。
    # (旧版只有 merge_runs 会写 provenance,于是"一次跑多遍"的产物反而没有出处。)
    res["provenance"] = (f"python tools/probe_leakage.py {'all' if a == 'all' else a} "
                         f"{epochs} {n_seed} {k} {n_repeat};"
                         f"每格 {n_seed} seed x {n_repeat} 遍 = {n_seed * n_repeat} runs")
    for l in loads:
        res["by_support_load"][str(l)] = run(l, epochs, n_seed, k, n_repeat)
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        json.dump(res, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[save] {OUT_JSON}  (已含支撑域 {sorted(res['by_support_load'], key=int)})")


if __name__ == "__main__":
    main()
