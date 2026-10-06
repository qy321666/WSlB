"""诊断:少样本目标域场景下,各种增广源到底值多少?

设定:**没有源域**(或源域不可用/不信任),只有目标设备上 k 个/类的带标签样本。
这是"数字孪生补样本"这一命题真正要解决的场景 —— 有标签数据极度稀缺。

对照:
  support_only   : 只用 k 个/类 的真实支撑样本训练(下界)
  + roll_aug     : 支撑样本做循环平移+高斯噪声的无信息增广(400/类)——控制"只是多了一批数据"
  + dt_aug       : 加上原物理孪生合成样本(400/类,源转速)
  + fp_rand_aug  : 加上"目标指纹着色白噪声"(400/类)—— 本文候选方法
  + fp_phys_aug  : 加上"目标指纹着色 + 目标转速冲击串"(400/类)—— 物理+指纹
  + src_rand_aug : 加上"源域指纹着色白噪声"(400/类)—— 不用任何目标标签
  + surrogate    : 相位随机化(保留每条支撑样本自身幅度谱,400/类)
  + true_aug     : 从**目标域训练集之外**的真实样本中取样(400/类)—— 上界(理想增广源)

用法: python tools/probe_fewshot.py [tgt_load] [src_load] [epochs] [n_seed] [n_repeat]
输出: results_paper/tab_fewshot.json(供 tools/verify_paper_numbers.py 逐条核对表 IX)

**n_repeat** 同 probe_fp/probe_leakage:同配置整跑 n_repeat 遍,报告 n_repeat x n_seed
次运行。表 IX 里有几行(尤其 fp_phys_aug)方差极大,单次 3 seed 的格子不可靠。
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import CLASS_SETS, build_cwru_domain, split_target_fewshot  # noqa: E402
from dt_synth import CWRU_FR, synth_dataset  # noqa: E402
from models import build_model  # noqa: E402
from train import evaluate, make_loader, rand_augment_dataset  # noqa: E402
from twin_calib import estimate_fingerprints, synth_target_like  # noqa: E402

WINDOW, N_AUG = 1024, 400
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(HERE, "results_paper", "tab_fewshot.json")
CONFIGS = ("support_only", "roll_aug", "dt_aug", "fp_rand_aug",
           "fp_phys_aug", "src_rand_aug", "surrogate", "true_aug")


def phase_surrogate(X_sup, y_sup, n_class, n_per_class, seed):
    """相位随机化代理样本:对每个支撑样本做傅里叶相位随机化(保留其幅度谱)。

    这是"不用类别均值"的天然竞争基线 —— 每类生成 n_per_class 条样本,但用的是
    单条支撑样本自己的幅度谱,而不是类别平均谱。用来回答"fp_aug 的增益是否只来自
    类别均值这个统计量"。
    """
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


def run(X_tr, y_tr, X_test, y_test, device, n_class, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    bb, cl, _ = build_model(n_class, device)
    opt = torch.optim.Adam(list(bb.parameters()) + list(cl.parameters()), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    loader = make_loader(X_tr, y_tr, 64, shuffle=True)
    for _ in range(epochs):
        bb.train(); cl.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = ce(cl(bb(xb.unsqueeze(1).to(device))), yb.to(device))
            loss.backward(); opt.step()
    return evaluate(bb, cl, X_test, y_test, device)["macro_f1"]


def result(tgt_load=3, src_load=0, epochs=40, n_seed=3, ks=(1, 3, 5), n_repeat=1):
    """跑完所有 k 与增广源,返回可直接核对表 IX 的字典(含配置与逐次运行原始值)。"""
    classes = CLASS_SETS[10]
    n_class = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_src, y_src = build_cwru_domain("data/cwru", src_load, classes, window=WINDOW)
    X_tgt, y_tgt = build_cwru_domain("data/cwru", tgt_load, classes, window=WINDOW)
    print(f"目标 load{tgt_load}(fr={CWRU_FR[tgt_load]:.2f}Hz),增广 {N_AUG}/类,"
          f"{epochs} epoch,{n_seed} seed x {n_repeat} 遍。**不使用源域训练**\n")

    y_gen = np.repeat(np.arange(n_class), N_AUG)
    header = f"{'k':>3} " + "".join(f"{n:>17}" for n in CONFIGS)
    print(header)
    print("-" * len(header))

    out = {"config": {"tgt_load": tgt_load, "src_load": src_load, "epochs": epochs,
                      "n_seed": n_seed, "n_repeat": n_repeat,
                      "n_aug_per_class": N_AUG, "window": WINDOW,
                      "classes": list(classes)},
           "by_k": {}}
    for k in ks:
        acc = {n: [] for n in CONFIGS}
        for rep in range(n_repeat):
            for seed in range(n_seed):
                sup_idx, pool_idx, test_idx = split_target_fewshot(X_tgt, y_tgt, k, seed)
                X_sup, y_sup = X_tgt[sup_idx], y_tgt[sup_idx]
                X_test, y_test = X_tgt[test_idx], y_tgt[test_idx]
                X_extra, y_extra = X_tgt[pool_idx], y_tgt[pool_idx]   # 真实但未用作支撑的样本

                acc["support_only"].append(run(X_sup, y_sup, X_test, y_test, device, n_class, epochs, seed))

                A = estimate_fingerprints(X_sup, y_sup, n_class, window=WINDOW)
                A_src = estimate_fingerprints(X_src, y_src, n_class, window=WINDOW)
                gens = {
                    "roll_aug": rand_augment_dataset(X_sup, y_sup, n_class, N_AUG, seed=seed)[0],
                    "dt_aug": synth_dataset(classes, N_AUG, WINDOW, 12000,
                                            CWRU_FR[src_load], seed=seed)[0],
                    "fp_rand_aug": synth_target_like(classes, N_AUG, WINDOW, 12000,
                                                     CWRU_FR[tgt_load], A, seed=seed, physics=False)[0],
                    "fp_phys_aug": synth_target_like(classes, N_AUG, WINDOW, 12000,
                                                     CWRU_FR[tgt_load], A, seed=seed, physics=True)[0],
                    "src_rand_aug": synth_target_like(classes, N_AUG, WINDOW, 12000,
                                                      CWRU_FR[tgt_load], A_src, seed=seed,
                                                      physics=False)[0],
                    "surrogate": phase_surrogate(X_sup, y_sup, n_class, N_AUG, seed)[0],
                }
                # 理想增广源:从真实目标域的"池"(既非支撑、也非测试)里按类取样。
                # 每类不足 N_AUG 时有放回抽样,保证形状与其它增广源一致(它只是上界参照)。
                rng = np.random.RandomState(seed)
                idx_true = []
                for c in range(n_class):
                    ci = np.where(y_extra == c)[0]
                    idx_true.append(rng.choice(ci, N_AUG, replace=len(ci) < N_AUG))
                idx_true = np.concatenate(idx_true)
                gens["true_aug"] = X_extra[idx_true]

                for n, Xg in gens.items():
                    acc[n].append(run(np.concatenate([X_sup, Xg]), np.concatenate([y_sup, y_gen]),
                                      X_test, y_test, device, n_class, epochs, seed))
            if n_repeat > 1:
                print(f"    k={k} 第{rep+1}遍完成(support_only {np.mean(acc['support_only'][-n_seed:]):.3f})",
                      flush=True)

        row = f"{k:>3} " + "".join(f"{np.mean(acc[n]):>9.3f}±{np.std(acc[n]):<7.3f}" for n in CONFIGS)
        print(row, flush=True)
        out["by_k"][str(k)] = {n: {"mean": float(np.mean(acc[n])),
                                   "std": float(np.std(acc[n])),
                                   "per_run": [float(v) for v in acc[n]],
                                   "n_runs": len(acc[n])} for n in CONFIGS}
    return out


def main():
    tgt_load = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    src_load = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    n_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    n_repeat = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    out = result(tgt_load, src_load, epochs, n_seed, n_repeat=n_repeat)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
