"""诊断:目标域指纹标定的增广域,到底有多接近真实目标域?

候选合成源(同一标尺、多种子取均值):
  dt            : 原物理孪生(按**源域**转速布置冲击串,物理参数已标定)
  fp_rand       : 目标支撑集指纹着色 + 白噪声激励(无物理结构)
  fp_phys       : 目标支撑集指纹着色 + **按目标转速**布置的物理冲击串
  fp_phys_env   : 同上,但指纹先做形态学开运算削掉故障谐波梳齿(避免梳齿重复计数)
  src_rand      : **源域**指纹着色 + 白噪声(不用任何目标标签)
  src_phys      : **源域**指纹着色 + 目标转速冲击串(不用任何目标标签)

对每种合成源测两件事:
  A. 只用它训练 -> 真实目标域测试集上的 macro_f1(10 类随机水平 0.10)
  B. 域判别器区分 "合成特征 vs 真实目标域特征" 的准确率(0.5 分不开,1.0 一眼分开)

src_* 系列用来回答一个关键问题:**类指纹是不是本身就是跨负载不变的?**
若是,则"目标域指纹标定"这一动机不成立 —— 源域就能给出同样的东西。

用法: python tools/probe_fp.py [tgt_load] [k] [n_per_class] [epochs] [seeds]
输出: results_paper/tab_synth.json(供 tools/verify_paper_numbers.py 逐条核对表 I)
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
from train import evaluate, make_loader  # noqa: E402
from twin_calib import estimate_fingerprints, fingerprint_transfer_cost, synth_target_like  # noqa: E402

WINDOW = 1024
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(HERE, "results_paper", "tab_synth.json")


def train_and_test(X_tr, y_tr, X_test, y_test, device, n_class, epochs, seed):
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
    return evaluate(bb, cl, X_test, y_test, device), bb


def disc_acc(bb, X_syn, X_real, device, seed=0):
    """在给定骨干特征上单独训一个域判别器,返回它区分两者的准确率。"""
    torch.manual_seed(seed)
    def feats(X, bs=512):
        bb.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.from_numpy(X[i:i + bs]).unsqueeze(1).to(device)
                out.append(bb(xb).cpu())
        return torch.cat(out)

    z_s = feats(X_syn[:2000]); z_r = feats(X_real[:2000])
    d = nn.Sequential(nn.Linear(z_s.shape[1], 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    od = torch.optim.Adam(d.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    Z = torch.cat([z_s, z_r]).to(device)
    L = torch.cat([torch.ones(len(z_s)), torch.zeros(len(z_r))]).to(device)
    for _ in range(400):
        od.zero_grad()
        idx = torch.randperm(len(Z))
        bce(d(Z[idx]).squeeze(-1), L[idx]).backward()
        od.step()
    with torch.no_grad():
        return ((d(Z).squeeze(-1) > 0).float() == L).float().mean().item()


def run(tgt_load=3, k=5, n_per_class=400, epochs=40, n_seed=3, src_load=0, n_repeat=1):
    """跑完所有合成源,返回一张可直接核对表 I 的字典(含配置与逐 seed 原始值)。

    配置一并落盘:表 I 的每个数字都要能追到"哪套参数跑出来的"。曾有过
    `tab:pb_leak` 因配置不可考而整表作废的先例(见接手文档 4.8)。

    **n_repeat**:同配置整跑 n_repeat 遍。这是必要的,不是奢侈 —— 本仓库的 GPU 内核
    非确定性足以让"10 个 seed 的均值"在两次同命令运行间摆动 0.03(实测 target
    fingerprint + white noise:runA 0.8935 / runB 0.9264,而单次 3 seed 更容易被
    偶然的紧致抽样骗到,曾给出 std 0.005 而真值约 0.08)。报告的数字取**全部
    n_repeat × n_seed 次运行**的均值 ± 标准差。
    """
    classes = CLASS_SETS[10]
    n_class = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_src, y_src = build_cwru_domain("data/cwru", src_load, classes, window=WINDOW)
    X_tgt, y_tgt = build_cwru_domain("data/cwru", tgt_load, classes, window=WINDOW)
    sup_idx, _, test_idx = split_target_fewshot(X_tgt, y_tgt, k, 0)
    X_sup, y_sup = X_tgt[sup_idx], y_tgt[sup_idx]
    X_test, y_test = X_tgt[test_idx], y_tgt[test_idx]
    print(f"源 load{src_load}(fr={CWRU_FR[src_load]:.2f}Hz) -> 目标 load{tgt_load}"
          f"(fr={CWRU_FR[tgt_load]:.2f}Hz)   支撑 {len(X_sup)}  测试 {len(X_test)}  k={k}  "
          f"每类 {n_per_class} 条  {epochs} epoch  {n_seed} seed x {n_repeat} 遍\n")

    # ---- 指纹估计质量 ----
    A_k = estimate_fingerprints(X_sup, y_sup, n_class, window=WINDOW)
    A_full = estimate_fingerprints(X_tgt, y_tgt, n_class, window=WINDOW)
    A_src = estimate_fingerprints(X_src, y_src, n_class, window=WINDOW)
    A_k_env = estimate_fingerprints(X_sup, y_sup, n_class, window=WINDOW, envelope=True)

    sim = fingerprint_transfer_cost(A_full, A_k)
    print(f"指纹估计质量: k={k} 个支撑样本 vs 全区  ->  平均余弦 {sim.mean():.3f}  最差 {sim.min():.3f}")
    sim_src = fingerprint_transfer_cost(A_full, A_src)
    print(f"**源域指纹 vs 目标域全区指纹**           ->  平均余弦 {sim_src.mean():.3f}  "
          f"最差 {sim_src.min():.3f}")
    print("   (逐类: " + "  ".join(f"{c}={s:.3f}" for c, s in zip(classes, sim_src)) + ")\n")

    y_gen = np.repeat(np.arange(n_class), n_per_class)
    sources = {
        "Physics twin (source speed)": lambda s: synth_dataset(
            classes, n_per_class, WINDOW, 12000, CWRU_FR[src_load], seed=s)[0],
        "Source fingerprint + white noise": lambda s: synth_target_like(
            classes, n_per_class, WINDOW, 12000, CWRU_FR[tgt_load], A_src, seed=s, physics=False)[0],
        "Source fingerprint + impulse train": lambda s: synth_target_like(
            classes, n_per_class, WINDOW, 12000, CWRU_FR[tgt_load], A_src, seed=s, physics=True)[0],
        "Target fingerprint + white noise": lambda s: synth_target_like(
            classes, n_per_class, WINDOW, 12000, CWRU_FR[tgt_load], A_k, seed=s, physics=False)[0],
        "Target fingerprint + impulse train": lambda s: synth_target_like(
            classes, n_per_class, WINDOW, 12000, CWRU_FR[tgt_load], A_k, seed=s, physics=True)[0],
        "Target fingerprint + impulse (comb-free)": lambda s: synth_target_like(
            classes, n_per_class, WINDOW, 12000, CWRU_FR[tgt_load], A_k_env, seed=s, physics=True)[0],
    }

    out = {"config": {"src_load": src_load, "tgt_load": tgt_load, "k": k,
                      "n_per_class": n_per_class, "epochs": epochs, "n_seed": n_seed,
                      "n_repeat": n_repeat, "window": WINDOW, "classes": list(classes)},
           "fingerprint_cos_sup_vs_full": [float(v) for v in sim],
           "fingerprint_cos_src_vs_full": [float(v) for v in sim_src],
           "sources": {}}
    print(f"{'合成源':<44}{'→真实目标域 macro_f1':>22}{'判别器可分性':>16}")
    print("-" * 84)
    for name, gen in sources.items():
        f1s, dscs = [], []
        for rep in range(n_repeat):
            for s in range(n_seed):
                X_syn = gen(s)
                m, bb = train_and_test(X_syn, y_gen, X_test, y_test, device, n_class, epochs, s)
                f1s.append(m["macro_f1"])
                dscs.append(disc_acc(bb, X_syn, X_test, device, s))
            if n_repeat > 1:
                print(f"    [{name} 第{rep+1}遍] {np.mean(f1s[-n_seed:]):.4f}", flush=True)
        print(f"{name:<44}{np.mean(f1s):>10.4f} ± {np.std(f1s):<8.4f}{np.mean(dscs):>12.4f}")
        out["sources"][name] = {"f1_per_run": [float(v) for v in f1s],
                                "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                                "n_runs": len(f1s), "disc_mean": float(np.mean(dscs))}

    print("\n(10 类随机水平 macro_f1=0.10;判别器 0.5=完全分不开, 1.0=一眼分开)")
    return out


def main():
    tgt_load = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_per_class = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    n_seed = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    n_repeat = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    out = run(tgt_load, k, n_per_class, epochs, n_seed, n_repeat=n_repeat)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
