"""诊断:孪生合成域到底有多"真"?

两个问题:
  A. 只在合成域上训练,直接测真实目标域 —— 能到多少?(sim-real gap 的直接度量)
  B. 域判别器区分 "合成特征 vs 真实源特征" 的准确率 —— 能不能一眼分开?

若 A 接近随机水平,说明孪生信号物理上不够逼真,主方法的动机(用孪生补样本)就不成立。

用法: python tools/probe_simreal.py [src_load] [tgt_load] [epochs]
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import CLASS_SETS, build_cwru_domain, split_target_fewshot  # noqa: E402
from dt_synth import CWRU_FR, synth_dataset  # noqa: E402
from models import build_model, grad_reverse  # noqa: E402
from train import compute_metrics, evaluate, make_loader  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)


def main():
    src_load = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tgt_load = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    window, classes = 1024, CLASS_SETS[10]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 合成域(全部用来训练)+ 真实源域(只用于度量分布差)+ 真实目标域(测试)
    X_dt, y_dt = synth_dataset(classes, 400, window, 12000, CWRU_FR[src_load], seed=0)
    X_src, y_src = build_cwru_domain("data/cwru", src_load, classes, window=window)
    X_tgt, y_tgt = build_cwru_domain("data/cwru", tgt_load, classes, window=window)
    _, _, test_idx = split_target_fewshot(X_tgt, y_tgt, 5, 0)
    X_test, y_test = X_tgt[test_idx], y_tgt[test_idx]

    print(f"[A] 仅用合成域训练 -> 真实目标域测试  (合成 n={len(X_dt)}, 测试 n={len(X_test)})")
    backbone, classifier, disc = build_model(len(classes), device)
    params = list(backbone.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    ce = nn.CrossEntropyLoss()
    loader = make_loader(X_dt, y_dt, 64, shuffle=True)
    for ep in range(1, epochs + 1):
        backbone.train(); classifier.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = ce(classifier(backbone(xb.unsqueeze(1).to(device))), yb.to(device))
            loss.backward(); opt.step()
    m = evaluate(backbone, classifier, X_test, y_test, device)
    print(f"    合成->真实 acc={m['acc']:.4f}  macro_f1={m['macro_f1']:.4f}   "
          f"(10 类随机水平 = 0.1000)")

    # 同规模对照:用同等数量的真实源域样本训练,测真实目标域
    print(f"[对照] 仅用真实源域训练(同规模) -> 真实目标域测试")
    n = len(X_dt)
    sub = np.random.RandomState(0).choice(len(X_src), min(n, len(X_src)), replace=False)
    backbone2, classifier2, _ = build_model(len(classes), device)
    opt2 = torch.optim.Adam(list(backbone2.parameters()) + list(classifier2.parameters()), lr=1e-3)
    loader2 = make_loader(X_src[sub], y_src[sub], 64, shuffle=True)
    for ep in range(1, epochs + 1):
        backbone2.train(); classifier2.train()
        for xb, yb in loader2:
            opt2.zero_grad()
            loss = ce(classifier2(backbone2(xb.unsqueeze(1).to(device))), yb.to(device))
            loss.backward(); opt2.step()
    m2 = evaluate(backbone2, classifier2, X_test, y_test, device)
    print(f"    真实源->真实目标 acc={m2['acc']:.4f}  macro_f1={m2['macro_f1']:.4f}")

    # [B] 域判别器区分 合成 vs 真实源(用上面训好的特征提取器)
    print(f"[B] 域判别器区分 '合成特征 vs 真实源特征'")
    def feats(bb, X, bs=512):
        bb.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.from_numpy(X[i:i + bs]).unsqueeze(1).to(device)
                out.append(bb(xb).cpu())
        return torch.cat(out)

    for name, bb in (("合成训练的骨干", backbone), ("真实源训练的骨干", backbone2)):
        z_syn = feats(bb, X_dt[:2000])
        z_real = feats(bb, X_src[:2000])
        d = nn.Sequential(nn.Linear(z_syn.shape[1], 64), nn.ReLU(),
                          nn.Linear(64, 1)).to(device)
        od = torch.optim.Adam(d.parameters(), lr=1e-3)
        bce = nn.BCEWithLogitsLoss()
        Z = torch.cat([z_syn, z_real]).to(device)
        L = torch.cat([torch.ones(len(z_syn)), torch.zeros(len(z_real))]).to(device)
        for _ in range(400):
            od.zero_grad()
            idx = torch.randperm(len(Z))
            bce(d(Z[idx]).squeeze(-1), L[idx]).backward()
            od.step()
        with torch.no_grad():
            acc = (((d(Z).squeeze(-1) > 0).float() == L).float().mean().item())
        print(f"    {name}: 判别准确率 = {acc:.4f}  (0.5=完全分不开, 1.0=一判就准)")


if __name__ == "__main__":
    main()
