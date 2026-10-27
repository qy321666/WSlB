"""换 backbone 复核三层分解:结论是数据的,还是这颗 ResNet 的?

表 V/VI 的每一项都由同一个 1D ResNet(118 368 参数)测出。审稿人可能问
"换个网络还成立吗"。这里把同一套 L1/L2/L3 划分、同一批 seed、同一份窗数据
搬到另一个网络上重跑,训练超参(Adam 1e-3、batch 64、200 epoch)保持不变。

两条 arch 路径:

    smallresnet  同一族 ResNet,但通道只有 1/4(out_dim 32,约 3 万参数)。
                 训练过程与原网络同构、收敛稳定 —— **论文若引这一项,用它**。

    cnn          另起炉灶的 3 层 1D CNN(无残差、GAP)。2026-10-27 实测这条
                 路线在本文配方下**训练不稳定**:seed 0 上 200 epoch 的训练集
                 准确率在 0.36–0.89 之间震荡,lr 降到 3e-4 仍不收敛。训练不收敛
                 的网络测出来的分解项由优化噪声主导,不能当作"对架构稳健"的证据,
                 故结果未写入论文;保留这条路径只为记下判据,免得日后重复踩坑。

产物 results_pb/pb_levels_<arch>.json,与 pb_levels.json 同结构;
不改动表 V/VI 的任何数字。

用法: python tools/probe_pb_backbone.py [epochs] [seeds] [--arch=smallresnet|cnn]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Backbone, Classifier  # noqa: E402
from pb_utils import CLASSES_LOC, OP_CONDITIONS, list_bearings  # noqa: E402
from probe_pb_levels import (N_FIT, N_HELD, RES, WINDOW, balanced_split,  # noqa: E402
                             load_all)
from train import evaluate, make_loader, set_seed  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "paderborn")


class AltCNN(nn.Module):
    """独立的 3 层 1D CNN(无残差)。见模块说明:本文配方下不收敛,勿用于结论。"""

    def __init__(self, n_class, ch=(32, 64, 128)):
        super().__init__()

        def block(cin, cout, k):
            return nn.Sequential(
                nn.Conv1d(cin, cout, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(2))

        self.features = nn.Sequential(block(1, ch[0], 7), block(ch[0], ch[1], 5),
                                      block(ch[1], ch[2], 3),
                                      nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(ch[2], n_class)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


def build(arch, n_class, device):
    """返回 (backbone, head);head 可为 Identity,统一交给 evaluate(bb, head, ...)。"""
    if arch == "cnn":
        return AltCNN(n_class).to(device), nn.Identity()
    bb = Backbone(channels=(16, 32, 64), out_dim=32).to(device)
    return bb, Classifier(bb.head.out_features, n_class).to(device)


def n_params(arch, n_class):
    dev = torch.device("cpu")
    bb, head = build(arch, n_class, dev)
    return sum(p.numel() for p in bb.parameters()) + sum(p.numel() for p in head.parameters())


def fit(model, head, X, y, device, epochs, seed):
    set_seed(seed)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train(); head.train()
        for xb, yb in make_loader(X, y, 64, shuffle=True):
            opt.zero_grad()
            ce(head(model(xb.unsqueeze(1).to(device))), yb.to(device)).backward()
            opt.step()
    return model, head


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    arch = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--arch=")),
                "smallresnet")
    epochs = int(pos[0]) if len(pos) > 0 else 200
    n_seed = int(pos[1]) if len(pos) > 1 else 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond = OP_CONDITIONS[3]
    X, y, br, num, lab = load_all(ROOT, cond)
    codes = [b for b in list_bearings(ROOT) if lab(b)]  # 与 probe_pb_levels 同一份轴承表
    n_par = n_params(arch, len(CLASSES_LOC))

    print(f"[换 backbone 复核] arch={arch},{n_par} 参数;{cond},{len(X)} 窗,"
          f"{epochs} epoch,{n_seed} seed,device={device.type}")
    hdr = (f"  {'seed':<6}{'L1 同录制':>12}{'L2 跨录制':>12}{'L3 跨试件':>12}"
           f"{'L1-L2':>9}{'L2-L3':>9}{'训练集':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rec = {"L1": [], "L2": [], "L3": [], "fit": []}
    for seed in range(n_seed):
        rng = np.random.RandomState(seed)
        tr_c, te_c = balanced_split(lab, codes, rng)
        tr_b = np.isin(br, tr_c)
        fit_pool = np.where(tr_b & (num <= N_FIT))[0]
        perm = rng.permutation(len(fit_pool))
        n_te = int(0.3 * len(fit_pool))
        l1, fit_idx = fit_pool[perm[:n_te]], fit_pool[perm[n_te:]]
        l2 = np.where(tr_b & (num > N_FIT))[0]
        l3 = np.where(np.isin(br, te_c) & (num <= N_FIT))[0]

        model, head = build(arch, len(CLASSES_LOC), device)
        model, head = fit(model, head, X[fit_idx], y[fit_idx], device, epochs, seed)
        m = {k: evaluate(model, head, X[i], y[i], device)["macro_f1"]
             for k, i in (("L1", l1), ("L2", l2), ("L3", l3))}
        m["fit"] = evaluate(model, head, X[fit_idx], y[fit_idx], device)["macro_f1"]
        for k in rec:
            rec[k].append(float(m[k]))
        print(f"  {seed:<6}{m['L1']:>12.4f}{m['L2']:>12.4f}{m['L3']:>12.4f}"
              f"{m['L1'] - m['L2']:>+9.4f}{m['L2'] - m['L3']:>+9.4f}{m['fit']:>10.4f}")

    mu = {k: float(np.mean(v)) for k, v in rec.items()}
    sd = {k: float(np.std(v)) for k, v in rec.items()}
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'均值':<6}{mu['L1']:>12.4f}{mu['L2']:>12.4f}{mu['L3']:>12.4f}"
          f"{mu['L1'] - mu['L2']:>+9.4f}{mu['L2'] - mu['L3']:>+9.4f}{mu['fit']:>10.4f}")
    print(f"\n  总乐观量 L1-L3 = {mu['L1'] - mu['L3']:+.4f}  "
          f"(会话项 {mu['L1'] - mu['L2']:+.4f} + 试件项 {mu['L2'] - mu['L3']:+.4f})")

    os.makedirs(RES, exist_ok=True)
    fn = os.path.join(RES, f"pb_levels_{arch}.json")
    json.dump({"cond": cond, "arch": arch, "n_params": int(n_par),
               "epochs": epochs, "seeds": n_seed,
               "per_seed": rec, "mean": mu, "std": sd,
               "n_fit_files": N_FIT, "n_held_files": N_HELD, "window": WINDOW},
              open(fn, "w"), indent=1)
    print(f"  -> {fn}")


if __name__ == "__main__":
    main()
