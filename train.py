"""论文一统一训练入口。

七种方法:
  src_pure    : **纯源域**训练,不碰任何目标域数据(无支撑集、无对抗)—— 真正的零标签下界
  source_only : 源域 + 目标支撑集(其余方法也都用支撑集,这一项是与它们对齐的公平下界)
  dann        : 源域 + 目标无标签池 做对抗域适应(DANN)           —— 同时是 A2 消融(去孪生域)
  dt_dann     : 源域 + 孪生合成域 + 目标支撑集,对抗域适应 + 可选伪标签自训练 —— 原主方法
  rand_dann   : A3 消融 —— 把孪生合成域换成"循环平移 + 高斯噪声"的无物理增广
  fp_dann     : **现主方法**。目标域指纹标定的生成式增广 ——
                用目标支撑集(k 个/类)估计每类的谱包络(指纹),再用它给白噪声着色,
                生成任意多份"目标域风格"的带标签样本。样本属目标域风格,不参与域判别器。
  fp_phys_dann: fp_dann 的物理消融 —— 同样用目标指纹着色,但激励源换成**按目标转速布置的
                物理冲击串**(BPFO/BPFI/BSF 随转速平移)。实测这一版**更差**,见 tools/probe_fp.py。
  (--pseudo 可叠加在 dt_dann / rand_dann 上)

方法选择依据(2026-10-05 实测,详见 接手文档.md 第 4 节):
  1. 原物理孪生 dt_dann 在 10 类 S0→S3 上(5 seed macro_f1)只有 0.815,
     低于其消融 dann(0.971)、A3 rand_dann(0.971),也低于仅用孪生训练的随机水平诊断;
  2. 目标域指纹着色白噪声(fp_dann)达到 **0.9996 ± 0.0005**,且方差比 dann 小两个数量级;
  3. 在指纹之上再叠加物理冲击串(fp_phys_dann, 0.981)反而变差。原因不是类别信息被破坏 ——
     实测它的类间谱距离(0.346)**高于** fp_dann(0.315),而是分布被推离真实域:
     冲击串在时域上近乎周期,给信号强加了确定性的相位结构,使域判别器可分性从 0.925 升到 0.973。
     真实信号更接近"幅度谱给定、相位随机"的代理信号,见 tools/probe_classsep.py。

类别体系由 --classes 选择:4(正常/滚动体/内圈/外圈)或 10(再按 007/014/021 尺寸细分)。

用法示例见 run_s2.sh 与 README。
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_utils import (
    CLASS_SETS,
    CLASSES_10,
    build_cwru_domain,
    split_source_trainval,
    split_target_fewshot,
)
from dt_synth import CWRU_FR, synth_dataset
from models import build_model, grad_reverse
from twin_calib import estimate_fingerprints, synth_target_like

CLASSES = CLASSES_10
# dt_*:孪生/无物理增广域,属"源侧",参与域判别器;fp_*:目标指纹标定的生成式增广,属目标域风格,不参与
DT_METHODS = ("dt_dann", "rand_dann")
FP_METHODS = ("fp_dann", "fp_phys_dann")
METHODS = ("src_pure", "source_only", "dann") + DT_METHODS + FP_METHODS


def compute_metrics(gt, pred):
    """用 numpy 计算准确率 / 宏平均 F1 / 每类召回率(不依赖 sklearn,避免导入顺序冲突)。"""
    acc = float((gt == pred).mean())
    n_class = int(gt.max()) + 1
    recalls, f1s = [], []
    for c in range(n_class):
        tp = ((gt == c) & (pred == c)).sum()
        p = (pred == c).sum()
        r = (gt == c).sum()
        rec = float(tp / r) if r > 0 else 0.0
        prec = float(tp / p) if p > 0 else 0.0
        recalls.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return {"acc": acc, "macro_f1": float(np.mean(f1s)), "recall": recalls}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=list(METHODS), default="dt_dann")
    p.add_argument("--classes", type=int, choices=[4, 10], default=10,
                   help="类别体系:4(正常/滚动体/内圈/外圈)或 10(再按 007/014/021 尺寸细分)")
    p.add_argument("--data_root", default="data/cwru", help="CWRU .mat 所在目录")
    p.add_argument("--src_load", type=int, default=0)
    p.add_argument("--tgt_load", type=int, default=3)
    p.add_argument("--k", type=int, default=5, help="目标域每类带标签样本数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--synth_per_class", type=int, default=400)
    p.add_argument("--w_dt", type=float, default=1.0, help="孪生合成域损失权重")
    p.add_argument("--w_sup", type=float, default=1.0, help="目标支撑集损失权重")
    p.add_argument("--pseudo", action="store_true", help="开启目标域伪标签自训练")
    p.add_argument("--pseudo_tau", type=float, default=0.90, help="伪标签置信度阈值")
    p.add_argument("--pseudo_every", type=int, default=5, help="每 N 个 epoch 更新一次伪标签")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--update", choices=["adam", "manual"], default="adam",
                   help="adam=标准优化器(默认); manual=纯张量手写梯度下降,"
                        "仅供无 GPU/受限沙箱环境做逻辑冒烟测试时使用")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 注意:这里**刻意不设** torch.backends.cudnn.deterministic=True。
    # 2026-10-05 试过,本仓库的小骨干 + 这个数据集上它会让部分 seed 训练直接崩掉
    # (seed 1 训练集 macro-F1 只有 0.52,而默认路径是 1.000)。复现性靠"报告多种子
    # 均值 ± 标准差"来保证,不靠逐位可复现。同配置同 seed 的两次运行经验上会差
    # 到小数点后第三位(高方差配置更多),比较差异时请以 std 为尺度。


def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def rand_augment_dataset(X_src, y_src, n_class, n_per_class, seed=0):
    """A3 消融用的无物理增广:对源域样本做循环平移 + 加高斯噪声。

    与孪生合成域保持**完全相同的样本量和训练管线**,唯一区别是不含任何物理建模。
    这样 A3 与 dt_dann 的差值就只能归因于"物理约束",而不是"多了一批数据"。
    """
    rng = np.random.RandomState(seed)
    X, y = [], []
    for c in range(n_class):
        idx = np.where(y_src == c)[0]
        if len(idx) == 0:
            continue
        for _ in range(n_per_class):
            w = X_src[rng.choice(idx)].astype(np.float32).copy()
            w = np.roll(w, rng.randint(0, len(w)))
            w = w + rng.normal(0, 0.5, size=w.shape).astype(np.float32)
            X.append((w - w.mean()) / (w.std() + 1e-6))
            y.append(c)
    return np.stack(X).astype("float32"), np.array(y, dtype="int64")


def fwd(backbone, x, device):
    """骨干前向包装。

    GPU 环境为透传。受限 CPU 环境(如沙箱/无驱动虚机)的 ATen 线程池在
    backward 后可能死锁,重设单线程可规避;对正常 CPU/GPU 机器无副作用。
    """
    if device.type == "cpu":
        torch.set_num_threads(1)
    return backbone(x)


@torch.no_grad()
def evaluate(backbone, classifier, X_test, y_test, device):
    backbone.eval()
    classifier.eval()
    preds, gts = [], []
    for i in range(0, len(X_test), 512):
        xb = torch.from_numpy(X_test[i : i + 512]).unsqueeze(1).to(device)
        logits = classifier(fwd(backbone, xb, device))
        preds.append(logits.argmax(1).cpu().numpy())
        gts.append(y_test[i : i + 512])
    pred = np.concatenate(preds)
    gt = np.concatenate(gts)
    return compute_metrics(gt, pred)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 非 GPU 环境下切单线程并禁用 oneDNN:规避部分无驱动/沙箱环境的 autograd 引擎与
    # CPU 卷积内核挂起(不影响正确性;GPU 环境不受影响)
    if device.type == "cpu":
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled = False
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    classes = CLASS_SETS[args.classes]
    # dt_* / fp_* 都会额外构造一个"增广域";dann / source_only 不构造
    uses_synth = args.method in DT_METHODS + FP_METHODS

    print(f"[env] device={device}  method={args.method}  classes={args.classes}  "
          f"S{args.src_load}->S{args.tgt_load}  k={args.k}  seed={args.seed}")

    # ---- 数据 ----
    cache = os.path.join(os.path.dirname(__file__), ".cache")
    X_src, y_src = build_cwru_domain(args.data_root, args.src_load, classes,
                                     window=args.window, cache_dir=cache)
    X_tgt, y_tgt = build_cwru_domain(args.data_root, args.tgt_load, classes,
                                     window=args.window, cache_dir=cache)
    print(f"[data] source n={len(X_src)}  target n={len(X_tgt)}")

    sup_idx, pool_idx, test_idx = split_target_fewshot(X_tgt, y_tgt, args.k, args.seed)
    X_sup, y_sup = X_tgt[sup_idx], y_tgt[sup_idx]
    X_pool, y_pool = X_tgt[pool_idx], y_tgt[pool_idx]   # y_pool 训练时不用(只用于伪标签生成时记录)
    X_test, y_test = X_tgt[test_idx], y_tgt[test_idx]
    print(f"[data] support={len(sup_idx)}  pool={len(pool_idx)}  test={len(test_idx)}")

    src_loader = make_loader(X_src, y_src, args.batch_size, shuffle=True)
    sup_loader = make_loader(X_sup, y_sup, args.batch_size, shuffle=True)
    pool_loader = make_loader(X_pool, y_pool, args.batch_size, shuffle=True)

    # ---- 增广域 ----
    #   dt_dann      : 物理孪生合成(源域风格)      -> 属源侧,参与域判别
    #   rand_dann    : 无物理增广(A3)              -> 属源侧,参与域判别
    #   fp_dann      : 目标指纹着色 + 白噪声        -> 目标域风格,不参与域判别(主方法)
    #   fp_phys_dann : 目标指纹着色 + 物理冲击串     -> 同上(物理消融,实测更差)
    synth_loader = None
    dt_in_domain = args.method in DT_METHODS
    if args.method in DT_METHODS:
        if args.method == "dt_dann":
            X_dt, y_dt = synth_dataset(classes, args.synth_per_class, args.window, 12000,
                                       CWRU_FR[args.src_load], seed=args.seed)
            print(f"[dt] 孪生合成域 n={len(X_dt)}")
        else:
            X_dt, y_dt = rand_augment_dataset(X_src, y_src, len(classes),
                                              args.synth_per_class, seed=args.seed)
            print(f"[dt] 无物理增广域(A3) n={len(X_dt)}")
        synth_loader = make_loader(X_dt, y_dt, args.batch_size, shuffle=True)
    elif args.method in FP_METHODS:
        # 指纹只从**目标支撑集**估计(k 个/类),与其它方法用支撑集算 CE 的是同一份信息
        A = estimate_fingerprints(X_sup, y_sup, len(classes), window=args.window)
        physics = args.method == "fp_phys_dann"
        X_dt, y_dt = synth_target_like(
            classes, args.synth_per_class, args.window, 12000,
            CWRU_FR[args.tgt_load], A, seed=args.seed, physics=physics)
        print(f"[fp] 目标指纹标定增广域 n={len(X_dt)}  physics={physics}")
        synth_loader = make_loader(X_dt, y_dt, args.batch_size, shuffle=True)

    # ---- 模型 ----
    n_class = len(classes)
    backbone, classifier, disc = build_model(n_class, device)
    all_params = list(backbone.parameters()) + list(classifier.parameters()) + list(disc.parameters())
    if args.update == "manual":
        opt = None   # 手写更新,避免受限环境下的优化器构造问题
    else:
        opt = torch.optim.Adam(all_params, lr=args.lr)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    # ---- 训练 ----
    n_batches = max(len(src_loader), 1)
    pseudo_buf = {"X": None, "y": None}
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        classifier.train()
        disc.train()
        p = epoch / args.epochs
        alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0   # DANN lambda 调度

        # 伪标签自训练:每 pseudo_every 个 epoch 用当前模型给无标签池打高置信伪标签
        if uses_synth and args.pseudo and epoch % args.pseudo_every == 0:
            backbone.eval()
            feats, confs, preds = [], [], []
            with torch.no_grad():
                for xb, _ in pool_loader:
                    z = fwd(backbone, xb.unsqueeze(1).to(device), device)
                    prob = torch.softmax(classifier(z), dim=1)
                    conf, pred = prob.max(1)
                    feats.append(z.cpu()); confs.append(conf.cpu()); preds.append(pred.cpu())
            feats = torch.cat(feats); confs = torch.cat(confs); preds = torch.cat(preds)
            mask = confs >= args.pseudo_tau
            if mask.sum() > 0:
                pseudo_buf = {"X": feats[mask].numpy(), "y": preds[mask].numpy()}
                print(f"    [pseudo] epoch {epoch}: 置信样本 {int(mask.sum())} 个")

        src_iter = iter(src_loader)
        sup_iter = iter(sup_loader)
        dt_iter = iter(synth_loader) if synth_loader is not None else None
        pool_iter = iter(pool_loader)

        total_loss = 0.0
        for it in range(n_batches):
            if args.update == "manual":
                for p in all_params:
                    p.grad = None   # 手动清梯度(等价 opt.zero_grad(set_to_none=True))
            else:
                opt.zero_grad()
            loss = torch.zeros(1, device=device)

            # 源域分类损失
            x_src, y_src_b = next(src_iter)
            z_src = fwd(backbone, x_src.unsqueeze(1).to(device), device)
            loss = loss + ce(classifier(z_src), y_src_b.to(device))

            # 目标支撑集损失(source_only 及以上的方法都用:少量目标标签是最自然的可用信息)。
            # src_pure 是唯一的例外 —— 它是"完全不碰目标域"的零标签下界。
            if args.method != "src_pure":
                try:
                    x_sup, y_sup_b = next(sup_iter)
                    z_sup = fwd(backbone, x_sup.unsqueeze(1).to(device), device)
                    loss = loss + args.w_sup * ce(classifier(z_sup), y_sup_b.to(device))
                except StopIteration:
                    pass

            # 孪生合成域损失(dt_dann)
            if dt_iter is not None:
                try:
                    x_dt, y_dt_b = next(dt_iter)
                    z_dt = fwd(backbone, x_dt.unsqueeze(1).to(device), device)
                    loss = loss + args.w_dt * ce(classifier(z_dt), y_dt_b.to(device))
                except StopIteration:
                    pass

            # 对抗域适应(dann / dt_dann / rand_dann / fp_*):源(含孪生) vs 目标无标签池
            if args.method not in ("src_pure", "source_only"):
                try:
                    x_pool, _ = next(pool_iter)
                    z_pool = fwd(backbone, x_pool.unsqueeze(1).to(device), device)
                except StopIteration:
                    x_pool, _ = next(iter(pool_loader))
                    z_pool = fwd(backbone, x_pool.unsqueeze(1).to(device), device)
                # 源侧特征:源批 + 增广批(仅 dt_*,即源域风格的孪生/无物理增广)。
                # 均不 detach:对抗梯度经 GRL 反向影响特征提取器。
                # fp_* 的增广样本本身已是目标域风格,若并入源侧会把判别器教反,故排除。
                feats_dom = [z_src]
                if dt_iter is not None and dt_in_domain:
                    # 取一个增广批做源侧域样本
                    try:
                        x_dt2, _ = next(dt_iter)
                        feats_dom.append(fwd(backbone, x_dt2.unsqueeze(1).to(device), device))
                    except StopIteration:
                        pass
                feats_dom = torch.cat(feats_dom)
                n_src = len(feats_dom)
                dom_feats = torch.cat([feats_dom, z_pool])
                dom_labels = torch.cat([
                    torch.ones(n_src), torch.zeros(len(z_pool))]).to(device)
                loss_dom = bce(disc(grad_reverse(dom_feats, alpha)), dom_labels)
                loss = loss + loss_dom

            # 伪标签自训练损失(高置信池样本作为带标签数据)
            if args.pseudo and pseudo_buf["X"] is not None:
                zp = torch.from_numpy(pseudo_buf["X"]).to(device)
                yp = torch.from_numpy(pseudo_buf["y"]).to(device)
                loss = loss + ce(classifier(zp), yp)

            loss.backward()
            if args.update == "manual":
                # 纯张量手写梯度下降:不引入优化器对象,便于受限环境冒烟测试。
                # 与标准优化器等价(仅支持无动量/无调度的最简更新)。
                with torch.no_grad():
                    for p in all_params:
                        if p.grad is not None:
                            p.add_(p.grad, alpha=-args.lr)
            else:
                opt.step()
            total_loss += loss.item()

        # ---- 每个 epoch 末尾在目标测试集上评估(仅监控) ----
        metrics = evaluate(backbone, classifier, X_test, y_test, device)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[{epoch}/{args.epochs}] loss={total_loss/n_batches:.4f} "
                  f"test_acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f}")

    # ---- 最终评估 ----
    metrics = evaluate(backbone, classifier, X_test, y_test, device)
    print(f"\n[final] method={args.method} classes={args.classes} "
          f"S{args.src_load}->S{args.tgt_load} "
          f"k={args.k} seed={args.seed} | "
          f"acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f} "
          f"recall={['%.3f' % r for r in metrics['recall']]}")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = (f"{args.method}_c{args.classes}_s{args.src_load}to{args.tgt_load}"
           f"_k{args.k}_seed{args.seed}")
    if uses_synth and args.pseudo:
        tag += "_pseudo"
    out_path = os.path.join(args.out_dir, f"{tag}.csv")
    with open(out_path, "w") as f:
        f.write("method,src_load,tgt_load,k,seed,acc,macro_f1,recall\n")
        f.write(f"{args.method},{args.src_load},{args.tgt_load},{args.k},{args.seed},"
                f"{metrics['acc']:.4f},{metrics['macro_f1']:.4f},"
                f"{';'.join(['%.3f' % r for r in metrics['recall']])}\n")
    print(f"[save] {out_path}  (用时 {(time.time()-t0)/60:.1f} 分钟)")

    # 受限 CPU 环境(如沙箱/无驱动虚机)下 torch 的 C++ 反初始化可能死锁,直接退出收尾;
    # GPU 环境正常运行到 Python 解释器退出。
    if device.type == "cpu":
        os._exit(0)


if __name__ == "__main__":
    main()
