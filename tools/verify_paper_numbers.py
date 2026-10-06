"""把稿件里的数字逐条重算,与正文/表格声称的值对照。

分两类,覆盖正文中每一个可核对的量:
  1. **纯数据统计量**(类均值谱的余弦 / 距离 / 频带能量占比 / 两两 dB 间距 / 台架线谱):
     秒级可算,直接由 data/ 下的原始数据重算 —— 即 `raw` 组。
  2. **训练得到的表**(I/III/V/VI/VII/VIII/IX):重跑要 GPU,故改为核对**落盘产物**
     (results_paper/、results_pb/、results/final/)与正文声称值是否一致 —— `artifacts` 组。
     ⚠️ 这一组验证的是"正文与产物一致",**不是**"产物能被逐位复现":在别的机器上重跑
     会得到略有不同的值(本仓实测 10 seed 均值两次同命令相差可达 0.033,见 GPU kernel
     非确定性一节),这是论文里如实写明的。

脚本存在的意义是实证的:2026-10-05 一天之内发现两处稿件数字复现不出来
(r=0.957、Paderborn 线谱能量 90--99.8%),都是靠人工核对才发现的;
把核对写成脚本,以后每次改正文都能一键跑。

**不重复实现**:全部直接调用各 probe 的公开函数,所以"验证脚本通过"与"probe 输出的数"
在实现上不可能分叉。若某个声称值对不上,要么正文错,要么该 probe 的参数与正文不一致。

覆盖情况:正文每张表都有对应产物,由本脚本核对(对照表见补充材料 TABLE_ARTIFACTS.md)。
唯一**不在**本脚本范围内的是图 1/图 2 的图形本身 —— 它们画的是已被核对的那些数
(图 1 的前三行即表 III 的均值行,图 2 的六个量在 §II 一组里逐个核对)。

用法: python tools/verify_paper_numbers.py [all|artifacts|raw|bench,classsep,diag2,rig]
      all        全部检查(需要 data/ 下有原始数据)
      artifacts  **只查有落盘产物的表**(表 I/III/V/VI/VII/VIII/IX)——
                 不需要原始数据,审稿人拿到补充材料包就能直接跑这一条
      raw        只查需要原始数据的纯统计量(表 II/IV、§II、§PB 台架线谱)
      退出码 1 表示有与正文不符的项(可直接接在 CI / 投稿前检查里)。
"""
import csv
import glob
import itertools
import json
import os
import sys

import numpy as np

# Windows 控制台默认 GBK,直接把中文写到 stdout 会乱码(实测),统一改 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import probe_classsep as CS  # noqa: E402
import probe_spectra as SP  # noqa: E402
import probe_spectral_numbers as SN  # noqa: E402
from data_utils import CLASS_SETS, build_cwru_domain  # noqa: E402
from pb_utils import (CLASSES_LOC, OP_CONDITIONS, SHAFT_HZ,  # noqa: E402
                      class_of_location, damage_table, list_bearings, list_files,
                      load_vibration)

PB = os.path.join(HERE, "data", "paderborn")
CW = 1024
FS_CW = 12000.0
PW = 4096
FS_PB = 64000.0
COND = OP_CONDITIONS[3]

FAILS = []
N_CHECKS = 0                   # 实际核对过的项数,结束时报出来(补充材料的 README 要引它)


def chk(section, name, got, claim, tol=0.002, tight=False):
    """got / claim 可为标量,也可为 (lo, hi) 区间。

    tight=True 时,声称的区间宽度必须与实测相当(容差 tol)。这条是为"区间写得比数据宽"
    准备的:写 50--57% 而实测恒为 54% ,读起来像估算而非测量,本身就该改。
    """
    global N_CHECKS
    N_CHECKS += 1
    if isinstance(claim, tuple):
        ok = (got[0] >= claim[0] - tol) and (got[1] <= claim[1] + tol)
        txt = f"{got[0]:.2f}..{got[1]:.2f}"
        ref = f"{claim[0]:.2f}..{claim[1]:.2f}"
        if ok and tight and (claim[1] - claim[0]) > (got[1] - got[0]) + 2 * tol:
            flag = "**LOOSE**   "
            FAILS.append((section, name + " (区间过宽)", txt, ref))
        else:
            flag = "OK      " if ok else "**MISMATCH**"
        if not ok:
            FAILS.append((section, name, txt, ref))
    else:
        ok = abs(got - claim) <= tol
        txt, ref = f"{got:.4f}", f"{claim:.4f}"
        flag = "OK      " if ok else "**MISMATCH**"
        if not ok:
            FAILS.append((section, name, txt, ref))
    print(f"  [{flag}] {name:<42} 算得 {txt:<16} 正文 {ref}")


# ------------------------------------------------------- 表 IV：跨基准对照
def table_bench():
    print("\n表 IV  tab:bench  —— 类均值幅度谱余弦(未平滑,两基准同一算法)")
    c = SP.cwru_part()
    for l in ("0", "3"):
        chk("表IV", f"CWRU 类间 load{l}", c["between_class_cos"][l], 0.626)
    chk("表IV", "CWRU 同类 load0 vs load1", c["same_class_across_load_cos"]["S0->S1"][0], 0.870)

    for tag, hp, w, b in (("原始", 0.0, 1.000, 1.000), ("高通 2 kHz", 2000.0, 0.992, 0.989)):
        p = SP.paderborn_part(PB, max_files=4, hp=hp)
        wi = p["within_class_across_bearing_cos"][0]
        bw = p["between_class_across_bearing_cos"][0]
        chk("表IV", f"Paderborn {tag} 类内", wi, w)
        chk("表IV", f"Paderborn {tag} 类间", bw, b)
        chk("表IV", f"Paderborn {tag} 裕度", wi - bw, 0.000 if not hp else 0.003)
    chk("表IV", "Paderborn 独立轴承数",
        SP.paderborn_part(PB, max_files=4)["n_bearings"], 21, tol=0)


# ------------------------------------------------------- 表 II：类间余弦距离
def table_classsep():
    print("\n表 II  tab:classsep  —— 类间平均余弦距离(窗长 1024,每类 400 样本)")
    classes = CLASS_SETS[10]
    nc = len(classes)
    n_pc = 400
    X3, y3 = build_cwru_domain(os.path.join(HERE, "data", "cwru"), 3, classes, window=CW)
    X0, y0 = build_cwru_domain(os.path.join(HERE, "data", "cwru"), 0, classes, window=CW)
    sup, _, _ = CS.split_target_fewshot(X3, y3, 5, 0)
    A_k = CS.estimate_fingerprints(X3[sup], y3[sup], nc, window=CW)
    A_env = CS.estimate_fingerprints(X3[sup], y3[sup], nc, window=CW, envelope=True)

    y_gen = np.repeat(np.arange(nc), n_pc)
    srcs = {
        "真实目标域 load3": (X3, y3, 0.374),
        "真实源域 load0": (X0, y0, 0.374),
        "原物理孪生": (CS.synth_dataset(classes, n_pc, CW, 12000, CS.CWRU_FR[0], seed=0)[0],
                       y_gen, 0.167),
        "指纹+白噪声": (CS.synth_target_like(classes, n_pc, CW, 12000, CS.CWRU_FR[3],
                                             A_k, seed=0, physics=False)[0], y_gen, 0.315),
        "指纹+冲击串": (CS.synth_target_like(classes, n_pc, CW, 12000, CS.CWRU_FR[3],
                                             A_k, seed=0, physics=True)[0], y_gen, 0.346),
        "去梳齿包络+冲击串": (CS.synth_target_like(classes, n_pc, CW, 12000, CS.CWRU_FR[3],
                                                   A_env, seed=0, physics=True)[0],
                             y_gen, 0.319),
    }
    for name, (X, y, claim) in srcs.items():
        mean_d, _, _ = CS.sep(CS.class_mean_spectra(X, y, nc))
        chk("表II", name, mean_d, claim)


# --------------------------------------- §Diagnosis II：平滑包络 + 频带能量 + dB
def diag2():
    print("\n§Diagnosis II  —— 平滑谱包络 A_c(f) 的跨负载稳定性")
    classes, X, y = SN.load_data()
    nc = len(classes)
    A = {l: SN.estimate_fingerprints(X[l], y[l], nc, window=SN.WINDOW, smooth=5)
         for l in (0, 1, 2, 3)}
    for t, claim in ((1, 0.954), (2, 0.939), (3, 0.924)):
        chk("§II", f"load0 -> load{t} 余弦",
            np.mean([SN.cos(A[0][i], A[t][i]) for i in range(nc)]), claim)
    chk("§II", "load0 对另三负载均值",
        np.mean([np.mean([SN.cos(A[0][i], A[t][i]) for i in range(nc)]) for t in (1, 2, 3)]),
        0.939)
    allp = [np.mean([SN.cos(A[s][i], A[t][i]) for i in range(nc)])
            for s, t in itertools.combinations((0, 1, 2, 3), 2)]
    chk("§II", "全部 6 对均值", np.mean(allp), 0.944)

    print("\n§Diagnosis II / 图 2  —— 2800--3300 Hz 频带能量占比与两两 dB 间距")
    be = SN.band_energy(classes, X, y, nc)
    r, t = be["真实 load0,九个故障类"], be["物理孪生, 九个故障类"]
    chk("§II", "真实九类能量占比",
        (r["min"], r["max"]), (9, 67), tol=1.0, tight=True)
    print(f"           └ 最小 {r['min_cls']} {r['min']:.0f}%  最大 {r['max_cls']} "
          f"{r['max']:.0f}%  均值 {r['mean']:.0f}%")
    chk("§II", "孪生九类能量占比",
        (t["min"], t["max"]), (48, 64), tol=1.0, tight=True)
    print(f"           └ 均值 {t['mean']:.0f}%  (正文 55%)")
    chk("§II", "孪生九类能量占比均值", t["mean"], 55.0, tol=1.0)

    d = SN.pairwise_db(classes, X[0], y[0], nc)
    rt, tw = d["真实 load0"], d["物理孪生"]
    chk("图2", "真实九类两两间距中位", rt["pair_med"], 6.7, tol=0.1)
    chk("图2", "真实九类两两间距最小", rt["pair_min"], 3.7, tol=0.1)
    chk("图2", "真实九类逐频点展宽中位", rt["span_med"], 18.8, tol=0.1)
    chk("图2", "孪生九类两两间距中位", tw["pair_med"], 2.5, tol=0.1)
    chk("图2", "孪生九类两两间距最小", tw["pair_min"], 0.9, tol=0.1)
    chk("图2", "孪生九类逐频点展宽中位", tw["span_med"], 6.5, tol=0.1)


# ------------------------------------------------------- §Paderborn 台架线谱
def rig_tone():
    print("\n§Paderborn  —— 台架线谱(21 颗轴承,4096 点窗)")
    tbl = damage_table(PB)
    lab = lambda b: class_of_location(b, tbl)  # noqa: E731
    codes = [b for b in list_bearings(PB) if lab(b)]
    chk("§PB", "轴承数", len(codes), 21, tol=0)
    chk("§PB", "4 x 轴频 (N15)", 4 * SHAFT_HZ["N15"], 100.0, tol=0.01)

    c0 = int(np.argmin(np.abs(np.fft.rfftfreq(PW, 1.0 / FS_PB) - 100.0)))
    peaks, one, near, five = [], [], [], []
    for b in codes:
        x = load_vibration(list_files(PB, b, COND)[0])[:131072]
        X = np.fft.rfft(x * np.hanning(len(x)))
        P = np.abs(X) ** 2
        pk = np.fft.rfftfreq(len(x), 1.0 / FS_PB)[1:][np.argmax(P[1:])]
        peaks.append(pk)
        n = len(x) // PW
        W_ = np.stack([x[i * PW:(i + 1) * PW] for i in range(n)])
        Pw = (np.abs(np.fft.rfft(W_ * np.hanning(PW), axis=-1)) ** 2).mean(0)
        for k, store in ((1, near), (2, five), (0, one)):
            store.append(100.0 * Pw[np.abs(np.arange(Pw.size) - c0) <= k].sum() / Pw.sum())
    chk("§PB", "谱峰频率范围 (Hz)", (float(min(peaks)), float(max(peaks))), (99.6, 100.1),
        tol=0.05, tight=True)
    chk("§PB", "单 bin 能量占比 (%)", (float(min(one)), float(max(one))), (53.97, 54.31),
        tol=0.05)
    chk("§PB", "峰±2 bin 能量占比 (%)", (float(min(five)), float(max(five))), (99.5, 99.8),
        tol=0.05, tight=True)
    print(f"           └ 单 bin {min(one):.2f}..{max(one):.2f}%   "
          f"±1 bin {min(near):.2f}..{max(near):.2f}%   "
          f"±2 bin {min(five):.2f}..{max(five):.2f}%")
    print("             注:单 bin 约 54%、左右各一个 bin 再占约 45%,正是 Hann 窗下单频线"
          "的 0.5 : 0.25 : 0.25 分布 —— 可交叉印证这就是单根纯音,不是宽带结构。")


# ------------------------------------------------------- 表 VII：Paderborn 无源少样本
def table_pb_fewshot():
    """表 VII 由 tools/probe_paderborn.py 的 part_d 产出,落盘 results_pb/d_twin_k*.json。

    与其它训练表不同,这里**有**落盘产物,所以可以直接核对(而不是只能凭记忆)。
    待核对项里数这张最险 —— 抽查时确实抓到一处:
    k=50 的"指纹标定(无物理)"0.16846 曾被写成 0.169(应为 0.168),
    且题注原写"three seeds",而 k=5 那一行实为 5 个 seed。
    """
    print("\n表 VII  tab:pb_fewshot  —— Paderborn 无源少样本(results_pb/d_twin_k*.json)")
    cols = ["真实支撑集(仅 k 个/类)", "纯孪生(无标定)", "指纹标定(无物理)", "指纹标定+物理"]
    claim = {5: (0.286, 0.166, 0.203, 0.199),
             20: (0.316, 0.172, 0.177, 0.266),
             50: (0.325, 0.166, 0.168, 0.231)}
    for k in (5, 20, 50):
        path = os.path.join(HERE, "results_pb", f"d_twin_k{k}.json")
        if not os.path.exists(path):
            chk("表VII", f"k={k} 落盘文件", np.nan, 0.0)
            continue
        d = json.load(open(path))
        for col, ref in zip(cols, claim[k]):
            v = d[col]
            chk("表VII", f"k={k:<2} {col} (n={len(v)})", float(np.mean(v)), ref, tol=0.0005)


# ------------------------------------------------------- 表 I：只训合成域的迁移
def table_synth():
    """表 I 由 tools/probe_fp.py 产出 results_paper/tab_synth.json。

    这一列是"多遍重复"的:2026-10-06 发现本仓 GPU 内核非确定性足以让 10 seed 的均值
    在两次同命令运行间摆动 0.03(target fp+white:runA 0.8935 / runB 0.9264),而单次
    3 seed 更会给出 std 0.005 的假紧致(真值约 0.077)。故正文改为报 20 次(2 遍 x 10 seed)。
    """
    print("\n表 I  tab:synth  —— 只训合成域、测真实目标域(results_paper/tab_synth.json)")
    path = os.path.join(HERE, "results_paper", "tab_synth.json")
    if not os.path.exists(path):
        print("  [--      ] 尚无落盘产物:先跑 python tools/probe_fp.py 3 5 400 40 10 2")
        return
    d = json.load(open(path, encoding="utf-8"))
    print(f"  (n_repeat={d['config'].get('n_repeat')}  x n_seed={d['config']['n_seed']})")
    claim = {"Physics twin (source speed)": (0.123, 0.030, 0.9999),
             "Source fingerprint + white noise": (0.587, 0.045, 0.9972),
             "Source fingerprint + impulse train": (0.491, 0.041, 0.9994),
             "Target fingerprint + impulse train": (0.651, 0.056, 0.9786),
             "Target fingerprint + impulse (comb-free)": (0.524, 0.061, 0.9999),
             "Target fingerprint + white noise": (0.910, 0.077, 0.9337)}
    for name, (m, s, disc) in claim.items():
        if name not in d["sources"]:
            chk("表I", f"{name} (缺)", np.nan, 0.0)
            continue
        v = d["sources"][name]
        chk("表I", f"{name[:38]:<38} F1", v["f1_mean"], m, tol=0.0005)
        chk("表I", f"{name[:38]:<38} F1 std", v["f1_std"], s, tol=0.0005)
        chk("表I", f"{name[:38]:<38} 判别器", v["disc_mean"], disc, tol=0.0005)
    # 正文 §Diagnosis I 的定性句:target fp + white 是最高分,叠物理反而更低
    src = d["sources"]
    if "Target fingerprint + white noise" in src:
        best = src["Target fingerprint + white noise"]["f1_mean"]
        for other in ("Target fingerprint + impulse train",
                      "Target fingerprint + impulse (comb-free)",
                      "Source fingerprint + white noise"):
            chk("表I", f"`{other[:30]}' 低于最高分", 1.0 if best > src[other]["f1_mean"] else 0.0,
                1.0, tol=0)


# ------------------------------------------------------- 表 VIII：CWRU 主表
def table_main():
    """表 VIII 的产物是 results/final/*.csv(7 方法 x 5 seed x 单一方向),已落盘。"""
    print("\n表 VIII tab:main  —— CWRU 十类 S0->S3,5 seed(results/final/*.csv)")
    claim = {"src_pure": (0.8188, 0.0421, 0.7307, 0.0225),
             "source_only": (0.8842, 0.0377, 0.8228, 0.0686),
             "dann": (0.9904, 0.0179, 0.9858, 0.0268),
             "rand_dann": (0.9512, 0.0553, 0.9223, 0.0977),
             "dt_dann": (0.9028, 0.1212, 0.9141, 0.0780),
             "fp_phys_dann": (0.9917, 0.0091, 0.9891, 0.0123),
             "fp_dann": (0.9998, 0.0003, 0.9998, 0.0004)}
    rows = {}
    for f in glob.glob(os.path.join(HERE, "results", "final", "*.csv")):
        for r in csv.DictReader(open(f)):
            rows.setdefault(r["method"], []).append((float(r["acc"]), float(r["macro_f1"])))
    for m, (ca, sa, cf, sf) in claim.items():
        if m not in rows:
            chk("表VIII", f"{m} (缺 CSV)", np.nan, 0.0)
            continue
        a = np.array([x[0] for x in rows[m]])
        f1 = np.array([x[1] for x in rows[m]])
        chk("表VIII", f"{m:<13} acc (n={len(a)})", float(a.mean()), ca, tol=0.0006)
        chk("表VIII", f"{m:<13} acc std", float(a.std()), sa, tol=0.0006)
        chk("表VIII", f"{m:<13} macro-F1", float(f1.mean()), cf, tol=0.0006)
        chk("表VIII", f"{m:<13} macro-F1 std", float(f1.std()), sf, tol=0.0006)


# ------------------------------------------------------- 表 V / VI：Paderborn 泄漏
def table_pb_levels():
    """表 V(tab:pb_leak)与表 VI(tab:pb_levels)同出 results_pb/pb_levels.json。"""
    print("\n表 V/VI  tab:pb_leak、tab:pb_levels  —— results_pb/pb_levels.json")
    path = os.path.join(HERE, "results_pb", "pb_levels.json")
    if not os.path.exists(path):
        chk("表V/VI", "落盘文件", np.nan, 0.0)
        return
    d = json.load(open(path, encoding="utf-8"))
    m, s = d["mean"], d["std"]
    chk("表V", "同录制 (L1) 均值", m["L1"], 0.938)
    chk("表V", "轴承不相交 (L3) 均值", m["L3"], 0.597)
    chk("表V", "乐观量 +L1-L3", m["L1"] - m["L3"], 0.340)
    chk("表V", "同录制 std", s["L1"], 0.013)
    chk("表V", "轴承不相交 std", s["L3"], 0.139)
    chk("表VI", "L2 跨录制同轴承", m["L2"], 0.755)
    chk("表VI", "会话项 L1-L2", m["L1"] - m["L2"], 0.182)
    chk("表VI", "试件项 L2-L3", m["L2"] - m["L3"], 0.158)


# ------------------------------------------------------- 表 III：CWRU 同录制泄漏
def table_leak():
    """表 III 由 tools/probe_leakage.py 产出 results_paper/tab_leak.json。"""
    print("\n表 III  tab:leak  —— results_paper/tab_leak.json")
    path = os.path.join(HERE, "results_paper", "tab_leak.json")
    if not os.path.exists(path):
        print("  [--      ] 尚无落盘产物:先跑 python tools/probe_leakage.py all")
        return
    d = json.load(open(path, encoding="utf-8"))["by_support_load"]
    # 表内值取自 6 次运行(3 seed x 2 遍)的合并产物,单元格四舍五入到 3 位。
    # 2026-10-06 修复 CWRU 99.mat 的读取(负载 2 的 NORMAL 类原先读了 1 hp 的录制)后整表重跑,
    # 下面的声称值随之更新 —— 变化最大的是 load2 行与各行的"测 load2"列。
    CLAIM = {  # 支撑域 -> (support_only, +fp_aug, +surrogate, +true_aug) 的 (same, cross, cross->3)
        "0": ((0.987, 0.868, 0.696), (0.985, 0.837, 0.709),
              (0.989, 0.826, 0.737), (1.000, 0.864, 0.695)),
        "1": ((0.988, 0.871, 0.684), (0.988, 0.837, 0.691),
              (0.992, 0.811, 0.661), (1.000, 0.894, 0.714)),
        "2": ((0.999, 0.871, 0.767), (0.946, 0.847, 0.733),
              (0.798, 0.684, 0.596), (1.000, 0.929, 0.881)),
        "3": ((0.998, 0.725, None), (0.995, 0.809, None),
              (0.997, 0.686, None), (1.000, 0.688, None)),
    }
    names = ("support_only", "+fp_aug", "+surrogate", "+true_aug")
    for L in ("0", "1", "2", "3"):
        if L not in d:
            chk("表III", f"支撑域 {L} (未跑)", np.nan, 0.0)
            continue
        for name, (same, cross, c3) in zip(names, CLAIM[L]):
            r = d[L]["rows"][name]
            chk("表III", f"load{L} {name:<12} same", r["same_mean"], same)
            chk("表III", f"load{L} {name:<12} cross", r["cross_mean"], cross)
            if c3 is not None:
                chk("表III", f"load{L} {name:<12} cross->3", r["cross_to"]["3"]["mean"], c3)
    # 均值行(tab:leak 底部)与 §Protocol 正文。
    # 注意:表 V(tab:pb_leak)里有一行 "\emph{CWRU, for comparison} … Support only"
    # 重复引用这里的 support_only 三个数(0.993 / 0.834 / 0.159)—— 改表 III 时必须同步改它,
    # 否则两处会各说各话(本脚本只核一处,因为另一处是同一来源)。
    for name, sm, cm, om in (("support_only", 0.993, 0.834, 0.159),
                             ("+fp_aug", 0.979, 0.832, 0.146),
                             ("+surrogate", 0.944, 0.752, 0.192),
                             ("+true_aug", 1.000, 0.844, 0.156)):
        rs = [d[L]["rows"][name] for L in ("0", "1", "2", "3") if L in d]
        if len(rs) < 4:
            continue
        chk("表III", f"均值行 {name:<12} same", float(np.mean([r["same_mean"] for r in rs])), sm)
        chk("表III", f"均值行 {name:<12} cross", float(np.mean([r["cross_mean"] for r in rs])), cm)
        chk("表III", f"均值行 {name:<12} 乐观量",
            float(np.mean([r["optimism"] for r in rs])), om)
    # §Protocol 里以区间形式给出的三个量(区间写法最容易与数据脱节,单独核)
    lab = [np.mean([d[L]["rows"][n]["optimism"] for L in "0123"]) for n in names]
    chk("表III", "带标签配置的均值乐观量 下限", min(lab), 0.146, tol=0.001)
    chk("表III", "带标签配置的均值乐观量 上限", max(lab), 0.192, tol=0.001)
    allop = [d[L]["rows"][n]["optimism"] for L in "0123" for n in names]
    chk("表III", "全部单元格最大乐观量", max(allop), 0.312, tol=0.001)
    cr = [d[L]["rows"]["support_only"]["cross_mean"] for L in "0123"]
    chk("表III", "support_only 跨录制 下限", min(cr), 0.725, tol=0.001)
    chk("表III", "support_only 跨录制 上限", max(cr), 0.871, tol=0.001)
    # §Protocol「指纹增广是正则化器」一段:只在支撑录制不典型时才占优
    for L in "012":
        chk("表III", f"load{L} 指纹增广劣于仅支撑集",
            d[L]["rows"]["support_only"]["cross_mean"] - d[L]["rows"]["+fp_aug"]["cross_mean"],
            0.031 if L == "0" else (0.034 if L == "1" else 0.024), tol=0.001)
    chk("表III", "load3 指纹增广优于仅支撑集",
        d["3"]["rows"]["+fp_aug"]["cross_mean"] - d["3"]["rows"]["support_only"]["cross_mean"],
        0.084, tol=0.001)
    # §Protocol「load 3 是异常工况」一段引用的无目标标签行
    if "source_only" in d["0"]["rows"]:
        src_same = {L: d[L]["rows"]["source_only"]["same_mean"] for L in ("0", "1", "2", "3")}
        chk("表III", "无标签源模型(load0--2 最高)", max(src_same[l] for l in "012"), 1.000)
        # 0.984 -> 0.933:load2 的 NORMAL 换成 2 hp 录制后,无标签源模型在 load2 上明显变差 ——
        # 原先那份 1 hp 录制恰好与源域更"像"。正文的 0.98--1.00 因此改为 0.93--1.00。
        chk("表III", "无标签源模型(load0--2 最低)", min(src_same[l] for l in "012"), 0.933)
        chk("表III", "无标签源模型测 load3", src_same["3"], 0.707)
        into3 = [d[L]["rows"]["support_only"]["cross_to"]["3"]["mean"] for L in ("0", "1", "2")]
        chk("表III", "支撑域(load0--2)测 load3 最低", min(into3), 0.684)
        chk("表III", "支撑域(load0--2)测 load3 最高", max(into3), 0.767)


# ------------------------------------------------------- 表 IX：CWRU 无源少样本
def table_fewshot():
    """表 IX 由 tools/probe_fewshot.py 产出 results_paper/tab_fewshot.json。"""
    print("\n表 IX  tab:fewshot  —— results_paper/tab_fewshot.json")
    path = os.path.join(HERE, "results_paper", "tab_fewshot.json")
    if not os.path.exists(path):
        print("  [--      ] 尚无落盘产物:先跑 python tools/probe_fewshot.py")
        return
    d = json.load(open(path, encoding="utf-8"))["by_k"]
    # 正文行名 -> {k: 声称均值};表内一角一格,是 6 次运行(3 seed x 2 遍)的均值。
    CLAIM = {"support_only": {1: 0.890, 3: 0.970, 5: 0.985},
             "dt_aug": {1: 0.558, 3: 0.777, 5: 0.885},
             "fp_rand_aug": {1: 0.977, 3: 0.995, 5: 0.993},
             "fp_phys_aug": {1: 0.947, 3: 0.983, 5: 0.658},
             "src_rand_aug": {1: 0.861, 3: 0.945, 5: 0.905},
             "surrogate": {1: 0.960, 3: 0.993, 5: 0.993},
             "roll_aug": {1: 0.952, 3: 0.991, 5: 0.994},
             "true_aug": {1: 1.000, 3: 1.000, 5: 1.000}}
    for name, per_k in CLAIM.items():
        for k, ref in per_k.items():
            if str(k) not in d or name not in d[str(k)]:
                chk("表IX", f"{name:<13} k={k} (缺)", np.nan, 0.0)
                continue
            chk("表IX", f"{name:<13} k={k}", d[str(k)][name]["mean"], ref, tol=0.001)
    if "5" in d and "fp_phys_aug" in d["5"]:
        r = d["5"]["fp_phys_aug"]
        chk("表IX", "dagger:k=5 fp+imp 标准差", r["std"], 0.46, tol=0.01)
        n_bad = sum(1 for v in r["per_run"] if v < 0.30)
        chk("表IX", "dagger:崩塌次数(6 次中)", n_bad, 2, tol=0.001)
    # 正文「源域指纹明显差于目标域指纹」一句引的两个 k
    if "src_rand_aug" in d["1"] and "fp_rand_aug" in d["1"]:
        chk("表IX", "正文 源指纹 k=1", d["1"]["src_rand_aug"]["mean"], 0.861, tol=0.001)
        chk("表IX", "正文 源指纹 k=5", d["5"]["src_rand_aug"]["mean"], 0.905, tol=0.001)
        chk("表IX", "正文 目标指纹 k=1", d["1"]["fp_rand_aug"]["mean"], 0.977, tol=0.001)
        chk("表IX", "正文 目标指纹 k=5", d["5"]["fp_rand_aug"]["mean"], 0.993, tol=0.001)
        chk("表IX", "正文 只用支撑集 k=1", d["1"]["support_only"]["mean"], 0.890, tol=0.001)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    runs = {"bench": table_bench, "classsep": table_classsep, "diag2": diag2,
            "rig": rig_tone, "pb_fewshot": table_pb_fewshot, "main": table_main,
            "pb_levels": table_pb_levels, "leak": table_leak, "fewshot": table_fewshot,
            "synth": table_synth}
    # 两组划分的依据是**数据依赖**,不是主题:artifacts 组只读本仓已落盘的 JSON/CSV,
    # 没有原始数据也能跑 —— 补充材料包就是靠这一点做到"审稿人开箱可复核"。
    groups = {"all": tuple(runs),
              "artifacts": ("synth", "leak", "fewshot", "main", "pb_levels", "pb_fewshot"),
              "raw": ("bench", "classsep", "diag2", "rig")}
    todo = groups.get(which, tuple(which.split(",")))
    for k in todo:
        if k not in runs:
            raise SystemExit(f"未知检查项 {k!r};可选:{', '.join(runs)} / {', '.join(groups)}")
    for k in todo:
        runs[k]()
    print("\n" + "=" * 78)
    print(f"本次共核对 {N_CHECKS} 项。")
    if FAILS:
        print(f"**{len(FAILS)} 项与正文不符,需要改正文**")
        for sec, n, got, ref in FAILS:
            print(f"   [{sec}] {n}: 算得 {got}, 正文 {ref}")
        sys.exit(1)
    print("全部通过:本仓所有有落盘产物、可自动核对的数字都已复现。")
    if which in ("all", "artifacts"):
        print("说明:artifacts 组只覆盖**有落盘产物**的表(I/III/V/VI/VII/VIII/IX);"
              "表 II/IV、§II 与 §PB 台架线谱是纯数据统计量,需要 data/ 下的原始数据,"
              "用 `python tools/verify_paper_numbers.py raw`。")
    for tag, path, cmd in (("表 I", "results_paper/tab_synth.json",
                            "tools/probe_fp.py 3 5 400 40 10 2"),
                           ("表 III", "results_paper/tab_leak.json",
                            "tools/probe_leakage.py all"),
                           ("表 IX", "results_paper/tab_fewshot.json",
                            "tools/probe_fewshot.py")):
        if not os.path.exists(os.path.join(HERE, path)):
            print(f"  [缺] {tag} 无落盘产物,先跑 python {cmd}")


if __name__ == "__main__":
    main()
