"""Paderborn(KAt)轴承数据集读取管线 —— 作为"类间差异由物理主导"的对照基准。

**为什么需要它**
CWRU 每个"类别×负载"只有**一次录制**(一颗轴承),所以"类间差异是录制指纹"这个结论
在 CWRU 上**无法证伪**:你没法把"同一类的不同轴承"放在一起比。Paderborn 可以 ——
每类包含**多颗不同的轴承**(6 颗健康轴承、12 颗人工损伤、14 颗真实损伤),
于是可以问两个 CWRU 问不了的问题:

1. **跨轴承的类内一致性**:同一类里不同轴承的谱像不像?(像 → 类信息是物理的)
2. **轴承不相交评估**:在 A 轴承上训练,在 B 轴承上测试,还准不准?

**数据组织**
  data/paderborn/<轴承码>/<轴承码>/<N?_M?_F?>_<轴承码>_<序号>.mat
  文件名前缀即工况:N09/N15 = 转速档(900/1500 rpm),M01/M07 = 负载扭矩,
  F04/F10 = 径向力。每个轴承 4 个工况 × 20 次测量 = 80 个 .mat。
  振动信号在 `Y` 里名为 `vibration_1` 的通道,栅格 HostService,64 kHz,256001 点(4 s)。

**两套类别定义**

1. `CLASSES_PB = ("HEALTHY","ARTIF","REAL")` —— 只看轴承码前缀,零转述风险:
     K00x -> HEALTHY   KAxx -> ARTIF(人工制造损伤)   KIxx -> REAL(加速寿命试验损伤)
2. `CLASSES_LOC = ("HEALTHY","IR","OR")` —— 按**损伤位置**(内圈/外圈)分,
   由 `damage_table()` 从每颗轴承自带的事实表 PDF 里解析。这一套才是"物理主导"的类别,
   因为内外圈损伤的**运动学**(BPFO/BPFI)正是物理孪生能建模的东西。

⚠️ 已知混淆:`damage_table()` 只在每颗轴承的**第一个**损伤上取位置。本仓库首批下载的
18 颗里 KA 全为 OR、KI 全为 IR,"位置"与"成因(人工/真实)"完全重合 —— 所以要
另下 KA08 等人工内圈轴承来解耦。用 `component` 分类时必须在正文里声明这一点。

用法:
    from pb_utils import build_pb_domain, list_bearings, OP_CONDITIONS, CLASSES_LOC
    X, y, meta = build_pb_domain("data/paderborn", "N15_M07_F10", CLASSES_LOC)
"""
import glob
import json
import os
import re
import subprocess

import numpy as np
import scipy.io as sio
from scipy.signal import sosfilt

SR = 64000                     # 振动通道采样率 (Hz)
WINDOW_DEFAULT = 1024
OP_CONDITIONS = ("N09_M07_F10", "N15_M01_F10", "N15_M07_F04", "N15_M07_F10")
SHAFT_HZ = {"N09": 900 / 60.0, "N15": 1500 / 60.0}   # 转速档 -> 转频

CLASSES_PB = ("HEALTHY", "ARTIF", "REAL")
CLASSES_LOC = ("HEALTHY", "IR", "OR")
_FNAME_RE = re.compile(r"^(N\d+)_(M\d+)_(F\d+)_([A-Z]+\d+)_(\d+)\.mat$", re.IGNORECASE)


def class_of(bearing):
    """轴承码 -> 类别名。K001->HEALTHY, KA04->ARTIF, KI01->REAL。"""
    m = re.match(r"^([A-Z]+)\d+$", bearing.upper())
    if m is None:
        raise ValueError(f"无法解析轴承码: {bearing}")
    return {"K": "HEALTHY", "KA": "ARTIF", "KI": "REAL"}.get(m.group(1), "OTHER")


def _pdf_field(txt, label):
    """在 pdftotext -table 输出里取某标签后的第一个值(标签可能在行中间)。"""
    for ln in txt.splitlines():
        m = re.search(r"(?:^|\s)" + re.escape(label) + r"(?=\s|$)", ln)
        if not m:
            continue
        parts = [p.strip() for p in re.split(r"\s{2,}", ln[m.end():]) if p.strip()]
        if parts:
            return parts[0]
    return None


def damage_table(root, use_cache=True):
    """解析每颗轴承自带的事实表 PDF,得到 {轴承码: {...}}。

    字段:`component`(OR/IR,只取第一处损伤)、`mode`(fatigue/artificial)、
    `method`(lifetime test / EDM machining / drilled / electric engraver)、`extent`。
    结果缓存到 <root>/damage_table.json。缺 pdftotext 时返回空 dict(调用方自行降级)。
    """
    cache = os.path.join(root, "damage_table.json")
    if use_cache and os.path.exists(cache):
        return json.load(open(cache))
    table = {}
    for code in list_bearings(root):
        pdfs = glob.glob(os.path.join(root, code, code, f"{code}.pdf"))
        if not pdfs:
            continue
        try:
            txt = subprocess.run(
                ["pdftotext", "-table", "-f", "2", "-l", "2", pdfs[0], "-"],
                capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            return {}
        table[code] = {
            "component": _pdf_field(txt, "Component"),
            "mode": _pdf_field(txt, "Mode"),
            "method": _pdf_field(txt, "Damage method"),
            "extent": _pdf_field(txt, "Extent of damage"),
        }
    if table:
        json.dump(table, open(cache, "w"), indent=1, ensure_ascii=False)
    return table


def class_of_location(bearing, table):
    """轴承码 + 事实表 -> 位置类别(HEALTHY / IR / OR);无信息时返回 None。"""
    if class_of(bearing) == "HEALTHY":
        return "HEALTHY"
    comp = (table.get(bearing.upper()) or {}).get("component")
    return comp if comp in ("IR", "OR") else None


def list_bearings(root):
    """列出已解压的轴承码(要求存在 <root>/<code>/<code>/ 目录)。"""
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        code = os.path.basename(d)
        if os.path.isdir(os.path.join(d, code)):
            out.append(code.upper())
    return out


def list_files(root, bearing, cond):
    """列出某轴承某工况下的 .mat 文件,按序号排序。"""
    d = os.path.join(root, bearing.upper(), bearing.upper())
    fs = []
    for p in glob.glob(os.path.join(d, f"*{bearing.upper()}_*.mat")):
        m = _FNAME_RE.match(os.path.basename(p))
        if m and f"{m.group(1)}_{m.group(2)}_{m.group(3)}".upper() == cond.upper():
            fs.append((int(m.group(5)), p))
    return [p for _, p in sorted(fs)]


def load_vibration(path):
    """读取 64 kHz 振动通道。"""
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    s = m[[k for k in m if not k.startswith("__")][0]]
    for e in np.atleast_1d(s.Y):
        nm = np.array(getattr(e, "Name", [])).ravel()
        rs = np.array(getattr(e, "Raster", [])).ravel()
        if len(nm) and str(nm[0]) == "vibration_1":
            return np.asarray(getattr(e, "Data"), dtype=np.float64)
        if len(rs) and str(rs[0]) == "HostService" and getattr(e, "Type", None) == 4:
            d = np.asarray(getattr(e, "Data"), dtype=np.float64)
            if len(d) > 100000:      # 兜底:64 kHz 通道就是它
                return d
    raise KeyError(f"未在 {path} 中找到 vibration_1")


def _windows(x, window, stride):
    n = (len(x) - window) // stride + 1
    if n <= 0:
        return np.zeros((0, window), dtype=np.float64)
    idx = np.arange(n)[:, None] * stride + np.arange(window)[None, :]
    return x[idx]


def build_pb_domain(root, cond, classes=CLASSES_PB, window=WINDOW_DEFAULT, stride=512,
                    bearings=None, max_files=None, zscore=True, label_fn=None, sos=None):
    """拼装一个工况下的数据集。

    返回 (X, y, meta):
      X      : (N, window) float32
      y      : (N,) int64,类别 id(按 classes 顺序)
      meta   : dict,含 'bearing'(轴承码数组)与 'file'(文件名数组)—— 供轴承不相交切分使用

    label_fn(bearing) -> 类别名(None 表示丢弃该轴承)。默认按轴承码前缀分类;
    要按损伤位置分类,传 `lambda b: class_of_location(b, damage_table(root))`。
    sos: 可选的 IIR SOS 滤波器(scipy.signal.butter(..., output='sos')),
    在切窗前作用到整条记录上。用途:本数据集振动被台架线谱(1500 rpm 下 100 Hz)
    占去 90% 以上能量,做谱比较前需先高通把它去掉。
    """
    if label_fn is None:
        label_fn = class_of
    codes = bearings if bearings is not None else list_bearings(root)
    X, y, mb, mf = [], [], [], []
    for code in codes:
        cls = label_fn(code)
        if cls not in classes:
            continue
        fs = list_files(root, code, cond)
        if max_files:
            fs = fs[:max_files]
        if not fs:
            continue
        for p in fs:
            x = load_vibration(p)
            if sos is not None:
                x = sosfilt(sos, x)
            w = _windows(x, window, stride)
            if len(w) == 0:
                continue
            if zscore:
                w = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-8)
            X.append(w.astype(np.float32))
            y.append(np.full(len(w), classes.index(cls), dtype=np.int64))
            mb.append(np.full(len(w), code, dtype=object))
            mf.append(np.full(len(w), os.path.basename(p), dtype=object))
    if not X:
        raise RuntimeError(f"工况 {cond} 下没有可用数据(检查 {root} 是否已解压)")
    X = np.concatenate(X); y = np.concatenate(y)
    meta = {"bearing": np.concatenate(mb), "file": np.concatenate(mf)}
    return X, y, meta


def char_freqs_pb(fr, n_balls=8, ball_d=6.75e-3, pitch_d=28.5e-3, contact=0.0):
    """6203 轴承的特征频率(供孪生使用)。"""
    d, D, Z = ball_d, pitch_d, n_balls
    c = np.cos(contact)
    return (Z / 2 * fr * (1 - d / D * c),          # BPFO
            Z / 2 * fr * (1 + d / D * c),          # BPFI
            D / (2 * d) * fr * (1 - (d / D * c) ** 2),   # BSF
            fr / 2 * (1 - d / D * c))              # FTF


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "data", "paderborn")
    codes = list_bearings(root)
    print("已解压轴承:", codes)
    for c in OP_CONDITIONS:
        print(f"  {c}: fr={SHAFT_HZ[c[:3]]:.2f}Hz  BPFO/BPFI={char_freqs_pb(SHAFT_HZ[c[:3]])[:2]}")
