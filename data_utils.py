"""数据加载与划分工具:当前实现 CWRU 12k 驱动端数据。

提供:
- CWRU .mat 文件解析(文件名正则,保持官方命名)
- 切窗 / z-score 归一化
- 目标域 few-shot 划分:支撑集(带标签) / 无标签池(训练用) / 测试集(hold-out,按时间序切分)
- 缓存为 .npy,避免重复预处理

注意:切窗采用 step = window//2,训练与测试按时间序切分(前段做训练/适配池,后段做测试),
避免相邻窗口跨划分带来的泄漏。
"""
import os
import re
import hashlib
import numpy as np
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# CWRU 文件解析
# ---------------------------------------------------------------------------

# 文件名如: Normal_0.mat / IR007_0.mat / B021_3.mat / OR014_1.mat
_FNAME_RE = re.compile(r"^(Normal|IR|OR|B)(\d*)_(\d)\.mat$", re.IGNORECASE)

# 类别名如: NORMAL / B / IR / OR(旧 4 类,尺寸由 size 参数统一给出)
#           NORMAL / B007 / B014 / B021 / IR007 / ... (10 类,尺寸写在类名里)
_CLS_RE = re.compile(r"^(NORMAL|B|IR|OR)(\d*)$", re.IGNORECASE)

CLASSES_4 = ("NORMAL", "B", "IR", "OR")
CLASSES_10 = ("NORMAL",
              "B007", "B014", "B021",
              "IR007", "IR014", "IR021",
              "OR007", "OR014", "OR021")
CLASS_SETS = {4: CLASSES_4, 10: CLASSES_10}


def parse_class(name, default_size="007"):
    """把类别名解析成 (类型, 尺寸)。

    'NORMAL' -> ('NORMAL', '')
    'B014'   -> ('B', '014')
    'B'      -> ('B', default_size)      # 旧 4 类写法,尺寸由调用方给定
    """
    m = _CLS_RE.match(name)
    if m is None:
        raise ValueError(f"无法解析类别名: {name}")
    typ = m.group(1).upper()
    if typ == "NORMAL":
        return typ, ""
    return typ, (m.group(2) or default_size)


def list_cwru_files(root):
    """递归扫描目录,返回 [(路径, 类型, 尺寸, 负载)]。"""
    files = []
    for dirpath, _, fnames in os.walk(root):
        for f in sorted(fnames):
            m = _FNAME_RE.match(f)
            if m:
                typ, size, load = m.group(1).upper(), m.group(2), int(m.group(3))
                files.append((os.path.join(dirpath, f), typ, size, load))
    return files


def load_de_channel(path):
    """读取 CWRU .mat 中的驱动端(DE)振动通道,返回一维 float64 数组。

    注意:loadmat 会附带 '__header__' 等元数据键(含 'de' 子串),必须跳过。

    ⚠️ 一个文件里出现**多路** DE 通道时必须报错,不能"取第一个"。
    实例:CWRU 官方的 2 hp 基线和 `99.mat` 除了 X099 还**附带了 1 hp 的 X098**
    (两个变量在同一文件里,X098 排在前面)。早期的实现在这里取第一个含 DE 的键,
    于是本仓"负载 2"的 NORMAL 类一直加载的是**负载 1 的录制**(逐点相同),
    直到 2026-10-06 才发现。故改为:多路即报错,由调用方明确指定。
    """
    data = loadmat(path)
    for cand in ("DE_time", "de_time", "De_time", "DE", "de", "De"):
        if cand in data:
            return np.asarray(data[cand]).ravel()
    keys = [k for k in data
            if not k.startswith("__") and re.search(r"DE", k, re.IGNORECASE)]
    if not keys:
        raise ValueError(f"在 {path} 中找不到 DE 通道")
    if len(keys) > 1:
        raise ValueError(
            f"{path} 里有 {len(keys)} 路 DE 通道 {keys} —— 无法判断该用哪一路。"
            f"这通常意味着该文件打包了两段录制(CWRU 的 99.mat 就是如此:"
            f"X098 = 1 hp、X099 = 2 hp 在同一文件里),必须先把想要的那一段"
            f"单独存成只含该通道的文件,再让加载器读 —— 取第一个键会静默读错录制。")
    return np.asarray(data[keys[0]]).ravel()


# ---------------------------------------------------------------------------
# 切窗与归一化
# ---------------------------------------------------------------------------


def segment(sig, window, step):
    """按固定窗长/步长切窗,返回 (n_windows, window) 数组。"""
    n = len(sig)
    if n < window:
        return np.zeros((0, window))
    starts = np.arange(0, n - window + 1, step)
    return np.stack([sig[i : i + window] for i in starts])


def zscore_per_window(X, eps=1e-6):
    """逐窗口 z-score 归一化(每个窗口减去自身均值、除以自身标准差)。"""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)


# ---------------------------------------------------------------------------
# 领域构建
# ---------------------------------------------------------------------------


def _dir_fingerprint(root):
    """数据目录指纹:文件名 + 大小 + mtime 的哈希。

    旧实现只哈希目录**路径**,于是"改了数据但路径没变"时旧缓存会被静默命中 ——
    2026-10-06 重写 `data/cwru/Normal_2.mat`(把误装的 1 hp 通道换成 2 hp)正踩中这一点:
    若不换键,修了数据也还是会跑旧数据。故键里必须含内容信息。
    """
    h = hashlib.md5()
    for p, _typ, _size, _load in list_cwru_files(root):
        st = os.stat(p)
        h.update(f"{os.path.basename(p)}|{st.st_size}|{int(st.st_mtime)}|".encode())
    return h.hexdigest()[:8]


def build_cwru_domain(root, load, classes=CLASSES_4,
                      size="007", window=1024, step=None, cache_dir=None):
    """构建 CWRU 某个负载下的完整数据集。

    classes 可以写旧 4 类 ("NORMAL","B","IR","OR",尺寸统一由 size 给出),
    也可以写 10 类 ("NORMAL","B007","B014",...,"OR021",尺寸写在类名里)。
    返回 (X, y):X 为 (n, window) float32 已归一化;y 为类别 id,顺序与 classes 一致。
    若 cache_dir 给定则缓存 .npy,缓存键绑定 (load, 类别集合, 数据目录指纹)。
    """
    if step is None:
        step = window // 2
    class_id = {c: i for i, c in enumerate(classes)}

    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        # 缓存键绑定数据目录指纹 + 类别集合,避免不同数据源(真数据/假数据)、
        # 不同类别切分(4 类/10 类)以及**数据内容本身的改动**互相污染
        root_fp = _dir_fingerprint(root)
        cls_fp = hashlib.md5((",".join(classes) + "|" + size).encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"cwru_l{load}_{cls_fp}_{root_fp}.npz")
        if os.path.exists(cache_path):
            d = np.load(cache_path)
            return d["X"], d["y"]

    files = list_cwru_files(root)
    X_list, y_list = [], []
    for cls in classes:
        typ, sz = parse_class(cls, size)
        if typ == "NORMAL":
            cls_files = [p for p, t, s, l in files if t == "NORMAL" and l == load]
        else:
            cls_files = [p for p, t, s, l in files
                         if t == typ and s == sz and l == load]
        if not cls_files:
            raise FileNotFoundError(
                f"找不到类别 {cls}(类型 {typ}、尺寸 {sz})、负载 {load} 的 CWRU 文件,"
                f"请检查 data_root")
        for path in cls_files:
            sig = load_de_channel(path)
            win = segment(sig, window, step)
            X_list.append(win)
            y_list.append(np.full(len(win), class_id[cls]))

    X = np.concatenate(X_list, axis=0).astype("float32")
    y = np.concatenate(y_list, axis=0).astype("int64")
    X = zscore_per_window(X)

    if cache_path is not None:
        np.savez(cache_path, X=X, y=y)
    return X, y


# ---------------------------------------------------------------------------
# 目标域 few-shot 划分
# ---------------------------------------------------------------------------


def split_target_fewshot(X, y, k, seed, test_frac=0.4):
    """把目标域数据划分为 支撑集 / 无标签池 / 测试集。

    规则:
    - 每个类别按时间序先取前 (1-test_frac) 部分作为"适配区",后 test_frac 部分为测试集。
    - 从适配区里每类随机抽 k 个作为支撑集(带标签),其余为无标签池(训练时标签隐藏)。
    - 支撑集抽取依赖 seed;不同 seed 得到不同划分,便于 5 次重复取均值。

    返回 (support_idx, pool_idx, test_idx),均为全局下标数组。
    """
    rng = np.random.RandomState(seed)
    support, pool, test = [], [], []
    n_class = int(y.max()) + 1
    for c in range(n_class):
        idx = np.where(y == c)[0]
        n_test = int(round(len(idx) * test_frac))
        test_idx = idx[len(idx) - n_test:]
        adapt_idx = idx[: len(idx) - n_test]

        perm = rng.permutation(len(adapt_idx))
        sup = adapt_idx[perm[:k]]
        pl = adapt_idx[perm[k:]]
        support.append(sup)
        pool.append(pl)
        test.append(test_idx)
    return (np.concatenate(support), np.concatenate(pool), np.concatenate(test))


def split_source_trainval(X, y, val_frac=0.15):
    """源域按时间序切分训练/验证(供监控,验证不参与测试)。"""
    n = len(y)
    n_val = int(round(n * val_frac))
    return X[: n - n_val], y[: n - n_val], X[n - n_val:], y[n - n_val:]
