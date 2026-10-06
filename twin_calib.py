"""目标域指纹标定的孪生增广(论文一方法侧的新尝试)。

**动机(来自实测诊断)**
物理孪生知道"故障冲击该按什么频率出现"——BPFO/BPFI/BSF 会随转速按比例平移,
这是可推广的物理规律。但孪生**不知道目标设备那颗轴承的传递特性**:实测 CWRU 各类的
谱形在负载间高度稳定(r≈0.96),说明类间差异来自"那次录制/那颗轴承的固有指纹",
物理模型无从预测。这正是原主方法 `dt_dann` 失败的根因:合成域的类间谱形几乎全同
(九类在 2800-3300Hz 一律约 53%),拿它训练只在真实域得到随机水平(macro_f1 0.10)。

**本模块的做法**
用目标域**支撑集**(k 个/类,带标签,是 few-shot 设定下合法可用的信息)估计每一类在
目标域的谱包络 —— 即"指纹";再用它去着色物理冲击串,生成**目标域风格**的带标签样本。
于是:类别结构由物理提供,域风格由支撑集提供,两者各司其职。

判别依据依然是留出的目标域测试集,支撑集只用于估计指纹(与已有方法用支撑集算 CE 损失
用的是同一份信息,没有额外泄漏)。
"""
import numpy as np

from dt_synth import _fault_freq, parse_cls


def estimate_fingerprints(X_sup, y_sup, n_class, window=1024, smooth=5, envelope=False):
    """从目标域支撑集估计每类的谱包络(指纹)。

    返回 A: (n_class, window//2+1) 的幅度包络,每行归一化到单位 L2 范数。
    样本很少(k 个)时用滑动平均做频域平滑,避免指纹全是毛刺。

    envelope=True 时再做一次形态学开运算(先腐蚀后膨胀),把**窄的故障谐波梳齿**削掉,
    只留传递函数的宽包络。保留这个开关是为了记录一个重要的否定结果:实测把它削掉后
    合成域迁移反而更差(0.486 vs 0.875),说明**细的梳齿结构本身就携带类别信息**,
    不能当噪声处理。物理结构帮不上忙的原因不是"信息被破坏",而是分布被推离真实域 ——
    见 `tools/probe_classsep.py` 的读法说明。
    """
    n_freq = window // 2 + 1
    A = np.zeros((n_class, n_freq), dtype=np.float64)
    win = np.hanning(window)
    for c in range(n_class):
        Xc = X_sup[y_sup == c]
        if len(Xc) == 0:
            A[c] = 1.0                       # 该类没有支撑样本 -> 用平坦指纹(退化为白噪声着色)
            continue
        S = np.abs(np.fft.rfft(Xc * win, axis=-1)).mean(axis=0)
        if smooth > 1:
            k = np.ones(smooth) / smooth
            S = np.convolve(S, k, mode="same")
        if envelope:
            S = _morph_open(S, 9)
        A[c] = S
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return A


def _morph_open(S, w):
    """一维形态学开运算(腐蚀 -> 膨胀),用于削掉窄谱峰、保留宽包络。"""
    n = len(S)
    pad = np.pad(S, w // 2, mode="edge")
    eroded = np.array([pad[i:i + w].min() for i in range(n)])
    pad = np.pad(eroded, w // 2, mode="edge")
    return np.array([pad[i:i + w].max() for i in range(n)])


def _excitation(typ, window, sr, fr, rng, slip=0.015):
    """激励源:故障类是特征频率处的冲击串;正常类是白噪声(无冲击)。"""
    if typ == "NORMAL":
        return rng.randn(window)
    period = 1.0 / _fault_freq(typ, fr)
    imp = np.zeros(window)
    # 冲击时刻:按目标转速下的特征频率周期布置 + 随机滑移
    tt = rng.uniform(0, period)
    while tt < window / sr:
        i0 = int(round(tt * sr))
        if 0 <= i0 < window:
            imp[i0] = 1.0
        tt += period + rng.normal(0, slip * period)
    return imp


def synth_target_like(classes, n_per_class, window, sr, fr_tgt, A, seed=0, physics=True):
    """用目标域指纹着色激励源,生成目标域风格的合成样本。

    关键物理点:冲击串按 **目标域转速** fr_tgt 布置,因此 BPFO/BPFI/BSF 会随转速
    正确平移 —— 这是孪生真正可推广的那部分知识;而谱的"形状"来自目标域指纹。

    physics=False 时退化为**无物理对照**:激励源一律换成白噪声(无冲击串、无特征频率),
    只保留"用目标域指纹着色"这一步。两者的差值即物理结构的净贡献。

    返回 (X, y):X 为 (n_class*n_per_class, window) float32;y 为类别 id。
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    X, y = [], []
    for cid, cls in enumerate(classes):
        typ, _ = parse_cls(cls)
        a = A[cid]
        for _ in range(n_per_class):
            exc = _excitation(typ, window, sr, fr_tgt, rng) if physics \
                else rng.randn(window)
            spec = np.fft.rfft(exc) * a
            w = np.fft.irfft(spec, window)
            X.append((w - w.mean()) / (w.std() + 1e-6))
            y.append(cid)
    np.random.set_state(rng_state)
    return np.stack(X).astype("float32"), np.array(y, dtype="int64")


def fingerprint_transfer_cost(A_true, A_est):
    """诊断用:估计指纹与真实指纹的余弦相似度(逐类)。"""
    An = A_true / (np.linalg.norm(A_true, axis=1, keepdims=True) + 1e-12)
    Ae = A_est / (np.linalg.norm(A_est, axis=1, keepdims=True) + 1e-12)
    return (An * Ae).sum(axis=1)
