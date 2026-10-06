"""物理仿真数字孪生信号合成器(论文一模块 1)。

基于滚动轴承局部故障的经典冲击响应建模思想:
- McFadden & Smith 点缺陷模型:故障产生的冲击按特征频率周期出现,且伴随随机滑移
  (见 Randall & Antoni, MSSP 2011 教程对"伪循环平稳"的讨论)。
- 每个冲击激励起系统共振:与"单自由度谐振系统的指数衰减冲击响应"卷积。
- 叠加转频谐波(确定性成分)与高斯噪声,调节信噪比模拟不同安装/载荷条件。

几何参数取 CWRU 使用的 SKF 6205-2RS 深沟球轴承。支持两套类别体系:
- 4 类:正常 / 滚动体(B) / 内圈(IR) / 外圈(OR)
- 10 类:在此基础上按故障尺寸细分(B007/B014/B021、IR007/…、OR007/…),
  尺寸通过谐振衰减常数与冲击强度调制。

用法:
    from dt_synth import synth_dataset
    X, y = synth_dataset(CLASSES_10, n_per_class=400, window=1024, sr=12000, fr=29.95)
"""
import re

import numpy as np

# SKF 6205-2RS 轴承几何参数
PITCH_D = 39.04e-3   # 节圆直径 (m)
BALL_D = 7.94e-3     # 滚动体直径 (m)
N_BALLS = 9          # 滚动体数量
CONTACT_ANGLE = 0.0  # 接触角 (rad)

# CWRU 各负载下的转频 (Hz)
CWRU_FR = {0: 1797 / 60, 1: 1772 / 60, 2: 1750 / 60, 3: 1730 / 60}


def char_freqs(fr):
    """给定转频 fr(Hz),计算 4 个故障特征频率(单位 Hz)。"""
    d, D, Z = BALL_D, PITCH_D, N_BALLS
    bpfo = Z / 2 * fr * (1 - d / D * np.cos(CONTACT_ANGLE))          # 外圈
    bpfi = Z / 2 * fr * (1 + d / D * np.cos(CONTACT_ANGLE))          # 内圈
    bsf = D / (2 * d) * fr * (1 - (d / D * np.cos(CONTACT_ANGLE)) ** 2)  # 滚动体
    ftf = fr / 2 * (1 - d / D * np.cos(CONTACT_ANGLE))               # 保持架
    return bpfo, bpfi, bsf, ftf


def _fault_freq(typ, fr):
    bpfo, bpfi, bsf, _ = char_freqs(fr)
    return {"B": bsf, "IR": bpfi, "OR": bpfo}[typ]


# 故障尺寸 -> 冲击相对强度(缺陷越大,冲击能量越强)。
# 尺寸的主要判别线索是下面的 _RESONANCE_BW(振铃时长),幅值只是辅助。
_SIZE_AMP = {"007": 1.0, "014": 1.5, "021": 2.0}

# ---------------------------------------------------------------------------
# 以下三组参数由**真实 CWRU 源域数据**标定,不要凭手感改。
# 标定脚本:tools/calibrate_spectrum.py(用正常类谱作背景模型、故障类谱作共振带模型)
# 复现判据:tools/compare_stats.py 里"各频段能量占比"应与真实列接近。
# ---------------------------------------------------------------------------

# 故障冲击激起的共振频率范围 (Hz)。实测真实故障类能量峰值在 3586 Hz,
# 半高宽 2754-3609 Hz;随机化范围覆盖这一带。
_RESONANCE_RANGE = (2850.0, 3550.0)

# 共振 -3dB 带宽 (Hz) —— 缺陷越大振铃越长、带宽越窄。衰减常数 = pi * BW。
_RESONANCE_BW = {"007": 200.0, "014": 120.0, "021": 80.0}

# 背景噪声着色曲线(25 段等分 0-6000Hz,归一化到最大 1)。
# 实测真实"正常类"信号 65% 的能量在 800-1500Hz,2500Hz 以上几乎为零;
# 用白噪声会凭空造出大量高频能量,这是孪生域被域判别器一眼分开的主因之一。
_NOISE_SHAPE = (0.555, 0.306, 0.253, 0.245, 1.000, 0.129, 0.078, 0.049, 0.280,
                0.026, 0.027, 0.021, 0.024, 0.022, 0.019, 0.010, 0.010, 0.012,
                0.014, 0.014, 0.030, 0.010, 0.006, 0.001, 0.000)


# 轴频谐波功率 / 背景噪声功率。实测真实"正常"类信号 0-200Hz 约占 18% 能量,
# 0.22/(1+0.22) ≈ 18%;而故障类里同样的谐波会被共振淹没到 <0.1%(实测如此)。
_HARMONIC_POWER_RATIO = 0.14

# 故障冲击功率 / 背景噪声功率(随机化模拟不同安装与载荷)。
# 实测真实故障类 3000-4500Hz 共振带占 73%、2000-3000Hz 占 25%,
# 背景(800-2000Hz)仅 0.4% —— 对应冲击约为背景的 60~100 倍。原实现等效只有约 1 倍。
_FAULT_POWER_RATIO = (40.0, 100.0)


def _colored_noise(n, sr, shape=_NOISE_SHAPE):
    """生成与真实背景噪声同谱形的着色噪声(单位方差)。"""
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    edges = np.linspace(0, sr / 2, len(shape) + 1)
    gain = np.zeros_like(freqs)
    for i, g in enumerate(shape):
        gain[(freqs >= edges[i]) & (freqs < edges[i + 1])] = g
    w = np.fft.rfft(np.random.randn(n))
    out = np.fft.irfft(w * gain, n)
    return out / (out.std() + 1e-12)

def parse_cls(cls):
    """解析类别名。

    'NORMAL' -> ('NORMAL', '')      'B014' -> ('B', '014')      'B' -> ('B', '007')
    支持旧的 4 类写法(不写尺寸时按 007 处理)。
    """
    m = re.match(r"^(NORMAL|B|IR|OR)(\d*)$", cls, re.IGNORECASE)
    if m is None:
        raise ValueError(f"无法解析类别名: {cls}")
    typ = m.group(1).upper()
    if typ == "NORMAL":
        return typ, ""
    return typ, (m.group(2) or "007")


def synth_signal(window, sr, fr, cls, fn=None, snr_db=None, slip=0.015):
    """生成一条长度为 window 的合成振动信号(已 z-score 归一化)。

    参数:
        window : 样本点数
        sr     : 采样率 (Hz)
        fr     : 转频 (Hz)
        cls    : 类别, 'NORMAL' | 'B007' | 'IR014' | 'OR021' 等(也接受旧的 'B'/'IR'/'OR')
        fn     : 系统共振频率 (Hz), 默认 [2000, 4000] 内随机
        snr_db : 信噪比 (dB), 默认 [-2, 6] 内随机
        slip   : 冲击滑移量(占冲击间隔的比例), 默认 1.5%
    """
    typ, size = parse_cls(cls)
    t = np.arange(window) / sr

    # 1) 背景噪声:与真实背景同谱形的着色噪声(单位方差)。一切功率以它为基准。
    noise = _colored_noise(window, sr)

    # 2) 轴频谐波:功率相对背景固定。真实"正常"类 0-200Hz 约占 18% 能量,
    #    而故障类里它几乎被共振淹没(<0.1%)—— 这个反差由功率比自然得到,
    #    不需要对正常/故障分别处理。
    h = np.zeros(window)
    for m in (1, 2, 3):
        h += np.random.uniform(0.3, 1.0) * np.sin(2 * np.pi * m * fr * t
                                                   + np.random.uniform(0, 2 * np.pi))
    h = h / (h.std() + 1e-12) * np.sqrt(_HARMONIC_POWER_RATIO)

    y = h + noise

    # 3) 故障冲击(正常类不添加)
    if typ != "NORMAL":
        f_fault = _fault_freq(typ, fr)
        period = 1.0 / f_fault

        # 冲击时刻:按特征频率周期布置 + 随机滑移(高斯)
        times = []
        tt = np.random.uniform(0, period)
        while tt < window / sr:
            times.append(tt)
            tt += period + np.random.normal(0, slip * period)

        # 系统冲击响应:单自由度谐振的指数衰减振荡
        if fn is None:
            fn = np.random.uniform(*_RESONANCE_RANGE)
        decay = np.pi * _RESONANCE_BW.get(size, 200.0)
        ring = np.exp(-decay * t) * np.sin(2 * np.pi * fn * t)

        imp = np.zeros(window)
        for ti in times:
            i0 = int(round(ti * sr))
            if 0 <= i0 < window:
                imp[i0] = 1.0
        res = np.convolve(imp, ring, "same")

        # 冲击功率相对背景的比值:随机化模拟不同安装/载荷,再按缺陷尺寸放大
        ratio = 10 ** (snr_db / 10.0) if snr_db is not None \
            else np.random.uniform(*_FAULT_POWER_RATIO)
        ratio *= _SIZE_AMP.get(size, 1.0)
        res = res / (res.std() + 1e-12) * np.sqrt(ratio)

        y = y + res

    return (y - y.mean()) / (y.std() + 1e-6)


def synth_dataset(classes, n_per_class, window, sr, fr, seed=0):
    """批量生成孪生合成数据集。

    返回 (X, y):X 为 (n_class * n_per_class, window) float32;y 为类别 id。
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    X, y = [], []
    for cid, cls in enumerate(classes):
        for _ in range(n_per_class):
            X.append(synth_signal(window, sr, fr, cls))
            y.append(cid)
    np.random.set_state(rng_state)
    return np.stack(X).astype("float32"), np.array(y, dtype="int64")


if __name__ == "__main__":
    # 自检:打印各类别特征频率,并把一条外圈故障信号存为 npy 供后续谱分析
    for load in (0, 3):
        fr = CWRU_FR[load]
        bpfo, bpfi, bsf, ftf = char_freqs(fr)
        print(f"load {load}: fr={fr:.2f}Hz  BPFO={bpfo:.2f} BPFI={bpfi:.2f} BSF={bsf:.2f} FTF={ftf:.2f}")
