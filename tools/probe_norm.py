"""探针:关闭逐窗 z-score 归一化,看跨负载域偏移是否重新出现(source_only 精度应下降)。

不修改原代码,只在内存中打补丁。为避免缓存污染,强制 cache_dir=None。
用法: python probe_norm.py <src> <tgt> <method> [epochs]
"""
import sys
import numpy as np

import data_utils

_orig_build = data_utils.build_cwru_domain


def build_no_cache(root, load, classes=("NORMAL", "B", "IR", "OR"),
                   size="007", window=1024, step=None, cache_dir=None):
    """绕开缓存:否则会读到已归一化的缓存数据。"""
    return _orig_build(root, load, classes, size, window, step, cache_dir=None)


data_utils.build_cwru_domain = build_no_cache
# 关闭逐窗 z-score(恒等映射)
data_utils.zscore_per_window = lambda X, eps=1e-6: np.asarray(X, dtype=np.float32)

import train  # noqa: E402

train.build_cwru_domain = build_no_cache

src, tgt, method = sys.argv[1], sys.argv[2], sys.argv[3]
epochs = sys.argv[4] if len(sys.argv) > 4 else "60"

sys.argv = ["train.py", "--method", method, "--src_load", src, "--tgt_load", tgt,
            "--k", "5", "--seed", "0", "--epochs", epochs,
            "--data_root", "data/cwru", "--out_dir", "results/probe_norm"]
train.main()
