"""探针:把类别从 4 类扩成 10 类(Normal + B/IR/OR × 007/014/021),看天花板是否被打破。

只做诊断,**不修改项目核心文件**:在内存中替换 data_utils.build_cwru_domain 与 train.CLASSES。

用法: python tools/probe_multiclass.py <src> <tgt> <method> [epochs]
      method 支持 source_only / dann(dt_dann 依赖 dt_synth 的 4 类特征频率映射,未扩展)
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import load_de_channel, segment, zscore_per_window  # noqa: E402

DATA = "data/cwru"

# 10 类:顺序即类别 id
CLASSES10 = ("NORMAL", "B007", "B014", "B021", "IR007", "IR014", "IR021",
             "OR007", "OR014", "OR021")
_FNAME = re.compile(r"^(Normal|IR|OR|B)(\d*)_(\d)\.mat$", re.IGNORECASE)


def build_cwru_domain_10(root, load, classes=CLASSES10, size=None,
                         window=1024, step=None, cache_dir=None):
    """构建某个负载下的 10 类数据集(不复用缓存,避免与 4 类缓存混淆)。"""
    if step is None:
        step = window // 2
    class_id = {c: i for i, c in enumerate(classes)}

    grouped = {}
    for f in sorted(os.listdir(root)):
        m = _FNAME.match(f)
        if not m:
            continue
        typ, sz, ld = m.group(1).upper(), m.group(2), int(m.group(3))
        if ld != load:
            continue
        cls = "NORMAL" if typ == "NORMAL" else f"{typ}{sz}"
        grouped.setdefault(cls, []).append(os.path.join(root, f))

    X_list, y_list = [], []
    for cls in classes:
        paths = grouped.get(cls)
        if not paths:
            raise FileNotFoundError(f"缺少类别 {cls} / 负载 {load} 的文件")
        for p in paths:
            win = segment(load_de_channel(p), window, step)
            X_list.append(win)
            y_list.append(np.full(len(win), class_id[cls]))

    X = np.concatenate(X_list, axis=0).astype("float32")
    y = np.concatenate(y_list, axis=0).astype("int64")
    return zscore_per_window(X), y


def main():
    import train
    train.build_cwru_domain = build_cwru_domain_10
    train.CLASSES = CLASSES10

    src, tgt, method = sys.argv[1], sys.argv[2], sys.argv[3]
    epochs = sys.argv[4] if len(sys.argv) > 4 else "60"

    sys.argv = ["train.py", "--method", method, "--src_load", src, "--tgt_load", tgt,
                "--k", "5", "--seed", "0", "--epochs", epochs,
                "--data_root", DATA, "--out_dir", "results/probe_10class"]
    train.main()


if __name__ == "__main__":
    main()
