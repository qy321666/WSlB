"""按官方直链重建 data/cwru/ 的 40 个 .mat,并改成加载器要求的命名。

**为什么需要这个脚本(而不是"下载后手动改名")**
CWRU 官方的 `99.mat`(2 hp 基线)里**同时装着两段录制**:1 hp 的 `X098` 与 2 hp 的
`X099`,而 `X098` 在文件里排在前面。任何"取第一个 DE 通道"的读法都会把"负载 2 的
NORMAL 类"静默读成**负载 1 的录制**(两段逐点相同) —— 本仓在 2026-10-06 之前正是
如此,表 III 因此整表重跑过。本脚本对 99.mat 显式取 `X099`,并对每个文件校验
"DE 通道恰好一路",把这类错误挡在下载环节。

编号取自各文件内的变量名(`X105_DE_time` -> 105),不是从文档抄的;可用
`--verify` 与已有的 data/cwru/ 逐点比对。

用法:
    python tools/prepare_cwru.py                    # 下载到 .cache/cwru_raw 并写出 data/cwru
    python tools/prepare_cwru.py --verify           # 只比对现有文件,不下载/不覆盖
    python tools/prepare_cwru.py --out DIR --raw DIR

数据来源:https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
直链规律:https://engineering.case.edu/sites/default/files/<编号>.mat
"""
import argparse
import os
import re
import sys
import urllib.request

import numpy as np
import scipy.io as sio

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://engineering.case.edu/sites/default/files/%d.mat"

# 每类的官方编号,按负载 0/1/2/3 排列。OR 取 @6:00 的那一组(编号 130--133 等),
# 与 CWRU 文档一致;本仓一直用的就是这一组。
MAP = {
    "IR007": (105, 106, 107, 108),
    "IR014": (169, 170, 171, 172),
    "IR021": (209, 210, 211, 212),
    "B007": (118, 119, 120, 121),
    "B014": (185, 186, 187, 188),
    "B021": (222, 223, 224, 225),
    "OR007": (130, 131, 132, 133),
    "OR014": (197, 198, 199, 200),
    "OR021": (234, 235, 236, 237),
}
# NORMAL 单列:97/98/100 各只含一段;99.mat 含 X098+X099,必须点名取 X099。
NORMAL = ((0, 97, "X097"), (1, 98, "X098"), (2, 99, "X099"), (3, 100, "X100"))

# 目标文件名由 data_utils._FNAME_RE 决定:^(Normal|IR|OR|B)(\d*)_(\d)\.mat$
# 所以 Normal 没有尺寸段,其余都是 <类型><尺寸>_<负载>.mat。


def fetch(num, raw_dir):
    """取官方 <num>.mat;已存在则不重复下。返回本地路径。"""
    os.makedirs(raw_dir, exist_ok=True)
    p = os.path.join(raw_dir, f"{num}.mat")
    if not os.path.exists(p):
        print(f"  下载 {num}.mat ...", flush=True)
        urllib.request.urlretrieve(BASE % num, p)
    return p


def pick_de(d, want=None):
    """从 .mat 里取唯一的 DE 通道;want 给出时必须取名字里含它的那一路。

    多路且未指定 -> 报错(宁可不转换,也不猜)。
    """
    keys = [k for k in d if not k.startswith("__") and re.search(r"DE", k, re.IGNORECASE)]
    if want is not None:
        keys = [k for k in keys if want in k]
    if len(keys) != 1:
        raise SystemExit(f"DE 通道数为 {len(keys)}({keys}),无法确定用哪一路")
    return keys[0]


def convert(num, raw_dir, out_dir, prefix, load, want=None):
    d = sio.loadmat(fetch(num, raw_dir))
    de = pick_de(d, want)
    num_in = re.match(r"X(\d+)_", de).group(1)
    assert int(num_in) == (int(want[1:]) if want else num), \
        f"{num}.mat 里的 {de} 编号与文件不符"
    out = {"DE_time": np.asarray(d[de]).reshape(-1, 1)}
    fe = de.replace("_DE_", "_FE_")
    if fe in d:
        out["FE_time"] = np.asarray(d[fe]).reshape(-1, 1)
    path = os.path.join(out_dir, f"{prefix}_{load}.mat")
    sio.savemat(path, out, format="5")
    return path, de, len(out["DE_time"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data", "cwru"))
    ap.add_argument("--raw", default=os.path.join(HERE, ".cache", "cwru_raw"))
    ap.add_argument("--verify", action="store_true",
                    help="只与 --out 里已有文件逐点比对,不下载也不写出")
    a = ap.parse_args()

    jobs = [(f"{k}", n, l, None) for k, nums in MAP.items() for l, n in enumerate(nums)]
    jobs += [("Normal", n, l, want) for l, n, want in NORMAL]

    n_ok = 0
    for prefix, num, load, want in jobs:
        name = f"{prefix}_{load}.mat"
        if a.verify:
            new = sio.loadmat(fetch(num, a.raw))
            de_new = pick_de(new, want)                  # 官方文件里指定/唯一的那一路
            exp = np.asarray(new[de_new]).ravel()
            p_old = os.path.join(a.out, name)
            if not os.path.exists(p_old):
                print(f"  [缺]     {name:<14} <- {num}.mat:{de_new}")
                continue
            old = sio.loadmat(p_old)
            cand = [k for k in old
                    if not k.startswith("__") and re.search(r"DE", k, re.IGNORECASE)]
            if len(cand) != 1:                           # 例如未修过的 99.mat:两路 DE
                print(f"  [多重]   {name:<14} <- {num}.mat:{de_new}  "
                      f"现有文件含 {len(cand)} 路 DE {cand}")
                continue
            got = np.asarray(old[cand[0]]).ravel()
            same = len(exp) == len(got) and np.array_equal(exp, got)
            print(f"  [{'OK' if same else 'MISMATCH'}] {name:<14} <- {num}.mat:{de_new}")
            n_ok += same
        else:
            path, de, n = convert(num, a.raw, a.out, prefix, load, want)
            print(f"  [write] {os.path.basename(path):<14} <- {num}.mat:{de}  ({n} 点)")
            n_ok += 1

    print(f"\n完成 {n_ok}/{len(jobs)} 个文件。")
    if a.verify and n_ok != len(jobs):
        print("有文件与官方不一致或结构不对 —— 不要用这份 data/cwru 跑实验。")
        sys.exit(1)


if __name__ == "__main__":
    main()
