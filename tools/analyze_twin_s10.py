"""把 Paderborn 无源小样本的 10-seed 重跑与入稿的 5/3-seed 版本并排比较。

正文表 VII 原来在 k=5 用 5 个 seed、k=20/50 各 3 个,差异落在"一个标准差以内",
审稿人据此可以说"正对照没有统计力度"。这里对每个 k 计算:
  - 四列均值 ± 标准差(旧 / 新)
  - 逐 seed 配对差(同一 seed 下 指纹标定+物理 − 指纹标定)
  - 配对符号检验(双侧精确二项,p = 2·P(X≥n_win | n, 1/2))
只读 results_pb/d_twin_k{k}.json 与 d_twin_k{k}_s10.json,不改任何产物。

用法: python tools/analyze_twin_s10.py
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results_pb")
NAMES = ["真实支撑集(仅 k 个/类)", "纯孪生(无标定)", "指纹标定(无物理)", "指纹标定+物理"]


def sign_test_p(d):
    """双侧精确符号检验,忽略差值为 0 的配对。"""
    d = np.asarray(d, dtype=float)
    n = int(np.sum(d != 0))
    if n == 0:
        return 1.0, 0, 0
    w = int(np.sum(d > 0))
    # 双侧:取较远的那一侧尾部 ×2(等价于 P(X>=max(w,n-w)) 的两倍)
    tail = sum(math.comb(n, i) for i in range(max(w, n - w), n + 1))
    return min(1.0, 2.0 * tail / 2 ** n), w, n


def load(k, suffix=""):
    p = os.path.join(RES, f"d_twin_k{k}{suffix}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def report(k):
    old, new = load(k), load(k, "_s10")
    if new is None:
        print(f"\n[k={k}] 还没有 _s10 产物")
        return
    print(f"\n[k={k}]  旧 n={len(old[NAMES[0]])}  新 n={len(new[NAMES[0]])}")
    print(f"    {'训练集':<24}{'旧 均值±标准差':>22}{'新 均值±标准差':>22}")
    for nm in NAMES:
        o, w = np.array(old[nm]), np.array(new[nm])
        print(f"    {nm:<24}{o.mean():>12.4f} ± {o.std():.4f}"
              f"{w.mean():>12.4f} ± {w.std():.4f}")

    pairs = [("指纹+物理 − 指纹", NAMES[3], NAMES[2]),
             ("指纹 − 纯孪生", NAMES[2], NAMES[1]),
             ("指纹+物理 − 纯孪生", NAMES[3], NAMES[1]),
             ("真实支撑 − 指纹+物理", NAMES[0], NAMES[3])]
    print(f"    {'配对(新)':<24}{'均值差':>10}{'胜/负':>10}{'符号检验 p':>12}")
    for label, a, b in pairs:
        d = np.array(new[a]) - np.array(new[b])
        p, w, n = sign_test_p(d)
        print(f"    {label:<24}{d.mean():>+10.4f}{f'{w}/{n - w}':>10}{p:>12.4f}")
    # 旧结果里同一配对的符号检验,看扩到 10 seed 是否改变了结论
    print(f"    {'(旧 n 的同配对)':<24}{'':>10}{'':>10}{'':>12}")
    for label, a, b in pairs:
        d = np.array(old[a]) - np.array(old[b])
        p, w, n = sign_test_p(d)
        print(f"    {label:<24}{d.mean():>+10.4f}{f'{w}/{n - w}':>10}{p:>12.4f}")


def main():
    for k in (5, 20, 50):
        report(k)


if __name__ == "__main__":
    main()
