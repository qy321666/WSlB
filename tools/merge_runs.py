"""把同一张表的多次重复运行合并成一条可复核产物。

存在的理由:本仓库 GPU 内核非确定性足以让"10 个 seed 的均值"在两次同命令运行间摆动
0.03,而 3 个种子的 ± 更是不可信(实测某次 std 0.005,真值 0.077)。次要表格的
做法是**同配置整跑多遍,报告全部 n_repeat x n_seed 次运行**的统计量。

runner 现在都支持直接一次跑多遍(probe_fp 的第 6 个参数、probe_leakage 的第 5 个);如果
已经分别跑过几遍(比如加参数之前遗留的输出),用这个脚本合并,不必重跑。

支持三种产物形状,按顶层键自动分派:
  fp       `sources`          (tab_synth.json)      —— 逐 (合成源) 平均
  leakage  `by_support_load`  (tab_leak.json)       —— 逐 (支撑域, 训练配置) 平均
  fewshot  `by_k`             (tab_fewshot.json)    —— 逐 (k, 增广源) 平均

用法: python tools/merge_runs.py out.json runA.json runB.json [...]
      配置(方向 / k / seed 数 / epoch)必须完全一致,否则拒绝合并 —— 不同实验拼在一
      起正是 2026-10-05 `tab:pb_leak` 整表作废的原因。
"""
import json
import sys

FIELDS = ("src_load", "tgt_load", "k", "n_per_class", "epochs", "n_seed",
          "n_aug_per_class", "window", "classes")


def _cfg_ok(base, other, where):
    for key in FIELDS:
        if key in base or key in other:
            if base.get(key) != other.get(key):
                raise SystemExit(f"{where} 配置不一致({key}): {base.get(key)} != "
                                 f"{other.get(key)} —— 不同实验不能合并成一行")


def _stats(vals):
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return m, sd


def _grab(d, *keys):
    """依次尝试若干旧键名,兼容加 n_repeat 之前的产物。"""
    for k in keys:
        if k in d:
            return d[k]
    return None


def merge_fp(runs):
    base = dict(runs[0]["config"])
    base["n_repeat"] = len(runs)
    out = {"config": base, "sources": {}}
    for name in runs[0]["sources"]:
        vals, discs = [], []
        for r in runs:
            s = r["sources"][name]
            vals += _grab(s, "f1_per_run", "f1_per_seed")
            discs.append(s["disc_mean"])
        m, sd = _stats(vals)
        out["sources"][name] = {"f1_per_run": vals, "f1_mean": m, "f1_std": sd,
                                "n_runs": len(vals),
                                "disc_mean": sum(discs) / len(discs)}
    for extra in ("fingerprint_cos_sup_vs_full", "fingerprint_cos_src_vs_full"):
        if extra in runs[0]:
            out[extra] = runs[0][extra]
    return out


def merge_leakage(runs):
    out = {"by_support_load": {}}
    for r in runs:
        for L, blk in r["by_support_load"].items():
            if L not in out["by_support_load"]:
                out["by_support_load"][L] = {"config": dict(blk["config"]), "rows": {}}
            else:
                _cfg_ok(out["by_support_load"][L]["config"], blk["config"], f"支撑域 {L}")
            dst = out["by_support_load"][L]
            for name, row in blk["rows"].items():
                if name not in dst["rows"]:
                    dst["rows"][name] = {"same_all": [], "cross_all": {},
                                         "optimism_all": []}
                d = dst["rows"][name]
                d["same_all"] += _grab(row, "same_per_run", "same_per_seed")
                d["optimism_all"].append(row["optimism"])
                for l, v in row["cross_to"].items():
                    d["cross_all"].setdefault(l, []).append(v["mean"])
    for L, blk in out["by_support_load"].items():
        blk["config"]["n_repeat"] = len(runs)
        for name, d in blk["rows"].items():
            same_m, same_sd = _stats(d.pop("same_all"))
            cross_m, _ = _stats([sum(d["cross_all"][l]) / len(d["cross_all"][l])
                                 for l in d["cross_all"]])
            d["same_mean"], d["same_std"] = same_m, same_sd
            d["cross_mean"] = cross_m
            d["optimism"] = same_m - cross_m
            d["n_runs"] = len(runs) * blk["config"]["n_seed"]
            d["cross_to"] = {l: {"mean": sum(v) / len(v)} for l, v in d["cross_all"].items()}
            d.pop("cross_all")
            d.pop("optimism_all")
    return out


def merge_fewshot(runs):
    base = dict(runs[0]["config"])
    base["n_repeat"] = len(runs)
    out = {"config": base, "by_k": {}}
    for kk in runs[0]["by_k"]:
        out["by_k"][kk] = {}
        for name in runs[0]["by_k"][kk]:
            vals = []
            for r in runs:
                s = r["by_k"][kk][name]
                vals += _grab(s, "per_run", "per_seed")
            m, sd = _stats(vals)
            out["by_k"][kk][name] = {"mean": m, "std": sd, "per_run": vals,
                                     "n_runs": len(vals)}
    return out


def merge(paths):
    runs = [json.load(open(p, encoding="utf-8")) for p in paths]
    if "sources" in runs[0]:
        for r in runs[1:]:
            _cfg_ok(runs[0]["config"], r["config"], "fp")
        return "fp", merge_fp(runs)
    if "by_support_load" in runs[0]:
        return "leakage", merge_leakage(runs)
    if "by_k" in runs[0]:
        for r in runs[1:]:
            _cfg_ok(runs[0]["config"], r["config"], "fewshot")
        return "fewshot", merge_fewshot(runs)
    raise SystemExit("认不出产物形状(顶层键应为 sources / by_support_load / by_k)")


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    shape, out = merge(sys.argv[2:])
    out["provenance"] = (f"由 {len(sys.argv) - 2} 次同配置运行合并"
                         f"({', '.join(sys.argv[2:])});n_repeat={len(sys.argv) - 2}")
    json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[save] {sys.argv[1]}  形状={shape}  {len(sys.argv) - 2} 遍合并")


if __name__ == "__main__":
    main()
