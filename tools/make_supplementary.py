"""把论文的补充材料包组装到 supplementary/。

**为什么要脚本**:补充材料 = 代码 + 落盘产物 + 两者对应关系的说明。手工打包时最容易
出错的两件事,恰好都是致命的:漏掉某个产物(于是某张表无法核对)、或包的布局与命令
不符(于是 README 里的命令跑不起来)。脚本把清单写死,缺一个文件就报错退出,并且让
包内保持仓库的目录布局 —— README 里给的 `python tools/verify_paper_numbers.py artifacts`
在包里可以直接跑。

结尾重算每个文件的 sha256 写进 MANIFEST.txt,审稿人可据此确认包未被改动。

用法: python tools/make_supplementary.py
      输出 <repo>/supplementary/(先清空再写,不保留上一次的残留)
"""
import glob
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(HERE, "supplementary")

# Windows 控制台默认 GBK,中文直接写 stdout 会乱码,统一改 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 环境:2026-10-06 在本机实测的版本(见 README §6)。torch 只在重跑训练表时才需要。
REQUIREMENTS = """numpy==2.4.6
scipy==1.17.1
matplotlib==3.11.2
tqdm
# 重跑训练表(tab:synth / tab:leak / tab:main / tab:fewshot / Paderborn 各表)才需要:
torch==2.11.0
"""

# ---- 仓库根下的模块(实验与数据处理)-----------------------------------------
ROOT_MODULES = ("data_utils.py", "dt_synth.py", "pb_utils.py", "twin_calib.py",
                "models.py", "train.py")
# 不进包的 tools/:只与 Windows 环境诊断有关,与论文无关
TOOLS_SKIP = ("diag_dll.py", "diag_integrity.py", "diag_loader.py")

# ---- 产物 -------------------------------------------------------------------
# 表 III 的运行日志已归位到 results_paper/tab_leak.log;表 I 的几份日志还在仓库根下,
# 一并收进来(它们是"这条命令确实跑过、跑出这个数"的直接记录)。
ARTIFACT_GLOBS = ("results_paper/*.json", "results_paper/*.log", "results_paper*.log",
                  "results_pb/*.json", "results_pb/*.log",
                  "results/final/*.csv", "results/*.csv")
# results_paper/pre_fix_x098/ 是修复前的旧产物留档(内部审计用),不进补充材料
ARTIFACT_SKIP_DIRS = ("pre_fix_x098",)
# 这些产物是表的依据,缺一个就说明实验没跑完 —— 宁可不打包
REQUIRED_ARTIFACTS = ("results_paper/tab_synth.json", "results_paper/tab_leak.json",
                      "results_paper/tab_fewshot.json", "results_pb/pb_levels.json",
                      "results_pb/d_twin_k5.json", "results_pb/d_twin_k20.json",
                      "results_pb/d_twin_k50.json")

PAPER_FILES = ("paper/manuscript.tex", "paper/manuscript_mssp.tex", "paper/refs.bib",
               "paper/highlights.txt", "paper/figs/fig_leakage.pdf",
               "paper/figs/fig_spectra.pdf")

README = r"""# Supplementary material

**Why Simulation-Based Sample Supplementation Fails in Cross-Domain Bearing Fault
Diagnosis: The Recording-Fingerprint Effect and an Evaluation Protocol That Reveals It**
Fangyu Zhang, Julong He

This package holds the code, the numerical artifacts and the figure sources behind every
quantitative claim in the paper. It mirrors the authors' working repository, so the
commands below are exactly the commands the authors ran; nothing was rewritten for
publication.

---

## 1. Check the numbers: no dataset download needed

```bash
cd supplementary
python tools/verify_paper_numbers.py artifacts
```

This reads every artifact in `results_paper/`, `results_pb/` and `results/final/`,
recomputes the quantity each table reports, compares it with the value printed in the
manuscript, and prints one line per check. **170 checks**; exit code 0 means all agree,
exit code 1 means a mismatch, and the offending rows are printed.

Requirements: Python 3 with numpy and scipy (`requirements.txt`).

## 2. The remaining checks need the raw data

```bash
python tools/verify_paper_numbers.py raw     # 35 further checks
python tools/verify_paper_numbers.py all     # all 205
```

`raw` recomputes the deterministic statistics of Table II, Table IV, Section
"Diagnosis II" and the bench-tone paragraph directly from the signals. The datasets are
public and are not redistributed here (Paderborn alone is ~15 GB).

### CWRU 12 kHz drive end

```bash
python tools/prepare_cwru.py            # downloads the 40 official files into data/cwru/
python tools/prepare_cwru.py --verify   # checks an existing data/cwru/ file by file
```

**One warning, and it matters.** CWRU's `99.mat` (the 2 hp normal baseline) contains
*two* recordings in a single file: `X098` (1 hp) and `X099` (2 hp), with `X098` first.
Any loader that takes "the first DE channel" silently assigns the load-2 NORMAL class the
**1 hp** recording — a byte-for-byte duplicate of `Normal_1.mat`. We hit exactly that
during this work: the CWRU shaft rates, recovered from the recordings themselves, are
1795.5 rpm for `X097` and 1772 / 1750 rpm for `X098` / `X099`, which identifies the two
channels unambiguously. `tools/prepare_cwru.py` picks `X099` explicitly for that file, and
`data_utils.load_de_channel` now refuses any file with more than one DE channel rather
than guessing.

### Paderborn (KAt)

22 bearings, `data/paderborn/<code>/<code>/<condition>_<code>_<index>.mat`, from
`https://groups.uni-paderborn.de/kat/BearingDataCenter/<code>.rar`. Note the class
definition used here (`pb_utils.CLASSES_LOC`) is by *damage location* (healthy / inner
race / outer race), parsed from the fact sheet PDF shipped with each bearing.

## 3. What the checks establish, and what they do not

Two different kinds of claim; they should not be conflated.

| | how it is verified | checks |
|---|---|---|
| Table II, Table IV, §Diagnosis II, bench tone | recomputed from the raw signals — deterministic given the data | 35 |
| Tables I, III, V, VI, VII, VIII, IX | the manuscript agrees with the stored artifacts | 170 |

For the second group the check establishes **manuscript ↔ artifact** agreement, not
**artifact ↔ fresh run**. Those tables come from training, and this hardware is not
run-to-run deterministic: we measured the same command producing 10-seed means up to
0.033 apart, and a 3-seed standard deviation wrong by more than an order of magnitude
(0.005 measured against 0.077 measured properly). The paper says so, and it is why the
secondary tables are reported over `n_repeat × n_seed` runs rather than over three seeds.
Re-running the training scripts on other hardware will give values close to, but not
identical to, the printed ones.

## 4. Reproducing the training tables

See `TABLE_ARTIFACTS.md` for the table-by-table mapping of artifact file, producing
command and verifying function.

## 5. Layout

```
tools/            all analysis code; the entry points are verify_paper_numbers.py
results_paper/    artifacts for Tables I, III, IX (+ the per-run files they merge)
results_pb/       artifacts for Tables V, VI, VII and the cross-benchmark spectra
results/final/    per-run CSVs for Table VIII (7 methods x 5 seeds)
paper/            the LaTeX sources and the two figure PDFs
data_utils.py, dt_synth.py, pb_utils.py, twin_calib.py, models.py, train.py
```

## 6. Environment

Windows 11, Python 3.11.1, numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2, torch 2.11.0
(CUDA). Training tables used one consumer GPU; no GPU-specific code paths are involved.

## 7. Integrity

`MANIFEST.txt` lists the SHA-256 of every file in this package.
"""

TABLE_ARTIFACTS = r"""# Which artifact backs which table

Every numbered table in the manuscript has an on-disk artifact, and
`tools/verify_paper_numbers.py` compares the manuscript's values against it. Run
`python tools/verify_paper_numbers.py artifacts` to perform all of the checks in the
"verified by" column that do not need raw data.

| Manuscript table | Artifact | Producing command | Verified by |
|---|---|---|---|
| Table I `tab:synth` | `results_paper/tab_synth.json` (merged from `tab_synth_10a.json`, `tab_synth_10b.json`; n_repeat=2 x 10 seeds) | `python tools/probe_fp.py 3 5 400 40 10 2` | `table_synth` |
| Table II `tab:classsep` | none — recomputed from `data/cwru` | `python tools/probe_classsep.py` | `table_classsep` (raw) |
| Table III `tab:leak` | `results_paper/tab_leak.json` (n_repeat=2 x 3 seeds) | `python tools/probe_leakage.py all 60 3 5 2` | `table_leak` |
| Table IV `tab:bench` | `results_pb/spectra_bench.json` (recomputed anyway) | `python tools/probe_spectra.py` | `table_bench` (raw) |
| Table V `tab:pb_leak` | `results_pb/pb_levels.json` (5 seeds, 200 epochs) | `python tools/probe_pb_levels.py 200 5` | `table_pb_levels` |
| Table VI `tab:pb_levels` | same file (same run) | same | `table_pb_levels` |
| Table VII `tab:pb_fewshot` | `results_pb/d_twin_k5.json`, `d_twin_k20.json`, `d_twin_k50.json` | `python tools/probe_paderborn.py d` | `table_pb_fewshot` |
| Table VIII `tab:main` | `results/final/*.csv` (7 methods x 5 seeds) | `bash run_s2.sh all` then `python tools/collect_results.py` | `table_main` |
| Table IX `tab:fewshot` | `results_paper/tab_fewshot.json` (n_repeat=2 x 3 seeds) | `python tools/probe_fewshot.py 3 0 40 3 2` | `table_fewshot` |
| Fig. 1 `fig:leakage` | `paper/figs/fig_leakage.pdf` | `python tools/make_figures.py` | the same numbers as Table III |
| Fig. 2 `fig:spectra` | `paper/figs/fig_spectra.pdf` | `python tools/make_figures.py` | the same numbers as §Diagnosis II |

Notes.

- Table VII's k=5 row averages 5 seeds; the k=20 and k=50 rows average 3 (see the array
  lengths in the JSON files).
- Tables V and VI come from one run by construction: the same training set is evaluated
  against three test sets that differ by one factor each, which is what makes the
  session/specimen decomposition meaningful.
- The merged files (`tab_synth.json`, `tab_leak.json`) were produced either by the runner's
  `n_repeat` argument or by `python tools/merge_runs.py out.json runA.json runB.json`,
  which refuses to merge runs whose configurations differ. The per-run inputs are kept in
  the same directories.
- `results_paper/tab_synth_run1.json`, `tab_synth_10a.json`, `tab_synth_10b.json` and
  `tab_synth_pooled.json` are retained as the provenance chain of the merged Table I
  artifact. For Table III the post-fix run was a single two-repeat invocation; its console
  transcript is `results_paper/tab_leak.log` and the command is recorded in the artifact's
  own `provenance` field.
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def collect():
    """返回 [(源相对路径, 包内相对路径)],缺文件即报错退出。"""
    items = [(f, f) for f in ROOT_MODULES]
    for p in sorted(glob.glob(os.path.join(HERE, "tools", "*.py"))):
        rel = os.path.relpath(p, HERE).replace("\\", "/")
        if os.path.basename(p) not in TOOLS_SKIP:
            items.append((rel, rel))
    for pat in ARTIFACT_GLOBS:
        hits = sorted(glob.glob(os.path.join(HERE, pat)))
        for p in hits:
            rel = os.path.relpath(p, HERE).replace("\\", "/")
            if any(d in rel.split("/") for d in ARTIFACT_SKIP_DIRS):
                continue
            items.append((rel, rel))
    items += [(f, f) for f in PAPER_FILES]

    missing = [src for src, _ in items if not os.path.exists(os.path.join(HERE, src))]
    if missing:
        print("**以下文件不存在,不打残包**:")
        for m in missing:
            print("   ", m)
        sys.exit(1)
    # 核心脚本必须在场:包的意义就是"开箱可核对",少一个就白打
    names = {os.path.basename(d) for _, d in items}
    for must in ("verify_paper_numbers.py", "probe_leakage.py", "probe_fp.py",
                 "probe_fewshot.py", "probe_paderborn.py", "prepare_cwru.py"):
        if must not in names:
            print(f"**缺少 {must},不打残包**")
            sys.exit(1)
    # 表所依赖的产物也必须在场
    have = {d for _, d in items}
    for must in REQUIRED_ARTIFACTS:
        if must not in have:
            print(f"**缺少产物 {must},不打残包**(先把对应的实验跑出来)")
            sys.exit(1)
    if not glob.glob(os.path.join(HERE, "results", "final", "*.csv")):
        print("**results/final/ 没有 CSV(表 VIII 的依据),不打残包**")
        sys.exit(1)
    return items


def main():
    items = collect()
    if os.path.isdir(DST):
        shutil.rmtree(DST)          # 先清空,避免上一轮的残留文件混进来
    for src, dst in items:
        d = os.path.join(DST, dst)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(os.path.join(HERE, src), d)
    os.makedirs(os.path.join(DST, "paper", "figs"), exist_ok=True)
    for name, text in (("README.md", README), ("TABLE_ARTIFACTS.md", TABLE_ARTIFACTS),
                       ("requirements.txt", REQUIREMENTS)):
        with open(os.path.join(DST, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)

    # 清单:逐文件 sha256,审稿人据此确认包未被改动
    rows = []
    for root, _dirs, files in os.walk(DST):
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, DST).replace("\\", "/")
            if rel == "MANIFEST.txt":
                continue
            rows.append((rel, os.path.getsize(p), sha256(p)))
    rows.sort()
    with open(os.path.join(DST, "MANIFEST.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("SHA-256 of every file in this supplementary package\n")
        fh.write(f"{len(rows)} files\n\n")
        for rel, size, digest in rows:
            fh.write(f"{digest}  {size:>9}  {rel}\n")

    total = sum(r[1] for r in rows)
    print(f"[write] {DST}")
    print(f"  {len(rows)} 个文件,{total/1e6:.2f} MB")
    print("  含 README.md / TABLE_ARTIFACTS.md / MANIFEST.txt / requirements.txt")


if __name__ == "__main__":
    main()
