"""投稿前自查:两个目标期刊的格式硬约束逐条检查。

覆盖两份稿子:
  paper/manuscript.tex        IEEEtran 版 -> IEEE TIM(后备)
  paper/manuscript_mssp.tex   elsarticle 版 -> MSSP(当前目标)

口径来源:
  * IEEE TIM:官方 Information for Authors 页(https://ieee-ims.org/publication/ieee-tim/information-authors)
    给出常规论文无页数上限、超页费按 IMS 的 Overlength Page Charge Agreement 收取;
    该 Agreement(2024-09 版)给的是每页 265 美元 / IMS 会员 220 美元。TIM 的
    guide 页对常规论文的摘要字数**没有**明文规定(只有短论文 150 词),所以这里按
    IEEE 通行惯例 250 词给出提示,不是硬性红线。
  * MSSP / Elsevier:ScienceDirect 的 guide-for-authors 屏蔽自动抓取(HTTP 403),
    本机也没装 agent-browser,所以**摘要 250 词、关键词 1–7、Highlights 3–5 条且每条
    85 字符**这几条来自二手来源,投稿前必须在 Editorial Manager 里再核一遍。
    订阅通道不收版面费这一点是 Elsevier 的标准模型,另需以投稿时的页面为准。

用法:  python tools/check_submission.py
"""
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(HERE, "paper")


def strip_tex(s):
    s = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", s)
    s = re.sub(r"[$\\{}~]", " ", s)
    return s


def probe_tex(path, label):
    src = open(path, encoding="utf-8").read()
    print(f"\n===== {label} =====  ({os.path.basename(path)})")

    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S)
    if not m:
        print("  ** 找不到 abstract,检查文件是否损坏 **")
        return
    words = [w for w in strip_tex(m.group(1)).split() if w.strip()]
    flag = "OK " if len(words) <= 250 else "**超**"
    print(f"  摘要词数        : {len(words):3d}  [{flag}] (上限 250)")

    km = re.search(r"\\begin\{(IEEEkeywords|keyword)\}(.*?)\\end\{\1\}", src, re.S)
    if km:
        # 注意:elsarticle 的分隔符是字面量 \sep,必须 re.escape ——
        # 直接写 r"\sep" 会被当成正则(\s + "ep"),于是 7 个关键词被算成 1 个。
        sep = "," if km.group(1) == "IEEEkeywords" else re.escape(r"\sep")
        terms = [t.strip() for t in re.split(sep, km.group(2)) if t.strip()]
        ordered = terms == sorted(terms, key=str.lower)
        print(f"  关键词          : {len(terms)} 个  [{'OK ' if 1 <= len(terms) <= 7 else '**超**'}]"
              f" (1-7);字母序:{'是' if ordered else '否'}")
    else:
        print("  关键词          : ** 未找到 **")

    n_fig = len(re.findall(r"\\begin\{figure\*?\}", src))
    n_tab = len(re.findall(r"\\begin\{table\*?\}", src))
    print(f"  图 / 表         : {n_fig} / {n_tab}")
    keys = set()
    for mm in re.findall(r"\\cite\{([^}]*)\}", src):
        keys |= {k.strip() for k in mm.split(",")}
    print(f"  引用文献数      : {len(keys)}")

    log = path.replace(".tex", ".log")
    if not os.path.exists(log):
        print("  (没有 .log,先编译一次)")
        return
    txt = open(log, encoding="utf-8", errors="ignore").read()
    mm = re.search(r"Output written on .*?\((\d+) pages", txt)
    pages = int(mm.group(1)) if mm else None
    overfull = txt.count("Overfull")
    undef = len(re.findall(r"Citation .* undefined|Reference .* undefined", txt))
    errors = len(re.findall(r"^!", txt, re.M))
    print(f"  编译页数        : {pages}")
    print(f"  Overfull / 未定义引用 / 报错 : {overfull} / {undef} / {errors}"
          f"  [{'OK' if overfull == 0 and undef == 0 and errors == 0 else '**需处理**'}]")
    return pages


def main():
    p1 = probe_tex(os.path.join(PAPER, "manuscript.tex"), "IEEEtran 版(目标 IEEE TIM,后备)")
    if p1:
        over = max(0, p1 - 8)
        print(f"  → TIM 免费 8 页,超 {over} 页:强制超页费约 ${over * 265}"
              f"(非会员)/ ${over * 220}(IMS 会员),另加税")

    p2 = probe_tex(os.path.join(PAPER, "manuscript_mssp.tex"), "elsarticle 版(目标 MSSP)")
    if p2:
        print("  → MSSP 订阅通道不按页收费,页数无上限(此处为双倍行距 preprint 页数)")

    hl = os.path.join(PAPER, "highlights.txt")
    print("\n===== MSSP Highlights =====")
    if not os.path.exists(hl):
        print("  ** 缺 highlights.txt —— Elsevier 必交材料 **")
        return
    n = 0
    for line in open(hl, encoding="utf-8"):
        m = re.match(r"^\d+\.\s+(.*?)\s*\[\d+\]\s*$", line.strip())
        if not m:
            continue
        n += 1
        text = m.group(1)
        ok = len(text) <= 85
        print(f"  [{n}] {len(text):3d} 字符  [{'OK ' if ok else '**超**'}]  {text}")
    print(f"  共 {n} 条  [{'OK ' if 3 <= n <= 5 else '**条数不合规**'}] (要求 3-5 条)")


if __name__ == "__main__":
    main()
