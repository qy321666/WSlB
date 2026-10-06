"""把 IEEEtran 版的 manuscript.tex 转成 Elsevier elsarticle 版(投 MSSP)。

**为什么用脚本而不是手改**:正文有 900 多行,手改一次就再也无法保证与 IEEEtran 版
同步。这里只做**机械替换**(导言区、浮动体环境、宽度宏、参考文献样式),正文一个字不碰;
改完两份稿子的正文仍逐字一致,任何一句论断的修改只需要在 IEEEtran 版里做一次。

改完必须编译验证(elsee 见 tools/check_submission.py):
    xelatex -> bibtex -> xelatex -> xelatex

用法: python tools/to_elsarticle.py
      输入 paper/manuscript.tex -> 输出 paper/manuscript_mssp.tex
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "paper", "manuscript.tex")
DST = os.path.join(HERE, "paper", "manuscript_mssp.tex")

# ---------------------------------------------------------------- 新的导言区 + 卷首
# 作者信息留成显式占位符:OpenAlex/Editorial Manager 要逐作者填 ORCID,不能编。
FRONT = r"""% !TEX program = xelatex
% Elsevier elsarticle 版(目标期刊:Mechanical Systems and Signal Processing)。
% 由 tools/to_elsarticle.py 从 IEEEtran 版 manuscript.tex 机械转换而来 —— 正文逐字一致,
% 改内容请改 IEEEtran 版再重跑转换,不要只改这一份。
\documentclass[preprint,12pt]{elsarticle}

\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{array}
\usepackage{multirow}
\usepackage{url}
% hyperref 必须排在 orcidlink 之前:orcidlink 内部会加载 hyperref,
% 若先加载 orcidlink,后面再 \usepackage[...]{hyperref} 会报 "Option clash"。
\usepackage[colorlinks=true,allcolors=blue]{hyperref}
\usepackage{orcidlink}

\newcommand{\ml}[1]{\textsc{#1}}

% elsarticle 的 \paragraph 是行内小标题,长标题后面紧跟正文时不给断行机会,
% 会留下 10~22pt 的 Overfull。放开一点点伸缩量即可消掉(纯排版,不改内容)。
\setlength{\emergencystretch}{2em}

\begin{document}

\begin{frontmatter}

\title{Why Simulation-Based Sample Supplementation Fails in Cross-Domain Bearing
Fault Diagnosis: The Recording-Fingerprint Effect and an Evaluation Protocol
That Reveals It}

% 作者信息(2026-10-06 由作者提供)。
% 注意:TeX Live 2026 的 elsarticle.cls 里**完全没有** orcid 支持
% (grep orcid elsarticle.cls 返回 0 处),所以 `\author[1]{Name}[orcid=...]` 那种写法
% 不会被解析,会原样排进 PDF(实测确实如此)。改用已安装的 orcidlink 包
% (包已在导言区加载 —— 它是 \usepackage,放在正文里会报 "Can be used only in preamble")。

\author[1]{Fangyu Zhang\,\orcidlink{0009-0009-7352-1484}\corref{cor1}}
\ead{fyzhang9925@mails.jlu.edu.cn}
\author[2]{Julong He\,\orcidlink{0009-0003-0143-7204}}
\ead{u202541560@xs.ustb.edu.cn}
\cortext[cor1]{Corresponding author.}
\affiliation[1]{organization={College of Mechanical and Aerospace Engineering,
                            Jilin University},
                city={Changchun},
                postcode={130025},
                country={China}}
\affiliation[2]{organization={National College for Excellent Engineers,
                            University of Science and Technology Beijing},
                city={Beijing},
                postcode={100083},
                country={China}}

\begin{abstract}
"""


def convert(src: str) -> str:
    # 1) 导言区 + 标题 + 作者 + 摘要开头 -> elsarticle 卷首
    m = re.search(r"\\begin\{abstract\}", src)
    if not m:
        raise SystemExit("找不到 \\begin{abstract}")
    body = FRONT + src[m.end():].lstrip("\n")

    # 2) 摘要与 IEEEkeywords 的收尾 -> elsarticle 的 keyword 环境。
    #    注意:原文在 \begin{IEEEkeywords} 之前已经有一个 \end{abstract},
    #    所以这里**只**产出 keyword 环境,不要再补 \end{abstract}(补了就是一个多余的收尾,
    #    LaTeX 会报 "frontmatter ... ended by \end{abstract}")。
    body = re.sub(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}",
                  lambda mm: "\\begin{keyword}\n"
                             + " \\sep\n".join(
                                 t.strip().rstrip(".") for t in mm.group(1).split(",") if t.strip())
                             + "\n\\end{keyword}\n\n\\end{frontmatter}",
                  body, count=1, flags=re.S)
    if "\\begin{keyword}" not in body:
        raise SystemExit("IEEEkeywords 替换失败")

    # 2b) 原 table* 的宽表(表 III,6 列)在单栏 preprint 下放不下,必须缩放。
    #     IEEEtran 里它靠跨双栏获得宽度;改成单栏后要把 tabular 包进 resizebox,
    #     否则会撑出 60pt 的 Overfull(实测就是这个数)。
    #     包装行带 elsarticle-wrap 标记,好让下面的自查把"我加的"和"原有的"区分开。
    def shrink_starred_table(mm):
        blk = mm.group(0)
        blk = re.sub(r"(\\begin\{tabular\})",
                     "\\\\resizebox{\\\\linewidth}{!}{%  % elsarticle-wrap\n\\1", blk, count=1)
        blk = re.sub(r"(\\end\{tabular\})", r"\1\n}  % elsarticle-wrap", blk, count=1)
        return blk

    body = re.sub(r"\\begin\{table\*\}.*?\\end\{table\*\}",
                  shrink_starred_table, body, flags=re.S)

    # 2c) 跨栏浮动体 -> 单栏(preprint 是单栏,table*/figure* 会报错)
    body = body.replace(r"\begin{table*}", r"\begin{table}")
    body = body.replace(r"\end{table*}", r"\end{table}")
    body = body.replace(r"\begin{figure*}", r"\begin{figure}")
    body = body.replace(r"\end{figure*}", r"\end{figure}")

    # 4) 其余宽度宏不动:elsarticle 单栏下 \columnwidth 与 \linewidth 等价,
    #    保留原样可以让下面的自查按字面比对。

    # 5) 参考文献样式
    body = body.replace(r"\bibliographystyle{IEEEtran}",
                        r"\bibliographystyle{elsarticle-num}")

    # 6) Elsevier 强制要求的结尾声明块(插在参考文献之前)。
    #    内容由作者提供(2026-10-06):无基金;生成式 AI 仅用于语言润色;
    #    CRediT 分工:一作承担其余全部角色,二作承担 Supervision / Validation /
    #    Data curation / Visualization。这些属署名与合规声明,不能由我代拟。
    decl = r"""% >>> elsarticle-decl BEGIN(转换脚本插入,不是正文)
\section*{CRediT authorship contribution statement}
\textbf{Fangyu Zhang:} Conceptualization, Methodology, Software, Formal analysis,
Investigation, Resources, Writing -- original draft, Writing -- review \& editing,
Project administration.
\textbf{Julong He:} Supervision, Validation, Data curation, Visualization.

\section*{Declaration of competing interest}
The authors declare that they have no known competing financial interests or
personal relationships that could have appeared to influence the work reported
in this paper.

\section*{Funding}
This research did not receive any specific grant from funding agencies in the
public, commercial, or not-for-profit sectors.

\section*{Data availability}
The CWRU 12\,kHz drive-end and Paderborn KAt datasets analysed here are public.
The code and the per-table artifacts that reproduce every number in this paper
are available at \url{https://github.com/qy321666/WSlB} (release \texttt{v1.0},
commit \texttt{f81f3c7}) and are also provided as supplementary material to this
submission.

\section*{Declaration of generative AI and AI-assisted technologies in the writing process}
During the preparation of this work the authors used ClawsGO Science Agent in
order to improve the language of the manuscript. After using this tool, the
authors reviewed and edited the content as needed and take full responsibility
for the content of the publication.

% <<< elsarticle-decl END
"""
    body = body.replace(r"\bibliographystyle{elsarticle-num}",
                        decl + "\n\\bibliographystyle{elsarticle-num}")

    return body


def main():
    src = open(SRC, encoding="utf-8").read()
    out = convert(src)
    open(DST, "w", encoding="utf-8").write(out)
    print(f"[write] {DST}")

    # 自查:两份稿子的正文只允许出现"机械替换"造成的差异,多一个字都要报出来。
    # 做法是把两侧都归一化掉机械标记后逐行比对 —— 任何一句改写都会立刻暴露。
    # 注意:转换脚本自己插的声明块用标记括起来,比对时在标记处截断(否则会被误判成改字)。
    MARK = "% >>> elsarticle-decl BEGIN"
    b_src = out[:out.index(MARK)] if MARK in out else out
    def norm(lines):
        out = []
        for ln in lines:
            if "elsarticle-wrap" in ln:
                continue                     # 宽表缩放包装是转换新加的,不算正文
            s = (ln.strip()
                  .replace(r"\begin{table*}", r"\begin{table}")
                  .replace(r"\end{table*}", r"\end{table}")
                  .replace(r"\begin{figure*}", r"\begin{figure}")
                  .replace(r"\end{figure*}", r"\end{figure}"))
            out.append(s)
        return out

    a = norm(src[src.index(r"\section{Introduction}"):src.index(r"\bibliographystyle")].splitlines())
    # b_src 已在声明块标记处截断,所以直接取到末尾即可(末尾就是原 \bibliographystyle 之前)
    b = norm(b_src[b_src.index(r"\section{Introduction}"):].splitlines())
    if a != b:
        bad = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
        print(f"**正文有 {len(bad)} 处非机械差异(或行数不同:{len(a)} vs {len(b)}),必须查**:")
        for i, x, y in bad[:10]:
            print(f"  行{i+1}\n    - {x}\n    + {y}")
        sys.exit(1)
    print("正文校验通过:只有浮动体/宽度宏被机械替换,文字未动")


if __name__ == "__main__":
    main()
