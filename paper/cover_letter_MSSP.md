# Cover Letter — Mechanical Systems and Signal Processing (Elsevier)

> **使用说明**:方括号 `[...]` 处必须由作者填写,我没有替你编造。填完转成 PDF 上传到
> Elsevier Editorial Manager。
> **Highlights 是必交材料**(3–5 条、每条 ≤85 字符),已在同目录 `highlights.txt` 备好。

---

\[日期\]

To the Editor-in-Chief
*Mechanical Systems and Signal Processing*

Dear Editor,

We submit for your consideration the manuscript **"Why Simulation-Based Sample
Supplementation Fails in Cross-Domain Bearing Fault Diagnosis: The
Recording-Fingerprint Effect and an Evaluation Protocol That Reveals It"**, by
Fangyu Zhang and Julong He, as a **research article**.

**Significance.** Physics-based simulation, including "digital twin"
generators, is a mainstream source of synthetic training samples for
cross-domain bearing fault diagnosis. This paper shows that on the CWRU 12 kHz
drive-end benchmark the approach cannot work, for a structural reason that no
improvement in simulator fidelity can remove: every "fault type × load" cell of
that benchmark is a *single recording*, and the class-conditional magnitude
spectrum is a stable property of that individual recording rather than of the
fault. A carefully calibrated McFadden–Smith twin therefore produces nine fault
classes that are spectrally indistinguishable; a classifier trained on twin
data alone reaches 0.123 macro-F1 on real data against a chance level of 0.10.

**Why MSSP.** This journal has published the two reference points our work
builds on directly: the CWRU benchmark study by Smith and Randall (MSSP 64,
2015) and the benchmarking critique of that dataset by Hendriks, Dumond and
Knox (MSSP 169, 2022). Our contribution is the methodological successor to the
latter — it identifies the *mechanism* behind the benchmarking problems those
papers report, and supplies a quantitative control for them.

**What may be of most interest to your readership.** While quantifying the above
we found that the standard few-shot protocol on this benchmark draws support
and test windows from the same recording, so both carry the same fingerprint.
Re-evaluating cross-recording lowers reported macro-F1 by 0.15–0.19 on average
(up to 0.31), and it removes the apparent advantage of every augmentation
method we tested — including adding 400 *real* labeled samples per class, which
lands within noise of the trivial baseline. We give a self-contained protocol
control that any paper in this area can run at negligible cost. On the
Paderborn dataset, where several independently manufactured bearings share each
class, the same optimism reproduces (+0.340 macro-F1), and it decomposes into a
session term (+0.182) and a specimen term (+0.158).

**Not a general indictment of physics-based augmentation.** The negative result
is about the *pairing* of a generator with a benchmark. On Paderborn our
physics-carrying augmentation is no longer the worst entry, and the ordering
between physics-driven and fingerprint-driven synthesis inverts between the two
benchmarks exactly where the proposed mechanism predicts. That inversion
functions as a positive control for the diagnosis.

**Reproducibility.** Every numeric value in the paper is recomputed from the raw
public data by a released verification script that fails on any mismatch; the
per-table artifacts backing the tables accompany the submission. Tables are
reported over repeated executions rather than the conventional three seeds,
because we measured run-to-run variation on this hardware large enough to make
a three-seed standard deviation misleading — this is stated in the paper.

**Declarations.** The manuscript is original, is not under consideration
elsewhere, and all authors have approved it. The authors declare no competing
interests. This research did not receive any specific grant from funding
agencies in the public, commercial, or not-for-profit sectors. Generative AI was
used only to improve the language of the manuscript, as declared in the
manuscript under Elsevier's heading "Declaration of generative AI and
AI-assisted technologies in the writing process"; all scientific content,
experiments and conclusions are the authors' own. Author contributions follow
the CRediT taxonomy: Fangyu Zhang — Conceptualization, Methodology, Software,
Formal analysis, Investigation, Resources, Writing – original draft, Writing –
review & editing, Project administration; Julong He — Supervision, Validation,
Data curation, Visualization.

Highlights are provided as a separate file. Supplementary material containing
the verification code and the per-table artifacts is included with the
submission, and the manuscript is available as an editable LaTeX source
together with the compiled PDF.

Thank you for considering our work.

Sincerely,

Fangyu Zhang (corresponding author)
College of Mechanical and Aerospace Engineering, Jilin University
Changchun 130025, China
fyzhang9925@mails.jlu.edu.cn · ORCID 0009-0009-7352-1484
on behalf of both authors, Fangyu Zhang and Julong He

---

## 投稿前对照表(MSSP / Elsevier)

| 项 | 状态 |
|---|---|
| 正文(可编辑源文件 LaTeX + PDF) | ✅ `manuscript_mssp.tex` + `manuscript_mssp.pdf`(25 页 preprint 双倍行距) |
| 摘要 ≤250 词 | ✅ 246 词 |
| 关键词 1–7 个 | ✅ 7 个 |
| **Highlights(3–5 条,每条 ≤85 字符)** | ✅ `highlights.txt`,已逐条计数 |
| Declaration of competing interest | ✅ 已写入正文(转换脚本自动插入) |
| 作者姓名 / 单位 / email / ORCID | ✅ **已填入两版稿件与投稿信**(Fangyu Zhang @ 吉林大学 / Julong He @ 北京科技大学) |
| CRediT 作者贡献声明 | ✅ 已写入正文:Fangyu Zhang 承担其余全部角色,Julong He 承担 Supervision / Validation / Data curation / Visualization |
| Data availability 声明 | ✅ 已写入正文:代码与逐表产物作为补充材料随投稿提供(若日后挂仓库,改这一句为 URL 即可) |
| 致谢 / 基金 | ✅ 无基金,已按 Elsevier 标准句式写入(Funding 一节的 no-specific-grant 句);原独立的致谢节已删 |
| 生成式 AI 使用披露 | ✅ 已按 Elsevier 模板措辞写入(仅语言润色,工具名为 ClawsGO Science Agent),节标题用 Elsevier 推荐标题 |
| 建议审稿人 | ⬜ 可选;若要填,建议从正文引用的同领域作者中选,但须避开发表过同一课题的合作者 |
| 版面费 | ✅ 订阅通道**不收费**(仅 Open Access 才付 APC) |

**注意两份稿子的关系**:`manuscript_mssp.tex` 由 `tools/to_elsarticle.py` 从
IEEEtran 版 `manuscript.tex` **机械转换**而来,正文逐字一致(脚本每次都会校验这一点)。
**要改内容请改 IEEEtran 版再重跑转换**,不要只改 elsarticle 版,否则两份稿子会分叉。
IEEEtran 版(10 页)保留作为改投 TIM 的后备。
