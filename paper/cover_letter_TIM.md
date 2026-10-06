# Cover Letter — IEEE Transactions on Instrumentation and Measurement

> **使用说明**:方括号 `[...]` 处必须由作者填写,我没有替你编造。填完后把本文件转成
> PDF(或直接粘贴到 Editorial Manager 的 cover-letter 框里)。
> 投稿入口:**http://www.editorialmanager.com/tim**(IEEE TIM 官方 Information for
> Authors 页给出的地址;网上仍在流传的 `tim.allentrack.net` 是旧信息,不要用)。

---

\[日期\]

To the Editor-in-Chief
*IEEE Transactions on Instrumentation and Measurement*

Dear Editor,

We submit for your consideration the manuscript **"Why Simulation-Based Sample
Supplementation Fails in Cross-Domain Bearing Fault Diagnosis: The
Recording-Fingerprint Effect and an Evaluation Protocol That Reveals It"**, by
Fangyu Zhang and Julong He, for publication as a **regular paper**.

**EDICS classification.** We suggest *(1) Automation, Fault Diagnosis,
Maintenance, and Testing — Fault diagnosis*, and *(2) Artificial Intelligence
for Instrumentation and Measurement — Machine learning*. The first is the
substantive home of the work; the second reflects its methodological content.
Both are taken from the I&M Society EDICS list.

**What the paper claims.** Physics-based simulation, including "digital twin"
generators, is a widely used source of synthetic training samples for
cross-domain bearing fault diagnosis. We show that on the CWRU 12 kHz
drive-end benchmark this approach cannot work, for a structural reason that no
improvement in simulator fidelity can remove: each "fault type × load" cell of
that benchmark is a *single recording*, and the class-conditional magnitude
spectrum is a stable property of the individual recording rather than of the
fault. A carefully calibrated twin therefore produces classes that are
spectrally indistinguishable, and a classifier trained on twin data alone
reaches 0.123 macro-F1 on real data, against a chance level of 0.10.

**What may be of most interest to your readership.** While quantifying the above
we found that the standard few-shot protocol on this benchmark draws support
and test windows from the same recording, so both carry that same fingerprint.
Re-evaluating cross-recording lowers reported macro-F1 by 0.15–0.19 on average
(up to 0.31), and it removes the apparent advantage of every augmentation
method we tested — including adding 400 *real* labeled samples per class, which
lands within noise of the trivial baseline. We give a self-contained protocol
control that any paper in this area can run at negligible cost, and we
recommend it explicitly. We also report a second benchmark (Paderborn) where
the class labels *are* physically meaningful, and show that the same
same-recording optimism reproduces there (0.340 macro-F1), decomposing into a
session term and a specimen term.

**Relation to prior work.** The negative result is about the *pairing* of a
generator with a benchmark, not about simulation-based augmentation as such:
on Paderborn our physics-carrying augmentation is no longer the worst entry,
and the ordering between physics-driven and fingerprint-driven synthesis
inverts between the two benchmarks exactly where the proposed mechanism
predicts. We are not aware of prior work that identifies the single-recording
structure of CWRU as the reason simulation-based supplementation fails, nor one
that quantifies the same-recording optimism of this benchmark with a
cross-recording control.

**Reproducibility.** Every numeric value in the paper is reproduced from the raw
public data by a released verification script, which recomputes each reported
figure and fails on any mismatch; the per-table artifacts backing the tables
are included in the supplementary material. The tables are reported over
multiple repeated executions rather than the conventional three seeds, because
we measured run-to-run variation on this hardware large enough to make a
three-seed standard deviation misleading; this is stated in the paper.

**Declarations.**
- The manuscript is original, has not been published previously, and is not
  under consideration elsewhere.
- All authors have approved the submission and agree to its content.
- The authors declare no conflict of interest.
- This work received no external funding.
- Author contributions (CRediT): Fangyu Zhang — Conceptualization, Methodology,
  Software, Formal analysis, Investigation, Resources, Writing – original draft,
  Writing – review & editing, Project administration; Julong He — Supervision,
  Validation, Data curation, Visualization.
- Generative AI was used only to improve the language of the manuscript. In
  accordance with IEEE policy on AI-generated content, the authors declare that
  no part of the technical content, experiments, results or conclusions was
  produced by an AI tool, and the authors take full responsibility for the
  content of the publication.

**Length and overlength charges.** The manuscript is 10 pages in the IEEE
two-column transactions format. We acknowledge the mandatory overlength page
charge for the two pages beyond eight and will submit the signed Overlength
Page Charge Agreement with the manuscript.

**Submission materials.** A single self-contained PDF is uploaded. \[如适用:
Supplementary material and a graphical abstract are also provided.\]

Thank you for considering our work.

Sincerely,

Fangyu Zhang (corresponding author)
College of Mechanical and Aerospace Engineering, Jilin University
Changchun 130025, China
fyzhang9925@mails.jlu.edu.cn · ORCID 0009-0009-7352-1484
on behalf of both authors, Fangyu Zhang and Julong He

---

## 投稿前还要准备的东西(逐条对照 TIM Information for Authors)

| 项 | 状态 |
|---|---|
| 每位作者姓名 / email / ORCID | ✅ 已填入(Fangyu Zhang @ 吉林大学 / Julong He @ 北京科技大学) |
| 标题、摘要 | ✅ 摘要已压到 246 词,索引词已按字母序 |
| 两个 EDICS 分类 | ✅ 已在本文件中给出建议 |
| 单一自包含、未加密的 PDF(≤100 MB) | ✅ 275 KB |
| 超页费协议 Overlength Page Charge Agreement | ⬜ 若超过 8 页,须在**投稿时**提交,且由一位有财务权的作者签署(不能是学生) |
| IEEE 版权协议 | ⬜ 录用后提交 |
| 经费 / 致谢 | ✅ 无基金(声明写在投稿信;IEEEtran 版正文没有致谢节,若改投 TIM 需在参考文献前补一节重复这几条) |
| AI 使用披露 | ✅ 已写明用于语言润色;同上,改投 TIM 时需在正文补 Acknowledgment 一节 |
| CRediT 分工 | ✅ 已写入投稿信(两版稿子中只有 elsarticle 版正文含 CRediT 节) |
| 作者简介 | 不适用(短论文/常规论文投稿阶段不需要) |

**页数成本**:当前 10 页,超 8 页免费额度 2 页 → 强制超页费约 **$530**(非会员)或
**$440**(I&M 会员),另加税。TIM 对常规论文**没有页数上限**,超页费是强制的但不会因此退稿。
若你希望压到 8 页以内,需要删掉约 2 页正文内容(我评估:可压缩的是 Paderborn 一节的
推导细节和部分方法描述,但会削弱论证链条,建议先确认)。
